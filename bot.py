import httpx
import os
import re
import json
import asyncio
import time
from datetime import date, timedelta, timezone, datetime
from collections import defaultdict, deque
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, RetryAfter, Conflict
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes
)
from openai import OpenAI, RateLimitError, APIError

# ============================================================
# KONFIGURASI
# ============================================================
TELEGRAM_TOKEN      = os.environ.get("TELEGRAM_TOKEN")
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY")
MCP_URL             = "https://garmin.amalgama.co/api/v1/mcp/48247257-4554-43df-9e93-e7dd3710c58a"
CHAT_ID             = os.environ.get("CHAT_ID")

# Model OpenRouter — ganti sesuai kebutuhan, contoh lain:
# "openai/gpt-4o-mini", "anthropic/claude-3-haiku", "google/gemini-flash-1.5"
MODEL_CHAT       = "anthropic/claude-opus-4-7"
TEMPERATURE_CHAT = 0.1          # sangat rendah → lebih faktual, minim halusinasi

CHECK_INTERVAL   = 600
STREAM_INTERVAL  = 1.5
MAX_HISTORY      = 8            # simpan 8 pasang user/assistant
GARMIN_CACHE_TTL = 300

WITA = timezone(timedelta(hours=8))
# ============================================================

openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/running-assistant-bot",
        "X-Title": "Running Assistant Bot",
    }
)
reported_activities  = set()
conversation_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY * 2))

garmin_cache      = {}   # {user_id: (timestamp, processed_data)}
last_garmin_data  = {}   # {user_id: processed_data}


# ════════════════════════════════════════════════════════════
#  KONVERSI & PRE-PROCESSING DATA GARMIN
#  → Semua angka dikonversi di Python, LLM tinggal display
# ════════════════════════════════════════════════════════════

def fmt_pace(seconds_per_meter: float) -> str:
    """detik/meter → 'M:SS/km'"""
    if not seconds_per_meter or seconds_per_meter <= 0:
        return "N/A"
    spk = seconds_per_meter * 1000
    return f"{int(spk // 60)}:{int(spk % 60):02d}/km"


def fmt_pace_from_speed(speed_ms: float) -> str:
    """m/s → 'M:SS/km'"""
    if not speed_ms or speed_ms <= 0:
        return "N/A"
    spk = 1000 / speed_ms
    return f"{int(spk // 60)}:{int(spk % 60):02d}/km"


def fmt_duration(seconds: float) -> str:
    """detik → 'Xj Ym' atau 'Ym Zd'"""
    s = int(seconds)
    if s <= 0:
        return "N/A"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h} jam {m} menit"
    if m > 0:
        return f"{m} menit {sec} detik"
    return f"{sec} detik"


def fmt_distance(meters: float) -> str:
    """meter → 'X.XX km'"""
    if not meters or meters <= 0:
        return "N/A"
    return f"{meters / 1000:.2f} km"


def fmt_hr(bpm) -> str:
    try:
        return f"{int(float(bpm))} bpm"
    except Exception:
        return "N/A"


def fmt_calories(cal) -> str:
    try:
        return f"{int(float(cal)):,} kcal".replace(",", ".")
    except Exception:
        return "N/A"


def convert_utc_to_wita(text: str) -> str:
    MONTHS = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    def replace_iso(m):
        try:
            dt = datetime.fromisoformat(m.group(0).replace("Z", "+00:00"))
            return dt.astimezone(WITA).strftime("%-d %b %Y %H:%M WITA")
        except Exception:
            return m.group(0)

    text = re.sub(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z',
        replace_iso, text
    )

    def replace_readable(m):
        try:
            day, mon, year, hour, minute = m.groups()
            dt = datetime(
                int(year), MONTHS[mon], int(day),
                int(hour), int(minute), tzinfo=timezone.utc
            )
            return dt.astimezone(WITA).strftime("%-d %b %Y %H:%M WITA")
        except Exception:
            return m.group(0)

    text = re.sub(
        r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\s+(\d{4})\s+(\d{2}):(\d{2})\s+UTC',
        replace_readable, text
    )

    def replace_space_iso(m):
        try:
            dt_str = m.group(0).replace(" UTC", "+00:00").replace(" ", "T", 1)
            dt = datetime.fromisoformat(dt_str)
            return dt.astimezone(WITA).strftime("%-d %b %Y %H:%M WITA")
        except Exception:
            return m.group(0)

    text = re.sub(
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC',
        replace_space_iso, text
    )
    return text


def preprocess_garmin_json(raw: str) -> str:
    """
    Coba parse JSON dari MCP, konversi semua unit ke human-readable.
    Kalau bukan JSON, lakukan konversi berbasis regex sebagai fallback.
    """
    # ── Coba parse sebagai JSON ──────────────────────────────
    try:
        obj = json.loads(raw)
        return _process_json_obj(obj)
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Coba extract JSON array / object dari dalam teks ────
    for pattern in [r'\[.*\]', r'\{.*\}']:
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                prefix = raw[:m.start()].strip()
                processed = _process_json_obj(obj)
                return f"{prefix}\n{processed}".strip() if prefix else processed
            except Exception:
                pass

    # ── Fallback: konversi regex pada teks biasa ────────────
    return _regex_convert(raw)


def _process_json_obj(obj) -> str:
    """Rekursif konversi semua field numerik Garmin dalam dict/list."""
    if isinstance(obj, list):
        parts = []
        for i, item in enumerate(obj, 1):
            parts.append(f"[Aktivitas #{i}]\n{_process_json_obj(item)}")
        return "\n\n".join(parts)

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            k_lower = k.lower()

            # ── Pace (detik/meter atau detik/km) ──────────
            if any(x in k_lower for x in ['pace', 'avgpace', 'averagepace']):
                try:
                    fv = float(v)
                    if 0 < fv < 1:           # detik/meter
                        result[k] = fmt_pace(fv)
                    elif fv >= 60:           # detik/km
                        result[k] = f"{int(fv // 60)}:{int(fv % 60):02d}/km"
                    else:
                        result[k] = str(v)
                except Exception:
                    result[k] = str(v)

            # ── Speed (m/s) ────────────────────────────────
            elif any(x in k_lower for x in ['speed', 'avgspeed', 'averagespeed',
                                             'maxspeed']):
                try:
                    fv = float(v)
                    if 0 < fv < 30:          # wajar m/s untuk lari
                        pace_str = fmt_pace_from_speed(fv)
                        result[k] = f"{fv:.2f} m/s ({pace_str})"
                    else:
                        result[k] = str(v)
                except Exception:
                    result[k] = str(v)

            # ── Distance (meter) ───────────────────────────
            elif any(x in k_lower for x in ['distance', 'totaldistance']):
                try:
                    fv = float(v)
                    if fv > 500:             # pasti meter
                        result[k] = fmt_distance(fv)
                    else:
                        result[k] = str(v)
                except Exception:
                    result[k] = str(v)

            # ── Duration (detik) ───────────────────────────
            elif any(x in k_lower for x in ['duration', 'elapsedtime',
                                             'movingtime', 'totaltime']):
                try:
                    fv = float(v)
                    if fv > 60:              # lebih dari 1 menit
                        result[k] = fmt_duration(fv)
                    else:
                        result[k] = str(v)
                except Exception:
                    result[k] = str(v)

            # ── Heart Rate ────────────────────────────────
            elif any(x in k_lower for x in ['heartrate', 'hr', 'avghr',
                                             'maxhr', 'averagehr']):
                try:
                    result[k] = fmt_hr(v)
                except Exception:
                    result[k] = str(v)

            # ── Calories ──────────────────────────────────
            elif any(x in k_lower for x in ['calorie', 'calories', 'kcal']):
                try:
                    result[k] = fmt_calories(v)
                except Exception:
                    result[k] = str(v)

            # ── Nested dict/list ──────────────────────────
            elif isinstance(v, (dict, list)):
                result[k] = json.loads(_process_json_obj(v)) \
                    if False else _process_json_obj(v)

            else:
                result[k] = v

        # Format sebagai teks terstruktur
        lines = []
        for k, v in result.items():
            if isinstance(v, str) and '\n' in v:
                lines.append(f"{k}:\n  {v.replace(chr(10), chr(10)+'  ')}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    # Primitive value
    return str(obj)


def _regex_convert(text: str) -> str:
    """Konversi berbasis regex kalau data bukan JSON."""

    # avgSpeed / maxSpeed (m/s) → pace
    def repl_speed(m):
        try:
            fv = float(m.group(2))
            if 0 < fv < 30:
                return f"{m.group(1)}: {fv:.2f} m/s ({fmt_pace_from_speed(fv)})"
        except Exception:
            pass
        return m.group(0)

    text = re.sub(
        r'(avg[Ss]peed|max[Ss]peed|[Ss]peed)\s*[=:]\s*([0-9.]+)',
        repl_speed, text
    )

    # avgPace (detik/meter atau detik/km)
    def repl_pace(m):
        try:
            fv = float(m.group(1))
            if 0 < fv < 1:
                return f"avgPace: {fmt_pace(fv)}"
            elif fv >= 60:
                return f"avgPace: {int(fv // 60)}:{int(fv % 60):02d}/km"
        except Exception:
            pass
        return m.group(0)

    text = re.sub(
        r'avg[Pp]ace\s*[=:]\s*([0-9.]+)',
        repl_pace, text
    )

    # distance (meter → km)
    def repl_dist(m):
        try:
            fv = float(m.group(1))
            if fv > 500:
                return f"distance: {fmt_distance(fv)}"
        except Exception:
            pass
        return m.group(0)

    text = re.sub(
        r'distance\s*[=:]\s*([0-9.]+)',
        repl_dist, text, flags=re.IGNORECASE
    )

    # duration (detik → jam/menit)
    def repl_dur(m):
        try:
            fv = float(m.group(1))
            if fv > 60:
                return f"duration: {fmt_duration(fv)}"
        except Exception:
            pass
        return m.group(0)

    text = re.sub(
        r'(duration|elapsedTime|movingTime)\s*[=:]\s*([0-9.]+)',
        lambda m: f"{m.group(1)}: {fmt_duration(float(m.group(2)))}"
            if float(m.group(2)) > 60 else m.group(0),
        text, flags=re.IGNORECASE
    )

    # HR
    text = re.sub(
        r'(avg[Hh][Rr]|max[Hh][Rr]|heartRate)\s*[=:]\s*([0-9]+)',
        lambda m: f"{m.group(1)}: {fmt_hr(m.group(2))}",
        text
    )

    # Calories
    text = re.sub(
        r'(calories?|kcal)\s*[=:]\s*([0-9.]+)',
        lambda m: f"{m.group(1)}: {fmt_calories(m.group(2))}",
        text, flags=re.IGNORECASE
    )

    return text


# ════════════════════════════════════════════════════════════
#  MARKDOWN → HTML TELEGRAM
# ════════════════════════════════════════════════════════════

def markdown_to_html(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'^\s*[\*\-]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__',     r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'`([^`\n]+?)`',  r'<code>\1</code>', text)
    text = re.sub(r'(?<!\w)\*(?!\w)', '', text)
    return text


def safe_html_for_streaming(text: str) -> str:
    html = markdown_to_html(text)
    open_code = html.count("<code>") - html.count("</code>")
    open_b    = html.count("<b>")    - html.count("</b>")
    if open_code > 0:
        html += "</code>" * open_code
    if open_b > 0:
        html += "</b>" * open_b
    return html


# ════════════════════════════════════════════════════════════
#  SYSTEM PROMPT  (sangat ketat, anti-halusinasi)
# ════════════════════════════════════════════════════════════

COACH_SYSTEM_PROMPT = """Kamu adalah Running Assistant, pelatih lari pribadi yang ramah.

═══════════════════════════════════════
ATURAN WAJIB — ANTI-HALUSINASI
═══════════════════════════════════════
1. HANYA gunakan angka yang TERSURAT di blok [DATA GARMIN].
2. DILARANG KERAS menghitung ulang, mengestimasi, atau mengisi sendiri angka yang tidak ada.
3. Semua angka (pace, jarak, durasi, HR, kalori) SUDAH dikonversi ke satuan benar.
   → LANGSUNG tampilkan, JANGAN konversi ulang.
4. Kalau data tidak ada → jawab: "Data tidak tersedia untuk itu."
5. JANGAN pernah menyebut angka yang tidak ada di [DATA GARMIN].

═══════════════════════════════════════
FORMAT ANGKA (HANYA display, jangan konversi)
═══════════════════════════════════════
• Jarak  : sudah dalam km   → tampilkan apa adanya, misal "12,34 km"
• Durasi : sudah dalam jam/menit → misal "1 jam 23 menit"
• Pace   : sudah dalam M:SS/km → misal "5:30/km"
• HR     : sudah dalam bpm  → misal "136 bpm"
• Kalori : sudah dalam kcal → misal "850 kcal"
• Waktu  : sudah dalam WITA → tampilkan apa adanya

═══════════════════════════════════════
KARAKTER & GAYA
═══════════════════════════════════════
• Ramah seperti teman, bahasa Indonesia santai
• Jawab SINGKAT & FOKUS — jangan kasih full report kalau ditanya 1 hal
• Sapaan/casual → balas natural 1-3 kalimat, tanpa data
• Ingat konteks percakapan sebelumnya

═══════════════════════════════════════
FORMAT JAWABAN (kalau diminta analisis lengkap)
═══════════════════════════════════════
📊 **Statistik Utama**
🏃 **Detail Aktivitas**
❤️ **Performa Tubuh**
💡 **Insight & Saran**

• Bold pakai **teks**
• Bullet pakai •
• Pisahkan section dengan baris kosong"""


# ════════════════════════════════════════════════════════════
#  KLASIFIKASI INTENT (pure regex, zero API call)
# ════════════════════════════════════════════════════════════

_DATA_KEYWORDS = re.compile(
    r'\b(data|lari|aktivitas|tanggal|pace|jarak|km|hr|denyut|nadi|kalori|'
    r'durasi|waktu|kecepatan|statistik|ringkasan|analisis|evaluasi|performa|'
    r'latihan|saran|rekomendasi|berikan|kasih|lihat|cek|tampilkan|tunjukkan|'
    r'bulan|minggu|hari|kemarin|tadi|terakhir|terbaru|terbaik|terjauh|'
    r'tercepat|average|rata|total|summary|report|hasil|laju|kecepatan)\b',
    re.IGNORECASE
)

_SAPAAN_ONLY = re.compile(
    r'^(halo|hallo|hai|hi|hey|pagi|siang|sore|malam|'
    r'oke|ok|iya|ya|siap|makasih|terima kasih|thx|thanks|'
    r'mantap|keren|bagus|sip|lanjut|done|selesai|'
    r'hehe|wkwk|😊|👍|🙏)[\s!.]*$',
    re.IGNORECASE
)

_FOLLOW_UP = re.compile(
    r'^(kenapa|knp|mengapa|terus|trus|lalu|lanjut|detailnya|'
    r'detail|maksudnya|maksud|gimana|bagaimana|contohnya|misalnya|'
    r'jelaskan|jelasin|lebih lanjut|selengkapnya)\??\s*$',
    re.IGNORECASE
)

def classify_intent(message: str) -> str:
    msg = message.strip()
    if _FOLLOW_UP.match(msg):
        return "follow_up"
    if _SAPAAN_ONLY.match(msg) and not _DATA_KEYWORDS.search(msg):
        return "sapaan"
    if _DATA_KEYWORDS.search(msg):
        if re.search(r'\b(analisis|ringkasan|evaluasi|summary|report|lengkap|menyeluruh|semua)\b',
                     msg, re.IGNORECASE):
            return "analisis_lengkap"
        if re.search(r'\b(saran|rekomendasi|tips|improve|tingkatkan|sebaiknya|harus)\b',
                     msg, re.IGNORECASE):
            return "saran_latihan"
        return "tanya_spesifik"
    return "tanya_spesifik"


# ════════════════════════════════════════════════════════════
#  MCP CALL
# ════════════════════════════════════════════════════════════

async def call_mcp(tool_name: str, arguments: dict = {}) -> str:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments}
            }
            r = await client.post(
                MCP_URL, json=payload,
                headers={"Content-Type": "application/json"}
            )
            data = r.json()
            result  = data.get("result", {})
            content = result.get("content", [])
            if content:
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts)
            return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"


# ════════════════════════════════════════════════════════════
#  FETCH & CACHE DATA GARMIN
# ════════════════════════════════════════════════════════════

async def fetch_garmin_data(user_id: int = 0):
    """
    Return data string yang sudah dikonversi, atau None kalau gagal.
    Data di-cache selama GARMIN_CACHE_TTL detik.
    """
    now = time.time()
    if user_id in garmin_cache:
        cached_time, cached_data = garmin_cache[user_id]
        if now - cached_time < GARMIN_CACHE_TTL:
            print(f"✓ Cache hit Garmin (age: {int(now - cached_time)}s)")
            return cached_data

    today      = date.today().isoformat()
    bulan_lalu = (date.today() - timedelta(days=30)).isoformat()

    aktivitas_raw = await call_mcp("list_activities", {
        "limit": 20, "from_date": bulan_lalu, "to_date": today
    })
    stats_raw = await call_mcp("get_activity_stats", {
        "from_date": bulan_lalu, "to_date": today
    })

    if "Error" in aktivitas_raw or "Error" in stats_raw:
        print(f"✗ Garmin fetch error")
        return None
    if not aktivitas_raw.strip() or not stats_raw.strip():
        print("✗ Garmin data kosong")
        return None

    # ── Pre-process: konversi semua unit di Python ───────────
    aktivitas_proc = preprocess_garmin_json(aktivitas_raw)
    stats_proc     = preprocess_garmin_json(stats_raw)

    # ── Konversi timestamp UTC → WITA ────────────────────────
    aktivitas_proc = convert_utc_to_wita(aktivitas_proc)
    stats_proc     = convert_utc_to_wita(stats_proc)

    data = (
        "=== AKTIVITAS 30 HARI TERAKHIR (sudah dikonversi) ===\n"
        f"{aktivitas_proc}\n\n"
        "=== STATISTIK KESELURUHAN (sudah dikonversi) ===\n"
        f"{stats_proc}"
    )

    garmin_cache[user_id]     = (now, data)
    last_garmin_data[user_id] = data
    print("✓ Garmin data fresh, di-cache")
    return data


def get_garmin_context(user_id: int) -> str:
    """Ambil data dari cache tanpa fetch — untuk follow_up."""
    if user_id in garmin_cache:
        return garmin_cache[user_id][1]
    return last_garmin_data.get(user_id, "")


# ════════════════════════════════════════════════════════════
#  BUILD PROMPT (grounding eksplisit)
# ════════════════════════════════════════════════════════════

def build_user_content(user_message: str, garmin_data: str, mode: str = "data") -> str:
    if mode == "data" and garmin_data:
        return (
            f"{user_message}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"[DATA GARMIN — SEMUA ANGKA SUDAH DIKONVERSI]\n"
            f"[WAJIB: HANYA gunakan angka dari sini. DILARANG menghitung ulang.]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{garmin_data}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"[AKHIR DATA GARMIN]"
        )
    else:
        return (
            f"{user_message}\n\n"
            f"[CATATAN SISTEM: Tidak ada data Garmin. "
            f"DILARANG menyebut angka apapun. "
            f"Kalau ditanya statistik lari, minta user ketik /ringkasan.]"
        )


# ════════════════════════════════════════════════════════════
#  STREAMING OPENROUTER
# ════════════════════════════════════════════════════════════

async def stream_to_telegram(
    user_id: int,
    user_message: str,
    user_content: str,
    message,
    context: ContextTypes.DEFAULT_TYPE,
    max_retries: int = 2
) -> str:
    messages = [{"role": "system", "content": COACH_SYSTEM_PROMPT}]
    messages.extend(list(conversation_history[user_id]))
    messages.append({"role": "user", "content": user_content})

    for attempt in range(max_retries + 1):
        try:
            return await _do_stream(user_id, user_message, messages, message, context)

        except RateLimitError as e:
            if attempt < max_retries:
                wait_time = 30
                m = re.search(r'try again in ([\d.]+)s', str(e))
                if m:
                    wait_time = float(m.group(1)) + 1
                print(f"⚠ Rate limit, tunggu {wait_time:.0f}s...")
                try:
                    await message.edit_text(f"⏳ Server sibuk, tunggu {int(wait_time)}s...")
                except Exception:
                    pass
                await asyncio.sleep(wait_time)
                continue
            try:
                await message.edit_text("😅 Server sibuk. Coba lagi 1-2 menit ya!")
            except Exception:
                pass
            return ""

        except APIError as e:
            print(f"OpenRouter API error: {e}")
            try:
                await message.edit_text("❌ Error dari server AI. Coba lagi ya!")
            except Exception:
                pass
            return ""

        except Exception as e:
            print(f"Stream error: {e}")
            try:
                await message.edit_text(f"❌ Error: {str(e)[:100]}")
            except Exception:
                pass
            return ""
    return ""


async def _do_stream(user_id, user_message, messages, message, context):
    loop = asyncio.get_event_loop()

    def get_stream():
        return openrouter_client.chat.completions.create(
            model=MODEL_CHAT,
            messages=messages,
            max_tokens=1024,
            temperature=TEMPERATURE_CHAT,
            stream=True
        )

    stream = await loop.run_in_executor(None, get_stream)

    full_text        = ""
    last_update_time = 0
    last_sent_html   = ""
    stream_iter      = iter(stream)

    def get_next_chunk(it):
        try:
            return next(it)
        except StopIteration:
            return None

    while True:
        chunk = await loop.run_in_executor(None, get_next_chunk, stream_iter)
        if chunk is None:
            break
        delta = chunk.choices[0].delta.content
        if delta:
            full_text += delta

        now = loop.time()
        if now - last_update_time >= STREAM_INTERVAL:
            html_text = safe_html_for_streaming(full_text) + " ▌"
            if html_text != last_sent_html:
                try:
                    await message.edit_text(html_text, parse_mode=ParseMode.HTML)
                    last_sent_html   = html_text
                    last_update_time = now
                    await context.bot.send_chat_action(
                        chat_id=message.chat_id, action=ChatAction.TYPING
                    )
                except RetryAfter as e:
                    await asyncio.sleep(e.retry_after)
                except BadRequest as e:
                    if "can't parse entities" in str(e).lower():
                        pass
                except Exception as e:
                    print(f"Stream edit error: {e}")

    if full_text:
        final_html = markdown_to_html(full_text)
        try:
            await message.edit_text(final_html, parse_mode=ParseMode.HTML)
        except BadRequest as e:
            if "can't parse entities" in str(e).lower():
                clean = re.sub(r'\*+', '', full_text)
                clean = re.sub(r'^\s*[-]\s+', '• ', clean, flags=re.MULTILINE)
                try:
                    await message.edit_text(clean)
                except Exception:
                    pass
        except Exception as e:
            print(f"Final edit error: {e}")

    # Simpan ke history (simpan user_message asli, bukan user_content yang panjang)
    conversation_history[user_id].append({"role": "user",      "content": user_message})
    conversation_history[user_id].append({"role": "assistant", "content": full_text})

    return full_text


# ════════════════════════════════════════════════════════════
#  AUTO-NOTIF (cek aktivitas baru)
# ════════════════════════════════════════════════════════════

async def cek_aktivitas_baru(context: ContextTypes.DEFAULT_TYPE):
    if not CHAT_ID:
        return

    today   = date.today().isoformat()
    kemarin = (date.today() - timedelta(days=2)).isoformat()

    try:
        hasil_raw = await call_mcp("list_activities", {
            "limit": 5, "from_date": kemarin, "to_date": today
        })
        if "Error" in hasil_raw or not hasil_raw.strip():
            return

        hasil = preprocess_garmin_json(hasil_raw)
        hasil = convert_utc_to_wita(hasil)

        if not reported_activities:
            reported_activities.add(hasil[:200])
            print("✓ Initial scan selesai")
            return

        signature = hasil[:200]
        if signature not in reported_activities:
            reported_activities.add(signature)
            user_content = build_user_content(
                "Ada aktivitas lari baru terdeteksi! "
                "Kasih ringkasan singkat (jarak km, pace M:SS/km, durasi jam:menit) "
                "plus kalimat motivasi pendek. "
                "HANYA gunakan angka dari data, jangan menambah apapun.",
                hasil, mode="data"
            )
            try:
                resp = openrouter_client.chat.completions.create(
                    model=MODEL_CHAT,
                    messages=[
                        {"role": "system", "content": COACH_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_content}
                    ],
                    max_tokens=350,
                    temperature=TEMPERATURE_CHAT
                )
                ringkasan = resp.choices[0].message.content
                html_text = (
                    f"🏃 <b>Aktivitas Lari Baru!</b>\n\n"
                    f"{markdown_to_html(ringkasan)}"
                )
                await context.bot.send_message(
                    chat_id=CHAT_ID, text=html_text, parse_mode=ParseMode.HTML
                )
            except RateLimitError:
                print("⚠ Rate limit di auto-notif, skip")
            except BadRequest:
                clean = re.sub(r'\*+', '', ringkasan)
                await context.bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"🏃 Aktivitas Lari Baru!\n\n{clean}"
                )
    except Exception as e:
        print(f"Error cek aktivitas: {e}")


# ════════════════════════════════════════════════════════════
#  ERROR HANDLER GLOBAL
# ════════════════════════════════════════════════════════════

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(context.error, Conflict):
        return
    print(f"⚠ Global error: {context.error}")
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "😅 Ada error. Coba lagi sebentar ya!"
            )
        except Exception:
            pass


# ════════════════════════════════════════════════════════════
#  COMMAND HANDLERS
# ════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nama    = update.effective_user.first_name
    chat_id = update.effective_chat.id
    conversation_history[update.effective_user.id].clear()

    welcome = (
        f"👟 Halo <b>{nama}</b>!\n\n"
        f"Aku <b>Running Assistant</b>, pelatih lari pribadi kamu. "
        f"Tanya apa aja seputar aktivitas Garmin kamu! 💪\n\n"
        f"📌 Chat ID: <code>{chat_id}</code>\n\n"
        f"<b>Contoh pertanyaan:</b>\n"
        f"• Berapa pace rata-rata saya?\n"
        f"• Lari terjauh saya kapan?\n"
        f"• Kasih analisis lengkap dong\n"
        f"• Saran latihan untuk improve\n\n"
        f"<b>Command:</b>\n"
        f"/ringkasan — analisis 30 hari\n"
        f"/cekkoneksi — tes koneksi Garmin\n"
        f"/reset — reset percakapan\n"
        f"/model — info model AI"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.HTML)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    conversation_history[uid].clear()
    garmin_cache.pop(uid, None)
    last_garmin_data.pop(uid, None)
    await update.message.reply_text("✅ Percakapan di-reset. Mulai fresh!")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = (
        f"🤖 <b>Model AI Info</b>\n\n"
        f"• <b>Chat:</b> <code>{MODEL_CHAT}</code>\n"
        f"• <b>Classifier:</b> <code>regex (built-in)</code>\n"
        f"• <b>Temperature:</b> <code>{TEMPERATURE_CHAT}</code>\n"
        f"• <b>Timezone:</b> <code>WITA (UTC+8)</code>\n\n"
        f"<i>Powered by OpenRouter</i>"
    )
    await update.message.reply_text(info, parse_mode=ParseMode.HTML)


async def cmd_ringkasan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    pesan = await update.message.reply_text("▌")
    data  = await fetch_garmin_data(uid)
    if data is None:
        await pesan.edit_text(
            "❌ <b>Tidak bisa mengambil data Garmin.</b>\n\n"
            "Coba /cekkoneksi untuk cek status.",
            parse_mode=ParseMode.HTML
        )
        return
    user_content = build_user_content(
        "Kasih analisis lengkap aktivitas lari aku 30 hari terakhir. "
        "Gunakan format: 📊 Statistik Utama, 🏃 Detail Aktivitas, "
        "❤️ Performa Tubuh, 💡 Insight & Saran. "
        "HANYA gunakan angka yang ada di data.",
        data, mode="data"
    )
    await stream_to_telegram(uid, user_content, user_content, pesan, context)


async def cmd_cekkoneksi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    pesan = await update.message.reply_text("🔍 Mengecek koneksi...")
    hasil = await call_mcp("list_activities", {"limit": 1})
    if "Error" in hasil:
        await pesan.edit_text(f"❌ Gagal:\n{hasil}")
    else:
        await pesan.edit_text(
            "✅ <b>Koneksi Garmin OK!</b>\nData siap dianalisis.",
            parse_mode=ParseMode.HTML
        )


# ════════════════════════════════════════════════════════════
#  MESSAGE HANDLER UTAMA
# ════════════════════════════════════════════════════════════

async def handle_pesan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid       = update.effective_user.id
    pertanyaan = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    intent = classify_intent(pertanyaan)
    print(f"[Intent: {intent}] {pertanyaan[:60]}")

    pesan = await update.message.reply_text("▌")

    # ── Sapaan murni ─────────────────────────────────────────
    if intent == "sapaan":
        user_content = build_user_content(pertanyaan, "", mode="no_data")
        await stream_to_telegram(uid, pertanyaan, user_content, pesan, context)
        return

    # ── Follow-up (pakai cache) ───────────────────────────────
    if intent == "follow_up":
        garmin_data = get_garmin_context(uid)
        if not garmin_data:
            await pesan.edit_text(
                "⚠️ Data Garmin belum dimuat.\n\n"
                "Tanya dulu sesuatu yang spesifik, atau ketik /ringkasan."
            )
            return
        user_content = build_user_content(pertanyaan, garmin_data, mode="data")
        await stream_to_telegram(uid, pertanyaan, user_content, pesan, context)
        return

    # ── Tanya spesifik / analisis / saran → fetch data ───────
    data = await fetch_garmin_data(uid)
    if data is None:
        await pesan.edit_text(
            "❌ <b>Data Garmin tidak bisa diambil.</b>\n\n"
            "• Koneksi ke Garmin mungkin terputus\n"
            "• Coba /cekkoneksi untuk cek status",
            parse_mode=ParseMode.HTML
        )
        return

    user_content = build_user_content(pertanyaan, data, mode="data")
    await stream_to_telegram(uid, pertanyaan, user_content, pesan, context)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("🤖 Running Assistant Bot berjalan...")
    print(f"   Chat model  : {MODEL_CHAT}")
    print(f"   Classifier  : regex (built-in)")
    print(f"   Temperature : {TEMPERATURE_CHAT}")
    print(f"   Timezone    : WITA (UTC+8)")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("ringkasan",   cmd_ringkasan))
    app.add_handler(CommandHandler("cekkoneksi",  cmd_cekkoneksi))
    app.add_handler(CommandHandler("reset",       cmd_reset))
    app.add_handler(CommandHandler("model",       cmd_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pesan))
    app.add_error_handler(error_handler)

    if CHAT_ID:
        app.job_queue.run_repeating(
            cek_aktivitas_baru, interval=CHECK_INTERVAL, first=10
        )
        print(f"✓ Auto-monitoring aktif (interval: {CHECK_INTERVAL}s)")
    else:
        print("⚠ CHAT_ID belum di-set, auto-notif nonaktif")

    app.run_polling()


if __name__ == "__main__":
    main()