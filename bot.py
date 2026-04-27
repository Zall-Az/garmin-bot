import httpx
import os
import re
import json
import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from collections import defaultdict, deque
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, RetryAfter, Conflict
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes
)
from groq import Groq, RateLimitError, APIError

# Opsional: deteksi timezone otomatis dari lokasi
# pip install timezonefinder
try:
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    LOCATION_SUPPORT = True
except ImportError:
    tf = None
    LOCATION_SUPPORT = False
    print("⚠ timezonefinder tidak terinstall. Fitur deteksi lokasi nonaktif.")
    print("  Install dengan: pip install timezonefinder")

# ============================================================
# KONFIGURASI
# ============================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
MCP_URL        = "https://garmin.amalgama.co/api/v1/mcp/48247257-4554-43df-9e93-e7dd3710c58a"
CHAT_ID        = os.environ.get("CHAT_ID")

# ⭐ MODEL CONFIG
MODEL_CHAT       = "openai/gpt-oss-120b"
MODEL_CLASSIFIER = "llama-3.1-8b-instant"

# Tuning parameters
CHECK_INTERVAL   = 600   # auto-cek aktivitas tiap 10 menit
STREAM_INTERVAL  = 1.5   # interval edit pesan saat streaming
MAX_HISTORY      = 8     # max pesan history per user
GARMIN_CACHE_TTL = 300   # cache data Garmin 5 menit

# Timezone default jika user belum set
DEFAULT_TZ = "Asia/Makassar"  # WITA — ganti sesuai kebutuhan
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)
reported_activities = set()
conversation_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

# Cache data Garmin per user
garmin_cache = {}   # {user_id: (timestamp, data)}

# Timezone per user — disimpan in-memory
# Untuk persistent, bisa ganti dengan sqlite/json file
user_timezones = {}  # {user_id: "Asia/Makassar"}

# Alias timezone yang mudah diingat
TIMEZONE_ALIASES = {
    # Indonesia
    "wib":      "Asia/Jakarta",
    "wita":     "Asia/Makassar",
    "wit":      "Asia/Jayapura",
    "jakarta":  "Asia/Jakarta",
    "makassar": "Asia/Makassar",
    "manado":   "Asia/Makassar",
    "bali":     "Asia/Makassar",
    "jayapura": "Asia/Jayapura",
    "papua":    "Asia/Jayapura",
    # Asia Tenggara
    "singapore":  "Asia/Singapore",
    "malaysia":   "Asia/Kuala_Lumpur",
    "thailand":   "Asia/Bangkok",
    "vietnam":    "Asia/Ho_Chi_Minh",
    "philippines": "Asia/Manila",
    "japan":      "Asia/Tokyo",
    "korea":      "Asia/Seoul",
    # Umum
    "utc":   "UTC",
    "gmt":   "UTC",
    "london": "Europe/London",
    "paris":  "Europe/Paris",
    "sydney": "Australia/Sydney",
    "ny":     "America/New_York",
    "la":     "America/Los_Angeles",
}


# ── Helper timezone ───────────────────────────────────────────
def get_user_tz(user_id: int) -> ZoneInfo:
    """Ambil timezone user, fallback ke DEFAULT_TZ."""
    tz_name = user_timezones.get(user_id, DEFAULT_TZ)
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo(DEFAULT_TZ)


def get_tz_name(user_id: int) -> str:
    return user_timezones.get(user_id, DEFAULT_TZ)


def convert_utc_to_local(text: str, user_id: int) -> str:
    """
    Konversi semua timestamp UTC dalam teks ke timezone lokal user.
    Mendukung format:
      - ISO 8601: 2026-04-26T20:54:00Z  atau  2026-04-26T20:54:00+00:00
      - Epoch integer (detik): 1745704440
    """
    tz = get_user_tz(user_id)

    # Format ISO 8601
    def replace_iso(m):
        raw = m.group(0)
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            local_dt = dt.astimezone(tz)
            return local_dt.strftime("%d %b %Y %H:%M %Z")
        except Exception:
            return raw

    text = re.sub(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})',
        replace_iso,
        text
    )

    # Format epoch (angka >= 10 digit yang masuk akal sebagai timestamp)
    def replace_epoch(m):
        raw = m.group(0)
        try:
            ts = int(raw)
            # Validasi: antara 2020 dan 2040
            if 1577836800 <= ts <= 2208988800:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                local_dt = dt.astimezone(tz)
                return local_dt.strftime("%d %b %Y %H:%M %Z")
        except Exception:
            pass
        return raw

    text = re.sub(r'\b(1[5-9]\d{8}|2[0-1]\d{8})\b', replace_epoch, text)

    return text


# ── Convert markdown ke HTML Telegram ─────────────────────────
def markdown_to_html(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = re.sub(r'^\s*[\*\-]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'`([^`\n]+?)`', r'<code>\1</code>', text)
    text = re.sub(r'(?<!\w)\*(?!\w)', '', text)
    return text


def safe_html_for_streaming(text: str) -> str:
    html = markdown_to_html(text)
    open_b    = html.count("<b>")    - html.count("</b>")
    open_code = html.count("<code>") - html.count("</code>")
    if open_code > 0:
        html += "</code>" * open_code
    if open_b > 0:
        html += "</b>" * open_b
    return html


# ── System prompts ────────────────────────────────────────────
INTENT_CLASSIFIER_PROMPT = """Klasifikasikan pesan user ke salah satu kategori:

- "sapaan": Pesan kasual seperti "halo", "hi", "pagi", "terima kasih", "oke" - tanpa pertanyaan data lari
- "tanya_spesifik": Pertanyaan ASPEK TERTENTU dari data lari (pace, denyut nadi, jarak hari ini, dll)
- "analisis_lengkap": Permintaan analisis menyeluruh (analisis dong, ringkasan, evaluasi performa)
- "saran_latihan": Minta saran/rekomendasi latihan
- "follow_up": Pertanyaan lanjutan singkat ("kenapa?", "terus?", "detailnya")
- "lain": Di luar konteks lari/Garmin

Jawab HANYA satu kata kategori."""


def build_coach_prompt(user_id: int) -> str:
    tz_name = get_tz_name(user_id)
    tz = get_user_tz(user_id)
    now_local = datetime.now(tz)
    tz_abbr = now_local.strftime("%Z")  # WIB / WITA / WIT / dll

    return f"""Kamu adalah Running Assistant, pelatih lari pribadi yang ramah dan conversational.

KARAKTER:
- Ramah seperti teman, bukan robot formal
- Bahasa Indonesia santai tapi profesional
- Inget konteks percakapan sebelumnya
- Motivatif dan supportive

TIMEZONE USER: {tz_name} ({tz_abbr})
- Semua data waktu yang diterima sudah dikonversi ke timezone user ini
- Tampilkan waktu sesuai timezone tersebut, JANGAN sebut UTC

ATURAN:

1. SAPAAN/CASUAL CHAT:
   - Balas natural dan singkat (1-3 kalimat)
   - JANGAN langsung kasih data atau analisis
   - Contoh: "Halo! Apa kabar? Ada yang mau dicek dari aktivitas lari kamu?"

2. PERTANYAAN SPESIFIK:
   - Jawab LANGSUNG dan SINGKAT, fokus ke yang ditanya saja
   - JANGAN kasih full report kalau cuma ditanya 1 hal
   - Contoh: "HR rata-rata kamu 30 hari terakhir adalah **136 bpm**. Tergolong intensitas sedang ya 👍"

3. ANALISIS LENGKAP (kalau diminta):
   - Pakai struktur lengkap dengan section + emoji:
     📊 **Statistik Utama**
     🏃 **Detail Aktivitas**
     ❤️ **Performa Tubuh**
     💡 **Insight & Saran**

4. ATURAN ANGKA (PENTING!):
   - Jarak: HANYA km (1-2 desimal). Contoh: "12,5 km"
   - Durasi: jam:menit. Contoh: "23 jam 30 menit"
   - Pace: menit:detik per km. Contoh: "5:30/km"
   - HR: tambah "bpm". Contoh: "136 bpm"
   - Kalori: tambah "kcal". Contoh: "8.431 kcal"
   - JANGAN tampilkan satuan raw (meter, detik)
   - WAKTU: Semua waktu sudah dalam {tz_abbr}, tampilkan apa adanya tanpa konversi tambahan

5. FORMATTING:
   - Bold pakai **double asterisk** untuk angka penting
   - Bullet pakai • (BUKAN * atau -)
   - Pisahkan section dengan baris kosong

6. KONTEKS:
   - Inget percakapan sebelumnya
   - Kalau user nanya "kenapa?" atau "terus?", lanjutkan dari topik tadi
   - Jangan ulang info yang udah dikasih sebelumnya"""


# ── Klasifikasi intent ────────────────────────────────────────
async def classify_intent(message: str) -> str:
    try:
        loop = asyncio.get_event_loop()

        def _classify():
            return groq_client.chat.completions.create(
                model=MODEL_CLASSIFIER,
                messages=[
                    {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
                    {"role": "user", "content": message}
                ],
                max_tokens=20,
                temperature=0.1
            )

        response = await loop.run_in_executor(None, _classify)
        intent = response.choices[0].message.content.strip().lower()

        valid = ["sapaan", "tanya_spesifik", "analisis_lengkap", "saran_latihan", "follow_up", "lain"]
        for v in valid:
            if v in intent:
                return v
        return "tanya_spesifik"
    except Exception as e:
        print(f"Intent classify error: {e}")
        return "tanya_spesifik"


# ── Panggil tool MCP ──────────────────────────────────────────
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
            result = data.get("result", {})
            content = result.get("content", [])
            if content:
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts)
            return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"


# ── Ambil data Garmin (dengan cache + konversi timezone) ──────
async def fetch_garmin_data(user_id: int = 0) -> str:
    now = time.time()

    # Cek cache — cache per (user_id, tz_name) agar beda tz tidak tabrakan
    tz_name = get_tz_name(user_id)
    cache_key = (user_id, tz_name)

    if cache_key in garmin_cache:
        cached_time, cached_data = garmin_cache[cache_key]
        if now - cached_time < GARMIN_CACHE_TTL:
            print(f"✓ Pakai cache Garmin (age: {int(now - cached_time)}s)")
            return cached_data

    today = date.today().isoformat()
    bulan_lalu = (date.today() - timedelta(days=30)).isoformat()

    aktivitas = await call_mcp("list_activities", {
        "limit": 20,
        "from_date": bulan_lalu,
        "to_date": today
    })
    stats = await call_mcp("get_activity_stats", {
        "from_date": bulan_lalu,
        "to_date": today
    })

    raw = f"=== AKTIVITAS 30 HARI TERAKHIR ===\n{aktivitas}\n\n=== STATISTIK ===\n{stats}"

    # Konversi semua timestamp ke timezone user
    data = convert_utc_to_local(raw, user_id)

    garmin_cache[cache_key] = (now, data)
    print(f"✓ Fetch Garmin baru, cached (tz: {tz_name})")

    return data


# ── Streaming Groq dengan retry ──────────────────────────────
async def stream_groq_to_telegram(
    user_id: int,
    user_message: str,
    extra_context: str,
    message,
    context: ContextTypes.DEFAULT_TYPE,
    max_retries: int = 2
) -> str:
    system_prompt = build_coach_prompt(user_id)

    messages = [{"role": "system", "content": system_prompt}]
    history = list(conversation_history[user_id])
    messages.extend(history)

    if extra_context:
        user_content = f"{user_message}\n\n[Data Garmin tersedia:]\n{extra_context}"
    else:
        user_content = user_message
    messages.append({"role": "user", "content": user_content})

    for attempt in range(max_retries + 1):
        try:
            return await _do_stream(user_id, user_message, messages, message, context)
        except RateLimitError as e:
            if attempt < max_retries:
                wait_time = 30
                match = re.search(r'try again in ([\d.]+)s', str(e))
                if match:
                    wait_time = float(match.group(1)) + 1

                print(f"⚠ Rate limit, tunggu {wait_time}s...")
                try:
                    await message.edit_text(
                        f"⏳ Server sibuk, tunggu sebentar ({int(wait_time)}s)..."
                    )
                except:
                    pass
                await asyncio.sleep(wait_time)
                continue
            else:
                try:
                    await message.edit_text(
                        "😅 Maaf, server lagi sibuk banget. Coba lagi 1-2 menit ya!"
                    )
                except:
                    pass
                return ""
        except APIError as e:
            print(f"Groq API error: {e}")
            try:
                await message.edit_text("❌ Ada error dari server AI. Coba lagi sebentar ya!")
            except:
                pass
            return ""
        except Exception as e:
            print(f"Stream error: {e}")
            try:
                await message.edit_text(f"❌ Terjadi error: {str(e)[:100]}\nCoba lagi ya!")
            except:
                pass
            return ""

    return ""


async def _do_stream(user_id, user_message, messages, message, context):
    def get_stream():
        return groq_client.chat.completions.create(
            model=MODEL_CHAT,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True
        )

    loop = asyncio.get_event_loop()
    stream = await loop.run_in_executor(None, get_stream)

    full_text = ""
    last_update_time = 0
    last_sent_html = ""

    def get_next_chunk(stream_iter):
        try:
            return next(stream_iter)
        except StopIteration:
            return None

    stream_iter = iter(stream)

    while True:
        chunk = await loop.run_in_executor(None, get_next_chunk, stream_iter)
        if chunk is None:
            break

        delta = chunk.choices[0].delta.content
        if delta:
            full_text += delta

        now = asyncio.get_event_loop().time()
        if now - last_update_time >= STREAM_INTERVAL:
            html_text = safe_html_for_streaming(full_text) + " ▌"
            if html_text != last_sent_html:
                try:
                    await message.edit_text(html_text, parse_mode=ParseMode.HTML)
                    last_sent_html = html_text
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
                clean_text = re.sub(r'\*+', '', full_text)
                clean_text = re.sub(r'^\s*[\-]\s+', '• ', clean_text, flags=re.MULTILINE)
                try:
                    await message.edit_text(clean_text)
                except BadRequest:
                    pass
        except Exception as e:
            print(f"Final edit error: {e}")

    conversation_history[user_id].append({"role": "user", "content": user_message})
    conversation_history[user_id].append({"role": "assistant", "content": full_text})

    return full_text


# ── Auto-notif aktivitas baru ─────────────────────────────────
async def cek_aktivitas_baru(context: ContextTypes.DEFAULT_TYPE):
    global reported_activities

    if not CHAT_ID:
        return

    today = date.today().isoformat()
    kemarin = (date.today() - timedelta(days=2)).isoformat()

    try:
        hasil = await call_mcp("list_activities", {
            "limit": 5,
            "from_date": kemarin,
            "to_date": today
        })

        if "Error" in hasil or not hasil.strip():
            return

        if not reported_activities:
            reported_activities.add(hasil[:200])
            print("✓ Initial scan selesai...")
            return

        signature = hasil[:200]
        if signature not in reported_activities:
            reported_activities.add(signature)

            # Konversi ke timezone CHAT_ID user (pakai user_id=0 → default tz)
            hasil_lokal = convert_utc_to_local(hasil, 0)

            try:
                response = groq_client.chat.completions.create(
                    model=MODEL_CHAT,
                    messages=[
                        {"role": "system", "content": build_coach_prompt(0)},
                        {
                            "role": "user",
                            "content": (
                                f"Data Garmin terbaru:\n{hasil_lokal}\n\n"
                                "Ada aktivitas lari baru baru aja kedeteksi! Kasih ringkasan singkat "
                                "(jarak km, pace, durasi jam:menit, waktu mulai) plus motivasi pendek."
                            )
                        }
                    ],
                    max_tokens=400,
                    temperature=0.7
                )
                ringkasan = response.choices[0].message.content
                html_text = f"🏃 <b>Aktivitas Lari Baru!</b>\n\n{markdown_to_html(ringkasan)}"

                await context.bot.send_message(
                    chat_id=CHAT_ID, text=html_text, parse_mode=ParseMode.HTML
                )
            except RateLimitError:
                print("⚠ Rate limit di auto-notif, skip...")
            except BadRequest:
                clean = re.sub(r'\*+', '', ringkasan)
                await context.bot.send_message(
                    chat_id=CHAT_ID, text=f"🏃 Aktivitas Lari Baru!\n\n{clean}"
                )

    except Exception as e:
        print(f"Error cek aktivitas: {e}")


# ── Error handler global ──────────────────────────────────────
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = context.error
    if isinstance(error, Conflict):
        return
    print(f"⚠ Global error: {error}")
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "😅 Ups, ada error nih. Coba lagi sebentar ya!"
            )
        except:
            pass


# ── Handlers ──────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nama = update.effective_user.first_name
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    conversation_history[user_id].clear()

    tz_name = get_tz_name(user_id)
    tz = get_user_tz(user_id)
    now_local = datetime.now(tz)
    tz_abbr = now_local.strftime("%Z")

    welcome = (
        f"👟 Halo <b>{nama}</b>!\n\n"
        f"Aku <b>Running Assistant</b>, pelatih lari pribadi kamu. "
        f"Tanya apa aja seputar aktivitas Garmin kamu! 💪\n\n"
        f"📌 Chat ID: <code>{chat_id}</code>\n"
        f"🕐 Timezone: <b>{tz_name}</b> ({tz_abbr})\n\n"
        f"<b>Contoh pertanyaan:</b>\n"
        f"• Berapa pace rata-rata saya?\n"
        f"• Lari terjauh saya kapan?\n"
        f"• Kasih analisis lengkap dong\n"
        f"• Saran latihan untuk improve\n\n"
        f"<b>Command:</b>\n"
        f"/ringkasan — analisis 30 hari\n"
        f"/cekkoneksi — tes Garmin\n"
        f"/settimezone — atur timezone\n"
        f"/mytimezone — lihat timezone aktif\n"
        f"/reset — reset percakapan\n"
        f"/model — info model AI"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.HTML)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_history[user_id].clear()
    # Hapus cache agar data di-fetch ulang dengan timezone terkini
    keys_to_del = [k for k in garmin_cache if k[0] == user_id]
    for k in keys_to_del:
        del garmin_cache[k]
    await update.message.reply_text("✅ Percakapan di-reset. Mulai fresh dari sini ya!")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = (
        f"🤖 <b>Model AI Info</b>\n\n"
        f"• <b>Chat utama:</b> <code>{MODEL_CHAT}</code>\n"
        f"• <b>Classifier:</b> <code>{MODEL_CLASSIFIER}</code>\n\n"
        f"<i>Powered by Groq</i>"
    )
    await update.message.reply_text(info, parse_mode=ParseMode.HTML)


async def cmd_mytimezone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tampilkan timezone aktif user beserta waktu sekarang."""
    user_id = update.effective_user.id
    tz_name = get_tz_name(user_id)
    tz = get_user_tz(user_id)
    now_local = datetime.now(tz)

    await update.message.reply_text(
        f"🕐 <b>Timezone kamu saat ini:</b>\n"
        f"<code>{tz_name}</code>\n\n"
        f"⏰ Waktu lokal sekarang:\n"
        f"<b>{now_local.strftime('%d %b %Y %H:%M %Z')}</b>\n\n"
        f"Untuk ganti: /settimezone",
        parse_mode=ParseMode.HTML
    )


async def cmd_settimezone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Set timezone user.
    Usage: /settimezone wita
           /settimezone Asia/Jakarta
           /settimezone  (tanpa argumen → tampilkan daftar)
    """
    user_id = update.effective_user.id
    args = context.args

    if not args:
        tz_name = get_tz_name(user_id)
        tz = get_user_tz(user_id)
        now_local = datetime.now(tz)

        await update.message.reply_text(
            f"🕐 <b>Timezone kamu sekarang:</b> <code>{tz_name}</code>\n"
            f"⏰ Waktu lokal: <b>{now_local.strftime('%d %b %Y %H:%M %Z')}</b>\n\n"
            f"<b>Cara ganti timezone:</b>\n"
            f"/settimezone wib\n"
            f"/settimezone wita\n"
            f"/settimezone wit\n"
            f"/settimezone singapore\n"
            f"/settimezone Asia/Tokyo\n\n"
            f"<b>Atau kirim 📍 Lokasi</b> untuk deteksi otomatis!\n\n"
            f"<b>Daftar alias tersedia:</b>\n"
            + "\n".join(
                f"• <code>{k}</code> → {v}"
                for k, v in TIMEZONE_ALIASES.items()
            ),
            parse_mode=ParseMode.HTML
        )
        return

    tz_input = args[0].strip()

    # Cek alias dulu
    tz_name = TIMEZONE_ALIASES.get(tz_input.lower(), tz_input)

    # Validasi timezone
    try:
        tz = ZoneInfo(tz_name)
        user_timezones[user_id] = tz_name

        # Hapus cache lama agar data di-fetch ulang
        keys_to_del = [k for k in garmin_cache if k[0] == user_id]
        for k in keys_to_del:
            del garmin_cache[k]

        now_local = datetime.now(tz)
        await update.message.reply_text(
            f"✅ <b>Timezone berhasil diubah!</b>\n\n"
            f"📍 Timezone: <code>{tz_name}</code>\n"
            f"⏰ Waktu lokal kamu sekarang:\n"
            f"<b>{now_local.strftime('%d %b %Y %H:%M %Z')}</b>\n\n"
            f"Data Garmin kamu akan ditampilkan dalam timezone ini mulai sekarang.",
            parse_mode=ParseMode.HTML
        )
    except ZoneInfoNotFoundError:
        await update.message.reply_text(
            f"❌ Timezone <code>{tz_input}</code> tidak dikenali.\n\n"
            f"Coba salah satu alias ini:\n"
            f"• <code>wib</code>, <code>wita</code>, <code>wit</code>\n"
            f"• <code>singapore</code>, <code>malaysia</code>\n\n"
            f"Atau format IANA lengkap:\n"
            f"• <code>Asia/Jakarta</code>\n"
            f"• <code>Asia/Singapore</code>\n\n"
            f"Lihat daftar lengkap: /settimezone",
            parse_mode=ParseMode.HTML
        )


async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Deteksi timezone otomatis dari lokasi yang dikirim user."""
    user_id = update.effective_user.id
    loc = update.message.location

    if not LOCATION_SUPPORT or tf is None:
        await update.message.reply_text(
            "⚠️ Fitur deteksi lokasi belum aktif di bot ini.\n"
            "Gunakan /settimezone untuk set timezone manual."
        )
        return

    tz_name = tf.timezone_at(lat=loc.latitude, lng=loc.longitude)

    if tz_name:
        user_timezones[user_id] = tz_name

        # Hapus cache lama
        keys_to_del = [k for k in garmin_cache if k[0] == user_id]
        for k in keys_to_del:
            del garmin_cache[k]

        tz = ZoneInfo(tz_name)
        now_local = datetime.now(tz)

        await update.message.reply_text(
            f"📍 <b>Lokasi terdeteksi!</b>\n\n"
            f"✅ Timezone: <code>{tz_name}</code>\n"
            f"⏰ Waktu lokal kamu:\n"
            f"<b>{now_local.strftime('%d %b %Y %H:%M %Z')}</b>\n\n"
            f"Data Garmin akan ditampilkan dalam timezone ini.",
            parse_mode=ParseMode.HTML
        )
    else:
        await update.message.reply_text(
            "❌ Gagal mendeteksi timezone dari lokasi ini.\n"
            "Coba /settimezone untuk set manual."
        )


async def cmd_ringkasan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    pesan = await update.message.reply_text("▌")
    data = await fetch_garmin_data(user_id)
    await stream_groq_to_telegram(
        user_id,
        "Kasih analisis lengkap aktivitas lari aku 30 hari terakhir dengan format section yang rapi (statistik utama, detail aktivitas, performa tubuh, dan insight).",
        data,
        pesan,
        context
    )


async def cmd_cekkoneksi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )
    pesan = await update.message.reply_text("🔍 Mengecek koneksi...")
    hasil = await call_mcp("list_activities", {"limit": 1})

    if "Error" in hasil:
        await pesan.edit_text(f"❌ Gagal terhubung:\n{hasil}")
    else:
        await pesan.edit_text(
            "✅ <b>Koneksi Garmin berhasil!</b>\nData siap dianalisis.",
            parse_mode=ParseMode.HTML
        )


async def handle_pesan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    pertanyaan = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    intent = await classify_intent(pertanyaan)
    print(f"Intent: {intent} | Pesan: {pertanyaan[:50]}")

    pesan = await update.message.reply_text("▌")

    if intent == "sapaan":
        await stream_groq_to_telegram(user_id, pertanyaan, "", pesan, context)
    elif intent == "lain":
        await stream_groq_to_telegram(
            user_id,
            pertanyaan + "\n\n[Note: User nanya hal di luar konteks lari. Jawab ramah dan arahkan balik ke topik lari.]",
            "",
            pesan,
            context
        )
    elif intent == "follow_up":
        await stream_groq_to_telegram(user_id, pertanyaan, "", pesan, context)
    else:
        data = await fetch_garmin_data(user_id)
        await stream_groq_to_telegram(user_id, pertanyaan, data, pesan, context)


# ── Main ──────────────────────────────────────────────────────
def main():
    print("🤖 Running Assistant Bot berjalan...")
    print(f"   Chat model:       {MODEL_CHAT}")
    print(f"   Classifier model: {MODEL_CLASSIFIER}")
    print(f"   Default timezone: {DEFAULT_TZ}")
    print(f"   Deteksi lokasi:   {'✓ aktif' if LOCATION_SUPPORT else '✗ nonaktif (install timezonefinder)'}")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",        cmd_start))
    app.add_handler(CommandHandler("ringkasan",    cmd_ringkasan))
    app.add_handler(CommandHandler("cekkoneksi",   cmd_cekkoneksi))
    app.add_handler(CommandHandler("reset",        cmd_reset))
    app.add_handler(CommandHandler("model",        cmd_model))
    app.add_handler(CommandHandler("settimezone",  cmd_settimezone))
    app.add_handler(CommandHandler("mytimezone",   cmd_mytimezone))
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pesan))

    app.add_error_handler(error_handler)

    if CHAT_ID:
        app.job_queue.run_repeating(
            cek_aktivitas_baru, interval=CHECK_INTERVAL, first=10
        )
        print(f"✓ Auto-monitoring aktif (cek tiap {CHECK_INTERVAL}s)")
    else:
        print("⚠ CHAT_ID belum di-set, auto-notif nonaktif")

    app.run_polling()


if __name__ == "__main__":
    main()