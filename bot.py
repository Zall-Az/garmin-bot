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
from groq import Groq, RateLimitError, APIError

# ============================================================
# KONFIGURASI
# ============================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
MCP_URL        = "https://garmin.amalgama.co/api/v1/mcp/48247257-4554-43df-9e93-e7dd3710c58a"
CHAT_ID        = os.environ.get("CHAT_ID")

# ⭐ MODEL CONFIG (gampang ganti di sini)
MODEL_CHAT       = "llama-3.3-70b-versatile"    # Model utama untuk jawab user (stabil, anti-halu)
MODEL_CLASSIFIER = "llama-3.1-8b-instant"       # Model kecil untuk classify intent

# Temperature (rendah = lebih akurat, tidak mengarang)
TEMPERATURE_CHAT = 0.3

# Tuning parameters
CHECK_INTERVAL   = 600   # auto-cek aktivitas tiap 10 menit
STREAM_INTERVAL  = 1.5   # interval edit pesan saat streaming
MAX_HISTORY      = 8     # max pesan history per user
GARMIN_CACHE_TTL = 300   # cache data Garmin 5 menit

# Timezone WITA = UTC+8
WITA = timezone(timedelta(hours=8))
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)
reported_activities = set()
conversation_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

# Cache data Garmin per user
garmin_cache = {}  # {user_id: (timestamp, data)}


# ── Konversi UTC → WITA ───────────────────────────────────────
def convert_utc_to_wita(text: str) -> str:
    """Konversi semua timestamp UTC dalam teks ke WITA (UTC+8)."""

    MONTHS = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    # Format ISO 8601: "2026-04-26T20:54:00.000Z" atau "2026-04-26T20:54:00Z"
    def replace_iso(m):
        try:
            dt = datetime.fromisoformat(m.group(0).replace("Z", "+00:00"))
            wita = dt.astimezone(WITA)
            return wita.strftime("%-d %b %Y %H:%M WITA")
        except Exception:
            return m.group(0)

    text = re.sub(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z',
        replace_iso, text
    )

    # Format "26 Apr 2026 20:54 UTC"
    def replace_readable(m):
        try:
            day, mon, year, hour, minute = m.groups()
            dt = datetime(
                int(year), MONTHS[mon], int(day),
                int(hour), int(minute),
                tzinfo=timezone.utc
            )
            wita = dt.astimezone(WITA)
            return wita.strftime("%-d %b %Y %H:%M WITA")
        except Exception:
            return m.group(0)

    text = re.sub(
        r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\s+(\d{4})\s+(\d{2}):(\d{2})\s+UTC',
        replace_readable, text
    )

    # Format "2026-04-26 20:54:00 UTC" (dengan spasi, bukan T)
    def replace_space_iso(m):
        try:
            dt_str = m.group(0).replace(" UTC", "+00:00").replace(" ", "T", 1)
            dt = datetime.fromisoformat(dt_str)
            wita = dt.astimezone(WITA)
            return wita.strftime("%-d %b %Y %H:%M WITA")
        except Exception:
            return m.group(0)

    text = re.sub(
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC',
        replace_space_iso, text
    )

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
    open_b = html.count("<b>") - html.count("</b>")
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


COACH_SYSTEM_PROMPT = """Kamu adalah Running Assistant, pelatih lari pribadi yang ramah dan conversational.

KARAKTER:
- Ramah seperti teman, bukan robot formal
- Bahasa Indonesia santai tapi profesional
- Inget konteks percakapan sebelumnya
- Motivatif dan supportive

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
   - Waktu: gunakan WITA (UTC+8). Format: "26 Apr 2026 pukul 04:54 WITA"

5. FORMATTING:
   - Bold pakai **double asterisk** untuk angka penting
   - Bullet pakai • (BUKAN * atau -)
   - Pisahkan section dengan baris kosong

6. KONTEKS:
   - Inget percakapan sebelumnya
   - Kalau user nanya "kenapa?" atau "terus?", lanjutkan dari topik tadi
   - Jangan ulang info yang udah dikasih sebelumnya

7. ANTI-HALUSINASI (WAJIB!):
   - JANGAN PERNAH mengarang angka atau data yang tidak ada di data Garmin
   - Kalau data Garmin tidak tersedia atau tidak relevan, bilang jujur:
     "Data tidak tersedia, coba tanya yang lebih spesifik ya!"
   - HANYA gunakan angka yang benar-benar ada di data yang diberikan
   - Kalau tidak yakin, tanya balik ke user daripada mengarang"""


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


# ── Ambil data Garmin (DENGAN CACHE) ──────────────────────────
async def fetch_garmin_data(user_id: int = 0) -> str:
    """Fetch data Garmin dengan cache 5 menit per user."""
    now = time.time()

    # Cek cache
    if user_id in garmin_cache:
        cached_time, cached_data = garmin_cache[user_id]
        if now - cached_time < GARMIN_CACHE_TTL:
            print(f"✓ Pakai cache Garmin (age: {int(now - cached_time)}s)")
            return cached_data

    # Fetch baru
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

    data = f"=== AKTIVITAS 30 HARI TERAKHIR ===\n{aktivitas}\n\n=== STATISTIK ===\n{stats}"

    # ✅ Konversi semua timestamp UTC → WITA sebelum disimpan ke cache
    data = convert_utc_to_wita(data)

    garmin_cache[user_id] = (now, data)
    print(f"✓ Fetch Garmin baru, cached (timestamp sudah WITA)")

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
    """Stream response dengan auto-retry kalau rate limit."""

    messages = [{"role": "system", "content": COACH_SYSTEM_PROMPT}]
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
                error_msg = str(e)
                match = re.search(r'try again in ([\d.]+)s', error_msg)
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
                await message.edit_text(
                    "❌ Ada error dari server AI. Coba lagi sebentar ya!"
                )
            except:
                pass
            return ""
        except Exception as e:
            print(f"Stream error: {e}")
            try:
                await message.edit_text(
                    f"❌ Terjadi error: {str(e)[:100]}\nCoba lagi ya!"
                )
            except:
                pass
            return ""

    return ""


async def _do_stream(user_id, user_message, messages, message, context):
    """Helper internal untuk streaming."""

    def get_stream():
        return groq_client.chat.completions.create(
            model=MODEL_CHAT,
            messages=messages,
            max_tokens=1024,
            temperature=TEMPERATURE_CHAT,
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
                        chat_id=message.chat_id,
                        action=ChatAction.TYPING
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

    # Simpan ke history
    conversation_history[user_id].append({"role": "user", "content": user_message})
    conversation_history[user_id].append({"role": "assistant", "content": full_text})

    return full_text


# ── Cek aktivitas baru (auto-notif) ───────────────────────────
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

        # ✅ Konversi timestamp UTC → WITA di notif otomatis juga
        hasil = convert_utc_to_wita(hasil)

        if not reported_activities:
            reported_activities.add(hasil[:200])
            print("✓ Initial scan selesai...")
            return

        signature = hasil[:200]
        if signature not in reported_activities:
            reported_activities.add(signature)

            try:
                response = groq_client.chat.completions.create(
                    model=MODEL_CHAT,
                    messages=[
                        {"role": "system", "content": COACH_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Data Garmin terbaru:\n{hasil}\n\n"
                                "Ada aktivitas lari baru baru aja kedeteksi! Kasih ringkasan singkat "
                                "(jarak km, pace, durasi jam:menit) plus motivasi pendek."
                            )
                        }
                    ],
                    max_tokens=400,
                    temperature=TEMPERATURE_CHAT
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
    """Tangani semua error yang lolos."""
    error = context.error

    # Skip Conflict error (instance ganda) biar log nggak penuh
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
        f"/cekkoneksi — tes Garmin\n"
        f"/reset — reset percakapan\n"
        f"/model — info model AI"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.HTML)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_history[user_id].clear()
    if user_id in garmin_cache:
        del garmin_cache[user_id]
    await update.message.reply_text("✅ Percakapan di-reset. Mulai fresh dari sini ya!")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Info model yang dipakai bot."""
    info = (
        f"🤖 <b>Model AI Info</b>\n\n"
        f"• <b>Chat utama:</b> <code>{MODEL_CHAT}</code>\n"
        f"• <b>Classifier:</b> <code>{MODEL_CLASSIFIER}</code>\n\n"
        f"<i>Powered by Groq</i>"
    )
    await update.message.reply_text(info, parse_mode=ParseMode.HTML)


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
        # tanya_spesifik / analisis_lengkap / saran_latihan
        data = await fetch_garmin_data(user_id)
        await stream_groq_to_telegram(user_id, pertanyaan, data, pesan, context)


# ── Main ──────────────────────────────────────────────────────
def main():
    print("🤖 Running Assistant Bot berjalan...")
    print(f"   Chat model:       {MODEL_CHAT}")
    print(f"   Classifier model: {MODEL_CLASSIFIER}")
    print(f"   Timezone output:  WITA (UTC+8)")

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
        print(f"✓ Auto-monitoring aktif (cek tiap {CHECK_INTERVAL}s)")
    else:
        print("⚠ CHAT_ID belum di-set, auto-notif nonaktif")

    app.run_polling()


if __name__ == "__main__":
    main()