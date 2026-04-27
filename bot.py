import httpx
import os
import re
import json
import asyncio
from datetime import date, timedelta
from collections import defaultdict, deque
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes
)
from groq import Groq

# ============================================================
# KONFIGURASI
# ============================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
MCP_URL        = "https://garmin.amalgama.co/api/v1/mcp/48247257-4554-43df-9e93-e7dd3710c58a"
CHAT_ID        = os.environ.get("CHAT_ID")
CHECK_INTERVAL = 300
STREAM_INTERVAL = 1.5
MAX_HISTORY = 10  # max history per user
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)
reported_activities = set()

# Conversation history per user (in-memory)
# Format: {user_id: deque([{role, content}, ...])}
conversation_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))


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
INTENT_CLASSIFIER_PROMPT = """Kamu adalah classifier untuk pesan pengguna ke bot pelatih lari.
Klasifikasikan pesan user ke salah satu kategori berikut:

- "sapaan": Pesan kasual seperti "halo", "hi", "pagi", "apa kabar", "gimana", terima kasih, dll. TANPA pertanyaan tentang data lari.
- "tanya_spesifik": Pertanyaan tentang ASPEK TERTENTU dari data lari (misal: "berapa pace saya?", "denyut nadi rata-rata?", "lari terjauh kapan?", "minggu ini lari berapa kali?")
- "analisis_lengkap": Permintaan analisis menyeluruh (misal: "analisis dong", "kasih ringkasan", "evaluasi performa saya", "gimana progress saya bulan ini?")
- "saran_latihan": Minta saran/rekomendasi (misal: "latihan apa yang cocok?", "saran dong", "gimana cara improve?")
- "follow_up": Pertanyaan lanjutan dari percakapan sebelumnya (misal: "kenapa begitu?", "terus gimana?", "kasih detail dong")
- "lain": Pertanyaan di luar konteks lari/Garmin

Jawab HANYA dengan satu kata kategori, tanpa penjelasan apapun."""


COACH_SYSTEM_PROMPT = """Kamu adalah pelatih lari pribadi yang ramah, cerdas, dan conversational. Namamu Running Assistant.

KARAKTER:
- Ramah seperti teman, bukan robot formal
- Bahasa Indonesia santai tapi profesional
- Motivatif dan supportive
- Inget konteks percakapan sebelumnya

ATURAN PENTING:

1. SAPAAN/CASUAL CHAT:
   - Balas natural dan singkat (1-3 kalimat)
   - JANGAN langsung kasih data atau analisis
   - Tanya apa yang bisa dibantu
   - Contoh: "Halo! Apa kabar? Ada yang mau dianalisis dari aktivitas lari kamu?"

2. PERTANYAAN SPESIFIK (denyut nadi, pace, jarak, dll):
   - Jawab LANGSUNG dan SINGKAT, fokus ke yang ditanya saja
   - JANGAN kasih full report kalau cuma ditanya 1 hal
   - Contoh: "HR rata-rata kamu 30 hari terakhir adalah 136 bpm. Tergolong intensitas sedang ya 👍"

3. ANALISIS LENGKAP (kalau diminta):
   - Pakai struktur lengkap dengan section + emoji
   - Format:
     📊 **Statistik Utama**
     🏃 **Detail Aktivitas**
     ❤️ **Performa Tubuh**
     💡 **Insight & Saran**

4. ATURAN ANGKA:
   - Jarak: HANYA km (1-2 desimal). Contoh: "12,5 km"
   - Durasi: jam dan menit. Contoh: "23 jam 30 menit"
   - Pace: menit:detik per km. Contoh: "5:30/km"
   - Heart rate: tambah "bpm". Contoh: "136 bpm"
   - JANGAN tampilkan satuan raw (meter, detik)

5. FORMATTING:
   - Bold pakai **double asterisk** untuk angka penting
   - Bullet pakai • (BUKAN * atau -)

6. KONTEKS:
   - Inget percakapan sebelumnya
   - Kalau user nanya "kenapa?" atau "terus?", lanjutkan dari topik tadi
   - Jangan ulang info yang udah dikasih sebelumnya
"""


# ── Klasifikasi intent (ringan, cepat) ────────────────────────
async def classify_intent(message: str) -> str:
    """Tentuin tipe pesan: sapaan / tanya_spesifik / analisis_lengkap / saran_latihan / follow_up / lain"""
    try:
        loop = asyncio.get_event_loop()
        
        def _classify():
            return groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # model kecil & cepat untuk classify
                messages=[
                    {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
                    {"role": "user", "content": message}
                ],
                max_tokens=20,
                temperature=0.1
            )
        
        response = await loop.run_in_executor(None, _classify)
        intent = response.choices[0].message.content.strip().lower()
        
        # Ekstrak kategori dari response
        valid = ["sapaan", "tanya_spesifik", "analisis_lengkap", "saran_latihan", "follow_up", "lain"]
        for v in valid:
            if v in intent:
                return v
        return "tanya_spesifik"  # default
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


# ── Ambil data Garmin ─────────────────────────────────────────
async def fetch_garmin_data() -> str:
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

    return f"=== AKTIVITAS 30 HARI TERAKHIR ===\n{aktivitas}\n\n=== STATISTIK ===\n{stats}"


# ── Streaming Groq ke Telegram (dengan history) ──────────────
async def stream_groq_to_telegram(
    user_id: int,
    user_message: str,
    extra_context: str,
    message,
    context: ContextTypes.DEFAULT_TYPE
) -> str:
    """Stream response dengan conversation history."""
    
    # Bangun messages dari history + context
    messages = [{"role": "system", "content": COACH_SYSTEM_PROMPT}]
    
    # Inject history sebelumnya
    history = list(conversation_history[user_id])
    messages.extend(history)
    
    # Tambah pesan user saat ini (dengan extra context kalau ada)
    if extra_context:
        user_content = f"{user_message}\n\n[Data Garmin tersedia:]\n{extra_context}"
    else:
        user_content = user_message
    messages.append({"role": "user", "content": user_content})

    def get_stream():
        return groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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

    # Final update
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

        if not reported_activities:
            reported_activities.add(hasil[:200])
            print("✓ Initial scan selesai...")
            return

        signature = hasil[:200]
        if signature not in reported_activities:
            reported_activities.add(signature)

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": COACH_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Data Garmin terbaru:\n{hasil}\n\n"
                            "Aktivitas lari baru baru aja kedeteksi! Kasih ringkasan singkat "
                            "(jarak km, pace, durasi jam:menit) plus motivasi pendek."
                        )
                    }
                ],
                max_tokens=400,
                temperature=0.7
            )
            ringkasan = response.choices[0].message.content
            html_text = f"🏃 <b>Aktivitas Lari Baru!</b>\n\n{markdown_to_html(ringkasan)}"

            try:
                await context.bot.send_message(
                    chat_id=CHAT_ID, text=html_text, parse_mode=ParseMode.HTML
                )
            except BadRequest:
                clean = re.sub(r'\*+', '', ringkasan)
                await context.bot.send_message(
                    chat_id=CHAT_ID, text=f"🏃 Aktivitas Lari Baru!\n\n{clean}"
                )

    except Exception as e:
        print(f"Error cek aktivitas: {e}")


# ── Handlers ──────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nama = update.effective_user.first_name
    chat_id = update.effective_chat.id
    
    # Reset history
    conversation_history[update.effective_user.id].clear()
    
    welcome = (
        f"👟 Halo <b>{nama}</b>!\n\n"
        f"Aku <b>Running Assistant</b>, pelatih lari pribadi kamu. "
        f"Tanya apa aja seputar aktivitas Garmin kamu, aku siap bantu! 💪\n\n"
        f"📌 Chat ID: <code>{chat_id}</code>\n\n"
        f"<b>Contoh pertanyaan:</b>\n"
        f"• Berapa pace rata-rata saya?\n"
        f"• Lari terjauh saya kapan?\n"
        f"• Kasih analisis lengkap dong\n"
        f"• Saran latihan untuk improve\n\n"
        f"<b>Command:</b>\n"
        f"/ringkasan — analisis 30 hari\n"
        f"/cekkoneksi — tes Garmin\n"
        f"/reset — reset percakapan"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.HTML)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset conversation history user."""
    user_id = update.effective_user.id
    conversation_history[user_id].clear()
    await update.message.reply_text("✅ Percakapan di-reset. Mulai fresh dari sini ya!")


async def cmd_ringkasan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.TYPING
    )

    pesan = await update.message.reply_text("▌")
    data = await fetch_garmin_data()

    await stream_groq_to_telegram(
        user_id,
        "Kasih analisis lengkap aktivitas lari aku 30 hari terakhir dengan format section yang rapi.",
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

    # STEP 1: Klasifikasi intent dulu
    intent = await classify_intent(pertanyaan)
    print(f"Intent: {intent} | Pesan: {pertanyaan[:50]}")

    # STEP 2: Tentuin perlu fetch data atau nggak
    pesan = await update.message.reply_text("▌")
    
    if intent == "sapaan":
        # Sapaan biasa - nggak perlu data Garmin
        await stream_groq_to_telegram(
            user_id,
            pertanyaan,
            "",  # nggak ada extra context
            pesan,
            context
        )
    elif intent == "lain":
        # Pertanyaan di luar konteks - balas tanpa data
        await stream_groq_to_telegram(
            user_id,
            pertanyaan + "\n\n[Note: User nanya hal di luar konteks lari. Jawab ramah dan arahkan balik ke topik lari.]",
            "",
            pesan,
            context
        )
    elif intent == "follow_up":
        # Follow-up - pakai history aja, biasanya cukup
        await stream_groq_to_telegram(
            user_id,
            pertanyaan,
            "",  # history udah ada di conversation_history
            pesan,
            context
        )
    else:
        # tanya_spesifik / analisis_lengkap / saran_latihan - perlu data
        data = await fetch_garmin_data()
        await stream_groq_to_telegram(
            user_id,
            pertanyaan,
            data,
            pesan,
            context
        )


# ── Main ──────────────────────────────────────────────────────
def main():
    print("🤖 Running Assistant Bot berjalan (conversational mode)...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("ringkasan",   cmd_ringkasan))
    app.add_handler(CommandHandler("cekkoneksi",  cmd_cekkoneksi))
    app.add_handler(CommandHandler("reset",       cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pesan))

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