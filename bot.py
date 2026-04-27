import httpx
import os
import re
import json
import asyncio
from datetime import date, timedelta
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
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)
reported_activities = set()


# ── Convert markdown ke HTML Telegram ─────────────────────────
def markdown_to_html(text: str) -> str:
    """Convert markdown standar ke HTML Telegram."""
    # Escape HTML special chars
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Convert "* item" / "- item" di awal baris → "• item"
    text = re.sub(r'^\s*[\*\-]\s+', '• ', text, flags=re.MULTILINE)

    # **bold** → <b>bold</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text, flags=re.DOTALL)

    # `code` → <code>code</code>
    text = re.sub(r'`([^`\n]+?)`', r'<code>\1</code>', text)

    # Hapus single asterisk yang nyangkut
    text = re.sub(r'(?<!\w)\*(?!\w)', '', text)

    return text


def safe_html_for_streaming(text: str) -> str:
    """Untuk streaming - tutup tag HTML yang masih kebuka."""
    html = markdown_to_html(text)

    open_b = html.count("<b>") - html.count("</b>")
    open_code = html.count("<code>") - html.count("</code>")

    if open_code > 0:
        html += "</code>" * open_code
    if open_b > 0:
        html += "</b>" * open_b

    return html


# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah pelatih lari pribadi yang ramah dan cerdas.
Jawab berdasarkan data Garmin yang diberikan dalam Bahasa Indonesia.

═══ ATURAN FORMAT WAJIB ═══

1. STRUKTUR JAWABAN:
   - Mulai dengan sapaan singkat 1 kalimat
   - Bagi dalam SECTION dengan judul emoji (lihat contoh di bawah)
   - Tutup dengan motivasi/saran 1-2 kalimat

2. JUDUL SECTION (pakai emoji + bold):
   📊 **Statistik Utama**
   🏃 **Detail Aktivitas**
   ❤️ **Performa Tubuh**
   💡 **Insight & Saran**

3. FORMATTING TEXT:
   - Bold pakai **double asterisk** untuk angka penting dan nama aktivitas
   - Bullet list pakai • (bullet character), JANGAN pakai * atau -
   - Pisahkan section dengan baris kosong

4. ATURAN ANGKA (PENTING!):
   - Jarak: HANYA km dengan 1-2 desimal. Contoh: "12,5 km"
     ❌ JANGAN: "12500 meter" atau "12500 m" atau "12,5 km (12500 meter)"
   - Durasi: format jam dan menit. Contoh: "23 jam 30 menit"
     ❌ JANGAN: "84769 detik" atau "84769 s"
   - Pace: format menit:detik per km. Contoh: "5:30/km"
   - Heart rate: bulatkan, tambah 'bpm'. Contoh: "136 bpm"
   - Kalori: tambah "kcal". Contoh: "8.431 kcal"
   - JANGAN PERNAH tampilkan satuan raw (meter, detik) di output

5. CONTOH JAWABAN YANG BENAR:

Halo! Berikut analisis performa lari kamu 30 hari terakhir.

📊 **Statistik Utama**
- Total aktivitas: **24 sesi**
- Total jarak: **124,3 km**
- Total durasi: **23 jam 30 menit**
- Kalori terbakar: **8.431 kcal**

🏃 **Detail Aktivitas**
- Lari: **16 sesi** (103,4 km)
- Strength training: **5 sesi**
- Padel: **1 sesi**
- Jalan kaki: **2 sesi**

❤️ **Performa Tubuh**
- HR rata-rata: **136 bpm**
- HR maksimum: **174 bpm**

💡 **Insight**
Konsistensi kamu keren banget! Coba tingkatkan jarak per sesi minggu depan untuk progress yang lebih baik.

═══ JAWABAN YANG SALAH (JANGAN!) ═══
❌ "124252,71 meter (sekitar 124,25 km)"  → harus "124,3 km" saja
❌ "* item list"  → harus "• item list"
❌ "84769 detik (sekitar 23,5 jam)"  → harus "23 jam 30 menit" saja
"""


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


# ── Ambil data Garmin lengkap ─────────────────────────────────
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


# ── Streaming Groq ke Telegram ───────────────────────────────
async def stream_groq_to_telegram(
    pertanyaan: str,
    data_garmin: str,
    message,
    context: ContextTypes.DEFAULT_TYPE
) -> str:
    """Stream response Groq ke Telegram dengan HTML rendering."""

    def get_stream():
        return groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Data Garmin saya:\n{data_garmin}\n\nPertanyaan: {pertanyaan}"
                }
            ],
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
                    await message.edit_text(
                        html_text,
                        parse_mode=ParseMode.HTML
                    )
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

    # Final update tanpa cursor
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
            print("✓ Initial scan selesai, monitoring aktivitas baru...")
            return

        signature = hasil[:200]
        if signature not in reported_activities:
            reported_activities.add(signature)

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Data Garmin terbaru:\n{hasil}\n\n"
                            "Ada aktivitas lari baru! Berikan ringkasan singkat dan motivasi. "
                            "Sebutkan jarak (km), pace, dan durasi (jam:menit)."
                        )
                    }
                ],
                max_tokens=512,
                temperature=0.7
            )
            ringkasan = response.choices[0].message.content
            html_text = f"🏃 <b>Aktivitas Lari Baru!</b>\n\n{markdown_to_html(ringkasan)}"

            try:
                await context.bot.send_message(
                    chat_id=CHAT_ID,
                    text=html_text,
                    parse_mode=ParseMode.HTML
                )
            except BadRequest:
                clean = re.sub(r'\*+', '', ringkasan)
                await context.bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"🏃 Aktivitas Lari Baru!\n\n{clean}"
                )

    except Exception as e:
        print(f"Error cek aktivitas: {e}")


# ── Handlers ──────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nama = update.effective_user.first_name
    chat_id = update.effective_chat.id
    welcome = (
        f"👟 Halo <b>{nama}</b>!\n\n"
        f"Saya <b>Running Assistant</b>, pelatih lari pribadimu yang siap "
        f"menganalisis data Garmin kamu kapan aja.\n\n"
        f"📌 Chat ID: <code>{chat_id}</code>\n\n"
        f"<b>Coba tanya apa aja:</b>\n"
        f"• Berapa total lari saya bulan ini?\n"
        f"• Gimana pace rata-rata saya?\n"
        f"• Aktivitas terakhir saya apa?\n"
        f"• Kasih saran latihan dong\n\n"
        f"<b>Command:</b>\n"
        f"/ringkasan — analisis lengkap 30 hari\n"
        f"/cekkoneksi — tes koneksi Garmin"
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.HTML)


async def cmd_ringkasan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

    pesan = await update.message.reply_text("▌")
    data = await fetch_garmin_data()

    await stream_groq_to_telegram(
        "Berikan ringkasan lengkap aktivitas lari 30 hari terakhir dengan format "
        "section yang rapi (statistik utama, detail aktivitas, performa tubuh, dan insight).",
        data,
        pesan,
        context
    )


async def cmd_cekkoneksi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )
    pesan = await update.message.reply_text("🔍 Mengecek koneksi...")
    hasil = await call_mcp("list_activities", {"limit": 1})

    if "Error" in hasil:
        await pesan.edit_text(f"❌ Gagal terhubung:\n{hasil}")
    else:
        await pesan.edit_text("✅ <b>Koneksi Garmin berhasil!</b>\nData siap dianalisis.", parse_mode=ParseMode.HTML)


async def handle_pesan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pertanyaan = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

    pesan = await update.message.reply_text("▌")
    data = await fetch_garmin_data()
    await stream_groq_to_telegram(pertanyaan, data, pesan, context)


# ── Main ──────────────────────────────────────────────────────
def main():
    print("🤖 Running Assistant Bot berjalan...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("ringkasan",   cmd_ringkasan))
    app.add_handler(CommandHandler("cekkoneksi",  cmd_cekkoneksi))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pesan))

    if CHAT_ID:
        app.job_queue.run_repeating(
            cek_aktivitas_baru,
            interval=CHECK_INTERVAL,
            first=10
        )
        print(f"✓ Auto-monitoring aktif (cek tiap {CHECK_INTERVAL}s)")
    else:
        print("⚠ CHAT_ID belum di-set, auto-notif nonaktif")

    app.run_polling()


if __name__ == "__main__":
    main()