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
STREAM_INTERVAL = 1.5  # naikin dikit biar lebih aman
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)
reported_activities = set()


# ── Convert markdown → HTML untuk Telegram ────────────────────
def markdown_to_html(text: str) -> str:
    """
    Convert markdown standar ke HTML Telegram.
    Telegram HTML support: <b>, <i>, <u>, <s>, <code>, <pre>, <a>
    """
    # Escape HTML special chars dulu (penting!)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # **bold** atau __bold__ → <b>bold</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text, flags=re.DOTALL)

    # *italic* → <i>italic</i> (hati-hati: harus single asterisk yang BUKAN list)
    # Pattern: * yang diapit char non-spasi
    text = re.sub(r'(?<![*\w])\*([^\*\n]+?)\*(?!\*)', r'<i>\1</i>', text)

    # `code` → <code>code</code>
    text = re.sub(r'`([^`\n]+?)`', r'<code>\1</code>', text)

    return text


def safe_html_for_streaming(text: str) -> str:
    """
    Untuk streaming: kalau text masih on-going, mungkin ada tag yang belum ditutup.
    Fungsi ini convert ke HTML, terus tutup tag yang masih buka.
    """
    html = markdown_to_html(text)

    # Hitung tag yang belum ditutup
    open_b = html.count("<b>") - html.count("</b>")
    open_i = html.count("<i>") - html.count("</i>")
    open_code = html.count("<code>") - html.count("</code>")

    # Tutup yang masih kebuka
    if open_code > 0:
        html += "</code>" * open_code
    if open_i > 0:
        html += "</i>" * open_i
    if open_b > 0:
        html += "</b>" * open_b

    return html


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


# ── Streaming Groq ke Telegram (HTML mode) ───────────────────
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
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah asisten pelatih lari pribadi yang cerdas. "
                        "Jawab berdasarkan data Garmin yang diberikan. "
                        "Gunakan Bahasa Indonesia yang ramah dan motivatif. "
                        "Berikan insight berguna tentang performa lari.\n\n"
                        "FORMAT PENTING:\n"
                        "- Gunakan **bold** (double asterisk) untuk judul/poin penting\n"
                        "- Gunakan bullet '•' untuk list (BUKAN tanda '*')\n"
                        "- Pisahkan section dengan baris kosong\n"
                        "- JANGAN pakai single asterisk '*' untuk list, gunakan '•'"
                    )
                },
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
                    # Kalau HTML masih invalid, tunggu chunk berikutnya
                    if "can't parse entities" in str(e).lower():
                        pass
                except Exception as e:
                    print(f"Stream edit error: {e}")

    # Final update — tanpa cursor
    if full_text:
        final_html = markdown_to_html(full_text)
        try:
            await message.edit_text(final_html, parse_mode=ParseMode.HTML)
        except BadRequest as e:
            if "can't parse entities" in str(e).lower():
                # Last resort: kirim plain text TANPA tanda markdown
                clean_text = re.sub(r'\*+', '', full_text)
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
                    {
                        "role": "system",
                        "content": (
                            "Kamu pelatih lari yang ramah. Jawab dalam Bahasa Indonesia. "
                            "Gunakan **bold** untuk poin penting dan '•' untuk list."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Data Garmin terbaru:\n{hasil}\n\n"
                            "Ada aktivitas lari baru! Berikan ringkasan singkat dan motivasi. "
                            "Sebutkan jarak, pace, dan durasinya."
                        )
                    }
                ],
                max_tokens=512,
                temperature=0.7
            )
            ringkasan = response.choices[0].message.content
            html_text = f"🏃 <b>Aktivitas Lari Baru Terdeteksi!</b>\n\n{markdown_to_html(ringkasan)}"

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
                    text=f"🏃 Aktivitas Lari Baru Terdeteksi!\n\n{clean}"
                )

    except Exception as e:
        print(f"Error cek aktivitas: {e}")


# ── Handlers ──────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nama = update.effective_user.first_name
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"👟 Halo <b>{nama}</b>! Saya bot pelatih larimu.\n\n"
        f"📌 Chat ID kamu: <code>{chat_id}</code>\n"
        "(set sebagai env <code>CHAT_ID</code> di Railway untuk auto-notif)\n\n"
        "Tanya apapun tentang aktivitas Garmin kamu:\n\n"
        "• Berapa total lari saya bulan ini?\n"
        "• Gimana pace rata-rata saya?\n"
        "• Aktivitas terakhir saya apa?\n\n"
        "/ringkasan — ringkasan 30 hari terakhir\n"
        "/cekkoneksi — cek koneksi Garmin",
        parse_mode=ParseMode.HTML
    )


async def cmd_ringkasan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

    pesan = await update.message.reply_text("▌")

    data = await fetch_garmin_data()

    await stream_groq_to_telegram(
        "Berikan ringkasan lengkap aktivitas lari 30 hari terakhir: "
        "total jarak, jumlah sesi, pace terbaik, kalori, dan tren performa.",
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
        await pesan.edit_text("✅ Koneksi ke Garmin berhasil! Data bisa dibaca.")


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
    print("🤖 Running Assistant Bot berjalan (HTML streaming)...")
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