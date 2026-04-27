import httpx
import os
import re
import json
import asyncio
from datetime import date, timedelta
from telegram import Update
from telegram.constants import ChatAction
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
CHECK_INTERVAL = 300   # cek aktivitas baru tiap 5 menit
STREAM_INTERVAL = 1.2  # interval edit message saat streaming (detik)
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)
reported_activities = set()


# ── Helper: Convert markdown standar ke format Telegram ───────
def convert_markdown(text: str) -> str:
    """
    Convert markdown standar (**bold**) ke Telegram markdown (*bold*).
    Telegram pakai single asterisk untuk bold, bukan double.
    """
    # **text** -> *text*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    return text


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


# ── Streaming Groq response ke Telegram ──────────────────────
async def stream_groq_to_telegram(
    pertanyaan: str,
    data_garmin: str,
    message,
    context: ContextTypes.DEFAULT_TYPE
) -> str:
    """Stream response Groq ke Telegram dengan markdown rendering."""

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
                        "Berikan insight berguna tentang performa lari. "
                        "Format jawaban dengan rapi: gunakan *bold* untuk judul/poin penting "
                        "(single asterisk, BUKAN double). Pakai bullet point '•' untuk list."
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
    last_sent_text = ""

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
        if now - last_update_time >= STREAM_INTERVAL and full_text != last_sent_text:
            try:
                display_text = convert_markdown(full_text) + " ▌"
                await message.edit_text(display_text, parse_mode="Markdown")
                last_sent_text = full_text
                last_update_time = now

                # Refresh typing indicator
                await context.bot.send_chat_action(
                    chat_id=message.chat_id,
                    action=ChatAction.TYPING
                )
            except RetryAfter as e:
                await asyncio.sleep(e.retry_after)
            except BadRequest as e:
                # Markdown invalid (tag belum tutup) → fallback plain text
                if "can't parse entities" in str(e).lower():
                    try:
                        await message.edit_text(full_text + " ▌")
                        last_sent_text = full_text
                        last_update_time = now
                    except BadRequest:
                        pass
                # Message belum berubah → skip
            except Exception as e:
                print(f"Stream edit error: {e}")

    # Final update tanpa cursor
    if full_text:
        try:
            final_text = convert_markdown(full_text)
            await message.edit_text(final_text, parse_mode="Markdown")
        except BadRequest as e:
            if "can't parse entities" in str(e).lower():
                # Markdown invalid → kirim plain text
                try:
                    await message.edit_text(full_text)
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

        # Initial scan: tandai semua yang ada sekarang sebagai "udah dilihat"
        if not reported_activities:
            reported_activities.add(hasil[:200])
            print("✓ Initial scan selesai, monitoring aktivitas baru...")
            return

        signature = hasil[:200]
        if signature not in reported_activities:
            reported_activities.add(signature)

            # Generate ringkasan (non-streaming untuk auto-notif)
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Kamu pelatih lari yang ramah dan motivatif. "
                            "Jawab dalam Bahasa Indonesia. Gunakan *bold* (single asterisk) "
                            "untuk poin penting."
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

            try:
                await context.bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"🏃 *Aktivitas Lari Baru Terdeteksi!*\n\n{convert_markdown(ringkasan)}",
                    parse_mode="Markdown"
                )
            except BadRequest:
                # Fallback plain text kalau markdown error
                await context.bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"🏃 Aktivitas Lari Baru Terdeteksi!\n\n{ringkasan}"
                )

    except Exception as e:
        print(f"Error cek aktivitas: {e}")


# ── Handlers ──────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nama = update.effective_user.first_name
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"👟 Halo {nama}! Saya bot pelatih larimu.\n\n"
        f"📌 Chat ID kamu: `{chat_id}`\n"
        "(set sebagai env `CHAT_ID` di Railway untuk auto-notif)\n\n"
        "Tanya apapun tentang aktivitas Garmin kamu:\n\n"
        "• Berapa total lari saya bulan ini?\n"
        "• Gimana pace rata-rata saya?\n"
        "• Aktivitas terakhir saya apa?\n\n"
        "/ringkasan — ringkasan 30 hari terakhir\n"
        "/cekkoneksi — cek koneksi Garmin",
        parse_mode="Markdown"
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
    print("🤖 Bot Garmin + Groq berjalan (streaming + markdown)...")
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