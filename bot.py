import httpx
import os
import json
import asyncio
from datetime import date, timedelta
from telegram import Update
from telegram.constants import ChatAction
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
CHAT_ID        = os.environ.get("CHAT_ID")  # untuk auto-notif
CHECK_INTERVAL = 300  # cek aktivitas baru tiap 5 menit
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)

# Simpan signature aktivitas yang udah dilaporkan
reported_activities = set()


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


# ── Tanya Groq ────────────────────────────────────────────────
def tanya_groq(pertanyaan: str, data_garmin: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten pelatih lari pribadi yang cerdas. "
                    "Jawab berdasarkan data Garmin yang diberikan. "
                    "Gunakan Bahasa Indonesia yang ramah dan motivatif. "
                    "Berikan insight berguna tentang performa lari."
                )
            },
            {
                "role": "user",
                "content": f"Data Garmin saya:\n{data_garmin}\n\nPertanyaan: {pertanyaan}"
            }
        ],
        max_tokens=1024,
        temperature=0.7
    )
    return response.choices[0].message.content


# ── Helper: animasi titik berjalan ────────────────────────────
async def animate_dots(message, stop_event: asyncio.Event):
    """Animasi titik . → .. → ... yang loop sampai stop_event di-set."""
    dots = [".", "..", "..."]
    i = 0
    while not stop_event.is_set():
        try:
            await message.edit_text(dots[i % 3])
            i += 1
            # Tunggu 0.5 detik atau sampai stop
            await asyncio.wait_for(stop_event.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except Exception:
            break


# ── Cek aktivitas baru (auto) ─────────────────────────────────
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

        # Pas pertama kali jalan, anggap aktivitas existing udah dilihat
        if not reported_activities:
            reported_activities.add(hasil[:200])
            print("✓ Initial scan selesai, monitoring aktivitas baru...")
            return

        # Cek apakah ada perubahan (aktivitas baru)
        signature = hasil[:200]
        if signature not in reported_activities:
            reported_activities.add(signature)

            ringkasan = tanya_groq(
                "Ada aktivitas lari baru! Berikan ringkasan singkat dan motivasi "
                "untuk aktivitas terbaru saya. Sebutkan jarak, pace, dan durasinya.",
                hasil
            )

            await context.bot.send_message(
                chat_id=CHAT_ID,
                text=f"🏃 *Aktivitas Lari Baru Terdeteksi!*\n\n{ringkasan}",
                parse_mode="Markdown"
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
    # Kirim "..." awal
    pesan_proses = await update.message.reply_text(".")
    stop_event = asyncio.Event()
    animasi = asyncio.create_task(animate_dots(pesan_proses, stop_event))

    try:
        # Typing indicator di header
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )

        data = await fetch_garmin_data()
        jawaban = tanya_groq(
            "Berikan ringkasan lengkap aktivitas lari 30 hari terakhir: "
            "total jarak, jumlah sesi, pace terbaik, kalori, dan tren performa.",
            data
        )
    finally:
        stop_event.set()
        await animasi

    await pesan_proses.edit_text(
        f"📊 *Ringkasan 30 Hari*\n\n{jawaban}",
        parse_mode="Markdown"
    )


async def cmd_cekkoneksi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pesan_proses = await update.message.reply_text(".")
    stop_event = asyncio.Event()
    animasi = asyncio.create_task(animate_dots(pesan_proses, stop_event))

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )
        hasil = await call_mcp("list_activities", {"limit": 1})
    finally:
        stop_event.set()
        await animasi

    if "Error" in hasil:
        await pesan_proses.edit_text(f"❌ Gagal terhubung:\n{hasil}")
    else:
        await pesan_proses.edit_text("✅ Koneksi ke Garmin berhasil! Data bisa dibaca.")


async def handle_pesan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pertanyaan = update.message.text

    # Kirim "..." dan mulai animasi
    pesan_proses = await update.message.reply_text(".")
    stop_event = asyncio.Event()
    animasi = asyncio.create_task(animate_dots(pesan_proses, stop_event))

    try:
        # Typing indicator di header chat
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )

        data = await fetch_garmin_data()

        # Refresh typing indicator (karena fetch tadi makan waktu)
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING
        )

        jawaban = tanya_groq(pertanyaan, data)
    finally:
        # Stop animasi
        stop_event.set()
        await animasi

    # Edit pesan "..." jadi jawaban final
    await pesan_proses.edit_text(jawaban)


# ── Main ──────────────────────────────────────────────────────
def main():
    print("🤖 Bot Garmin + Groq berjalan...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("ringkasan",   cmd_ringkasan))
    app.add_handler(CommandHandler("cekkoneksi",  cmd_cekkoneksi))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pesan))

    # Auto-monitoring aktivitas baru
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