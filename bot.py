import httpx
import os
import json
from datetime import date, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from groq import Groq

# ============================================================
# KONFIGURASI
# ============================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY") # ← ganti ini
MCP_URL        = "https://garmin.amalgama.co/api/v1/mcp/48247257-4554-43df-9e93-e7dd3710c58a"
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)


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

    # Ambil 20 aktivitas terbaru bulan ini
    aktivitas = await call_mcp("list_activities", {
        "limit": 20,
        "from_date": bulan_lalu,
        "to_date": today
    })

    # Ambil statistik agregat
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


# ── Handlers ──────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nama = update.effective_user.first_name
    await update.message.reply_text(
        f"👟 Halo {nama}! Saya bot pelatih larimu.\n\n"
        "Tanya apapun tentang aktivitas Garmin kamu:\n\n"
        "• Berapa total lari saya bulan ini?\n"
        "• Gimana pace rata-rata saya?\n"
        "• Aktivitas terakhir saya apa?\n"
        "• Kapan saya lari paling jauh?\n\n"
        "/ringkasan — ringkasan 30 hari terakhir\n"
        "/cekkoneksi — cek koneksi Garmin"
    )

async def cmd_ringkasan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Mengambil data Garmin kamu...")
    data = await fetch_garmin_data()
    jawaban = tanya_groq(
        "Berikan ringkasan lengkap aktivitas lari 30 hari terakhir: "
        "total jarak, jumlah sesi, pace terbaik, kalori, dan tren performa.",
        data
    )
    await update.message.reply_text(f"📊 *Ringkasan 30 Hari*\n\n{jawaban}", parse_mode="Markdown")

async def cmd_cekkoneksi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Mengecek koneksi ke Garmin...")
    hasil = await call_mcp("list_activities", {"limit": 1})
    if "Error" in hasil:
        await update.message.reply_text(f"❌ Gagal terhubung:\n{hasil}")
    else:
        await update.message.reply_text("✅ Koneksi ke Garmin berhasil! Data bisa dibaca.")

async def handle_pesan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pertanyaan = update.message.text
    await update.message.reply_text("⏳ Menganalisis data Garmin kamu...")
    data = await fetch_garmin_data()
    jawaban = tanya_groq(pertanyaan, data)
    await update.message.reply_text(jawaban)


# ── Main ──────────────────────────────────────────────────────
def main():
    print("🤖 Bot Garmin + Groq berjalan...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("ringkasan",   cmd_ringkasan))
    app.add_handler(CommandHandler("cekkoneksi",  cmd_cekkoneksi))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pesan))
    app.run_polling()

if __name__ == "__main__":
    main()