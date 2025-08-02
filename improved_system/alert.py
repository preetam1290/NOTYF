import os
import time
import requests
import sys
import smtplib
from email.mime.text import MIMEText
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# === Load .env file ===
load_dotenv()

# === Configuration ===
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Email settings (fallback alternative)
SMTP_SERVER = os.environ.get("SMTP_SERVER")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_TO = os.environ.get("EMAIL_TO")

# Bot parameters
SYMBOL = "ETHUSDT"
INTERVAL = Client.KLINE_INTERVAL_5MINUTE
LOOKBACK = 250
FIB_LOW_PCT = 0.595
FIB_HIGH_PCT = 0.65
POLL_INTERVAL = 5 * 60

# Initialize Binance client
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)


def get_telegram_chat_id():
    """Fetch the latest chat_id from bot updates (for initial setup)."""
    if not TELEGRAM_TOKEN:
        print("Telegram token not configured.")
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    try:
        resp = requests.get(url).json()
        if not resp.get("ok") or not resp.get("result"):
            print("No updates found. Send a message to your bot first.")
            return None
        # extract chat_id from last update
        last = resp["result"][-1]
        chat_id = last["message"]["chat"]["id"]
        print(f"Detected Telegram chat_id: {chat_id}")
        return str(chat_id)
    except Exception as e:
        print(f"Error fetching updates: {e}")
        return None


def send_telegram_message(text: str) -> bool:
    """Attempt to send via Telegram. Returns True on success."""
    global TELEGRAM_CHAT_ID
    # If chat_id not set, try to detect automatically
    if not TELEGRAM_CHAT_ID:
        TELEGRAM_CHAT_ID = get_telegram_chat_id()
        if not TELEGRAM_CHAT_ID:
            return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=payload)
        if resp.status_code == 200:
            print("✅ Telegram message sent.")
            return True
        else:
            print(f"❌ Telegram error: {resp.status_code} | {resp.text}")
            return False
    except Exception as e:
        print(f"❌ Exception sending Telegram: {e}")
        return False


def send_email(text: str):
    """Send alert via email fallback."""
    if not all([SMTP_SERVER, EMAIL_USER, EMAIL_PASSWORD, EMAIL_TO]):
        print("🚫 Email credentials not fully configured.")
        return
    msg = MIMEText(text)
    msg["Subject"] = "ETH Fibonacci Zone Alert"
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, [EMAIL_TO], msg.as_string())
        server.quit()
        print("✅ Email alert sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


def send_alert(text: str):
    """Unified alert: try Telegram first, then email fallback."""
    if not send_telegram_message(text):
        print("🔄 Falling back to email...")
        send_email(text)


def fetch_klines():
    try:
        return client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK + 2)
    except BinanceAPIException as e:
        print(f"Binance API error: {e}")
        return []


def compute_fib_zone(klines):
    highs = [float(k[2]) for k in klines[-(LOOKBACK + 1):-1]]
    lows = [float(k[3]) for k in klines[-(LOOKBACK + 1):-1]]
    swing_high, swing_low = max(highs), min(lows)
    price_range = swing_high - swing_low
    fib_low = swing_high - price_range * FIB_LOW_PCT
    fib_high = swing_high - price_range * FIB_HIGH_PCT
    return fib_low, fib_high


def monitor():
    prev_in_zone = False
    while True:
        klines = fetch_klines()
        if not klines:
            time.sleep(POLL_INTERVAL)
            continue
        fib_low, fib_high = compute_fib_zone(klines)
        last_close = float(klines[-2][4])
        timestamp = klines[-2][0]
        ts_readable = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamp / 1000))
        in_zone = fib_high <= last_close <= fib_low
        if in_zone and not prev_in_zone:
            text = (
                f"🚨 *ETHUSDT* Fibonacci zone hit!\n"
                f"Time: `{ts_readable}` UTC\n"
                f"Price: `{last_close}` USDT\n"
                f"Zone: {fib_high:.2f} - {fib_low:.2f}`"
            )
            send_alert(text)
        prev_in_zone = in_zone
        time.sleep(POLL_INTERVAL)


def demo():
    sample = (
        "🚀 *DEMO: ETH Fibonacci Zone!*\n"
        "Time: `2025-07-26 12:00:00` UTC\n"
        "Price: `3140.42` USDT\n"
        "Zone: 3130.00 - 3050.00"
    )
    send_alert(sample)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("📦 DEMO mode...")
        demo()
    else:
        print("🚀 Starting monitor...")
        monitor()
