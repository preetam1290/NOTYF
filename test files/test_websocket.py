import websocket
import json
import ssl

def on_message(ws, message):
    data = json.loads(message)
    print(f"✅ Received data: {data['k']['c']} (close price)")

def on_error(ws, error):
    print(f"❌ WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"🔌 Connection closed: {close_status_code} - {close_msg}")

def on_open(ws):
    print("🔌 WebSocket connected successfully!")

# Test connection
socket_url = "wss://stream.binance.com:9443/ws/ethusdt@kline_5m"
print(f"Testing connection to: {socket_url}")

ws = websocket.WebSocketApp(
    socket_url,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
    on_open=on_open
)

# Disable SSL verification if needed
ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
