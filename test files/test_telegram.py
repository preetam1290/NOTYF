#!/usr/bin/env python3
"""
Telegram Bot Test Script
Sends a demo message to verify bot functionality
"""

import requests
import json
from datetime import datetime

# Your bot configuration
BOT_TOKEN = "8459911523:AAHXBnaNywpvSNU59MhD6G0RqJn993VKhUU"
CHAT_ID = "-4808548016"

def send_test_message():
    """Send a test message to Telegram"""
    
    # Create test message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    message = f"""🚀 TELEGRAM BOT TEST MESSAGE 🚀

✅ Bot Status: CONNECTED
📱 Chat ID: {CHAT_ID}
🕐 Time: {timestamp}
🔧 Test: Configuration Verification

Hey Preetam! 👋
Your trading bot is working perfectly!

🎯 This is a demo message to confirm:
• Telegram bot token is valid
• Chat ID is correct
• Bot has permission to send messages
• All systems are operational

Ready to receive trading alerts! 📊"""

    # Send message via Telegram Bot API
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message
    }
    
    try:
        print("🔄 Sending test message to Telegram...")
        response = requests.post(url, json=payload, timeout=10)
        response_data = response.json()
        
        if response.status_code == 200 and response_data.get('ok', False):
            print("✅ SUCCESS: Test message sent successfully!")
            print(f"📱 Message ID: {response_data.get('result', {}).get('message_id', 'N/A')}")
            print(f"📊 Chat ID: {response_data.get('result', {}).get('chat', {}).get('id', 'N/A')}")
            return True
        else:
            error_description = response_data.get('description', 'Unknown error')
            print(f"❌ FAILED: {response.status_code} - {error_description}")
            
            # Provide troubleshooting tips
            if "chat not found" in error_description.lower():
                print("\n🔧 TROUBLESHOOTING:")
                print("1. Make sure the bot is added to your chat/group")
                print("2. Send a message to the bot first if it's a private chat")
                print(f"3. Verify chat ID: {CHAT_ID}")
                print("4. For groups, chat ID should start with '-'")
            elif "bot was blocked" in error_description.lower():
                print("\n🔧 SOLUTION: The bot was blocked by the user. Please unblock it.")
            elif "not enough rights" in error_description.lower():
                print("\n🔧 SOLUTION: The bot doesn't have permission to send messages. Check bot permissions.")
            
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"🌐 NETWORK ERROR: {e}")
        return False
    except Exception as e:
        print(f"💥 UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TELEGRAM BOT TEST SCRIPT")
    print("=" * 40)
    print(f"🤖 Bot Token: {BOT_TOKEN[:10]}...")
    print(f"💬 Chat ID: {CHAT_ID}")
    print("=" * 40)
    
    success = send_test_message()
    
    if success:
        print("\n🎉 CONGRATULATIONS!")
        print("Your Telegram bot is working perfectly!")
        print("You should have received the test message.")
    else:
        print("\n⚠️ TROUBLESHOOTING NEEDED")
        print("Please check the error messages above and fix the issues.")
