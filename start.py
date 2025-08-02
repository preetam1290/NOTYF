#!/usr/bin/env python3
"""
Start script for Render deployment
Handles graceful startup and error recovery
"""

import subprocess
import sys
import time
import os
from datetime import datetime

def start_trading_bot():
    """Start the trading bot with error handling"""
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"🚀 Starting Fibonacci Alert System - Attempt {retry_count + 1}")
            print(f"📅 Time: {datetime.now()}")
            
            # Start the trading bot
            process = subprocess.Popen([
                sys.executable, "trading_bot.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Monitor the process
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.strip())
            
            # If we reach here, the process ended
            return_code = process.wait()
            print(f"⚠️ Trading bot exited with code: {return_code}")
            
            if return_code == 0:
                print("✅ Trading bot exited gracefully")
                break
            else:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = min(60 * retry_count, 300)  # Max 5 minutes
                    print(f"🔄 Restarting in {wait_time} seconds...")
                    time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("⛔ Received interrupt signal, shutting down...")
            break
        except Exception as e:
            print(f"💥 Error starting trading bot: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"🔄 Retrying in 60 seconds...")
                time.sleep(60)
    
    if retry_count >= max_retries:
        print(f"❌ Failed to start trading bot after {max_retries} attempts")
        sys.exit(1)

if __name__ == "__main__":
    start_trading_bot()
