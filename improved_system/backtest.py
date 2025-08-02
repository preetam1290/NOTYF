# ============================================================================
# BACKTEST MODULE - Historical Data Simulation
# ============================================================================

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from binance.client import Client
import sys
import os
import requests
import importlib.util

# Import the main trading system
sys.path.append('C:\\Users\\GIGABYTE\\stetragy programing')

# Import FibonacciAlertSystem class from system V3.py
spec = importlib.util.spec_from_file_location("system_v3", "C:\\Users\\GIGABYTE\\stetragy programing\\system V3.py")
system_v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(system_v3)
FibonacciAlertSystem = system_v3.FibonacciAlertSystem

class BacktestSystem:
    """Backtest the trading system with historical data"""
    
    def __init__(self, api_key, api_secret, symbol='ETHUSDT', interval='5m'):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.interval = interval
        self.client = Client(api_key, api_secret)
        
        # Backtest results tracking
        self.alerts_generated = []
        self.fibonacci_hits = []
        self.ema_intersections = []
        self.volume_alerts = []
        
    def get_historical_data_range(self, start_date, end_date, limit=30):
        """Get historical data for a specific date range"""
        try:
            print(f"🔍 Fetching historical data from {start_date} to {end_date}")
            
            klines = self.client.get_historical_klines(
                self.symbol, 
                self.interval, 
                start_date,
                end_date,
                limit=limit
            )
            
            if not klines:
                print("❌ No historical data received")
                return None
            
            data = []
            for kline in klines:
                data.append({
                    'timestamp': pd.to_datetime(kline[0], unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ Loaded {len(df)} historical bars")
            return df
            
        except Exception as e:
            print(f"❌ Error getting historical data: {e}")
            return None
    
    def run_backtest(self, start_date, end_date, speed_multiplier=1):
        """Run backtest simulation"""
        print("🚀 Starting Backtest Simulation")
        print("=" * 60)
        
        # Get historical data
        historical_data = self.get_historical_data_range(start_date, end_date)
        
        if historical_data is None:
            print("❌ Failed to get historical data for backtest")
            return
        
        # Initialize trading system in debug mode
        system = FibonacciAlertSystem(
            api_key=self.api_key,
            api_secret=self.api_secret,
            symbol=self.symbol,
            interval=self.interval,
            debug=True
        )
        
        # Disable Telegram alerts for backtest
        system.telegram_config['enable_telegram_alerts'] = True
        
        print(f"📊 Simulating {len(historical_data)} candles...")
        print(f"⚡ Speed: {speed_multiplier}x normal")
        
        # Process each candle
        for i, (timestamp, row) in enumerate(historical_data.iterrows()):
            print(f"\n🔧 DEBUG: Processing candle {i+1}/{len(historical_data)} - {timestamp}")
            
            # Create new data point
            new_data = {
                'timestamp': timestamp,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            
            # Add to system DataFrame
            new_row = pd.DataFrame([new_data])
            new_row.set_index('timestamp', inplace=True)
            system.df = pd.concat([system.df, new_row])
            
            # Keep only last max_bars
            if len(system.df) > system.max_bars:
                system.df = system.df.tail(system.max_bars)
            
            # Calculate indicators (only if we have enough data)
            try:
                system.calculate_volume_ml_indicators()
                system.calculate_volume_ml_score()
                system.calculate_fibonacci_levels()
                system.calculate_confluence_score()
                
                # Check for alerts (only if indicators are calculated)
                if len(system.df) >= system.vol_lookback:
                    fib_alerts = system.check_fibonacci_alerts()
                    ema_intersection = system.check_ema_intersection()
                else:
                    fib_alerts = []
                    ema_intersection = False
                    if system.debug:
                        print(f"🔧 DEBUG: Skipping alerts - need {system.vol_lookback} bars, have {len(system.df)}")
            except Exception as e:
                if system.debug:
                    print(f"🔧 DEBUG: Error in calculations: {e}")
                fib_alerts = []
                ema_intersection = False
            
            # Track alerts
            if fib_alerts:
                for alert in fib_alerts:
                    alert['timestamp'] = timestamp
                    self.fibonacci_hits.append(alert)
                    print(f"🎯 Fibonacci Alert: {alert['level']} at ${alert['current_price']:.4f}")
            
            if ema_intersection:
                self.ema_intersections.append({
                    'timestamp': timestamp,
                    'price': row['close']
                })
                print(f"🔄 EMA Intersection at ${row['close']:.4f}")
            
            # Print periodic analysis
            if i % 50 == 0:  # Every 50 candles
                try:
                    analysis = system.analyze_current_state()
                    if analysis:
                        print(f"\n📊 ANALYSIS UPDATE - Candle {i+1}")
                        print(f"Price: ${analysis['price']:.2f}")
                        if 'volume_analysis' in analysis and analysis['volume_analysis']:
                            print(f"Volume ML Score: {analysis['volume_analysis']['vol_ml_score']}")
                        if 'confluence_score' in analysis:
                            print(f"Confluence Score: {analysis['confluence_score']}")
                except Exception as e:
                    if system.debug:
                        print(f"🔧 DEBUG: Analysis error at candle {i+1}: {e}")
            
            # Simulate time delay (adjust based on speed_multiplier)
            if speed_multiplier < 10:  # Only add delay for slower speeds
                time.sleep(0.1 / speed_multiplier)
        
        # Print backtest results
        self.print_backtest_results()
    
    def send_backtest_results_to_telegram(self):
        """Send backtest results summary to Telegram"""
        try:
            # Create a summary message
            results_message = f"📊 **BACKTEST RESULTS SUMMARY** 📊\n\n"
            results_message += f"💹 Symbol: {self.symbol}\n"
            results_message += f"⏱️ Interval: {self.interval}\n\n"
            results_message += f"🎯 Total Fibonacci Alerts: {len(self.fibonacci_hits)}\n"
            results_message += f"🔄 Total EMA Intersections: {len(self.ema_intersections)}\n\n"
            
            if self.fibonacci_hits:
                fib_595_count = len([alert for alert in self.fibonacci_hits if alert['level'] == '59.5%'])
                fib_650_count = len([alert for alert in self.fibonacci_hits if alert['level'] == '65.0%'])
                
                results_message += f"🎯 **FIBONACCI BREAKDOWN:**\n"
                results_message += f"• Fibonacci 59.5% hits: {fib_595_count}\n"
                results_message += f"• Fibonacci 65.0% hits: {fib_650_count}\n\n"
                
                if self.fibonacci_hits:
                    results_message += f"📋 **RECENT FIBONACCI ALERTS:**\n"
                    for alert in self.fibonacci_hits[-3:]:  # Show last 3
                        results_message += f"• {alert['timestamp'].strftime('%m-%d %H:%M')} - Fib {alert['level']} at ${alert['current_price']:.2f}\n"
            
            if self.ema_intersections:
                results_message += f"\n🔄 **EMA INTERSECTION SUMMARY:**\n"
                results_message += f"• Total intersections: {len(self.ema_intersections)}\n\n"
                
                results_message += f"📋 **RECENT EMA INTERSECTIONS:**\n"
                for intersection in self.ema_intersections[-3:]:  # Show last 3
                    results_message += f"• {intersection['timestamp'].strftime('%m-%d %H:%M')} - Price: ${intersection['price']:.2f}\n"
            
            # Send via telegram bot (reuse the telegram config from the system)
            bot_token = '8309174583:AAEzQNLmpwHbJ-NCVLoR5aSoqzwlncR3QoE'
            chat_id = '6080078099'
            
            response = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={'chat_id': chat_id, 'text': results_message, 'parse_mode': 'Markdown'},
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ Backtest results sent to Telegram!")
            else:
                print(f"⚠️ Failed to send backtest results to Telegram: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error sending backtest results to Telegram: {e}")
    
    def print_backtest_results(self):
        """Print comprehensive backtest results"""
        print("\n" + "=" * 80)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"Total Fibonacci Alerts: {len(self.fibonacci_hits)}")
        print(f"Total EMA Intersections: {len(self.ema_intersections)}")
        
        # Send backtest results to Telegram
        self.send_backtest_results_to_telegram()
        
        if self.fibonacci_hits:
            print("\n🎯 FIBONACCI ALERT BREAKDOWN:")
            fib_595_count = len([alert for alert in self.fibonacci_hits if alert['level'] == '59.5%'])
            fib_650_count = len([alert for alert in self.fibonacci_hits if alert['level'] == '65.0%'])
            
            print(f"  • Fibonacci 59.5% hits: {fib_595_count}")
            print(f"  • Fibonacci 65.0% hits: {fib_650_count}")
            
            print("\n📋 RECENT FIBONACCI ALERTS:")
            for alert in self.fibonacci_hits[-5:]:  # Show last 5
                print(f"  {alert['timestamp']} - Fib {alert['level']} at ${alert['current_price']:.4f}")
        
        if self.ema_intersections:
            print("\n🔄 EMA INTERSECTION SUMMARY:")
            print(f"  • Total intersections: {len(self.ema_intersections)}")
            
            print("\n📋 RECENT EMA INTERSECTIONS:")
            for intersection in self.ema_intersections[-5:]:  # Show last 5
                print(f"  {intersection['timestamp']} - Price: ${intersection['price']:.4f}")
        
        print("\n" + "=" * 80)

def run_quick_backtest():
    """Run a quick backtest for demonstration"""
    # Your API credentials
    API_KEY = 'WSPoE9L04b8xukvqh7li4WT1TEqNnGob4ZCanZ4p2lTy2AbjYxzjV8q9lPcARRNI'
    API_SECRET = 'J4PoDOoyBhpMTKIBBHGa1eCz8gNG19N85yTxfwFGG5tNcPUVgbrwKA51lox451Xg'
    
    # Initialize backtest
    backtest = BacktestSystem(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='ETHUSDT',
        interval='5m'
    )
    
    # Run backtest for last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"🎯 Running backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    backtest.run_backtest(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        speed_multiplier=100  # 100x speed
    )

if __name__ == "__main__":
    run_quick_backtest()
