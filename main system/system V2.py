import pandas as pd
import numpy as np
import websocket
import json
import threading
import time
from datetime import datetime, timedelta
from binance.client import Client
import warnings
import logging


warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedTechnicalAnalysis:
    def __init__(self, api_key, api_secret, symbol='ETHUSDT', interval='5m', sms_config_path='config.json'):
        """
        Advanced Technical Analysis Suite for live trading data
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            symbol: Trading pair (default: ETHUSDT)
            interval: Timeframe (default: 5m)
            sms_config_path: Path to SMS configuration file (default: config.json)
        """
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.interval = interval
        self.df = pd.DataFrame()
        self.ws = None
        self.websocket_thread = None # To keep track of the WebSocket thread
        self.is_running = True # Control flag for the main loop and WebSocket
        
        

        # Parameters (matching Pine Script inputs)
        self.vol_lookback = 30
        self.vol_threshold = 2.0
        self.ml_sensitivity = 2.0
        self.swing_lookback = 249
        self.ema_length = 50
        
        # Alert system parameters
        self.fib_tolerance = 0.1  # Percentage tolerance for Fibonacci level matching (0.1% = 0.001)
        self.last_fib_alert_time = {}  # Track last alert time for each level
        self.alert_cooldown = 300  # 5 minutes cooldown between alerts for same level
        
        # Data storage
        self.price_data = []
        self.max_bars = 500  # Keep last 500 bars
        
        logging.info(f"Initializing Advanced Technical Analysis for {symbol} on {interval} timeframe...")
        logging.info(f"Fibonacci Alert System Enabled - Tolerance: {self.fib_tolerance}%")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI using pandas (no TA-Lib needed)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ema(self, prices, period):
        """Calculate EMA using pandas (no TA-Lib needed)"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
        
    def get_historical_data(self, limit=500):
        """Get historical kline data from Binance"""
        try:
            klines = self.client.get_historical_klines(
                self.symbol, self.interval, f"{limit} hours ago UTC"
            )
            
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
            
            self.df = pd.DataFrame(data)
            self.df.set_index('timestamp', inplace=True)
            logging.info(f"Loaded {len(self.df)} historical bars")
            return True
            
        except Exception as e:
            logging.error(f"Error getting historical data: {e}", exc_info=True)
            return False
    
    def calculate_volume_ml_indicators(self):
        """Calculate Volume ML indicators matching Pine Script logic"""
        if len(self.df) < self.vol_lookback:
            return
        
        # Basic volume statistics
        self.df['vol_avg'] = self.df['volume'].rolling(window=self.vol_lookback).mean()
        self.df['vol_std'] = self.df['volume'].rolling(window=self.vol_lookback).std()
        self.df['vol_normalized'] = (self.df['volume'] - self.df['vol_avg']) / self.df['vol_std']
        
        # Price and volume changes
        self.df['price_change'] = self.df['close'].diff()
        self.df['volume_change'] = self.df['volume'].diff()
        
        # Volume-Price correlation
        self.df['vp_correlation'] = self.df['price_change'].rolling(20).corr(self.df['volume_change'])
        
        # Volume RSI and EMAs (using custom functions)
        self.df['vol_rsi'] = self.calculate_rsi(self.df['volume'], 14)
        self.df['vol_ema_short'] = self.calculate_ema(self.df['volume'], 8)
        self.df['vol_ema_long'] = self.calculate_ema(self.df['volume'], 21)
        self.df['vol_momentum'] = ((self.df['vol_ema_short'] - self.df['vol_ema_long']) / self.df['vol_ema_long']) * 100
        
        # Volume patterns
        self.df['vol_spike'] = self.df['volume'] > (self.df['vol_avg'] * self.vol_threshold)
        self.df['vol_dry_up'] = self.df['volume'] < (self.df['vol_avg'] * 0.5)
        self.df['vol_increasing'] = (self.df['volume'] > self.df['volume'].shift(1)) & (self.df['volume'].shift(1) > self.df['volume'].shift(2))
        self.df['vol_decreasing'] = (self.df['volume'] < self.df['volume'].shift(1)) & (self.df['volume'].shift(1) < self.df['volume'].shift(2))
        
        # Smart Money Flow Analysis
        hlc2 = (self.df['high'] + self.df['low']) / 2
        self.df['buying_pressure'] = np.where(self.df['close'] > hlc2, self.df['volume'], 0)
        self.df['selling_pressure'] = np.where(self.df['close'] < hlc2, self.df['volume'], 0)
        
        self.df['smart_money_flow'] = (self.df['buying_pressure'] - self.df['selling_pressure']).rolling(21).mean()
        self.df['smart_money_ratio'] = (self.df['smart_money_flow'] / self.df['vol_avg']) * 100
        
        # VWAP and deviation (cumulative calculation reset daily)
        hlc3 = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # For a more accurate daily VWAP, you'd need to reset at market open
        # Here's a rolling VWAP approximation for the last 100 periods
        rolling_periods = min(100, len(self.df))
        rolling_vol_price = (hlc3 * self.df['volume']).rolling(rolling_periods).sum()
        rolling_volume = self.df['volume'].rolling(rolling_periods).sum()
        self.df['vwap'] = rolling_vol_price / rolling_volume
        
        self.df['vwap_deviation'] = ((self.df['close'] - self.df['vwap']) / self.df['vwap']) * 100
        
    def calculate_volume_ml_score(self):
        """Calculate Volume ML Score matching Pine Script logic"""
        if len(self.df) == 0:
            return
        
        vol_ml_score = pd.Series(0.0, index=self.df.index)
        
        # Volume spike analysis
        condition1 = self.df['vol_spike'] & (self.df['price_change'] > 0)
        condition2 = self.df['vol_spike'] & (self.df['price_change'] < 0)
        vol_ml_score = np.where(condition1, vol_ml_score + 30, vol_ml_score)
        vol_ml_score = np.where(condition2, vol_ml_score - 25, vol_ml_score)
        
        # Volume momentum
        condition3 = self.df['vol_momentum'] > 10
        condition4 = self.df['vol_momentum'] < -10
        vol_ml_score = np.where(condition3, vol_ml_score + 20, vol_ml_score)
        vol_ml_score = np.where(condition4, vol_ml_score - 15, vol_ml_score)
        
        # Smart money flow
        condition5 = self.df['smart_money_ratio'] > 5
        condition6 = self.df['smart_money_ratio'] < -5
        vol_ml_score = np.where(condition5, vol_ml_score + 25, vol_ml_score)
        vol_ml_score = np.where(condition6, vol_ml_score - 20, vol_ml_score)
        
        # Volume-price correlation
        condition7 = self.df['vp_correlation'] > 0.3
        condition8 = self.df['vp_correlation'] < -0.3
        vol_ml_score = np.where(condition7, vol_ml_score + 15, vol_ml_score)
        vol_ml_score = np.where(condition8, vol_ml_score - 10, vol_ml_score)
        
        # VWAP analysis
        condition9 = (self.df['close'] > self.df['vwap']) & (self.df['volume'] > self.df['vol_avg'])
        condition10 = (self.df['close'] < self.df['vwap']) & (self.df['volume'] > self.df['vol_avg'])
        vol_ml_score = np.where(condition9, vol_ml_score + 10, vol_ml_score)
        vol_ml_score = np.where(condition10, vol_ml_score - 10, vol_ml_score)
        
        # Volume pattern recognition
        condition11 = self.df['vol_increasing'] & (self.df['price_change'] > 0)
        condition12 = self.df['vol_decreasing'] & (self.df['price_change'] < 0)
        vol_ml_score = np.where(condition11, vol_ml_score + 15, vol_ml_score)
        vol_ml_score = np.where(condition12, vol_ml_score - 10, vol_ml_score)
        
        # Apply ML sensitivity
        vol_ml_score = vol_ml_score * self.ml_sensitivity
        
        self.df['vol_ml_score'] = vol_ml_score
        
        # Volume trend classification
        self.df['vol_trend'] = np.where(
            self.df['vol_ml_score'] > 30, 'ACCUMULATION',
            np.where(self.df['vol_ml_score'] < -30, 'DISTRIBUTION', 'NEUTRAL')
        )
    
    def calculate_fibonacci_levels(self):
        """Calculate Fibonacci retracement levels (Magic Zone Strategy)"""
        if len(self.df) < self.swing_lookback:
            return
        
        # Calculate swing high and low
        self.df['swing_high'] = self.df['high'].rolling(window=self.swing_lookback).max()
        self.df['swing_low'] = self.df['low'].rolling(window=self.swing_lookback).min()
        
        # Calculate Fibonacci levels
        price_range = self.df['swing_high'] - self.df['swing_low']
        self.df['fib_595'] = self.df['swing_high'] - price_range * 0.595
        self.df['fib_650'] = self.df['swing_high'] - price_range * 0.65
        
        # Calculate EMA (using custom function)
        self.df['ema_50'] = self.calculate_ema(self.df['close'], self.ema_length)
        
        # Calculate RSI with period 9
        self.df['rsi'] = self.calculate_rsi(self.df['close'], 9)
    
    def check_fibonacci_alerts(self):
        """Check if current price matches Fibonacci levels and send alerts"""
        if len(self.df) == 0:
            return
        
        latest = self.df.iloc[-1]
        current_price = latest['close']
        current_time = time.time()
        
        # Check if we have valid Fibonacci levels
        if pd.isna(latest['fib_595']) or pd.isna(latest['fib_650']):
            return
        
        fib_595 = latest['fib_595']
        fib_650 = latest['fib_650']
        
        # Calculate tolerance range for each level
        tolerance_595 = fib_595 * (self.fib_tolerance / 100)
        tolerance_650 = fib_650 * (self.fib_tolerance / 100)
        
        # Check Fibonacci 59.5% level
        if (abs(current_price - fib_595) <= tolerance_595):
            alert_key = 'fib_595'
            if (alert_key not in self.last_fib_alert_time or 
                current_time - self.last_fib_alert_time[alert_key] > self.alert_cooldown):
                
                self.send_fibonacci_alert('59.5%', current_price, fib_595)
                self.last_fib_alert_time[alert_key] = current_time
        
        # Check Fibonacci 65.0% level
        if (abs(current_price - fib_650) <= tolerance_650):
            alert_key = 'fib_650'
            if (alert_key not in self.last_fib_alert_time or 
                current_time - self.last_fib_alert_time[alert_key] > self.alert_cooldown):
                
                self.send_fibonacci_alert('65.0%', current_price, fib_650)
                self.last_fib_alert_time[alert_key] = current_time
    
    def send_fibonacci_alert(self, fib_level, current_price, fib_price):
        """Send Fibonacci level alert"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        alert_message = (
            "\n" + "🚨" * 50 + "\n"
            "🚨 FIBONACCI LEVEL ALERT 🚨\n"
            "🚨" * 50 + "\n"
            "Hey! Preetam need to look the market bro!\n"
            f"Symbol: {self.symbol}\n"
            f"Time: {timestamp}\n"
            f"Current Price: ${current_price:.4f}\n"
            f"Fibonacci {fib_level} Level: ${fib_price:.4f}\n"
            f"Difference: ${abs(current_price - fib_price):.4f}\n"
            f"Price is {'ABOVE' if current_price > fib_price else 'BELOW'} Fibonacci {fib_level}\n"
            "🚨" * 50 + "\n"
            "Action Required: Check the market immediately!\n"
            "🚨" * 50
        )
        
        # Preserve existing console logging behavior
        logging.warning(alert_message) # Use warning level for alerts
        
        # Send SMS alert if SMS manager is available
        if self.sms_manager:
            sms_text = (
                f"🚨 FIBONACCI ALERT 🚨\n"
                f"Symbol: {self.symbol}\n"
                f"Price: ${current_price:.4f}\n"
                f"Fib {fib_level}: ${fib_price:.4f}\n"
                f"Diff: ${abs(current_price - fib_price):.4f}\n"
                f"Price {'ABOVE' if current_price > fib_price else 'BELOW'} Fib {fib_level}\n"
                f"Time: {timestamp}"
            )
            try:
                sid, status = self.sms_manager.send_message(sms_text, f"fibonacci_{fib_level.replace('.', '_')}")
                if status == "sent":
                    logging.info(f"SMS alert sent successfully for Fibonacci {fib_level} level")
                elif status == "blocked":
                    logging.info(f"SMS alert blocked by rate limiting for Fibonacci {fib_level} level")
                else:
                    logging.warning(f"SMS alert failed for Fibonacci {fib_level} level")
            except Exception as e:
                logging.error(f"Error sending SMS alert: {e}")
        
        # Preserve existing log file behavior
        self.log_fibonacci_alert(fib_level, current_price, fib_price, timestamp)
    
    def log_fibonacci_alert(self, fib_level, current_price, fib_price, timestamp):
        """Log Fibonacci alerts to file"""
        try:
            with open(f"fibonacci_alerts_{self.symbol}.log", "a") as f:
                f.write(f"{timestamp} | {self.symbol} | Fib {fib_level} | Price: ${current_price:.4f} | "
                       f"Fib Level: ${fib_price:.4f} | Diff: ${abs(current_price - fib_price):.4f}\n")
        except Exception as e:
            logging.error(f"Error logging alert: {e}", exc_info=True)
    
    def _send_rsi_alert(self, alert_type, rsi_value, current_time):
        """Send RSI-based SMS alert"""
        if not self.sms_manager:
            return
        
        # Get current price from the latest data
        if len(self.df) == 0:
            return
        
        current_price = self.df.iloc[-1]['close']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sms_text = (
            f"🚨 RSI ALERT 🚨\n"
            f"{alert_type}\n"
            f"Symbol: {self.symbol}\n"
            f"Price: ${current_price:.4f}\n"
            f"RSI(9): {rsi_value}\n"
            f"Time: {timestamp}"
        )
        
        try:
            tag = "rsi_" + alert_type.lower().replace(" ", "_")
            sid, status = self.sms_manager.send_message(sms_text, tag)
            if status == "sent":
                logging.info(f"RSI SMS alert sent successfully: {alert_type}")
            elif status == "blocked":
                logging.info(f"RSI SMS alert blocked by rate limiting: {alert_type}")
            else:
                logging.warning(f"RSI SMS alert failed: {alert_type}")
        except Exception as e:
            logging.error(f"Error sending RSI SMS alert: {e}")
    
    def _send_volume_alert(self, alert_type, score, current_price):
        """Send volume-based SMS alert"""
        if not self.sms_manager:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sms_text = (
            f"🚨 VOLUME ALERT 🚨\n"
            f"{alert_type}\n"
            f"Symbol: {self.symbol}\n"
            f"Price: ${current_price:.4f}\n"
            f"ML Score: {score:.0f}\n"
            f"Time: {timestamp}"
        )
        
        try:
            tag = "volume_" + alert_type.lower().replace(" ", "_")
            sid, status = self.sms_manager.send_message(sms_text, tag)
            if status == "sent":
                logging.info(f"Volume SMS alert sent successfully: {alert_type}")
            elif status == "blocked":
                logging.info(f"Volume SMS alert blocked by rate limiting: {alert_type}")
            else:
                logging.warning(f"Volume SMS alert failed: {alert_type}")
        except Exception as e:
            logging.error(f"Error sending volume SMS alert: {e}")
    
    def _send_confluence_alert(self, alert_type, score, current_price):
        """Send confluence-based SMS alert"""
        if not self.sms_manager:
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sms_text = (
            f"🚨 CONFLUENCE ALERT 🚨\n"
            f"{alert_type}\n"
            f"Symbol: {self.symbol}\n"
            f"Price: ${current_price:.4f}\n"
            f"Score: {score:.0f}\n"
            f"Time: {timestamp}"
        )
        
        try:
            tag = "confluence_" + alert_type.lower().replace(" ", "_")
            sid, status = self.sms_manager.send_message(sms_text, tag)
            if status == "sent":
                logging.info(f"Confluence SMS alert sent successfully: {alert_type}")
            elif status == "blocked":
                logging.info(f"Confluence SMS alert blocked by rate limiting: {alert_type}")
            else:
                logging.warning(f"Confluence SMS alert failed: {alert_type}")
        except Exception as e:
            logging.error(f"Error sending confluence SMS alert: {e}")
    
    def calculate_confluence_score(self):
        """Calculate confluence score"""
        if len(self.df) == 0:
            return
        
        confluence_score = pd.Series(0.0, index=self.df.index)
        
        # Trend momentum
        momentum = self.df['close'].diff(5)
        confluence_score = np.where(momentum > 0, confluence_score + 15, confluence_score - 15)
        
        # Market structure analysis
        recent_high = self.df['high'].rolling(10).max().shift(1)
        recent_low = self.df['low'].rolling(10).min().shift(1)
        
        market_structure = np.where(
            self.df['close'] > recent_high, 1,
            np.where(self.df['close'] < recent_low, -1, 0)
        )
        
        confluence_score = np.where(market_structure == 1, confluence_score + 10, confluence_score)
        confluence_score = np.where(market_structure == -1, confluence_score - 10, confluence_score)
        
        # Volume confluence
        confluence_score = np.where(self.df['vol_ml_score'] > 20, confluence_score + 20, confluence_score)
        confluence_score = np.where(self.df['vol_ml_score'] < -20, confluence_score - 15, confluence_score)
        
        self.df['confluence_score'] = confluence_score
    
    def analyze_current_state(self):
        """Analyze current market state and return summary"""
        if len(self.df) == 0:
            return None
        
        latest = self.df.iloc[-1]
        current_time = self.df.index[-1]
        
        # Handle NaN values safely
        def safe_round(value, decimals=2):
            return round(float(value), decimals) if pd.notna(value) else 0.0
        
        # Volume analysis
        vol_ratio = latest['volume'] / latest['vol_avg'] if pd.notna(latest['vol_avg']) and latest['vol_avg'] > 0 else 0
        vol_ratio_status = 'HIGH' if vol_ratio > 1.5 else 'LOW' if vol_ratio < 0.5 else 'NORMAL'
        
        vol_rsi_status = 'OVERBOUGHT' if pd.notna(latest['vol_rsi']) and latest['vol_rsi'] > 70 else 'OVERSOLD' if pd.notna(latest['vol_rsi']) and latest['vol_rsi'] < 30 else 'NEUTRAL'
        
        vol_momentum_status = 'BULLISH' if pd.notna(latest['vol_momentum']) and latest['vol_momentum'] > 10 else 'BEARISH' if pd.notna(latest['vol_momentum']) and latest['vol_momentum'] < -10 else 'NEUTRAL'
        
        smart_money_status = 'BUYING' if pd.notna(latest['smart_money_ratio']) and latest['smart_money_ratio'] > 5 else 'SELLING' if pd.notna(latest['smart_money_ratio']) and latest['smart_money_ratio'] < -5 else 'NEUTRAL'
        
        vp_correlation_status = 'STRONG +' if pd.notna(latest['vp_correlation']) and latest['vp_correlation'] > 0.3 else 'STRONG -' if pd.notna(latest['vp_correlation']) and latest['vp_correlation'] < -0.3 else 'WEAK'
        
        vwap_dev_status = 'ABOVE' if pd.notna(latest['vwap_deviation']) and latest['vwap_deviation'] > 1 else 'BELOW' if pd.notna(latest['vwap_deviation']) and latest['vwap_deviation'] < -1 else 'NEUTRAL'
        
        # RSI analysis
        rsi_value = safe_round(latest['rsi'], 1)
        if rsi_value >= 70:
            rsi_status = 'OVERBOUGHT'
            self._send_rsi_alert("RSI Overbought", rsi_value, current_time)
        elif rsi_value <= 30:
            rsi_status = 'OVERSOLD'
            self._send_rsi_alert("RSI Oversold", rsi_value, current_time)
        else:
            rsi_status = 'NEUTRAL'
        
        # Volume pattern
        if pd.notna(latest['vol_spike']) and latest['vol_spike']:
            vol_pattern = 'SPIKE'
        elif pd.notna(latest['vol_dry_up']) and latest['vol_dry_up']:
            vol_pattern = 'DRY UP'
        elif pd.notna(latest['vol_increasing']) and latest['vol_increasing']:
            vol_pattern = 'RISING'
        elif pd.notna(latest['vol_decreasing']) and latest['vol_decreasing']:
            vol_pattern = 'FALLING'
        else:
            vol_pattern = 'STABLE'
        
        analysis = {
            'timestamp': current_time,
            'price': safe_round(latest['close']),
            'volume_analysis': {
                'vol_ratio': safe_round(vol_ratio, 2),
                'vol_ratio_status': vol_ratio_status,
                'vol_rsi': safe_round(latest['vol_rsi'], 1),
                'vol_rsi_status': vol_rsi_status,
                'vol_momentum': safe_round(latest['vol_momentum'], 1),
                'vol_momentum_status': vol_momentum_status,
                'smart_money_ratio': safe_round(latest['smart_money_ratio'], 1),
                'smart_money_status': smart_money_status,
                'vp_correlation': safe_round(latest['vp_correlation'], 2),
                'vp_correlation_status': vp_correlation_status,
                'vwap_deviation': safe_round(latest['vwap_deviation'], 2),
                'vwap_dev_status': vwap_dev_status,
                'vol_pattern': vol_pattern,
                'vol_ml_score': safe_round(latest['vol_ml_score'], 0),
                'vol_trend': latest['vol_trend'] if pd.notna(latest['vol_trend']) else 'NEUTRAL'
            },
            'technical_levels': {
                'ema_50': safe_round(latest['ema_50'], 2),
                'fib_595': safe_round(latest['fib_595'], 2),
                'fib_650': safe_round(latest['fib_650'], 2),
                'vwap': safe_round(latest['vwap'], 2),
                'rsi': rsi_value,
                'rsi_status': rsi_status
            },
            'confluence_score': safe_round(latest['confluence_score'], 0)
        }
        
        return analysis
    
    def on_message(self, ws, message):
        """Handle WebSocket messages"""
        try:
            data = json.loads(message)
            kline = data['k']
            
            # Only process closed candles
            if kline['x']:  # is_closed
                new_data = {
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }
                
                # Add new data to DataFrame
                new_row = pd.DataFrame([new_data])
                new_row.set_index('timestamp', inplace=True)
                self.df = pd.concat([self.df, new_row])
                
                # Keep only last max_bars
                if len(self.df) > self.max_bars:
                    self.df = self.df.tail(self.max_bars)
                
                # Recalculate all indicators
                self.calculate_volume_ml_indicators()
                self.calculate_volume_ml_score()
                self.calculate_fibonacci_levels()
                self.calculate_confluence_score()
                
                # Check for Fibonacci alerts
                self.check_fibonacci_alerts()
                
                # Print analysis
                analysis = self.analyze_current_state()
                if analysis:
                    self.print_analysis(analysis)
                    
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from WebSocket message: {e}. Message: {message[:200]}...", exc_info=True)
        except KeyError as e:
            logging.error(f"Missing key '{e}' in WebSocket message. Message: {message[:200]}...", exc_info=True)
        except Exception as e:
            logging.error(f"Unhandled error processing WebSocket message: {e}", exc_info=True)
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logging.error(f"WebSocket error: {error}", exc_info=True)
        if self.is_running: # Only attempt reconnection if the system is still intended to be running
            logging.warning("WebSocket connection encountered an error. Attempting to reconnect...")
            self.reconnect_websocket()
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logging.info(f"WebSocket connection closed. Code: {close_status_code}, Message: {close_msg}")
        # Attempt to reconnect if the closure was not intentional and system is running
        if self.is_running and close_status_code not in [1000, 1001]: # 1000: Normal Closure, 1001: Going Away
            logging.warning("WebSocket closed unexpectedly. Attempting to reconnect...")
            self.reconnect_websocket()
        elif not self.is_running:
            logging.info("WebSocket closed intentionally.")
    
    def on_open(self, ws):
        """Handle WebSocket open"""
        logging.info(f"WebSocket connected for {self.symbol}")
    
    def reconnect_websocket(self):
        """Attempts to reconnect the WebSocket."""
        if self.ws:
            try:
                self.ws.close() # Attempt to close the existing connection gracefully
            except Exception as e:
                logging.debug(f"Error closing old WebSocket: {e}")
            self.ws = None # Clear the old WebSocket object
        
        if self.is_running: # Only try to reconnect if the system is still active
            logging.info("Attempting to reconnect WebSocket in 5 seconds...")
            time.sleep(5) # Wait before retrying to avoid hammering the server
            
            try:
                self.start_live_analysis_websocket_only() # Call a method that only starts the WS
            except Exception as e:
                logging.error(f"Failed to reconnect WebSocket: {e}", exc_info=True)
                # Consider implementing a more sophisticated retry mechanism (e.g., exponential backoff) here
        else:
            logging.info("Not attempting to reconnect as the system is shutting down.")

    def start_live_analysis_websocket_only(self):
        """Starts only the WebSocket connection. Used for initial setup and reconnection."""
        socket = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"
        
        self.ws = websocket.WebSocketApp(
            socket,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # Run in a separate thread to avoid blocking the main thread
        # This is crucial for reconnection attempts to not block the program
        self.websocket_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.websocket_thread.start()
        logging.info(f"WebSocket thread started for {self.symbol}.")

    def print_analysis(self, analysis):
        """Print formatted analysis"""
        # Using logging.info for structured output
        logging.info("\n" + "="*80)
        logging.info(f"ADVANCED TECHNICAL ANALYSIS - {self.symbol}")
        logging.info(f"Time: {analysis['timestamp']}")
        logging.info(f"Price: ${analysis['price']:.2f}")
        logging.info("="*80)
        
        logging.info("\nVOLUME ML ANALYSIS:")
        vol = analysis['volume_analysis']
        logging.info(f"  Vol Ratio: {vol['vol_ratio']} ({vol['vol_ratio_status']})")
        logging.info(f"  Vol RSI: {vol['vol_rsi']} ({vol['vol_rsi_status']})")
        logging.info(f"  Vol Momentum: {vol['vol_momentum']}% ({vol['vol_momentum_status']})")
        logging.info(f"  Smart Money: {vol['smart_money_ratio']}% ({vol['smart_money_status']})")
        logging.info(f"  VP Correlation: {vol['vp_correlation']} ({vol['vp_correlation_status']})")
        logging.info(f"  VWAP Deviation: {vol['vwap_deviation']}% ({vol['vwap_dev_status']})")
        logging.info(f"  Volume Pattern: {vol['vol_pattern']}")
        logging.info(f"  ML Score: {vol['vol_ml_score']} ({vol['vol_trend']})")
        
        logging.info(f"\nTECHNICAL LEVELS:")
        levels = analysis['technical_levels']
        logging.info(f"  EMA 50: ${levels['ema_50']:.2f}")
        logging.info(f"  Fibonacci 59.5%: ${levels['fib_595']:.2f}")
        logging.info(f"  Fibonacci 65.0%: ${levels['fib_650']:.2f}")
        logging.info(f"  VWAP: ${levels['vwap']:.2f}")
        logging.info(f"  RSI(9): {levels['rsi']} ({levels['rsi_status']})")
        
        logging.info(f"\nCONFLUENCE SCORE: {analysis['confluence_score']}")
        
        # Calculate distance to Fibonacci levels
        current_price = analysis['price']
        fib_595_distance = abs(current_price - levels['fib_595']) if levels['fib_595'] > 0 else 0
        fib_650_distance = abs(current_price - levels['fib_650']) if levels['fib_650'] > 0 else 0
        
        logging.info(f"\nFIBONACCI DISTANCE:")
        logging.info(f"  Distance to 59.5%: ${fib_595_distance:.4f}")
        logging.info(f"  Distance to 65.0%: ${fib_650_distance:.4f}")
        
        # Alerts
        if vol['vol_ml_score'] > 50:
            alert_msg = "🚨 ALERT: Strong Volume Accumulation Detected!"
            logging.warning(alert_msg)
            self._send_volume_alert("Strong Volume Accumulation", vol['vol_ml_score'], current_price)
        elif vol['vol_ml_score'] < -50:
            alert_msg = "🚨 ALERT: Strong Volume Distribution Detected!"
            logging.warning(alert_msg)
            self._send_volume_alert("Strong Volume Distribution", vol['vol_ml_score'], current_price)
        
        if analysis['confluence_score'] > 70:
            alert_msg = "🚨 ALERT: High Bullish Confluence!"
            logging.warning(alert_msg)
            self._send_confluence_alert("High Bullish Confluence", analysis['confluence_score'], current_price)
        elif analysis['confluence_score'] < -70:
            alert_msg = "🚨 ALERT: High Bearish Confluence!"
            logging.warning(alert_msg)
            self._send_confluence_alert("High Bearish Confluence", analysis['confluence_score'], current_price)
    
    def start_live_analysis(self):
        """Start live analysis with WebSocket"""
        # Get historical data first (only once at the beginning)
        if self.df.empty: # Only load historical data if DataFrame is empty
            if not self.get_historical_data():
                logging.error("Failed to get historical data. Exiting.")
                self.is_running = False
                return
            
            # Calculate initial indicators after historical data is loaded
            self.calculate_volume_ml_indicators()
            self.calculate_volume_ml_score()
            self.calculate_fibonacci_levels()
            self.calculate_confluence_score()
            
            # Check for initial Fibonacci alerts
            self.check_fibonacci_alerts()
            
            # Print initial analysis
            analysis = self.analyze_current_state()
            if analysis:
                self.print_analysis(analysis)
        
        logging.info(f"Starting live analysis for {self.symbol}...")
        logging.info("📊 Fibonacci Alert System is ACTIVE!")
        logging.info(f"📊 Alert tolerance: ±{self.fib_tolerance}%")
        logging.info(f"📊 Alert cooldown: {self.alert_cooldown} seconds")
        
        # Start WebSocket in a separate thread
        self.start_live_analysis_websocket_only()

    def stop_analysis(self):
        """Stops the analysis and closes the WebSocket connection."""
        logging.info("Stopping analysis...")
        self.is_running = False # Set flag to stop reconnection attempts
        if self.ws:
            try:
                self.ws.close()
                logging.info("WebSocket connection closed.")
            except Exception as e:
                logging.error(f"Error closing WebSocket: {e}", exc_info=True)
        if self.websocket_thread and self.websocket_thread.is_alive():
            self.websocket_thread.join(timeout=5) # Wait for the thread to finish
            if self.websocket_thread.is_alive():
                logging.warning("WebSocket thread did not terminate gracefully.")


# Example usage
if __name__ == "__main__":
    # Replace with your Binance API credentials
    API_KEY = "your_binance_api_key_here"
    API_SECRET = "your_binance_api_secret_here"
    
    # Initialize the analysis system
    analyzer = AdvancedTechnicalAnalysis(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='ETHUSDT',
        interval='5m'
    )
    
    # Optional: Customize alert settings
    analyzer.fib_tolerance = 0.05  # 0.05% tolerance (tighter)
    analyzer.alert_cooldown = 180  # 3 minutes cooldown
    
    try:
        analyzer.start_live_analysis()
        # Keep the main thread alive while the WebSocket runs in a daemon thread
        # This loop will run until KeyboardInterrupt or analyzer.is_running becomes False
        while analyzer.is_running:
            time.sleep(1) 
    except KeyboardInterrupt:
        logging.info("\nKeyboardInterrupt detected.")
    except Exception as e:
        logging.critical(f"An unhandled error occurred in main execution: {e}", exc_info=True)
    finally:
        analyzer.stop_analysis() # Ensure cleanup happens
        logging.info("Analysis stopped.")

