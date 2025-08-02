# ============================================================================
# DATA FETCHER - Async WebSocket with Exponential Backoff
# ============================================================================

import asyncio
import json
import logging
import time
from typing import Callable, Optional, Dict, Any
import pandas as pd
import websockets
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import TradingConfig

class AsyncDataFetcher:
    """Async data fetcher with exponential backoff retry logic"""
    
    def __init__(self, config: TradingConfig, on_new_data: Callable[[Dict[str, Any]], None]):
        self.config = config
        self.logger = logging.getLogger('TradingSystem.DataFetcher')
        self.on_new_data = on_new_data
        
        # Initialize Binance client
        self.client = Client(config.api_key, config.api_secret)
        
        # WebSocket connection state
        self.websocket = None
        self.is_running = False
        self.retry_count = 0
        self.last_heartbeat = 0
        
        # Connection monitoring
        self.last_data_time = 0
        self.connection_timeout = 30  # 30 seconds without data = reconnect
        
        self.logger.info(f"DataFetcher initialized for {config.symbol}")
    
    async def get_historical_data(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Get historical kline data from Binance API"""
        try:
            self.logger.info(f"Fetching {limit} historical bars for {self.config.symbol}")
            
            # Get historical data
            klines = self.client.get_historical_klines(
                self.config.symbol, 
                self.config.interval, 
                f"{limit} hours ago UTC"
            )
            
            if not klines:
                self.logger.error("No historical data received")
                return None
            
            # Convert to DataFrame
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
            
            self.logger.info(f"Successfully loaded {len(df)} historical bars")
            return df
            
        except BinanceAPIException as e:
            self.logger.error(f"Binance API error while fetching historical data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error while fetching historical data: {e}")
            return None
    
    async def start_websocket(self):
        """Start WebSocket connection with retry logic"""
        self.is_running = True
        self.retry_count = 0
        
        while self.is_running:
            try:
                await self._connect_websocket()
            except Exception as e:
                self.logger.error(f"WebSocket connection failed: {e}")
                
                if not self.is_running:
                    break
                
                # Calculate retry delay with exponential backoff
                retry_delay = min(
                    self.config.initial_retry_delay * (self.config.backoff_multiplier ** self.retry_count),
                    self.config.max_retry_delay
                )
                
                self.retry_count += 1
                
                if self.retry_count >= self.config.max_reconnect_attempts:
                    self.logger.error(f"Max retry attempts ({self.config.max_reconnect_attempts}) reached. Stopping...")
                    self.is_running = False
                    break
                
                self.logger.info(f"Retrying in {retry_delay} seconds (attempt {self.retry_count}/{self.config.max_reconnect_attempts})")
                await asyncio.sleep(retry_delay)
    
    async def _connect_websocket(self):
        """Connect to Binance WebSocket"""
        uri = f"wss://stream.binance.com:9443/ws/{self.config.symbol.lower()}@kline_{self.config.interval}"
        
        self.logger.info(f"Connecting to WebSocket: {uri}")
        
        async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as websocket:
            self.websocket = websocket
            self.retry_count = 0  # Reset retry count on successful connection
            self.last_data_time = time.time()
            
            self.logger.info(f"WebSocket connected for {self.config.symbol}")
            
            # Start heartbeat monitoring
            heartbeat_task = asyncio.create_task(self._monitor_connection())
            
            try:
                async for message in websocket:
                    if not self.is_running:
                        break
                    
                    await self._handle_message(message)
                    self.last_data_time = time.time()
                    
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"WebSocket connection closed: {e}")
            except Exception as e:
                self.logger.error(f"Error in WebSocket message loop: {e}")
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            kline = data.get('k')
            
            if not kline:
                return
            
            # Only process closed candles
            if kline.get('x'):  # is_closed
                candle_data = {
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'is_closed': True
                }
                
                # Call the callback function
                self.on_new_data(candle_data)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode WebSocket message: {e}")
        except KeyError as e:
            self.logger.error(f"Missing key in WebSocket message: {e}")
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
    async def _monitor_connection(self):
        """Monitor WebSocket connection health"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                current_time = time.time()
                time_since_last_data = current_time - self.last_data_time
                
                if time_since_last_data > self.connection_timeout:
                    self.logger.warning(f"No data received for {time_since_last_data:.1f} seconds. Connection may be stale.")
                    # Force reconnection by raising an exception
                    raise ConnectionError("Connection timeout - no data received")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Connection monitor error: {e}")
                raise
    
    def stop(self):
        """Stop the WebSocket connection"""
        self.logger.info("Stopping WebSocket connection...")
        self.is_running = False
        
        if self.websocket:
            asyncio.create_task(self.websocket.close())
