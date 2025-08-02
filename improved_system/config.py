# ============================================================================
# CONFIGURATION FILE - Improved Trading System
# ============================================================================

import logging
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TradingConfig:
    """Trading system configuration"""
    # Binance API Configuration
    api_key: str
    api_secret: str
    symbol: str = 'ETHUSDT'
    interval: str = '5m'
    
    # Core Trading Parameters
    swing_lookback: int = 249
    ema_length: int = 50
    max_bars: int = 500
    
    # Volume Parameters
    vol_lookback: int = 30
    vol_threshold: float = 2.0
    ml_sensitivity: float = 2.0
    
    # Alert System Parameters
    fib_tolerance: float = 0.1  # Percentage tolerance for Fibonacci level matching
    alert_cooldown: int = 300   # 5 minutes cooldown between alerts for same level
    
    # WebSocket Retry Configuration
    max_reconnect_attempts: int = 10
    initial_retry_delay: int = 1
    max_retry_delay: int = 60
    backoff_multiplier: float = 2.0
    
    # Telegram Configuration
    telegram_enabled: bool = True
    telegram_bot_token: str = '8459911523:AAHXBnaNywpvSNU59MhD6G0RqJn993VKhUU'
    telegram_chat_id: str = '6080078099'
    telegram_retry_attempts: int = 3
    telegram_retry_delay: int = 5
    
    # Logging Configuration
    log_level: str = 'INFO'
    log_file: str = 'trading_system.log'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Global configuration instance
config: Optional[TradingConfig] = None

def setup_logging(log_level: str = 'INFO', log_file: str = 'trading_system.log') -> logging.Logger:
    """Setup logging configuration"""
    # Create logger
    logger = logging.getLogger('TradingSystem')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_config(api_key: str, api_secret: str, **kwargs) -> TradingConfig:
    """Load configuration with provided API credentials"""
    global config
    config = TradingConfig(
        api_key=api_key,
        api_secret=api_secret,
        **kwargs
    )
    return config
