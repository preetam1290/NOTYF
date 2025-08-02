# ============================================================================
# NOTIFIER - Telegram Notifier Implementation
# ============================================================================

import logging
import asyncio
import requests
from config import TradingConfig

class TelegramNotifier:
    """Telegram Notifier for sending alerts"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger('TradingSystem.TelegramNotifier')

    async def send_message(self, message: str):
        """Send a message to the configured Telegram chat"""
        if not self.config.telegram_enabled:
            self.logger.info("Telegram alerts are disabled")
            return

        try:
            retries = 0
            while retries < self.config.telegram_retry_attempts:
                try:
                    response = requests.post(
                        f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage",
                        json={'chat_id': self.config.telegram_chat_id, 'text': message},
                        timeout=10
                    )
                    response.raise_for_status()
                    self.logger.info("Telegram message sent successfully")
                    return
                except requests.exceptions.HTTPError as http_err:
                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after", self.config.telegram_retry_delay)
                        self.logger.warning(f"Too many requests. Retrying after {retry_after} seconds...")
                        await asyncio.sleep(int(retry_after))
                    else:
                        self.logger.error(f"HTTP error occurred: {http_err}")
                        break
                except Exception as err:
                    self.logger.error(f"Error sending message: {err}")
                    break
                retries += 1
            self.logger.error("Failed to send telegram message after several attempts")
        except Exception as e:
            self.logger.error(f"Unexpected error in send_message: {e}")

