#!/usr/bin/env python3
"""
Health check script for Render deployment
This ensures the service stays alive
"""

import time
import os
from datetime import datetime

def health_check():
    """Simple health check function"""
    while True:
        try:
            # Create a simple heartbeat file
            with open('/tmp/heartbeat.txt', 'w') as f:
                f.write(f"Alive at {datetime.now()}")
            
            print(f"Health check: Service is running at {datetime.now()}")
            time.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            print(f"Health check error: {e}")
            time.sleep(60)  # Wait 1 minute on error

if __name__ == "__main__":
    health_check()
