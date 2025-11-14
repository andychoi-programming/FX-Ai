"""Check what session is currently detected"""

import MetaTrader5 as mt5
from utils.time_manager import TimeManager

mt5.initialize()

time_manager = TimeManager()
current_session = time_manager.get_current_session()

print(f"Current session: {current_session}")

# Check preferred sessions from config
import json
with open('config/config.json', 'r') as f:
    config = json.load(f)

preferred = config.get('trading_rules', {}).get('session_filter', {}).get('preferred_sessions', [])
print(f"Preferred sessions: {preferred}")
print(f"Is current session preferred: {current_session in preferred}")

mt5.shutdown()