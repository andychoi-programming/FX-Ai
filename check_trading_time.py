#!/usr/bin/env python3
"""
Check current time for optimal trading hours
"""
from datetime import datetime
import pytz

def check_trading_time():
    print("🕐 TRADING TIME CHECK")
    print("=" * 50)

    gmt = pytz.timezone('GMT')
    now_gmt = datetime.now(gmt)
    now_local = datetime.now()

    print(f"Current GMT time: {now_gmt.strftime('%H:%M:%S')}")
    print(f"Current local time: {now_local.strftime('%H:%M:%S')}")

    hour = now_gmt.hour
    weekday = now_gmt.weekday()  # 0=Monday, 6=Sunday

    print(f"Day of week: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][weekday]}")

    # Check if weekend
    if weekday >= 5:  # Saturday or Sunday
        print("❌ WEEKEND - Forex markets are closed!")
        return False

    # Check trading sessions
    if 8 <= hour < 12:
        print("✅ London session - Good trading time")
        print("   (High liquidity, good volatility)")
        return True
    elif 13 <= hour < 17:
        print("✅ London/NY overlap - BEST trading time")
        print("   (Maximum liquidity and volatility)")
        return True
    elif 17 <= hour < 21:
        print("✅ NY session - Good trading time")
        print("   (High liquidity, news events)")
        return True
    elif 21 <= hour < 24 or 0 <= hour < 1:
        print("⚠️  Asian session - Lower liquidity")
        print("   (Consider waiting for better hours)")
        return True
    else:
        print("⚠️  Very early Asian session - Very low liquidity")
        print("   (Not recommended for trading)")
        return False

if __name__ == "__main__":
    check_trading_time()