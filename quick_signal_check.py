"""Check current signal strengths vs new threshold"""

import asyncio
import MetaTrader5 as mt5
from app.application import FXAiApplication

async def check():
    app = FXAiApplication()
    await app.initialize_components()

    # Test one symbol
    symbol = "EURUSD"

    # Get data
    h1_data = app.market_data_manager.get_bars(symbol, mt5.TIMEFRAME_H1, 200)
    if h1_data is None:
        print("❌ No data")
        return

    market_data = {'H1': h1_data}

    # Get scores
    tech = app.technical_analyzer.analyze(symbol, market_data)
    fund = app.fundamental_collector.get_news_sentiment(symbol)['score']
    sent = app.sentiment_analyzer.analyze(symbol, market_data)

    # Calculate signal (same formula as system)
    signal = tech * 0.6 + fund * 0.25 + sent * 0.15

    # Get threshold
    threshold = app.time_manager.get_session_signal_threshold(app.config)

    print(f"🔍 CURRENT STATUS for {symbol}:")
    print(f"   Technical:   {tech:.3f}")
    print(f"   Fundamental: {fund:.3f}")
    print(f"   Sentiment:   {sent:.3f}")
    print(f"   Combined:    {signal:.3f}")
    print(f"   Threshold:   {threshold:.3f}")
    print()

    if signal >= threshold:
        print(f"✅ SIGNAL ABOVE THRESHOLD - WOULD TRADE!")
        print(f"   Gap: +{signal - threshold:.3f} (strong enough)")
    else:
        gap = threshold - signal
        print(f"❌ SIGNAL BELOW THRESHOLD - BLOCKED")
        print(f"   Gap: -{gap:.3f} (need to be {gap:.3f} stronger)")
        print()
        print(f"💡 To enable trading:")
        print(f"   Option 1: Lower threshold to {signal - 0.02:.3f}")
        print(f"   Option 2: Wait for stronger market signals")

    app.shutdown()

asyncio.run(check())