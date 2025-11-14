"""
Diagnose why system finds no trading opportunities
Shows actual signal strengths vs threshold
"""

import asyncio
import sys
from app.application import FXAiApplication

async def diagnose():
    print("🔍 DIAGNOSING NO TRADE OPPORTUNITIES\n")
    print("="*70)

    # Initialize app
    app = FXAiApplication()
    await app.initialize_components()

    # Get configuration
    config = app.config
    symbols = config.get('trading', {}).get('symbols', [])[:5]  # Test first 5

    print(f"Testing {len(symbols)} symbols...")
    print(f"Current server time: {app.get_current_mt5_time()}")
    print(f"Current session: {app.time_manager.get_current_session()}")
    print(f"Signal threshold: {app.time_manager.get_session_signal_threshold(config)}")
    print("="*70)

    import MetaTrader5 as mt5

    for symbol in symbols:
        try:
            # Get market data
            h1_data = app.market_data_manager.get_bars(symbol, mt5.TIMEFRAME_H1, 200)
            if h1_data is None or len(h1_data) < 50:
                print(f"\n{symbol}: ❌ Insufficient data")
                continue

            market_data = {'H1': h1_data}

            # Get tick data for current price
            tick_data = app.market_data_manager.get_market_data(symbol)
            if tick_data:
                bid = tick_data.get('bid', 0)
                ask = tick_data.get('ask', 0)
                current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
            else:
                current_price = h1_data['close'].iloc[-1]

            # Get analysis scores
            technical_score = app.technical_analyzer.analyze(symbol, market_data)
            fundamental_score = app.fundamental_collector.get_news_sentiment(symbol)['score']
            sentiment_score = app.sentiment_analyzer.analyze(symbol, market_data)

            # Apply session boost
            current_session = app.time_manager.get_current_session()
            if current_session == 'london':
                fundamental_score = min(0.6, fundamental_score + 0.1)
                sentiment_score = min(0.6, sentiment_score + 0.1)
            elif current_session == 'new_york':
                fundamental_score = min(0.65, fundamental_score + 0.15)
                sentiment_score = min(0.65, sentiment_score + 0.15)

            # Calculate combined signal
            if app.adaptive_learning and current_session not in ['london', 'new_york']:
                signal_strength = app.adaptive_learning.calculate_signal_strength(
                    technical_score, fundamental_score, sentiment_score, 0.0)
            else:
                signal_strength = (technical_score * 0.6 +
                                 fundamental_score * 0.25 +
                                 sentiment_score * 0.15)

            # Get threshold
            min_strength = app.time_manager.get_session_signal_threshold(config)

            # Display results
            print(f"\n{symbol}:")
            print(f"  Price: {current_price:.5f}")
            print(f"  Technical:   {technical_score:.3f}")
            print(f"  Fundamental: {fundamental_score:.3f}")
            print(f"  Sentiment:   {sentiment_score:.3f}")
            print(f"  Combined:    {signal_strength:.3f}")
            print(f"  Threshold:   {min_strength:.3f}")

            if signal_strength >= min_strength:
                print(f"  Status: ✅ WOULD TRADE")
            else:
                gap = min_strength - signal_strength
                print(f"  Status: ❌ TOO WEAK (need +{gap:.3f})")

        except Exception as e:
            print(f"\n{symbol}: ❌ Error - {e}")

    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS:")
    print("="*70)

    # Analyze session
    current_session = app.time_manager.get_current_session()
    if current_session in ['sydney', 'tokyo']:
        print("⚠️  Currently in LOW LIQUIDITY session (Sydney/Tokyo)")
        print("   - Weak signals are NORMAL")
        print("   - System correctly refusing weak trades")
        print("   - WAIT for London session (08:00 server time)")
    elif current_session == 'london':
        print("✅ Currently in GOOD session (London)")
        print("   - If still no signals, market may be ranging")
        print("   - Consider lowering threshold temporarily")
    elif current_session == 'new_york':
        print("✅ Currently in GOOD session (New York)")
        print("   - If still no signals, check market conditions")
    else:
        print("⚠️  Currently OUTSIDE major sessions")
        print("   - Low activity is expected")

    print("\n📊 SIGNAL STRENGTH ANALYSIS:")
    # More detailed analysis would go here

    app.shutdown()

if __name__ == "__main__":
    asyncio.run(diagnose())