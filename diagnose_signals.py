"""
Diagnostic script to check trading signal generation in detail
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.application import FXAiApplication
import MetaTrader5 as mt5

async def diagnose_signals():
    """Diagnose why no trading signals are being generated"""
    print("🔍 DIAGNOSTIC: Checking Trading Signal Generation")
    print("=" * 60)

    app = FXAiApplication()

    try:
        # Initialize components
        await app.initialize_components()

        # Check if pre-trading checklist passes
        checklist_passed = await app.trading_orchestrator.pre_trading_checklist()
        print(f"Pre-trading checklist: {'PASS' if checklist_passed else 'FAIL'}")

        if not checklist_passed:
            print("❌ Cannot proceed - pre-trading checks failed")
            return

        # Test a few key symbols
        test_symbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDJPY']

        print(f"\n📊 Analyzing {len(test_symbols)} test symbols in detail...")
        print("-" * 60)

        for symbol in test_symbols:
            print(f"\n🔎 Analyzing {symbol}:")
            print("-" * 30)

            try:
                # Check if we can trade this symbol
                can_trade, reason = app.trading_orchestrator.risk_manager.can_trade(symbol)
                print(f"Risk Manager: {'✅ Can trade' if can_trade else '❌ Cannot trade'} - {reason}")

                if not can_trade:
                    continue

                # Check trading hours
                if hasattr(app, 'schedule_manager') and app.schedule_manager:
                    can_trade_hours = app.schedule_manager.can_trade_symbol(symbol)
                    next_time = app.schedule_manager.get_next_trading_time(symbol)
                    print(f"Trading Hours: {'✅ OK' if can_trade_hours else '❌ Outside hours'} - Next: {next_time}")
                else:
                    print("Trading Hours: ⚠️ No schedule manager")

                # Check for existing positions/orders
                existing_orders = mt5.orders_get(symbol=symbol)
                our_orders = [o for o in (existing_orders or []) if hasattr(o, 'magic') and o.magic == app.magic_number]
                print(f"Pending Orders: {len(our_orders)} from our system")

                existing_positions = mt5.positions_get(symbol=symbol)
                print(f"Open Positions: {len(existing_positions or [])}")

                # Get market data
                h1_data = app.market_data_manager.get_bars(symbol, mt5.TIMEFRAME_H1, 200)
                print(f"Market Data: {'✅ Available' if h1_data is not None and len(h1_data) >= 50 else '❌ Insufficient'} ({len(h1_data) if h1_data else 0} bars)")

                if not h1_data or len(h1_data) < 50:
                    continue

                # Get current price
                tick_data = app.market_data_manager.get_market_data(symbol)
                if tick_data:
                    bid = tick_data.get('bid', 0)
                    ask = tick_data.get('ask', 0)
                    current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else tick_data.get('last', 0)
                    print(f"Current Price: {current_price:.5f} (bid: {bid}, ask: {ask})")
                else:
                    current_price = h1_data['close'].iloc[-1] if len(h1_data) > 0 else 0
                    print(f"Current Price: {current_price:.5f} (from H1 close)")

                if current_price <= 0:
                    print("❌ Invalid price, skipping")
                    continue

                # Get analyzer scores
                market_data = {'H1': h1_data}

                technical_score = app.technical_analyzer.analyze(symbol, market_data)
                fundamental_score = app.fundamental_collector.get_news_sentiment(symbol)['score']
                sentiment_score = app.sentiment_analyzer.analyze(symbol, market_data)

                print(f"Technical Score: {technical_score:.3f}")
                print(f"Fundamental Score: {fundamental_score:.3f}")
                print(f"Sentiment Score: {sentiment_score:.3f}")

                # Get ML score
                ml_score = 0.0
                ml_confidence = 0.0
                if app.ml_predictor:
                    try:
                        technical_signals = app.technical_analyzer.analyze_symbol(symbol, market_data)
                        ml_prediction = await app.ml_predictor.predict(symbol, h1_data, technical_signals, 'H1')
                        if ml_prediction and 'confidence' in ml_prediction:
                            ml_score = ml_prediction['confidence']
                            ml_confidence = ml_prediction.get('confidence', 0.0)
                        print(f"ML Score: {ml_score:.3f} (confidence: {ml_confidence:.3f})")
                    except Exception as e:
                        print(f"ML Score: ❌ Error - {e}")
                else:
                    print("ML Score: ⚠️ No ML predictor available")

                # Calculate signal strength
                current_session = app.trading_orchestrator.time_manager.get_current_session()
                print(f"Current Session: {current_session}")

                # Apply session boosts
                boosted_fundamental = fundamental_score
                boosted_sentiment = sentiment_score
                if current_session == 'london':
                    boosted_fundamental = min(0.6, fundamental_score + 0.1)
                    boosted_sentiment = min(0.6, sentiment_score + 0.1)
                elif current_session == 'new_york':
                    boosted_fundamental = min(0.65, fundamental_score + 0.15)
                    boosted_sentiment = min(0.65, sentiment_score + 0.15)

                # Calculate combined signal strength
                if app.trading_orchestrator.adaptive_learning and current_session not in ['london', 'new_york']:
                    signal_strength = app.trading_orchestrator.adaptive_learning.calculate_signal_strength(
                        technical_score, boosted_fundamental, boosted_sentiment, ml_score)
                else:
                    if ml_confidence > 0.6:
                        signal_strength = (technical_score * 0.4 + boosted_fundamental * 0.2 +
                                         boosted_sentiment * 0.15 + ml_score * 0.25)
                    else:
                        signal_strength = (technical_score * 0.6 + boosted_fundamental * 0.25 + boosted_sentiment * 0.15)

                print(f"Signal Strength: {signal_strength:.3f}")

                # Check minimum threshold
                min_strength = app.trading_orchestrator.time_manager.get_session_signal_threshold(app.config)
                print(f"Minimum Threshold: {min_strength:.3f}")

                if signal_strength >= min_strength:
                    direction = 'BUY' if technical_score > 0.5 else 'SELL'
                    print(f"🎯 TRADE SIGNAL: {direction} (Strength: {signal_strength:.3f})")
                else:
                    print(f"❌ No signal: Strength {signal_strength:.3f} < threshold {min_strength:.3f}")

            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")

        print("\n" + "=" * 60)
        print("🎯 CONCLUSION:")
        print("If all symbols show low signal strength, then there really are no opportunities.")
        print("This is NORMAL during Tokyo session (low liquidity).")
        print("Try running this during London/NY sessions for better opportunities.")

    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        app.shutdown()

if __name__ == "__main__":
    asyncio.run(diagnose_signals())