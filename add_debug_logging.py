"""
FX-Ai Debug Logging Patch
==========================

This script provides code snippets to add to your FX-Ai system to see
EXACTLY why signals are being rejected.

INSTRUCTIONS:
1. Locate your main.py or trading_engine.py file
2. Find the signal analysis loop (where it analyzes 30 symbols)
3. Add the logging code shown below BEFORE the validation checks
4. Restart FX-Ai
5. Watch logs to see actual signal values

"""

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║                    FX-Ai Debug Logging Patch                          ║
║                                                                        ║
║  Add this code to your trading system to see why signals fail        ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: Locate Your Trading Loop
═════════════════════════════════

Find the section in main.py or trading_engine.py that looks like:

    for symbol in symbols:
        # Analyze symbol
        # Generate signals
        # Check if trade should execute

This is typically in a function like:
  • analyze_symbols()
  • generate_signals()
  • trading_loop()
  • main_loop()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 2: Add This Code AFTER Signal Calculation
═══════════════════════════════════════════════

After you calculate signal components (technical_score, ml_prediction, etc.)
but BEFORE the validation checks, add this logging:

""")

print("""
# ============================================================================
# DEBUG LOGGING - Add this to your trading loop
# ============================================================================

# After calculating all signal components, add this:

logger.info(f"\\n{'='*70}")
logger.info(f"SIGNAL ANALYSIS: {symbol}")
logger.info(f"{'='*70}")

# Log individual components
logger.info(f"  COMPONENTS:")
logger.info(f"    Technical Score:    {technical_score:.3f} (weight: 0.25)")
logger.info(f"    ML Prediction:      {ml_prediction:.3f} (weight: 0.30)")
logger.info(f"    Sentiment Score:    {sentiment_score:.3f} (weight: 0.20)")
logger.info(f"    Fundamental Score:  {fundamental_score:.3f} (weight: 0.15)")
logger.info(f"    S/R Score:          {sr_score:.3f} (weight: 0.10)")

# Calculate and log combined signal
signal_strength = (
    0.25 * technical_score +
    0.30 * ml_prediction +
    0.20 * sentiment_score +
    0.15 * fundamental_score +
    0.10 * sr_score
)

logger.info(f"\\n  COMBINED SIGNAL STRENGTH: {signal_strength:.3f}")
logger.info(f"  Required threshold: {min_signal_strength}")

if signal_strength >= min_signal_strength:
    logger.info(f"  ✅ PASS: Signal strength sufficient")
else:
    logger.info(f"  ❌ FAIL: Signal too weak ({signal_strength:.3f} < {min_signal_strength})")
    continue  # Skip to next symbol

# Log market conditions
tick = mt5.symbol_info_tick(symbol)
info = mt5.symbol_info(symbol)
if 'JPY' in symbol:
    pip_size = 0.01
else:
    pip_size = 0.0001
current_spread = (tick.ask - tick.bid) / pip_size

logger.info(f"\\n  MARKET CONDITIONS:")
logger.info(f"    Current Spread: {current_spread:.1f} pips")
logger.info(f"    Max allowed: {max_spread} pips")

if current_spread > max_spread:
    logger.info(f"  ❌ FAIL: Spread too high")
    continue

logger.info(f"  ✅ PASS: Spread acceptable")

# Check daily limit
if risk_manager.has_traded_today(symbol):
    logger.info(f"\\n  ❌ FAIL: Already traded this symbol today")
    continue

logger.info(f"\\n  ✅ PASS: Daily limit OK")

# If we get here, all basic checks passed
logger.info(f"\\n  🎯 SIGNAL PASSED ALL CHECKS - Proceeding to trade setup")
logger.info(f"{'='*70}\\n")

# ============================================================================
# END DEBUG LOGGING
# ============================================================================

""")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 3: Alternative - Add to Specific Locations
════════════════════════════════════════════════

If you can't find the main loop, add logging at these specific points:

""")

print("""
Location 1: After Technical Analysis
─────────────────────────────────────
# In technical_analyzer.py or wherever technical analysis happens

logger.info(f"{symbol} Technical Analysis: Score={technical_score:.3f}, "
            f"VWAP={vwap_signal}, EMA={ema_signal}, RSI={rsi_value:.1f}")


Location 2: After ML Prediction
────────────────────────────────
# In ml_predictor.py or wherever ML prediction happens

logger.info(f"{symbol} ML Prediction: {ml_prediction:.3f}, "
            f"Direction={'BUY' if ml_prediction > 0.5 else 'SELL'}, "
            f"Confidence={abs(ml_prediction - 0.5) * 2:.3f}")


Location 3: After Sentiment Analysis
─────────────────────────────────────
# In sentiment_analyzer.py

logger.info(f"{symbol} Sentiment: Score={sentiment_score:.3f}, "
            f"Sources={len(sentiment_sources)}, "
            f"Bullish={bullish_count}, Bearish={bearish_count}")


Location 4: In Risk Manager Validation
───────────────────────────────────────
# In risk_manager.py - can_trade() method

logger.info(f"{symbol} Risk Check: "
            f"Traded today={self.has_traded_today(symbol)}, "
            f"Daily limit={self.max_trades_per_day}, "
            f"Current positions={len(self.current_positions)}")


Location 5: Before Trade Rejection
───────────────────────────────────
# Wherever signals are rejected, add:

logger.warning(f"{symbol} SIGNAL REJECTED - "
               f"Reason: {rejection_reason}, "
               f"Signal: {signal_strength:.3f}, "
               f"Spread: {current_spread:.1f}, "
               f"Threshold: {min_signal_strength}")

""")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 4: View the New Logs
══════════════════════════

After restarting FX-Ai with the new logging:

    # View in real-time
    tail -f logs/fxai_*.log | grep -E "(SIGNAL|COMPONENTS|PASS|FAIL|REJECT)"

    # View last 100 signal analyses
    grep -A 20 "SIGNAL ANALYSIS" logs/fxai_*.log | tail -100

    # Count how many symbols failed each check
    grep "FAIL:" logs/fxai_*.log | cut -d: -f3 | sort | uniq -c

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 5: Analyze Results
════════════════════════

Look for patterns in the logs:

Pattern 1: All symbols show "Signal too weak"
└→ Lower min_signal_strength threshold in config

Pattern 2: All symbols show "Spread too high"
└→ Increase max_spread or trade during better hours

Pattern 3: All symbols show "Already traded today"
└→ Reset daily limits or wait until tomorrow

Pattern 4: Mix of reasons
└→ System is working correctly, just being selective

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICK TEST: Add This Single Line
═════════════════════════════════

If you want to quickly see signal values with minimal code changes,
add this ONE line wherever signals are calculated:

    logger.info(f"{symbol} Signal: {signal_strength:.3f} "
                f"(Tech:{tech:.2f} ML:{ml:.2f} Sent:{sent:.2f} Fund:{fund:.2f})")

This will at least show you the signal values for each symbol.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FILE LOCATIONS TO CHECK
════════════════════════

Your FX-Ai source files are likely in:
  • C:\\Users\\andyc\\python\\FX-Ai\\
  
Key files to edit:
  • main.py                       (main trading loop)
  • core/trading_engine.py        (trade execution)
  • analysis/technical_analyzer.py (technical analysis)
  • ai/ml_predictor.py            (ML predictions)
  • core/risk_manager.py          (risk checks)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AFTER ADDING LOGGING
═════════════════════

1. Restart FX-Ai system
2. Watch logs for detailed signal information
3. Identify which check is failing most often
4. Adjust configuration accordingly
5. Re-run system

Expected output with new logging:

    ======================================================================
    SIGNAL ANALYSIS: EURUSD
    ======================================================================
      COMPONENTS:
        Technical Score:    0.612 (weight: 0.25)
        ML Prediction:      0.548 (weight: 0.30)
        Sentiment Score:    0.423 (weight: 0.20)
        Fundamental Score:  0.501 (weight: 0.15)
        S/R Score:          0.387 (weight: 0.10)

      COMBINED SIGNAL STRENGTH: 0.515
      Required threshold: 0.4
      ✅ PASS: Signal strength sufficient

      MARKET CONDITIONS:
        Current Spread: 1.2 pips
        Max allowed: 3.0 pips
      ✅ PASS: Spread acceptable

      ✅ PASS: Daily limit OK

      🎯 SIGNAL PASSED ALL CHECKS - Proceeding to trade setup
    ======================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TROUBLESHOOTING
═══════════════

Issue: "NameError: 'logger' is not defined"
Fix: Add at top of file: import logging; logger = logging.getLogger(__name__)

Issue: Variables like 'technical_score' not found
Fix: Use whatever variable names your code actually uses

Issue: Logs still show no detail
Fix: Check logger level is set to INFO, not WARNING or ERROR

Issue: Too many log messages
Fix: Only add logging inside the symbol loop, not for every data point

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUMMARY
═══════

The problem is that your current logs say "No opportunities found" but don't
explain WHY. Adding detailed logging will show you:

  ✓ What signal strength values are being generated
  ✓ Which validation check is failing
  ✓ Whether it's spreads, signal strength, or daily limits
  ✓ How close signals are to passing threshold

This information will let you make informed decisions about whether to adjust
your configuration or accept that the system is working correctly but being
selective about trade quality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After adding this logging and restarting FX-Ai, run:

    tail -f logs/fxai_*.log

And you'll see exactly why each symbol is accepted or rejected!

""")

if __name__ == "__main__":
    pass