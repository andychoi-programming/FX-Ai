import MetaTrader5 as mt5
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_filling_modes():
    """Test which filling modes work with TIOMarkets"""

    # Initialize MT5
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return

    logger.info("MT5 initialized successfully")

    # Test symbol
    symbol = "EURUSD"  # Common symbol to test

    # Select symbol
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Failed to select {symbol}")
        mt5.shutdown()
        return

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Symbol {symbol} not found")
        mt5.shutdown()
        return

    logger.info(f"Testing filling modes for {symbol}")
    logger.info(f"Symbol filling modes: {symbol_info.filling_mode}")

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"Failed to get tick for {symbol}")
        mt5.shutdown()
        return

    # Test each filling mode
    filling_modes = [
        ("IOC", mt5.ORDER_FILLING_IOC),
        ("RETURN", mt5.ORDER_FILLING_RETURN),
        ("FOK", mt5.ORDER_FILLING_FOK),
        ("NONE_0", 0),  # Try 0
        ("NONE_2", 2),  # Try 2 even if not supported
    ]

    working_modes = []

    for name, mode in filling_modes:
        logger.info(f"Testing {name} (value: {mode})")

        # Create test order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,  # Small test volume
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "deviation": 10,
            "magic": 123456,
            "comment": f"Test {name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mode,
        }

        # Use order_check instead of actual order to test
        result = mt5.order_check(request)

        if result and (result.retcode == mt5.TRADE_RETCODE_DONE or result.retcode == 0):
            logger.info(f"✅ {name} works! Retcode: {result.retcode}")
            working_modes.append((name, mode))
        else:
            retcode = result.retcode if result else "None"
            comment = result.comment if result and hasattr(result, 'comment') else "No comment"
            logger.error(f"❌ {name} failed! Retcode: {retcode}, Comment: {comment}")

    logger.info(f"\nWorking filling modes: {working_modes}")

    if working_modes:
        logger.info(f"Recommended mode: {working_modes[0][0]} (value: {working_modes[0][1]})")
    else:
        logger.error("No filling modes work!")

    mt5.shutdown()

if __name__ == "__main__":
    test_filling_modes()