"""
FX-Ai Position Size Calculator Test
Verifies correct lot size calculation for $50 risk
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

def calculate_correct_lot_size(symbol, stop_loss_pips, risk_amount=50.0):
    """
    Calculate the correct lot size for a fixed dollar risk

    This is the CORRECT formula that should be used
    """

    # Get symbol information
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"{Fore.RED}Symbol {symbol} not found")
        return 0.0

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"{Fore.RED}Cannot get price for {symbol}")
        return 0.0

    # Symbol specifications
    contract_size = symbol_info.trade_contract_size
    point = symbol_info.point
    digits = symbol_info.digits

    # Determine pip size (handle 5-digit brokers)
    if digits == 3 or digits == 5:
        pip_size = point * 10
    else:
        pip_size = point

    # Calculate pip value per lot
    # Different calculations for different pairs
    if symbol == "USDJPY" or symbol.endswith("JPY"):
        # For JPY pairs
        pip_value_per_lot = (pip_size * contract_size) / tick.bid
    elif symbol.startswith("USD"):
        # USD as base currency
        pip_value_per_lot = pip_size * contract_size / tick.bid
    elif symbol.endswith("USD"):
        # USD as quote currency (most common)
        pip_value_per_lot = pip_size * contract_size
    elif "XAU" in symbol:
        # Gold
        pip_value_per_lot = pip_size * contract_size
    elif "XAG" in symbol:
        # Silver
        pip_value_per_lot = pip_size * contract_size
    else:
        # Cross pairs - simplified
        pip_value_per_lot = (pip_size * contract_size) / tick.bid

    # CORRECT FORMULA: Lot Size = Risk Amount / (Stop Loss Pips × Pip Value per Lot)
    lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)

    # Round to broker's lot step
    lot_step = symbol_info.volume_step
    lot_size = round(lot_size / lot_step) * lot_step

    # Apply broker limits
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max

    if lot_size < min_lot:
        lot_size = min_lot
    elif lot_size > max_lot:
        lot_size = max_lot

    return lot_size, pip_value_per_lot


def test_position_sizing():
    """Test position sizing for common scenarios"""

    print("\n" + "="*70)
    print(f"{Fore.CYAN}{Style.BRIGHT}FX-Ai Position Size Calculator - $50 Risk Test")
    print("="*70)

    # Initialize MT5
    if not mt5.initialize():
        print(f"{Fore.RED}Failed to initialize MT5")
        print("Please ensure MetaTrader 5 is running")
        return

    # Get account info
    account = mt5.account_info()
    if account:
        print(f"\n{Fore.GREEN}Account Information:")
        print(f"  Login: {account.login}")
        print(f"  Balance: ${account.balance:.2f}")
        print(f"  Leverage: 1:{account.leverage}")
        print(f"  Currency: {account.currency}")

    print(f"\n{Fore.YELLOW}Testing Position Sizes for $50 Risk:")
    print("-"*70)

    # Test cases
    test_scenarios = [
        ("EURUSD", 10, 50),
        ("EURUSD", 15, 50),
        ("EURUSD", 20, 50),
        ("GBPUSD", 15, 50),
        ("USDJPY", 20, 50),
        ("AUDUSD", 15, 50),
        ("XAUUSD", 30, 50),
        ("XAUUSD", 50, 50),
    ]

    results = []

    for symbol, sl_pips, risk in test_scenarios:
        # Select symbol
        if not mt5.symbol_select(symbol, True):
            print(f"{Fore.RED}Cannot select {symbol}")
            continue

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue

        # Calculate lot size
        lot_size, pip_value = calculate_correct_lot_size(symbol, sl_pips, risk)

        # Calculate actual risk with this lot size
        actual_risk = lot_size * sl_pips * pip_value

        # Get symbol info for display
        symbol_info = mt5.symbol_info(symbol)

        # Store results
        result = {
            'Symbol': symbol,
            'Price': tick.bid,
            'SL Pips': sl_pips,
            'Target Risk': f"${risk}",
            'Lot Size': lot_size,
            'Pip Value/Lot': f"${pip_value:.2f}",
            'Actual Risk': f"${actual_risk:.2f}",
            'Min Lot': symbol_info.volume_min,
            'Max Lot': symbol_info.volume_max
        }
        results.append(result)

        # Display result
        print(f"\n{Fore.CYAN}{symbol} @ {tick.bid:.5f}")
        print(f"  Stop Loss: {sl_pips} pips")
        print(f"  {'Target Risk:':15} ${risk:.2f}")
        print(f"  {'Pip Value/Lot:':15} ${pip_value:.2f}")
        print(f"  {Fore.GREEN}{'LOT SIZE:':15} {lot_size:.2f} lots")
        print(f"  {'Actual Risk:':15} ${actual_risk:.2f}")

        # Check if risk is within acceptable range (±10%)
        risk_accuracy = (actual_risk / risk) * 100
        if 90 <= risk_accuracy <= 110:
            print(f"  {Fore.GREEN}✓ Risk Accuracy: {risk_accuracy:.1f}%")
        else:
            print(f"  {Fore.YELLOW}⚠ Risk Accuracy: {risk_accuracy:.1f}%")

    # Display summary table
    print("\n" + "="*70)
    print(f"{Fore.CYAN}Summary Table:")
    print("-"*70)

    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))

    # Show the problem with your current trade
    print("\n" + "="*70)
    print(f"{Fore.RED}Your Current Problem:")
    print("-"*70)
    print(f"Your screenshot shows:")
    print(f"  Symbol: EURUSD")
    print(f"  Volume: {Fore.RED}2.34 lots (WAY TOO HIGH!)")
    print(f"  Loss: -$84.24 (exceeds $50 risk)")
    print(f"\n{Fore.GREEN}With correct calculation:")
    print(f"  For EURUSD with 20 pip SL and $50 risk")
    print(f"  Correct lot size should be: ~0.05 lots")
    print(f"  Your 2.34 lots risks about $468 (!)")

    # Provide fix instructions
    print("\n" + "="*70)
    print(f"{Fore.YELLOW}How to Fix:")
    print("-"*70)
    print("1. Copy risk_manager_fixed.py to your core/ folder")
    print("2. Replace the old risk_manager.py")
    print("3. Restart FX-Ai")
    print("4. The system will now use correct position sizing")

    mt5.shutdown()
    print("\n" + "="*70)


def calculate_your_actual_risk():
    """Calculate the actual risk of the trade shown in screenshot"""

    print("\n" + "="*70)
    print(f"{Fore.RED}Analyzing Your Current Trade:")
    print("="*70)

    # Your trade details from screenshot
    symbol = "EURUSD"
    volume = 2.34  # lots
    entry_price = 1.16123
    stop_loss = 1.16233
    current_price = 1.16159

    # Calculate pip difference
    sl_distance_price = abs(entry_price - stop_loss)
    sl_distance_pips = sl_distance_price * 10000  # For EURUSD

    # Standard pip value for EURUSD (1 standard lot)
    pip_value_per_lot = 10.0  # $10 per pip per lot for EURUSD

    # Calculate actual risk
    actual_risk = volume * sl_distance_pips * pip_value_per_lot

    print(f"Trade Analysis:")
    print(f"  Symbol: {symbol}")
    print(f"  Volume: {Fore.RED}{volume} lots")
    print(f"  Entry: {entry_price}")
    print(f"  Stop Loss: {stop_loss}")
    print(f"  SL Distance: {sl_distance_pips:.1f} pips")
    print(f"  {Fore.RED}Actual Risk: ${actual_risk:.2f}")
    print(f"  {Fore.GREEN}Target Risk: $50.00")
    print(f"  {Fore.RED}Over-risked by: ${actual_risk - 50:.2f} ({(actual_risk/50)*100:.0f}% of target)")

    # Calculate what the lot size should have been
    correct_lot_size = 50.0 / (sl_distance_pips * pip_value_per_lot)

    print(f"\n{Fore.GREEN}Correct Position Size:")
    print(f"  Should be: {correct_lot_size:.3f} lots")
    print(f"  You used: {volume:.2f} lots")
    print(f"  {Fore.RED}That's {(volume/correct_lot_size):.1f}x too large!")


if __name__ == "__main__":
    print(f"{Fore.CYAN}FX-Ai Position Size Test Tool")
    print("This will verify position sizing calculations\n")

    # First analyze the problematic trade
    calculate_your_actual_risk()

    # Then test correct calculations
    response = input(f"\n{Fore.YELLOW}Test correct position sizing? (y/n): ")
    if response.lower() == 'y':
        test_position_sizing()

    input(f"\n{Fore.CYAN}Press Enter to exit...")