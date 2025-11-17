from core.risk_manager import RiskManager

rm = RiskManager(config={}, db_path=None, mt5_connector=None)

# Test XAUUSD
print("="*60)
print("XAUUSD (Gold) Testing")
print("="*60)
risk_amount = 50.0
stop_loss_pips = 20.0

lots = rm.calculate_position_size('XAUUSD', stop_loss_pips, risk_amount)
pip_value_per_lot = 10.0  # Should be $10
actual_risk = lots * stop_loss_pips * pip_value_per_lot

print(f"Risk Target: ${risk_amount}")
print(f"Stop Loss: {stop_loss_pips} pips")
print(f"Calculated Position Size: {lots:.4f} lots")
print(f"Pip Value per Lot: ${pip_value_per_lot}")
print(f"Actual Risk: ${actual_risk:.2f}")
print(f"Error: ${abs(actual_risk - risk_amount):.2f}")

if abs(actual_risk - risk_amount) < 1.0:
    print("✅ XAUUSD calculation CORRECT")
else:
    print(f"❌ XAUUSD calculation WRONG - Off by ${abs(actual_risk - risk_amount):.2f}")

print("\n" + "="*60)
print("XAGUSD (Silver) Testing")
print("="*60)

lots = rm.calculate_position_size('XAGUSD', stop_loss_pips, risk_amount)
pip_value_per_lot = 50.0  # Should be $50
actual_risk = lots * stop_loss_pips * pip_value_per_lot

print(f"Risk Target: ${risk_amount}")
print(f"Stop Loss: {stop_loss_pips} pips")
print(f"Calculated Position Size: {lots:.4f} lots")
print(f"Pip Value per Lot: ${pip_value_per_lot}")
print(f"Actual Risk: ${actual_risk:.2f}")
print(f"Error: ${abs(actual_risk - risk_amount):.2f}")

if abs(actual_risk - risk_amount) < 1.0:
    print("✅ XAGUSD calculation CORRECT")
else:
    print(f"❌ XAGUSD calculation WRONG - Off by ${abs(actual_risk - risk_amount):.2f}")