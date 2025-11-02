import json

# Load optimized parameters
data = json.load(open('models/parameter_optimization/optimal_parameters.json'))

# Analyze parameter variations across all symbols
print('=== SYMBOL-SPECIFIC TRADING LOGIC ANALYSIS ===')
print()

# Check entry time variations
entry_times = {}
sl_tp_combinations = {}
breakeven_settings = {}
trailing_settings = {}

for symbol, symbol_data in data.items():
    if 'H1' in symbol_data:
        params = symbol_data['H1']['optimal_params']
        entry_key = f"{params.get('entry_hour_start', '?')}-{params.get('entry_hour_end', '?')}"
        risk_key = f"SL{params.get('sl_pips', '?')}/TP{params.get('tp_pips', '?')}"
        breakeven_key = f"BE{params.get('breakeven_trigger', '?')}"
        trailing_key = f"TA{params.get('trailing_activation', '?')}/TD{params.get('trailing_distance', '?')}"

        entry_times[entry_key] = entry_times.get(entry_key, 0) + 1
        sl_tp_combinations[risk_key] = sl_tp_combinations.get(risk_key, 0) + 1
        breakeven_settings[breakeven_key] = breakeven_settings.get(breakeven_key, 0) + 1
        trailing_settings[trailing_key] = trailing_settings.get(trailing_key, 0) + 1

print('ENTRY TIME VARIATIONS:')
for time_range, count in entry_times.items():
    print(f"  {time_range}: {count} symbols")
print()

print('RISK PARAMETER COMBINATIONS:')
for risk_combo, count in sl_tp_combinations.items():
    print(f"  {risk_combo}: {count} symbols")
print()

print('BREAKEVEN SETTINGS:')
for be_setting, count in breakeven_settings.items():
    print(f"  {be_setting}: {count} symbols")
print()

print('TRAILING STOP SETTINGS:')
for ts_setting, count in trailing_settings.items():
    print(f"  {ts_setting}: {count} symbols")
print()

print('SAMPLE SYMBOL COMPARISON:')
symbols = ['AUDCAD', 'EURUSD', 'GBPJPY', 'USDJPY', 'NZDUSD']
print('=' * 80)
for symbol in symbols:
    if symbol in data:
        params = data[symbol]['H1']['optimal_params']
        print(f'{symbol}:')
        print(f'  Entry: {params.get("entry_hour_start", "?")}-{params.get("entry_hour_end", "?")} hours')
        print(f'  Risk: SL{params.get("sl_pips", "?")}/TP{params.get("tp_pips", "?")} pips')
        print(f'  Breakeven: {params.get("breakeven_trigger", "?")} pips')
        print(f'  Trailing: Activate@{params.get("trailing_activation", "?")}pips, Distance@{params.get("trailing_distance", "?")}pips')
        print(f'  Day-of-week: Mon+{params.get("monday_entry_delay", "?")}h, Fri-{params.get("friday_early_exit", "?")}h')
        print()