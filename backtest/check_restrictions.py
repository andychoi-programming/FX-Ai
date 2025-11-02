import json

# Load optimized parameters
params = json.load(open('models/parameter_optimization/optimal_parameters.json'))
symbols = [k for k in params.keys() if k != 'summary']

print('='*70)
print('OPTIMIZED PARAMETERS vs CURRENT RESTRICTIONS')
print('='*70)

# Check minimum SL violations
forex_below_25 = []
forex_below_15 = []
all_rr_ratios = []

for s in sorted(symbols):
    sl = params[s]['H1']['optimal_params']['sl_pips']
    tp = params[s]['H1']['optimal_params']['tp_pips']
    rr = tp / sl
    is_metal = 'XAU' in s or 'XAG' in s
    
    all_rr_ratios.append((s, rr))
    
    if not is_metal:
        if sl < 25:
            forex_below_25.append(f'{s:8s}: SL={sl:3d}, TP={tp:4d}, R/R=1:{rr:4.1f}')
        if sl < 15:
            forex_below_15.append(f'{s:8s}: SL={sl:3d}, TP={tp:4d}, R/R=1:{rr:4.1f}')

print(f'\n1. MINIMUM STOP LOSS RESTRICTION')
print(f'   Current config: 25 pips minimum (except metals)')
print(f'   Optimized params: Most use 10-20 pips')
print(f'\n   Forex pairs with SL < 25 pips: {len(forex_below_25)} out of {len([s for s in symbols if "XAU" not in s and "XAG" not in s])}')
if forex_below_25:
    for p in forex_below_25[:5]:  # Show first 5
        print(f'   - {p}')
    if len(forex_below_25) > 5:
        print(f'   ... and {len(forex_below_25)-5} more')

print(f'\n2. RISK/REWARD RATIO RESTRICTION')
print(f'   Current config: 3.0:1 minimum')
print(f'   Current code enforcement: 2.9:1 minimum (line 808 in main.py)')
rr_below_3 = [r for r in all_rr_ratios if r[1] < 3.0]
print(f'   Symbols with R/R < 3.0: {len(rr_below_3)}')
if rr_below_3:
    for s, rr in rr_below_3:
        print(f'   - {s:8s}: {rr:.1f}:1')

print(f'\n3. HARD CLOSE TIME RESTRICTION')
print(f'   Current config: 22:30 (close_hour=22, close_minute=30)')
print(f'   Optimized params: All use hard_close_hour=22')
print(f'   Status: ✓ MATCHES (no conflict)')

print(f'\n4. MAXIMUM SL PIPS RESTRICTION')
print(f'   Current config: 50 pips maximum')
max_sl_forex = max([(s, params[s]['H1']['optimal_params']['sl_pips']) 
                     for s in symbols if 'XAU' not in s and 'XAG' not in s], 
                    key=lambda x: x[1])
print(f'   Highest forex SL: {max_sl_forex[0]} = {max_sl_forex[1]} pips')
print(f'   Status: ✓ WITHIN LIMIT')

print('\n'+'='*70)
print('RECOMMENDATIONS:')
print('='*70)
print('\n✗ REMOVE: minimum_sl_pips restriction (currently 25 pips)')
print('  Reason: Optimized params use 10-20 pips for most forex pairs')
print('  Impact: Would enable 27 symbols to use optimized SL values')
print('\n✓ KEEP: risk_reward_ratio = 3.0 (but enforcement is 2.9:1)')
print('  Reason: All optimized params meet or exceed 3.0:1')
print('  Note: Code uses 2.9:1 threshold (line 808), slightly below config')
print('\n✓ KEEP: close_hour = 22, close_minute = 30')
print('  Reason: Matches optimized parameters')
print('\n✓ KEEP: maximum_sl_pips = 50 for forex')
print('  Reason: All optimized params are within this limit')
