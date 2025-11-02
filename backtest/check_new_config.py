from sklearn.model_selection import ParameterGrid

params = {
    'sl_pips': [15, 20, 25, 30, 35],
    'tp_pips': [60, 75, 90, 105, 120],
    'breakeven_trigger': [15, 20, 25],
    'trailing_activation': [20, 25],
    'trailing_distance': [10, 15],
    'entry_hour_start': [6, 8],
    'entry_hour_end': [16, 18],
    'max_holding_hours': [6, 12],
    'monday_entry_delay': [10, 12],
    'friday_early_exit': [19, 20],  # Assuming you meant PM hours, reduced to 2 options
    'hard_close_hour': [22],
    'best_entry_days': [
        ['Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
    ]
}

all_combos = list(ParameterGrid(params))
valid = [p for p in all_combos if p['tp_pips']/p['sl_pips'] >= 3.0]

print('NEW CONFIGURATION: Monday [10,12] + Friday [10,13,16]')
print('='*70)
print(f'Total combinations: {len(all_combos):,}')
print(f'Valid (R/R >= 1:3): {len(valid):,}')
print(f'Filtered out: {len(all_combos)-len(valid):,}')
print()

mins = len(valid) * 0.25 / 60
hrs = mins / 60
total_hrs = hrs * 30

print(f'Per symbol: {len(valid):,} combos × 0.25s = {mins:.1f} min ({hrs:.2f} hr)')
print(f'Total (30 symbols): {total_hrs:.1f} hours')
print(f'Utilization: {total_hrs/24*100:.1f}% of 24 hours')
print()

if total_hrs <= 24:
    print(f'✓ FITS IN 24-HOUR WINDOW')
    print(f'✓ Safety buffer: {24-total_hrs:.1f} hours')
else:
    print(f'⚠ EXCEEDS 24 hours by {total_hrs-24:.1f} hours')
    print(f'  Total time needed: {total_hrs:.1f} hours')
