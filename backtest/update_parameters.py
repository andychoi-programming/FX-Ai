import json
import os
from pathlib import Path

# Load existing parameters
params_file = Path("models/parameter_optimization/optimal_parameters.json")
with open(params_file, 'r') as f:
    params = json.load(f)

# Enhanced defaults for day-of-week timing
enhanced_defaults = {
    'monday_entry_delay': 10,  # Enter later on Monday
    'friday_early_exit': 17,   # Exit earlier on Friday
    'best_entry_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday'],  # Standard weekdays
    'avoid_exit_days': ['Friday']  # Avoid Friday exits
}

# Update all symbol parameters with enhanced timing
for symbol, timeframes in params.items():
    for timeframe, data in timeframes.items():
        if 'optimal_params' in data:
            # Add new parameters if they don't exist
            for key, value in enhanced_defaults.items():
                if key not in data['optimal_params']:
                    data['optimal_params'][key] = value

# Save updated parameters
with open(params_file, 'w') as f:
    json.dump(params, f, indent=2)

print(f"Updated parameters for {len(params)} symbols with enhanced day-of-week timing")