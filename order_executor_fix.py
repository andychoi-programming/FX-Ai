# FIX FOR MT5 FILLING MODE ISSUE
# Add this to your order request in order_executor.py

request["type_filling"] = 0  # ORDER_FILLING_FOK

# This fixes the 10030 error (invalid filling mode)
# Recommended by MT5 filling mode diagnostic on 2025-11-18 20:20:10
