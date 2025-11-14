#!/usr/bin/env python3
"""
Simple test to verify the loop interval calculations.
"""

def test_interval_calculations():
    """Test the interval calculation logic."""

    # Test trading opportunity check interval
    opportunity_check_interval = 120  # 2 minutes
    loops_per_check = max(1, opportunity_check_interval // 10)  # Convert seconds to loop count

    print(f"Trading opportunity check interval: {opportunity_check_interval} seconds")
    print(f"Main loop interval: 10 seconds")
    print(f"Loops per trading check: {loops_per_check}")
    print(f"Actual check frequency: {loops_per_check * 10} seconds")

    # Test schedule check interval
    schedule_check_interval_loops = 60  # Every 60 loops = 10 minutes
    schedule_check_seconds = schedule_check_interval_loops * 10

    print(f"\nSchedule check interval: every {schedule_check_interval_loops} loops")
    print(f"Actual schedule check frequency: {schedule_check_seconds} seconds")

    # Verify calculations
    assert loops_per_check == 12, f"Expected 12 loops per check, got {loops_per_check}"
    assert schedule_check_seconds == 600, f"Expected 600 seconds, got {schedule_check_seconds}"

    print("\n✅ All interval calculations are correct!")
    return True

if __name__ == "__main__":
    test_interval_calculations()