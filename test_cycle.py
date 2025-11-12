"""
Single cycle test of FX-Ai Trading System
"""

import asyncio
import logging
import sys
import traceback
from app.application import FXAiApplication


async def test_single_cycle():
    """Test single trading cycle"""
    try:
        # Create and initialize application
        app = FXAiApplication()

        # Initialize components
        if not await app.initialize_components():
            print("Failed to initialize components")
            return

        # Get the trading orchestrator
        orchestrator = app.trading_orchestrator

        # Run one cycle of opportunity checking
        print("Running single trading cycle...")
        await orchestrator._check_trading_opportunities()
        print("Cycle complete!")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(test_single_cycle())