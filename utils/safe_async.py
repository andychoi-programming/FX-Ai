"""
Safe async utilities for FX-Ai
Prevents async/await errors
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

async def safe_await(coro_or_func, *args, **kwargs):
    """
    Safely await a coroutine or function
    Handles both async and sync functions
    """
    try:
        if asyncio.iscoroutine(coro_or_func):
            # It's already a coroutine, just await it
            return await coro_or_func
        elif asyncio.iscoroutinefunction(coro_or_func):
            # It's an async function, call and await it
            return await coro_or_func(*args, **kwargs)
        elif callable(coro_or_func):
            # It's a regular function, run it in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, coro_or_func, *args, **kwargs)
        else:
            # Not callable, just return None
            logger.warning(f"safe_await called with non-callable: {type(coro_or_func)}")
            return None
    except Exception as e:
        logger.error(f"Error in safe_await: {e}")
        return None


async def safe_close_positions(trading_engine):
    """Safely close all positions regardless of method type"""
    try:
        if not trading_engine:
            logger.warning("No trading engine provided")
            return

        if hasattr(trading_engine, 'close_all_positions'):
            method = trading_engine.close_all_positions

            if asyncio.iscoroutinefunction(method):
                await method()
            else:
                # Run sync method in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, method)

            logger.info("Positions closed successfully")
        else:
            logger.warning("Trading engine has no close_all_positions method")

    except Exception as e:
        logger.error(f"Error closing positions: {e}")
