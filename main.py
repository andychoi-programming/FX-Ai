"""
FX-Ai Trading System - Main Entry Point

This is the simplified main entry point that uses the modular application architecture.
"""

import asyncio
import logging
import sys
import traceback
import os
import argparse
from app.application import FXAiApplication


def main():
    """Main entry point with crash protection"""
    parser = argparse.ArgumentParser(description='FX-Ai Trading System')
    parser.add_argument('--mode', choices=['live', 'backtest'], default='live',
                       help='Trading mode (default: live)')
    parser.add_argument('--config', default='config/config.json',
                       help='Configuration file path')

    args = parser.parse_args()

    try:
        # Create and run application
        app = FXAiApplication(mode=args.mode, config_path=args.config)

        # Run async main
        asyncio.run(app.run())

    except KeyboardInterrupt:
        logging.info("System stopped by user (Ctrl+C)")
        sys.exit(0)

    except Exception as e:
        # Log the full crash details
        logging.critical("=" * 70)
        logging.critical("FATAL CRASH DETECTED")
        logging.critical("=" * 70)
        logging.critical(f"Error: {e}")
        logging.critical("Full traceback:")
        logging.critical(traceback.format_exc())
        logging.critical("=" * 70)

        # Also write to separate crash log
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/crash_log.txt", "a", encoding="utf-8") as f:
                from datetime import datetime
                f.write(f"\n{'=' * 70}\n")
                f.write(f"CRASH at {datetime.now()}\n")
                f.write(f"{'=' * 70}\n")
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
                f.write(f"{'=' * 70}\n\n")
        except BaseException:
            pass

        # Exit with error code so watchdog knows it crashed
        sys.exit(1)


if __name__ == "__main__":
    main()
