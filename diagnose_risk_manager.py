#!/usr/bin/env python3
"""
Diagnose why risk manager is not available during trading
"""
import sys
sys.path.insert(0, '.')

from app.application import FXAiApplication
from utils.config_loader import ConfigLoader

def diagnose():
    print("Diagnosing Risk Manager initialization...\n")

    try:
        config = ConfigLoader().load_config()
        app = FXAiApplication()
        
        # Actually initialize components
        print("Initializing application components...")
        import asyncio
        asyncio.run(app.initialize_components())
        print("Components initialized.\n")

        # Check if components exist
        print("Checking component initialization:")

        # Access the orchestrator
        if hasattr(app, 'trading_orchestrator'):
            orch = app.trading_orchestrator
            print("  Trading Orchestrator: ✓ exists")

            # Check trading engine
            if hasattr(orch, 'trading_engine'):
                engine = orch.trading_engine
                print(f"  Trading Engine in Orchestrator: {'✓' if engine is not None else '✗ is None'}")

                if engine is not None:
                    # Check if engine has risk manager
                    if hasattr(engine, 'risk_manager'):
                        rm = engine.risk_manager
                        print(f"  Risk Manager in Engine: {'✓' if rm is not None else '✗ is None'}")
                    else:
                        print("  Risk Manager in Engine: ✗ attribute doesn't exist")

                    # Check if engine has order executor
                    if hasattr(engine, 'order_executor'):
                        executor = engine.order_executor
                        print("  Order Executor: ✓ exists")

                        # Check if executor has risk manager
                        if hasattr(executor, 'risk_manager'):
                            rm = executor.risk_manager
                            print(f"  Risk Manager in Executor: {'✓' if rm is not None else '✗ is None'}")

                            if rm is not None:
                                # Test the method
                                min_rr = rm._get_symbol_min_rr('XAGUSD')
                                print(f"\n  Test: XAGUSD min R:R = {min_rr}:1 (should be 2.5)")
                                min_rr_eurgbp = rm._get_symbol_min_rr('EURGBP')
                                print(f"  Test: EURGBP min R:R = {min_rr_eurgbp}:1 (should be 2.0)")
                            else:
                                print("\n  ❌ PROBLEM: Risk Manager is None in Order Executor")
                        else:
                            print("  Risk Manager in Executor: ✗ attribute doesn't exist")
                    else:
                        print("  Order Executor: ✗ attribute doesn't exist")
                else:
                    print("  ❌ PROBLEM: Trading Engine is None in Orchestrator")
            else:
                print("  Trading Engine in Orchestrator: ✗ attribute doesn't exist")
        else:
            print("  Trading Orchestrator: ✗ attribute doesn't exist")

        print("\n" + "="*60)
        print("DIAGNOSIS COMPLETE")
        print("="*60)

    except Exception as e:
        print(f"Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose()