import asyncio
import sys
sys.path.append('.')
from app.application import FXAiApplication

async def check_orders():
    app = FXAiApplication(mode='live')
    if await app.initialize_components():
        print('Checking pending orders and positions...')
        
        # Check pending orders
        pending_orders = app.mt5.get_orders()
        print(f'Pending orders: {len(pending_orders)}')
        for order in pending_orders:
            print(f'  Order {order.get("ticket")}: {order.get("symbol")} {order.get("type")} magic={order.get("magic")}')
        
        # Check open positions
        positions = app.mt5.get_positions()
        print(f'Open positions: {len(positions)}')
        for pos in positions:
            print(f'  Position {pos.get("ticket")}: {pos.get("symbol")} magic={pos.get("magic")}')
    else:
        print('Failed to initialize')

if __name__ == "__main__":
    asyncio.run(check_orders())