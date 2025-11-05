"""
FX-Ai Risk Manager - Fixed Version
Proper position size calculation for fixed dollar risk
"""

import logging
import MetaTrader5 as mt5
from typing import Dict, Optional, Tuple
import math

# Set up logger
logger = logging.getLogger('FX-Ai.RiskManager')

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management with proper position sizing for fixed dollar risk"""
    
    def __init__(self, config: dict):
        """Initialize risk manager"""
        self.config = config
        
        # Risk parameters
        self.risk_per_trade = config.get('trading', {}).get('risk_per_trade', 50.0)  # Dollar risk
        self.max_positions = config.get('trading', {}).get('max_positions', 3)
        self.max_daily_loss = config.get('trading', {}).get('max_daily_loss', 200.0)
        self.max_spread = config.get('trading', {}).get('max_spread', 3.0)
        
        # Position tracking
        self.open_positions = {}
        self.daily_loss = 0.0
        
        # Cooldown tracking to prevent immediate reopening after losses
        self.symbol_cooldowns = {}  # symbol -> cooldown_end_time
        self.cooldown_minutes = config.get('risk_management', {}).get('symbol_cooldown_minutes', 5)  # 5 minute cooldown
        
        # Daily trade tracking per symbol (ONE TRADE PER SYMBOL PER DAY)
        self.daily_trades_per_symbol = {}  # symbol -> {'date': 'YYYY-MM-DD', 'count': N}
        self.max_trades_per_symbol_per_day = 1  # Hard limit: 1 trade per symbol per day
        
        logger.info(f"Risk Manager initialized with ${self.risk_per_trade} risk per trade, max_positions={self.max_positions}")
        logger.info(f"Daily trade limit: {self.max_trades_per_symbol_per_day} trade per symbol per day")
    
    def calculate_position_size(self, 
                               symbol: str, 
                               stop_loss_pips: float = 20,
                               risk_override: Optional[float] = None) -> float:
        """
        Calculate position size based on fixed dollar risk
        
        Args:
            symbol: Trading symbol
            stop_loss_pips: Stop loss distance in pips
            risk_override: Override risk amount if provided
            
        Returns:
            Calculated lot size
        """
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return 0.01
            
            # Use specialized calculation for metals
            if "XAU" in symbol or "XAG" in symbol:
                return self.calculate_position_size_metals(symbol, stop_loss_pips, 
                                                         risk_override if risk_override else self.risk_per_trade)
            
            # Use override risk or default
            risk_amount = risk_override if risk_override else self.risk_per_trade
            
            # Get current price for calculations
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Cannot get tick for {symbol}")
                return 0.01
            
            # FIXED CALCULATION METHOD FOR CROSS PAIRS
            # Calculate pip value correctly for all pair types
            
            if symbol.endswith("USD"):
                # Direct pairs (EURUSD, GBPUSD, etc.)
                pip_value_per_lot = 10.0  # $10 per pip for 1 lot
                
            elif symbol.startswith("USD"):
                # Inverse pairs (USDCHF, USDJPY, etc.)
                if "JPY" in symbol:
                    pip_value_per_lot = (0.01 * 100000) / tick.bid
                else:
                    pip_value_per_lot = (0.0001 * 100000) / tick.bid
                    
            else:
                # CROSS PAIRS (EURGBP, EURJPY, GBPJPY, etc.)
                # Need to convert through the quote currency
                quote_currency = symbol[-3:]  # Last 3 chars (GBP, JPY, etc.)
                
                # Get conversion rate
                if quote_currency == "JPY":
                    # For XXX/JPY crosses
                    usdjpy_tick = mt5.symbol_info_tick("USDJPY")
                    if usdjpy_tick:
                        pip_value_per_lot = (0.01 * 100000) / usdjpy_tick.bid
                    else:
                        pip_value_per_lot = 6.5  # Fallback estimate
                        
                else:
                    # For other crosses like EURGBP
                    # Need GBP/USD rate to convert GBP pips to USD
                    conversion_symbol = f"{quote_currency}USD"
                    conversion_tick = mt5.symbol_info_tick(conversion_symbol)
                    
                    if conversion_tick:
                        # EURGBP pip value = 10 GBP × GBPUSD rate
                        pip_value_per_lot = 10.0 * conversion_tick.bid
                    else:
                        # Fallback if can't get conversion
                        logger.warning(f"Can't get {conversion_symbol} rate, using estimate")
                        pip_value_per_lot = 13.0  # Rough estimate
            
            # Calculate lot size
            if stop_loss_pips <= 0:
                logger.warning(f"Invalid stop loss pips: {stop_loss_pips}, using default 20")
                stop_loss_pips = 20
                
            lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
            
            # Round to lot step
            lot_step = symbol_info.volume_step
            lot_size = round(lot_size / lot_step) * lot_step
            
            # Apply broker limits
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            
            if lot_size < min_lot:
                logger.warning(f"Calculated lot size {lot_size:.4f} below minimum {min_lot}")
                lot_size = min_lot
            elif lot_size > max_lot:
                logger.warning(f"Calculated lot size {lot_size:.4f} above maximum {max_lot}")
                lot_size = max_lot
            
            # Verify the calculation
            actual_risk = lot_size * stop_loss_pips * pip_value_per_lot
            logger.info(f"{symbol}: {lot_size:.2f} lots × {stop_loss_pips} pips × ${pip_value_per_lot:.2f} = ${actual_risk:.2f} risk")
            
            # Safety check
            if actual_risk > risk_amount * 1.1:  # Allow 10% tolerance
                # Reduce lot size
                lot_size = (risk_amount * 0.95) / (stop_loss_pips * pip_value_per_lot)
                lot_size = round(lot_size / lot_step) * lot_step
                lot_size = max(min_lot, lot_size)
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01
    
    def calculate_position_size_metals(self, symbol, stop_loss_pips, risk_amount=50):
        """Special calculation for Gold and Silver"""
        
        symbol_info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        
        if "XAU" in symbol:  # Gold
            # For Gold: 1 pip = $0.10 movement = $10 per lot
            # So 1 pip = $10 for 1 lot, $1 for 0.1 lot, $0.10 for 0.01 lot
            
            # If stop loss is 500 pips (typical for gold)
            # Risk per 0.01 lot = 500 * $0.10 = $50
            # So for $50 risk with 500 pip SL = 0.01 lots
            
            pip_value_per_001_lot = 0.10  # $0.10 per pip for 0.01 lot
            lot_size = risk_amount / (stop_loss_pips * pip_value_per_001_lot * 100)
            
        elif "XAG" in symbol:  # Silver  
            # For Silver: 1 pip = $0.001 movement = $50 per lot
            # 1 pip = $50 for 1 lot, $5 for 0.1 lot, $0.50 for 0.01 lot
            
            # If stop loss is 500 pips
            # Risk per 0.01 lot = 500 * $0.50 = $250 (too much!)
            # Need 0.002 lots for $50 risk
            
            pip_value_per_001_lot = 0.50  # $0.50 per pip for 0.01 lot
            lot_size = risk_amount / (stop_loss_pips * pip_value_per_001_lot * 100)
            
            # Check if we can trade with broker's minimum
            min_lot = symbol_info.volume_min
            min_risk = stop_loss_pips * pip_value_per_001_lot * min_lot * 100
            
            if min_risk > risk_amount:
                logger.warning(f"Cannot trade {symbol}: Minimum risk ${min_risk:.2f} > ${risk_amount}")
                return 0  # Cannot trade within risk limits
        
        # Round to lot step
        lot_step = symbol_info.volume_step
        lot_size = round(lot_size / lot_step) * lot_step
        
        # Apply limits
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        return lot_size
    
    def calculate_stop_loss_price(self, 
                                 symbol: str, 
                                 order_type: str,
                                 entry_price: float,
                                 stop_loss_pips: float = 20) -> float:
        """Calculate stop loss price based on pips - Fixed for JPY pairs"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.0
            
            # Correct pip size for JPY pairs
            if "JPY" in symbol:
                pip_size = 0.01  # JPY pairs: 1 pip = 0.01 price units
            else:
                pip_size = 0.0001  # Standard pairs: 1 pip = 0.0001 price units
            
            # Calculate distance in price units
            distance = stop_loss_pips * pip_size
            
            # Calculate SL price
            if order_type.upper() == "BUY":
                sl_price = entry_price - distance
            else:  # SELL
                sl_price = entry_price + distance
            
            return sl_price
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return 0.0
    
    def calculate_take_profit_price(self,
                                   symbol: str,
                                   order_type: str,
                                   entry_price: float,
                                   take_profit_pips: float = 40) -> float:
        """Calculate take profit price based on pips"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.0
            
            point = symbol_info.point
            digits = symbol_info.digits
            
            # Determine pip size
            if digits == 3 or digits == 5:
                pip_size = point * 10
            else:
                pip_size = point
            
            # Calculate TP price based on order type
            if order_type.upper() == "BUY":
                tp_price = entry_price + (take_profit_pips * pip_size)
            else:  # SELL
                tp_price = entry_price - (take_profit_pips * pip_size)
            
            # Round to symbol digits
            tp_price = round(tp_price, digits)
            
            return tp_price
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return 0.0
    
    def validate_trade_risk(self, 
                           symbol: str, 
                           lot_size: float,
                           stop_loss_pips: float) -> Tuple[bool, str]:
        """
        Validate if trade meets risk parameters
        
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check daily loss limit
            if self.daily_loss >= self.max_daily_loss:
                return False, f"Daily loss limit reached: ${self.daily_loss:.2f}"
            
            # Check max positions
            positions = mt5.positions_get()
            if positions and len(positions) >= self.max_positions:
                return False, f"Max positions reached: {len(positions)}/{self.max_positions}"
            
            # Check spread
            tick = mt5.symbol_info_tick(symbol)
            symbol_info = mt5.symbol_info(symbol)
            if tick and symbol_info:
                spread_points = tick.ask - tick.bid
                spread_pips = spread_points / symbol_info.point
                
                if symbol_info.digits == 3 or symbol_info.digits == 5:
                    spread_pips = spread_pips / 10
                
                if spread_pips > self.max_spread:
                    return False, f"Spread too high: {spread_pips:.1f} pips"
            
            # Estimate risk for this trade
            estimated_risk = self.estimate_trade_risk(symbol, lot_size, stop_loss_pips)
            
            if estimated_risk > self.risk_per_trade * 1.2:  # 20% tolerance
                return False, f"Risk too high: ${estimated_risk:.2f} > ${self.risk_per_trade * 1.2:.2f}"
            
            return True, "Trade validated"
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"
    
    def estimate_trade_risk(self, symbol: str, lot_size: float, stop_loss_pips: float) -> float:
        """Estimate the risk amount for a trade"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return self.risk_per_trade

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return self.risk_per_trade

            # Calculate pip value
            contract_size = symbol_info.trade_contract_size
            point = symbol_info.point
            digits = symbol_info.digits

            # Determine pip size based on symbol type
            if "XAU" in symbol or "GOLD" in symbol or "XAG" in symbol:
                pip_size = point * 10  # Metals: 1 pip = 10 points
            elif digits == 3 or digits == 5:
                pip_size = point * 10
            else:
                pip_size = point

            # Calculate pip value per lot
            if "JPY" in symbol and symbol.endswith("JPY"):
                pip_value = (pip_size * contract_size) / tick.bid
            elif symbol.startswith("USD"):
                pip_value = pip_size * contract_size
            elif symbol.endswith("USD"):
                pip_value = pip_size * contract_size
            elif "XAU" in symbol or "GOLD" in symbol:
                pip_value = pip_size * contract_size  # For XAUUSD: 0.1 * 100 = $10 per pip per lot
            elif "XAG" in symbol:
                pip_value = pip_size * contract_size  # For XAGUSD: similar calculation
            else:
                # Get current price for cross pairs
                pip_value = (pip_size * contract_size) / tick.bid

            # Calculate risk
            risk_amount = lot_size * stop_loss_pips * pip_value

            return risk_amount

        except Exception as e:
            logger.error(f"Error estimating risk: {e}")
            return self.risk_per_trade
    
    def update_daily_loss(self, profit: float):
        """Update daily loss tracker"""
        if profit < 0:
            self.daily_loss += abs(profit)
            logger.info(f"Daily loss updated: ${self.daily_loss:.2f}")
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_loss = 0.0
        self.open_positions.clear()
        self.daily_trades_per_symbol.clear()  # Reset daily trade counter
        logger.info("Daily stats reset (including per-symbol trade counters)")
    
    def has_traded_today(self, symbol: str) -> bool:
        """Check if symbol has already been traded today (MT5 server time)"""
        from datetime import datetime
        
        # Get MT5 server time
        server_time = mt5.symbol_info_tick(symbol)
        if server_time:
            current_date = datetime.fromtimestamp(server_time.time).strftime('%Y-%m-%d')
        else:
            # Fallback to local time if MT5 not available
            current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check if symbol has trade record for today
        if symbol in self.daily_trades_per_symbol:
            trade_info = self.daily_trades_per_symbol[symbol]
            
            # If it's a new day, reset the counter
            if trade_info['date'] != current_date:
                self.daily_trades_per_symbol[symbol] = {'date': current_date, 'count': 0}
                return False
            
            # Check if already traded today
            if trade_info['count'] >= self.max_trades_per_symbol_per_day:
                logger.info(f"{symbol}: Already traded today ({current_date}), skipping")
                return True
        
        return False
    
    def record_trade(self, symbol: str):
        """Record that a trade was executed for this symbol today"""
        from datetime import datetime
        
        # Get MT5 server time
        server_time = mt5.symbol_info_tick(symbol)
        if server_time:
            current_date = datetime.fromtimestamp(server_time.time).strftime('%Y-%m-%d')
        else:
            current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Update trade counter
        if symbol not in self.daily_trades_per_symbol:
            self.daily_trades_per_symbol[symbol] = {'date': current_date, 'count': 0}
        
        self.daily_trades_per_symbol[symbol]['count'] += 1
        logger.info(f"{symbol}: Trade recorded for {current_date} (count: {self.daily_trades_per_symbol[symbol]['count']})")
    
    def can_trade(self, symbol: str) -> bool:
        """Quick check if trading is allowed"""
        from datetime import datetime
        
        logger.debug(f"Checking if can trade {symbol}: daily_loss={self.daily_loss}, max_daily_loss={self.max_daily_loss}")
        
        # CHECK #1: Daily trade limit per symbol (ONE TRADE PER DAY)
        if self.has_traded_today(symbol):
            logger.warning(f"{symbol}: Already traded today - ONE trade per symbol per day limit")
            return False
        
        # CHECK #2: Daily loss limit
        if self.daily_loss >= self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: ${self.daily_loss:.2f}")
            return False

        try:
            positions = mt5.positions_get()
            position_count = len(positions) if positions else 0
            logger.debug(f"Positions query result type: {type(positions)}, value: {positions}")
            logger.debug(f"Max positions: {self.max_positions}, current positions: {position_count}")
            
            if positions is None:
                logger.warning(f"MT5 positions_get() returned None for {symbol}, allowing trade but this indicates connection issues")
                # If we can't get positions, allow trading but log the issue
                positions = []
            elif not isinstance(positions, (list, tuple)):
                logger.warning(f"MT5 positions_get() returned unexpected type {type(positions)} for {symbol}, allowing trade")
                positions = []
            
            position_count = len(positions) if positions else 0
            logger.debug(f"Current position count: {position_count}, max_positions: {self.max_positions}")
            
            if position_count >= self.max_positions:
                logger.warning(f"Max positions reached: {position_count} >= {self.max_positions}")
                return False

            # Check if we already have a position on this symbol (optional)
            prevent_multiple = self.config.get('trading', {}).get('prevent_multiple_positions_per_symbol', True)
            logger.debug(f"Prevent multiple positions per symbol: {prevent_multiple}")
            
            if prevent_multiple:
                symbol_positions = [p for p in positions if hasattr(p, 'symbol') and p.symbol == symbol] if positions else []
                logger.debug(f"Existing positions on {symbol}: {len(symbol_positions)}")
                if symbol_positions:
                    logger.warning(f"Already have {len(symbol_positions)} position(s) on {symbol}, skipping")
                    return False

        except Exception as e:
            logger.error(f"Error checking positions for {symbol}: {e}, allowing trade")
            # If there's an error getting positions, allow trading
            pass

        # Check cooldown period after losses
        current_time = datetime.now()
        if symbol in self.symbol_cooldowns:
            cooldown_end = self.symbol_cooldowns[symbol]
            if current_time < cooldown_end:
                remaining_minutes = (cooldown_end - current_time).total_seconds() / 60
                logger.warning(f"Symbol {symbol} is in cooldown for {remaining_minutes:.1f} more minutes")
                return False
            else:
                # Cooldown expired, remove it
                del self.symbol_cooldowns[symbol]

        logger.debug(f"Can trade {symbol}: True")
        return True
    
    def calculate_risk_for_lot_size(self, symbol: str, lot_size: float, stop_loss_pips: float) -> float:
        """Calculate the dollar risk for a given lot size and stop loss pips"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return 0.0
            
            # Get pip value calculation (reuse logic from calculate_position_size)
            contract_size = symbol_info.trade_contract_size
            point = symbol_info.point
            digits = symbol_info.digits
            
            # Determine pip size
            if "XAU" in symbol or "GOLD" in symbol or "XAG" in symbol:
                pip_size = point * 10  # Metals: 1 pip = 10 points
            elif digits == 3 or digits == 5:
                pip_size = point * 10
            else:
                pip_size = point
            
            # Calculate pip value per lot
            if "JPY" in symbol and symbol.endswith("JPY"):
                pip_value_per_lot = pip_size * contract_size
            elif symbol.startswith("USD"):
                pip_value_per_lot = pip_size * contract_size
            elif symbol.endswith("USD"):
                pip_value_per_lot = pip_size * contract_size
            elif "XAU" in symbol or "GOLD" in symbol:
                pip_value_per_lot = pip_size * contract_size  # For XAUUSD: 0.1 * 100 = $10 per pip per lot
            elif "XAG" in symbol:
                pip_value_per_lot = pip_size * contract_size  # For XAGUSD: similar calculation
            else:
                # Get current price for cross pairs
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    pip_value_per_lot = (pip_size * contract_size) / tick.bid
                else:
                    pip_value_per_lot = pip_size * contract_size
            
            # Convert to account currency if needed
            account_info = mt5.account_info()
            if account_info:
                account_currency = account_info.currency
                if "JPY" in symbol and symbol.endswith("JPY") and account_currency == "USD":
                    usdjpy_tick = mt5.symbol_info_tick("USDJPY")
                    if usdjpy_tick:
                        pip_value_per_lot = pip_value_per_lot / usdjpy_tick.bid
            
            # Calculate risk: lot_size * stop_loss_pips * pip_value_per_lot
            risk_amount = lot_size * stop_loss_pips * pip_value_per_lot
            
            return risk_amount
            
        except Exception as e:
            logger.error(f"Error calculating risk for lot size: {e}")
            return 0.0
    
    def record_trade_result(self, symbol: str, profit: float):
        """Record trade result and set cooldowns if needed"""
        from datetime import datetime, timedelta
        
        # Update daily loss
        if profit < 0:
            self.daily_loss += abs(profit)
            
            # Set cooldown for losing symbol
            cooldown_end = datetime.now() + timedelta(minutes=self.cooldown_minutes)
            self.symbol_cooldowns[symbol] = cooldown_end
            logger.info(f"Set {self.cooldown_minutes} minute cooldown for {symbol} after ${abs(profit):.2f} loss")
    
    def get_risk_summary(self) -> Dict:
        """Get current risk status summary"""
        positions = mt5.positions_get()
        current_positions = len(positions) if positions else 0
        
        total_exposure = 0.0
        if positions:
            for pos in positions:
                total_exposure += pos.volume * pos.price_current
        
        return {
            'daily_loss': self.daily_loss,
            'max_daily_loss': self.max_daily_loss,
            'current_positions': len(positions) if positions else 0,
            'max_positions': self.max_positions,
            'total_exposure': total_exposure,
            'risk_per_trade': self.risk_per_trade,
            'can_trade': self.can_trade("")
        }
    
    def check_trade_risk(self, symbol: str, ml_signal: Dict, current_capital: float) -> Dict:
        """
        Check if a trade meets risk management criteria for backtesting
        
        Args:
            symbol: Trading symbol
            ml_signal: ML prediction signal dict with 'direction' and 'confidence'
            current_capital: Current account capital
            
        Returns:
            Dict with 'approved' (bool) and 'max_volume' (float)
        """
        try:
            # Check if we can trade this symbol
            if not self.can_trade(symbol):
                return {'approved': False, 'max_volume': 0.0}
            
            # Check daily loss limit
            if self.daily_loss >= self.max_daily_loss:
                return {'approved': False, 'max_volume': 0.0}
            
            # Check max positions
            positions = mt5.positions_get()
            if positions and len(positions) >= self.max_positions:
                return {'approved': False, 'max_volume': 0.0}
            
            # Calculate maximum position size based on risk per trade
            stop_loss_pips = 20  # Conservative 20 pip stop loss
            max_volume = self.calculate_position_size(symbol, stop_loss_pips, self.risk_per_trade)
            
            # Limit volume based on current capital (max 2% of capital per trade)
            capital_limit = current_capital * 0.02
            max_volume = min(max_volume, capital_limit / (stop_loss_pips * 10.0))  # Simplified pip value
            
            # Ensure minimum volume
            max_volume = max(max_volume, 0.01)
            
            return {
                'approved': True,
                'max_volume': round(max_volume, 2)
            }
            
        except Exception as e:
            logger.error(f"Error in check_trade_risk: {e}")
            return {'approved': False, 'max_volume': 0.0}