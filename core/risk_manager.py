"""
FX-Ai Risk Manager - Fixed Version
Proper position size calculation for fixed dollar risk
"""

import logging
import MetaTrader5 as mt5  # type: ignore
import sqlite3
import os
from datetime import datetime, UTC
from typing import Dict, Optional, Tuple

# Set up logger
logger = logging.getLogger('FX-Ai.RiskManager')

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management with proper position sizing for fixed dollar risk"""
    
    def __init__(self, config: dict, db_path: Optional[str] = None, mt5_connector=None):
        """Initialize risk manager"""
        self.config = config
        self.db_path = db_path
        self.mt5_connector = mt5_connector
        
        # Try to read from new trading_rules section, fallback to old locations
        trading_rules = config.get('trading_rules', {})
        trading_config = config.get('trading', {})
        risk_config = config.get('risk_management', {})
        
        # Risk parameters (prefer trading_rules, fallback to legacy)
        risk_limits = trading_rules.get('risk_limits', {})
        self.risk_per_trade = risk_limits.get('risk_per_trade', trading_config.get('risk_per_trade', 50.0))
        self.max_daily_loss = risk_limits.get('max_daily_loss', trading_config.get('max_daily_loss', 200.0))
        
        # Position limits
        position_limits = trading_rules.get('position_limits', {})
        self.max_positions = position_limits.get('max_positions', trading_config.get('max_positions', 3))
        self.max_trades_per_symbol_per_day = position_limits.get('max_trades_per_symbol_per_day', 3)
        
        # Entry rules
        entry_rules = trading_rules.get('entry_rules', {})
        self.max_spread = entry_rules.get('max_spread', trading_config.get('max_spread', 3.0))
        
        # Cooldown rules
        cooldown_rules = trading_rules.get('cooldown_rules', {})
        self.cooldown_minutes = cooldown_rules.get('symbol_cooldown_minutes', risk_config.get('symbol_cooldown_minutes', 5))
        
        # Position tracking
        self.open_positions = {}
        self.daily_loss = 0.0
        
        # Cooldown tracking to prevent immediate reopening after losses
        self.symbol_cooldowns = {}  # symbol -> cooldown_end_time
        
        # Daily trade tracking per symbol (up to 3 trades per symbol per day)
        self.daily_trades_per_symbol = {}  # symbol -> {'date': 'YYYY-MM-DD', 'count': N}
        
        # Daily loss reset tracking
        self.last_reset_date = None  # Track when daily loss was last reset
        
        # Load persistent daily trade counts if database path provided
        if self.db_path:
            self._load_daily_trade_counts()
        
        # Check and reset daily loss if it's a new day
        self._check_and_reset_daily_loss()
        
        logger.info(f"Risk Manager initialized with ${self.risk_per_trade} risk per trade, max_positions={self.max_positions}")
        logger.info(f"Daily trade limit: {self.max_trades_per_symbol_per_day} trade per symbol per day")
        logger.info(f"Max spread: {self.max_spread} pips, Cooldown: {self.cooldown_minutes} minutes")
    
    def _load_daily_trade_counts(self):
        """Load daily trade counts from database for persistence across restarts"""
        if not self.db_path:
            logger.info("No database path provided, using in-memory daily trade counts only")
            return
            
        try:
            if not os.path.exists(self.db_path):
                logger.info("Database not found, starting with empty daily trade counts")
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get today's date using MT5 server time (consistent with trading logic)
            current_date, current_timestamp, success = self._get_mt5_server_date_reliable()
            if not success:
                logger.warning("Cannot get MT5 server time for loading daily counts - using local time as fallback")
                today = datetime.now().strftime('%Y-%m-%d')
            else:
                today = current_date
            
            # Load today's daily trade counts
            cursor.execute('''
                SELECT symbol, trade_date, trade_count
                FROM daily_trade_counts
                WHERE trade_date = ?
            ''', (today,))
            
            loaded_count = 0
            for row in cursor.fetchall():
                symbol, trade_date, count = row
                self.daily_trades_per_symbol[symbol] = {
                    'date': trade_date, 
                    'count': count,
                    'timestamp': current_timestamp if success else None
                }
                loaded_count += 1
            
            conn.close()
            logger.info(f"Loaded {loaded_count} persistent daily trade counts from database for {today}")
            
        except Exception as e:
            logger.error(f"Error loading daily trade counts from database: {e}")
            # Continue with empty counts if database load fails
    
    def _save_daily_trade_count(self, symbol: str, trade_date: str, count: int):
        """Save daily trade count to database"""
        if not self.db_path:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert or replace the daily trade count
            cursor.execute('''
                INSERT OR REPLACE INTO daily_trade_counts
                (symbol, trade_date, trade_count, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (symbol, trade_date, count))
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved daily trade count for {symbol}: {count} on {trade_date}")
            
        except Exception as e:
            logger.error(f"Error saving daily trade count for {symbol}: {e}")

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
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return 0.01
            
            # Use specialized calculation for metals
            if "XAU" in symbol or "XAG" in symbol:
                return self.calculate_position_size_metals(symbol, stop_loss_pips, 
                                                         int(risk_override if risk_override else self.risk_per_trade))
            
            # Use override risk or default
            risk_amount = risk_override if risk_override else self.risk_per_trade
            
            # Get current price for calculations
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
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
                    usdjpy_tick = mt5.symbol_info_tick("USDJPY")  # type: ignore
                    if usdjpy_tick:
                        pip_value_per_lot = (0.01 * 100000) / usdjpy_tick.bid
                    else:
                        pip_value_per_lot = 6.5  # Fallback estimate
                        
                else:
                    # For other crosses like EURGBP
                    # Need GBP/USD rate to convert GBP pips to USD
                    conversion_symbol = f"{quote_currency}USD"
                    conversion_tick = mt5.symbol_info_tick(conversion_symbol)  # type: ignore
                    
                    if conversion_tick:
                        # EURGBP pip value = 10 GBP x GBPUSD rate
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
            logger.info(f"{symbol}: {lot_size:.2f} lots x {stop_loss_pips} pips x ${pip_value_per_lot:.2f} = ${actual_risk:.2f} risk")
            
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
        
        symbol_info = mt5.symbol_info(symbol)  # type: ignore
        tick = mt5.symbol_info_tick(symbol)  # type: ignore
        
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
        """Calculate stop loss price based on pips - Fixed for JPY pairs and metals"""
        try:
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                return 0.0
            
            # Correct pip size for different symbol types
            if "XAU" in symbol:
                pip_size = 0.10  # Gold: 1 pip = 0.10 price units
            elif "XAG" in symbol:
                pip_size = 0.01  # Silver: 1 pip = 0.01 price units (practical)
            elif "JPY" in symbol:
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
        """Calculate take profit price based on pips - Fixed for metals"""
        try:
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                return 0.0
            
            # Correct pip size for different symbol types
            if "XAU" in symbol:
                pip_size = 0.10  # Gold: 1 pip = 0.10 price units
            elif "XAG" in symbol:
                pip_size = 0.01  # Silver: 1 pip = 0.01 price units (practical)
            elif "JPY" in symbol:
                pip_size = 0.01  # JPY pairs: 1 pip = 0.01 price units
            else:
                pip_size = 0.0001  # Standard pairs: 1 pip = 0.0001 price units
            
            # Calculate TP price based on order type
            if order_type.upper() == "BUY":
                tp_price = entry_price + (take_profit_pips * pip_size)
            else:  # SELL
                tp_price = entry_price - (take_profit_pips * pip_size)
            
            # Round to symbol digits
            tp_price = round(tp_price, symbol_info.digits)
            
            return tp_price
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return 0.0

    def calculate_stop_loss_take_profit(self,
                                       symbol: str,
                                       entry_price: float,
                                       direction: str) -> Dict[str, float]:
        """Calculate stop loss and take profit levels for a trade"""
        try:
            # Get default SL/TP pips from config
            sl_pips = self.config.get('trading', {}).get('default_sl_pips', 20)
            tp_pips = self.config.get('trading', {}).get('default_tp_pips', 60)
            
            # Calculate SL and TP prices
            sl_price = self.calculate_stop_loss_price(symbol, direction, entry_price, sl_pips)
            tp_price = self.calculate_take_profit_price(symbol, direction, entry_price, tp_pips)
            
            return {
                'stop_loss': sl_price,
                'take_profit': tp_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating stop loss and take profit: {e}")
            return {'stop_loss': 0.0, 'take_profit': 0.0}
    
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
            positions = mt5.positions_get()  # type: ignore
            if positions and len(positions) >= self.max_positions:
                return False, f"Max positions reached: {len(positions)}/{self.max_positions}"
            
            # Check spread
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
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
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                return self.risk_per_trade

            tick = mt5.symbol_info_tick(symbol)  # type: ignore
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
    
    def _check_and_reset_daily_loss(self):
        """
        Check if it's a new trading day and reset daily loss if needed.
        Uses MT5 server time for consistent daily resets across restarts.
        """
        try:
            current_date, current_timestamp, success = self._get_mt5_server_date_reliable()
            
            if not success:
                logger.warning("Cannot get MT5 server time for daily loss reset check - using local time as fallback")
                from datetime import datetime
                current_date = datetime.now().strftime('%Y-%m-%d')
                current_timestamp = None
            
            # Check if we need to reset (first run or new day)
            if self.last_reset_date is None or self.last_reset_date != current_date:
                # Reset daily loss for new day
                old_loss = self.daily_loss
                self.daily_loss = 0.0
                self.last_reset_date = current_date
                
                if old_loss > 0:
                    logger.info(
                        f"NEW TRADING DAY DETECTED - Daily loss reset\n"
                        f"  Previous day loss: ${old_loss:.2f}\n"
                        f"  New day: {current_date}\n"
                        f"  MT5 Timestamp: {current_timestamp}\n"
                        f"  Daily loss reset to $0.00"
                    )
                else:
                    logger.info(f"Daily loss initialized for new trading day: {current_date}")
                    
        except Exception as e:
            logger.error(f"Error checking daily loss reset: {e}")
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        # Get current date for reset tracking
        current_date, _, success = self._get_mt5_server_date_reliable()
        if not success:
            current_date = datetime.now().strftime('%Y-%m-%d')
        
        self.daily_loss = 0.0
        self.last_reset_date = current_date
        self.open_positions.clear()
        self.daily_trades_per_symbol.clear()  # Reset daily trade counter
        logger.info(f"Daily stats manually reset for {current_date} (including per-symbol trade counters)")
    
    def _get_mt5_server_date_reliable(self) -> tuple:
        """
        Get MT5 server date with NO FALLBACKS to local time
        CRITICAL FIX: This ensures consistent time source for daily trade tracking
        
        Returns:
            tuple: (date_string, timestamp, success)
                   If success=False, DO NOT allow trading
        """
        try:
            # Method 1: Use MT5 connector if available (preferred)
            if self.mt5_connector:
                try:
                    logger.debug(f"RiskManager: Trying MT5 connector for server time")
                    server_time = self.mt5_connector.get_server_time()
                    if server_time:
                        date_str = server_time.strftime('%Y-%m-%d')
                        timestamp = server_time.timestamp()
                        logger.debug(f"RiskManager: MT5 connector time SUCCESS: {date_str} (timestamp: {timestamp})")
                        return (date_str, timestamp, True)
                    else:
                        logger.warning(f"RiskManager: MT5 connector returned None")
                except Exception as e:
                    logger.warning(f"RiskManager: MT5 connector time failed: {e}")
            
            # Method 2: Direct mt5.time_current() - MOST RELIABLE
            logger.debug(f"RiskManager: Trying direct mt5.time_current()")
            if hasattr(mt5, 'time_current'):
                server_timestamp = mt5.time_current()  # type: ignore
                if server_timestamp and server_timestamp > 0:
                    server_time_utc = datetime.fromtimestamp(server_timestamp, tz=UTC)
                    date_str = server_time_utc.strftime('%Y-%m-%d')
                    logger.debug(f"RiskManager: mt5.time_current() SUCCESS = {date_str} (timestamp: {server_timestamp})")
                    return (date_str, server_timestamp, True)
                else:
                    logger.warning(f"RiskManager: mt5.time_current() returned invalid value: {server_timestamp}")
            
            # Method 3: Get tick time from EURUSD (reliable major pair)
            logger.debug(f"RiskManager: Trying EURUSD tick time")
            tick = mt5.symbol_info_tick('EURUSD')  # type: ignore
            if tick and hasattr(tick, 'time') and tick.time > 0:
                server_time_utc = datetime.fromtimestamp(tick.time, tz=UTC)
                date_str = server_time_utc.strftime('%Y-%m-%d')
                logger.warning(f"RiskManager: EURUSD tick time SUCCESS = {date_str} (timestamp: {tick.time})")
                return (date_str, tick.time, True)
            else:
                logger.warning(f"RiskManager: EURUSD tick failed - tick: {tick}, time: {tick.time if tick else None}")
            
            # CRITICAL: If we can't get MT5 time, DO NOT ALLOW TRADING
            logger.error("CRITICAL: Cannot get MT5 server time - BLOCKING all trades for safety")
            return (None, None, False)
            
        except Exception as e:
            logger.error(f"Exception getting MT5 server time: {e} - BLOCKING trades")
            return (None, None, False)
    
    def has_traded_today(self, symbol: str) -> bool:
        """
        Check if symbol has already been traded today (MT5 server time)
        FIXED: Now uses consistent MT5 time source with NO fallbacks to local time
        Only checks trade count limits, not existing positions
        """
        # NOTE: Existing open positions don't prevent new trades
        # The system can manage multiple positions or add to existing ones

        current_date, current_timestamp, success = self._get_mt5_server_date_reliable()
        
        if not success:
            # FAIL-SAFE: If we can't get MT5 time, block trading
            logger.error(f"{symbol}: Cannot verify MT5 server date - BLOCKING trade for safety")
            return True  # Block trading
        
        logger.debug(f"{symbol}: Checking daily trades - MT5 date: {current_date}, timestamp: {current_timestamp}")
        
        # Check if we have a record for this symbol
        if symbol not in self.daily_trades_per_symbol:
            logger.info(f"{symbol}: No trades recorded yet today ({current_date})")
            return False
        
        trade_info = self.daily_trades_per_symbol[symbol]
        recorded_date = trade_info['date']
        recorded_timestamp = trade_info.get('timestamp', 0)
        trade_count = trade_info['count']
        
        # If it's a new day, reset counter
        if recorded_date != current_date:
            logger.info(
                f"{symbol}: New day detected - resetting counter\n"
                f"  Previous: {recorded_date} (timestamp: {recorded_timestamp})\n"
                f"  Current:  {current_date} (timestamp: {current_timestamp})\n"
                f"  Time difference: {current_timestamp - recorded_timestamp:.0f} seconds"
            )
            self.daily_trades_per_symbol[symbol] = {
                'date': current_date, 
                'count': 0, 
                'timestamp': current_timestamp
            }
            return False
        
        # Same day - check count
        if trade_count >= self.max_trades_per_symbol_per_day:
            logger.warning(
                f"{symbol}: Already traded today - BLOCKING\n"
                f"  Date: {current_date}\n"
                f"  Trades today: {trade_count}/{self.max_trades_per_symbol_per_day}\n"
                f"  First trade timestamp: {recorded_timestamp}\n"
                f"  Current timestamp: {current_timestamp}\n"
                f"  Time since first trade: {current_timestamp - recorded_timestamp:.0f} seconds"
            )
            return True
        
        return False

    def record_trade(self, symbol: str):
        """
        Record that a trade was executed for this symbol today
        FIXED: Now uses consistent MT5 time source with NO fallbacks
        """
        current_date, current_timestamp, success = self._get_mt5_server_date_reliable()
        
        if not success:
            logger.error(f"{symbol}: Cannot record trade - MT5 time unavailable")
            return
        
        # Update trade record with both date and timestamp
        if symbol not in self.daily_trades_per_symbol:
            self.daily_trades_per_symbol[symbol] = {
                'date': current_date, 
                'count': 0, 
                'timestamp': current_timestamp
            }
        
        # Increment counter
        self.daily_trades_per_symbol[symbol]['count'] += 1
        self.daily_trades_per_symbol[symbol]['timestamp'] = current_timestamp
        
        logger.warning(
            f"{symbol}: TRADE RECORDED\n"
            f"  MT5 Date: {current_date}\n"
            f"  MT5 Timestamp: {current_timestamp} ({datetime.fromtimestamp(current_timestamp, tz=UTC)})\n"
            f"  Trades Today: {self.daily_trades_per_symbol[symbol]['count']}/{self.max_trades_per_symbol_per_day}\n"
            f"  NO MORE TRADES ALLOWED TODAY FOR {symbol}"
        )
        
        # Save to database for persistence
        self._save_daily_trade_count(symbol, current_date, self.daily_trades_per_symbol[symbol]['count'])

    def can_trade(self, symbol: str) -> tuple[bool, str]:
        """Quick check if trading is allowed
        
        Returns:
            tuple: (can_trade: bool, reason: str)
        """
        
        # Check and reset daily loss if it's a new trading day
        self._check_and_reset_daily_loss()
        
        logger.debug(f"Checking if can trade {symbol}: daily_loss={self.daily_loss}, max_daily_loss={self.max_daily_loss}")
        
        # CHECK #1: Daily trade limit per symbol (up to 3 trades per day)
        if self.has_traded_today(symbol):
            reason = f"{symbol}: Daily trade limit reached - {self.max_trades_per_symbol_per_day} trades per symbol per day maximum"
            logger.warning(reason)
            return False, reason
        
        # CHECK #2: Daily loss limit
        if self.daily_loss >= self.max_daily_loss:
            reason = f"Daily loss limit reached: ${self.daily_loss:.2f}"
            logger.warning(reason)
            return False, reason

        try:
            positions = mt5.positions_get()  # type: ignore
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
                reason = f"Max positions reached: {position_count} >= {self.max_positions}"
                logger.warning(reason)
                return False, reason

            # Check if we already have a position on this symbol (optional)
            prevent_multiple = self.config.get('trading', {}).get('prevent_multiple_positions_per_symbol', True)
            logger.debug(f"Prevent multiple positions per symbol: {prevent_multiple}")
            
            if prevent_multiple:
                symbol_positions = [p for p in positions if hasattr(p, 'symbol') and p.symbol == symbol] if positions else []
                logger.debug(f"Existing positions on {symbol}: {len(symbol_positions)}")
                if symbol_positions:
                    # Allow trading on symbols with existing positions for position management
                    # but still check daily trade limits
                    logger.info(f"{symbol}: Has {len(symbol_positions)} existing position(s) - allowing trade for position management")
                    # Continue to daily trade limit checks below

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
                reason = f"Symbol {symbol} is in cooldown for {remaining_minutes:.1f} more minutes"
                logger.warning(reason)
                return False, reason
            else:
                # Cooldown expired, remove it
                del self.symbol_cooldowns[symbol]

        logger.debug(f"Can trade {symbol}: True")
        return True, "OK"
    
    def calculate_risk_for_lot_size(self, symbol: str, lot_size: float, stop_loss_pips: float) -> float:
        """Calculate the dollar risk for a given lot size and stop loss pips"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
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
                tick = mt5.symbol_info_tick(symbol)  # type: ignore
                if tick:
                    pip_value_per_lot = (pip_size * contract_size) / tick.bid
                else:
                    pip_value_per_lot = pip_size * contract_size
            
            # Convert to account currency if needed
            account_info = mt5.account_info()  # type: ignore
            if account_info:
                account_currency = account_info.currency
                if "JPY" in symbol and symbol.endswith("JPY") and account_currency == "USD":
                    usdjpy_tick = mt5.symbol_info_tick("USDJPY")  # type: ignore
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
        positions = mt5.positions_get()  # type: ignore
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
            positions = mt5.positions_get()  # type: ignore
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

    def check_emergency_stop(self) -> bool:
        """
        Check if emergency stop conditions are met

        Returns:
            bool: True if emergency stop should be triggered
        """
        try:
            # Check and reset daily loss if it's a new day
            self._check_and_reset_daily_loss()

            # Emergency stop conditions
            emergency_config = self.config.get('emergency_stop', {})

            # Check daily loss limit (emergency threshold - higher than normal trading limit)
            emergency_loss_limit = emergency_config.get('emergency_loss_limit', 500.0)  # $500 emergency limit
            if self.daily_loss >= emergency_loss_limit:
                logger.critical(f"Emergency stop triggered: Daily loss ${self.daily_loss:.2f} exceeds emergency limit ${emergency_loss_limit:.2f}")
                return True

            # Check if we've exceeded maximum consecutive losses
            max_consecutive_losses = emergency_config.get('max_consecutive_losses', 10)
            # This would require tracking consecutive losses - for now, just check daily loss

            # Check if MT5 connection is lost (this would be a major emergency)
            if hasattr(self, 'mt5_connector') and self.mt5_connector:
                if not self.mt5_connector.connected:
                    logger.critical("Emergency stop triggered: MT5 connection lost")
                    return True

            # Check for extreme market conditions (this could be expanded)
            # For now, just return False as no emergency conditions met

            return False

        except Exception as e:
            logger.error(f"Error checking emergency stop conditions: {e}")
            # In case of error, be conservative and trigger emergency stop
            return True