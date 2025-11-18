#!/usr/bin/env python3
"""
Fix for FX-Ai OrderManager Missing Method
This file contains utility functions that can be added to the OrderManager class.
These are reference implementations - the actual fixes have been applied to core/order_executor.py
"""

# NOTE: The methods below are reference implementations.
# The actual fixes have been applied directly to the OrderManager class in core/order_executor.py
# This file is kept for reference only.

def calculate_min_stop_distance(symbol: str, logger=None, config=None) -> float:
    """
    Calculate minimum stop distance based on broker requirements and symbol type
    
    Args:
        symbol: Trading symbol
        logger: Optional logger instance
        config: Optional config dictionary
        
    Returns:
        Minimum stop distance in price units
    """
    try:
        import MetaTrader5 as mt5
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            if logger:
                logger.error(f"Cannot get symbol info for {symbol}")
            # Return a default based on symbol type
            if symbol in ['XAUUSD', 'XAGUSD']:
                return 0.50  # 50 pips for metals
            else:
                return 0.0010  # 10 pips for forex
                
        # Get the minimum stop level from broker
        stops_level = symbol_info.stops_level * symbol_info.point
        
        # Define minimum pip requirements by symbol type
        if symbol in ['XAUUSD', 'XAGUSD']:
            # Metals require larger minimum stop
            min_pips = 50
        elif 'JPY' in symbol:
            # JPY pairs
            min_pips = 15
        else:
            # Regular forex pairs
            min_pips = 10
            
        # Convert pips to price distance
        min_distance_from_pips = min_pips * symbol_info.point
        
        # Use the larger of broker requirement or our minimum
        min_stop_distance = max(stops_level, min_distance_from_pips)
        
        # Add a small buffer for safety
        min_stop_distance *= 1.1  # 10% buffer
        
        if logger:
            logger.info(f"{symbol}: Min stop distance: {min_stop_distance:.5f} ({min_stop_distance/symbol_info.point:.1f} pips)")
        
        return min_stop_distance
        
    except Exception as e:
        if logger:
            logger.error(f"Error calculating min stop distance for {symbol}: {e}")
        # Return safe defaults
        if symbol in ['XAUUSD', 'XAGUSD']:
            return 0.50
        elif 'JPY' in symbol:
            return 0.15
        else:
            return 0.0010

def validate_stop_distance(symbol: str, stop_distance: float, logger=None, config=None) -> float:
    """
    Validate and adjust stop distance to meet minimum requirements
    
    Args:
        symbol: Trading symbol
        stop_distance: Proposed stop distance
        logger: Optional logger instance
        config: Optional config dictionary
        
    Returns:
        Valid stop distance (adjusted if necessary)
    """
    try:
        import MetaTrader5 as mt5
        
        # Get minimum required distance
        min_distance = calculate_min_stop_distance(symbol, logger, config)
        
        # Ensure stop distance meets minimum
        if stop_distance < min_distance:
            if logger:
                logger.warning(f"{symbol}: Stop distance {stop_distance:.5f} below minimum {min_distance:.5f}, adjusting")
            return min_distance
            
        # Check maximum allowed distance from config
        max_sl_pips = 50  # Default
        if config:
            max_sl_pips = config.get('trading_rules', {}).get('stop_loss_rules', {}).get('max_sl_pips', 50)
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            max_distance = max_sl_pips * symbol_info.point
            if stop_distance > max_distance:
                if logger:
                    logger.warning(f"{symbol}: Stop distance {stop_distance:.5f} above maximum {max_distance:.5f}, adjusting")
                return max_distance
                
        return stop_distance
        
    except Exception as e:
        if logger:
            logger.error(f"Error validating stop distance for {symbol}: {e}")
        return stop_distance  # Return original if validation fails

def calculate_position_size_with_min_stop(symbol: str, stop_distance: float, risk_amount: float, logger=None, config=None) -> float:
    """
    Calculate position size considering minimum stop distance requirements
    
    Args:
        symbol: Trading symbol
        stop_distance: Stop loss distance in price units
        risk_amount: Risk amount in account currency (e.g., $50)
        logger: Optional logger instance
        config: Optional config dictionary
        
    Returns:
        Position size in lots
    """
    try:
        import MetaTrader5 as mt5
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            if logger:
                logger.error(f"Cannot get symbol info for {symbol}")
            return 0.01  # Minimum lot size
            
        # Ensure stop distance meets minimum requirements
        validated_stop_distance = validate_stop_distance(symbol, stop_distance, logger, config)
        
        # Calculate pip value per lot
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        
        if tick_size == 0:
            if logger:
                logger.error(f"Invalid tick size for {symbol}")
            return 0.01
            
        # Calculate pip value
        pip_value = (tick_value * symbol_info.point) / tick_size
        
        # Calculate position size
        if validated_stop_distance > 0 and pip_value > 0:
            # Calculate lots needed for the risk amount
            stop_in_pips = validated_stop_distance / symbol_info.point
            position_size = risk_amount / (stop_in_pips * pip_value)
            
            # Round to valid lot step
            lot_step = symbol_info.volume_step
            position_size = round(position_size / lot_step) * lot_step
            
            # Apply min/max constraints
            position_size = max(symbol_info.volume_min, min(position_size, symbol_info.volume_max))
            
            if logger:
                logger.info(f"{symbol}: Position size: {position_size:.2f} lots for ${risk_amount:.2f} risk")
            
            return position_size
        else:
            if logger:
                logger.error(f"Invalid calculation parameters for {symbol}")
            return symbol_info.volume_min
            
    except Exception as e:
        if logger:
            logger.error(f"Error calculating position size for {symbol}: {e}")
        return 0.01  # Safe minimum

# Additional helper function that might be needed
def get_minimum_stops(symbol: str, logger=None, config=None) -> dict:
    """
    Get minimum stop and target distances for a symbol
    
    Args:
        symbol: Trading symbol
        logger: Optional logger instance
        config: Optional config dictionary
        
    Returns:
        Dictionary with 'stop_distance' and 'target_distance' keys
    """
    try:
        import MetaTrader5 as mt5
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            # Return defaults
            if symbol in ['XAUUSD', 'XAGUSD']:
                return {'stop_distance': 0.50, 'target_distance': 1.00}
            else:
                return {'stop_distance': 0.0010, 'target_distance': 0.0020}
                
        # Get broker's minimum stop level
        min_stop_points = symbol_info.stops_level
        if min_stop_points == 0:
            min_stop_points = 10  # Default if broker doesn't specify
            
        min_stop_distance = min_stop_points * symbol_info.point
        
        # Ensure minimum risk/reward ratio
        min_rr_ratio = 2.0  # Default
        if config:
            min_rr_ratio = config.get('trading_rules', {}).get('entry_rules', {}).get('min_risk_reward_ratio', 2.0)
        min_target_distance = min_stop_distance * min_rr_ratio
        
        return {
            'stop_distance': min_stop_distance,
            'target_distance': min_target_distance
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting minimum stops for {symbol}: {e}")
        return {'stop_distance': 0.0010, 'target_distance': 0.0020}
