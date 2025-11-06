"""
Risk Validation Utilities
Validates risk calculations and prevents common errors
"""
import MetaTrader5 as mt5
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class RiskValidator:
    @staticmethod
    def validate_risk_calculation(position, symbol: str) -> Tuple[bool, str, float]:
        """
        Validate risk calculation for a position
        Returns: (is_valid, error_message, calculated_risk_pips)
        """
        try:
            if not position.sl or position.sl <= 0:
                return False, "No stop loss set", 0

            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if not symbol_info:
                return False, "Symbol info unavailable", 0

            point = symbol_info.point

            # Calculate risk using absolute distance (correct method)
            if position.type == mt5.ORDER_TYPE_BUY:
                risk_distance = abs(position.sl - position.price_open)
            else:  # SELL
                risk_distance = abs(position.price_open - position.sl)

            # Convert to pips
            risk_pips = risk_distance / point
            if 'JPY' in symbol:
                risk_pips = risk_pips / 100
            elif 'XAU' in symbol or 'XAG' in symbol:
                risk_pips = risk_pips / 10  # Metals: 1 pip = 0.10 units

            # Validate risk is reasonable - asset-specific limits
            if risk_pips <= 0:
                return False, f"Invalid risk calculation: {risk_pips:.1f} pips", risk_pips

            # Different risk limits for different asset classes
            if 'XAU' in symbol or 'XAG' in symbol:  # Metals
                min_risk = 50   # 50 pips minimum for metals
                max_risk = 5000 # 5000 pips maximum for metals (reasonable for high-volatility assets)
            else:  # Forex pairs
                min_risk = 5    # 5 pips minimum
                max_risk = 500  # 500 pips maximum

            if risk_pips < min_risk:
                return False, f"Risk too tight: {risk_pips:.1f} pips (minimum {min_risk} pips)", risk_pips

            if risk_pips > max_risk:
                return False, f"Risk too wide: {risk_pips:.1f} pips (maximum {max_risk} pips)", risk_pips

            return True, "Risk calculation valid", risk_pips

        except Exception as e:
            return False, f"Risk calculation error: {e}", 0

    @staticmethod
    def validate_sl_positioning(position, symbol: str) -> Tuple[bool, str]:
        """
        Validate that stop loss is positioned correctly relative to entry
        """
        try:
            if not position.sl or position.sl <= 0:
                return False, "No stop loss set"

            # For BUY positions, SL should be below entry
            if position.type == mt5.ORDER_TYPE_BUY:
                if position.sl >= position.price_open:
                    return False, f"BUY position SL ({position.sl:.5f}) should be below entry ({position.price_open:.5f})"
            else:  # SELL positions, SL should be above entry
                if position.sl <= position.price_open:
                    return False, f"SELL position SL ({position.sl:.5f}) should be above entry ({position.price_open:.5f})"

            return True, "Stop loss positioning correct"

        except Exception as e:
            return False, f"SL positioning validation error: {e}"

    @staticmethod
    def check_broker_limits(position, symbol: str) -> Tuple[bool, str]:
        """
        Check if SL/TP meet broker minimum requirements
        """
        try:
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if not symbol_info:
                return False, "Symbol info unavailable"

            min_stop_points = getattr(symbol_info, 'trade_stops_level', 0)
            if min_stop_points <= 0:
                return True, "No broker limits to check"

            min_stop_distance = min_stop_points * symbol_info.point

            # Check SL distance from current price
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
            if tick:
                current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
                sl_distance = abs(current_price - position.sl)

                if sl_distance < min_stop_distance:
                    return False, f"SL too close to current price: {sl_distance:.5f} < {min_stop_distance:.5f}"

            # Check TP distance from current price (if set)
            if position.tp and position.tp > 0:
                tp_distance = abs(current_price - position.tp)
                if tp_distance < min_stop_distance:
                    return False, f"TP too close to current price: {tp_distance:.5f} < {min_stop_distance:.5f}"

            return True, "Broker limits satisfied"

        except Exception as e:
            return False, f"Broker limits check error: {e}"

    @staticmethod
    def comprehensive_position_check(position) -> dict:
        """
        Run all validations on a position
        Returns dict with validation results
        """
        symbol = position.symbol
        results = {
            'symbol': symbol,
            'ticket': position.ticket,
            'validations': {},
            'overall_valid': True,
            'issues': []
        }

        # Risk calculation validation
        valid, message, risk_pips = RiskValidator.validate_risk_calculation(position, symbol)
        results['validations']['risk_calculation'] = {'valid': valid, 'message': message, 'risk_pips': risk_pips}
        if not valid:
            results['overall_valid'] = False
            results['issues'].append(f"Risk: {message}")

        # SL positioning validation
        valid, message = RiskValidator.validate_sl_positioning(position, symbol)
        results['validations']['sl_positioning'] = {'valid': valid, 'message': message}
        if not valid:
            results['overall_valid'] = False
            results['issues'].append(f"SL Position: {message}")

        # Broker limits validation
        valid, message = RiskValidator.check_broker_limits(position, symbol)
        results['validations']['broker_limits'] = {'valid': valid, 'message': message}
        if not valid:
            results['overall_valid'] = False
            results['issues'].append(f"Broker: {message}")

        return results