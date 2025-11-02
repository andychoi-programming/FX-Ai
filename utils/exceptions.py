"""
Custom Exception Classes for FX-Ai Trading System
Provides specific exception types for better error handling and debugging
"""


class FXAiException(Exception):
    """Base exception for all FX-Ai errors"""
    pass


# MT5 Connection Errors
class MT5ConnectionError(FXAiException):
    """Raised when MT5 connection fails or is lost"""
    pass


class MT5InitializationError(MT5ConnectionError):
    """Raised when MT5 fails to initialize"""
    pass


class MT5LoginError(MT5ConnectionError):
    """Raised when MT5 login fails"""
    pass


# Trading Errors
class OrderRejectedError(FXAiException):
    """Raised when broker rejects an order"""
    pass


class InsufficientFundsError(FXAiException):
    """Raised when account has insufficient funds for trade"""
    pass


class InvalidOrderParametersError(FXAiException):
    """Raised when order parameters are invalid (SL/TP/volume)"""
    pass


class MaxPositionsReachedError(FXAiException):
    """Raised when maximum number of positions is reached"""
    pass


class SpreadTooHighError(FXAiException):
    """Raised when spread exceeds configured maximum"""
    pass


# Risk Management Errors
class RiskLimitExceededError(FXAiException):
    """Raised when trade would exceed risk limits"""
    pass


class DailyLossLimitError(FXAiException):
    """Raised when daily loss limit is reached"""
    pass


class CorrelationLimitError(FXAiException):
    """Raised when new position would violate correlation limits"""
    pass


# Data Errors
class MarketDataError(FXAiException):
    """Raised when market data is unavailable or invalid"""
    pass


class InsufficientDataError(FXAiException):
    """Raised when not enough historical data for analysis"""
    pass


# Backwards-compatible generic data error expected by some callers
class DataError(MarketDataError):
    """Generic data error (alias for MarketDataError)"""
    pass


# ML Model Errors
class ModelLoadError(FXAiException):
    """Raised when ML model fails to load"""
    pass


class PredictionError(FXAiException):
    """Raised when ML prediction fails"""
    pass


class ModelNotTrainedError(FXAiException):
    """Raised when attempting to use untrained model"""
    pass


# Configuration Errors
class ConfigurationError(FXAiException):
    """Raised when configuration is invalid or missing"""
    pass


class MissingParametersError(ConfigurationError):
    """Raised when required parameters are missing"""
    pass


class InvalidConfigValueError(ConfigurationError):
    """Raised when configuration value is invalid"""
    pass


# Generic validation error for config/data validation failures
class ValidationError(ConfigurationError):
    """Raised when data or configuration validation fails"""
    pass


# Market Condition Errors
class MarketClosedError(FXAiException):
    """Raised when attempting to trade while market is closed"""
    pass


class HighVolatilityError(FXAiException):
    """Raised when volatility exceeds safe trading levels"""
    pass


class NewsEventBlockedError(FXAiException):
    """Raised when trading is blocked due to news event"""
    pass


# System Errors
class SystemHealthError(FXAiException):
    """Raised when system health check fails"""
    pass


class ResourceExhaustedError(FXAiException):
    """Raised when system resources (CPU/memory) are exhausted"""
    pass


def handle_mt5_error(error_code: int, operation: str = "") -> Exception:
    """
    Convert MT5 error codes to specific exceptions
    
    Args:
        error_code: MT5 error code
        operation: Description of the operation that failed
        
    Returns:
        Appropriate exception for the error code
    """
    error_messages = {
        10004: "Requote",
        10006: "Request rejected",
        10007: "Request canceled",
        10008: "Order placed",
        10009: "Request done",
        10010: "Request partially filled",
        10011: "Request processing error",
        10012: "Invalid request",
        10013: "Invalid volume",
        10014: "Invalid price",
        10015: "Invalid stops",
        10016: "Trade disabled",
        10017: "Market closed",
        10018: "Not enough money",
        10019: "Price changed",
        10020: "No quotes",
        10021: "Invalid order expiration",
        10022: "Order state changed",
        10023: "Too many requests",
        10024: "No changes in request",
        10025: "Autotrading disabled",
        10026: "Autotrading disabled on server",
        10027: "Order locked for processing",
        10028: "Long positions only",
        10029: "Short positions only",
        10030: "Close only positions allowed",
    }
    
    message = f"{operation}: {error_messages.get(error_code, f'Unknown error {error_code}')}"
    
    # Map specific codes to specific exceptions
    if error_code == 10018:
        return InsufficientFundsError(message)
    elif error_code in [10013, 10014, 10015]:
        return InvalidOrderParametersError(message)
    elif error_code in [10006, 10012]:
        return OrderRejectedError(message)
    elif error_code in [10016, 10017]:
        return MarketClosedError(message)
    else:
        return OrderRejectedError(message)
