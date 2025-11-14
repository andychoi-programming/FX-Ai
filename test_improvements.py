#!/usr/bin/env python3
"""
Test each improvement independently
"""
import sys
import asyncio
sys.path.insert(0, '.')

from app.application import FXAiApplication
from core.risk_manager import RiskManager
from analysis.technical_analyzer import TechnicalAnalyzer
from analysis.sentiment_analyzer import SentimentAnalyzer
from ai.ml_predictor import MLPredictor
from utils.config_loader import ConfigLoader
from utils.circuit_breaker import CircuitBreaker


async def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("🧪 Testing Circuit Breaker...")

    circuit_breaker = CircuitBreaker()

    # Simulate failures
    for i in range(5):
        try:
            circuit_breaker.call(lambda: 1/0)  # Intentional error
        except:
            pass

    status = circuit_breaker.get_status()
    assert status['is_open'], "Circuit breaker should be open after 5 failures"
    print("✅ Circuit breaker test passed")


async def test_data_freshness():
    """Test data freshness checks"""
    print("🧪 Testing Data Freshness...")

    config = ConfigLoader().load_config()
    technical_analyzer = TechnicalAnalyzer(config)
    sentiment_analyzer = SentimentAnalyzer(config)

    # Check all analyzers have freshness methods
    assert hasattr(technical_analyzer, 'is_data_fresh')
    assert hasattr(sentiment_analyzer, 'is_data_fresh')
    print("✅ Data freshness test passed")


async def test_position_validation():
    """Test position size validation"""
    print("🧪 Testing Position Validation...")

    config = ConfigLoader().load_config()
    risk_manager = RiskManager(config)

    # Test with oversized position
    result = risk_manager.validate_position_size(
        'EURUSD',
        100.0,  # Way too large
        10000.0  # $10k account
    )

    assert not result, "Should reject oversized position"
    print("✅ Position validation test passed")


async def test_feature_consistency():
    """Test feature consistency validation"""
    print("🧪 Testing Feature Consistency...")

    config = ConfigLoader().load_config()
    ml_predictor = MLPredictor(config)

    # Test with complete feature set
    test_features = {
        'returns_1d': 0.01,
        'returns_5d': 0.05,
        'volatility_5d': 0.02,
        'volatility_20d': 0.03,
        'volume_ratio': 1.2,
        'rsi_norm': 0.6,
        'vwap_position': 0.55,
        'bb_position': 0.45,
        'macd_signal': 0.1,
        'trend_strength': 0.7,
        'momentum': 0.8,
        'support_resistance': 0.6,
        'regime_score': 0.75
    }

    assert ml_predictor.validate_feature_consistency(test_features), "Feature validation should pass"
    print("✅ Feature consistency test passed")


async def test_daily_limits():
    """Test daily limit tracker"""
    print("🧪 Testing Daily Limits...")

    from app.trading_orchestrator import DailyLimitTracker

    tracker = DailyLimitTracker()

    # Test normal operation
    assert tracker.can_trade('EURUSD'), "Should allow first trade"
    tracker.record_trade('EURUSD')

    # Test symbol limit
    assert not tracker.can_trade('EURUSD'), "Should block second EURUSD trade"
    assert tracker.can_trade('GBPUSD'), "Should allow different symbol"

    print("✅ Daily limits test passed")


async def test_performance_monitoring():
    """Test performance monitoring"""
    print("🧪 Testing Performance Monitoring...")

    from utils.performance_monitor import PerformanceTracker, monitor_performance
    import time

    tracker = PerformanceTracker()

    # Test tracking
    tracker.track_execution('test_operation', 1.5)
    tracker.track_execution('test_operation', 2.0)

    metrics = tracker.get_metrics_report()
    assert 'test_operation' in metrics, "Should track operations"
    assert metrics['test_operation']['avg'] > 0, "Should calculate averages"

    print("✅ Performance monitoring test passed")


async def run_all_tests():
    """Run all component tests"""
    print("🚀 RUNNING COMPONENT TESTS")
    print("=" * 50)

    tests = [
        test_circuit_breaker,
        test_data_freshness,
        test_position_validation,
        test_feature_consistency,
        test_daily_limits,
        test_performance_monitoring
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎯 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - check implementation")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)