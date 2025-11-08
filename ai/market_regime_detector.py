#!/usr/bin/env python3
"""
Market Regime Detection Module
Detects different market conditions to adapt trading strategies
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"

@dataclass
class RegimeAnalysis:
    """Container for regime analysis results"""
    primary_regime: MarketRegime
    secondary_regimes: List[MarketRegime]
    confidence: float
    adx_value: float
    volatility_ratio: float
    trend_strength: float
    regime_score: float

class MarketRegimeDetector:
    """Detects and classifies market regimes for adaptive trading"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Regime detection thresholds
        self.adx_thresholds = {
            'trending': self.config.get('regime_adx_trending', 25),
            'strong_trend': self.config.get('regime_adx_strong', 40)
        }

        self.volatility_thresholds = {
            'high_vol': self.config.get('regime_vol_high', 1.5),
            'low_vol': self.config.get('regime_vol_low', 0.7)
        }

        self.trend_thresholds = {
            'bull_market': self.config.get('regime_bull_threshold', 0.02),  # 2% above 200MA
            'bear_market': self.config.get('regime_bear_threshold', -0.02)  # 2% below 200MA
        }

        # Moving average periods for regime detection
        self.ma_periods = {
            'short': self.config.get('regime_ma_short', 20),
            'medium': self.config.get('regime_ma_medium', 50),
            'long': self.config.get('regime_ma_long', 200)
        }

        logger.info("Market Regime Detector initialized")

    def analyze_regime(self, symbol: str, price_data: pd.DataFrame) -> RegimeAnalysis:
        """
        Analyze current market regime for a symbol

        Args:
            symbol: Trading symbol
            price_data: OHLCV data with at least 200 periods

        Returns:
            RegimeAnalysis with primary and secondary regimes
        """
        try:
            if len(price_data) < 200:
                logger.warning(f"Insufficient data for {symbol} regime analysis")
                return self._default_regime()

            # Calculate regime indicators
            adx_value = self._calculate_adx(price_data)
            volatility_ratio = self._calculate_volatility_ratio(price_data)
            trend_direction = self._calculate_trend_direction(price_data)
            trend_strength = self._calculate_trend_strength(price_data)

            # Determine primary regime
            primary_regime = self._classify_primary_regime(
                adx_value, volatility_ratio, trend_direction, trend_strength
            )

            # Determine secondary regimes
            secondary_regimes = self._classify_secondary_regimes(
                adx_value, volatility_ratio, trend_direction, trend_strength
            )

            # Calculate overall confidence
            confidence = self._calculate_regime_confidence(
                adx_value, volatility_ratio, trend_strength
            )

            # Calculate regime score for adaptive learning
            regime_score = self._calculate_regime_score(
                primary_regime, secondary_regimes, confidence
            )

            return RegimeAnalysis(
                primary_regime=primary_regime,
                secondary_regimes=secondary_regimes,
                confidence=confidence,
                adx_value=adx_value,
                volatility_ratio=volatility_ratio,
                trend_strength=trend_strength,
                regime_score=regime_score
            )

        except Exception as e:
            logger.error(f"Error analyzing regime for {symbol}: {e}")
            return self._default_regime()

    def _calculate_adx(self, data: pd.DataFrame) -> float:
        """Calculate Average Directional Index"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate Directional Movement
            dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low),
                             np.maximum(high - high.shift(1), 0), 0)
            dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)),
                               np.maximum(low.shift(1) - low, 0), 0)

            # Calculate Directional Indicators
            period = 14
            atr = tr.rolling(period).mean()
            di_plus = 100 * (pd.Series(dm_plus).rolling(period).mean() / atr)
            di_minus = 100 * (pd.Series(dm_minus).rolling(period).mean() / atr)

            # Calculate ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(period).mean()

            return adx.iloc[-1] if not adx.empty else 20.0

        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return 20.0

    def _calculate_volatility_ratio(self, data: pd.DataFrame) -> float:
        """Calculate current volatility relative to historical average"""
        try:
            # Use ATR as volatility measure
            high = data['high']
            low = data['low']
            close = data['close']

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr_current = tr.rolling(14).mean().iloc[-1]
            atr_average = tr.rolling(100).mean().iloc[-1]

            return atr_current / atr_average if atr_average > 0 else 1.0

        except Exception as e:
            logger.error(f"Error calculating volatility ratio: {e}")
            return 1.0

    def _calculate_trend_direction(self, data: pd.DataFrame) -> float:
        """Calculate trend direction (-1 to 1, negative = downtrend)"""
        try:
            close = data['close']
            ma_short = close.rolling(self.ma_periods['short']).mean()
            ma_long = close.rolling(self.ma_periods['long']).mean()

            current_price = close.iloc[-1]
            long_ma = ma_long.iloc[-1]

            # Trend direction based on price vs long MA
            trend_direction = (current_price - long_ma) / long_ma

            return trend_direction

        except Exception as e:
            logger.error(f"Error calculating trend direction: {e}")
            return 0.0

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)"""
        try:
            close = data['close']
            ma_short = close.rolling(self.ma_periods['short']).mean()
            ma_medium = close.rolling(self.ma_periods['medium']).mean()

            # Trend strength based on MA alignment
            short_above_medium = (ma_short > ma_medium).rolling(20).mean()
            trend_strength = abs(short_above_medium.iloc[-1] - 0.5) * 2

            return min(trend_strength, 1.0)

        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.5

    def _classify_primary_regime(self, adx: float, vol_ratio: float,
                                trend_dir: float, trend_strength: float) -> MarketRegime:
        """Classify the primary market regime"""
        # Strong trending market
        if adx >= self.adx_thresholds['strong_trend']:
            if trend_dir > self.trend_thresholds['bull_market']:
                return MarketRegime.TRENDING_UP
            elif trend_dir < self.trend_thresholds['bear_market']:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.TRENDING_UP if trend_dir > 0 else MarketRegime.TRENDING_DOWN

        # Moderate trending market
        elif adx >= self.adx_thresholds['trending']:
            if trend_dir > 0.005:  # Slightly above MA
                return MarketRegime.BULL_MARKET
            elif trend_dir < -0.005:  # Slightly below MA
                return MarketRegime.BEAR_MARKET
            else:
                return MarketRegime.SIDEWAYS

        # Ranging market
        else:
            if vol_ratio >= self.volatility_thresholds['high_vol']:
                return MarketRegime.HIGH_VOLATILITY
            elif vol_ratio <= self.volatility_thresholds['low_vol']:
                return MarketRegime.LOW_VOLATILITY
            else:
                return MarketRegime.RANGING

    def _classify_secondary_regimes(self, adx: float, vol_ratio: float,
                                   trend_dir: float, trend_strength: float) -> List[MarketRegime]:
        """Classify secondary market regimes"""
        secondary = []

        # Add volatility regime
        if vol_ratio >= self.volatility_thresholds['high_vol']:
            secondary.append(MarketRegime.HIGH_VOLATILITY)
        elif vol_ratio <= self.volatility_thresholds['low_vol']:
            secondary.append(MarketRegime.LOW_VOLATILITY)

        # Add trend strength
        if trend_strength > 0.7:
            if trend_dir > 0:
                secondary.append(MarketRegime.BULL_MARKET)
            else:
                secondary.append(MarketRegime.BEAR_MARKET)

        return secondary

    def _calculate_regime_confidence(self, adx: float, vol_ratio: float, trend_strength: float) -> float:
        """Calculate confidence in regime classification (0-1)"""
        # Higher ADX = higher confidence in trending regime
        adx_confidence = min(adx / 50.0, 1.0)

        # Volatility extremes = higher confidence
        vol_confidence = abs(vol_ratio - 1.0) * 2
        vol_confidence = min(vol_confidence, 1.0)

        # Trend strength confidence
        trend_confidence = trend_strength

        # Average confidence
        return (adx_confidence + vol_confidence + trend_confidence) / 3.0

    def _calculate_regime_score(self, primary: MarketRegime, secondary: List[MarketRegime], confidence: float) -> float:
        """Calculate numerical score for regime (used in adaptive learning)"""
        base_scores = {
            MarketRegime.TRENDING_UP: 1.0,
            MarketRegime.TRENDING_DOWN: -1.0,
            MarketRegime.BULL_MARKET: 0.7,
            MarketRegime.BEAR_MARKET: -0.7,
            MarketRegime.RANGING: 0.0,
            MarketRegime.HIGH_VOLATILITY: 0.3,
            MarketRegime.LOW_VOLATILITY: -0.3,
            MarketRegime.SIDEWAYS: 0.0
        }

        score = base_scores.get(primary, 0.0)

        # Adjust for secondary regimes
        for regime in secondary:
            score += base_scores.get(regime, 0.0) * 0.3  # Secondary weight

        # Apply confidence
        return score * confidence

    def _default_regime(self) -> RegimeAnalysis:
        """Return default regime when analysis fails"""
        return RegimeAnalysis(
            primary_regime=MarketRegime.RANGING,
            secondary_regimes=[],
            confidence=0.5,
            adx_value=20.0,
            volatility_ratio=1.0,
            trend_strength=0.5,
            regime_score=0.0
        )

    def get_regime_adapted_parameters(self, regime: RegimeAnalysis) -> Dict:
        """Get trading parameters adapted to current regime"""
        base_params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_signal_strength': 0.6,
            'trailing_stop_distance': 20,
            'risk_multiplier': 1.0,
            'max_holding_hours': 4
        }

        # Adapt parameters based on regime
        if regime.primary_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # Strong trends: wider stops, longer holding
            base_params.update({
                'trailing_stop_distance': 30,
                'max_holding_hours': 8,
                'min_signal_strength': 0.7
            })
        elif regime.primary_regime == MarketRegime.HIGH_VOLATILITY:
            # High vol: tighter stops, smaller positions
            base_params.update({
                'risk_multiplier': 0.8,
                'trailing_stop_distance': 15,
                'max_holding_hours': 2
            })
        elif regime.primary_regime == MarketRegime.LOW_VOLATILITY:
            # Low vol: wider stops, breakout focus
            base_params.update({
                'trailing_stop_distance': 25,
                'max_holding_hours': 6,
                'rsi_oversold': 25,
                'rsi_overbought': 75
            })

        return base_params