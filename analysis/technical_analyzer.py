"""
Technical Analyzer Module
Performs technical analysis on price data and generates trading signals
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.getLogger(__name__).warning("TA-Lib not available, using simplified calculations")

class TechnicalAnalyzer:
    """Performs technical analysis and generates signals"""

    def __init__(self, config: Dict):
        """
        Initialize technical analyzer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Indicator parameters
        self.vwap_period = config.get('vwap_period', 20)
        self.ema_fast = config.get('ema_fast', 9)
        self.ema_slow = config.get('ema_slow', 21)
        self.rsi_period = config.get('rsi_period', 14)
        self.atr_period = config.get('atr_period', 14)

    def analyze_symbol(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Perform complete technical analysis on symbol data from multiple timeframes

        Args:
            symbol: Trading symbol
            data: Dictionary of OHLCV DataFrames by timeframe {'M1': df, 'M5': df, 'H1': df, 'H4': df, 'D1': df}

        Returns:
            dict: Technical analysis results
        """
        try:
            if data is None or not isinstance(data, dict) or 'H1' not in data:
                return {}

            # Use H1 as primary timeframe for most analysis
            primary_data = data['H1']
            if primary_data is None or len(primary_data) < 50:
                return {}

            # Ensure volume column exists
            if 'volume' not in primary_data.columns:
                primary_data = primary_data.copy()
                primary_data['volume'] = primary_data.get('tick_volume', 1)

            analysis = {}

            # VWAP Analysis (primary timeframe)
            analysis['vwap'] = self._calculate_vwap(primary_data)

            # EMA Analysis (primary timeframe)
            analysis['ema'] = self._calculate_ema_signals(primary_data)

            # RSI Analysis (primary timeframe)
            analysis['rsi'] = self._calculate_rsi_signals(primary_data)

            # ATR for volatility (multi-timeframe)
            analysis['atr'] = self._calculate_multi_timeframe_atr(data)

            # Support/Resistance levels (primary timeframe)
            analysis['support_resistance'] = self._find_support_resistance(primary_data)

            # Volume analysis (primary timeframe)
            analysis['volume'] = self._analyze_volume(primary_data)

            # Calculate overall score from individual signals
            analysis['overall_score'] = self._calculate_overall_score(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {}

    async def analyze(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Async wrapper for analyze_symbol method

        Args:
            symbol: Trading symbol
            data: Dictionary of OHLCV DataFrames by timeframe

        Returns:
            dict: Technical analysis results
        """
        return self.analyze_symbol(symbol, data)

    def _calculate_vwap(self, data: pd.DataFrame) -> Dict:
        """
        Calculate VWAP and position relative to it

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: VWAP analysis
        """
        try:
            # Calculate VWAP
            data_copy = data.copy()
            data_copy['typical_price'] = (data_copy['high'] + data_copy['low'] + data_copy['close']) / 3
            data_copy['pv'] = data_copy['typical_price'] * data_copy['volume']
            data_copy['cumulative_pv'] = data_copy['pv'].cumsum()
            data_copy['cumulative_volume'] = data_copy['volume'].cumsum()
            data_copy['vwap'] = data_copy['cumulative_pv'] / data_copy['cumulative_volume']

            current_price = data_copy['close'].iloc[-1]
            current_vwap = data_copy['vwap'].iloc[-1]

            return {
                'value': current_vwap,
                'position': 'above' if current_price > current_vwap else 'below',
                'distance': abs(current_price - current_vwap) / current_price
            }

        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return {}

    def _calculate_ema_signals(self, data: pd.DataFrame) -> Dict:
        """
        Calculate EMA signals and trend analysis

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: EMA analysis
        """
        try:
            # Calculate EMAs manually if TA-Lib not available
            if TALIB_AVAILABLE:
                ema_fast = talib.EMA(data['close'], timeperiod=self.ema_fast)
                ema_slow = talib.EMA(data['close'], timeperiod=self.ema_slow)
            else:
                ema_fast = self._calculate_ema(data['close'], self.ema_fast)
                ema_slow = self._calculate_ema(data['close'], self.ema_slow)

            # Get current values
            fast_current = ema_fast.iloc[-1]
            slow_current = ema_slow.iloc[-1]
            fast_prev = ema_fast.iloc[-2]
            slow_prev = ema_slow.iloc[-2]

            # Determine trend
            if fast_current > slow_current and fast_prev <= slow_prev:
                trend = 'bullish_crossover'
            elif fast_current < slow_current and fast_prev >= slow_prev:
                trend = 'bearish_crossover'
            elif fast_current > slow_current:
                trend = 'bullish'
            else:
                trend = 'bearish'

            return {
                'fast_ema': fast_current,
                'slow_ema': slow_current,
                'trend': trend,
                'separation': abs(fast_current - slow_current) / slow_current
            }

        except Exception as e:
            self.logger.error(f"Error calculating EMA signals: {e}")
            return {}

    def _calculate_rsi_signals(self, data: pd.DataFrame) -> Dict:
        """
        Calculate RSI and generate signals

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: RSI analysis
        """
        try:
            if TALIB_AVAILABLE:
                rsi = talib.RSI(data['close'], timeperiod=self.rsi_period)
            else:
                rsi = self._calculate_rsi(data['close'], self.rsi_period)

            current_rsi = rsi.iloc[-1]

            # Determine RSI signal
            if current_rsi < 30:
                signal = 'oversold'
            elif current_rsi > 70:
                signal = 'overbought'
            else:
                signal = 'neutral'

            return {
                'value': current_rsi,
                'signal': signal,
                'divergence': self._check_rsi_divergence(data, rsi)
            }

        except Exception as e:
            self.logger.error(f"Error calculating RSI signals: {e}")
            return {}

    def _calculate_multi_timeframe_atr(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate Average True Range from multiple timeframes and combine them

        Args:
            data: Dictionary of DataFrames by timeframe

        Returns:
            dict: Multi-timeframe ATR analysis
        """
        try:
            timeframe_weights = {
                'M1': 0.1,   # 10% weight - short-term noise
                'M5': 0.2,   # 20% weight - short-term trends
                'H1': 0.4,   # 40% weight - primary timeframe
                'H4': 0.2,   # 20% weight - medium-term trends
                'D1': 0.1    # 10% weight - long-term context
            }

            atr_values = []
            valid_timeframes = 0

            for timeframe, weight in timeframe_weights.items():
                if timeframe in data and data[timeframe] is not None:
                    df = data[timeframe]
                    if len(df) >= 20:  # Minimum bars for ATR calculation
                        # Calculate ATR for this timeframe
                        if TALIB_AVAILABLE:
                            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
                        else:
                            atr = self._calculate_atr_manual(df, self.atr_period)

                        current_atr = atr.iloc[-1]
                        if not pd.isna(current_atr) and current_atr > 0:
                            atr_values.append(current_atr * weight)
                            valid_timeframes += 1

            if not atr_values:
                return {}

            # Calculate weighted average ATR
            combined_atr = sum(atr_values) / sum(timeframe_weights[tf] for tf in timeframe_weights.keys() 
                                                if any(tf in data and data[tf] is not None and len(data[tf]) >= 20 for _ in [None]))

            # Use primary timeframe (H1) price for percentage calculation
            primary_price = data['H1']['close'].iloc[-1]
            atr_percentage = (combined_atr / primary_price) * 100

            return {
                'value': combined_atr,
                'percentage': atr_percentage,
                'volatility': 'high' if atr_percentage > 1.5 else 'low',  # Adjusted threshold for multi-timeframe
                'timeframes_used': valid_timeframes,
                'method': 'multi_timeframe_weighted'
            }

        except Exception as e:
            self.logger.error(f"Error calculating multi-timeframe ATR: {e}")
            # Fallback to single timeframe ATR
            if 'H1' in data and data['H1'] is not None:
                return self._calculate_atr_manual(data['H1'], self.atr_period).iloc[-1]
            return {}

    def _find_support_resistance(self, data: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Find support and resistance levels

        Args:
            data: OHLCV DataFrame
            lookback: Number of bars to look back

        Returns:
            dict: Support/resistance levels
        """
        try:
            recent_data = data.tail(lookback)

            # Simple pivot points
            high = recent_data['high'].max()
            low = recent_data['low'].min()

            # Find local highs and lows
            pivot_highs = []
            pivot_lows = []

            for i in range(2, len(recent_data) - 2):
                if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                    pivot_highs.append(recent_data['high'].iloc[i])

                if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                    pivot_lows.append(recent_data['low'].iloc[i])

            # Get nearest levels
            current_price = recent_data['close'].iloc[-1]

            resistance_levels = sorted([h for h in pivot_highs if h > current_price])
            support_levels = sorted([s for s in pivot_lows if s < current_price], reverse=True)

            return {
                'next_resistance': resistance_levels[0] if resistance_levels else None,
                'next_support': support_levels[0] if support_levels else None,
                'near_resistance': len(resistance_levels) > 0 and (resistance_levels[0] - current_price) / current_price < 0.005,
                'near_support': len(support_levels) > 0 and (current_price - support_levels[0]) / current_price < 0.005
            }

        except Exception as e:
            self.logger.error(f"Error finding support/resistance: {e}")
            return {}

    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """
        Analyze volume patterns

        Args:
            data: OHLCV DataFrame

        Returns:
            dict: Volume analysis
        """
        try:
            recent_volume = data['volume'].tail(20)
            avg_volume = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]

            # Volume trend
            if TALIB_AVAILABLE:
                volume_ma = talib.SMA(recent_volume, timeperiod=10)
            else:
                volume_ma = recent_volume.rolling(10).mean()

            volume_trend = 'increasing' if volume_ma.iloc[-1] > volume_ma.iloc[-2] else 'decreasing'

            return {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                'increasing': volume_trend == 'increasing'
            }

        except Exception as e:
            self.logger.error(f"Error analyzing volume: {e}")
            return {}

    def _check_rsi_divergence(self, data: pd.DataFrame, rsi: pd.Series) -> str:
        """
        Check for RSI divergence

        Args:
            data: OHLCV DataFrame
            rsi: RSI series

        Returns:
            str: Divergence type or 'none'
        """
        try:
            # Simple divergence check - compare last 5 bars
            price_recent = data['close'].tail(5)
            rsi_recent = rsi.tail(5)

            price_trend = 'up' if price_recent.iloc[-1] > price_recent.iloc[0] else 'down'
            rsi_trend = 'up' if rsi_recent.iloc[-1] > rsi_recent.iloc[0] else 'down'

            if price_trend != rsi_trend:
                if price_trend == 'up' and rsi_trend == 'down':
                    return 'bearish_divergence'
                elif price_trend == 'down' and rsi_trend == 'up':
                    return 'bullish_divergence'

            return 'none'

        except Exception as e:
            return 'none'

    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average manually

        Args:
            data: Price data series
            period: EMA period

        Returns:
            pd.Series: EMA values
        """
        return data.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI manually

        Args:
            data: Price data series
            period: RSI period

        Returns:
            pd.Series: RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr_manual(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate ATR manually

        Args:
            data: OHLCV DataFrame
            period: ATR period

        Returns:
            pd.Series: ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR
        return tr.rolling(window=period).mean()

    def _calculate_overall_score(self, analysis: Dict) -> float:
        """
        Calculate overall technical score from individual signals

        Args:
            analysis: Dictionary with individual analysis results

        Returns:
            float: Overall score between 0 and 1
        """
        try:
            scores = []

            # VWAP score (0.2 weight)
            vwap = analysis.get('vwap', {})
            if vwap.get('position') == 'above':
                vwap_score = 0.6  # Bullish
            elif vwap.get('position') == 'below':
                vwap_score = 0.4  # Bearish
            else:
                vwap_score = 0.5  # Neutral
            scores.append((vwap_score, 0.2))

            # EMA score (0.3 weight)
            ema = analysis.get('ema', {})
            ema_trend = ema.get('trend', 'neutral')
            if ema_trend == 'bullish':
                ema_score = 0.7
            elif ema_trend == 'bearish':
                ema_score = 0.3
            else:
                ema_score = 0.5
            scores.append((ema_score, 0.3))

            # RSI score (0.25 weight)
            rsi = analysis.get('rsi', {})
            rsi_signal = rsi.get('signal', 'neutral')
            if rsi_signal == 'oversold':
                rsi_score = 0.8  # Buy signal
            elif rsi_signal == 'overbought':
                rsi_score = 0.2  # Sell signal
            else:
                rsi_score = 0.5  # Neutral
            scores.append((rsi_score, 0.25))

            # Volume score (0.15 weight)
            volume = analysis.get('volume', {})
            volume_signal = volume.get('signal', 'neutral')
            if volume_signal == 'increasing':
                volume_score = 0.6
            elif volume_signal == 'decreasing':
                volume_score = 0.4
            else:
                volume_score = 0.5
            scores.append((volume_score, 0.15))

            # Support/Resistance score (0.1 weight)
            sr = analysis.get('support_resistance', {})
            sr_signal = sr.get('signal', 'neutral')
            if sr_signal == 'support':
                sr_score = 0.7  # Near support
            elif sr_signal == 'resistance':
                sr_score = 0.3  # Near resistance
            else:
                sr_score = 0.5
            scores.append((sr_score, 0.1))

            # Calculate weighted average
            total_score = 0
            total_weight = 0
            for score, weight in scores:
                total_score += score * weight
                total_weight += weight

            if total_weight == 0:
                return 0.5

            return total_score / total_weight

        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 0.5