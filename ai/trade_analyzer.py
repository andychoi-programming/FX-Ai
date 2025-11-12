"""
FX-Ai Trade Analyzer Module
Handles trade analysis, timing optimization, and performance evaluation
"""

import os
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TradeAnalyzer:
    """
    Trade analysis and timing optimization for the adaptive learning system.
    Analyzes trade patterns, timing, and performance metrics.
    """

    def __init__(self, db_path: str = None):
        """Initialize the trade analyzer"""
        self.db_path = db_path or os.path.join('data', 'performance_history.db')

    def analyze_forced_closure_timing(self, trade_data: dict):
        """Analyze timing patterns for forced closures to improve holding time optimization"""
        try:
            # For forced closures, we analyze the duration to understand if our time limits are appropriate
            duration_minutes = trade_data.get('duration_minutes', 0)
            profit_pct = trade_data.get('profit_pct', 0)
            reason = trade_data.get('closure_reason', '')

            # Store forced closure analysis
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO forced_closure_analysis
                (timestamp, symbol, duration_minutes, profit_pct, closure_reason)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol, duration_minutes)
                DO UPDATE SET
                    profit_pct = (profit_pct + excluded.profit_pct) / 2,
                    count = count + 1,
                    last_updated = excluded.timestamp
            ''', (
                trade_data['timestamp'],
                trade_data['symbol'],
                duration_minutes,
                profit_pct,
                reason
            ))

            conn.commit()
            conn.close()

            # If this was a time-based closure with profit, consider adjusting optimal times
            if 'time' in reason.lower() and profit_pct > 0:
                self.evaluate_time_based_closure_effectiveness(trade_data)

        except Exception as e:
            logger.error(f"Error analyzing forced closure timing: {e}")

    def evaluate_time_based_closure_effectiveness(self, trade_data: dict):
        """Evaluate if time-based closures are happening too early or too late"""
        try:
            symbol = trade_data['symbol']
            duration_minutes = trade_data['duration_minutes']
            profit_pct = trade_data['profit_pct']

            # Get current optimal holding time for this symbol
            current_optimal = self.get_symbol_optimal_holding_time(symbol)

            if current_optimal.get('found', False):
                optimal_hours = current_optimal['optimal_holding_hours']
                optimal_minutes = optimal_hours * 60

                # If we closed early but had profit, consider extending optimal time
                if duration_minutes < optimal_minutes and profit_pct > 2:  # Had >2% profit
                    logger.info(f"Time-based closure for {symbol} may be too early. "
                              f"Closed at {duration_minutes}min with {profit_pct:.2f}% profit, "
                              f"optimal is {optimal_minutes}min. Consider increasing optimal holding time.")

                # If we hit max time with significant profit, consider increasing max time
                elif duration_minutes >= 480 and profit_pct > 5:  # 8 hours with >5% profit
                    logger.info(f"Max holding time may be too short for {symbol}. "
                              f"Closed at {duration_minutes}min with {profit_pct:.2f}% profit.")

        except Exception as e:
            logger.error(f"Error evaluating time-based closure effectiveness: {e}")

    def analyze_symbol_holding_performance(self, symbol: str) -> Optional[dict]:
        """Analyze holding time performance for a symbol, including forced closures"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get trades for this symbol
            cursor.execute('''
                SELECT profit_pct, duration_minutes, closure_reason
                FROM trades
                WHERE symbol = ? AND duration_minutes > 0
                ORDER BY timestamp DESC
                LIMIT 1000
            ''', (symbol,))

            trades = cursor.fetchall()
            conn.close()

            if len(trades) < 20:
                return None

            # Group trades by duration buckets
            buckets = {
                '0-1h': [], '1-2h': [], '2-3h': [], '3-4h': [],
                '4-6h': [], '6-8h': [], '8-12h': [], '12-24h': [], '24h+': []
            }

            forced_closures = {bucket: [] for bucket in buckets.keys()}

            for profit_pct, duration_min, closure_reason in trades:
                hours = duration_min / 60.0

                # Determine bucket
                if hours < 1:
                    bucket = '0-1h'
                elif hours < 2:
                    bucket = '1-2h'
                elif hours < 3:
                    bucket = '2-3h'
                elif hours < 4:
                    bucket = '3-4h'
                elif hours < 6:
                    bucket = '4-6h'
                elif hours < 8:
                    bucket = '6-8h'
                elif hours < 12:
                    bucket = '8-12h'
                elif hours < 24:
                    bucket = '12-24h'
                else:
                    bucket = '24h+'

                buckets[bucket].append(profit_pct)

                # Track forced closures separately
                if closure_reason and ('time' in closure_reason.lower() or 'max' in closure_reason.lower()):
                    forced_closures[bucket].append(profit_pct)

            # Calculate average profit by bucket
            avg_profit_by_duration = {}
            forced_closure_rates = {}

            for bucket, profits in buckets.items():
                if profits:
                    avg_profit_by_duration[bucket] = np.mean(profits)
                    forced_count = len(forced_closures[bucket])
                    total_count = len(profits)
                    forced_closure_rates[bucket] = forced_count / total_count if total_count > 0 else 0

            # Find best performing bucket
            if avg_profit_by_duration:
                best_bucket = max(avg_profit_by_duration.items(), key=lambda x: x[1])
            else:
                best_bucket = None

            # Count natural vs forced trades
            total_trades = len(trades)
            forced_trades = sum(len(forced_closures[bucket]) for bucket in forced_closures.keys())
            natural_trades = total_trades - forced_trades

            return {
                'symbol': symbol,
                'total_trades': total_trades,
                'natural_trades': natural_trades,
                'forced_trades': forced_trades,
                'avg_profit_by_duration': avg_profit_by_duration,
                'forced_closure_rates': forced_closure_rates,
                'best_bucket': best_bucket
            }

        except Exception as e:
            logger.error(f"Error analyzing symbol holding performance for {symbol}: {e}")
            return None

    def calculate_optimal_holding_times(self, symbol: str) -> Optional[dict]:
        """Calculate optimal holding times for a specific symbol, considering forced closures"""
        try:
            analysis = self.analyze_symbol_holding_performance(symbol)
            if not analysis or not analysis['best_bucket']:
                return None

            best_bucket = analysis['best_bucket'][0]
            best_avg_profit = analysis['best_bucket'][1]
            forced_rates = analysis.get('forced_closure_rates', {})

            # Convert bucket to optimal holding hours
            bucket_ranges = {
                '0-1h': (0.5, 60, 15),    # optimal: 0.5h, max: 1h, min: 15min
                '1-2h': (1.5, 120, 30),   # optimal: 1.5h, max: 2h, min: 30min
                '2-3h': (2.5, 180, 60),   # optimal: 2.5h, max: 3h, min: 1h
                '3-4h': (3.5, 240, 90),   # optimal: 3.5h, max: 4h, min: 1.5h
                '4-6h': (5.0, 360, 120),  # optimal: 5h, max: 6h, min: 2h
                '6-8h': (7.0, 480, 180),  # optimal: 7h, max: 8h, min: 3h
                '8-12h': (10.0, 720, 240),  # optimal: 10h, max: 12h, min: 4h
                '12-24h': (18.0, 1440, 360),  # optimal: 18h, max: 24h, min: 6h
                '24h+': (36.0, 2880, 720)  # optimal: 36h, max: 48h, min: 12h
            }

            if best_bucket in bucket_ranges:
                optimal_hours, max_minutes, min_minutes = bucket_ranges[best_bucket]

                # Adjust optimal times based on forced closure analysis
                # If forced closure rate in best bucket is high (>30%), consider extending time
                best_bucket_forced_rate = forced_rates.get(best_bucket, 0)
                if best_bucket_forced_rate > 0.3 and best_avg_profit > 0.1:  # High forced rate but good profits
                    # Extend optimal time by 25% to capture more profit potential
                    optimal_hours *= 1.25
                    max_minutes = min(max_minutes * 1.25, 480)  # Cap at 8 hours
                    logger.info(f"Extending optimal holding time for {symbol} due to high forced closure rate ({best_bucket_forced_rate:.1%}) with good profits")

                # If forced closures in earlier buckets have good profits, consider shortening time
                early_buckets = ['0-1h', '1-2h', '2-3h']
                high_profit_forced_closures = 0
                for bucket in early_buckets:
                    if bucket in forced_rates and forced_rates[bucket] > 0.2:
                        bucket_profits = analysis['avg_profit_by_duration'].get(bucket, 0)
                        if bucket_profits > 0.05:  # Good profits in early forced closures
                            high_profit_forced_closures += 1

                if high_profit_forced_closures >= 2:
                    # Shorten optimal time since good profits are being cut off early
                    optimal_hours *= 0.8
                    min_minutes = max(min_minutes * 0.8, 15)  # Minimum 15 minutes
                    logger.info(f"Shortening optimal holding time for {symbol} due to profitable early forced closures")

                # Calculate confidence score based on profit difference and sample size
                all_profits = list(analysis['avg_profit_by_duration'].values())
                if len(all_profits) > 1:
                    profit_std = np.std(all_profits)
                    base_confidence = min(1.0, best_avg_profit / (profit_std + 0.01))
                    # Adjust confidence based on forced closure data availability
                    forced_closure_ratio = analysis['forced_trades'] / max(analysis['total_trades'], 1)
                    confidence = base_confidence * (0.8 + 0.2 * forced_closure_ratio)  # Boost confidence with forced closure data
                else:
                    confidence = 0.5

                return {
                    'symbol': symbol,
                    'optimal_holding_hours': optimal_hours,
                    'max_holding_minutes': max_minutes,
                    'min_holding_minutes': min_minutes,
                    'best_bucket': best_bucket,
                    'best_avg_profit': best_avg_profit,
                    'total_trades': analysis['total_trades'],
                    'natural_trades': analysis['natural_trades'],
                    'forced_trades': analysis['forced_trades'],
                    'forced_closure_rate_best_bucket': best_bucket_forced_rate,
                    'confidence_score': confidence,
                    'avg_profit_by_duration': json.dumps(analysis['avg_profit_by_duration']),
                    'forced_closure_rates': json.dumps(forced_rates)
                }

            return None

        except Exception as e:
            logger.error(f"Error calculating optimal holding times for {symbol}: {e}")
            return None

    def update_symbol_holding_times(self, symbol: str):
        """Update optimal holding times for a symbol in the database"""
        try:
            optimal_times = self.calculate_optimal_holding_times(symbol)
            if not optimal_times:
                return

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert or replace symbol holding times
            cursor.execute('''
                INSERT OR REPLACE INTO symbol_holding_times
                (symbol, optimal_holding_hours, max_holding_minutes,
                 min_holding_minutes, avg_profit_by_duration, total_trades,
                 last_updated, confidence_score, natural_trades, forced_trades,
                 forced_closure_rate_best_bucket, forced_closure_rates)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                optimal_times['symbol'],
                optimal_times['optimal_holding_hours'],
                optimal_times['max_holding_minutes'],
                optimal_times['min_holding_minutes'],
                optimal_times['avg_profit_by_duration'],
                optimal_times['total_trades'],
                datetime.now(),
                optimal_times['confidence_score'],
                optimal_times.get('natural_trades'),
                optimal_times.get('forced_trades'),
                optimal_times.get('forced_closure_rate_best_bucket'),
                optimal_times.get('forced_closure_rates')
            ))

            conn.commit()
            conn.close()

            logger.info(
                f"Updated optimal holding times for {symbol}: {
                    optimal_times['optimal_holding_hours']}h " f"(confidence: {
                    optimal_times['confidence_score']:.2f})")

        except Exception as e:
            logger.error(
                f"Error updating symbol holding times for {symbol}: {e}")

    def get_symbol_optimal_holding_time(self, symbol: str) -> dict:
        """Get optimal holding times for a specific symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT optimal_holding_hours, max_holding_minutes,
                       min_holding_minutes,
                       confidence_score, total_trades
                FROM symbol_holding_times
                WHERE symbol = ?
            ''', (symbol,))

            result = cursor.fetchone()
            conn.close()

            if result:
                optimal_hours, max_minutes, min_minutes, confidence, \
                    total_trades = result
                return {
                    'optimal_holding_hours': optimal_hours,
                    'max_holding_minutes': max_minutes,
                    'min_holding_minutes': min_minutes,
                    'confidence_score': confidence,
                    'total_trades': total_trades,
                    'found': True
                }
            else:
                # Return default global parameters if no symbol-specific data
                return {
                    'optimal_holding_hours': 4.0,
                    'max_holding_minutes': 480,
                    'min_holding_minutes': 15,
                    'confidence_score': 0.0,
                    'total_trades': 0,
                    'found': False}

        except Exception as e:
            logger.error(
                f"Error getting symbol optimal holding time for {symbol}: {e}")
            return {
                'optimal_holding_hours': 4.0,
                'max_holding_minutes': 480,
                'min_holding_minutes': 15,
                'confidence_score': 0.0,
                'total_trades': 0,
                'found': False}

    def update_all_symbol_holding_times(self):
        """Update optimal holding times for all symbols"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all symbols with at least 20 trades
            cursor.execute('''
                SELECT symbol, COUNT(*) as trade_count
                FROM trades
                GROUP BY symbol
                HAVING trade_count >= 20
            ''')

            symbols = cursor.fetchall()
            conn.close()

            for symbol, trade_count in symbols:
                logger.info(
                    f"Updating holding times for {symbol} "
                    f"({trade_count} trades)")
                self.update_symbol_holding_times(symbol)

        except Exception as e:
            logger.error(f"Error updating all symbol holding times: {e}")

    def get_all_symbol_holding_times(self) -> dict:
        """Get optimal holding times for all symbols"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''SELECT symbol, optimal_holding_hours, max_holding_minutes,
                       min_holding_minutes,
                       confidence_score, total_trades, last_updated
                FROM symbol_holding_times
                ORDER BY confidence_score DESC, total_trades DESC
            ''')

            results = cursor.fetchall()
            conn.close()

            symbol_times = {}
            for row in results:
                symbol, opt_hours, max_min, min_min, confidence, \
                    trades, last_updated = row
                symbol_times[symbol] = {
                    'optimal_holding_hours': opt_hours,
                    'max_holding_minutes': max_min,
                    'min_holding_minutes': min_min,
                    'confidence_score': confidence,
                    'total_trades': trades,
                    'last_updated': last_updated
                }

            return symbol_times

        except Exception as e:
            logger.error(f"Error getting all symbol holding times: {e}")
            return {}

    def analyze_entry_timing(self):
        """Analyze profitable entry timing patterns across multiple timeframes"""
        logger.info("Analyzing comprehensive temporal patterns...")

        try:
            # Get symbols from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT DISTINCT symbol FROM trades
                WHERE timestamp > datetime('now', '-730 days')
            ''')

            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()

            for symbol in symbols:
                # Get recent trades for this symbol (2 years for comprehensive analysis)
                trades_df = self.get_recent_trades_df(symbol, days=730)

                if len(trades_df) < 50:  # Need more trades for temporal analysis
                    logger.info(f"Skipping {symbol}: insufficient trades ({len(trades_df)})")
                    continue

                # Prepare temporal features
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                trades_df['hour'] = trades_df['timestamp'].dt.hour
                trades_df['day_of_week'] = trades_df['timestamp'].dt.dayofweek
                trades_df['day_of_month'] = trades_df['timestamp'].dt.day
                trades_df['week_of_year'] = trades_df['timestamp'].dt.isocalendar().week
                trades_df['month_of_year'] = trades_df['timestamp'].dt.month
                trades_df['year'] = trades_df['timestamp'].dt.year

                # Calculate additional metrics
                trades_df['is_profitable'] = trades_df['profit'] > 0

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # 1. Analyze HOURLY patterns (existing)
                self._analyze_hourly_patterns(cursor, symbol, trades_df)

                # 2. Analyze DAILY patterns
                self._analyze_daily_patterns(cursor, symbol, trades_df)

                # 3. Analyze WEEKLY patterns
                self._analyze_weekly_patterns(cursor, symbol, trades_df)

                # 4. Analyze MONTHLY patterns
                self._analyze_monthly_patterns(cursor, symbol, trades_df)

                # 5. Analyze YEARLY patterns
                self._analyze_yearly_patterns(cursor, symbol, trades_df)

                conn.commit()
                conn.close()

                logger.info(f"Completed comprehensive temporal analysis for {symbol}")

        except Exception as e:
            logger.error(f"Error in comprehensive temporal analysis: {e}")

    def _analyze_hourly_patterns(self, cursor, symbol: str, trades_df: pd.DataFrame):
        """Analyze hourly trading patterns"""
        try:
            hourly_performance = trades_df.groupby('hour').agg({
                'profit': ['count', 'mean', lambda x: (x > 0).mean()],
                'profit_pct': 'mean'
            }).round(4)

            hourly_performance.columns = ['total_trades', 'avg_profit', 'win_rate', 'avg_profit_pct']
            hourly_performance = hourly_performance.reset_index()

            for _, row in hourly_performance.iterrows():
                if row['total_trades'] >= 5:  # Minimum trades per hour
                    cursor.execute('''
                        INSERT OR REPLACE INTO entry_timing_analysis
                        (symbol, hour_of_day, day_of_week, market_volatility, spread_pips,
                         total_trades, profitable_trades, avg_profit, win_rate, last_updated)
                        VALUES (?, ?, -1, 0, 0, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, int(row['hour']), int(row['total_trades']),
                        int(row['total_trades'] * row['win_rate']),
                        row['avg_profit'], row['win_rate'], datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error analyzing hourly patterns for {symbol}: {e}")

    def _analyze_daily_patterns(self, cursor, symbol: str, trades_df: pd.DataFrame):
        """Analyze daily trading patterns"""
        try:
            # Group by day of month, month, and year
            daily_performance = trades_df.groupby(['day_of_month', 'month_of_year', 'year']).agg({
                'profit': ['count', 'mean', 'sum', lambda x: (x > 0).mean()],
                'profit_pct': 'mean'
            }).round(4)

            daily_performance.columns = ['total_trades', 'avg_profit', 'total_profit', 'win_rate', 'avg_profit_pct']
            daily_performance = daily_performance.reset_index()

            # Calculate additional metrics
            daily_performance['sharpe_ratio'] = daily_performance.apply(
                lambda row: self._calculate_sharpe_ratio(trades_df, row.name), axis=1
            )
            daily_performance['max_drawdown'] = daily_performance.apply(
                lambda row: self._calculate_max_drawdown(trades_df, row.name), axis=1
            )

            for _, row in daily_performance.iterrows():
                if row['total_trades'] >= 3:  # Minimum trades per day
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_temporal_analysis
                        (symbol, day_of_month, month_of_year, year, total_trades,
                         profitable_trades, avg_profit, win_rate, avg_volatility,
                         avg_spread_pips, sharpe_ratio, max_drawdown, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                    ''', (
                        symbol, int(row['day_of_month']), int(row['month_of_year']),
                        int(row['year']), int(row['total_trades']),
                        int(row['total_trades'] * row['win_rate']),
                        row['avg_profit'], row['win_rate'],
                        row['sharpe_ratio'], row['max_drawdown'], datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error analyzing daily patterns for {symbol}: {e}")

    def _analyze_weekly_patterns(self, cursor, symbol: str, trades_df: pd.DataFrame):
        """Analyze weekly trading patterns"""
        try:
            weekly_performance = trades_df.groupby(['week_of_year', 'year']).agg({
                'profit': ['count', 'mean', 'sum', lambda x: (x > 0).mean()],
                'profit_pct': 'mean'
            }).round(4)

            weekly_performance.columns = ['total_trades', 'avg_profit', 'total_profit', 'win_rate', 'avg_profit_pct']
            weekly_performance = weekly_performance.reset_index()

            weekly_performance['sharpe_ratio'] = weekly_performance.apply(
                lambda row: self._calculate_sharpe_ratio(trades_df, row.name), axis=1
            )
            weekly_performance['max_drawdown'] = weekly_performance.apply(
                lambda row: self._calculate_max_drawdown(trades_df, row.name), axis=1
            )

            for _, row in weekly_performance.iterrows():
                if row['total_trades'] >= 5:  # Minimum trades per week
                    cursor.execute('''
                        INSERT OR REPLACE INTO weekly_temporal_analysis
                        (symbol, week_of_year, year, total_trades, profitable_trades,
                         avg_profit, win_rate, avg_volatility, avg_spread_pips,
                         sharpe_ratio, max_drawdown, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                    ''', (
                        symbol, int(row['week_of_year']), int(row['year']),
                        int(row['total_trades']),
                        int(row['total_trades'] * row['win_rate']),
                        row['avg_profit'], row['win_rate'],
                        row['sharpe_ratio'], row['max_drawdown'], datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error analyzing weekly patterns for {symbol}: {e}")

    def _analyze_monthly_patterns(self, cursor, symbol: str, trades_df: pd.DataFrame):
        """Analyze monthly trading patterns"""
        try:
            monthly_performance = trades_df.groupby(['month_of_year', 'year']).agg({
                'profit': ['count', 'mean', 'sum', lambda x: (x > 0).mean()],
                'profit_pct': 'mean'
            }).round(4)

            monthly_performance.columns = ['total_trades', 'avg_profit', 'total_profit', 'win_rate', 'avg_profit_pct']
            monthly_performance = monthly_performance.reset_index()

            monthly_performance['sharpe_ratio'] = monthly_performance.apply(
                lambda row: self._calculate_sharpe_ratio(trades_df, row.name), axis=1
            )
            monthly_performance['max_drawdown'] = monthly_performance.apply(
                lambda row: self._calculate_max_drawdown(trades_df, row.name), axis=1
            )

            for _, row in monthly_performance.iterrows():
                if row['total_trades'] >= 10:  # Minimum trades per month
                    cursor.execute('''
                        INSERT OR REPLACE INTO monthly_temporal_analysis
                        (symbol, month_of_year, year, total_trades, profitable_trades,
                         avg_profit, win_rate, avg_volatility, avg_spread_pips,
                         sharpe_ratio, max_drawdown, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                    ''', (
                        symbol, int(row['month_of_year']), int(row['year']),
                        int(row['total_trades']),
                        int(row['total_trades'] * row['win_rate']),
                        row['avg_profit'], row['win_rate'],
                        row['sharpe_ratio'], row['max_drawdown'], datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error analyzing monthly patterns for {symbol}: {e}")

    def _analyze_yearly_patterns(self, cursor, symbol: str, trades_df: pd.DataFrame):
        """Analyze yearly trading patterns"""
        try:
            yearly_performance = trades_df.groupby('year').agg({
                'profit': ['count', 'mean', 'sum', lambda x: (x > 0).mean()],
                'profit_pct': 'mean'
            }).round(4)

            yearly_performance.columns = ['total_trades', 'avg_profit', 'total_profit', 'win_rate', 'avg_profit_pct']
            yearly_performance = yearly_performance.reset_index()

            yearly_performance['sharpe_ratio'] = yearly_performance.apply(
                lambda row: self._calculate_sharpe_ratio(trades_df, row.name), axis=1
            )
            yearly_performance['max_drawdown'] = yearly_performance.apply(
                lambda row: self._calculate_max_drawdown(trades_df, row.name), axis=1
            )

            for _, row in yearly_performance.iterrows():
                if row['total_trades'] >= 20:  # Minimum trades per year
                    cursor.execute('''
                        INSERT OR REPLACE INTO yearly_temporal_analysis
                        (symbol, year, total_trades, profitable_trades, avg_profit,
                         win_rate, avg_volatility, avg_spread_pips, sharpe_ratio,
                         max_drawdown, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                    ''', (
                        symbol, int(row['year']), int(row['total_trades']),
                        int(row['total_trades'] * row['win_rate']),
                        row['avg_profit'], row['win_rate'],
                        row['sharpe_ratio'], row['max_drawdown'], datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error analyzing yearly patterns for {symbol}: {e}")

    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame, group_key) -> float:
        """Calculate Sharpe ratio for a group of trades"""
        try:
            # This is a simplified Sharpe ratio calculation
            # In practice, you'd want daily returns and risk-free rate
            if len(trades_df) < 5:
                return 0.0

            returns = trades_df['profit_pct']
            if returns.std() == 0:
                return 0.0

            return (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, trades_df: pd.DataFrame, group_key) -> float:
        """Calculate maximum drawdown for a group of trades"""
        try:
            if len(trades_df) < 3:
                return 0.0

            cumulative = (1 + trades_df['profit_pct']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        except Exception:
            return 0.0

    def get_recent_trades_df(self, symbol: str, days: int) -> pd.DataFrame:
        """Get recent trades from database as DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f'''
                SELECT * FROM trades
                WHERE symbol = ? AND timestamp > datetime('now', '-{days} days')
            '''
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {e}")
            return pd.DataFrame()