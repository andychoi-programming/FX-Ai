"""
FX-Ai Learning Database Module
Handles all database operations for the adaptive learning system
"""

import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class LearningDatabase:
    """
    Database operations for the adaptive learning system.
    Handles trade recording, performance tracking, and data retrieval.
    """

    def __init__(self, db_path: str = None):
        """Initialize database connection"""
        self.db_path = db_path or os.path.join('data', 'performance_history.db')
        os.makedirs('data', exist_ok=True)

    def init_database(self):
        """Initialize SQLite database for trade and performance tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                volume REAL,
                profit REAL,
                profit_pct REAL,
                signal_strength REAL,
                ml_score REAL,
                technical_score REAL,
                sentiment_score REAL,
                duration_minutes INTEGER,
                model_version TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                model_type TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                avg_profit REAL,
                total_trades INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameter_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                parameter_name TEXT,
                old_value REAL,
                new_value REAL,
                improvement_pct REAL,
                validation_score REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbol_holding_times (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE,
                optimal_holding_hours REAL,
                max_holding_minutes INTEGER,
                min_holding_minutes INTEGER,
                avg_profit_by_duration TEXT,  -- JSON duration buckets
                total_trades INTEGER,
                last_updated DATETIME,
                confidence_score REAL
            )
        ''')

        # Entry timing analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entry_timing_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                hour_of_day INTEGER,  -- 0-23
                day_of_week INTEGER,  -- 0-6 (Monday-Sunday)
                market_volatility REAL,  -- ATR normalized volatility
                spread_pips REAL,      -- Spread in pips at entry
                total_trades INTEGER,
                profitable_trades INTEGER,
                avg_profit REAL,
                win_rate REAL,
                last_updated DATETIME
            )
        ''')

        # Daily temporal analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_temporal_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                day_of_month INTEGER,  -- 1-31
                month_of_year INTEGER, -- 1-12
                year INTEGER,          -- Year
                total_trades INTEGER,
                profitable_trades INTEGER,
                avg_profit REAL,
                win_rate REAL,
                avg_volatility REAL,
                avg_spread_pips REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                last_updated DATETIME,
                UNIQUE(symbol, day_of_month, month_of_year, year)
            )
        ''')

        # Weekly temporal analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_temporal_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                week_of_year INTEGER, -- 1-52
                year INTEGER,          -- Year
                total_trades INTEGER,
                profitable_trades INTEGER,
                avg_profit REAL,
                win_rate REAL,
                avg_volatility REAL,
                avg_spread_pips REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                last_updated DATETIME,
                UNIQUE(symbol, week_of_year, year)
            )
        ''')

        # Monthly temporal analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monthly_temporal_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                month_of_year INTEGER, -- 1-12
                year INTEGER,          -- Year
                total_trades INTEGER,
                profitable_trades INTEGER,
                avg_profit REAL,
                win_rate REAL,
                avg_volatility REAL,
                avg_spread_pips REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                last_updated DATETIME,
                UNIQUE(symbol, month_of_year, year)
            )
        ''')

        # Yearly temporal analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS yearly_temporal_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                year INTEGER,          -- Year
                total_trades INTEGER,
                profitable_trades INTEGER,
                avg_profit REAL,
                win_rate REAL,
                avg_volatility REAL,
                avg_spread_pips REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                last_updated DATETIME,
                UNIQUE(symbol, year)
            )
        ''')

        # Per-symbol SL/TP optimization table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbol_sl_tp_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE,
                optimal_sl_atr_multiplier REAL,
                optimal_tp_atr_multiplier REAL,
                optimal_rr_ratio REAL,  -- Risk-reward ratio
                avg_win_rate REAL,
                avg_profit_factor REAL,
                total_trades INTEGER,
                last_updated DATETIME,
                confidence_score REAL
            )
        ''')

        # Entry filter learning table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entry_filters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                filter_type TEXT,  -- 'time_filter', 'volatility_filter', etc.
                condition_value REAL,
                should_enter BOOLEAN,  -- Whether to enter when met
                total_trades INTEGER,
                profitable_trades INTEGER,
                win_rate REAL,
                last_updated DATETIME
            )
        ''')

        # Technical indicator optimization table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicator_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                indicator_name TEXT,  -- 'vwap', 'ema', 'rsi', 'atr', etc.
                parameter_name TEXT,  -- 'period', 'fast_period', etc.
                optimal_value REAL,
                performance_score REAL,  -- win_rate, profit_factor, etc.
                total_trades INTEGER,
                last_updated DATETIME,
                confidence_score REAL
            )
        ''')

        # Fundamental weight optimization table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundamental_weight_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT,  -- 'myfxbook', 'fxstreet', 'fxblue', etc.
                optimal_weight REAL,
                prediction_accuracy REAL,
                total_predictions INTEGER,
                last_updated DATETIME,
                market_condition TEXT  -- 'trending', 'ranging', etc.
            )
        ''')

        # Economic calendar impact table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_calendar_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT,
                event_impact TEXT,  -- 'high', 'medium', 'low'
                hours_before_event INTEGER,
                hours_after_event INTEGER,
                avg_trade_performance REAL,
                total_trades INTEGER,
                should_avoid_trading BOOLEAN,
                last_updated DATETIME,
                currency_pair TEXT
            )
        ''')

        # Interest rate impact table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interest_rate_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                currency TEXT,
                rate_change REAL,  -- percentage change
                time_horizon TEXT,  -- '1h', '4h', '1d', '1w'
                avg_price_movement REAL,
                total_observations INTEGER,
                correlation_strength REAL,
                last_updated DATETIME
            )
        ''')

        # Sentiment parameter optimization table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_parameter_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parameter_name TEXT,  -- 'sentiment_threshold', etc.
                optimal_value REAL,
                performance_impact REAL,
                total_trades INTEGER,
                last_updated DATETIME,
                market_condition TEXT
            )
        ''')

        # Position adjustment tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                old_sl REAL,
                old_tp REAL,
                new_sl REAL,
                new_tp REAL,
                adjustment_reason TEXT,
                adjustment_timestamp TEXT NOT NULL
            )
        ''')

        # Adjustment performance analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adjustment_performance_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date TEXT NOT NULL,
                success_rate REAL NOT NULL,
                total_adjustments INTEGER NOT NULL,
                successful_adjustments INTEGER NOT NULL,
                avg_profit_impact REAL
            )
        ''')

        # Daily trade counts for persistent risk management
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_trade_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                trade_count INTEGER NOT NULL DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, trade_date)
            )
        ''')

        # Regime parameter optimization table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regime_parameter_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                regime_type TEXT,
                parameter_name TEXT,
                optimal_value REAL,
                performance_score REAL,
                total_trades INTEGER,
                last_updated DATETIME,
                confidence_score REAL,
                UNIQUE(symbol, regime_type, parameter_name)
            )
        ''')

        # Analyzer accuracy tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyzer_accuracy_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                analyzer_type TEXT,
                accuracy_score REAL,
                total_evaluations INTEGER,
                market_regime TEXT,
                last_updated DATETIME
            )
        ''')

        # Forced closure analysis table for learning from time-based exits
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forced_closure_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                duration_minutes INTEGER,
                profit_pct REAL,
                closure_reason TEXT,
                count INTEGER DEFAULT 1,
                last_updated DATETIME,
                UNIQUE(symbol, duration_minutes)
            )
        ''')

        # Add new columns to trades table if they don't exist (for backward compatibility)
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN closure_reason TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN forced_closure BOOLEAN DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add new columns to symbol_holding_times table for forced closure analysis
        try:
            cursor.execute("ALTER TABLE symbol_holding_times ADD COLUMN natural_trades INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE symbol_holding_times ADD COLUMN forced_trades INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE symbol_holding_times ADD COLUMN forced_closure_rate_best_bucket REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE symbol_holding_times ADD COLUMN forced_closure_rates TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        conn.commit()
        conn.close()

    def load_daily_trade_counts(self) -> dict:
        """Load daily trade counts from database for risk management persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get today's date in YYYY-MM-DD format
            today = datetime.now().strftime('%Y-%m-%d')

            # Load all daily trade counts, but only keep today's data
            cursor.execute('''
                SELECT symbol, trade_date, trade_count
                FROM daily_trade_counts
                WHERE trade_date = ?
            ''', (today,))

            daily_counts = {}
            for row in cursor.fetchall():
                symbol, trade_date, count = row
                daily_counts[symbol] = {'date': trade_date, 'count': count}

            conn.close()
            logger.info(f"Loaded {len(daily_counts)} daily trade counts from database")
            return daily_counts

        except Exception as e:
            logger.error(f"Error loading daily trade counts: {e}")
            return {}

    def save_daily_trade_count(self, symbol: str, trade_date: str, count: int):
        """Save daily trade count to database"""
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

    def record_trade(self, trade_data: dict):
        """Record a completed trade in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO trades (
                    timestamp, symbol, direction, entry_price, exit_price,
                    volume, profit, profit_pct, signal_strength, ml_score,
                    technical_score, sentiment_score, duration_minutes, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('timestamp', datetime.now()),
                trade_data.get('symbol'),
                trade_data.get('direction'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('volume', 0.01),
                trade_data.get('profit', 0),
                trade_data.get('profit_pct', 0),
                trade_data.get('signal_strength', 0),
                trade_data.get('ml_score', 0),
                trade_data.get('technical_score', 0),
                trade_data.get('sentiment_score', 0),
                trade_data.get('duration_minutes', 0),
                trade_data.get('model_version', 'unknown')
            ))

            conn.commit()
            conn.close()
            logger.debug(f"Recorded trade for {trade_data.get('symbol')}")

        except Exception as e:
            logger.error(f"Error recording trade: {e}")

    def record_trade_closure(self, ticket: int, reason: str, entry_price: float, exit_price: float, symbol: str = None):
        """Record trade closure information for learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate profit/loss
            profit = exit_price - entry_price if entry_price and exit_price else 0
            profit_pct = (profit / entry_price * 100) if entry_price and entry_price != 0 else 0

            # Update the trade record with closure information
            cursor.execute('''
                UPDATE trades
                SET exit_price = ?, profit = ?, profit_pct = ?, closure_reason = ?
                WHERE id = (SELECT MAX(id) FROM trades WHERE symbol = ?)
            ''', (exit_price, profit, profit_pct, reason, symbol))

            conn.commit()
            conn.close()
            logger.debug(f"Recorded trade closure for ticket {ticket}, reason: {reason}")

        except Exception as e:
            logger.error(f"Error recording trade closure: {e}")

    def record_position_adjustment(self, ticket: int, symbol: str, old_sl: float, old_tp: float, new_sl: float, new_tp: float, reason: str):
        """Record position SL/TP adjustment for learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert position adjustment record
            cursor.execute('''
                INSERT INTO position_adjustments 
                (ticket, symbol, old_sl, old_tp, new_sl, new_tp, adjustment_reason, adjustment_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (ticket, symbol, old_sl, old_tp, new_sl, new_tp, reason))

            conn.commit()
            conn.close()
            logger.debug(f"Recorded position adjustment for ticket {ticket}: SL {old_sl:.5f}→{new_sl:.5f}, TP {old_tp:.5f}→{new_tp:.5f}")

        except Exception as e:
            logger.error(f"Error recording position adjustment: {e}")

    def get_recent_trades(self, symbol: str, days: int) -> pd.DataFrame:
        """Get recent trades for a symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT * FROM trades
                WHERE symbol = ? AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days)

            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            return df

        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {e}")
            return pd.DataFrame()

    def save_signal_weights(self):
        """Placeholder for signal weights saving - to be implemented"""
        pass

    def _save_regime_parameters(self, symbol: str, regime: str, params: dict, performance_score: float):
        """Save regime-specific parameters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for param_name, param_value in params.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO regime_parameter_optimization
                    (symbol, regime_type, parameter_name, optimal_value, performance_score, total_trades, last_updated, confidence_score)
                    VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP, ?)
                ''', (symbol, regime, param_name, param_value, performance_score, 0.5))

            conn.commit()
            conn.close()
            logger.debug(f"Saved regime parameters for {symbol} in {regime} regime")

        except Exception as e:
            logger.error(f"Error saving regime parameters: {e}")

    def _load_regime_parameters(self, symbol: str, regime: str) -> Dict:
        """Load regime-specific parameters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT parameter_name, optimal_value, performance_score
                FROM regime_parameter_optimization
                WHERE symbol = ? AND regime_type = ?
            ''', (symbol, regime))

            params = {}
            for row in cursor.fetchall():
                param_name, optimal_value, performance_score = row
                params[param_name] = optimal_value

            conn.close()
            return params

        except Exception as e:
            logger.error(f"Error loading regime parameters: {e}")
            return {}

    def _save_log_learning_state(self):
        """Placeholder for log learning state saving"""
        pass

    def analyze_trade_timing_patterns(self, trade_data: dict):
        """
        Analyze timing patterns for trades that close naturally (not SL/TP).
        Learns optimal entry and exit timing when stop loss and take profit are not reached.
        """
        try:
            # Determine if this was a natural trade closure
            # Natural trades are those that didn't hit SL/TP and weren't forced closed by time limits
            is_natural_closure = self._is_natural_trade_closure(trade_data)

            if not is_natural_closure:
                return  # Skip timing analysis for forced closures

            # Extract timing information
            entry_timestamp = trade_data['timestamp']
            if isinstance(entry_timestamp, str):
                entry_time = datetime.fromisoformat(entry_timestamp.replace('Z', '+00:00'))
            else:
                entry_time = entry_timestamp

            exit_time = datetime.now()  # Approximate exit time
            duration_minutes = trade_data.get('duration_minutes', 0)

            # Calculate exit time more accurately
            if duration_minutes > 0:
                exit_time = entry_time + timedelta(minutes=duration_minutes)

            # Extract timing features
            hour_of_day = entry_time.hour  # 0-23
            day_of_week = entry_time.weekday()  # 0-6 (Monday-Sunday)

            # Get market conditions at entry (if available)
            market_volatility = trade_data.get('volatility', 0.001)  # ATR-based
            spread_pips = trade_data.get('spread_pips', 0.1)  # Spread at entry

            # Trade outcome
            profit_pct = trade_data.get('profit_pct', 0)
            is_profitable = profit_pct > 0

            # Store timing analysis in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if entry timing record exists
            cursor.execute('''
                SELECT total_trades, profitable_trades, avg_profit
                FROM entry_timing_analysis
                WHERE symbol = ? AND hour_of_day = ? AND day_of_week = ?
            ''', (trade_data['symbol'], hour_of_day, day_of_week))

            existing_record = cursor.fetchone()

            if existing_record:
                # Update existing record
                total_trades, profitable_trades, avg_profit = existing_record
                new_total_trades = total_trades + 1
                new_profitable_trades = profitable_trades + (1 if is_profitable else 0)
                new_avg_profit = ((avg_profit * total_trades) + profit_pct) / new_total_trades
                new_win_rate = new_profitable_trades / new_total_trades

                cursor.execute('''
                    UPDATE entry_timing_analysis
                    SET total_trades = ?, profitable_trades = ?, avg_profit = ?,
                        win_rate = ?, market_volatility = ?, spread_pips = ?, last_updated = ?
                    WHERE symbol = ? AND hour_of_day = ? AND day_of_week = ?
                ''', (
                    new_total_trades, new_profitable_trades, new_avg_profit, new_win_rate,
                    market_volatility, spread_pips, datetime.now(),
                    trade_data['symbol'], hour_of_day, day_of_week
                ))
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO entry_timing_analysis
                    (symbol, hour_of_day, day_of_week, market_volatility, spread_pips,
                     total_trades, profitable_trades, avg_profit, win_rate, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data['symbol'], hour_of_day, day_of_week, market_volatility, spread_pips,
                    1, 1 if is_profitable else 0, profit_pct, 1.0 if is_profitable else 0.0, datetime.now()
                ))

            # Also analyze exit timing patterns
            exit_hour = exit_time.hour
            exit_day_of_week = exit_time.weekday()

            # Store exit timing analysis (similar structure but for exit patterns)
            cursor.execute('''
                SELECT total_trades, profitable_trades, avg_profit
                FROM entry_timing_analysis
                WHERE symbol = ? AND hour_of_day = ? AND day_of_week = ?
            ''', (trade_data['symbol'], exit_hour, exit_day_of_week))

            exit_record = cursor.fetchone()

            if exit_record:
                # Update exit timing record
                total_trades, profitable_trades, avg_profit = exit_record
                new_total_trades = total_trades + 1
                new_profitable_trades = profitable_trades + (1 if is_profitable else 0)
                new_avg_profit = ((avg_profit * total_trades) + profit_pct) / new_total_trades
                new_win_rate = new_profitable_trades / new_total_trades

                cursor.execute('''
                    UPDATE entry_timing_analysis
                    SET total_trades = ?, profitable_trades = ?, avg_profit = ?,
                        win_rate = ?, last_updated = ?
                    WHERE symbol = ? AND hour_of_day = ? AND day_of_week = ?
                ''', (
                    new_total_trades, new_profitable_trades, new_avg_profit, new_win_rate, datetime.now(),
                    trade_data['symbol'], exit_hour, exit_day_of_week
                ))
            else:
                # Insert new exit timing record
                cursor.execute('''
                    INSERT INTO entry_timing_analysis
                    (symbol, hour_of_day, day_of_week, total_trades, profitable_trades,
                     avg_profit, win_rate, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data['symbol'], exit_hour, exit_day_of_week,
                    1, 1 if is_profitable else 0, profit_pct, 1.0 if is_profitable else 0.0, datetime.now()
                ))

            conn.commit()
            conn.close()

            logger.debug(f"Analyzed timing patterns for {trade_data['symbol']} natural trade: "
                        f"entry_hour={hour_of_day}, exit_hour={exit_hour}, profitable={is_profitable}")

        except Exception as e:
            logger.error(f"Error analyzing trade timing patterns: {e}")

    def _is_natural_trade_closure(self, trade_data: dict) -> bool:
        """
        Determine if a trade closed naturally (not by SL/TP or forced time closure).

        Returns True if the trade closed due to market movement without hitting predefined limits.
        """
        try:
            # Check if trade was forced closed by time limits
            # This would be indicated by specific closure reasons in the trade data
            closure_reason = trade_data.get('closure_reason', 'natural_exit')

            if closure_reason in ['market_close_buffer', 'weekend_closure', 'max_time', 'optimal_time']:
                return False  # Forced closure

            # Check if trade hit stop loss or take profit
            # This is harder to determine without exact SL/TP levels, but we can use heuristics

            # For now, assume trades that close with moderate profit/loss are natural
            # Trades with extreme losses might have hit SL, extreme gains might have hit TP
            profit_pct = abs(trade_data.get('profit_pct', 0))

            # If profit/loss is extremely high, likely hit SL/TP
            if profit_pct > 2.0:  # More than 2% gain/loss likely hit TP/SL
                return False

            # If duration is very short (< 5 minutes), might be noise, not natural
            duration_minutes = trade_data.get('duration_minutes', 0)
            if duration_minutes < 5:
                return False

            return True  # Assume natural closure

        except Exception as e:
            logger.error(f"Error determining natural trade closure: {e}")
            return False

    def get_optimal_entry_timing(self, symbol: str) -> dict:
        """
        Get optimal entry timing recommendations for a symbol based on historical analysis.

        Returns:
            dict: Timing recommendations with confidence scores
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get current time
            now = datetime.now()
            current_hour = now.hour
            current_day = now.weekday()

            # Find best entry hours for this symbol
            cursor.execute('''
                SELECT hour_of_day, day_of_week, win_rate, avg_profit, total_trades
                FROM entry_timing_analysis
                WHERE symbol = ? AND total_trades >= 5
                ORDER BY win_rate DESC, avg_profit DESC
                LIMIT 5
            ''', (symbol,))

            best_timings = cursor.fetchall()

            # Get current timing performance
            cursor.execute('''
                SELECT win_rate, avg_profit, total_trades
                FROM entry_timing_analysis
                WHERE symbol = ? AND hour_of_day = ? AND day_of_week = ?
            ''', (symbol, current_hour, current_day))

            current_timing = cursor.fetchone()

            conn.close()

            # Calculate timing score
            timing_score = 0.5  # Neutral default

            if current_timing and current_timing[2] >= 5:  # At least 5 trades
                current_win_rate = current_timing[0]
                current_avg_profit = current_timing[1]

                # Compare to best timings
                if best_timings:
                    best_win_rate = best_timings[0][2]  # Best win rate
                    best_avg_profit = best_timings[0][3]  # Best avg profit

                    # Calculate relative performance
                    win_rate_score = min(current_win_rate / best_win_rate, 1.0) if best_win_rate > 0 else 0.5
                    profit_score = min(abs(current_avg_profit) / abs(best_avg_profit), 1.0) if best_avg_profit != 0 else 0.5

                    timing_score = (win_rate_score + profit_score) / 2
                else:
                    # No historical data, use current performance
                    timing_score = min(current_win_rate, 1.0)

            return {
                'timing_score': timing_score,
                'current_hour': current_hour,
                'current_day': current_day,
                'recommended': timing_score > 0.6,  # Recommend if score > 60%
                'confidence': min(current_timing[2] / 20, 1.0) if current_timing else 0.0
            }

        except Exception as e:
            logger.error(f"Error getting optimal entry timing: {e}")
            return {
                'timing_score': 0.5,
                'recommended': True,  # Default to allowing trades
                'confidence': 0.0
            }

    def record_analyzer_accuracy(self, symbol: str, analyzer_type: str, accuracy_score: float, market_regime: str = None):
        """Record analyzer accuracy for performance tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert or update analyzer accuracy
            cursor.execute('''
                INSERT INTO analyzer_accuracy_tracking
                (timestamp, symbol, analyzer_type, accuracy_score, total_evaluations, market_regime, last_updated)
                VALUES (?, ?, ?, ?, 1, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol, analyzer_type) DO UPDATE SET
                    accuracy_score = (accuracy_score * total_evaluations + ?) / (total_evaluations + 1),
                    total_evaluations = total_evaluations + 1,
                    market_regime = ?,
                    last_updated = CURRENT_TIMESTAMP
            ''', (
                datetime.now(),
                symbol,
                analyzer_type,
                accuracy_score,
                market_regime or 'unknown',
                accuracy_score,
                market_regime or 'unknown'
            ))

            conn.commit()
            conn.close()

            logger.debug(f"Recorded {analyzer_type} accuracy for {symbol}: {accuracy_score:.3f}")

        except Exception as e:
            logger.error(f"Error recording analyzer accuracy: {e}")

    def record_model_performance(self, symbol: str, metrics: dict):
        """Record model performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO model_performance (
                    timestamp, symbol, model_type, accuracy, precision,
                    recall, sharpe_ratio, max_drawdown, win_rate, avg_profit, total_trades
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                symbol,
                'ensemble',
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('win_rate', 0),
                metrics.get('avg_profit', 0),
                metrics.get('total_trades', 0)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error recording model performance: {e}")

    def record_parameter_change(self, param_name: str, old_value: float, new_value: float, score: float):
        """Record parameter optimization"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            improvement = ((new_value - old_value) / old_value * 100) if old_value != 0 else 0

            cursor.execute('''
                INSERT INTO parameter_optimization (
                    timestamp, parameter_name, old_value, new_value,
                    improvement_pct, validation_score
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                param_name,
                old_value,
                new_value,
                improvement,
                score
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error recording parameter change: {e}")

    def clean_old_data(self):
        """Clean old data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Keep only last 5 years of trades (1825 days)
            cursor.execute('''
                DELETE FROM trades
                WHERE timestamp < datetime('now', '-1825 days')
            ''')

            # Clean old model performance data (keep last 2 years)
            cursor.execute('''
                DELETE FROM model_performance
                WHERE timestamp < datetime('now', '-730 days')
            ''')

            # Clean old parameter optimization data (keep last 1 year)
            cursor.execute('''
                DELETE FROM parameter_optimization
                WHERE timestamp < datetime('now', '-365 days')
            ''')

            conn.commit()
            conn.close()

            logger.info("Cleaned old data from database")

        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")