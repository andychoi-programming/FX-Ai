# Real-time Fundamental Analyzer
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

class FundamentalAnalyzer:
    """Real-time fundamental analyzer for economic data and market analysis"""

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.thread = None

        # Data storage
        self.economic_data = {}
        self.last_update = None
        self.update_interval = self.config.get('fundamental_update_interval', 3600)  # 1 hour

        # Economic indicators that affect forex markets
        self.key_indicators = [
            'GDP', 'CPI', 'PPI', 'Unemployment', 'Retail Sales',
            'Industrial Production', 'Housing Starts', 'Trade Balance',
            'Interest Rates', 'Consumer Confidence', 'PMI'
        ]

        # Currency impact weights for different indicators
        self.currency_weights = {
            'USD': {'GDP': 0.9, 'Unemployment': 0.8, 'CPI': 0.7, 'Fed Rate': 1.0},
            'EUR': {'GDP': 0.8, 'CPI': 0.6, 'ECB Rate': 1.0, 'PMI': 0.7},
            'GBP': {'GDP': 0.8, 'CPI': 0.7, 'BOE Rate': 1.0, 'Unemployment': 0.6},
            'JPY': {'GDP': 0.6, 'CPI': 0.5, 'BOJ Rate': 1.0, 'Trade Balance': 0.8},
            'AUD': {'GDP': 0.7, 'Employment': 0.8, 'RBA Rate': 1.0, 'Commodity Prices': 0.9},
            'CAD': {'GDP': 0.7, 'Employment': 0.7, 'BOC Rate': 1.0, 'Oil Prices': 0.9},
            'CHF': {'CPI': 0.6, 'SNB Rate': 1.0, 'Trade Balance': 0.7},
            'NZD': {'GDP': 0.7, 'Employment': 0.8, 'RBNZ Rate': 1.0, 'Dairy Prices': 0.8}
        }

        # Risk adjustment factors based on economic events
        self.event_risk_multipliers = {
            'high_impact': 1.5,    # Increase SL by 50%
            'medium_impact': 1.2,  # Increase SL by 20%
            'low_impact': 1.1,     # Increase SL by 10%
            'no_impact': 1.0       # No change
        }

    def start(self):
        """Start the analyzer"""
        self.running = True
        self.logger.info("FundamentalAnalyzer started")
        return True

    def stop(self):
        """Stop the analyzer"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.logger.info("FundamentalAnalyzer stopped")
        return True

    def should_avoid_trading(self, symbol: Optional[str] = None) -> bool:
        """Check if trading should be avoided due to high-impact economic events"""
        try:
            high_impact_events = self.get_high_impact_events()

            if not high_impact_events:
                return False

            # Check for high-impact events in the next 2 hours
            now = datetime.now()
            two_hours_later = now + timedelta(hours=2)

            for event in high_impact_events:
                event_time = event.get('datetime') or event.get('time')
                if event_time:
                    try:
                        if isinstance(event_time, str):
                            event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))

                        # If event is within next 2 hours, avoid trading
                        if now <= event_time <= two_hours_later:
                            impact = event.get('impact', 'low')
                            if impact in ['high', 'medium']:
                                self.logger.warning(f"Avoiding trading due to {impact} impact event: {event.get('title', 'Unknown')}")
                                return True
                    except Exception as e:
                        self.logger.debug(f"Error parsing event time: {e}")
                        continue

            return False

        except Exception as e:
            self.logger.error(f"Error checking trading avoidance: {e}")
            return False

    def get_market_bias(self, symbol: Optional[str] = None) -> str:
        """Get market bias for symbol based on fundamental analysis"""
        try:
            if not symbol:
                return "neutral"

            # Calculate score directly to avoid recursion
            score = 0.5

            # Adjust based on interest rate differentials for forex pairs
            if len(symbol) == 6 and symbol not in ['XAUUSD', 'XAGUSD']:
                base_currency = symbol[:3]
                quote_currency = symbol[3:]

                interest_rates = self.get_interest_rates()
                base_rate = interest_rates.get(base_currency, 0)
                quote_rate = interest_rates.get(quote_currency, 0)
                rate_diff = base_rate - quote_rate

                # Normalize rate differential to 0-1 scale (typical range: -6 to +6)
                score = 0.5 + (rate_diff / 12.0)  # Normalize -6 to +6 -> 0 to 1
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            # Determine bias based on score
            if score > 0.7:
                return "bullish"
            elif score < 0.3:
                return "bearish"
            else:
                return "neutral"

        except Exception as e:
            self.logger.error(f"Error getting market bias for {symbol}: {e}")
            return "neutral"

    def get_economic_calendar(self, hours_ahead: int = 24) -> List[Dict]:
        """Get economic calendar for the next hours_ahead hours"""
        try:
            # In a real implementation, this would fetch from an API like Forex Factory
            # For now, return a simulated calendar with realistic events

            now = datetime.now()
            end_time = now + timedelta(hours=hours_ahead)

            # Simulated economic events (in production, fetch from API)
            simulated_events = [
                {
                    'title': 'US Non-Farm Payrolls',
                    'country': 'USD',
                    'datetime': (now + timedelta(hours=6)).isoformat(),
                    'impact': 'high',
                    'forecast': '200K',
                    'previous': '150K'
                },
                {
                    'title': 'EU CPI m/m',
                    'country': 'EUR',
                    'datetime': (now + timedelta(hours=12)).isoformat(),
                    'impact': 'high',
                    'forecast': '0.3%',
                    'previous': '0.4%'
                },
                {
                    'title': 'GBP GDP q/q',
                    'country': 'GBP',
                    'datetime': (now + timedelta(hours=18)).isoformat(),
                    'impact': 'medium',
                    'forecast': '0.2%',
                    'previous': '0.1%'
                },
                {
                    'title': 'CAD Employment Change',
                    'country': 'CAD',
                    'datetime': (now + timedelta(hours=22)).isoformat(),
                    'impact': 'medium',
                    'forecast': '20K',
                    'previous': '15K'
                }
            ]

            # Filter events within the time window
            upcoming_events = []
            for event in simulated_events:
                event_time = datetime.fromisoformat(event['datetime'])
                if now <= event_time <= end_time:
                    upcoming_events.append(event)

            return upcoming_events

        except Exception as e:
            self.logger.error(f"Error getting economic calendar: {e}")
            return []

    def get_high_impact_events(self) -> List[Dict]:
        """Get high impact economic events"""
        try:
            all_events = self.get_economic_calendar(48)  # Look ahead 48 hours
            high_impact_events = [
                event for event in all_events
                if event.get('impact') == 'high'
            ]
            return high_impact_events

        except Exception as e:
            self.logger.error(f"Error getting high impact events: {e}")
            return []

    def get_news_sentiment(self, symbol: Optional[str] = None) -> Dict:
        """Get news sentiment analysis"""
        try:
            # In a real implementation, this would analyze news feeds
            # For now, return simulated sentiment data

            # Simulate sentiment based on recent economic events
            base_sentiment = 0.5  # Neutral

            # Adjust based on recent high-impact events
            high_impact_events = self.get_high_impact_events()
            recent_events = [
                event for event in high_impact_events
                if self._is_recent_event(event, hours=24)
            ]

            if recent_events:
                # Recent high-impact events can create volatility
                base_sentiment = 0.6  # Slightly bullish due to event-driven volatility

            # Symbol-specific adjustments
            if symbol:
                if symbol in ['XAUUSD', 'XAGUSD']:
                    # Metals often react to USD sentiment
                    base_sentiment = 0.55
                elif 'JPY' in symbol:
                    # JPY often has different sentiment drivers
                    base_sentiment = 0.45

            return {
                'score': base_sentiment,
                'news_count': len(recent_events),
                'last_update': datetime.now().isoformat(),
                'sentiment': 'bullish' if base_sentiment > 0.6 else 'bearish' if base_sentiment < 0.4 else 'neutral'
            }

        except Exception as e:
            self.logger.error(f"Error getting news sentiment: {e}")
            return {'score': 0.5, 'news_count': 0, 'last_update': None}

    def get_interest_rates(self) -> Dict[str, float]:
        """Get current interest rates for major currencies"""
        try:
            # Current major central bank rates (as of late 2024/early 2025)
            # In a real implementation, this would fetch from financial APIs
            rates = {
                'USD': 4.50,  # Federal Reserve
                'EUR': 3.75,  # European Central Bank
                'GBP': 4.75,  # Bank of England
                'JPY': -0.10, # Bank of Japan
                'AUD': 4.35,  # Reserve Bank of Australia
                'CAD': 4.50,  # Bank of Canada
                'CHF': 1.50,  # Swiss National Bank
                'NZD': 5.25   # Reserve Bank of New Zealand
            }

            # Store for use in analysis
            self.economic_data['interest_rates'] = rates
            self.economic_data['rates_last_update'] = datetime.now()

            return rates

        except Exception as e:
            self.logger.error(f"Error getting interest rates: {e}")
            return {
                'USD': 4.50, 'EUR': 3.75, 'GBP': 4.75, 'JPY': -0.10,
                'AUD': 4.35, 'CAD': 4.50, 'CHF': 1.50, 'NZD': 5.25
            }

    def _is_recent_event(self, event: Dict, hours: int = 24) -> bool:
        """Check if an event occurred within the last N hours"""
        try:
            event_time = event.get('datetime') or event.get('time')
            if event_time:
                if isinstance(event_time, str):
                    event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))

                now = datetime.now()
                time_diff = (now - event_time).total_seconds() / 3600  # hours
                return 0 <= time_diff <= hours

            return False

        except Exception:
            return False

    def get_recent_events(self, minutes: int = 5) -> List[Dict]:
        """Get economic events that occurred within the last N minutes"""
        try:
            all_events = self.get_economic_calendar(hours_ahead=1)  # Get next hour's events
            recent_events = []

            for event in all_events:
                if self._is_recent_event(event, hours=minutes/60):  # Convert minutes to hours
                    recent_events.append(event)

            return recent_events

        except Exception as e:
            self.logger.error(f"Error getting recent events: {e}")
            return []

    def get_breaking_news(self, symbol: Optional[str] = None, minutes: int = 5) -> Dict:
        """Get breaking news analysis for ongoing trades"""
        try:
            recent_events = self.get_recent_events(minutes=minutes)

            # Filter for high-impact events
            high_impact_events = [e for e in recent_events if e.get('impact') in ['high', 'medium']]

            result = {
                'has_breaking_news': len(high_impact_events) > 0,
                'events': high_impact_events,
                'severity': 'high' if any(e.get('impact') == 'high' for e in high_impact_events) else 'medium' if high_impact_events else 'low',
                'direction': self._analyze_news_direction(high_impact_events, symbol),
                'recommendation': self._get_news_recommendation(high_impact_events, symbol)
            }

            return result

        except Exception as e:
            self.logger.error(f"Error getting breaking news: {e}")
            return {
                'has_breaking_news': False,
                'events': [],
                'severity': 'low',
                'direction': 'neutral',
                'recommendation': 'hold'
            }

    def _analyze_news_direction(self, events: List[Dict], symbol: Optional[str]) -> str:
        """Analyze if news direction favors or hurts the position"""
        if not events or not symbol:
            return 'neutral'

        # Simple analysis based on currency pairs
        base_currency = symbol[:3] if symbol else None
        quote_currency = symbol[3:] if symbol else None

        positive_events = 0
        negative_events = 0

        for event in events:
            currency = event.get('currency', '')
            if currency in [base_currency, quote_currency]:
                # This affects our pair
                if event.get('forecast') == 'better':
                    positive_events += 1
                elif event.get('forecast') == 'worse':
                    negative_events += 1

        if positive_events > negative_events:
            return 'favorable'
        elif negative_events > positive_events:
            return 'adverse'
        else:
            return 'neutral'

    def _get_news_recommendation(self, events: List[Dict], symbol: Optional[str]) -> str:
        """Get trading recommendation based on news events"""
        if not events:
            return 'hold'

        severity = 'high' if any(e.get('impact') == 'high' for e in events) else 'medium'
        direction = self._analyze_news_direction(events, symbol)

        if severity == 'high':
            if direction == 'adverse':
                return 'close_position'
            elif direction == 'favorable':
                return 'lock_profits'
            else:
                return 'tighten_stops'
        else:
            if direction == 'adverse':
                return 'tighten_stops'
            elif direction == 'favorable':
                return 'extend_targets'
            else:
                return 'monitor'

    def get_sl_tp_adjustments(self, symbol: str, base_sl_pips: float, base_tp_pips: float) -> Dict[str, Any]:
        """Get SL/TP adjustments based on fundamental analysis"""
        try:
            adjustments = {
                'sl_pips': base_sl_pips,
                'tp_pips': base_tp_pips,
                'reason': 'base_values'
            }

            # Check for upcoming high-impact events
            high_impact_events = self.get_high_impact_events()
            upcoming_high_impact = False

            now = datetime.now()
            next_4_hours = now + timedelta(hours=4)

            for event in high_impact_events:
                event_time = event.get('datetime') or event.get('time')
                if event_time:
                    try:
                        if isinstance(event_time, str):
                            event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))

                        if now <= event_time <= next_4_hours:
                            upcoming_high_impact = True
                            break
                    except Exception:
                        continue

            if upcoming_high_impact:
                # Increase SL by 50% for high-impact events
                adjustments['sl_pips'] = base_sl_pips * 1.5
                adjustments['reason'] = 'high_impact_event'
                self.logger.info(f"Adjusting SL for {symbol} due to upcoming high-impact event: {base_sl_pips} -> {adjustments['sl_pips']}")

            # Check market volatility based on interest rate differentials
            if len(symbol) == 6 and symbol not in ['XAUUSD', 'XAGUSD']:
                base_currency = symbol[:3]
                quote_currency = symbol[3:]

                rates = self.get_interest_rates()
                base_rate = rates.get(base_currency, 0)
                quote_rate = rates.get(quote_currency, 0)
                rate_diff = abs(base_rate - quote_rate)

                # High interest rate differential increases volatility
                if rate_diff > 2.0:  # More than 2% difference
                    adjustments['sl_pips'] = base_sl_pips * 1.2
                    adjustments['tp_pips'] = base_tp_pips * 1.1  # Slightly increase TP too
                    adjustments['reason'] = 'high_rate_volatility'

            return adjustments

        except Exception as e:
            self.logger.error(f"Error getting SL/TP adjustments for {symbol}: {e}")
            return {
                'sl_pips': base_sl_pips,
                'tp_pips': base_tp_pips,
                'reason': 'error_fallback'
            }

    def collect(self) -> Dict[str, Any]:
        """Collect fundamental data (compatibility method)"""
        try:
            # Get economic events
            economic_events = self.get_economic_calendar(24)
            high_impact_events = self.get_high_impact_events()

            # Check if high-impact news is coming within 1 hour
            now = datetime.now()
            high_impact_soon = False

            for event in high_impact_events:
                event_time = event.get('datetime') or event.get('time')
                if event_time:
                    try:
                        if isinstance(event_time, str):
                            event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                        time_until_event = (event_time - now).total_seconds() / 3600  # hours
                        if 0 <= time_until_event <= 1:  # Within next hour
                            high_impact_soon = True
                            break
                    except Exception:
                        pass

            # Calculate fundamental scores per symbol based on economic data
            symbol_scores = {}
            interest_rates = self.get_interest_rates()

            # Calculate fundamental score for each major currency pair
            currency_pairs = [
                'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
                'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY', 'EURAUD', 'EURCHF', 'AUDNZD',
                'NZDJPY', 'GBPAUD', 'GBPCAD', 'EURNZD', 'AUDCAD', 'GBPCHF', 'AUDCHF',
                'EURCAD', 'CADJPY', 'GBPNZD', 'CADCHF', 'CHFJPY', 'NZDCAD', 'NZDCHF',
                'XAUUSD', 'XAGUSD'
            ]

            for symbol in currency_pairs:
                # Default neutral score
                score = 0.5

                # Adjust based on interest rate differentials for forex pairs
                if len(symbol) == 6 and symbol not in ['XAUUSD', 'XAGUSD']:
                    base_currency = symbol[:3]
                    quote_currency = symbol[3:]

                    base_rate = interest_rates.get(base_currency, 0)
                    quote_rate = interest_rates.get(quote_currency, 0)
                    rate_diff = base_rate - quote_rate

                    # Normalize rate differential to 0-1 scale (typical range: -6 to +6)
                    score = 0.5 + (rate_diff / 12.0)  # Normalize -6 to +6 -> 0 to 1
                    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

                # Add high-impact news flag and bias
                # Calculate bias directly to avoid recursion
                bias = "neutral"
                if score > 0.7:
                    bias = "bullish"
                elif score < 0.3:
                    bias = "bearish"

                symbol_scores[symbol] = {
                    'score': score,
                    'high_impact_news_soon': high_impact_soon,
                    'bias': bias
                }

            return {
                'status': 'collected',
                'economic_events': economic_events,
                'high_impact_events': high_impact_events,
                'high_impact_news_soon': high_impact_soon,
                'interest_rates': interest_rates,
                'market_sentiment': self.get_news_sentiment(),
                **symbol_scores  # Spread symbol scores into the dict
            }
        except Exception as e:
            self.logger.error(f'Error collecting fundamental data: {e}')
            return {'status': 'error', 'error': str(e)}

    def is_data_current(self) -> bool:
        """Check if data is current"""
        if not self.last_update:
            return False

        time_since_update = (datetime.now() - self.last_update).total_seconds()
        return time_since_update < self.update_interval

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of analyzer state"""
        return {
            'running': self.running,
            'mode': 'real_time_fundamental',
            'data_current': self.is_data_current(),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'high_impact_events_count': len(self.get_high_impact_events()),
            'should_avoid_trading': self.should_avoid_trading()
        }

# Alias for compatibility with old code
FundamentalDataCollector = FundamentalAnalyzer 
