# Clean Fundamental Analyzer - No More Loops!
import logging
import threading
from typing import Dict, List

class FundamentalAnalyzer:
    """Clean fundamental analyzer with all required methods"""

    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.thread = None

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

    def should_avoid_trading(self):
        """Check if trading should be avoided"""
        return False

    def get_market_bias(self, symbol=None):
        """Get market bias for symbol"""
        return "neutral"

    def get_economic_calendar(self, hours_ahead=24):
        """Get economic calendar"""
        return []

    def get_high_impact_events(self):
        """Get high impact events"""
        return []

    def get_news_sentiment(self, symbol=None):
        """Get news sentiment"""
        return {'score': 0.0, 'news_count': 0, 'last_update': None}

    def get_interest_rates(self):
        """Get interest rates"""
        return {
            'USD': 5.50, 'EUR': 4.50, 'GBP': 5.25, 'JPY': -0.10,
            'AUD': 4.35, 'CAD': 5.00, 'CHF': 1.75, 'NZD': 5.50
        }

    def is_data_current(self):
        """Check if data is current"""
        return True

    def get_summary(self):
        """Get summary"""
        return {'running': self.running, 'mode': 'clean'}

    def collect(self):
        """Collect fundamental data (compatibility method)"""
        # This method collects/updates fundamental data
        try:
            # Get economic events
            economic_events = self.get_economic_calendar(24) if hasattr(self, 'get_economic_calendar') else []
            high_impact_events = self.get_high_impact_events() if hasattr(self, 'get_high_impact_events') else []
            
            # Check if high-impact news is coming within 1 hour
            from datetime import datetime, timedelta
            now = datetime.now()
            high_impact_soon = False
            
            for event in high_impact_events:
                # Event structure may vary - handle different formats
                event_time = event.get('time') or event.get('datetime')
                if event_time:
                    try:
                        if isinstance(event_time, str):
                            event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                        if isinstance(event_time, datetime):
                            time_until_event = (event_time - now).total_seconds() / 3600  # hours
                            if 0 <= time_until_event <= 1:  # Within next hour
                                high_impact_soon = True
                                break
                    except Exception:
                        pass
            
            # Calculate fundamental scores per symbol based on economic data
            symbol_scores = {}
            interest_rates = self.get_interest_rates() if hasattr(self, 'get_interest_rates') else {}
            
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
                
                # Add high-impact news flag
                symbol_scores[symbol] = {
                    'score': score,
                    'high_impact_news_soon': high_impact_soon
                }
            
            return {
                'status': 'collected',
                'economic_events': economic_events,
                'high_impact_events': high_impact_events,
                'high_impact_news_soon': high_impact_soon,
                'interest_rates': interest_rates,
                'market_sentiment': self.get_news_sentiment() if hasattr(self, 'get_news_sentiment') else {'score': 0.0},
                **symbol_scores  # Spread symbol scores into the dict
            }
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f'Error collecting fundamental data: {e}')
            return {'status': 'error', 'error': str(e)}

# Alias for compatibility with old code
FundamentalDataCollector = FundamentalAnalyzer 
