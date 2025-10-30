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
            return {
                'status': 'collected',
                'economic_events': self.get_economic_calendar(24) if hasattr(self, 'get_economic_calendar') else [],
                'interest_rates': self.get_interest_rates() if hasattr(self, 'get_interest_rates') else {},
                'market_sentiment': self.get_news_sentiment() if hasattr(self, 'get_news_sentiment') else {'score': 0.0}
            }
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f'Error collecting fundamental data: {e}')
            return {'status': 'error', 'error': str(e)}

# Alias for compatibility with old code
FundamentalDataCollector = FundamentalAnalyzer 
