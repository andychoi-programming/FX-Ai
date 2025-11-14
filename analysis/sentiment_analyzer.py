"""
Sentiment Analyzer Module
Analyzes market sentiment from news, social media, and retail positioning
"""

import logging
import re
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from textblob import TextBlob

try:
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.getLogger(__name__).warning("TextBlob not available, using simplified sentiment analysis")

class SentimentAnalyzer:
    """Analyzes market sentiment from various sources"""

    def __init__(self, config: Dict):
        """
        Initialize sentiment analyzer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Sentiment tracking
        self.sentiment_history = {}
        self.retail_sentiment = {}

        # Data freshness tracking
        self.last_update = None

        # Keywords for sentiment analysis
        self.bullish_keywords = [
            'bullish', 'buy', 'long', 'up', 'rise', 'gain', 'strong', 'positive',
            'growth', 'recovery', 'optimism', 'confidence', 'rally'
        ]

        self.bearish_keywords = [
            'bearish', 'sell', 'short', 'down', 'fall', 'drop', 'weak', 'negative',
            'decline', 'recession', 'pessimism', 'fear', 'crash'
        ]

    def is_data_fresh(self, max_age_minutes: int = 60) -> bool:
        """
        Check if analyzer is ready for use

        Args:
            max_age_minutes: Maximum age of data in minutes (not used for sentiment analyzer)

        Returns:
            bool: True if analyzer is initialized and ready
        """
        # Sentiment analyzer is always "fresh" since it analyzes data on-demand
        # The freshness check is more relevant for real-time data feeds
        return True

    async def analyze_sentiment(self, symbol: str, news_data: Optional[List[Dict]] = None,
                               social_data: Optional[List[Dict]] = None) -> Dict:
        """
        Perform comprehensive sentiment analysis

        Args:
            symbol: Trading symbol
            news_data: News articles data
            social_data: Social media data

        Returns:
            dict: Sentiment analysis results
        """
        try:
            sentiment_scores = {}

            # Analyze news sentiment
            if news_data:
                sentiment_scores['news'] = self._analyze_news_sentiment(news_data)

            # Analyze social media sentiment
            if social_data:
                sentiment_scores['social'] = self._analyze_social_sentiment(social_data)

            # Get retail positioning (contrarian indicator)
            sentiment_scores['retail'] = self._get_retail_sentiment(symbol)

            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(sentiment_scores)

            # Store in history
            self.sentiment_history[symbol] = {
                'timestamp': datetime.now(),
                'scores': sentiment_scores,
                'overall': overall_sentiment
            }

            return {
                'overall_sentiment': overall_sentiment,
                'components': sentiment_scores,
                'retail_long_percentage': sentiment_scores['retail'].get('long_percentage', 50),
                'retail_short_percentage': sentiment_scores['retail'].get('short_percentage', 50),
                'sentiment_trend': self._calculate_sentiment_trend(symbol)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {}

    def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """
        Analyze sentiment from news articles

        Args:
            news_data: List of news articles

        Returns:
            dict: News sentiment analysis
        """
        try:
            if not news_data:
                return {'score': 0, 'confidence': 0}

            total_score = 0
            total_confidence = 0
            article_count = 0

            for article in news_data:
                title = article.get('title', '')
                content = article.get('content', '')

                # Analyze title (weighted more heavily)
                title_sentiment = self._analyze_text_sentiment(title)
                content_sentiment = self._analyze_text_sentiment(content)

                # Combine with weights
                combined_score = (title_sentiment['score'] * 0.7) + (content_sentiment['score'] * 0.3)
                combined_confidence = min(title_sentiment['confidence'], content_sentiment['confidence'])

                total_score += combined_score
                total_confidence += combined_confidence
                article_count += 1

            if article_count == 0:
                return {'score': 0, 'confidence': 0}

            return {
                'score': total_score / article_count,
                'confidence': total_confidence / article_count,
                'article_count': article_count
            }

        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return {'score': 0, 'confidence': 0}

    def _analyze_social_sentiment(self, social_data: List[Dict]) -> Dict:
        """
        Analyze sentiment from social media data

        Args:
            social_data: List of social media posts

        Returns:
            dict: Social sentiment analysis
        """
        try:
            if not social_data:
                return {'score': 0, 'volume': 0}

            total_score = 0
            total_volume = 0

            for post in social_data:
                text = post.get('text', '')
                sentiment = self._analyze_text_sentiment(text)
                weight = post.get('engagement', 1)  # Likes, retweets, etc.

                total_score += sentiment['score'] * weight
                total_volume += weight

            if total_volume == 0:
                return {'score': 0, 'volume': 0}

            return {
                'score': total_score / total_volume,
                'volume': total_volume
            }

        except Exception as e:
            self.logger.error(f"Error analyzing social sentiment: {e}")
            return {'score': 0, 'volume': 0}

    def _analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using TextBlob or fallback

        Args:
            text: Text to analyze

        Returns:
            dict: Sentiment analysis
        """
        try:
            if not text:
                return {'score': 0, 'confidence': 0}

            # Clean text
            text = self._clean_text(text)

            if TEXTBLOB_AVAILABLE:
                # Use TextBlob for sentiment analysis
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # type: ignore
                subjectivity = blob.sentiment.subjectivity  # type: ignore

                # Enhance with keyword analysis
                keyword_score = self._keyword_sentiment_analysis(text)

                # Combine scores
                combined_score = (polarity * 0.7) + (keyword_score * 0.3)

                return {
                    'score': combined_score,
                    'confidence': subjectivity,  # Subjectivity as confidence measure
                    'polarity': polarity,
                    'keyword_score': keyword_score
                }
            else:
                # Fallback to keyword-based analysis only
                keyword_score = self._keyword_sentiment_analysis(text)
                return {
                    'score': keyword_score,
                    'confidence': 0.5,  # Fixed confidence for keyword-only analysis
                    'polarity': keyword_score,
                    'keyword_score': keyword_score
                }

        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {e}")
            return {'score': 0, 'confidence': 0}

    def _keyword_sentiment_analysis(self, text: str) -> float:
        """
        Analyze sentiment based on bullish/bearish keywords

        Args:
            text: Text to analyze

        Returns:
            float: Keyword-based sentiment score (-1 to 1)
        """
        try:
            text_lower = text.lower()

            bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
            bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)

            total_keywords = bullish_count + bearish_count

            if total_keywords == 0:
                return 0

            # Calculate score
            score = (bullish_count - bearish_count) / total_keywords
            return max(-1, min(1, score))

        except Exception as e:
            return 0

    def _clean_text(self, text: str) -> str:
        """
        Clean text for sentiment analysis

        Args:
            text: Raw text

        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _get_retail_sentiment(self, symbol: str) -> Dict:
        """
        Get retail trader positioning (contrarian indicator)

        Args:
            symbol: Trading symbol

        Returns:
            dict: Retail sentiment data
        """
        try:
            # Mock retail sentiment data
            # In real implementation, this would come from brokers' sentiment data
            base_sentiment = {
                'EURUSD': {'long': 65, 'short': 35},
                'GBPUSD': {'long': 45, 'short': 55},
                'USDJPY': {'long': 50, 'short': 50},
                'AUDUSD': {'long': 70, 'short': 30},
                'USDCAD': {'long': 40, 'short': 60}
            }

            sentiment = base_sentiment.get(symbol, {'long': 50, 'short': 50})

            # Calculate contrarian score (0-1 scale)
            # When retail is heavily long (>60%), score tends toward bearish (0.3)
            # When retail is heavily short (<40%), score tends toward bullish (0.7)
            # Neutral at 50%
            long_pct = sentiment['long']
            if long_pct > 60:
                contrarian_score = 0.3  # Bearish contrarian signal
            elif long_pct < 40:
                contrarian_score = 0.7  # Bullish contrarian signal
            else:
                contrarian_score = 0.5  # Neutral

            return {
                'long_percentage': sentiment['long'],
                'short_percentage': sentiment['short'],
                'score': contrarian_score,  # Add score for overall calculation
                'contrarian_signal': 'buy' if sentiment['long'] < 40 else ('sell' if sentiment['long'] > 60 else 'neutral')
            }

        except Exception as e:
            self.logger.error(f"Error getting retail sentiment for {symbol}: {e}")
            return {'long_percentage': 50, 'short_percentage': 50}

    def _calculate_overall_sentiment(self, sentiment_scores: Dict) -> float:
        """
        Calculate overall sentiment score

        Args:
            sentiment_scores: Individual sentiment scores

        Returns:
            float: Overall sentiment (-1 to 1)
        """
        try:
            weights = {
                'news': 0.4,
                'social': 0.3,
                'retail': 0.3
            }

            overall_score = 0
            total_weight = 0

            for source, score_data in sentiment_scores.items():
                if source in weights:
                    if isinstance(score_data, dict):
                        score = score_data.get('score', 0)
                    else:
                        score = score_data

                    overall_score += score * weights[source]
                    total_weight += weights[source]

            return overall_score / total_weight if total_weight > 0 else 0

        except Exception as e:
            self.logger.error(f"Error calculating overall sentiment: {e}")
            return 0

    def _calculate_sentiment_trend(self, symbol: str) -> str:
        """
        Calculate sentiment trend over time

        Args:
            symbol: Trading symbol

        Returns:
            str: Sentiment trend
        """
        try:
            history = self.sentiment_history.get(symbol, [])
            if len(history) < 2:
                return 'stable'

            recent_scores = [h['overall'] for h in history[-5:]]  # Last 5 readings

            if len(recent_scores) < 2:
                return 'stable'

            # Calculate trend
            first_half = np.mean(recent_scores[:len(recent_scores)//2])
            second_half = np.mean(recent_scores[len(recent_scores)//2:])

            diff = second_half - first_half

            if diff > 0.1:
                return 'improving'
            elif diff < -0.1:
                return 'deteriorating'
            else:
                return 'stable'

        except Exception as e:
            return 'stable'
    
    def stop(self):
        """Stop the sentiment analyzer"""
        self.logger.info("SentimentAnalyzer stopped")
    
    def start(self):
        """Start the sentiment analyzer"""
        self.logger.info("SentimentAnalyzer started")

    def get_sl_tp_adjustments(self, symbol: str, base_sl_pips: float, base_tp_pips: float,
                             sentiment_result: Dict) -> Dict:
        """
        Get SL/TP adjustments based on sentiment analysis

        Args:
            symbol: Trading symbol
            base_sl_pips: Base stop loss in pips
            base_tp_pips: Base take profit in pips
            sentiment_result: Sentiment analysis results

        Returns:
            dict: Adjusted SL/TP values with reason
        """
        try:
            adjustments = {
                'sl_pips': base_sl_pips,
                'tp_pips': base_tp_pips,
                'reason': 'base_values'
            }

            # Get sentiment score
            sentiment_score = sentiment_result.get('overall_score', 0.5)

            # Strong bullish sentiment: extend TP, maintain SL
            if sentiment_score > 0.7:
                adjustments['tp_pips'] = base_tp_pips * 1.2  # Extend TP by 20%
                adjustments['reason'] = 'strong_bullish_sentiment'

            # Strong bearish sentiment: extend TP, maintain SL
            elif sentiment_score < 0.3:
                adjustments['tp_pips'] = base_tp_pips * 1.2  # Extend TP by 20%
                adjustments['reason'] = 'strong_bearish_sentiment'

            # Neutral sentiment: slightly tighten both (reduced from 5% to 2%)
            elif 0.4 <= sentiment_score <= 0.6:
                adjustments['sl_pips'] = base_sl_pips * 0.98  # Tighten SL by 2%
                adjustments['tp_pips'] = base_tp_pips * 0.98  # Tighten TP by 2%
                adjustments['reason'] = 'neutral_sentiment'

            return adjustments

        except Exception as e:
            self.logger.error(f"Error getting sentiment SL/TP adjustments for {symbol}: {e}")
            return {
                'sl_pips': base_sl_pips,
                'tp_pips': base_tp_pips,
                'reason': 'error_fallback'
            }

    def analyze(self, symbol: str, market_data: Optional[Dict] = None) -> float:
        """
        Synchronous analyze method for trading orchestrator compatibility

        Args:
            symbol: Trading symbol
            market_data: Market data (not used for sentiment analysis)

        Returns:
            float: Sentiment score between 0 and 1
        """
        try:
            # Use asyncio.create_task for non-blocking sentiment updates if in async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a task for async execution and return cached/default value
                    asyncio.create_task(self._async_analyze_sentiment(symbol))
                    # Return last cached sentiment or neutral
                    return getattr(self, '_last_sentiment', {}).get(symbol, 0.5)
            except RuntimeError:
                # No event loop, create one for synchronous execution
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the async analysis with timeout protection
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(loop.run_until_complete, self._analyze_with_timeout(symbol))
                result = future.result(timeout=30)  # 30 second timeout

            # Extract the overall sentiment score
            overall_sentiment = result.get('overall_sentiment', 0.5)

            # Cache the result
            if not hasattr(self, '_last_sentiment'):
                self._last_sentiment = {}
            self._last_sentiment[symbol] = overall_sentiment

            # Convert to 0-1 scale if needed
            if isinstance(overall_sentiment, str):
                # Convert sentiment strings to numeric
                sentiment_map = {'bullish': 0.7, 'neutral': 0.5, 'bearish': 0.3}
                return sentiment_map.get(overall_sentiment.lower(), 0.5)
            elif isinstance(overall_sentiment, (int, float)):
                # Ensure it's in 0-1 range
                return max(0.0, min(1.0, float(overall_sentiment)))
            else:
                return 0.5

        except concurrent.futures.TimeoutError:
            self.logger.warning(f"Sentiment analysis timeout for {symbol} - using cached/default value")
            return getattr(self, '_last_sentiment', {}).get(symbol, 0.5)
        except Exception as e:
            self.logger.error(f"Error in synchronous sentiment analysis for {symbol}: {e}")
            return 0.5  # Return neutral sentiment on error

    async def _analyze_with_timeout(self, symbol: str) -> Dict:
        """Analyze sentiment with timeout protection"""
        try:
            return await asyncio.wait_for(self.analyze_sentiment(symbol), timeout=25.0)
        except asyncio.TimeoutError:
            self.logger.warning(f"Sentiment analysis timed out for {symbol}")
            return {'overall_sentiment': 0.5, 'error': 'timeout'}
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis for {symbol}: {e}")
            return {'overall_sentiment': 0.5, 'error': str(e)}

    async def _async_analyze_sentiment(self, symbol: str):
        """Async sentiment analysis for background updates"""
        try:
            result = await self._analyze_with_timeout(symbol)
            overall_sentiment = result.get('overall_sentiment', 0.5)

            # Cache the result
            if not hasattr(self, '_last_sentiment'):
                self._last_sentiment = {}
            self._last_sentiment[symbol] = overall_sentiment

            self.logger.debug(f"Background sentiment updated for {symbol}: {overall_sentiment}")
        except Exception as e:
            self.logger.error(f"Error in background sentiment analysis for {symbol}: {e}")