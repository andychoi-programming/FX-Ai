"""
Sentiment Analyzer Module
Analyzes market sentiment from news, social media, and retail positioning
"""

import logging
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from textblob import TextBlob
import asyncio

try:
    from textblob import TextBlob
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

        # Keywords for sentiment analysis
        self.bullish_keywords = [
            'bullish', 'buy', 'long', 'up', 'rise', 'gain', 'strong', 'positive',
            'growth', 'recovery', 'optimism', 'confidence', 'rally'
        ]

        self.bearish_keywords = [
            'bearish', 'sell', 'short', 'down', 'fall', 'drop', 'weak', 'negative',
            'decline', 'recession', 'pessimism', 'fear', 'crash'
        ]

    async def analyze_sentiment(self, symbol: str, news_data: List[Dict] = None,
                               social_data: List[Dict] = None) -> Dict:
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
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

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

            return {
                'long_percentage': sentiment['long'],
                'short_percentage': sentiment['short'],
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
    
    async def analyze(self):
        """Analyze sentiment for all symbols"""
        results = {}
        for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:  # Default symbols
            results[symbol] = await self.analyze_sentiment(symbol)
        return results