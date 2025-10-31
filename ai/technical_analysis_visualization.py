"""
Technical Analysis Visualization Module for FX-Ai Trading System

This module provides comprehensive visualization capabilities for technical analysis,
including interactive charts, indicator overlays, and pattern visualization.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrow
from matplotlib.collections import LineCollection
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    theme: str = 'dark'
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    colors: Dict[str, str] = None

    def __post_init__(self):
        if self.colors is None:
            if self.theme == 'dark':
                self.colors = {
                    'background': '#1a1a1a',
                    'text': '#ffffff',
                    'up': '#00ff00',
                    'down': '#ff4444',
                    'neutral': '#888888',
                    'support': '#00ffff',
                    'resistance': '#ff8800',
                    'trend': '#ffff00',
                    'volume': '#4444ff'
                }
            else:
                self.colors = {
                    'background': '#ffffff',
                    'text': '#000000',
                    'up': '#228B22',
                    'down': '#DC143C',
                    'neutral': '#666666',
                    'support': '#00CED1',
                    'resistance': '#FF6347',
                    'trend': '#FFD700',
                    'volume': '#4169E1'
                }


@dataclass
class ChartElement:
    """Represents a chart element for visualization"""
    element_type: str
    data: Any
    style: Dict[str, Any]
    label: Optional[str] = None


class TechnicalAnalysisVisualization:
    """
    Comprehensive technical analysis visualization system
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize technical analysis visualization

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Visualization configuration
        viz_config = config.get('visualization', {})
        self.enabled = viz_config.get('enabled', True)

        # Chart settings
        self.viz_config = VisualizationConfig(
            theme=viz_config.get('theme', 'dark'),
            figsize=tuple(viz_config.get('figsize', [12, 8])),
            dpi=viz_config.get('dpi', 100)
        )

        # Interactive settings
        self.interactive = viz_config.get('interactive', True)
        self.max_points = viz_config.get('max_points', 1000)

        # Element settings
        self.show_volume = viz_config.get('show_volume', True)
        self.show_indicators = viz_config.get('show_indicators', True)
        self.show_patterns = viz_config.get('show_patterns', True)
        self.show_trendlines = viz_config.get('show_trendlines', True)
        self.show_support_resistance = viz_config.get('show_support_resistance', True)

        # Indicator settings
        self.indicator_config = viz_config.get('indicators', {
            'moving_averages': ['SMA_20', 'SMA_50', 'EMA_21'],
            'oscillators': ['RSI', 'MACD', 'Stochastic'],
            'volatility': ['BBANDS', 'ATR'],
            'volume': ['Volume', 'OBV']
        })

        # Initialize plotting style
        self._setup_plotting_style()

        self.logger.info("Technical Analysis Visualization initialized")

    def _setup_plotting_style(self):
        """Setup matplotlib and seaborn plotting style"""
        try:
            # Set matplotlib style
            if self.viz_config.theme == 'dark':
                plt.style.use('dark_background')
            else:
                plt.style.use('default')

            # Set seaborn style
            sns.set_style("darkgrid" if self.viz_config.theme == 'dark' else "whitegrid")

            # Custom rcParams
            plt.rcParams.update({
                'figure.figsize': self.viz_config.figsize,
                'figure.dpi': self.viz_config.dpi,
                'axes.labelsize': 10,
                'axes.titlesize': 12,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 9,
                'font.size': 9
            })

        except Exception as e:
            self.logger.warning(f"Error setting up plotting style: {e}")

    def create_comprehensive_chart(self, data: pd.DataFrame, symbol: str,
                                 indicators: Dict[str, pd.Series] = None,
                                 patterns: List[Any] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive technical analysis chart

        Args:
            data: OHLCV price data
            symbol: Trading symbol
            indicators: Dictionary of technical indicators
            patterns: List of detected patterns
            save_path: Optional path to save chart

        Returns:
            Matplotlib figure
        """
        try:
            if self.interactive:
                return self._create_interactive_chart(data, symbol, indicators, patterns, save_path)
            else:
                return self._create_static_chart(data, symbol, indicators, patterns, save_path)

        except Exception as e:
            self.logger.error(f"Error creating comprehensive chart: {e}")
            return plt.figure()

    def _create_static_chart(self, data: pd.DataFrame, symbol: str,
                           indicators: Dict[str, pd.Series] = None,
                           patterns: List[Any] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Create static matplotlib chart"""
        try:
            # Limit data points for performance
            if len(data) > self.max_points:
                data = data.tail(self.max_points)

            # Create subplots
            if self.show_volume:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.viz_config.figsize,
                                             gridspec_kw={'height_ratios': [3, 1]})
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=self.viz_config.figsize)
                ax2 = None

            # Set background colors
            ax1.set_facecolor(self.viz_config.colors['background'])
            if ax2:
                ax2.set_facecolor(self.viz_config.colors['background'])

            # Plot price data
            self._plot_price_data(ax1, data, symbol)

            # Plot indicators
            if indicators and self.show_indicators:
                self._plot_indicators(ax1, indicators)

            # Plot patterns
            if patterns and self.show_patterns:
                self._plot_patterns(ax1, patterns, data)

            # Plot volume
            if ax2 and self.show_volume:
                self._plot_volume(ax2, data)

            # Add trendlines and support/resistance
            if self.show_trendlines:
                self._plot_trendlines(ax1, data)

            if self.show_support_resistance:
                self._plot_support_resistance(ax1, data)

            # Format axes
            self._format_axes(ax1, ax2, data)

            # Add title
            title = f"{symbol} Technical Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            ax1.set_title(title, color=self.viz_config.colors['text'], fontsize=14)

            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=self.viz_config.dpi,
                           facecolor=self.viz_config.colors['background'],
                           bbox_inches='tight')
                self.logger.info(f"Chart saved to {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Error creating static chart: {e}")
            return plt.figure()

    def _create_interactive_chart(self, data: pd.DataFrame, symbol: str,
                                indicators: Dict[str, pd.Series] = None,
                                patterns: List[Any] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """Create interactive plotly chart"""
        try:
            # Limit data points for performance
            if len(data) > self.max_points:
                data = data.tail(self.max_points)

            # Create subplots
            if self.show_volume:
                fig = make_subplots(rows=2, cols=1,
                                  shared_xaxes=True,
                                  vertical_spacing=0.1,
                                  subplot_titles=(f'{symbol} Price', 'Volume'),
                                  row_width=[0.7, 0.3])
            else:
                fig = make_subplots(rows=1, cols=1,
                                  subplot_titles=(f'{symbol} Price',))

            # Plot candlesticks
            self._plot_interactive_candlesticks(fig, data, row=1, col=1)

            # Plot indicators
            if indicators and self.show_indicators:
                self._plot_interactive_indicators(fig, indicators, row=1, col=1)

            # Plot patterns
            if patterns and self.show_patterns:
                self._plot_interactive_patterns(fig, patterns, data, row=1, col=1)

            # Plot volume
            if self.show_volume:
                self._plot_interactive_volume(fig, data, row=2, col=1)

            # Update layout
            self._update_interactive_layout(fig, symbol)

            # Save if requested
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Interactive chart saved to {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Error creating interactive chart: {e}")
            return make_subplots()

    def _plot_price_data(self, ax, data: pd.DataFrame, symbol: str):
        """Plot price data with candlesticks"""
        try:
            # Plot candlesticks
            self._plot_candlesticks(ax, data)

            # Plot price line
            ax.plot(data.index, data['close'],
                   color=self.viz_config.colors['neutral'],
                   linewidth=1, alpha=0.7, label='Close')

        except Exception as e:
            self.logger.warning(f"Error plotting price data: {e}")

    def _plot_candlesticks(self, ax, data: pd.DataFrame):
        """Plot candlestick chart"""
        try:
            # Calculate body positions
            data = data.copy()
            data['body_high'] = data[['open', 'close']].max(axis=1)
            data['body_low'] = data[['open', 'close']].min(axis=1)
            data['color'] = np.where(data['close'] >= data['open'],
                                   self.viz_config.colors['up'],
                                   self.viz_config.colors['down'])

            # Plot candles
            for idx, row in enumerate(data.itertuples()):
                # High-low line
                ax.vlines(idx, row.low, row.high,
                         color=row.color, linewidth=1, alpha=0.8)

                # Body
                body_height = abs(row.close - row.open)
                if body_height > 0:
                    ax.add_patch(Rectangle(
                        (idx - 0.4, row.body_low),
                        0.8, body_height,
                        facecolor=row.color,
                        edgecolor=row.color,
                        alpha=0.8
                    ))

        except Exception as e:
            self.logger.warning(f"Error plotting candlesticks: {e}")

    def _plot_indicators(self, ax, indicators: Dict[str, pd.Series]):
        """Plot technical indicators"""
        try:
            # Moving averages
            ma_indicators = [ind for ind in indicators.keys()
                           if any(ma in ind.upper() for ma in ['SMA', 'EMA', 'WMA'])]
            colors = ['blue', 'red', 'green', 'orange', 'purple']

            for i, ind_name in enumerate(ma_indicators[:5]):  # Limit to 5
                if ind_name in indicators:
                    ax.plot(indicators[ind_name].index, indicators[ind_name].values,
                           color=colors[i % len(colors)],
                           linewidth=1.5, alpha=0.8, label=ind_name)

            # Bollinger Bands
            if 'BBANDS_UPPER' in indicators and 'BBANDS_LOWER' in indicators:
                ax.fill_between(indicators['BBANDS_UPPER'].index,
                              indicators['BBANDS_UPPER'].values,
                              indicators['BBANDS_LOWER'].values,
                              color='gray', alpha=0.1, label='Bollinger Bands')

        except Exception as e:
            self.logger.warning(f"Error plotting indicators: {e}")

    def _plot_patterns(self, ax, patterns: List[Any], data: pd.DataFrame):
        """Plot detected patterns"""
        try:
            for pattern in patterns:
                if hasattr(pattern, 'pattern_type') and hasattr(pattern, 'vertices'):
                    # Convert pattern vertices to data coordinates
                    vertices = self._pattern_vertices_to_data_coords(pattern.vertices, data)

                    if len(vertices) >= 3:
                        # Create polygon
                        poly = Polygon(vertices,
                                     facecolor='none',
                                     edgecolor=self._get_pattern_color(pattern.pattern_type),
                                     linewidth=2, alpha=0.7)
                        ax.add_patch(poly)

                        # Add label
                        centroid = np.mean(vertices, axis=0)
                        ax.annotate(pattern.pattern_type,
                                  xy=centroid,
                                  xytext=(10, 10),
                                  textcoords='offset points',
                                  fontsize=8,
                                  color=self._get_pattern_color(pattern.pattern_type),
                                  bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='white', alpha=0.8))

        except Exception as e:
            self.logger.warning(f"Error plotting patterns: {e}")

    def _plot_volume(self, ax, data: pd.DataFrame):
        """Plot volume histogram"""
        try:
            if 'volume' in data.columns:
                # Color volume bars
                colors = [self.viz_config.colors['up'] if close >= open else self.viz_config.colors['down']
                         for close, open in zip(data['close'], data['open'])]

                ax.bar(range(len(data)), data['volume'],
                      color=colors, alpha=0.7, width=0.8)

                ax.set_ylabel('Volume', color=self.viz_config.colors['text'])
                ax.tick_params(axis='y', colors=self.viz_config.colors['text'])

        except Exception as e:
            self.logger.warning(f"Error plotting volume: {e}")

    def _plot_trendlines(self, ax, data: pd.DataFrame):
        """Plot trendlines"""
        try:
            # Simple trendline detection (could be enhanced)
            if len(data) > 20:
                # Calculate linear regression trendline
                x = np.arange(len(data))
                y = data['close'].values

                # Fit trendline
                coeffs = np.polyfit(x, y, 1)
                trendline = np.polyval(coeffs, x)

                ax.plot(data.index, trendline,
                       color=self.viz_config.colors['trend'],
                       linewidth=2, alpha=0.8, label='Trendline')

        except Exception as e:
            self.logger.warning(f"Error plotting trendlines: {e}")

    def _plot_support_resistance(self, ax, data: pd.DataFrame):
        """Plot support and resistance levels"""
        try:
            # Simple S/R detection using local minima/maxima
            window = min(20, len(data) // 4)

            if len(data) > window:
                # Find local maxima (resistance)
                resistance_levels = []
                for i in range(window, len(data) - window):
                    if data['high'].iloc[i] == max(data['high'].iloc[i-window:i+window+1]):
                        resistance_levels.append(data['high'].iloc[i])

                # Find local minima (support)
                support_levels = []
                for i in range(window, len(data) - window):
                    if data['low'].iloc[i] == min(data['low'].iloc[i-window:i+window+1]):
                        support_levels.append(data['low'].iloc[i])

                # Plot significant levels
                for level in resistance_levels[-3:]:  # Last 3 resistance levels
                    ax.axhline(y=level, color=self.viz_config.colors['resistance'],
                             linestyle='--', alpha=0.6, linewidth=1)

                for level in support_levels[-3:]:  # Last 3 support levels
                    ax.axhline(y=level, color=self.viz_config.colors['support'],
                             linestyle='--', alpha=0.6, linewidth=1)

        except Exception as e:
            self.logger.warning(f"Error plotting support/resistance: {e}")

    def _plot_interactive_candlesticks(self, fig, data: pd.DataFrame, row: int, col: int):
        """Plot interactive candlesticks"""
        try:
            # Add candlestick trace
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color=self.viz_config.colors['up'],
                decreasing_line_color=self.viz_config.colors['down']
            ), row=row, col=col)

        except Exception as e:
            self.logger.warning(f"Error plotting interactive candlesticks: {e}")

    def _plot_interactive_indicators(self, fig, indicators: Dict[str, pd.Series], row: int, col: int):
        """Plot interactive indicators"""
        try:
            # Moving averages
            ma_indicators = [ind for ind in indicators.keys()
                           if any(ma in ind.upper() for ma in ['SMA', 'EMA'])]

            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, ind_name in enumerate(ma_indicators[:3]):  # Limit to 3
                if ind_name in indicators:
                    fig.add_trace(go.Scatter(
                        x=indicators[ind_name].index,
                        y=indicators[ind_name].values,
                        mode='lines',
                        name=ind_name,
                        line=dict(color=colors[i], width=2)
                    ), row=row, col=col)

        except Exception as e:
            self.logger.warning(f"Error plotting interactive indicators: {e}")

    def _plot_interactive_patterns(self, fig, patterns: List[Any], data: pd.DataFrame, row: int, col: int):
        """Plot interactive patterns"""
        try:
            for pattern in patterns[:5]:  # Limit to 5 patterns
                if hasattr(pattern, 'pattern_type') and hasattr(pattern, 'vertices'):
                    vertices = self._pattern_vertices_to_data_coords(pattern.vertices, data)

                    if len(vertices) >= 3:
                        # Add pattern shape
                        fig.add_trace(go.Scatter(
                            x=[v[0] for v in vertices] + [vertices[0][0]],
                            y=[v[1] for v in vertices] + [vertices[0][1]],
                            mode='lines',
                            name=pattern.pattern_type,
                            line=dict(color=self._get_pattern_color(pattern.pattern_type), width=3),
                            fill='toself',
                            fillcolor=self._get_pattern_color(pattern.pattern_type),
                            opacity=0.3
                        ), row=row, col=col)

        except Exception as e:
            self.logger.warning(f"Error plotting interactive patterns: {e}")

    def _plot_interactive_volume(self, fig, data: pd.DataFrame, row: int, col: int):
        """Plot interactive volume"""
        try:
            if 'volume' in data.columns:
                colors = [self.viz_config.colors['up'] if close >= open else self.viz_config.colors['down']
                         for close, open in zip(data['close'], data['open'])]

                fig.add_trace(go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ), row=row, col=col)

        except Exception as e:
            self.logger.warning(f"Error plotting interactive volume: {e}")

    def _update_interactive_layout(self, fig, symbol: str):
        """Update interactive chart layout"""
        try:
            title = f"{symbol} Technical Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            fig.update_layout(
                title=title,
                paper_bgcolor=self.viz_config.colors['background'],
                plot_bgcolor=self.viz_config.colors['background'],
                font=dict(color=self.viz_config.colors['text']),
                xaxis_rangeslider_visible=False,
                showlegend=True
            )

            # Update axes
            fig.update_xaxes(showgrid=True, gridcolor='gray', gridwidth=0.5)
            fig.update_yaxes(showgrid=True, gridcolor='gray', gridwidth=0.5)

        except Exception as e:
            self.logger.warning(f"Error updating interactive layout: {e}")

    def _format_axes(self, ax1, ax2, data: pd.DataFrame):
        """Format chart axes"""
        try:
            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

            # Set colors
            ax1.tick_params(colors=self.viz_config.colors['text'])
            ax1.xaxis.label.set_color(self.viz_config.colors['text'])
            ax1.yaxis.label.set_color(self.viz_config.colors['text'])

            if ax2:
                ax2.tick_params(colors=self.viz_config.colors['text'])
                ax2.xaxis.label.set_color(self.viz_config.colors['text'])
                ax2.yaxis.label.set_color(self.viz_config.colors['text'])

            # Add grid
            ax1.grid(True, alpha=0.3)
            if ax2:
                ax2.grid(True, alpha=0.3)

            # Add legend
            ax1.legend(loc='upper left', facecolor=self.viz_config.colors['background'],
                      edgecolor=self.viz_config.colors['text'])

        except Exception as e:
            self.logger.warning(f"Error formatting axes: {e}")

    def _pattern_vertices_to_data_coords(self, vertices: List[Tuple[int, int]],
                                       data: pd.DataFrame) -> List[Tuple[float, float]]:
        """Convert pattern vertices to data coordinates"""
        try:
            data_coords = []

            for vx, vy in vertices:
                # Convert x coordinate (index) to datetime
                if 0 <= vx < len(data):
                    x_coord = data.index[vx]
                else:
                    continue

                # Convert y coordinate (pixel) to price
                price_range = data['high'].max() - data['low'].min()
                y_coord = data['low'].min() + (vy / 1000) * price_range  # Assuming 1000px height

                data_coords.append((x_coord, y_coord))

            return data_coords

        except Exception:
            return []

    def _get_pattern_color(self, pattern_type: str) -> str:
        """Get color for pattern type"""
        color_map = {
            'head_and_shoulders': '#FF4444',
            'double_top': '#44FF44',
            'double_bottom': '#4444FF',
            'triangle_ascending': '#FFFF44',
            'triangle_descending': '#FF44FF',
            'wedge': '#44FFFF',
            'flag': '#888888',
            'pennant': '#FF8844',
            'cup_and_handle': '#8844FF'
        }
        return color_map.get(pattern_type, '#FFFFFF')

    def create_indicator_dashboard(self, data: pd.DataFrame, symbol: str,
                                 indicators: Dict[str, pd.Series],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create indicator dashboard with multiple subplots

        Args:
            data: Price data
            symbol: Trading symbol
            indicators: Technical indicators
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        try:
            # Create subplot grid
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            axes = axes.flatten()

            # Set background colors
            for ax in axes:
                ax.set_facecolor(self.viz_config.colors['background'])
                ax.tick_params(colors=self.viz_config.colors['text'])

            # Plot 1: Price with MAs
            self._plot_price_with_mas(axes[0], data, indicators)

            # Plot 2: RSI
            if 'RSI' in indicators:
                self._plot_rsi(axes[1], indicators['RSI'])

            # Plot 3: MACD
            if all(ind in indicators for ind in ['MACD', 'MACD_SIGNAL', 'MACD_HIST']):
                self._plot_macd(axes[2], indicators)

            # Plot 4: Bollinger Bands
            if all(ind in indicators for ind in ['BBANDS_UPPER', 'BBANDS_LOWER']):
                self._plot_bollinger_bands(axes[3], data, indicators)

            # Plot 5: Stochastic
            if all(ind in indicators for ind in ['STOCH_K', 'STOCH_D']):
                self._plot_stochastic(axes[4], indicators)

            # Plot 6: Volume
            if 'volume' in data.columns:
                self._plot_volume_chart(axes[5], data)

            # Set title
            fig.suptitle(f'{symbol} Technical Indicator Dashboard',
                        color=self.viz_config.colors['text'], fontsize=16)

            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=self.viz_config.dpi,
                           facecolor=self.viz_config.colors['background'],
                           bbox_inches='tight')
                self.logger.info(f"Indicator dashboard saved to {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Error creating indicator dashboard: {e}")
            return plt.figure()

    def _plot_price_with_mas(self, ax, data: pd.DataFrame, indicators: Dict[str, pd.Series]):
        """Plot price with moving averages"""
        try:
            ax.plot(data.index, data['close'], color=self.viz_config.colors['neutral'],
                   linewidth=1.5, label='Close')

            # Plot moving averages
            ma_colors = ['blue', 'red', 'green']
            ma_names = ['SMA_20', 'SMA_50', 'EMA_21']

            for i, ma_name in enumerate(ma_names):
                if ma_name in indicators:
                    ax.plot(indicators[ma_name].index, indicators[ma_name].values,
                           color=ma_colors[i], linewidth=1.5, label=ma_name)

            ax.set_title('Price & Moving Averages', color=self.viz_config.colors['text'])
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            self.logger.warning(f"Error plotting price with MAs: {e}")

    def _plot_rsi(self, ax, rsi: pd.Series):
        """Plot RSI indicator"""
        try:
            ax.plot(rsi.index, rsi.values, color='purple', linewidth=1.5, label='RSI')
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)

            ax.set_ylim(0, 100)
            ax.set_title('RSI (14)', color=self.viz_config.colors['text'])
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            self.logger.warning(f"Error plotting RSI: {e}")

    def _plot_macd(self, ax, indicators: Dict[str, pd.Series]):
        """Plot MACD indicator"""
        try:
            ax.plot(indicators['MACD'].index, indicators['MACD'].values,
                   color='blue', linewidth=1.5, label='MACD')
            ax.plot(indicators['MACD_SIGNAL'].index, indicators['MACD_SIGNAL'].values,
                   color='red', linewidth=1.5, label='Signal')
            ax.bar(indicators['MACD_HIST'].index, indicators['MACD_HIST'].values,
                  color=['green' if x >= 0 else 'red' for x in indicators['MACD_HIST'].values],
                  alpha=0.7, label='Histogram')

            ax.set_title('MACD (12,26,9)', color=self.viz_config.colors['text'])
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            self.logger.warning(f"Error plotting MACD: {e}")

    def _plot_bollinger_bands(self, ax, data: pd.DataFrame, indicators: Dict[str, pd.Series]):
        """Plot Bollinger Bands"""
        try:
            ax.plot(data.index, data['close'], color=self.viz_config.colors['neutral'],
                   linewidth=1, label='Close')
            ax.plot(indicators['BBANDS_UPPER'].index, indicators['BBANDS_UPPER'].values,
                   color='red', linewidth=1, label='Upper Band')
            ax.plot(indicators['BBANDS_LOWER'].index, indicators['BBANDS_LOWER'].values,
                   color='green', linewidth=1, label='Lower Band')

            if 'BBANDS_MIDDLE' in indicators:
                ax.plot(indicators['BBANDS_MIDDLE'].index, indicators['BBANDS_MIDDLE'].values,
                       color='blue', linewidth=1, label='Middle Band')

            ax.fill_between(indicators['BBANDS_UPPER'].index,
                          indicators['BBANDS_UPPER'].values,
                          indicators['BBANDS_LOWER'].values,
                          color='gray', alpha=0.1)

            ax.set_title('Bollinger Bands (20,2)', color=self.viz_config.colors['text'])
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            self.logger.warning(f"Error plotting Bollinger Bands: {e}")

    def _plot_stochastic(self, ax, indicators: Dict[str, pd.Series]):
        """Plot Stochastic oscillator"""
        try:
            ax.plot(indicators['STOCH_K'].index, indicators['STOCH_K'].values,
                   color='blue', linewidth=1.5, label='%K')
            ax.plot(indicators['STOCH_D'].index, indicators['STOCH_D'].values,
                   color='red', linewidth=1.5, label='%D')
            ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold')

            ax.set_ylim(0, 100)
            ax.set_title('Stochastic (14,3,3)', color=self.viz_config.colors['text'])
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            self.logger.warning(f"Error plotting Stochastic: {e}")

    def _plot_volume_chart(self, ax, data: pd.DataFrame):
        """Plot volume chart"""
        try:
            colors = [self.viz_config.colors['up'] if close >= open else self.viz_config.colors['down']
                     for close, open in zip(data['close'], data['open'])]

            ax.bar(data.index, data['volume'], color=colors, alpha=0.7, width=0.8)
            ax.set_title('Volume', color=self.viz_config.colors['text'])
            ax.grid(True, alpha=0.3)

        except Exception as e:
            self.logger.warning(f"Error plotting volume: {e}")

    def create_heatmap_analysis(self, correlation_matrix: pd.DataFrame,
                              symbol: str, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create correlation heatmap for multiple assets

        Args:
            correlation_matrix: Correlation matrix
            symbol: Base symbol
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=self.viz_config.figsize)

            # Create heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                       center=0, square=True, linewidths=0.5, ax=ax,
                       cbar_kws={'shrink': 0.8})

            ax.set_title(f'{symbol} Correlation Heatmap', color=self.viz_config.colors['text'])
            ax.tick_params(colors=self.viz_config.colors['text'])

            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=self.viz_config.dpi,
                           facecolor=self.viz_config.colors['background'],
                           bbox_inches='tight')
                self.logger.info(f"Correlation heatmap saved to {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {e}")
            return plt.figure()

    def create_performance_chart(self, equity_curve: pd.Series,
                               benchmark: pd.Series = None,
                               drawdown: pd.Series = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create performance visualization chart

        Args:
            equity_curve: Portfolio equity curve
            benchmark: Benchmark series
            drawdown: Drawdown series
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.viz_config.figsize,
                                         gridspec_kw={'height_ratios': [2, 1]})

            # Set background colors
            ax1.set_facecolor(self.viz_config.colors['background'])
            ax2.set_facecolor(self.viz_config.colors['background'])

            # Plot equity curve
            ax1.plot(equity_curve.index, equity_curve.values,
                    color=self.viz_config.colors['up'], linewidth=2, label='Strategy')

            if benchmark is not None:
                ax1.plot(benchmark.index, benchmark.values,
                        color=self.viz_config.colors['neutral'], linewidth=2, label='Benchmark')

            ax1.set_title('Portfolio Performance', color=self.viz_config.colors['text'])
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot drawdown
            if drawdown is not None:
                ax2.fill_between(drawdown.index, 0, drawdown.values,
                               color=self.viz_config.colors['down'], alpha=0.7)
                ax2.set_title('Drawdown', color=self.viz_config.colors['text'])
            else:
                # Calculate drawdown if not provided
                peak = equity_curve.expanding().max()
                drawdown_calc = (equity_curve - peak) / peak
                ax2.fill_between(drawdown_calc.index, 0, drawdown_calc.values,
                               color=self.viz_config.colors['down'], alpha=0.7)
                ax2.set_title('Drawdown', color=self.viz_config.colors['text'])

            ax2.grid(True, alpha=0.3)

            # Format axes
            for ax in [ax1, ax2]:
                ax.tick_params(colors=self.viz_config.colors['text'])
                ax.xaxis.label.set_color(self.viz_config.colors['text'])
                ax.yaxis.label.set_color(self.viz_config.colors['text'])

            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=self.viz_config.dpi,
                           facecolor=self.viz_config.colors['background'],
                           bbox_inches='tight')
                self.logger.info(f"Performance chart saved to {save_path}")

            return fig

        except Exception as e:
            self.logger.error(f"Error creating performance chart: {e}")
            return plt.figure()