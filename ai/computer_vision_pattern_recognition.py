"""
Computer Vision Pattern Recognition Module for FX-Ai Trading System

This module implements computer vision algorithms for chart pattern recognition,
technical analysis visualization, and image-based feature extraction.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatternMatch:
    """Represents a detected chart pattern"""
    pattern_type: str
    confidence: float
    start_idx: int
    end_idx: int
    vertices: List[Tuple[int, int]]
    pattern_features: Dict[str, float]
    timestamp: datetime


@dataclass
class ChartImage:
    """Represents a processed chart image"""
    image: np.ndarray
    width: int
    height: int
    price_range: Tuple[float, float]
    time_range: Tuple[datetime, datetime]
    patterns: List[PatternMatch]


class ComputerVisionPatternRecognition:
    """
    Computer vision system for chart pattern recognition and analysis
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize computer vision pattern recognition

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # CV configuration
        cv_config = config.get('computer_vision', {})
        self.enabled = cv_config.get('enabled', True)

        # Pattern recognition parameters
        pattern_config = cv_config.get('pattern_recognition', {})
        self.min_pattern_confidence = pattern_config.get('min_confidence', 0.6)
        self.pattern_types = pattern_config.get('pattern_types', [
            'head_and_shoulders', 'double_top', 'double_bottom',
            'triangle_ascending', 'triangle_descending', 'wedge',
            'flag', 'pennant', 'cup_and_handle'
        ])

        # Image processing parameters
        image_config = cv_config.get('image_processing', {})
        self.image_width = image_config.get('width', 800)
        self.image_height = image_config.get('height', 600)
        self.line_thickness = image_config.get('line_thickness', 2)
        self.color_scheme = image_config.get('color_scheme', 'dark')

        # Feature extraction parameters
        feature_config = cv_config.get('feature_extraction', {})
        self.feature_scales = feature_config.get('scales', [1.0, 0.5, 2.0])
        self.morphology_operations = feature_config.get('morphology', True)

        # Visualization parameters
        viz_config = cv_config.get('visualization', {})
        self.show_patterns = viz_config.get('show_patterns', True)
        self.show_trendlines = viz_config.get('show_trendlines', True)
        self.show_support_resistance = viz_config.get('show_support_resistance', True)

        # Pattern templates and models
        self.pattern_templates = {}
        self.feature_extractors = {}

        # Initialize components
        self._initialize_cv_components()

        self.logger.info("Computer Vision Pattern Recognition initialized")

    def _initialize_cv_components(self):
        """Initialize computer vision components"""
        try:
            # Initialize pattern templates
            self._load_pattern_templates()

            # Initialize feature extractors
            self._initialize_feature_extractors()

            self.logger.info("CV components initialized")

        except Exception as e:
            self.logger.error(f"Error initializing CV components: {e}")

    def _load_pattern_templates(self):
        """Load predefined pattern templates"""
        try:
            # Define pattern templates as normalized shapes
            self.pattern_templates = {
                'head_and_shoulders': self._create_head_shoulders_template(),
                'double_top': self._create_double_top_template(),
                'double_bottom': self._create_double_bottom_template(),
                'triangle_ascending': self._create_triangle_template('ascending'),
                'triangle_descending': self._create_triangle_template('descending'),
                'wedge': self._create_wedge_template(),
                'flag': self._create_flag_template(),
                'pennant': self._create_pennant_template(),
                'cup_and_handle': self._create_cup_handle_template()
            }

            self.logger.info(f"Loaded {len(self.pattern_templates)} pattern templates")

        except Exception as e:
            self.logger.error(f"Error loading pattern templates: {e}")

    def _initialize_feature_extractors(self):
        """Initialize feature extraction models"""
        try:
            # Initialize clustering for pattern segmentation
            self.cluster_model = KMeans(n_clusters=5, random_state=42)
            self.scaler = StandardScaler()

            self.logger.info("Feature extractors initialized")

        except Exception as e:
            self.logger.error(f"Error initializing feature extractors: {e}")

    def create_chart_image(self, data: pd.DataFrame, symbol: str,
                          timeframe: str = 'H1') -> ChartImage:
        """
        Create chart image from price data

        Args:
            data: OHLC price data
            symbol: Trading symbol
            timeframe: Chart timeframe

        Returns:
            Processed chart image
        """
        try:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

            # Set color scheme
            if self.color_scheme == 'dark':
                plt.style.use('dark_background')
                bg_color = '#1a1a1a'
                line_color = '#00ff00'
                up_color = '#00ff00'
                down_color = '#ff4444'
            else:
                bg_color = 'white'
                line_color = 'black'
                up_color = 'green'
                down_color = 'red'

            ax.set_facecolor(bg_color)

            # Plot candlestick chart
            self._plot_candlesticks(ax, data, up_color, down_color)

            # Add technical indicators
            self._add_technical_indicators(ax, data)

            # Set labels and formatting
            ax.set_title(f'{symbol} {timeframe} Chart', color=line_color)
            ax.set_xlabel('Time', color=line_color)
            ax.set_ylabel('Price', color=line_color)
            ax.tick_params(colors=line_color)
            ax.grid(True, alpha=0.3)

            # Convert to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            plt.close(fig)

            # Create ChartImage object
            price_min = data['low'].min()
            price_max = data['high'].max()
            time_start = data.index[0] if hasattr(data.index, '__getitem__') else datetime.now()
            time_end = data.index[-1] if hasattr(data.index, '__getitem__') else datetime.now()

            chart_image = ChartImage(
                image=image,
                width=image.shape[1],
                height=image.shape[0],
                price_range=(price_min, price_max),
                time_range=(time_start, time_end),
                patterns=[]
            )

            return chart_image

        except Exception as e:
            self.logger.error(f"Error creating chart image: {e}")
            # Return empty image
            empty_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            return ChartImage(
                image=empty_image,
                width=self.image_width,
                height=self.image_height,
                price_range=(0, 0),
                time_range=(datetime.now(), datetime.now()),
                patterns=[]
            )

    def _plot_candlesticks(self, ax, data: pd.DataFrame, up_color: str, down_color: str):
        """Plot candlestick chart"""
        try:
            # Calculate body positions
            data = data.copy()
            data['body_high'] = data[['open', 'close']].max(axis=1)
            data['body_low'] = data[['open', 'close']].min(axis=1)
            data['color'] = np.where(data['close'] >= data['open'], up_color, down_color)

            # Plot candles
            for idx, row in data.iterrows():
                # High-low line
                ax.vlines(idx, row['low'], row['high'], color=row['color'], linewidth=1)

                # Body
                body_height = abs(row['close'] - row['open'])
                if body_height > 0:
                    ax.add_patch(Rectangle(
                        (idx - 0.4, row['body_low']),
                        0.8, body_height,
                        facecolor=row['color'],
                        edgecolor=row['color']
                    ))

        except Exception as e:
            self.logger.warning(f"Error plotting candlesticks: {e}")

    def _add_technical_indicators(self, ax, data: pd.DataFrame):
        """Add technical indicators to chart"""
        try:
            if len(data) < 50:
                return

            # Add moving averages
            if 'close' in data.columns:
                ma20 = data['close'].rolling(20).mean()
                ma50 = data['close'].rolling(50).mean()

                ax.plot(data.index, ma20, color='blue', linewidth=1, alpha=0.7, label='MA20')
                ax.plot(data.index, ma50, color='red', linewidth=1, alpha=0.7, label='MA50')

            # Add RSI if available
            if 'rsi' in data.columns:
                ax2 = ax.twinx()
                ax2.plot(data.index, data['rsi'], color='purple', linewidth=1, alpha=0.7, label='RSI')
                ax2.set_ylim(0, 100)
                ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)

        except Exception as e:
            self.logger.warning(f"Error adding technical indicators: {e}")

    def detect_patterns(self, chart_image: ChartImage, data: pd.DataFrame) -> List[PatternMatch]:
        """
        Detect chart patterns in the image

        Args:
            chart_image: Processed chart image
            data: Original price data

        Returns:
            List of detected patterns
        """
        try:
            detected_patterns = []

            # Convert to grayscale for processing
            gray = cv2.cvtColor(chart_image.image, cv2.COLOR_BGR2GRAY)

            # Apply preprocessing
            processed = self._preprocess_image(gray)

            # Detect patterns using template matching
            for pattern_type, template in self.pattern_templates.items():
                try:
                    matches = self._match_pattern_template(processed, template, pattern_type)

                    for match in matches:
                        if match.confidence >= self.min_pattern_confidence:
                            # Convert image coordinates to data indices
                            start_idx, end_idx = self._image_coords_to_data_indices(
                                match, chart_image, len(data)
                            )

                            pattern_match = PatternMatch(
                                pattern_type=pattern_type,
                                confidence=match.confidence,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                vertices=match.vertices,
                                pattern_features=match.features,
                                timestamp=datetime.now()
                            )

                            detected_patterns.append(pattern_match)

                except Exception as e:
                    self.logger.warning(f"Error detecting {pattern_type}: {e}")
                    continue

            # Sort by confidence
            detected_patterns.sort(key=lambda x: x.confidence, reverse=True)

            # Update chart image
            chart_image.patterns = detected_patterns

            self.logger.info(f"Detected {len(detected_patterns)} patterns")
            return detected_patterns

        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for pattern detection"""
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Apply morphological operations
            if self.morphology_operations:
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            return edges

        except Exception as e:
            self.logger.warning(f"Error preprocessing image: {e}")
            return image

    def _match_pattern_template(self, image: np.ndarray, template: Dict[str, Any],
                              pattern_type: str) -> List[Any]:
        """Match pattern template against image"""
        try:
            matches = []

            # Template matching
            if 'template_image' in template:
                result = cv2.matchTemplate(image, template['template_image'], cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= self.min_pattern_confidence)

                for pt in zip(*locations[::-1]):
                    match = type('Match', (), {
                        'confidence': result[pt[1], pt[0]],
                        'vertices': [(pt[0], pt[1]), (pt[0] + template['width'], pt[1] + template['height'])],
                        'features': {'template_match': result[pt[1], pt[0]]}
                    })()
                    matches.append(match)

            # Geometric pattern detection
            elif 'geometric_features' in template:
                geometric_matches = self._detect_geometric_pattern(image, template, pattern_type)
                matches.extend(geometric_matches)

            return matches

        except Exception as e:
            self.logger.warning(f"Error matching {pattern_type} template: {e}")
            return []

    def _detect_geometric_pattern(self, image: np.ndarray, template: Dict[str, Any],
                                pattern_type: str) -> List[Any]:
        """Detect geometric patterns using computer vision"""
        try:
            matches = []

            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Calculate contour features
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                # Calculate shape features
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # Approximate polygon
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                num_vertices = len(approx)

                # Match against template features
                confidence = self._calculate_geometric_confidence(
                    template, area, circularity, solidity, num_vertices
                )

                if confidence >= self.min_pattern_confidence:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)

                    match = type('Match', (), {
                        'confidence': confidence,
                        'vertices': [(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                        'features': {
                            'area': area,
                            'circularity': circularity,
                            'solidity': solidity,
                            'vertices': num_vertices
                        }
                    })()
                    matches.append(match)

            return matches

        except Exception as e:
            self.logger.warning(f"Error detecting geometric {pattern_type}: {e}")
            return []

    def _calculate_geometric_confidence(self, template: Dict[str, Any], area: float,
                                      circularity: float, solidity: float,
                                      num_vertices: int) -> float:
        """Calculate confidence score for geometric pattern match"""
        try:
            features = template.get('geometric_features', {})

            confidence = 1.0

            # Area match
            if 'area_range' in features:
                min_area, max_area = features['area_range']
                if not (min_area <= area <= max_area):
                    confidence *= 0.5

            # Circularity match
            if 'circularity_range' in features:
                min_circ, max_circ = features['circularity_range']
                if not (min_circ <= circularity <= max_circ):
                    confidence *= 0.7

            # Solidity match
            if 'solidity_range' in features:
                min_solid, max_solid = features['solidity_range']
                if not (min_solid <= solidity <= max_solid):
                    confidence *= 0.8

            # Vertex count match
            if 'vertex_range' in features:
                min_vert, max_vert = features['vertex_range']
                if not (min_vert <= num_vertices <= max_vert):
                    confidence *= 0.6

            return confidence

        except Exception:
            return 0.0

    def _image_coords_to_data_indices(self, match: Any, chart_image: ChartImage,
                                    data_length: int) -> Tuple[int, int]:
        """Convert image coordinates to data indices"""
        try:
            # Get pattern vertices
            vertices = match.vertices
            if not vertices:
                return 0, data_length - 1

            # Find min/max x coordinates (time dimension)
            x_coords = [v[0] for v in vertices]
            min_x = min(x_coords)
            max_x = max(x_coords)

            # Convert to data indices
            width = chart_image.width
            start_idx = int((min_x / width) * data_length)
            end_idx = int((max_x / width) * data_length)

            # Ensure valid indices
            start_idx = max(0, min(start_idx, data_length - 1))
            end_idx = max(start_idx + 1, min(end_idx, data_length - 1))

            return start_idx, end_idx

        except Exception:
            return 0, data_length - 1

    def extract_image_features(self, chart_image: ChartImage) -> Dict[str, float]:
        """
        Extract features from chart image

        Args:
            chart_image: Processed chart image

        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}

            # Convert to grayscale
            gray = cv2.cvtColor(chart_image.image, cv2.COLOR_BGR2GRAY)

            # Multi-scale feature extraction
            for scale in self.feature_scales:
                scaled_features = self._extract_scale_features(gray, scale)
                features.update({f"{k}_scale_{scale}": v for k, v in scaled_features.items()})

            # Pattern-based features
            pattern_features = self._extract_pattern_features(chart_image)
            features.update(pattern_features)

            # Statistical features
            stat_features = self._extract_statistical_features(gray)
            features.update(stat_features)

            self.logger.info(f"Extracted {len(features)} image features")
            return features

        except Exception as e:
            self.logger.error(f"Error extracting image features: {e}")
            return {}

    def _extract_scale_features(self, image: np.ndarray, scale: float) -> Dict[str, float]:
        """Extract features at specific scale"""
        try:
            # Resize image
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            scaled = cv2.resize(image, (new_width, new_height))

            features = {}

            # HOG features
            hog = cv2.HOGDescriptor()
            hog_features = hog.compute(scaled)
            if hog_features is not None:
                features['hog_mean'] = np.mean(hog_features)
                features['hog_std'] = np.std(hog_features)

            # LBP features
            lbp = self._compute_lbp(scaled)
            features['lbp_mean'] = np.mean(lbp)
            features['lbp_std'] = np.std(lbp)

            return features

        except Exception:
            return {}

    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """Compute Local Binary Patterns"""
        try:
            # Simple LBP implementation
            height, width = image.shape
            lbp = np.zeros((height-2, width-2), dtype=np.uint8)

            for i in range(1, height-1):
                for j in range(1, width-1):
                    center = image[i, j]
                    pattern = 0
                    pattern |= (image[i-1, j-1] >= center) << 7
                    pattern |= (image[i-1, j] >= center) << 6
                    pattern |= (image[i-1, j+1] >= center) << 5
                    pattern |= (image[i, j+1] >= center) << 4
                    pattern |= (image[i+1, j+1] >= center) << 3
                    pattern |= (image[i+1, j] >= center) << 2
                    pattern |= (image[i+1, j-1] >= center) << 1
                    pattern |= (image[i, j-1] >= center) << 0
                    lbp[i-1, j-1] = pattern

            return lbp.flatten()

        except Exception:
            return np.array([])

    def _extract_pattern_features(self, chart_image: ChartImage) -> Dict[str, float]:
        """Extract pattern-based features"""
        try:
            features = {}

            # Count patterns by type
            pattern_counts = {}
            total_confidence = 0.0

            for pattern in chart_image.patterns:
                pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
                total_confidence += pattern.confidence

            # Pattern density features
            features['total_patterns'] = len(chart_image.patterns)
            features['pattern_density'] = len(chart_image.patterns) / max(1, chart_image.width * chart_image.height / 10000)
            features['avg_pattern_confidence'] = total_confidence / max(1, len(chart_image.patterns))

            # Specific pattern counts
            for pattern_type in self.pattern_types:
                features[f'{pattern_type}_count'] = pattern_counts.get(pattern_type, 0)

            return features

        except Exception:
            return {}

    def _extract_statistical_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from image"""
        try:
            features = {}

            # Basic statistics
            features['mean_intensity'] = np.mean(image)
            features['std_intensity'] = np.std(image)
            features['min_intensity'] = np.min(image)
            features['max_intensity'] = np.max(image)

            # Histogram features
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.flatten() / np.sum(hist)

            features['hist_entropy'] = -np.sum(hist * np.log2(hist + 1e-8))
            features['hist_uniformity'] = np.sum(hist ** 2)

            # Edge features
            edges = cv2.Canny(image, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (image.shape[0] * image.shape[1])

            return features

        except Exception:
            return {}

    def visualize_patterns(self, chart_image: ChartImage, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected patterns on chart

        Args:
            chart_image: Chart image with patterns
            save_path: Optional path to save visualization

        Returns:
            Visualization image
        """
        try:
            # Create copy of image
            vis_image = chart_image.image.copy()

            # Draw patterns
            for pattern in chart_image.patterns:
                if pattern.confidence < self.min_pattern_confidence:
                    continue

                # Draw pattern outline
                vertices = np.array(pattern.vertices, np.int32)
                color = self._get_pattern_color(pattern.pattern_type)

                # Draw polygon
                cv2.polylines(vis_image, [vertices], True, color, self.line_thickness)

                # Add pattern label
                if vertices.size > 0:
                    centroid = np.mean(vertices, axis=0).astype(int)
                    label = f"{pattern.pattern_type}: {pattern.confidence:.2f}"
                    cv2.putText(vis_image, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, color, 1, cv2.LINE_AA)

            # Save if requested
            if save_path:
                cv2.imwrite(save_path, vis_image)
                self.logger.info(f"Pattern visualization saved to {save_path}")

            return vis_image

        except Exception as e:
            self.logger.error(f"Error visualizing patterns: {e}")
            return chart_image.image

    def _get_pattern_color(self, pattern_type: str) -> Tuple[int, int, int]:
        """Get color for pattern type"""
        color_map = {
            'head_and_shoulders': (255, 0, 0),    # Red
            'double_top': (0, 255, 0),           # Green
            'double_bottom': (0, 0, 255),        # Blue
            'triangle_ascending': (255, 255, 0), # Yellow
            'triangle_descending': (255, 0, 255), # Magenta
            'wedge': (0, 255, 255),              # Cyan
            'flag': (128, 128, 128),             # Gray
            'pennant': (255, 128, 0),            # Orange
            'cup_and_handle': (128, 0, 128)      # Purple
        }
        return color_map.get(pattern_type, (255, 255, 255))

    def create_pattern_templates(self):
        """Create pattern templates for matching"""
        # This would be implemented with actual template images
        # For now, using geometric features
        pass

    def _create_head_shoulders_template(self) -> Dict[str, Any]:
        """Create head and shoulders pattern template"""
        return {
            'geometric_features': {
                'area_range': (1000, 50000),
                'circularity_range': (0.1, 0.5),
                'solidity_range': (0.7, 0.95),
                'vertex_range': (5, 15)
            }
        }

    def _create_double_top_template(self) -> Dict[str, Any]:
        """Create double top pattern template"""
        return {
            'geometric_features': {
                'area_range': (500, 20000),
                'circularity_range': (0.2, 0.6),
                'solidity_range': (0.8, 0.98),
                'vertex_range': (3, 8)
            }
        }

    def _create_double_bottom_template(self) -> Dict[str, Any]:
        """Create double bottom pattern template"""
        return {
            'geometric_features': {
                'area_range': (500, 20000),
                'circularity_range': (0.2, 0.6),
                'solidity_range': (0.8, 0.98),
                'vertex_range': (3, 8)
            }
        }

    def _create_triangle_template(self, direction: str) -> Dict[str, Any]:
        """Create triangle pattern template"""
        return {
            'geometric_features': {
                'area_range': (2000, 100000),
                'circularity_range': (0.05, 0.3),
                'solidity_range': (0.9, 0.99),
                'vertex_range': (3, 6)
            }
        }

    def _create_wedge_template(self) -> Dict[str, Any]:
        """Create wedge pattern template"""
        return {
            'geometric_features': {
                'area_range': (1000, 50000),
                'circularity_range': (0.1, 0.4),
                'solidity_range': (0.85, 0.98),
                'vertex_range': (3, 7)
            }
        }

    def _create_flag_template(self) -> Dict[str, Any]:
        """Create flag pattern template"""
        return {
            'geometric_features': {
                'area_range': (300, 10000),
                'circularity_range': (0.3, 0.7),
                'solidity_range': (0.8, 0.97),
                'vertex_range': (4, 10)
            }
        }

    def _create_pennant_template(self) -> Dict[str, Any]:
        """Create pennant pattern template"""
        return {
            'geometric_features': {
                'area_range': (200, 5000),
                'circularity_range': (0.4, 0.8),
                'solidity_range': (0.75, 0.95),
                'vertex_range': (3, 6)
            }
        }

    def _create_cup_handle_template(self) -> Dict[str, Any]:
        """Create cup and handle pattern template"""
        return {
            'geometric_features': {
                'area_range': (5000, 200000),
                'circularity_range': (0.02, 0.2),
                'solidity_range': (0.8, 0.98),
                'vertex_range': (6, 20)
            }
        }