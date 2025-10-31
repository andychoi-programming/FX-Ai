"""
Image-Based Feature Extraction Module for FX-Ai Trading System

This module implements advanced image-based feature extraction from price charts,
converting visual patterns into numerical features for enhanced trading signals.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import cv2
from skimage import feature, filters, morphology, measure
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from scipy import ndimage, stats
from scipy.fft import fft2, fftshift
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ImageFeatures:
    """Container for extracted image features"""
    timestamp: datetime
    symbol: str
    basic_stats: Dict[str, float]
    texture_features: Dict[str, float]
    shape_features: Dict[str, float]
    frequency_features: Dict[str, float]
    pattern_features: Dict[str, float]
    fractal_features: Dict[str, float]
    combined_features: np.ndarray


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction process"""
    features: List[ImageFeatures]
    feature_matrix: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]


class ImageBasedFeatureExtraction:
    """
    Advanced image-based feature extraction for price charts
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image-based feature extraction

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Feature extraction configuration
        feature_config = config.get('image_features', {})
        self.enabled = feature_config.get('enabled', True)

        # Image processing parameters
        image_config = feature_config.get('image_processing', {})
        self.image_width = image_config.get('width', 224)
        self.image_height = image_config.get('height', 224)
        self.channels = image_config.get('channels', 3)
        self.normalization = image_config.get('normalization', 'zscore')

        # Feature extraction parameters
        extraction_config = feature_config.get('extraction', {})
        self.extract_basic_stats = extraction_config.get('basic_stats', True)
        self.extract_texture = extraction_config.get('texture', True)
        self.extract_shape = extraction_config.get('shape', True)
        self.extract_frequency = extraction_config.get('frequency', True)
        self.extract_patterns = extraction_config.get('patterns', True)
        self.extract_fractal = extraction_config.get('fractal', True)

        # Advanced parameters
        self.hog_orientations = extraction_config.get('hog_orientations', 9)
        self.hog_pixels_per_cell = extraction_config.get('hog_pixels_per_cell', (8, 8))
        self.hog_cells_per_block = extraction_config.get('hog_cells_per_block', (2, 2))
        self.lbp_radius = extraction_config.get('lbp_radius', 3)
        self.lbp_points = extraction_config.get('lbp_points', 24)
        self.gabor_scales = extraction_config.get('gabor_scales', [1, 2, 4])

        # Dimensionality reduction
        self.pca_components = extraction_config.get('pca_components', 50)
        self.use_pca = extraction_config.get('use_pca', True)

        # Feature scaling
        self.feature_scaler = StandardScaler()
        self.pca_model = PCA(n_components=self.pca_components)

        # Feature history for temporal analysis
        self.feature_history = {}
        self.max_history = extraction_config.get('max_history', 1000)

        # Initialize feature extractors
        self._initialize_feature_extractors()

        self.logger.info("Image-Based Feature Extraction initialized")

    def _initialize_feature_extractors(self):
        """Initialize feature extraction components"""
        try:
            # Initialize Gabor filters for texture analysis
            self.gabor_kernels = self._create_gabor_kernels()

            # Initialize clustering for pattern segmentation
            self.pattern_cluster_model = KMeans(n_clusters=10, random_state=42)

            self.logger.info("Feature extractors initialized")

        except Exception as e:
            self.logger.error(f"Error initializing feature extractors: {e}")

    def _create_gabor_kernels(self) -> List[np.ndarray]:
        """Create Gabor filter kernels for texture analysis"""
        try:
            kernels = []

            for theta in np.arange(0, np.pi, np.pi / 8):  # 8 orientations
                for sigma in self.gabor_scales:  # Different scales
                    for frequency in [0.1, 0.2]:  # Different frequencies
                        kernel = cv2.getGaborKernel(
                            (21, 21), sigma, theta, 10 * frequency,
                            0.5, 0, ktype=cv2.CV_32F
                        )
                        kernels.append(kernel)

            return kernels

        except Exception as e:
            self.logger.warning(f"Error creating Gabor kernels: {e}")
            return []

    def extract_features_from_chart(self, chart_image: np.ndarray,
                                  symbol: str, timestamp: datetime) -> ImageFeatures:
        """
        Extract comprehensive features from chart image

        Args:
            chart_image: Chart image as numpy array
            symbol: Trading symbol
            timestamp: Timestamp of the chart

        Returns:
            Extracted image features
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(chart_image)

            # Extract different types of features
            basic_stats = self._extract_basic_stats(processed_image) if self.extract_basic_stats else {}
            texture_features = self._extract_texture_features(processed_image) if self.extract_texture else {}
            shape_features = self._extract_shape_features(processed_image) if self.extract_shape else {}
            frequency_features = self._extract_frequency_features(processed_image) if self.extract_frequency else {}
            pattern_features = self._extract_pattern_features(processed_image) if self.extract_patterns else {}
            fractal_features = self._extract_fractal_features(processed_image) if self.extract_fractal else {}

            # Combine all features
            combined_features = self._combine_features([
                basic_stats, texture_features, shape_features,
                frequency_features, pattern_features, fractal_features
            ])

            # Create feature object
            image_features = ImageFeatures(
                timestamp=timestamp,
                symbol=symbol,
                basic_stats=basic_stats,
                texture_features=texture_features,
                shape_features=shape_features,
                frequency_features=frequency_features,
                pattern_features=pattern_features,
                fractal_features=fractal_features,
                combined_features=combined_features
            )

            # Store in history
            if symbol not in self.feature_history:
                self.feature_history[symbol] = []

            self.feature_history[symbol].append(image_features)

            # Limit history size
            if len(self.feature_history[symbol]) > self.max_history:
                self.feature_history[symbol] = self.feature_history[symbol][-self.max_history:]

            self.logger.info(f"Extracted {len(combined_features)} features for {symbol}")
            return image_features

        except Exception as e:
            self.logger.error(f"Error extracting features from chart: {e}")
            return self._create_empty_features(symbol, timestamp)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess chart image for feature extraction"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Resize to standard dimensions
            resized = cv2.resize(gray, (self.image_width, self.image_height))

            # Apply normalization
            if self.normalization == 'zscore':
                resized = (resized - np.mean(resized)) / (np.std(resized) + 1e-8)
            elif self.normalization == 'minmax':
                scaler = MinMaxScaler()
                resized = scaler.fit_transform(resized.reshape(-1, 1)).reshape(resized.shape)

            # Apply slight Gaussian blur to reduce noise
            processed = cv2.GaussianBlur(resized.astype(np.float32), (3, 3), 0)

            return processed

        except Exception as e:
            self.logger.warning(f"Error preprocessing image: {e}")
            return image

    def _extract_basic_stats(self, image: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features"""
        try:
            features = {}

            # Basic statistics
            features['mean_intensity'] = np.mean(image)
            features['std_intensity'] = np.std(image)
            features['min_intensity'] = np.min(image)
            features['max_intensity'] = np.max(image)
            features['median_intensity'] = np.median(image)

            # Percentiles
            for p in [10, 25, 75, 90]:
                features[f'percentile_{p}'] = np.percentile(image, p)

            # Image moments
            moments = cv2.moments(image)
            features['moment_m00'] = moments['m00']
            features['moment_mu02'] = moments['mu02']
            features['moment_mu20'] = moments['mu20']
            features['moment_mu11'] = moments['mu11']

            # Entropy
            hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 1))
            hist = hist / np.sum(hist)
            features['entropy'] = -np.sum(hist * np.log2(hist + 1e-8))

            # Contrast and homogeneity (from GLCM)
            glcm = feature.graycomatrix((image * 255).astype(np.uint8),
                                      distances=[1], angles=[0],
                                      levels=256, symmetric=True, normed=True)
            features['contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
            features['homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]

            return features

        except Exception as e:
            self.logger.warning(f"Error extracting basic stats: {e}")
            return {}

    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture-based features"""
        try:
            features = {}

            # HOG features
            hog_features = hog(image, orientations=self.hog_orientations,
                             pixels_per_cell=self.hog_pixels_per_cell,
                             cells_per_block=self.hog_cells_per_block,
                             feature_vector=True)
            features['hog_mean'] = np.mean(hog_features)
            features['hog_std'] = np.std(hog_features)
            features['hog_max'] = np.max(hog_features)

            # LBP features
            lbp = local_binary_pattern(image, self.lbp_points, self.lbp_radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), density=True)
            for i, val in enumerate(lbp_hist):
                features[f'lbp_hist_{i}'] = val

            # Gabor filter responses
            for i, kernel in enumerate(self.gabor_kernels[:10]):  # Limit to 10 kernels
                filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                features[f'gabor_energy_{i}'] = np.sum(filtered ** 2)
                features[f'gabor_mean_{i}'] = np.mean(filtered)
                features[f'gabor_std_{i}'] = np.std(filtered)

            # Haralick texture features
            glcm = feature.graycomatrix((image * 255).astype(np.uint8),
                                      distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                      levels=256, symmetric=True, normed=True)

            features['asm'] = np.mean(feature.graycoprops(glcm, 'ASM'))
            features['correlation'] = np.mean(feature.graycoprops(glcm, 'correlation'))
            features['energy'] = np.mean(feature.graycoprops(glcm, 'energy'))

            return features

        except Exception as e:
            self.logger.warning(f"Error extracting texture features: {e}")
            return {}

    def _extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features"""
        try:
            features = {}

            # Edge detection
            edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (image.shape[0] * image.shape[1])

            # Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Largest contour features
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)

                features['contour_area'] = area
                features['contour_perimeter'] = perimeter
                features['contour_circularity'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                features['bounding_width'] = w
                features['bounding_height'] = h
                features['bounding_aspect_ratio'] = w / h if h > 0 else 0

                # Convex hull
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                features['convexity'] = area / hull_area if hull_area > 0 else 0

                # Contour approximation
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                features['approx_vertices'] = len(approx)

            # Skeleton analysis
            skeleton = morphology.skeletonize(edges > 0)
            features['skeleton_density'] = np.sum(skeleton) / (image.shape[0] * image.shape[1])

            # Region properties
            labeled_image = measure.label(edges > 0)
            regions = measure.regionprops(labeled_image)

            if regions:
                largest_region = max(regions, key=lambda r: r.area)
                features['region_area'] = largest_region.area
                features['region_perimeter'] = largest_region.perimeter
                features['region_eccentricity'] = largest_region.eccentricity
                features['region_solidity'] = largest_region.solidity

            return features

        except Exception as e:
            self.logger.warning(f"Error extracting shape features: {e}")
            return {}

    def _extract_frequency_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features"""
        try:
            features = {}

            # 2D FFT
            fft_image = fft2(image)
            fft_shifted = fftshift(fft_image)

            # Magnitude spectrum
            magnitude = np.abs(fft_shifted)
            features['fft_mean_magnitude'] = np.mean(magnitude)
            features['fft_std_magnitude'] = np.std(magnitude)
            features['fft_max_magnitude'] = np.max(magnitude)

            # Spectral centroid
            rows, cols = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
            total_magnitude = np.sum(magnitude)
            if total_magnitude > 0:
                centroid_row = np.sum(rows * magnitude) / total_magnitude
                centroid_col = np.sum(cols * magnitude) / total_magnitude
                features['spectral_centroid_row'] = centroid_row
                features['spectral_centroid_col'] = centroid_col

            # High frequency content
            center_row, center_col = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            high_freq_mask = np.zeros_like(magnitude, dtype=bool)
            high_freq_mask[center_row-10:center_row+10, center_col-10:center_col+10] = True
            features['high_freq_energy'] = np.sum(magnitude[~high_freq_mask])
            features['low_freq_energy'] = np.sum(magnitude[high_freq_mask])

            # Spectral moments
            features['spectral_moment_2'] = np.sum(magnitude ** 2)
            features['spectral_moment_3'] = np.sum(magnitude ** 3)
            features['spectral_moment_4'] = np.sum(magnitude ** 4)

            # Wavelet-like features (simplified)
            # Low-pass filter
            kernel = np.ones((5, 5)) / 25
            low_pass = cv2.filter2D(image, -1, kernel)
            features['low_pass_energy'] = np.sum(low_pass ** 2)

            # High-pass filter
            high_pass = image - low_pass
            features['high_pass_energy'] = np.sum(high_pass ** 2)

            return features

        except Exception as e:
            self.logger.warning(f"Error extracting frequency features: {e}")
            return {}

    def _extract_pattern_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract pattern-based features"""
        try:
            features = {}

            # Line detection using Hough transform
            edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 50)

            if lines is not None:
                features['num_lines'] = len(lines)
                # Analyze line orientations
                orientations = lines[:, 0, 1]
                features['line_orientation_mean'] = np.mean(orientations)
                features['line_orientation_std'] = np.std(orientations)
                # Count horizontal and vertical lines
                horizontal = np.sum(np.abs(orientations) < np.pi/18)  # Within 10 degrees of horizontal
                vertical = np.sum(np.abs(orientations - np.pi/2) < np.pi/18)  # Within 10 degrees of vertical
                features['horizontal_lines'] = horizontal
                features['vertical_lines'] = vertical
            else:
                features['num_lines'] = 0
                features['line_orientation_mean'] = 0
                features['line_orientation_std'] = 0
                features['horizontal_lines'] = 0
                features['vertical_lines'] = 0

            # Corner detection
            corners = cv2.goodFeaturesToTrack((image * 255).astype(np.uint8), 100, 0.01, 10)
            features['num_corners'] = corners.shape[0] if corners is not None else 0

            # Blob detection
            detector = cv2.SimpleBlobDetector_create()
            keypoints = detector.detect((image * 255).astype(np.uint8))
            features['num_blobs'] = len(keypoints)

            # Pattern clustering features
            # Sample points from image
            sample_points = image.flatten()[::100]  # Sample every 100th pixel
            if len(sample_points) > 10:
                try:
                    # Cluster the sampled points
                    sample_points = sample_points.reshape(-1, 1)
                    clusters = self.pattern_cluster_model.fit_predict(sample_points)

                    # Cluster statistics
                    unique_clusters, counts = np.unique(clusters, return_counts=True)
                    features['num_clusters'] = len(unique_clusters)
                    features['largest_cluster_size'] = np.max(counts)
                    features['cluster_entropy'] = -np.sum((counts / len(clusters)) * np.log2(counts / len(clusters)))

                except Exception:
                    features['num_clusters'] = 0
                    features['largest_cluster_size'] = 0
                    features['cluster_entropy'] = 0

            return features

        except Exception as e:
            self.logger.warning(f"Error extracting pattern features: {e}")
            return {}

    def _extract_fractal_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract fractal dimension features"""
        try:
            features = {}

            # Box counting method for fractal dimension
            def box_count(img, box_sizes):
                """Calculate box counting dimension"""
                dimensions = []
                for size in box_sizes:
                    # Create grid
                    h, w = img.shape
                    boxes_h = int(np.ceil(h / size))
                    boxes_w = int(np.ceil(w / size))

                    # Count boxes with pixels
                    count = 0
                    for i in range(boxes_h):
                        for j in range(boxes_w):
                            box = img[i*size:(i+1)*size, j*size:(j+1)*size]
                            if np.sum(box) > 0:
                                count += 1

                    dimensions.append((size, count))

                return dimensions

            # Binary image for fractal analysis
            binary = (image > np.mean(image)).astype(np.uint8)

            # Box sizes for analysis
            box_sizes = [2, 4, 8, 16, 32]

            # Calculate fractal dimension
            dimensions = box_count(binary, box_sizes)

            if len(dimensions) > 2:
                # Log-log regression for fractal dimension
                sizes = np.array([d[0] for d in dimensions])
                counts = np.array([d[1] for d in dimensions])

                log_sizes = np.log(1/sizes)
                log_counts = np.log(counts)

                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
                features['fractal_dimension'] = -slope  # Negative because we use 1/sizes
                features['fractal_r_squared'] = r_value ** 2
            else:
                features['fractal_dimension'] = 0
                features['fractal_r_squared'] = 0

            # Lacunarity analysis
            def calculate_lacunarity(img, box_sizes):
                """Calculate lacunarity for different box sizes"""
                lacunarities = []
                for size in box_sizes:
                    h, w = img.shape
                    boxes_h = int(np.floor(h / size))
                    boxes_w = int(np.floor(w / size))

                    if boxes_h == 0 or boxes_w == 0:
                        continue

                    box_masses = []
                    for i in range(boxes_h):
                        for j in range(boxes_w):
                            box = img[i*size:(i+1)*size, j*size:(j+1)*size]
                            box_masses.append(np.sum(box))

                    box_masses = np.array(box_masses)
                    mean_mass = np.mean(box_masses)
                    var_mass = np.var(box_masses)

                    if mean_mass > 0:
                        lacunarity = var_mass / (mean_mass ** 2) + 1
                        lacunarities.append(lacunarity)
                    else:
                        lacunarities.append(1.0)

                return lacunarities

            lacunarities = calculate_lacunarity(binary, box_sizes[:3])  # Use smaller boxes
            features['mean_lacunarity'] = np.mean(lacunarities) if lacunarities else 0
            features['std_lacunarity'] = np.std(lacunarities) if lacunarities else 0

            return features

        except Exception as e:
            self.logger.warning(f"Error extracting fractal features: {e}")
            return {}

    def _combine_features(self, feature_dicts: List[Dict[str, float]]) -> np.ndarray:
        """Combine all feature dictionaries into a single array"""
        try:
            all_features = {}
            for feature_dict in feature_dicts:
                all_features.update(feature_dict)

            # Convert to numpy array
            feature_values = np.array(list(all_features.values()))

            # Handle NaN and infinite values
            feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=1e6, neginf=-1e6)

            # Apply scaling
            if len(feature_values) > 0:
                feature_values = self.feature_scaler.fit_transform(feature_values.reshape(1, -1)).flatten()

                # Apply PCA if enabled
                if self.use_pca and len(feature_values) > self.pca_components:
                    feature_values = self.pca_model.fit_transform(feature_values.reshape(1, -1)).flatten()

            return feature_values

        except Exception as e:
            self.logger.warning(f"Error combining features: {e}")
            return np.array([])

    def _create_empty_features(self, symbol: str, timestamp: datetime) -> ImageFeatures:
        """Create empty features object for error cases"""
        return ImageFeatures(
            timestamp=timestamp,
            symbol=symbol,
            basic_stats={},
            texture_features={},
            shape_features={},
            frequency_features={},
            pattern_features={},
            fractal_features={},
            combined_features=np.array([])
        )

    def extract_temporal_features(self, symbol: str, window_size: int = 10) -> Optional[np.ndarray]:
        """
        Extract temporal features from feature history

        Args:
            symbol: Trading symbol
            window_size: Size of temporal window

        Returns:
            Temporal features array
        """
        try:
            if symbol not in self.feature_history:
                return None

            history = self.feature_history[symbol]
            if len(history) < window_size:
                return None

            # Get recent features
            recent_features = [f.combined_features for f in history[-window_size:]]

            # Stack features
            temporal_matrix = np.stack(recent_features)

            # Calculate temporal statistics
            temporal_features = []

            # Mean and std across time
            temporal_features.extend(np.mean(temporal_matrix, axis=0))
            temporal_features.extend(np.std(temporal_matrix, axis=0))

            # Trend features (linear regression slope)
            for i in range(temporal_matrix.shape[1]):
                try:
                    slope, _, _, _, _ = stats.linregress(range(window_size), temporal_matrix[:, i])
                    temporal_features.append(slope)
                except:
                    temporal_features.append(0.0)

            # Autocorrelation features
            for i in range(min(5, temporal_matrix.shape[1])):  # First 5 features
                try:
                    autocorr = np.correlate(temporal_matrix[:, i], temporal_matrix[:, i], mode='full')
                    autocorr = autocorr[autocorr.size // 2:]  # Get positive lags
                    temporal_features.extend(autocorr[1:4])  # First 3 autocorrelations
                except:
                    temporal_features.extend([0.0, 0.0, 0.0])

            return np.array(temporal_features)

        except Exception as e:
            self.logger.warning(f"Error extracting temporal features: {e}")
            return None

    def batch_extract_features(self, chart_images: List[np.ndarray],
                             symbols: List[str], timestamps: List[datetime]) -> FeatureExtractionResult:
        """
        Extract features from batch of chart images

        Args:
            chart_images: List of chart images
            symbols: List of trading symbols
            timestamps: List of timestamps

        Returns:
            Batch feature extraction result
        """
        try:
            features = []

            for image, symbol, timestamp in zip(chart_images, symbols, timestamps):
                image_features = self.extract_features_from_chart(image, symbol, timestamp)
                features.append(image_features)

            # Create feature matrix
            feature_matrix = np.stack([f.combined_features for f in features])

            # Get feature names (simplified)
            feature_names = [f'feature_{i}' for i in range(feature_matrix.shape[1])]

            # Create metadata
            metadata = {
                'num_images': len(chart_images),
                'feature_dimensions': feature_matrix.shape[1],
                'extraction_timestamp': datetime.now(),
                'config': self.config
            }

            result = FeatureExtractionResult(
                features=features,
                feature_matrix=feature_matrix,
                feature_names=feature_names,
                metadata=metadata
            )

            self.logger.info(f"Batch extracted features from {len(chart_images)} images")
            return result

        except Exception as e:
            self.logger.error(f"Error in batch feature extraction: {e}")
            return FeatureExtractionResult(
                features=[],
                feature_matrix=np.array([]),
                feature_names=[],
                metadata={'error': str(e)}
            )

    def get_feature_importance(self, feature_matrix: np.ndarray,
                             target_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance using correlation with targets

        Args:
            feature_matrix: Feature matrix
            target_values: Target values for importance calculation

        Returns:
            Feature importance dictionary
        """
        try:
            importance = {}

            for i in range(feature_matrix.shape[1]):
                feature_values = feature_matrix[:, i]

                # Calculate correlation
                if np.std(feature_values) > 0 and np.std(target_values) > 0:
                    correlation = np.corrcoef(feature_values, target_values)[0, 1]
                    importance[f'feature_{i}'] = abs(correlation)
                else:
                    importance[f'feature_{i}'] = 0.0

            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            return importance

        except Exception as e:
            self.logger.warning(f"Error calculating feature importance: {e}")
            return {}

    def save_feature_extractor(self, filepath: str) -> None:
        """Save feature extractor state"""
        try:
            import pickle

            state = {
                'config': self.config,
                'feature_scaler': self.feature_scaler,
                'pca_model': self.pca_model,
                'pattern_cluster_model': self.pattern_cluster_model,
                'gabor_kernels': self.gabor_kernels,
                'feature_history': self.feature_history,
                'timestamp': datetime.now()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            self.logger.info(f"Feature extractor saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving feature extractor: {e}")

    def load_feature_extractor(self, filepath: str) -> None:
        """Load feature extractor state"""
        try:
            import pickle

            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.config = state.get('config', {})
            self.feature_scaler = state.get('feature_scaler', StandardScaler())
            self.pca_model = state.get('pca_model', PCA())
            self.pattern_cluster_model = state.get('pattern_cluster_model', KMeans())
            self.gabor_kernels = state.get('gabor_kernels', [])
            self.feature_history = state.get('feature_history', {})

            self.logger.info(f"Feature extractor loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading feature extractor: {e}")

    def get_feature_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get feature statistics for a symbol"""
        try:
            if symbol not in self.feature_history:
                return {}

            history = self.feature_history[symbol]
            if not history:
                return {}

            # Calculate statistics
            feature_matrices = [f.combined_features for f in history]
            feature_matrix = np.stack(feature_matrices)

            stats = {
                'num_samples': len(history),
                'feature_dimensions': feature_matrix.shape[1],
                'mean_features': np.mean(feature_matrix, axis=0).tolist(),
                'std_features': np.std(feature_matrix, axis=0).tolist(),
                'min_features': np.min(feature_matrix, axis=0).tolist(),
                'max_features': np.max(feature_matrix, axis=0).tolist(),
                'last_update': history[-1].timestamp
            }

            return stats

        except Exception as e:
            self.logger.warning(f"Error getting feature statistics: {e}")
            return {}