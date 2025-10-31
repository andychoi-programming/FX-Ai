"""
System Health Monitoring Module for FX-Ai Trading System

This module provides comprehensive monitoring of system health including:
- Memory and CPU usage tracking
- API response time monitoring
- Error rate tracking and alerting
- Performance metrics collection
- Automated health status reporting
- Resource usage alerts and warnings
"""

import psutil
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import json
import os


@dataclass
class HealthMetrics:
    """Container for system health metrics"""
    timestamp: datetime
    memory_usage_percent: float
    cpu_usage_percent: float
    disk_usage_percent: float
    network_connections: int
    thread_count: int
    process_uptime: float
    api_response_times: Dict[str, float] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthThresholds:
    """Configurable thresholds for health monitoring"""
    memory_warning: float = 80.0  # Memory usage warning threshold (%)
    memory_critical: float = 90.0  # Memory usage critical threshold (%)
    cpu_warning: float = 70.0  # CPU usage warning threshold (%)
    cpu_critical: float = 85.0  # CPU usage critical threshold (%)
    disk_warning: float = 85.0  # Disk usage warning threshold (%)
    disk_critical: float = 95.0  # Disk usage critical threshold (%)
    api_response_warning: float = 5.0  # API response time warning threshold (seconds)
    api_response_critical: float = 10.0  # API response time critical threshold (seconds)
    error_rate_warning: float = 0.1  # Error rate warning threshold (fraction)
    error_rate_critical: float = 0.25  # Error rate critical threshold (fraction)
    max_consecutive_failures: int = 5  # Max consecutive API failures before alert


class SystemHealthMonitor:
    """
    Comprehensive system health monitoring with automated alerts and performance tracking
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the system health monitor

        Args:
            config: Configuration dictionary with monitoring settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Monitoring settings
        self.enabled = config.get('enabled', True)
        self.monitoring_interval = config.get('monitoring_interval', 30)  # seconds
        self.metrics_history_size = config.get('metrics_history_size', 1000)
        self.alert_cooldown = config.get('alert_cooldown', 300)  # 5 minutes between alerts

        # Thresholds
        thresholds_config = config.get('thresholds', {})
        self.thresholds = HealthThresholds(
            memory_warning=thresholds_config.get('memory_warning', 80.0),
            memory_critical=thresholds_config.get('memory_critical', 90.0),
            cpu_warning=thresholds_config.get('cpu_warning', 70.0),
            cpu_critical=thresholds_config.get('cpu_critical', 85.0),
            disk_warning=thresholds_config.get('disk_warning', 85.0),
            disk_critical=thresholds_config.get('disk_critical', 95.0),
            api_response_warning=thresholds_config.get('api_response_warning', 5.0),
            api_response_critical=thresholds_config.get('api_response_critical', 10.0),
            error_rate_warning=thresholds_config.get('error_rate_warning', 0.1),
            error_rate_critical=thresholds_config.get('error_rate_critical', 0.25),
            max_consecutive_failures=thresholds_config.get('max_consecutive_failures', 5)
        )

        # Data structures
        self.metrics_history: deque = deque(maxlen=self.metrics_history_size)
        self.api_calls: Dict[str, List[Tuple[datetime, float]]] = {}
        self.errors: Dict[str, List[Tuple[datetime, str]]] = {}
        self.alerts_history: List[Dict[str, Any]] = []
        self.last_alert_times: Dict[str, datetime] = {}

        # Monitoring state
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.start_time = datetime.now()

        # Performance tracking
        self.api_call_counts: Dict[str, int] = {}
        self.api_response_times: Dict[str, List[float]] = {}
        self.consecutive_failures: Dict[str, int] = {}

        self.logger.info("System Health Monitor initialized")

    def start_monitoring(self) -> None:
        """Start the background monitoring thread"""
        if not self.enabled:
            self.logger.info("System health monitoring is disabled")
            return

        if self.is_monitoring:
            self.logger.warning("Monitoring is already running")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info(f"System health monitoring started (interval: {self.monitoring_interval}s)")

    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("System health monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs in background thread"""
        while self.is_monitoring:
            try:
                # Collect current metrics
                metrics = self._collect_system_metrics()

                # Store metrics
                self.metrics_history.append(metrics)

                # Check health status
                health_status = self._check_health_status(metrics)

                # Generate alerts if needed
                if health_status['status'] != 'healthy':
                    self._generate_alerts(health_status, metrics)

                # Clean up old data periodically
                if len(self.metrics_history) % 100 == 0:  # Every 100 measurements
                    self._cleanup_old_data()

                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_system_metrics(self) -> HealthMetrics:
        """Collect comprehensive system health metrics"""
        try:
            # Get process information
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=1.0)

            # System-wide metrics
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent(interval=1.0)
            disk_usage = psutil.disk_usage('/')

            # Network connections
            network_connections = len(psutil.net_connections())

            # Thread count
            thread_count = process.num_threads()

            # Process uptime
            process_uptime = time.time() - process.create_time()

            # API response times (rolling averages)
            api_response_times = {}
            for api_name, times in self.api_response_times.items():
                if times:
                    api_response_times[api_name] = np.mean(times[-100:])  # Last 100 calls

            # Error counts (recent errors)
            error_counts = {}
            cutoff_time = datetime.now() - timedelta(hours=1)  # Last hour
            for error_type, errors in self.errors.items():
                recent_errors = [e for e in errors if e[0] > cutoff_time]
                error_counts[error_type] = len(recent_errors)

            # Performance metrics
            performance_metrics = {
                'total_api_calls': sum(self.api_call_counts.values()),
                'unique_apis': len(self.api_calls),
                'total_errors': sum(error_counts.values()),
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
            }

            return HealthMetrics(
                timestamp=datetime.now(),
                memory_usage_percent=system_memory.percent,
                cpu_usage_percent=system_cpu,
                disk_usage_percent=disk_usage.percent,
                network_connections=network_connections,
                thread_count=thread_count,
                process_uptime=process_uptime,
                api_response_times=api_response_times,
                error_counts=error_counts,
                performance_metrics=performance_metrics
            )

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            # Return minimal metrics on error
            return HealthMetrics(
                timestamp=datetime.now(),
                memory_usage_percent=0.0,
                cpu_usage_percent=0.0,
                disk_usage_percent=0.0,
                network_connections=0,
                thread_count=0,
                process_uptime=0.0
            )

    def _check_health_status(self, metrics: HealthMetrics) -> Dict[str, Any]:
        """Check overall health status based on metrics and thresholds"""
        issues = []
        warnings = []
        critical_issues = []

        # Memory checks
        if metrics.memory_usage_percent >= self.thresholds.memory_critical:
            critical_issues.append(f"Memory usage critical: {metrics.memory_usage_percent:.1f}%")
        elif metrics.memory_usage_percent >= self.thresholds.memory_warning:
            warnings.append(f"Memory usage high: {metrics.memory_usage_percent:.1f}%")

        # CPU checks
        if metrics.cpu_usage_percent >= self.thresholds.cpu_critical:
            critical_issues.append(f"CPU usage critical: {metrics.cpu_usage_percent:.1f}%")
        elif metrics.cpu_usage_percent >= self.thresholds.cpu_warning:
            warnings.append(f"CPU usage high: {metrics.cpu_usage_percent:.1f}%")

        # Disk checks
        if metrics.disk_usage_percent >= self.thresholds.disk_critical:
            critical_issues.append(f"Disk usage critical: {metrics.disk_usage_percent:.1f}%")
        elif metrics.disk_usage_percent >= self.thresholds.disk_warning:
            warnings.append(f"Disk usage high: {metrics.disk_usage_percent:.1f}%")

        # API response time checks
        for api_name, response_time in metrics.api_response_times.items():
            if response_time >= self.thresholds.api_response_critical:
                critical_issues.append(f"{api_name} response time critical: {response_time:.2f}s")
            elif response_time >= self.thresholds.api_response_warning:
                warnings.append(f"{api_name} response time slow: {response_time:.2f}s")

        # Error rate checks
        total_calls = sum(self.api_call_counts.values())
        total_errors = sum(metrics.error_counts.values())

        if total_calls > 0:
            error_rate = total_errors / total_calls
            if error_rate >= self.thresholds.error_rate_critical:
                critical_issues.append(f"Error rate critical: {error_rate:.3f}")
            elif error_rate >= self.thresholds.error_rate_warning:
                warnings.append(f"Error rate high: {error_rate:.3f}")

        # Consecutive failure checks
        for api_name, failures in self.consecutive_failures.items():
            if failures >= self.thresholds.max_consecutive_failures:
                critical_issues.append(f"{api_name} has {failures} consecutive failures")

        # Determine overall status
        if critical_issues:
            status = 'critical'
        elif warnings:
            status = 'warning'
        else:
            status = 'healthy'

        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'critical_issues': critical_issues,
            'timestamp': metrics.timestamp
        }

    def _generate_alerts(self, health_status: Dict[str, Any], metrics: HealthMetrics) -> None:
        """Generate alerts based on health status"""
        current_time = datetime.now()

        # Check alert cooldown
        alert_key = health_status['status']
        last_alert = self.last_alert_times.get(alert_key)

        if last_alert and (current_time - last_alert).total_seconds() < self.alert_cooldown:
            return  # Too soon for another alert

        # Create alert
        alert = {
            'timestamp': current_time,
            'status': health_status['status'],
            'issues': health_status.get('issues', []),
            'warnings': health_status.get('warnings', []),
            'critical_issues': health_status.get('critical_issues', []),
            'metrics': {
                'memory_percent': metrics.memory_usage_percent,
                'cpu_percent': metrics.cpu_usage_percent,
                'disk_percent': metrics.disk_usage_percent,
                'api_response_times': metrics.api_response_times,
                'error_counts': metrics.error_counts
            }
        }

        # Store alert
        self.alerts_history.append(alert)
        self.last_alert_times[alert_key] = current_time

        # Log alert
        alert_message = f"Health Alert [{alert['status'].upper()}]: "
        if alert['critical_issues']:
            alert_message += f"Critical: {', '.join(alert['critical_issues'])}"
        if alert['warnings']:
            alert_message += f" Warnings: {', '.join(alert['warnings'])}"

        if alert['status'] == 'critical':
            self.logger.critical(alert_message)
        elif alert['status'] == 'warning':
            self.logger.warning(alert_message)
        else:
            self.logger.info(alert_message)

    def record_api_call(self, api_name: str, response_time: float, success: bool = True,
                       error_message: str = None) -> None:
        """Record an API call for monitoring"""
        if not self.enabled:
            return

        current_time = datetime.now()

        # Record response time
        if api_name not in self.api_response_times:
            self.api_response_times[api_name] = []
        self.api_response_times[api_name].append(response_time)

        # Keep only recent response times (last 1000 calls per API)
        if len(self.api_response_times[api_name]) > 1000:
            self.api_response_times[api_name] = self.api_response_times[api_name][-1000:]

        # Record call count
        self.api_call_counts[api_name] = self.api_call_counts.get(api_name, 0) + 1

        # Record API call timing
        if api_name not in self.api_calls:
            self.api_calls[api_name] = []
        self.api_calls[api_name].append((current_time, response_time))

        # Keep only recent calls (last 1000)
        if len(self.api_calls[api_name]) > 1000:
            self.api_calls[api_name] = self.api_calls[api_name][-1000:]

        # Handle failures
        if not success:
            if api_name not in self.errors:
                self.errors[api_name] = []
            self.errors[api_name].append((current_time, error_message or "Unknown error"))

            # Update consecutive failures
            self.consecutive_failures[api_name] = self.consecutive_failures.get(api_name, 0) + 1
        else:
            # Reset consecutive failures on success
            self.consecutive_failures[api_name] = 0

    def record_error(self, error_type: str, error_message: str) -> None:
        """Record a general error for monitoring"""
        if not self.enabled:
            return

        current_time = datetime.now()

        if error_type not in self.errors:
            self.errors[error_type] = []
        self.errors[error_type].append((current_time, error_message))

        # Keep only recent errors (last 1000 per type)
        if len(self.errors[error_type]) > 1000:
            self.errors[error_type] = self.errors[error_type][-1000:]

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        if not self.metrics_history:
            return {'status': 'no_data', 'message': 'No metrics collected yet'}

        latest_metrics = self.metrics_history[-1]

        # Calculate averages over last hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        if recent_metrics:
            avg_memory = np.mean([m.memory_usage_percent for m in recent_metrics])
            avg_cpu = np.mean([m.cpu_usage_percent for m in recent_metrics])
            avg_disk = np.mean([m.disk_usage_percent for m in recent_metrics])
        else:
            avg_memory = latest_metrics.memory_usage_percent
            avg_cpu = latest_metrics.cpu_usage_percent
            avg_disk = latest_metrics.disk_usage_percent

        # Get current health status
        health_status = self._check_health_status(latest_metrics)

        return {
            'status': health_status['status'],
            'timestamp': latest_metrics.timestamp,
            'system_metrics': {
                'memory_usage_percent': latest_metrics.memory_usage_percent,
                'cpu_usage_percent': latest_metrics.cpu_usage_percent,
                'disk_usage_percent': latest_metrics.disk_usage_percent,
                'network_connections': latest_metrics.network_connections,
                'thread_count': latest_metrics.thread_count,
                'process_uptime_hours': latest_metrics.process_uptime / 3600
            },
            'averages_last_hour': {
                'memory_percent': avg_memory,
                'cpu_percent': avg_cpu,
                'disk_percent': avg_disk
            },
            'api_metrics': {
                'response_times': latest_metrics.api_response_times,
                'call_counts': self.api_call_counts,
                'error_counts': latest_metrics.error_counts,
                'consecutive_failures': self.consecutive_failures
            },
            'performance_metrics': latest_metrics.performance_metrics,
            'issues': health_status.get('issues', []),
            'warnings': health_status.get('warnings', []),
            'critical_issues': health_status.get('critical_issues', []),
            'recent_alerts': self.alerts_history[-10:] if self.alerts_history else []  # Last 10 alerts
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        stats = {
            'total_api_calls': sum(self.api_call_counts.values()),
            'total_errors': sum(len(errors) for errors in self.errors.values()),
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'monitoring_enabled': self.is_monitoring,
            'metrics_collected': len(self.metrics_history),
            'alerts_generated': len(self.alerts_history)
        }

        # API-specific stats
        api_stats = {}
        for api_name in self.api_calls.keys():
            response_times = self.api_response_times.get(api_name, [])
            if response_times:
                api_stats[api_name] = {
                    'call_count': self.api_call_counts.get(api_name, 0),
                    'avg_response_time': np.mean(response_times),
                    'min_response_time': np.min(response_times),
                    'max_response_time': np.max(response_times),
                    'p95_response_time': np.percentile(response_times, 95),
                    'error_count': len(self.errors.get(api_name, [])),
                    'consecutive_failures': self.consecutive_failures.get(api_name, 0)
                }

        stats['api_stats'] = api_stats
        return stats

    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data to prevent memory issues"""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of data

        # Clean up API calls
        for api_name in list(self.api_calls.keys()):
            self.api_calls[api_name] = [
                call for call in self.api_calls[api_name]
                if call[0] > cutoff_time
            ]
            if not self.api_calls[api_name]:
                del self.api_calls[api_name]

        # Clean up errors
        for error_type in list(self.errors.keys()):
            self.errors[error_type] = [
                error for error in self.errors[error_type]
                if error[0] > cutoff_time
            ]
            if not self.errors[error_type]:
                del self.errors[error_type]

        # Clean up old alerts (keep last 100)
        if len(self.alerts_history) > 100:
            self.alerts_history = self.alerts_history[-100:]

        self.logger.debug("Cleaned up old monitoring data")

    def export_health_data(self, filepath: str) -> None:
        """Export health monitoring data to JSON file"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'config': {
                'enabled': self.enabled,
                'monitoring_interval': self.monitoring_interval,
                'thresholds': {
                    'memory_warning': self.thresholds.memory_warning,
                    'memory_critical': self.thresholds.memory_critical,
                    'cpu_warning': self.thresholds.cpu_warning,
                    'cpu_critical': self.thresholds.cpu_critical,
                    'disk_warning': self.thresholds.disk_warning,
                    'disk_critical': self.thresholds.disk_critical,
                    'api_response_warning': self.thresholds.api_response_warning,
                    'api_response_critical': self.thresholds.api_response_critical,
                    'error_rate_warning': self.thresholds.error_rate_warning,
                    'error_rate_critical': self.thresholds.error_rate_critical,
                    'max_consecutive_failures': self.thresholds.max_consecutive_failures
                }
            },
            'current_health': self.get_health_report(),
            'performance_stats': self.get_performance_stats(),
            'recent_alerts': self.alerts_history[-50:] if self.alerts_history else [],  # Last 50 alerts
            'metrics_history': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'memory_usage_percent': m.memory_usage_percent,
                    'cpu_usage_percent': m.cpu_usage_percent,
                    'disk_usage_percent': m.disk_usage_percent,
                    'network_connections': m.network_connections,
                    'thread_count': m.thread_count,
                    'process_uptime': m.process_uptime,
                    'api_response_times': m.api_response_times,
                    'error_counts': m.error_counts,
                    'performance_metrics': m.performance_metrics
                }
                for m in list(self.metrics_history)[-100:]  # Last 100 metrics
            ]
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"Health data exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export health data: {e}")