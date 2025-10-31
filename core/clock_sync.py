"""
FX-Ai Clock Synchronization Module
Ensures accurate time synchronization between local system, NTP servers, and MT5 platform
Critical for precise trade timing and signal execution
"""

import logging
import time
import threading
import socket
import struct
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple
import pytz

logger = logging.getLogger(__name__)

class ClockSynchronizer:
    """
    Advanced clock synchronization system for trading applications.
    Maintains accurate time alignment between local system, NTP servers, and MT5 platform.
    """

    def __init__(self, mt5_connector=None, sync_interval: int = 300, max_drift: float = 1.0):
        """
        Initialize clock synchronizer

        Args:
            mt5_connector: MT5 connector instance for server time checks
            sync_interval: Seconds between synchronization checks (default: 5 minutes)
            max_drift: Maximum allowed time drift in seconds before correction (default: 1.0s)
        """
        self.mt5_connector = mt5_connector
        self.sync_interval = sync_interval
        self.max_drift = max_drift

        # NTP servers (reliable public NTP servers)
        self.ntp_servers = [
            ('pool.ntp.org', 123),
            ('time.nist.gov', 123),
            ('time.google.com', 123),
            ('ntp.ubuntu.com', 123)
        ]

        # Synchronization status
        self.last_sync_time = None
        self.time_drift = 0.0
        self.is_synchronized = False
        self.sync_thread = None
        self.running = False

        # Timezone handling
        self.utc_tz = pytz.UTC
        self.local_tz = pytz.timezone('UTC')  # Default to UTC, can be configured

        logger.info("Clock Synchronizer initialized")

    def start_sync_thread(self):
        """Start the background synchronization thread"""
        if self.sync_thread is None or not self.sync_thread.is_alive():
            self.running = True
            self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self.sync_thread.start()
            logger.info("Clock synchronization thread started")

    def stop_sync_thread(self):
        """Stop the background synchronization thread"""
        self.running = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5.0)
            logger.info("Clock synchronization thread stopped")

    def _sync_loop(self):
        """Main synchronization loop running in background thread"""
        while self.running:
            try:
                self.perform_sync()
            except Exception as e:
                logger.error(f"Clock sync error: {e}")

            # Wait for next sync interval
            time.sleep(self.sync_interval)

    def perform_sync(self) -> Dict:
        """
        Perform complete time synchronization check

        Returns:
            Dict containing sync results and status
        """
        results = {
            'timestamp': datetime.now(timezone.utc),
            'ntp_time': None,
            'mt5_time': None,
            'local_time': datetime.now(timezone.utc),
            'drift_ntp': None,
            'drift_mt5': None,
            'is_synced': False,
            'corrections_made': []
        }

        try:
            # Get NTP time
            ntp_time = self._get_ntp_time()
            if ntp_time:
                results['ntp_time'] = ntp_time
                drift_ntp = abs((results['local_time'] - ntp_time).total_seconds())
                results['drift_ntp'] = drift_ntp

                # Check if NTP drift exceeds threshold
                if drift_ntp > self.max_drift:
                    logger.warning(f"NTP time drift detected: {drift_ntp:.3f}s")
                    # Note: We don't auto-correct system time as it requires admin privileges
                    # Instead, we log the issue and use NTP time for calculations

            # Get MT5 server time
            if self.mt5_connector:
                mt5_time = self.mt5_connector.get_server_time()
                if mt5_time:
                    results['mt5_time'] = mt5_time
                    drift_mt5 = abs((results['local_time'] - mt5_time).total_seconds())
                    results['drift_mt5'] = drift_mt5

                    if drift_mt5 > self.max_drift:
                        logger.warning(f"MT5 time drift detected: {drift_mt5:.3f}s")

            # Determine overall sync status
            max_drift = max(
                results['drift_ntp'] or 0,
                results['drift_mt5'] or 0
            )

            results['is_synced'] = max_drift <= self.max_drift
            self.is_synchronized = results['is_synced']
            self.last_sync_time = results['timestamp']
            self.time_drift = max_drift

            # Log sync status
            if results['is_synced']:
                logger.info(f"Clock synchronized - Max drift: {max_drift:.3f}s")
            else:
                logger.warning(f"Clock out of sync - Max drift: {max_drift:.3f}s")

        except Exception as e:
            logger.error(f"Sync check failed: {e}")

        return results

    def _get_ntp_time(self) -> Optional[datetime]:
        """
        Get time from NTP servers with fallback using raw NTP protocol

        Returns:
            UTC datetime from NTP or None if all servers fail
        """
        for server, port in self.ntp_servers:
            try:
                # Create NTP request packet
                ntp_request = b'\x1b' + 47 * b'\0'

                # Create socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(5)

                # Send request
                sock.sendto(ntp_request, (server, port))

                # Receive response
                response, _ = sock.recvfrom(1024)
                sock.close()

                # Parse NTP response
                if len(response) >= 48:
                    # Extract transmit timestamp (bytes 40-47)
                    tx_timestamp = struct.unpack('!I', response[40:44])[0]
                    tx_fraction = struct.unpack('!I', response[44:48])[0]

                    # Convert NTP timestamp to Unix timestamp
                    # NTP epoch is 1900-01-01, Unix epoch is 1970-01-01
                    # Difference: 70 years = 2208988800 seconds
                    unix_timestamp = tx_timestamp - 2208988800 + (tx_fraction / 2**32)

                    ntp_time = datetime.fromtimestamp(unix_timestamp, timezone.utc)
                    logger.debug(f"NTP sync successful with {server}")
                    return ntp_time

            except (socket.timeout, socket.gaierror, struct.error, OSError) as e:
                logger.debug(f"NTP server {server} failed: {e}")
                continue

        logger.warning("All NTP servers failed - unable to get network time")
        return None

    def get_synced_time(self) -> datetime:
        """
        Get the most accurate available time (NTP > MT5 > Local)

        Returns:
            Most accurate UTC datetime available
        """
        # Try NTP first
        ntp_time = self._get_ntp_time()
        if ntp_time:
            return ntp_time

        # Fall back to MT5 server time
        if self.mt5_connector:
            mt5_time = self.mt5_connector.get_server_time()
            if mt5_time:
                return mt5_time

        # Last resort: local system time
        return datetime.now(timezone.utc)

    def get_sync_status(self) -> Dict:
        """
        Get current synchronization status

        Returns:
            Dict with current sync information
        """
        return {
            'is_synchronized': self.is_synchronized,
            'last_sync_time': self.last_sync_time,
            'time_drift': self.time_drift,
            'max_allowed_drift': self.max_drift,
            'sync_interval': self.sync_interval,
            'thread_running': self.sync_thread.is_alive() if self.sync_thread else False
        }

    def force_sync(self) -> Dict:
        """
        Force immediate synchronization check

        Returns:
            Sync results dict
        """
        logger.info("Forced clock synchronization check")
        return self.perform_sync()

    def set_timezone(self, timezone_name: str):
        """
        Set the local timezone for time conversions

        Args:
            timezone_name: IANA timezone name (e.g., 'America/New_York')
        """
        try:
            self.local_tz = pytz.timezone(timezone_name)
            logger.info(f"Timezone set to {timezone_name}")
        except pytz.exceptions.UnknownTimeZoneError:
            logger.error(f"Unknown timezone: {timezone_name}")

    def convert_to_local(self, utc_time: datetime) -> datetime:
        """
        Convert UTC datetime to local timezone

        Args:
            utc_time: UTC datetime

        Returns:
            Local timezone datetime
        """
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=self.utc_tz)

        return utc_time.astimezone(self.local_tz)

    def __del__(self):
        """Cleanup on destruction"""
        self.stop_sync_thread()