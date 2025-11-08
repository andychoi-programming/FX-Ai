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
from datetime import datetime, timezone
from typing import Optional, Dict
import pytz

logger = logging.getLogger(__name__)

class ClockSynchronizer:
    """
    Advanced clock synchronization system for trading applications.
    Maintains accurate time alignment between local system, NTP servers, and MT5 platform.
    """

    def __init__(self, mt5_connector=None, sync_interval: int = 300, max_drift: float = 1.0, mt5_timezone_tolerance: float = 43200.0):
        """
        Initialize clock synchronizer

        Args:
            mt5_connector: MT5 connector instance for server time checks
            sync_interval: Seconds between synchronization checks (default: 5 minutes)
            max_drift: Maximum allowed time drift in seconds before correction (default: 1.0s)
            mt5_timezone_tolerance: Maximum allowed MT5 time difference in seconds (default: 12 hours)
                                   since MT5 might report times in broker's local timezone
        """
        self.mt5_connector = mt5_connector
        self.sync_interval = sync_interval
        self.max_drift = max_drift
        self.mt5_timezone_tolerance = mt5_timezone_tolerance

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

                    # Check if MT5 drift exceeds timezone tolerance (MT5 might be in different timezone)
                    if drift_mt5 > self.mt5_timezone_tolerance:
                        logger.warning(f"MT5 time significantly different (possible timezone issue): {drift_mt5:.1f}s")
                    elif drift_mt5 > self.max_drift:
                        logger.info(f"MT5 time drift detected: {drift_mt5:.3f}s")

            # Determine overall sync status - based primarily on NTP
            # MT5 time is used for informational purposes only since it might be in different timezone
            ntp_drift = results['drift_ntp'] or float('inf')
            results['is_synced'] = ntp_drift <= self.max_drift

            # Store the effective drift (NTP only)
            self.time_drift = ntp_drift if ntp_drift != float('inf') else 0
            self.is_synchronized = results['is_synced']
            self.last_sync_time = results['timestamp']

            # Log sync status
            if results['is_synced']:
                drift_msg = f"NTP drift: {ntp_drift:.3f}s" if ntp_drift != float('inf') else "NTP unavailable"
                logger.info(f"Clock synchronized - {drift_msg}")
            else:
                drift_msg = f"NTP drift: {ntp_drift:.3f}s" if ntp_drift != float('inf') else "NTP unavailable"
                logger.warning(f"Clock out of sync - {drift_msg}")

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
        Get the most accurate available time (MT5 > NTP cache > Local)

        Returns:
            Most accurate datetime available (uses MT5 server time for trading)
        """
        # Use MT5 server time first (most relevant for trading)
        if self.mt5_connector:
            try:
                mt5_time = self.mt5_connector.get_server_time()
                if mt5_time:
                    return mt5_time
            except Exception:
                pass
        
        # Fall back to local system time (NTP queries only in background sync thread)
        return datetime.now(timezone.utc)
    
    def get_ntp_synced_time(self) -> datetime:
        """
        Get NTP-synchronized time (slow - queries NTP servers)
        Use get_synced_time() for fast MT5 server time instead

        Returns:
            NTP time or local time as fallback
        """
        # Try NTP (slow - queries network)
        ntp_time = self._get_ntp_time()
        if ntp_time:
            return ntp_time

        # Fall back to local system time
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
            'mt5_timezone_tolerance': self.mt5_timezone_tolerance,
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