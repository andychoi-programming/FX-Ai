"""
FX-Ai Learning Scheduler Module
Handles scheduling and background task management for the adaptive learning system
"""

import time
import threading
import schedule
import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class LearningScheduler:
    """
    Scheduler for managing periodic learning tasks and background processing.
    Handles task scheduling, thread management, and health monitoring.
    """

    def __init__(self, retrain_interval: int = 24):
        """Initialize the learning scheduler"""
        self.retrain_interval = retrain_interval
        self.learning_thread: Optional[threading.Thread] = None
        self.task_callbacks: Dict[str, Callable] = {}

        # Register task callbacks
        self._register_task_callbacks()

    def _register_task_callbacks(self):
        """Register all scheduled task callbacks"""
        self.task_callbacks = {
            'retrain_models': self._default_retrain_models,
            'evaluate_performance': self._default_evaluate_performance,
            'optimize_parameters': self._default_optimize_parameters,
            'adjust_signal_weights': self._default_adjust_signal_weights,
            'update_all_symbol_holding_times': self._default_update_holding_times,
            'analyze_entry_timing': self._default_analyze_entry_timing,
            'optimize_symbol_sl_tp': self._default_optimize_sl_tp,
            'update_entry_filters': self._default_update_entry_filters,
            'optimize_technical_indicators': self._default_optimize_technical_indicators,
            'optimize_fundamental_weights': self._default_optimize_fundamental_weights,
            'analyze_economic_calendar_impact': self._default_analyze_economic_calendar,
            'analyze_interest_rate_impact': self._default_analyze_interest_rate,
            'optimize_sentiment_parameters': self._default_optimize_sentiment,
            'analyze_adjustment_performance': self._default_analyze_adjustment_performance,
            'clean_old_data': self._default_clean_old_data,
        }

    def register_task_callback(self, task_name: str, callback: Callable):
        """Register a custom callback for a scheduled task"""
        self.task_callbacks[task_name] = callback
        logger.debug(f"Registered custom callback for task: {task_name}")

    def schedule_tasks(self):
        """Schedule periodic learning tasks"""
        # Model retraining
        schedule.every(self.retrain_interval).hours.do(self.task_callbacks['retrain_models'])

        # Performance evaluation
        schedule.every(6).hours.do(self.task_callbacks['evaluate_performance'])

        # Parameter optimization
        schedule.every(12).hours.do(self.task_callbacks['optimize_parameters'])

        # Signal weight adjustment
        schedule.every(4).hours.do(self.task_callbacks['adjust_signal_weights'])

        # Symbol-specific holding time optimization
        schedule.every(24).hours.do(self.task_callbacks['update_all_symbol_holding_times'])

        # Entry timing analysis
        schedule.every(12).hours.do(self.task_callbacks['analyze_entry_timing'])

        # Per-symbol SL/TP optimization
        schedule.every(24).hours.do(self.task_callbacks['optimize_symbol_sl_tp'])

        # Entry filter learning
        schedule.every(8).hours.do(self.task_callbacks['update_entry_filters'])

        # Technical indicator optimization
        schedule.every(24).hours.do(self.task_callbacks['optimize_technical_indicators'])

        # Fundamental weight optimization
        schedule.every(12).hours.do(self.task_callbacks['optimize_fundamental_weights'])

        # Economic calendar learning
        schedule.every(6).hours.do(self.task_callbacks['analyze_economic_calendar_impact'])

        # Interest rate impact analysis
        schedule.every(24).hours.do(self.task_callbacks['analyze_interest_rate_impact'])

        # Sentiment parameter optimization
        schedule.every(12).hours.do(self.task_callbacks['optimize_sentiment_parameters'])

        # Position adjustment performance analysis
        schedule.every(24).hours.do(self.task_callbacks['analyze_adjustment_performance'])

        # Clean old data
        schedule.every(1).day.at("00:00").do(self.task_callbacks['clean_old_data'])

        logger.info(f"Scheduled {len(schedule.jobs)} learning tasks")

    def run_continuous_learning(self):
        """Background thread for continuous learning"""
        logger.info("Starting continuous learning thread")
        loop_count = 0
        last_successful_run = time.time()

        while True:
            try:
                loop_count += 1

                # Check if schedule.run_pending() is working
                schedule.run_pending()

                # Log every 10 minutes to show thread is alive
                current_time = time.time()
                if loop_count % 10 == 0:
                    logger.debug(f"Continuous learning thread alive - loop {loop_count}, {len(schedule.jobs)} scheduled jobs")

                # Check if any jobs should have run recently
                if current_time - last_successful_run > 3600:  # No successful runs in an hour
                    logger.warning("No scheduled jobs have run in the last hour - checking scheduler health")
                    self._check_scheduler_health()

                last_successful_run = current_time

                # Wait 60 seconds, but allow interruption
                time.sleep(60)

            except Exception as e:
                logger.error(f"Error in continuous learning loop {loop_count}: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

                # Don't exit the thread on error - just log and continue
                try:
                    time.sleep(30)  # Wait a bit longer after an error
                except Exception as e:
                    logger.warning(f"Error during thread sleep: {e}")

    def _check_scheduler_health(self):
        """Check if the scheduler is working properly"""
        try:
            logger.info(f"Scheduler health check: {len(schedule.jobs)} jobs registered")

            # Check if any jobs are due to run soon
            current_time = time.time()
            due_jobs = 0
            for job in schedule.jobs:
                if job.next_run <= current_time + 300:  # Due within 5 minutes
                    due_jobs += 1
                    func_name = getattr(job.job_func, '__name__', str(job.job_func))
                    logger.info(f"Job due soon: {func_name} at {time.ctime(job.next_run)}")

            if due_jobs == 0:
                logger.warning("No jobs due to run in the next 5 minutes")

            # Check for overdue jobs (should have run in last hour)
            overdue_jobs = [job for job in schedule.jobs if job.next_run < current_time - 3600]
            if overdue_jobs:
                logger.warning(f"Found {len(overdue_jobs)} jobs that should have run in the last hour")

            # Check if jobs are scheduled too far in the future
            future_jobs = [job for job in schedule.jobs if job.next_run > current_time + 86400]  # More than 24 hours
            if len(future_jobs) > len(schedule.jobs) * 0.8:  # Most jobs too far in future
                logger.warning("Most scheduled jobs are too far in the future - possible scheduling issue")

            # Try to manually run pending jobs
            logger.info("Manually running pending jobs...")
            schedule.run_pending()
            logger.info("Manual run_pending() completed")

        except Exception as e:
            logger.error(f"Error in scheduler health check: {e}")

    def restart_learning_thread(self):
        """Restart the continuous learning thread if it's not running"""
        try:
            if self.learning_thread and self.learning_thread.is_alive():
                logger.info("Learning thread is already running")
                return

            logger.info("Restarting continuous learning thread...")

            # Start a new learning thread
            self.learning_thread = threading.Thread(
                target=self.run_continuous_learning,
                daemon=True,
                name="ContinuousLearning"
            )
            self.learning_thread.start()
            logger.info("Continuous learning thread restarted")

        except Exception as e:
            logger.error(f"Error restarting learning thread: {e}")

    def get_thread_status(self) -> Dict[str, Any]:
        """Get status of the continuous learning thread"""
        try:
            if hasattr(self, 'learning_thread'):
                is_alive = self.learning_thread.is_alive()
                thread_name = self.learning_thread.name

                # If thread is dead, try to restart it
                if not is_alive:
                    logger.warning("Continuous learning thread is dead - attempting restart")
                    self.restart_learning_thread()
                    # Check again after restart attempt
                    time.sleep(0.1)
                    is_alive = self.learning_thread.is_alive() if hasattr(self, 'learning_thread') else False

                return {
                    'thread_alive': is_alive,
                    'thread_name': thread_name,
                    'scheduled_jobs': len(schedule.jobs),
                    'jobs_due_soon': sum(1 for job in schedule.jobs if job.next_run <= time.time() + 300)
                }
            else:
                return {'thread_alive': False, 'error': 'Thread not initialized'}
        except Exception as e:
            logger.error(f"Error getting thread status: {e}")
            return {'thread_alive': False, 'error': str(e)}

    def stop_scheduler(self):
        """Stop the scheduler and cleanup"""
        try:
            # Clear all scheduled jobs
            schedule.clear()
            logger.info("Cleared all scheduled jobs")

            # Stop the learning thread
            if hasattr(self, 'learning_thread') and self.learning_thread.is_alive():
                logger.info("Stopping continuous learning thread...")
                # Note: Daemon threads will be terminated when main thread exits
                self.learning_thread = None

        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

    # Default task implementations (placeholders that can be overridden)
    def _default_retrain_models(self):
        """Default model retraining task"""
        logger.info("Running scheduled model retraining")
        # Placeholder - will be overridden by main manager

    def _default_evaluate_performance(self):
        """Default performance evaluation task"""
        logger.info("Running scheduled performance evaluation")
        # Placeholder - will be overridden by main manager

    def _default_optimize_parameters(self):
        """Default parameter optimization task"""
        logger.info("Running scheduled parameter optimization")
        # Placeholder - will be overridden by main manager

    def _default_adjust_signal_weights(self):
        """Default signal weight adjustment task"""
        logger.info("Running scheduled signal weight adjustment")
        # Placeholder - will be overridden by main manager

    def _default_update_holding_times(self):
        """Default holding time optimization task"""
        logger.info("Running scheduled holding time optimization")
        # Placeholder - will be overridden by main manager

    def _default_analyze_entry_timing(self):
        """Default entry timing analysis task"""
        logger.info("Running scheduled entry timing analysis")
        # Placeholder - will be overridden by main manager

    def _default_optimize_sl_tp(self):
        """Default SL/TP optimization task"""
        logger.info("Running scheduled SL/TP optimization")
        # Placeholder - will be overridden by main manager

    def _default_update_entry_filters(self):
        """Default entry filter learning task"""
        logger.info("Running scheduled entry filter learning")
        # Placeholder - will be overridden by main manager

    def _default_optimize_technical_indicators(self):
        """Default technical indicator optimization task"""
        logger.info("Running scheduled technical indicator optimization")
        # Placeholder - will be overridden by main manager

    def _default_optimize_fundamental_weights(self):
        """Default fundamental weight optimization task"""
        logger.info("Running scheduled fundamental weight optimization")
        # Placeholder - will be overridden by main manager

    def _default_analyze_economic_calendar(self):
        """Default economic calendar analysis task"""
        logger.info("Running scheduled economic calendar analysis")
        # Placeholder - will be overridden by main manager

    def _default_analyze_interest_rate(self):
        """Default interest rate impact analysis task"""
        logger.info("Running scheduled interest rate impact analysis")
        # Placeholder - will be overridden by main manager

    def _default_optimize_sentiment(self):
        """Default sentiment parameter optimization task"""
        logger.info("Running scheduled sentiment parameter optimization")
        # Placeholder - will be overridden by main manager

    def _default_analyze_adjustment_performance(self):
        """Default position adjustment performance analysis task"""
        logger.info("Running scheduled adjustment performance analysis")
        # Placeholder - will be overridden by main manager

    def _default_clean_old_data(self):
        """Default data cleanup task"""
        logger.info("Running scheduled data cleanup")
        # Placeholder - will be overridden by main manager