"""
Task scheduling for the vast.ai automation system.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import PriorityQueue
from typing import Any, Dict, Optional
from collections.abc import Callable

from ..utils.exceptions import StateManagementError
from ..utils.logging import get_logger, log_function_call

logger = get_logger(__name__)


class TaskType(Enum):
    """Types of tasks that can be queued."""

    REFRESH_INSTANCES = "refresh_instances"
    CHECK_INSTANCE_STATUS = "check_instance_status"
    PREPARE_INSTANCE = "prepare_instance"
    START_JOB = "start_job"
    CHECK_JOB_STATUS = "check_job_status"
    DOWNLOAD_RESULTS = "download_results"
    ASSIGN_JOBS = "assign_jobs"
    DISCOVER_JOBS = "discover_jobs"


@dataclass
class Task:
    """Task definition."""

    task_type: TaskType
    instance_id: str | None = None
    job_file: str | None = None
    priority: int = 0
    scheduled_time: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Task scheduler with priority queue."""

    def __init__(self) -> None:
        self.task_queue: PriorityQueue[tuple[int, float, Task]] = PriorityQueue()
        self.running = False
        self.worker_thread: threading.Thread | None = None
        self.task_handlers: dict[TaskType, Callable[..., Any]] = {}
        self.task_intervals: dict[TaskType, int] = {
            TaskType.REFRESH_INSTANCES: 30,  # 30 seconds
            TaskType.DISCOVER_JOBS: 60,  # 60 seconds
            TaskType.ASSIGN_JOBS: 3,  # 3 seconds
        }
        self.last_task_times: dict[TaskType, datetime] = {}

    @log_function_call
    def create_task(self, task_type: TaskType, **kwargs: Any) -> Task:
        """Create a new task."""
        task = Task(task_type=task_type, **kwargs)
        logger.debug(f"Created task: {task_type.value}")
        return task

    @log_function_call
    def schedule_task(self, task: Task, priority: int = 0) -> bool:
        """Schedule a task with priority."""
        try:
            # Add timestamp for ordering
            timestamp = time.time()

            # Put task in priority queue (lower number = higher priority)
            self.task_queue.put((priority, timestamp, task))

            logger.debug(
                f"Scheduled task: {task.task_type.value} (priority: {priority})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to schedule task: {e}")
            return False

    @log_function_call
    def get_next_task(self) -> Task | None:
        """Get the next task from the queue."""
        try:
            if self.task_queue.empty():
                return None

            priority, timestamp, task = self.task_queue.get_nowait()

            # Check if task is ready to execute
            if task.scheduled_time > datetime.now():
                # Put task back in queue
                self.task_queue.put((priority, timestamp, task))
                return None

            logger.debug(f"Retrieved task: {task.task_type.value}")
            return task

        except Exception as e:
            logger.error(f"Failed to get next task: {e}")
            return None

    @log_function_call
    def mark_task_completed(self, task_id: str) -> bool:
        """Mark a task as completed."""
        try:
            logger.debug(f"Task completed: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark task as completed: {e}")
            return False

    @log_function_call
    def schedule_instance_preparation(self, instance: Any) -> bool:
        """Schedule instance preparation task."""
        try:
            task = self.create_task(
                TaskType.PREPARE_INSTANCE,
                instance_id=str(instance.contract_id),
                priority=1,
            )
            return self.schedule_task(task, priority=1)
        except Exception as e:
            logger.error(f"Failed to schedule instance preparation: {e}")
            return False

    @log_function_call
    def schedule_job_execution(self, instance: Any, job_file: str) -> bool:
        """Schedule job execution task."""
        try:
            task = self.create_task(
                TaskType.START_JOB,
                instance_id=str(instance.contract_id),
                job_file=job_file,
                priority=1,
            )
            return self.schedule_task(task, priority=1)
        except Exception as e:
            logger.error(f"Failed to schedule job execution: {e}")
            return False

    @log_function_call
    def schedule_result_collection(self, instance: Any, job_file: str) -> bool:
        """Schedule result collection task."""
        try:
            task = self.create_task(
                TaskType.DOWNLOAD_RESULTS,
                instance_id=str(instance.contract_id),
                job_file=job_file,
                priority=2,
            )
            return self.schedule_task(task, priority=2)
        except Exception as e:
            logger.error(f"Failed to schedule result collection: {e}")
            return False

    @log_function_call
    def reschedule_failed_task(self, task: Task, delay: timedelta) -> bool:
        """Reschedule a failed task with delay."""
        try:
            task.scheduled_time = datetime.now() + delay
            task.priority += 1  # Lower priority for retry

            logger.info(
                f"Rescheduling failed task {task.task_type.value} with {delay} delay"
            )
            return self.schedule_task(task, priority=task.priority)
        except Exception as e:
            logger.error(f"Failed to reschedule task: {e}")
            return False

    @log_function_call
    def schedule_periodic_tasks(self) -> None:
        """Schedule periodic tasks based on intervals."""
        try:
            current_time = datetime.now()

            for task_type, interval_seconds in self.task_intervals.items():
                last_time = self.last_task_times.get(task_type, datetime.min)

                if (current_time - last_time).total_seconds() >= interval_seconds:
                    task = self.create_task(task_type, priority=5)
                    self.schedule_task(task, priority=5)
                    self.last_task_times[task_type] = current_time

                    logger.debug(f"Scheduled periodic task: {task_type.value}")

        except Exception as e:
            logger.error(f"Failed to schedule periodic tasks: {e}")

    @log_function_call
    def register_task_handler(
        self, task_type: TaskType, handler: Callable[..., Any]
    ) -> None:
        """Register a task handler."""
        try:
            self.task_handlers[task_type] = handler
            logger.debug(f"Registered handler for task type: {task_type.value}")
        except Exception as e:
            logger.error(f"Failed to register task handler: {e}")

    @log_function_call
    def execute_task(self, task: Task) -> bool:
        """Execute a task using its registered handler."""
        try:
            if task.task_type not in self.task_handlers:
                logger.warning(
                    f"No handler registered for task type: {task.task_type.value}"
                )
                return False

            handler = self.task_handlers[task.task_type]
            logger.debug(f"Executing task: {task.task_type.value}")

            result = handler(task)

            if result:
                logger.debug(f"Task completed successfully: {task.task_type.value}")
            else:
                logger.warning(f"Task failed: {task.task_type.value}")

            return result

        except Exception as e:
            logger.error(f"Failed to execute task {task.task_type.value}: {e}")
            return False

    @log_function_call
    def start_scheduler(self) -> None:
        """Start the task scheduler."""
        try:
            if self.running:
                logger.warning("Scheduler is already running")
                return

            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

            logger.info("✅ Task scheduler started")

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")

    @log_function_call
    def stop_scheduler(self) -> None:
        """Stop the task scheduler."""
        try:
            if not self.running:
                logger.warning("Scheduler is not running")
                return

            self.running = False

            if self.worker_thread:
                self.worker_thread.join(timeout=5)

            logger.info("✅ Task scheduler stopped")

        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")

    def _worker_loop(self) -> None:
        """Worker loop for processing tasks."""
        logger.info("Task scheduler worker loop started")

        while self.running:
            try:
                # Schedule periodic tasks
                self.schedule_periodic_tasks()

                # Process next task
                task = self.get_next_task()
                if task:
                    self.execute_task(task)
                else:
                    # No tasks ready, sleep briefly
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in scheduler worker loop: {e}")
                time.sleep(1)

        logger.info("Task scheduler worker loop stopped")


# Global scheduler instance
_scheduler: TaskScheduler | None = None


def get_scheduler() -> TaskScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
