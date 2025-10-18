"""
Celery application configuration for density-engine.

This module sets up the Celery application with Redis backend and
appropriate configuration for reliable task processing.
"""

import os
from typing import Dict

from celery import Celery

from .types import CeleryConfig

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB_BROKER = int(os.getenv("REDIS_DB_BROKER", "0"))
REDIS_DB_BACKEND = int(os.getenv("REDIS_DB_BACKEND", "1"))

# Celery configuration
CELERY_BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_BROKER}"
CELERY_RESULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_BACKEND}"

# Create Celery app
app = Celery(
    "density_engine",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    result_expires=86400,  # 24 hours
)

# Celery configuration
app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task tracking
    task_track_started=True,
    task_acks_late=True,  # Important for recovery
    worker_prefetch_multiplier=1,  # One task at a time per worker
    # Result backend
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    # Task routing
    task_routes={
        "density_engine.tasks.jobs.*": {"queue": "jobs"},
        "density_engine.tasks.instances.*": {"queue": "instances"},
        "density_engine.tasks.cluster.*": {"queue": "cluster"},
    },
    # Worker configuration
    worker_hijack_root_logger=False,
    worker_log_color=False,
    # Task execution
    task_always_eager=False,
    task_eager_propagates=True,
    # Retry configuration
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Optional: Configure beat scheduler for periodic tasks
app.conf.beat_schedule = {
    "discover-instances": {
        "task": "density_engine.tasks.instances.discover_instances",
        "schedule": 60.0,  # Every 60 seconds
    },
    "health-check-instances": {
        "task": "density_engine.tasks.instances.health_check_all_instances",
        "schedule": 30.0,  # Every 30 seconds
    },
    "monitor-jobs": {
        "task": "density_engine.tasks.jobs.monitor_all_jobs",
        "schedule": 10.0,  # Every 10 seconds
    },
}

app.conf.timezone = "UTC"


def get_celery_config() -> CeleryConfig:
    """Get current Celery configuration for debugging."""
    return {
        "broker_url": app.conf.broker_url,
        "result_backend": app.conf.result_backend,
        "task_routes": app.conf.task_routes,
        "beat_schedule": app.conf.beat_schedule,
    }


if __name__ == "__main__":
    app.start()
