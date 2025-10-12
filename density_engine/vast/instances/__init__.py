"""
Instance management modules for the vast.ai automation system.
"""

from .discovery import InstanceInfo, discover_active_instances, discover_new_instances
from .lifecycle import (
    restart_instance,
    start_instance,
    stop_instance,
    terminate_instance,
)
from .monitoring import HealthStatus, check_instance_health, get_instance_summary
from .preparation import prepare_instance_for_jobs, verify_instance_readiness

__all__ = [
    "InstanceInfo",
    "discover_active_instances",
    "discover_new_instances",
    "prepare_instance_for_jobs",
    "verify_instance_readiness",
    "check_instance_health",
    "get_instance_summary",
    "HealthStatus",
    "start_instance",
    "stop_instance",
    "restart_instance",
    "terminate_instance",
]
