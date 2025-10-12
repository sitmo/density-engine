"""
State management for the vast.ai automation system.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.exceptions import StateManagementError
from ..utils.logging import get_logger, log_function_call
from .files import JobState

logger = get_logger(__name__)


class InstanceStatus(Enum):
    """Instance status."""

    STARTING = "starting"
    RUNNING = "running"
    PREPARING = "preparing"
    READY = "ready"
    BUSY = "busy"
    COMPLETED = "completed"
    ERROR = "error"
    IDLE = "idle"
    ASSIGNED = "assigned"
    FINISHED = "finished"


@dataclass
class InstanceState:
    """Instance state information."""

    contract_id: str
    ssh_host: str
    ssh_port: int
    status: InstanceStatus
    last_updated: datetime
    job_state: str = "idle"
    current_job: str | None = None
    is_prepared: bool = False  # Track if instance is prepared and verified
    preparation_verified_at: datetime | None = (
        None  # When preparation was last verified
    )
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class JobStateInfo:
    """Job state information."""

    job_file: str
    state: JobState
    assigned_instance: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    job_args: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_minutes: int = 60
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class StateManager:
    """Manages system state persistence."""

    def __init__(self, state_file: Path = Path("vast_state.json")) -> None:
        self.state_file = state_file
        self.instances: dict[str, InstanceState] = {}
        self.jobs: dict[str, JobStateInfo] = {}
        self.load_state()

    @log_function_call
    def load_state(self) -> dict[str, Any]:
        """Load state from file."""
        try:
            if not self.state_file.exists():
                logger.info("No existing state file found, starting fresh")
                return {}

            with open(self.state_file) as f:
                data = json.load(f)

            # Load instances
            instances_data = data.get("instances", {})
            for instance_id, instance_data in instances_data.items():
                # Convert datetime strings back to datetime objects
                for time_field in ["last_updated", "preparation_verified_at"]:
                    if time_field in instance_data and instance_data[time_field]:
                        instance_data[time_field] = datetime.fromisoformat(
                            instance_data[time_field]
                        )

                # Convert status string back to enum
                if "status" in instance_data:
                    instance_data["status"] = InstanceStatus(instance_data["status"])

                self.instances[instance_id] = InstanceState(**instance_data)

            # Load jobs
            jobs_data = data.get("jobs", {})
            for job_file, job_data in jobs_data.items():
                # Convert datetime strings back to datetime objects
                for time_field in ["start_time", "end_time"]:
                    if time_field in job_data and job_data[time_field]:
                        job_data[time_field] = datetime.fromisoformat(
                            job_data[time_field]
                        )

                # Convert state string back to enum
                if "state" in job_data:
                    job_data["state"] = JobState(job_data["state"])

                self.jobs[job_file] = JobStateInfo(**job_data)

            logger.info(
                f"Loaded state: {len(self.instances)} instances, {len(self.jobs)} jobs"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            raise StateManagementError(f"Failed to load state: {e}")

    @log_function_call
    def save_state(self) -> bool:
        """Save state to file."""
        try:
            # Convert instances to serializable format
            instances_data = {}
            for instance_id, instance in self.instances.items():
                instance_dict = asdict(instance)
                instance_dict["last_updated"] = instance.last_updated.isoformat()
                if instance.preparation_verified_at:
                    instance_dict["preparation_verified_at"] = (
                        instance.preparation_verified_at.isoformat()
                    )
                instance_dict["status"] = instance.status.value
                instances_data[instance_id] = instance_dict

            # Convert jobs to serializable format
            jobs_data = {}
            for job_file, job in self.jobs.items():
                job_dict = asdict(job)
                if job.start_time:
                    job_dict["start_time"] = job.start_time.isoformat()
                if job.end_time:
                    job_dict["end_time"] = job.end_time.isoformat()
                job_dict["state"] = job.state.value
                jobs_data[job_file] = job_dict

            state_data = {
                "instances": instances_data,
                "jobs": jobs_data,
                "last_saved": datetime.now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            logger.debug(f"State saved to {self.state_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise StateManagementError(f"Failed to save state: {e}")

    @log_function_call
    def clear_state(self) -> bool:
        """Clear all cached state and start fresh."""
        try:
            # Clear in-memory state
            self.instances.clear()
            self.jobs.clear()

            # Remove state file if it exists
            if self.state_file.exists():
                self.state_file.unlink()
                logger.info(f"Removed state file: {self.state_file}")

            logger.info("✅ State cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to clear state: {e}")
            raise StateManagementError(f"Failed to clear state: {e}")

    @log_function_call
    def update_instance_state(self, instance_id: str, updates: dict[str, Any]) -> bool:
        """Update instance state."""
        try:
            if instance_id not in self.instances:
                logger.warning(f"Instance {instance_id} not found in state")
                return False

            instance = self.instances[instance_id]

            # Update fields
            for key, value in updates.items():
                if hasattr(instance, key):
                    # Handle status field specially - convert string to enum
                    if key == "status" and isinstance(value, str):
                        setattr(instance, key, InstanceStatus(value))
                    else:
                        setattr(instance, key, value)
                else:
                    instance.metadata[key] = value

            instance.last_updated = datetime.now()
            self.save_state()

            logger.debug(f"Updated instance {instance_id} state")
            return True

        except Exception as e:
            logger.error(f"Failed to update instance state: {e}")
            raise StateManagementError(f"Failed to update instance state: {e}")

    @log_function_call
    def update_job_state(self, job_file: str, updates: dict[str, Any]) -> bool:
        """Update job state."""
        try:
            if job_file not in self.jobs:
                logger.warning(f"Job {job_file} not found in state")
                return False

            job = self.jobs[job_file]

            # Update fields
            for key, value in updates.items():
                if hasattr(job, key):
                    setattr(job, key, value)
                else:
                    job.metadata[key] = value

            self.save_state()

            logger.debug(f"Updated job {job_file} state")
            return True

        except Exception as e:
            logger.error(f"Failed to update job state: {e}")
            raise StateManagementError(f"Failed to update job state: {e}")

    @log_function_call
    def get_idle_instances(self) -> list[InstanceState]:
        """Get all idle instances that are prepared and ready for jobs."""
        idle_instances = []
        for instance in self.instances.values():
            if instance.job_state == "idle" and instance.is_prepared:
                idle_instances.append(instance)

        logger.debug(f"Found {len(idle_instances)} idle and prepared instances")
        return idle_instances

    @log_function_call
    def get_available_jobs(self) -> list[JobStateInfo]:
        """Get all available jobs."""
        available_jobs = []
        for job in self.jobs.values():
            if job.state == JobState.TODO:
                available_jobs.append(job)

        logger.debug(f"Found {len(available_jobs)} available jobs")
        return available_jobs

    @log_function_call
    def mark_job_assigned(self, job_file: str, instance_id: str) -> bool:
        """Mark a job as assigned to an instance."""
        try:
            # Update job state
            self.update_job_state(
                job_file,
                {
                    "state": JobState.RUNNING,
                    "assigned_instance": instance_id,
                    "start_time": datetime.now(),
                },
            )

            # Update instance state
            self.update_instance_state(
                instance_id, {"job_state": "assigned", "current_job": job_file}
            )

            logger.info(f"Marked job {job_file} as assigned to instance {instance_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to mark job as assigned: {e}")
            return False

    @log_function_call
    def mark_job_completed(self, job_file: str, success: bool) -> bool:
        """Mark a job as completed."""
        try:
            job_state = JobState.COMPLETED if success else JobState.FAILED

            # Update job state
            self.update_job_state(
                job_file, {"state": job_state, "end_time": datetime.now()}
            )

            # Update instance state
            if job_file in self.jobs:
                instance_id = self.jobs[job_file].assigned_instance
                if instance_id and instance_id in self.instances:
                    self.update_instance_state(
                        instance_id, {"job_state": "idle", "current_job": None}
                    )

            logger.info(
                f"Marked job {job_file} as {'completed' if success else 'failed'}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to mark job as completed: {e}")
            return False

    @log_function_call
    def mark_instance_prepared(self, instance_id: str) -> bool:
        """Mark an instance as prepared and verified."""
        try:
            self.update_instance_state(
                instance_id,
                {
                    "is_prepared": True,
                    "preparation_verified_at": datetime.now(),
                    "status": InstanceStatus.READY.value,
                },
            )
            logger.info(f"Marked instance {instance_id} as prepared and verified")
            return True

        except Exception as e:
            logger.error(f"Failed to mark instance as prepared: {e}")
            return False

    @log_function_call
    def get_unprepared_instances(self) -> list[str]:
        """Get list of instance IDs that are not prepared."""
        unprepared = []
        for instance_id, instance_state in self.instances.items():
            if not instance_state.is_prepared:
                unprepared.append(instance_id)
        return unprepared


# Global state manager instance
_state_manager: StateManager | None = None


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
