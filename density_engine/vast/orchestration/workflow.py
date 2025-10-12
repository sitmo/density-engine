"""
Workflow management for the vast.ai automation system.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ..core.state import get_state_manager
from ..utils.logging import get_logger, log_function_call
from .coordinator import OrchestrationCoordinator, get_coordinator
from .scheduler import TaskScheduler, get_scheduler

logger = get_logger(__name__)


class WorkflowState(Enum):
    """Workflow states."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class WorkflowMetrics:
    """Workflow metrics."""

    total_runtime: float
    jobs_processed: int
    jobs_successful: int
    jobs_failed: int
    instances_used: int
    average_job_time: float


class WorkflowManager:
    """Manages the overall workflow execution."""

    def __init__(self) -> None:
        self.coordinator = get_coordinator()
        self.scheduler = get_scheduler()
        self.state_manager = get_state_manager()
        self.state = WorkflowState.STOPPED
        self.start_time: float | None = None
        self.metrics = WorkflowMetrics(0.0, 0, 0, 0, 0, 0.0)

    @log_function_call
    def initialize_workflow(self) -> bool:
        """Initialize the workflow."""
        try:
            logger.info("Initializing workflow")

            self.state = WorkflowState.INITIALIZING

            # Load existing state
            self.state_manager.load_state()

            # Initialize scheduler and coordinator
            self.coordinator._register_task_handlers()

            logger.info("✅ Workflow initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize workflow: {e}")
            self.state = WorkflowState.ERROR
            return False

    @log_function_call
    def start_workflow(self) -> bool:
        """Start the workflow."""
        try:
            if self.state == WorkflowState.RUNNING:
                logger.warning("Workflow is already running")
                return True

            logger.info("Starting workflow")

            # Initialize if not already done
            if self.state == WorkflowState.STOPPED:
                if not self.initialize_workflow():
                    return False

            # Start orchestration
            self.coordinator.start_orchestration()

            # Record start time
            self.start_time = time.time()
            self.state = WorkflowState.RUNNING

            logger.info("✅ Workflow started")
            return True

        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            self.state = WorkflowState.ERROR
            return False

    @log_function_call
    def pause_workflow(self) -> bool:
        """Pause the workflow."""
        try:
            if self.state != WorkflowState.RUNNING:
                logger.warning("Workflow is not running, cannot pause")
                return False

            logger.info("Pausing workflow")

            self.state = WorkflowState.PAUSED

            logger.info("✅ Workflow paused")
            return True

        except Exception as e:
            logger.error(f"Failed to pause workflow: {e}")
            return False

    @log_function_call
    def resume_workflow(self) -> bool:
        """Resume the workflow."""
        try:
            if self.state != WorkflowState.PAUSED:
                logger.warning("Workflow is not paused, cannot resume")
                return False

            logger.info("Resuming workflow")

            self.state = WorkflowState.RUNNING

            logger.info("✅ Workflow resumed")
            return True

        except Exception as e:
            logger.error(f"Failed to resume workflow: {e}")
            return False

    @log_function_call
    def stop_workflow(self) -> bool:
        """Stop the workflow."""
        try:
            if self.state == WorkflowState.STOPPED:
                logger.warning("Workflow is already stopped")
                return True

            logger.info("Stopping workflow")

            self.state = WorkflowState.STOPPING

            # Stop orchestration
            self.coordinator.stop_orchestration()

            # Update metrics
            if self.start_time:
                self.metrics.total_runtime = time.time() - self.start_time

            self.state = WorkflowState.STOPPED

            logger.info("✅ Workflow stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop workflow: {e}")
            self.state = WorkflowState.ERROR
            return False

    @log_function_call
    def run_workflow_cycle(self) -> bool:
        """Run one cycle of the workflow."""
        try:
            if self.state != WorkflowState.RUNNING:
                return False

            # Get orchestration state
            orchestration_state = self.coordinator.get_orchestration_state()

            # Update metrics
            self.metrics.jobs_processed = orchestration_state.completed_jobs
            self.metrics.jobs_successful = (
                orchestration_state.completed_jobs
            )  # Simplified
            self.metrics.instances_used = orchestration_state.active_instances

            # Log status
            logger.info(
                f"Workflow cycle: {orchestration_state.active_instances} active instances, "
                f"{orchestration_state.idle_instances} idle, "
                f"{orchestration_state.pending_jobs} pending jobs, "
                f"{orchestration_state.running_jobs} running jobs"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to run workflow cycle: {e}")
            return False

    @log_function_call
    def run_workflow(self, max_iterations: int | None = None) -> bool:
        """Run the workflow for a specified number of iterations."""
        try:
            if not self.start_workflow():
                return False

            logger.info(f"Running workflow (max iterations: {max_iterations})")

            iteration = 0
            while self.state == WorkflowState.RUNNING:
                # Run one cycle
                if not self.run_workflow_cycle():
                    break

                iteration += 1

                # Check if we've reached max iterations
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached maximum iterations ({max_iterations})")
                    break

                # Sleep between cycles
                time.sleep(10)  # Increased from 5 to 10 seconds to reduce API hammering

            # Stop the workflow
            self.stop_workflow()

            logger.info(f"✅ Workflow completed after {iteration} iterations")
            return True

        except KeyboardInterrupt:
            logger.info("Workflow interrupted by user")
            self.stop_workflow()
            return False
        except Exception as e:
            logger.error(f"Failed to run workflow: {e}")
            self.stop_workflow()
            return False

    @log_function_call
    def get_workflow_status(self) -> dict[str, Any]:
        """Get current workflow status."""
        try:
            orchestration_state = self.coordinator.get_orchestration_state()

            status = {
                "workflow_state": self.state.value,
                "total_runtime": self.metrics.total_runtime,
                "instances": {
                    "total": orchestration_state.total_instances,
                    "active": orchestration_state.active_instances,
                    "idle": orchestration_state.idle_instances,
                    "busy": orchestration_state.busy_instances,
                },
                "jobs": {
                    "total": orchestration_state.total_jobs,
                    "pending": orchestration_state.pending_jobs,
                    "running": orchestration_state.running_jobs,
                    "completed": orchestration_state.completed_jobs,
                },
                "metrics": {
                    "jobs_processed": self.metrics.jobs_processed,
                    "jobs_successful": self.metrics.jobs_successful,
                    "jobs_failed": self.metrics.jobs_failed,
                    "instances_used": self.metrics.instances_used,
                    "average_job_time": self.metrics.average_job_time,
                },
            }

            return status

        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return {"error": str(e)}

    @log_function_call
    def print_workflow_status(self) -> None:
        """Print current workflow status."""
        try:
            status = self.get_workflow_status()

            print("\n" + "=" * 60)
            print("VAST.AI AUTOMATION WORKFLOW STATUS")
            print("=" * 60)
            print(f"Workflow State: {status['workflow_state']}")
            print(f"Total Runtime: {status['total_runtime']:.1f} seconds")
            print()
            print("INSTANCES:")
            print(f"  Total: {status['instances']['total']}")
            print(f"  Active: {status['instances']['active']}")
            print(f"  Idle: {status['instances']['idle']}")
            print(f"  Busy: {status['instances']['busy']}")
            print()
            print("JOBS:")
            print(f"  Total: {status['jobs']['total']}")
            print(f"  Pending: {status['jobs']['pending']}")
            print(f"  Running: {status['jobs']['running']}")
            print(f"  Completed: {status['jobs']['completed']}")
            print()
            print("METRICS:")
            print(f"  Jobs Processed: {status['metrics']['jobs_processed']}")
            print(f"  Jobs Successful: {status['metrics']['jobs_successful']}")
            print(f"  Jobs Failed: {status['metrics']['jobs_failed']}")
            print(f"  Instances Used: {status['metrics']['instances_used']}")
            print(
                f"  Average Job Time: {status['metrics']['average_job_time']:.1f} seconds"
            )
            print("=" * 60)

        except Exception as e:
            logger.error(f"Failed to print workflow status: {e}")


# Global workflow manager instance
_workflow_manager: WorkflowManager | None = None


def get_workflow_manager() -> WorkflowManager:
    """Get the global workflow manager instance."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
    return _workflow_manager
