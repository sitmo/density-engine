"""
Orchestration coordinator for the vast.ai automation system.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.files import JobState, move_job_file
from ..core.jobs import discover_job_files, prioritize_jobs
from ..core.state import InstanceState, InstanceStatus, get_state_manager
from ..execution.job_runner import execute_job_on_instance
from ..execution.process_monitor import detect_job_completion, monitor_job_execution
from ..execution.result_handler import collect_job_results, process_job_results
from ..instances.discovery import InstanceInfo, discover_active_instances
from ..instances.monitoring import check_instance_health, get_instance_summary
from ..instances.preparation import prepare_instance_for_jobs, verify_instance_readiness
from ..utils.exceptions import StateManagementError
from ..utils.logging import get_logger, log_function_call
from .scheduler import Task, TaskScheduler, TaskType, get_scheduler

logger = get_logger(__name__)


@dataclass
class OrchestrationState:
    """Current orchestration state."""

    total_instances: int
    active_instances: int
    idle_instances: int
    busy_instances: int
    total_jobs: int
    pending_jobs: int
    running_jobs: int
    completed_jobs: int


class OrchestrationCoordinator:
    """Main orchestration coordinator."""

    def __init__(self) -> None:
        self.state_manager = get_state_manager()
        self.scheduler = get_scheduler()
        self.running = False
        self._register_task_handlers()

    @log_function_call
    def _register_task_handlers(self) -> None:
        """Register task handlers with the scheduler."""
        try:
            self.scheduler.register_task_handler(
                TaskType.REFRESH_INSTANCES, self._handle_refresh_instances
            )
            self.scheduler.register_task_handler(
                TaskType.CHECK_INSTANCE_STATUS, self._handle_check_instance_status
            )
            self.scheduler.register_task_handler(
                TaskType.PREPARE_INSTANCE, self._handle_prepare_instance
            )
            self.scheduler.register_task_handler(
                TaskType.START_JOB, self._handle_start_job
            )
            self.scheduler.register_task_handler(
                TaskType.CHECK_JOB_STATUS, self._handle_check_job_status
            )
            self.scheduler.register_task_handler(
                TaskType.DOWNLOAD_RESULTS, self._handle_download_results
            )
            self.scheduler.register_task_handler(
                TaskType.ASSIGN_JOBS, self._handle_assign_jobs
            )
            self.scheduler.register_task_handler(
                TaskType.DISCOVER_JOBS, self._handle_discover_jobs
            )

            logger.info("✅ Task handlers registered")

        except Exception as e:
            logger.error(f"Failed to register task handlers: {e}")

    @log_function_call
    def _handle_refresh_instances(self, task: Task) -> bool:
        """Handle instance refresh task."""
        try:
            logger.debug("Refreshing instances")

            # Discover active instances
            active_instances = discover_active_instances()

            # Update state manager
            for instance in active_instances:
                instance_id = str(instance.contract_id)

                if instance_id not in self.state_manager.instances:
                    # New instance - starts as unprepared
                    instance_state = InstanceState(
                        contract_id=instance_id,
                        ssh_host=instance.ssh_host,
                        ssh_port=instance.ssh_port,
                        status=InstanceStatus.STARTING,
                        last_updated=task.scheduled_time,
                        is_prepared=False,  # New instances are not prepared
                        preparation_verified_at=None,
                    )
                    self.state_manager.instances[instance_id] = instance_state
                    logger.info(
                        f"Discovered new instance: {instance_id} (not prepared)"
                    )
                else:
                    # Update existing instance - preserve preparation status
                    existing_instance = self.state_manager.instances[instance_id]
                    self.state_manager.update_instance_state(
                        instance_id,
                        {
                            "ssh_host": instance.ssh_host,
                            "ssh_port": instance.ssh_port,
                            "last_updated": task.scheduled_time,
                            # Preserve preparation status - don't reset it
                        },
                    )
                    logger.debug(
                        f"Updated existing instance: {instance_id} (prepared: {existing_instance.is_prepared})"
                    )

            self.state_manager.save_state()
            logger.info(f"Refreshed {len(active_instances)} instances")
            return True

        except Exception as e:
            logger.error(f"Failed to refresh instances: {e}")
            return False

    @log_function_call
    def _handle_check_instance_status(self, task: Task) -> bool:
        """Handle instance status check task."""
        try:
            instance_id = task.instance_id
            if not instance_id:
                logger.warning("No instance ID provided for status check")
                return False

            logger.debug(f"Checking status of instance {instance_id}")

            # Get instance from state
            if instance_id not in self.state_manager.instances:
                logger.warning(f"Instance {instance_id} not found in state")
                return False

            instance_state = self.state_manager.instances[instance_id]

            # Check SSH connectivity
            from .core.ssh import test_ssh_connectivity

            ssh_connected = test_ssh_connectivity(
                instance_state.ssh_host, instance_state.ssh_port
            )

            if ssh_connected:
                # Update status to running
                self.state_manager.update_instance_state(
                    instance_id, {"status": InstanceStatus.RUNNING.value}
                )
                logger.debug(f"Instance {instance_id} is running")
            else:
                logger.debug(f"Instance {instance_id} SSH not ready")

            return True

        except Exception as e:
            logger.error(f"Failed to check instance status: {e}")
            return False

    @log_function_call
    def _handle_prepare_instance(self, task: Task) -> bool:
        """Handle instance preparation task."""
        try:
            instance_id = task.instance_id
            if not instance_id:
                logger.warning("No instance ID provided for preparation")
                return False

            logger.info(f"Preparing instance {instance_id}")

            # Get instance from state
            if instance_id not in self.state_manager.instances:
                logger.warning(f"Instance {instance_id} not found in state")
                return False

            instance_state = self.state_manager.instances[instance_id]

            # Create InstanceInfo object
            instance_info = InstanceInfo(
                contract_id=int(instance_id),
                machine_id=0,
                gpu_name="Unknown",
                price_per_hour=0.0,
                ssh_host=instance_state.ssh_host,
                ssh_port=instance_state.ssh_port,
                status="running",
                public_ipaddr=instance_state.ssh_host,
                ports={},
            )

            # Prepare the instance
            success = prepare_instance_for_jobs(instance_info)

            if success:
                # Mark instance as prepared and verified
                self.state_manager.mark_instance_prepared(instance_id)
                logger.info(
                    f"✅ Instance {instance_id} prepared and verified successfully"
                )
            else:
                # Mark as error
                self.state_manager.update_instance_state(
                    instance_id, {"status": InstanceStatus.ERROR.value}
                )
                logger.error(f"❌ Instance {instance_id} preparation failed")

            return success

        except Exception as e:
            logger.error(f"Failed to prepare instance: {e}")
            return False

    @log_function_call
    def _handle_start_job(self, task: Task) -> bool:
        """Handle job start task."""
        try:
            instance_id = task.instance_id
            job_file = task.job_file

            if not instance_id or not job_file:
                logger.warning("Missing instance ID or job file for job start")
                return False

            logger.info(f"Starting job {job_file} on instance {instance_id}")

            # Get instance from state
            if instance_id not in self.state_manager.instances:
                logger.warning(f"Instance {instance_id} not found in state")
                return False

            instance_state = self.state_manager.instances[instance_id]

            # Create InstanceInfo object
            instance_info = InstanceInfo(
                contract_id=int(instance_id),
                machine_id=0,
                gpu_name="Unknown",
                price_per_hour=0.0,
                ssh_host=instance_state.ssh_host,
                ssh_port=instance_state.ssh_port,
                status="running",
                public_ipaddr=instance_state.ssh_host,
                ports={},
            )

            # Move job file to running state
            if not move_job_file(job_file, JobState.TODO, JobState.RUNNING):
                logger.error(f"Failed to move job file {job_file} to running state")
                return False

            # Execute job
            success = execute_job_on_instance(
                instance_info, Path(f"jobs/running/{job_file}")
            )

            if success:
                # Update instance state
                self.state_manager.update_instance_state(
                    instance_id, {"job_state": "running", "current_job": job_file}
                )

                # Update job state
                self.state_manager.update_job_state(
                    job_file,
                    {"state": JobState.RUNNING, "assigned_instance": instance_id},
                )

                logger.info(f"✅ Job {job_file} started on instance {instance_id}")
            else:
                # Move job back to todo
                move_job_file(job_file, JobState.RUNNING, JobState.TODO)
                logger.error(
                    f"❌ Failed to start job {job_file} on instance {instance_id}"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to start job: {e}")
            return False

    @log_function_call
    def _handle_check_job_status(self, task: Task) -> bool:
        """Handle job status check task."""
        try:
            instance_id = task.instance_id
            job_file = task.job_file

            if not instance_id or not job_file:
                logger.warning("Missing instance ID or job file for job status check")
                return False

            logger.debug(
                f"Checking job status for {job_file} on instance {instance_id}"
            )

            # Get instance from state
            if instance_id not in self.state_manager.instances:
                logger.warning(f"Instance {instance_id} not found in state")
                return False

            instance_state = self.state_manager.instances[instance_id]

            # Create InstanceInfo object
            instance_info = InstanceInfo(
                contract_id=int(instance_id),
                machine_id=0,
                gpu_name="Unknown",
                price_per_hour=0.0,
                ssh_host=instance_state.ssh_host,
                ssh_port=instance_state.ssh_port,
                status="running",
                public_ipaddr=instance_state.ssh_host,
                ports={},
            )

            # Check job completion
            completion_status = detect_job_completion(instance_info, job_file)

            if completion_status.completed:
                if completion_status.success:
                    # Job completed successfully
                    move_job_file(job_file, JobState.RUNNING, JobState.COMPLETED)
                    self.state_manager.mark_job_completed(job_file, success=True)

                    # Schedule result collection
                    self.scheduler.schedule_result_collection(instance_info, job_file)

                    logger.info(f"✅ Job {job_file} completed successfully")
                else:
                    # Job failed
                    move_job_file(job_file, JobState.RUNNING, JobState.FAILED)
                    self.state_manager.mark_job_completed(job_file, success=False)

                    logger.error(
                        f"❌ Job {job_file} failed: {completion_status.error_message}"
                    )

                # Update instance state
                self.state_manager.update_instance_state(
                    instance_id, {"job_state": "idle", "current_job": None}
                )

            return True

        except Exception as e:
            logger.error(f"Failed to check job status: {e}")
            return False

    @log_function_call
    def _handle_download_results(self, task: Task) -> bool:
        """Handle result download task."""
        try:
            instance_id = task.instance_id
            job_file = task.job_file

            if not instance_id or not job_file:
                logger.warning("Missing instance ID or job file for result download")
                return False

            logger.info(
                f"Downloading results for job {job_file} from instance {instance_id}"
            )

            # Get instance from state
            if instance_id not in self.state_manager.instances:
                logger.warning(f"Instance {instance_id} not found in state")
                return False

            instance_state = self.state_manager.instances[instance_id]

            # Create InstanceInfo object
            instance_info = InstanceInfo(
                contract_id=int(instance_id),
                machine_id=0,
                gpu_name="Unknown",
                price_per_hour=0.0,
                ssh_host=instance_state.ssh_host,
                ssh_port=instance_state.ssh_port,
                status="running",
                public_ipaddr=instance_state.ssh_host,
                ports={},
            )

            # Collect results
            result_files = collect_job_results(instance_info, job_file)

            if result_files:
                # Process results
                processed_results = process_job_results(job_file, result_files)

                if processed_results.success:
                    logger.info(
                        f"✅ Downloaded {len(result_files)} result files for job {job_file}"
                    )
                else:
                    logger.error(f"❌ Failed to process results for job {job_file}")
            else:
                logger.warning(f"No result files found for job {job_file}")

            return True

        except Exception as e:
            logger.error(f"Failed to download results: {e}")
            return False

    @log_function_call
    def _handle_assign_jobs(self, task: Task) -> bool:
        """Handle job assignment task."""
        try:
            logger.debug("Assigning jobs to idle instances")

            # First, check for unprepared instances and schedule preparation
            self._schedule_preparation_for_unprepared_instances()

            # Get idle instances (only prepared ones)
            idle_instances = self.state_manager.get_idle_instances()

            if not idle_instances:
                logger.debug(
                    "No idle and prepared instances available for job assignment"
                )
                return True

            # Get available jobs
            available_jobs = self.state_manager.get_available_jobs()

            if not available_jobs:
                logger.debug("No available jobs for assignment")
                return True

            # Assign jobs to idle instances
            assigned_count = 0
            for instance in idle_instances:
                if assigned_count >= len(available_jobs):
                    break

                job = available_jobs[assigned_count]

                # Schedule job execution
                if self.scheduler.schedule_job_execution(instance, job.job_file):
                    # Mark job as assigned
                    self.state_manager.mark_job_assigned(
                        job.job_file, str(instance.contract_id)
                    )
                    assigned_count += 1

                    logger.info(
                        f"✅ Assigned job {job.job_file} to instance {instance.contract_id}"
                    )

            logger.info(f"Assigned {assigned_count} jobs to instances")
            return True

        except Exception as e:
            logger.error(f"Failed to assign jobs: {e}")
            return False

    @log_function_call
    def _schedule_preparation_for_unprepared_instances(self):
        """Schedule preparation tasks for instances that are not prepared."""
        try:
            unprepared_instances = self.state_manager.get_unprepared_instances()

            if not unprepared_instances:
                return

            logger.info(
                f"Found {len(unprepared_instances)} unprepared instances, scheduling preparation..."
            )

            for instance_id in unprepared_instances:
                instance_state = self.state_manager.instances[instance_id]

                # Only prepare instances that are running (not starting/error)
                if instance_state.status in [
                    InstanceStatus.RUNNING,
                    InstanceStatus.STARTING,
                ]:
                    # Schedule preparation task
                    prep_task = self.scheduler.create_task(
                        TaskType.PREPARE_INSTANCE,
                        instance_id=instance_id,
                        priority=2,  # High priority for preparation
                    )
                    self.scheduler.schedule_task(prep_task, priority=2)
                    logger.info(f"Scheduled preparation for instance {instance_id}")

        except Exception as e:
            logger.error(
                f"Failed to schedule preparation for unprepared instances: {e}"
            )

    @log_function_call
    def _handle_discover_jobs(self, task: Task) -> bool:
        """Handle job discovery task."""
        try:
            logger.debug("Discovering job files")

            # Discover job files
            job_files = discover_job_files(Path("jobs/todo"))

            if not job_files:
                logger.debug("No job files found")
                return True

            # Prioritize jobs
            prioritized_jobs = prioritize_jobs(job_files)

            # Update state manager
            for job_file in prioritized_jobs:
                job_name = job_file.name

                if job_name not in self.state_manager.jobs:
                    from ..core.files import JobState
                    from ..core.state import JobStateInfo

                    job_info = JobStateInfo(job_file=job_name, state=JobState.TODO)
                    self.state_manager.jobs[job_name] = job_info
                    logger.debug(f"Discovered new job: {job_name}")

            self.state_manager.save_state()
            logger.info(f"Discovered {len(prioritized_jobs)} job files")
            return True

        except Exception as e:
            logger.error(f"Failed to discover jobs: {e}")
            return False

    @log_function_call
    def get_orchestration_state(self) -> OrchestrationState:
        """Get current orchestration state with live verification."""
        try:
            # Get live instance data from vast.ai API
            live_instances = discover_active_instances()

            # Count instances by live status
            total_instances = len(live_instances)
            active_instances = sum(
                1
                for i in live_instances
                if i.status in [InstanceStatus.RUNNING, InstanceStatus.READY]
            )

            # Count jobs by live verification (only check a few to avoid overwhelming the system)
            total_jobs = len(self.state_manager.jobs)
            pending_jobs = sum(
                1 for j in self.state_manager.jobs.values() if j.state == JobState.TODO
            )

            # For running jobs, do live verification on a sample to get accurate count
            # This is expensive, so we'll do it selectively
            actually_running_jobs = 0
            actually_idle_instances = 0

            # Check instances that are marked as having running jobs
            for instance_state in self.state_manager.instances.values():
                if (
                    instance_state.job_state in ["running", "assigned"]
                    and instance_state.current_job
                ):
                    # Find the corresponding live instance
                    live_instance = None
                    for li in live_instances:
                        if str(li.contract_id) == instance_state.contract_id:
                            live_instance = li
                            break

                    if live_instance:
                        # Check if the job is actually running
                        try:
                            from ..execution.process_monitor import (
                                monitor_job_execution,
                            )

                            job_status = monitor_job_execution(
                                live_instance, instance_state.current_job
                            )
                            if job_status.value == "running":
                                actually_running_jobs += 1
                            else:
                                # Job is not actually running, mark as completed/failed
                                logger.info(
                                    f"Job {instance_state.current_job} on instance {instance_state.contract_id} is not actually running (status: {job_status.value})"
                                )
                                # Update state to reflect reality
                                job_info = self.state_manager.jobs.get(
                                    instance_state.current_job
                                )
                                if job_info:
                                    job_info.state = (
                                        JobState.COMPLETED
                                        if job_status.value == "completed"
                                        else JobState.FAILED
                                    )
                                    job_info.end_time = datetime.now()
                                    instance_state.job_state = "idle"
                                    instance_state.current_job = None
                        except Exception as e:
                            logger.warning(
                                f"Could not verify job status for {instance_state.current_job}: {e}"
                            )
                            # If we can't verify, assume the job has finished (conservative approach)
                            logger.info(
                                f"Assuming job {instance_state.current_job} has finished due to verification failure"
                            )
                            job_info = self.state_manager.jobs.get(
                                instance_state.current_job
                            )
                            if job_info:
                                job_info.state = (
                                    JobState.FAILED
                                )  # Mark as failed since we can't verify
                                job_info.end_time = datetime.now()
                                instance_state.job_state = "idle"
                                instance_state.current_job = None
                    else:
                        logger.warning(
                            f"Instance {instance_state.contract_id} not found in live instances"
                        )

            # Count idle instances (instances that are ready and not assigned to jobs)
            for live_instance in live_instances:
                if live_instance.status in [
                    InstanceStatus.RUNNING,
                    InstanceStatus.READY,
                ]:
                    # Check if this instance has any running jobs
                    has_running_job = False
                    for job_info in self.state_manager.jobs.values():
                        if (
                            job_info.state == JobState.RUNNING
                            and job_info.assigned_instance
                            == str(live_instance.contract_id)
                        ):
                            has_running_job = True
                            break

                    if not has_running_job:
                        actually_idle_instances += 1

            completed_jobs = sum(
                1
                for j in self.state_manager.jobs.values()
                if j.state == JobState.COMPLETED
            )

            state = OrchestrationState(
                total_instances=total_instances,
                active_instances=active_instances,
                idle_instances=actually_idle_instances,
                busy_instances=total_instances - actually_idle_instances,
                total_jobs=total_jobs,
                pending_jobs=pending_jobs,
                running_jobs=actually_running_jobs,
                completed_jobs=completed_jobs,
            )

            return state

        except Exception as e:
            logger.error(f"Failed to get orchestration state: {e}")
            return OrchestrationState(0, 0, 0, 0, 0, 0, 0, 0)

    @log_function_call
    def start_orchestration(self) -> None:
        """Start the orchestration system."""
        try:
            if self.running:
                logger.warning("Orchestration is already running")
                return

            self.running = True

            # Start the scheduler
            self.scheduler.start_scheduler()

            # Perform startup verification of instance preparation status
            self._verify_instance_preparation_on_startup()

            logger.info("✅ Orchestration system started")

        except Exception as e:
            logger.error(f"Failed to start orchestration: {e}")

    @log_function_call
    def _verify_instance_preparation_on_startup(self) -> None:
        """Verify preparation status of all instances on startup."""
        try:
            logger.info("🔍 Verifying instance preparation status on startup...")

            # Get all instances from state
            instances_to_check = list(self.state_manager.instances.keys())

            if not instances_to_check:
                logger.info("No instances found in state to verify")
                return

            logger.info(f"Found {len(instances_to_check)} instances to verify")

            for instance_id in instances_to_check:
                instance_state = self.state_manager.instances[instance_id]

                # Skip if already marked as prepared
                if instance_state.is_prepared:
                    logger.info(
                        f"Instance {instance_id} already marked as prepared, skipping verification"
                    )
                    continue

                logger.info(
                    f"Verifying preparation status for instance {instance_id}..."
                )

                # Create InstanceInfo object for verification
                instance_info = InstanceInfo(
                    contract_id=int(instance_id),
                    machine_id=0,
                    gpu_name="Unknown",
                    price_per_hour=0.0,
                    ssh_host=instance_state.ssh_host,
                    ssh_port=instance_state.ssh_port,
                    status="running",
                    public_ipaddr=instance_state.ssh_host,
                    ports={},
                )

                # Verify instance readiness
                try:
                    from ..core.ssh import create_ssh_connection

                    ssh_client = create_ssh_connection(
                        instance_state.ssh_host, instance_state.ssh_port
                    )
                    ssh_client.connect()

                    try:
                        from ..instances.preparation import verify_instance_readiness

                        is_ready = verify_instance_readiness(instance_info, ssh_client)

                        if is_ready:
                            logger.info(
                                f"✅ Instance {instance_id} is prepared and ready"
                            )
                            self.state_manager.mark_instance_prepared(instance_id)
                        else:
                            logger.info(
                                f"⚠️ Instance {instance_id} is not prepared, will be prepared when needed"
                            )
                            # Keep as STARTING status, will be prepared when needed

                    finally:
                        ssh_client.close()

                except Exception as e:
                    logger.warning(f"Failed to verify instance {instance_id}: {e}")
                    # Keep instance as STARTING, will be prepared when needed

            logger.info("✅ Instance preparation verification completed")

        except Exception as e:
            logger.error(f"Failed to verify instance preparation on startup: {e}")

    @log_function_call
    def stop_orchestration(self):
        """Stop the orchestration system."""
        try:
            if not self.running:
                logger.warning("Orchestration is not running")
                return

            self.running = False

            # Stop the scheduler
            self.scheduler.stop_scheduler()

            logger.info("✅ Orchestration system stopped")

        except Exception as e:
            logger.error(f"Failed to stop orchestration: {e}")


# Global coordinator instance
_coordinator: OrchestrationCoordinator | None = None


def get_coordinator() -> OrchestrationCoordinator:
    """Get the global coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = OrchestrationCoordinator()
    return _coordinator
