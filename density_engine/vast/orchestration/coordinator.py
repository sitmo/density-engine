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
                    # New instance - check if already prepared before marking as unprepared
                    logger.info(f"Discovered new instance: {instance_id}, checking if already prepared...")
                    
                    # Create InstanceInfo for verification
                    instance_info = InstanceInfo(
                        contract_id=int(instance_id),
                        machine_id=0,
                        gpu_name="Unknown",
                        price_per_hour=0.0,
                        ssh_host=instance.ssh_host,
                        ssh_port=instance.ssh_port,
                        status="running",
                        public_ipaddr=instance.ssh_host,
                        ports={},
                    )
                    
                    # Check if instance is already prepared
                    from ..instances.preparation import verify_instance_readiness
                    is_already_prepared = verify_instance_readiness(instance_info)
                    
                    instance_state = InstanceState(
                        contract_id=instance_id,
                        ssh_host=instance.ssh_host,
                        ssh_port=instance.ssh_port,
                        status=InstanceStatus.RUNNING if is_already_prepared else InstanceStatus.STARTING,
                        last_updated=task.scheduled_time,
                        is_prepared=is_already_prepared,
                        preparation_verified_at=task.scheduled_time if is_already_prepared else None,
                    )
                    self.state_manager.instances[instance_id] = instance_state
                    
                    if is_already_prepared:
                        logger.info(f"✅ Instance {instance_id} is already prepared and ready!")
                    else:
                        logger.info(f"Instance {instance_id} needs preparation")
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

            # Check if instance is already prepared
            if instance_state.is_prepared:
                logger.info(f"Instance {instance_id} is already prepared, skipping preparation")
                return True

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

            return success  # type: ignore[return]

        except Exception as e:
            logger.error(f"Failed to prepare instance: {e}")
            return False

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

            # Execute job (upload from local jobs/running/ but to remote root directory)
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
                logger.debug(f"Job {job_file} assignment skipped (likely already running)")

            return success  # type: ignore[return]

        except Exception as e:
            logger.error(f"Failed to start job: {e}")
            return False

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
                    # Job failed - check if we should retry
                    job_info = self.state_manager.get_job(job_file)
                    if job_info and job_info.retry_count < job_info.max_retries:
                        # Retry the job
                        job_info.retry_count += 1
                        logger.warning(
                            f"🔄 Retrying job {job_file} (attempt {job_info.retry_count}/{job_info.max_retries}): {completion_status.error_message}"
                        )
                        
                        # Move job back to todo for retry
                        move_job_file(job_file, JobState.RUNNING, JobState.TODO)
                        self.state_manager.update_job_state(job_file, {
                            "state": JobState.TODO,
                            "assigned_instance": None,
                            "retry_count": job_info.retry_count
                        })
                        
                        # Update instance state
                        self.state_manager.update_instance_state(
                            instance_id, {"job_state": "idle", "current_job": None}
                        )
                    else:
                        # Max retries exceeded, mark as permanently failed
                        move_job_file(job_file, JobState.RUNNING, JobState.FAILED)
                        self.state_manager.mark_job_completed(job_file, success=False)

                        logger.error(
                            f"❌ Job {job_file} permanently failed after {job_info.retry_count if job_info else 0} retries: {completion_status.error_message}"
                        )
                        
                        # Update instance state
                        self.state_manager.update_instance_state(
                            instance_id, {"job_state": "idle", "current_job": None}
                        )

            return True

        except Exception as e:
            logger.error(f"Failed to check job status: {e}")
            return False

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

                # Check remote instance status before assigning new job
                from ..execution.process_monitor import compute_instance_job_status
                remote_status = compute_instance_job_status(instance)
                
                # CRITICAL: Only assign if there are pending jobs (CSV files without parquet files)
                if remote_status.pending_jobs == 0:
                    logger.info(f"Instance {instance.contract_id} has no pending jobs (all CSV files have parquet files), skipping assignment")
                    continue
                
                # CRITICAL: Check if ANY run_garch_jobs.py is running (ONLY ONE JOB PER INSTANCE)
                if "running" in remote_status.status_summary and "0 running" not in remote_status.status_summary:
                    logger.warning(f"Instance {instance.contract_id} already has running jobs, skipping assignment to prevent multiple jobs")
                    continue

                job = available_jobs[assigned_count]

                # Check if job is already assigned or running to prevent duplicates
                job_info = self.state_manager.get_job(job.job_file)
                if job_info and job_info.state in [JobState.RUNNING, JobState.COMPLETED]:
                    logger.debug(f"Job {job.job_file} is already {job_info.state.value}, skipping duplicate assignment")
                    continue

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
    def _schedule_preparation_for_unprepared_instances(self) -> None:
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
        """Get current orchestration state using cached data to avoid API hammering."""
        try:
            # Use cached instance data instead of making API calls every time
            # This prevents hammering the vast.ai API with frequent requests
            
            # Count instances from cached state
            total_instances = len(self.state_manager.instances)
            active_instances = sum(
                1
                for instance in self.state_manager.instances.values()
                if instance.status in [InstanceStatus.RUNNING, InstanceStatus.READY]
            )
            idle_instances = len(self.state_manager.get_idle_instances())
            
            # Count jobs from cached state
            pending_jobs = len(self.state_manager.get_available_jobs())
            running_jobs = sum(
                1
                for job in self.state_manager.jobs.values()
                if job.state.value == "running"
            )
            completed_jobs = sum(
                1
                for job in self.state_manager.jobs.values()
                if job.state.value == "completed"
            )
            total_jobs = len(self.state_manager.jobs)

            # Create orchestration state using cached data
            busy_instances = total_instances - idle_instances
            state = OrchestrationState(
                total_instances=total_instances,
                active_instances=active_instances,
                idle_instances=idle_instances,
                busy_instances=busy_instances,
                pending_jobs=pending_jobs,
                running_jobs=running_jobs,
                completed_jobs=completed_jobs,
                total_jobs=total_jobs,
            )

            logger.debug(
                f"Orchestration state: {active_instances} active, {idle_instances} idle, "
                f"{pending_jobs} pending, {running_jobs} running, {completed_jobs} completed"
            )

            return state

        except Exception as e:
            logger.error(f"Failed to get orchestration state: {e}")
            # Return empty state on error
            return OrchestrationState(
                total_instances=0,
                active_instances=0,
                idle_instances=0,
                busy_instances=0,
                pending_jobs=0,
                running_jobs=0,
                completed_jobs=0,
                total_jobs=0,
            )

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
    def stop_orchestration(self) -> None:
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
