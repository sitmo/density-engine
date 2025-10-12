"""
Job execution for the vast.ai automation system.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.jobs import parse_job_file
from ..core.ssh import SSHClient, create_ssh_connection, execute_command, upload_file
from ..instances.discovery import InstanceInfo
from ..utils.exceptions import JobExecutionError
from ..utils.logging import get_logger, log_function_call

logger = get_logger(__name__)


@dataclass
class ProcessInfo:
    """Process information."""

    pid: str
    command: str
    start_time: str
    status: str


@dataclass
class JobExecutionResult:
    """Job execution result."""

    success: bool
    job_file: str
    instance_id: str
    start_time: str
    end_time: str | None
    log_file: str
    error_message: str | None = None


@log_function_call
def upload_job_file(instance: InstanceInfo, job_file: Path) -> bool:
    """Upload a job file to an instance."""
    try:
        logger.info(f"Uploading job file {job_file} to instance {instance.contract_id}")

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Ensure remote directory exists
            remote_dir = "/root/density-engine"
            mkdir_cmd = f"mkdir -p {remote_dir}"
            result = execute_command(ssh_client, mkdir_cmd, timeout=10)
            if not result.success:
                logger.warning(
                    f"Failed to create remote directory {remote_dir}: {result.stderr}"
                )

            # Upload the job file directly to /root/density-engine/ (root directory)
            # Extract just the filename, ignoring any subdirectory structure
            remote_path = f"{remote_dir}/{job_file.name}"
            success = upload_file(ssh_client, str(job_file), remote_path)

            if success:
                logger.info(f"✅ Job file {job_file.name} uploaded successfully")
                return True
            else:
                logger.error(f"❌ Failed to upload job file {job_file.name}")
                return False

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to upload job file: {e}")
        return False


@log_function_call
def start_job_process(
    instance: InstanceInfo, job_file: str, args: dict[str, Any]
) -> ProcessInfo:
    """Start a job process on an instance."""
    try:
        logger.info(
            f"Starting job process {job_file} on instance {instance.contract_id}"
        )

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Generate job command
            cmd = generate_job_command(job_file, args)

            # Create log file path
            log_file = f"/root/density-engine/{job_file}.log"

            # Start job in background using nohup
            background_cmd = (
                f"cd /root/density-engine && nohup bash -c 'source /venv/main/bin/activate && {cmd}' > {log_file} 2>&1 &"
            )

            logger.info(f"🚀 Executing command: {background_cmd}")
            result = execute_command(ssh_client, background_cmd, timeout=30)

            if not result.success:
                logger.info(f"Job already running or duplicate prevented: {result.stderr}")
                # This is not an error - it's expected behavior when job is already running
                # Return a ProcessInfo indicating the job is already running
                return ProcessInfo(
                    pid="already_running",
                    command=cmd,
                    start_time=str(
                        ssh_client.execute_command("date", timeout=5).stdout.strip()
                    ),
                    status="already_running",
                )

            # Give the job a moment to start and write to log file
            import time
            time.sleep(2)
            
            # Check if log file exists and has content
            log_check_cmd = f"test -f {log_file} && wc -l {log_file} || echo 'log file not found'"
            log_check_result = execute_command(ssh_client, log_check_cmd, timeout=10)
            if log_check_result.success:
                logger.info(f"📄 Log file status: {log_check_result.stdout.strip()}")
            
            # Try to get the process ID
            pid_cmd = f"ps aux | grep 'python3 scripts/run_garch_jobs.py {job_file}' | grep -v grep | awk '{{print $2}}'"
            pid_result = execute_command(ssh_client, pid_cmd, timeout=10)

            pid = "unknown"
            if pid_result.success and pid_result.stdout.strip():
                pid = pid_result.stdout.strip().split("\n")[0]
                logger.info(f"📋 Job started with PID: {pid}")
            else:
                # Exit code 1 from grep is normal when no process is found - not an error
                logger.info(f"📋 No running process found for job {job_file} (this is normal if job finished quickly)")
                # Check if the job failed immediately by looking at the log
                if log_check_result.success and "log file not found" not in log_check_result.stdout:
                    cat_cmd = f"cat {log_file}"
                    cat_result = execute_command(ssh_client, cat_cmd, timeout=10)
                    if cat_result.success and cat_result.stdout.strip():
                        logger.error(f"Job may have failed immediately. Log content: {cat_result.stdout.strip()}")

            process_info = ProcessInfo(
                pid=pid,
                command=cmd,
                start_time=str(
                    ssh_client.execute_command("date", timeout=5).stdout.strip()
                ),
                status="running",
            )

            logger.info(f"✅ Job {job_file} started in background")
            return process_info

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to start job process: {e}")
        raise JobExecutionError(f"Failed to start job process: {e}")


@log_function_call
def generate_job_command(job_file: str, args: dict[str, Any]) -> str:
    """Generate the command to run a job."""
    try:
        # Build arguments string
        args_str = " ".join([f"--{key} {value}" for key, value in args.items()])

        # Generate the command (without virtual environment activation - handled by bash wrapper)
        cmd = f"python3 scripts/run_garch_jobs.py /root/density-engine/{job_file} {args_str}"

        logger.debug(f"Generated command: {cmd}")
        return cmd

    except Exception as e:
        logger.error(f"Failed to generate job command: {e}")
        raise JobExecutionError(f"Failed to generate job command: {e}")


@log_function_call
def create_job_log_file(instance: InstanceInfo, job_file: str) -> str:
    """Create a job log file path."""
    log_file = f"/root/density-engine/{job_file}.log"
    logger.debug(f"Job log file: {log_file}")
    return log_file


@log_function_call
def execute_job_on_instance(instance: InstanceInfo, job_file: Path) -> bool:
    """Execute a job on an instance."""
    try:
        logger.info(f"Executing job {job_file.name} on instance {instance.contract_id}")

        # Upload job file
        if not upload_job_file(instance, job_file):
            return False

        # Use default job arguments
        args = {"num_sim": 1000000, "num_quantiles": 512, "stride": 1}

        # Start job process
        process_info = start_job_process(instance, job_file.name, args)

        logger.info(f"✅ Job {job_file.name} execution started")
        return True

    except Exception as e:
        logger.error(f"Failed to execute job on instance: {e}")
        return False


@log_function_call
def run_job_with_monitoring(
    instance: InstanceInfo, job_file: Path
) -> JobExecutionResult:
    """Run a job with monitoring."""
    try:
        logger.info(
            f"Running job {job_file.name} with monitoring on instance {instance.contract_id}"
        )

        # Upload job file
        if not upload_job_file(instance, job_file):
            return JobExecutionResult(
                success=False,
                job_file=job_file.name,
                instance_id=str(instance.contract_id),
                start_time="",
                end_time="",
                log_file="",
                error_message="Failed to upload job file",
            )

        # Use default job arguments
        args = {"num_sim": 1000000, "num_quantiles": 512, "stride": 1}

        # Start job process
        process_info = start_job_process(instance, job_file.name, args)

        # Create log file path
        log_file = create_job_log_file(instance, job_file.name)

        result = JobExecutionResult(
            success=True,
            job_file=job_file.name,
            instance_id=str(instance.contract_id),
            start_time=process_info.start_time,
            end_time=None,
            log_file=log_file,
        )

        logger.info(f"✅ Job {job_file.name} started with monitoring")
        return result

    except Exception as e:
        logger.error(f"Failed to run job with monitoring: {e}")
        return JobExecutionResult(
            success=False,
            job_file=job_file.name,
            instance_id=str(instance.contract_id),
            start_time="",
            end_time="",
            log_file="",
            error_message=str(e),
        )


@log_function_call
def handle_job_execution_error(
    instance: InstanceInfo, job_file: str, error: Exception
) -> bool:
    """Handle job execution error."""
    try:
        logger.error(
            f"Handling job execution error for {job_file} on instance {instance.contract_id}: {error}"
        )

        # Log the error
        logger.error(f"Job execution failed: {error}")

        # Update instance state to indicate error
        from ..core.state import get_state_manager

        state_manager = get_state_manager()
        state_manager.update_instance_state(
            str(instance.contract_id), {"job_state": "error", "current_job": None}
        )

        # Mark job as failed
        state_manager.mark_job_completed(job_file, success=False)

        logger.info(f"✅ Job execution error handled for {job_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to handle job execution error: {e}")
        return False
