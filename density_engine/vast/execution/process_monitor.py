"""
Process monitoring for the vast.ai automation system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from ..core.ssh import SSHClient, create_ssh_connection, execute_command
from ..instances.discovery import InstanceInfo
from ..utils.exceptions import JobExecutionError
from ..utils.logging import get_logger, log_function_call
from .job_runner import ProcessInfo

logger = get_logger(__name__)


@dataclass
class InstanceJobStatus:
    """Status of jobs on an instance."""

    csv_files_count: int
    parquet_files_count: int
    csv_files: list[str]
    parquet_files: list[str]
    completed_jobs: int
    pending_jobs: int
    status_summary: str


class JobStatus(Enum):
    """Job status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class JobLogInfo:
    """Job log information."""

    log_file: str
    content: str
    last_lines: list[str]
    has_errors: bool
    has_completion: bool


@dataclass
class CompletionStatus:
    """Job completion status."""

    completed: bool
    success: bool
    status: JobStatus
    error_message: str | None = None
    result_files: list[str] | None = None


@log_function_call
def find_process_by_name(
    instance: InstanceInfo, process_name: str
) -> list[ProcessInfo]:
    """Find processes by name on an instance."""
    try:
        logger.debug(
            f"Finding processes by name '{process_name}' on instance {instance.contract_id}"
        )

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Use ps aux to find processes (exclude bash wrappers)
            cmd = f"ps aux | grep '{process_name}' | grep -v grep | grep -v 'bash -c'"
            logger.info(f"Executing process search command: {cmd}")
            result = execute_command(ssh_client, cmd, timeout=10)

            logger.info(
                f"Process search result: success={result.success}, stdout='{result.stdout.strip()}', stderr='{result.stderr.strip()}'"
            )

            processes = []
            if result.success and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 11:
                        pid = parts[1]
                        command = " ".join(parts[10:])

                        process_info = ProcessInfo(
                            pid=pid,
                            command=command,
                            start_time="",  # ps aux doesn't show start time
                            status="running",
                        )
                        processes.append(process_info)

            logger.debug(f"Found {len(processes)} processes matching '{process_name}'")
            return processes

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to find processes by name: {e}")
        return []


def check_process_running(instance: InstanceInfo, process_id: str) -> bool:
    """Check if a process is running on an instance."""
    try:
        logger.debug(
            f"Checking if process {process_id} is running on instance {instance.contract_id}"
        )

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Check if process exists
            cmd = f"ps -p {process_id} > /dev/null 2>&1 && echo 'running' || echo 'finished'"
            result = execute_command(ssh_client, cmd, timeout=10)

            is_running = result.success and result.stdout.strip() == "running"
            logger.debug(
                f"Process {process_id} is {'running' if is_running else 'not running'}"
            )
            return is_running  # type: ignore[return]

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to check process running: {e}")
        return False


def get_process_output(instance: InstanceInfo, log_file: str) -> str:
    """Get process output from log file."""
    try:
        logger.debug(
            f"Getting process output from {log_file} on instance {instance.contract_id}"
        )

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Check if log file exists
            check_cmd = f"test -f {log_file} && echo 'exists' || echo 'missing'"
            check_result = execute_command(ssh_client, check_cmd, timeout=10)

            if not check_result.success or check_result.stdout.strip() != "exists":
                logger.warning(f"Log file {log_file} does not exist")
                return ""

            # Get log file content
            cat_cmd = f"cat {log_file}"
            result = execute_command(ssh_client, cat_cmd, timeout=30)

            if result.success:
                logger.debug(f"Retrieved {len(result.stdout)} characters from log file")
                return result.stdout  # type: ignore[return]
            else:
                logger.warning(f"Failed to read log file: {result.stderr}")
                return ""

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to get process output: {e}")
        return ""


@log_function_call
def parse_job_log(log_content: str) -> JobLogInfo:
    """Parse job log content."""
    try:
        logger.debug("Parsing job log content")

        lines = log_content.split("\n")
        last_lines = lines[-20:] if len(lines) > 20 else lines  # Last 20 lines

        # Check for errors
        has_errors = any(
            error_indicator in log_content.lower()
            for error_indicator in ["error", "failed", "exception", "traceback"]
        )

        # Check for completion
        has_completion = any(
            completion_indicator in log_content.lower()
            for completion_indicator in [
                "completed successfully",
                "finished",
                "done",
                "success",
            ]
        )

        log_info = JobLogInfo(
            log_file="",
            content=log_content,
            last_lines=last_lines,
            has_errors=has_errors,
            has_completion=has_completion,
        )

        logger.debug(
            f"Parsed log: {len(lines)} lines, errors: {has_errors}, completion: {has_completion}"
        )
        return log_info

    except Exception as e:
        logger.error(f"Failed to parse job log: {e}")
        return JobLogInfo(
            log_file="",
            content=log_content,
            last_lines=[],
            has_errors=False,
            has_completion=False,
        )


@log_function_call
def monitor_job_execution(instance: InstanceInfo, job_file: str) -> JobStatus:
    """Monitor job execution status."""
    try:
        logger.debug(
            f"Monitoring job execution for {job_file} on instance {instance.contract_id}"
        )

        # Find processes running the job
        processes = find_process_by_name(instance, f"run_garch_jobs.py.*{job_file}")

        if processes:
            logger.debug(f"Job {job_file} is running (PID: {processes[0].pid})")
            return JobStatus.RUNNING
        else:
            # Check log file for completion status
            log_file = f"/root/density-engine/{job_file}.log"
            log_content = get_process_output(instance, log_file)

            if log_content:
                log_info = parse_job_log(log_content)

                if log_info.has_completion:
                    logger.info(f"Job {job_file} completed successfully")
                    return JobStatus.COMPLETED
                elif log_info.has_errors:
                    logger.warning(f"Job {job_file} failed with errors")
                    return JobStatus.FAILED
                else:
                    logger.warning(f"Job {job_file} status unknown")
                    return JobStatus.UNKNOWN
            else:
                logger.warning(f"Job {job_file} not running and no log file found")
                return JobStatus.UNKNOWN

    except Exception as e:
        logger.error(f"Failed to monitor job execution: {e}")
        return JobStatus.UNKNOWN


@log_function_call
def find_parquet_files_for_job(instance: InstanceInfo, job_file: str) -> list[str]:
    """Find parquet files that start with the job file name."""
    try:
        logger.debug(
            f"Looking for parquet files for job {job_file} on instance {instance.contract_id}"
        )

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Get job file base name (without .csv extension)
            job_base = job_file.replace(".csv", "")

            # Look for parquet files that start with the job base name
            find_cmd = f"find /root/density-engine -name '{job_base}*.parquet' -type f"
            result = execute_command(ssh_client, find_cmd, timeout=10)

            if result.success and result.stdout.strip():
                parquet_files = [
                    f.strip() for f in result.stdout.strip().split("\n") if f.strip()
                ]
                logger.info(
                    f"Found {len(parquet_files)} parquet files for job {job_file}: {parquet_files}"
                )
                return parquet_files
            else:
                logger.debug(f"No parquet files found for job {job_file}")
                return []

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to find parquet files for job {job_file}: {e}")
        return []


@log_function_call
def compute_instance_job_status(instance: InstanceInfo) -> InstanceJobStatus:
    """Compute instance job status using simple process detection + file comparison."""
    try:
        logger.debug(f"Computing job status for instance {instance.contract_id}")

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # 1. Check for running jobs using ps aux (count only actual Python processes, not bash wrappers)
            ps_cmd = "ps aux | grep 'python3 scripts/run_garch_jobs.py' | grep -v grep | grep -v 'bash -c' | wc -l"
            ps_result = execute_command(ssh_client, ps_cmd, timeout=10)
            running_jobs = int(ps_result.stdout.strip()) if ps_result.success else 0

            # Debug: Also get the actual process list (filtered)
            ps_debug_cmd = "ps aux | grep 'python3 scripts/run_garch_jobs.py' | grep -v grep | grep -v 'bash -c'"
            ps_debug_result = execute_command(ssh_client, ps_debug_cmd, timeout=10)
            if ps_debug_result.success and ps_debug_result.stdout.strip():
                lines = ps_debug_result.stdout.strip().split("\n")
                logger.info(f"Found {len(lines)} actual Python processes:")
                for i, line in enumerate(lines, 1):
                    logger.info(f"  Process {i}: {line}")
            else:
                logger.info("No actual Python processes found")

            # 2. Get list of CSV files (input jobs)
            csv_cmd = "find /root/density-engine -name '*.csv' 2>/dev/null | wc -l"
            csv_result = execute_command(ssh_client, csv_cmd, timeout=10)
            csv_count = int(csv_result.stdout.strip()) if csv_result.success else 0

            # 3. Get list of parquet files (completed jobs) - same directory as CSV files
            parquet_cmd = (
                "find /root/density-engine -name '*.parquet' 2>/dev/null | wc -l"
            )
            parquet_result = execute_command(ssh_client, parquet_cmd, timeout=10)
            parquet_count = (
                int(parquet_result.stdout.strip()) if parquet_result.success else 0
            )

            # 4. Calculate not completed jobs
            not_completed_jobs = csv_count - parquet_count

            # 5. Determine if we can start a new job
            can_start_job = (running_jobs == 0) and (not_completed_jobs > 0)

            status = InstanceJobStatus(
                csv_files_count=csv_count,
                parquet_files_count=parquet_count,
                csv_files=[],  # Not needed for simple logic
                parquet_files=[],  # Not needed for simple logic
                completed_jobs=parquet_count,
                pending_jobs=not_completed_jobs,
                status_summary=f"{parquet_count} completed, {not_completed_jobs} not completed, {running_jobs} running",
            )

            logger.info(
                f"Instance {instance.contract_id}: {status.status_summary}, can_start: {can_start_job}"
            )
            return status

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to compute instance job status: {e}")
        return InstanceJobStatus(
            csv_files_count=0,
            parquet_files_count=0,
            csv_files=[],
            parquet_files=[],
            completed_jobs=0,
            pending_jobs=0,
            status_summary="Error computing status",
        )


@log_function_call
def detect_job_completion(instance: InstanceInfo, job_file: str) -> CompletionStatus:
    """Detect if a job has completed."""
    try:
        logger.debug(
            f"Detecting job completion for {job_file} on instance {instance.contract_id}"
        )

        # Check if process is still running (look for the specific job file in the command)
        processes = find_process_by_name(instance, f"run_garch_jobs.py.*{job_file}")

        # Debug logging
        logger.info(f"Looking for process pattern: 'run_garch_jobs.py.*{job_file}'")
        logger.info(f"Found {len(processes)} matching processes")
        for proc in processes:
            logger.info(f"  Process: PID={proc.pid}, Command={proc.command}")

        if processes:
            # Job is still running
            return CompletionStatus(
                completed=False, success=False, status=JobStatus.RUNNING
            )

        # Job is not running, check for parquet files
        parquet_files = find_parquet_files_for_job(instance, job_file)

        if parquet_files:
            # Found parquet files - job completed successfully
            logger.info(
                f"✅ Found {len(parquet_files)} parquet files for job {job_file}"
            )

            return CompletionStatus(
                completed=True,
                success=True,
                status=JobStatus.COMPLETED,
                result_files=parquet_files,
            )

        # No parquet files found, check log for errors
        log_file = f"/root/density-engine/{job_file}.log"
        log_content = get_process_output(instance, log_file)

        if not log_content:
            return CompletionStatus(
                completed=True,
                success=False,
                status=JobStatus.UNKNOWN,
                error_message="No log file found and no parquet files",
            )

        log_info = parse_job_log(log_content)

        if log_info.has_errors:
            # Log the full error details for debugging
            logger.error(f"Job {job_file} failed. Full log content:")
            logger.error(f"Log content: {log_content}")

            # Extract the actual error message from the log
            error_lines = [
                line
                for line in log_info.last_lines
                if any(
                    error_indicator in line.lower()
                    for error_indicator in ["error", "failed", "exception", "traceback"]
                )
            ]
            error_message = "Job failed with errors"
            if error_lines:
                error_message += (
                    f": {'; '.join(error_lines[-3:])}"  # Last 3 error lines
                )

            return CompletionStatus(
                completed=True,
                success=False,
                status=JobStatus.FAILED,
                error_message=error_message,
            )
        else:
            # Process not running, no parquet files, no errors in log
            # This could be a timeout or unexpected termination
            return CompletionStatus(
                completed=True,
                success=False,
                status=JobStatus.TIMEOUT,
                error_message="Job terminated without producing results",
            )

    except Exception as e:
        logger.error(f"Failed to detect job completion: {e}")
        return CompletionStatus(
            completed=True,
            success=False,
            status=JobStatus.UNKNOWN,
            error_message=str(e),
        )


@log_function_call
def handle_job_timeout(instance: InstanceInfo, job_file: str) -> bool:
    """Handle job timeout."""
    try:
        logger.warning(
            f"Handling job timeout for {job_file} on instance {instance.contract_id}"
        )

        # Find and kill the job process
        processes = find_process_by_name(instance, f"garch.*{job_file}")

        if processes:
            # Create SSH connection
            ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
            ssh_client.connect()

            try:
                # Kill the process
                for process in processes:
                    kill_cmd = f"kill -9 {process.pid}"
                    result = execute_command(ssh_client, kill_cmd, timeout=10)

                    if result.success:
                        logger.info(f"✅ Killed job process {process.pid}")
                    else:
                        logger.warning(
                            f"Failed to kill process {process.pid}: {result.stderr}"
                        )

            finally:
                ssh_client.close()

        # Update job state
        from ..core.state import get_state_manager

        state_manager = get_state_manager()
        state_manager.mark_job_completed(job_file, success=False)

        logger.info(f"✅ Job timeout handled for {job_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to handle job timeout: {e}")
        return False
