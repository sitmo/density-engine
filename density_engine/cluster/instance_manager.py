"""
Instance manager for individual cluster instances.

This module handles the per-instance logic including SSH communication,
job execution, and status monitoring.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .job_manager import JobManager
from .ssh_client import SSHClient

logger = logging.getLogger("INS")


class InstanceManager:
    """Manages a single cluster instance."""

    def __init__(
        self,
        instance_id: str,
        host: str,
        port: int,
        job_manager: JobManager,
        ssh_key_files: list[str],
        cluster_manager: Any | None = None,
        repo_dir: str = "/root/density-engine",
        venv_python: str = "/venv/main/bin/python",
        preparation_marker: str = "/root/density-engine/.prepared",
        required_packages: list[str] | None = None,
        github_repo: str = "https://github.com/sitmo/density-engine.git",
    ) -> None:
        self.instance_id = instance_id
        self.host = host
        self.port = port
        self.job_manager = job_manager

        # Debug logging
        logger.debug(f"Created InstanceManager {instance_id}: {host}:{port}")
        self.cluster_manager = cluster_manager
        self.repo_dir = repo_dir
        self.venv_python = venv_python
        self.preparation_marker = preparation_marker
        self.required_packages = required_packages or [
            "pandas",
            "pyarrow",
            "scipy",
            "numpy",
        ]
        self.github_repo = github_repo

        # SSH client
        self.ssh = SSHClient(
            host=host,
            port=port,
            username="root",
            key_files=ssh_key_files,
        )

        # Instance state
        self._closed = False
        self.last_check_time: float = 0.0
        self.next_check_time: float = 0.0
        self.error_count: int = 0
        self.last_success_time: float = 0.0
        self.last_reachable_time: float = 0.0  # Track when instance was last reachable

        # API state information
        self.api_state: str = "unknown"
        self.api_data: dict[str, Any] = {}

        # Job state
        self.current_job: str | None = None
        self.job_start_time: float = 0.0
        self.job_last_activity: float = 0.0

        # Preparation state
        self.is_prepared: bool = False
        self.preparation_checked: bool = False

    @property
    def host_port(self) -> str:
        """Get host:port string for logging."""
        return f"{self.host}:{self.port}"

    def update_api_state(self, api_data: dict[str, Any]) -> None:
        """Update API state information from cluster manager."""
        self.api_data = api_data

        # Extract state information from API data
        state_fields = ["cur_state", "actual_status", "intended_status", "status"]
        for field in state_fields:
            if field in api_data and api_data[field]:
                self.api_state = str(api_data[field]).lower()
                break

    async def get_instance_status(self) -> dict[str, Any]:
        """Get complete instance status including job information."""
        try:
            # Try to communicate over SSH (implicit connection check)
            conn = await self.ssh.ensure_connection()

            # Check preparation status
            prep_result = await conn.run(
                f"test -f {self.preparation_marker}", check=False
            )
            prepared = prep_result.exit_status == 0

            # Check available output files
            output_files = await self._get_output_files()

            # Check job running status
            job_running, job_info = await self._check_job_running()

            # Get job status analysis only if instance is prepared
            job_status = None
            if prepared:
                job_status = await self.get_instance_job_status()
                # Note: Crashed job detection is now handled by the health check system

            # Update success tracking
            current_time = time.time()
            self.last_success_time = current_time
            self.last_reachable_time = current_time  # Update last reachable time
            self.error_count = 0

            return {
                "success": True,
                "prepared": prepared,
                "output_files": output_files,
                "last_check": current_time,
                "last_reachable": self.last_reachable_time,
                "error_count": self.error_count,
                "api_state": self.api_state,
                "job_running": job_running,
                "job_file": job_info.get("file") if job_running else None,
                "job_start_time": job_info.get("start_time") if job_running else None,
                "job_duration_seconds": (
                    job_info.get("duration") if job_running else None
                ),
                "job_pid": job_info.get("pid") if job_running else None,
                "job_last_activity": (
                    job_info.get("last_activity") if job_running else None
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get status for {self.host_port}: {e}")
            self.error_count += 1
            current_time = time.time()
            return {
                "success": False,
                "prepared": False,
                "output_files": [],
                "last_check": current_time,
                "last_reachable": self.last_reachable_time,  # Keep last reachable time
                "error_count": self.error_count,
                "api_state": self.api_state,
                "job_running": False,
                "job_file": None,
                "job_start_time": None,
                "job_duration_seconds": None,
                "job_pid": None,
                "job_last_activity": None,
            }

    async def _get_output_files(self) -> list[str]:
        """Get list of output files on the instance."""
        try:
            conn = await self.ssh.ensure_connection()

            # First check if the repo directory exists
            check_dir = await conn.run(f"test -d {self.repo_dir}", check=False)
            if check_dir.exit_status != 0:
                # Directory doesn't exist yet (instance not prepared)
                return []

            result = await conn.run(
                f"find {self.repo_dir} -name '*.parquet' -type f", check=False
            )
            if result.exit_status == 0:
                if result.stdout is not None:
                    stdout_str = (
                        result.stdout.decode()
                        if isinstance(result.stdout, bytes)
                        else result.stdout
                    )
                    files = [
                        line.strip() for line in stdout_str.splitlines() if line.strip()
                    ]
                    return [Path(f).name for f in files]  # Return just filenames
                return []
            return []
        except Exception:
            return []

    async def _check_job_running(self) -> tuple[bool, dict[str, Any]]:
        """Check if a job is currently running on the instance."""
        try:
            conn = await self.ssh.ensure_connection()

            # First check if the repo directory exists
            check_dir = await conn.run(f"test -d {self.repo_dir}", check=False)
            if check_dir.exit_status != 0:
                # Directory doesn't exist yet (instance not prepared)
                return False, {}

            # Check for running garch jobs using ps aux
            result = await conn.run(
                "ps aux | grep 'scripts/run_garch_jobs.py' | grep -v grep", check=False
            )

            if (
                result.exit_status == 0
                and result.stdout is not None
                and result.stdout.strip()
            ):
                # Parse ps aux output
                # Example: root 20515 101 0.7 14081040 1053068 ? Rl 23:48 8:16 /venv/main/bin/python -u scripts/run_garch_jobs.py /root/density-engine/garch_training_job_63000_63099.csv --num_sim 10000000 --num_quantiles 512 --stride 1
                stdout_str = (
                    result.stdout.decode()
                    if isinstance(result.stdout, bytes)
                    else result.stdout
                )
                lines = stdout_str.strip().split("\n")
                for line in lines:
                    if "scripts/run_garch_jobs.py" in line:
                        parts = line.split()
                        if len(parts) >= 11:  # Ensure we have enough parts
                            pid = int(parts[1])

                            # Extract job file from command line arguments
                            job_file = None
                            for i, part in enumerate(parts):
                                part_str = str(part)
                                if (
                                    part_str.endswith(".csv")
                                    and "/root/density-engine/" in part_str
                                ):
                                    job_file = part_str.split("/")[
                                        -1
                                    ]  # Get just the filename
                                    break

                            if job_file:
                                # Get job start time and last activity
                                start_time = await self._get_job_start_time(job_file)
                                last_activity = await self._get_job_last_activity(
                                    job_file
                                )

                                return True, {
                                    "file": job_file,
                                    "pid": pid,
                                    "start_time": start_time,
                                    "duration": (
                                        time.time() - start_time if start_time else 0
                                    ),
                                    "last_activity": last_activity,
                                }

            return False, {}

        except Exception:
            return False, {}

    async def _get_job_start_time(self, job_file: str) -> float:
        """Get job start time from process start time."""
        try:
            conn = await self.ssh.ensure_connection()
            # Get process list and parse in Python
            result = await conn.run(
                f"ps aux | grep 'scripts/run_garch_jobs.py.*{job_file}' | grep -v grep",
                check=False,
            )
            if (
                result.exit_status == 0
                and result.stdout is not None
                and result.stdout.strip()
            ):
                stdout_str = (
                    result.stdout.decode()
                    if isinstance(result.stdout, bytes)
                    else result.stdout
                )
                lines = stdout_str.strip().split("\n")
                for line in lines:
                    if "scripts/run_garch_jobs.py" in line:
                        parts = line.split()
                        if len(parts) >= 11:
                            pid = str(parts[1])
                            # Get process start time from /proc/PID/stat
                            stat_result = await conn.run(
                                f"stat -c %Y /proc/{pid}", check=False
                            )
                            if (
                                stat_result.exit_status == 0
                                and stat_result.stdout is not None
                            ):
                                stdout_str = (
                                    stat_result.stdout.decode()
                                    if isinstance(stat_result.stdout, bytes)
                                    else stat_result.stdout
                                )
                                return float(stdout_str.strip())
        except Exception:
            pass
        return 0.0

    async def _get_job_last_activity(self, job_file: str) -> float:
        """Get job last activity time from log file."""
        try:
            conn = await self.ssh.ensure_connection()
            log_file = f"{self.repo_dir}/{job_file}.log"
            result = await conn.run(f"stat -c %Y {log_file}", check=False)
            if result.exit_status == 0 and result.stdout is not None:
                stdout_str = (
                    result.stdout.decode()
                    if isinstance(result.stdout, bytes)
                    else result.stdout
                )
                return float(stdout_str.strip())
        except Exception:
            pass
        return 0.0

    async def is_job_healthy(self, instance_status: dict[str, Any]) -> bool:
        """Check if job is healthy and making progress."""
        if not instance_status.get("job_running", False):
            return True

        try:
            job_file = instance_status.get("job_file")
            if not job_file:
                return False

            conn = await self.ssh.ensure_connection()

            # Check if process is still running by looking for it in ps aux
            result = await conn.run(
                f"ps aux | grep 'scripts/run_garch_jobs.py.*{job_file}' | grep -v grep",
                check=False,
            )
            if (
                result.exit_status != 0
                or result.stdout is None
                or not result.stdout.strip()
            ):
                # Process not found - check if job completed successfully
                logger.warning(
                    f"Job process not found for {job_file} on {self.host_port}"
                )

                # Check if parquet output file exists (job completed successfully)
                parquet_pattern = (
                    f"{self.repo_dir}/{job_file.split('.csv')[0]}_*_streaming_*.parquet"
                )
                result = await conn.run(f"find {parquet_pattern} -type f", check=False)
                if (
                    result.exit_status == 0
                    and result.stdout is not None
                    and result.stdout.strip()
                ):
                    logger.info(
                        f"Job {job_file} completed successfully on {self.host_port}"
                    )
                    return True  # Job completed successfully

                return False  # Process not found and no output file

            # Check for recent log activity using multiple methods
            log_file = f"{self.repo_dir}/{job_file}.log"

            # Method 1: Check if log file has been modified in the last minute
            result = await conn.run(f"find {log_file} -mmin -1 -type f", check=False)
            if result.exit_status != 0:
                logger.warning(
                    f"Log file not found or not recently modified for {job_file} on {self.host_port}"
                )
                return False

            # Method 2: Check if log file has reasonable size (not empty, not too small)
            result = await conn.run(f"wc -c < {log_file}", check=False)
            if result.exit_status == 0 and result.stdout is not None:
                try:
                    stdout_str = (
                        result.stdout.decode()
                        if isinstance(result.stdout, bytes)
                        else result.stdout
                    )
                    current_size = int(stdout_str.strip())
                    if current_size < 100:  # Log file too small, likely not working
                        logger.warning(
                            f"Log file too small ({current_size} bytes) for {job_file} on {self.host_port}"
                        )
                        return False
                except ValueError:
                    pass

            # Method 3: Check for recent log entries (last few lines)
            result = await conn.run(f"tail -n 5 {log_file}", check=False)
            if (
                result.exit_status == 0
                and result.stdout is not None
                and result.stdout.strip()
            ):
                # We have recent log content
                logger.debug(f"Job {job_file} appears healthy on {self.host_port}")
                return True

            logger.warning(
                f"No recent log content found for {job_file} on {self.host_port}"
            )
            return False

        except Exception as e:
            logger.error(f"Failed to check job health for {self.host_port}: {e}")
            return False

    async def handle_output_files(self, instance_status: dict[str, Any]) -> None:
        """Handle downloading output files from the instance."""
        output_files = instance_status.get("output_files", [])
        if not output_files:
            return

        try:
            sftp = await self.ssh.get_sftp()

            for file_name in output_files:
                remote_path = f"{self.repo_dir}/{file_name}"
                local_path = self.job_manager.output_dir / file_name

                # Skip if already downloaded
                if local_path.exists():
                    continue

                # Check if file is stable (not being written)
                if await self._is_file_stable(remote_path):
                    try:
                        await sftp.get(remote_path, str(local_path))
                        logger.info(f"✅ Downloaded output file: {file_name}")

                        # Update statistics counter
                        if self.cluster_manager:
                            self.cluster_manager.update_stat("outputs_downloaded")
                    except Exception as e:
                        logger.error(f"Failed to download {file_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to handle output files for {self.host_port}: {e}")

    async def _is_file_stable(self, remote_path: str) -> bool:
        """Check if a file is stable (not being written)."""
        try:
            conn = await self.ssh.ensure_connection()

            # Get file size
            result1 = await conn.run(f"stat -c %s {remote_path}", check=False)
            if result1.exit_status != 0:
                return False

            if result1.stdout is not None:
                stdout_str = (
                    result1.stdout.decode()
                    if isinstance(result1.stdout, bytes)
                    else result1.stdout
                )
                size1 = int(stdout_str.strip())
            else:
                return False

            # Wait a bit and check again
            await asyncio.sleep(2)

            result2 = await conn.run(f"stat -c %s {remote_path}", check=False)
            if result2.exit_status != 0:
                return False

            if result2.stdout is not None:
                stdout_str = (
                    result2.stdout.decode()
                    if isinstance(result2.stdout, bytes)
                    else result2.stdout
                )
                size2 = int(stdout_str.strip())
            else:
                return False

            return size1 == size2

        except Exception:
            return False

    async def handle_job_failure(self, instance_status: dict[str, Any]) -> None:
        """Handle failed or stuck jobs."""
        job_file = instance_status.get("job_file")
        if not job_file:
            return

        try:
            conn = await self.ssh.ensure_connection()

            # Find and kill the stuck/failed process
            result = await conn.run(
                f"ps aux | grep 'scripts/run_garch_jobs.py.*{job_file}' | grep -v grep | awk '{{print $2}}'",
                check=False,
            )
            if (
                result.exit_status == 0
                and result.stdout is not None
                and result.stdout.strip()
            ):
                stdout_str = (
                    result.stdout.decode()
                    if isinstance(result.stdout, bytes)
                    else result.stdout
                )
                pid = stdout_str.strip().split("\n")[0]
                # Kill the process
                await conn.run(f"kill -TERM {pid}", check=False)
                await asyncio.sleep(2)
                await conn.run(f"kill -KILL {pid}", check=False)
                logger.info(f"Killed process {pid} for job {job_file}")

            # Return job to queue
            self.job_manager.return_job_to_queue(job_file, "job failed")

            # ATOMICALLY clean up the failed job files from the instance
            await self._cleanup_failed_job(job_file)

            # Reset job state
            self.current_job = None
            self.job_start_time = 0.0
            self.job_last_activity = 0.0

            # Enhanced logging
            host_port = f"{self.host}:{self.port}"
            logger.warning(
                f"🔄 JOB RETURNED: {job_file} → queue (failed on {host_port})"
            )

            # Update statistics counter
            if self.cluster_manager:
                self.cluster_manager.update_stat("jobs_returned")

        except Exception as e:
            logger.error(f"Failed to handle job failure for {self.host_port}: {e}")

    async def _is_job_completed(self, instance_status: dict[str, Any]) -> bool:
        """Check if job completed successfully (process stopped but output exists)."""
        job_file = instance_status.get("job_file")
        if not job_file:
            return False

        try:
            conn = await self.ssh.ensure_connection()

            # Check if parquet output file exists
            parquet_pattern = (
                f"{self.repo_dir}/{job_file.split('.csv')[0]}_*_streaming_*.parquet"
            )
            result = await conn.run(f"find {parquet_pattern} -type f", check=False)
            if (
                result.exit_status == 0
                and result.stdout is not None
                and result.stdout.strip()
            ):
                return True

            return False
        except Exception:
            return False

    async def _is_job_still_running(self, job_name: str) -> bool:
        """Check if a specific job is still running."""
        try:
            conn = await self.ssh.ensure_connection()

            # Check if process is running for this specific job
            result = await conn.run(
                f"ps aux | grep 'scripts/run_garch_jobs.py.*{job_name}' | grep -v grep",
                check=False,
            )
            if (
                result.exit_status == 0
                and result.stdout is not None
                and result.stdout.strip()
            ):
                return True

            return False
        except Exception:
            return False

    async def _handle_job_completion(self, instance_status: dict[str, Any]) -> None:
        """Handle successful job completion."""
        job_file = instance_status.get("job_file")
        if not job_file:
            return

        try:
            # Mark job as completed in job manager
            self.job_manager.mark_job_completed(job_file)

            # Reset job state
            self.current_job = None
            self.job_start_time = 0.0
            self.job_last_activity = 0.0

            # Enhanced logging
            host_port = f"{self.host}:{self.port}"
            logger.info(f"✅ JOB COMPLETED: {job_file} on {host_port}")

        except Exception as e:
            logger.error(f"Failed to handle job completion for {self.host_port}: {e}")

    async def ensure_prepared(self) -> bool:
        """Ensure instance is prepared for jobs."""
        if self.preparation_checked and self.is_prepared:
            return True

        try:
            conn = await self.ssh.ensure_connection()

            # Check if already prepared
            result = await conn.run(f"test -f {self.preparation_marker}", check=False)
            if result.exit_status == 0:
                self.is_prepared = True
                self.preparation_checked = True
                return True

            # Run preparation
            logger.info(f"Preparing instance {self.host_port}")
            success = await self._run_preparation(conn)

            if success:
                self.is_prepared = True
                self.preparation_checked = True
                logger.info(f"Instance {self.host_port} prepared successfully")
            else:
                logger.error(f"Failed to prepare instance {self.host_port}")

            return success

        except Exception as e:
            logger.error(f"Failed to ensure preparation for {self.host_port}: {e}")
            return False

    async def _run_preparation(self, conn: Any) -> bool:
        """Run the preparation process on the instance."""
        try:
            logger.info(f"Starting preparation for {self.host_port}")

            # Step 1: Verify Python environment
            if not await self._verify_python_environment(conn):
                return False

            # Step 2: Check PyTorch availability
            if not await self._check_pytorch(conn):
                return False

            # Step 3: Setup repository
            if not await self._setup_repository(conn):
                return False

            # Step 4: Install required Python packages
            if not await self._install_python_packages(conn):
                return False

            # Step 5: Run smoke test
            if not await self._run_smoke_test(conn):
                return False

            # Step 6: Create preparation marker
            result = await conn.run(f"touch {self.preparation_marker}", check=False)
            if result.exit_status != 0:
                logger.error(
                    f"Failed to create preparation marker for {self.host_port}"
                )
                return False

            logger.info(f"✅ Preparation completed successfully for {self.host_port}")
            return True

        except Exception as e:
            logger.error(f"Preparation failed for {self.host_port}: {e}")
            return False

    async def get_instance_job_status(self) -> dict[str, Any]:
        """Analyze job files on the instance and return status information."""
        try:
            conn = await self.ssh.ensure_connection()

            # Get list of all files in the repo directory
            result = await conn.run(f"ls -la {self.repo_dir}/", check=False)
            if result.exit_status != 0:
                logger.error(
                    f"Failed to list files in {self.repo_dir} for {self.host_port}"
                )
                return {"success": False, "error": "Failed to list files"}

            # Parse file listing
            if result.stdout is not None:
                stdout_str = (
                    result.stdout.decode()
                    if isinstance(result.stdout, bytes)
                    else result.stdout
                )
                files = self._parse_file_listing(stdout_str)
            else:
                files = []

            # Analyze job files
            job_analysis = self._analyze_job_files(files)

            return {
                "success": True,
                "unprocessed_jobs": job_analysis["unprocessed_jobs"],
                "completed_jobs": job_analysis["completed_jobs"],
                "parquet_files_to_download": job_analysis["parquet_files_to_download"],
                "log_files": job_analysis["log_files"],
                "total_files": len(files),
            }

        except Exception as e:
            logger.error(f"Failed to get job status for {self.host_port}: {e}")
            return {"success": False, "error": str(e)}

    def _parse_file_listing(self, ls_output: str) -> list[dict[str, Any]]:
        """Parse ls -la output into structured file information."""
        files = []
        for line in ls_output.strip().split("\n"):
            if not line.strip() or line.startswith("total"):
                continue

            parts = line.split()
            if len(parts) >= 9:
                # Extract file information
                permissions = parts[0]
                size = int(parts[4]) if parts[4].isdigit() else 0
                date_parts = parts[5:8]  # Month Day Time
                filename = " ".join(parts[8:])  # Handle filenames with spaces

                files.append(
                    {
                        "name": filename,
                        "size": size,
                        "permissions": permissions,
                        "date": " ".join(date_parts),
                        "is_csv": filename.endswith(".csv"),
                        "is_log": filename.endswith(".csv.log"),
                        "is_parquet": filename.endswith(".parquet"),
                        "is_pid": filename.endswith(".pid"),
                    }
                )

        return files

    def _analyze_job_files(self, files: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze job files to identify unprocessed jobs and files to download."""
        # Group files by job name (everything before .csv)
        job_groups: dict[str, dict[str, Any]] = {}

        for file_info in files:
            filename = file_info["name"]

            if file_info["is_csv"] and not file_info["is_log"]:
                # This is a job file (CSV)
                job_name = filename[:-4]  # Remove .csv extension
                if job_name not in job_groups:
                    job_groups[job_name] = {
                        "csv": None,
                        "log": None,
                        "parquet": [],
                        "pid": None,
                    }
                job_groups[job_name]["csv"] = file_info

            elif file_info["is_log"]:
                # This is a log file
                job_name = filename[:-8]  # Remove .csv.log extension
                if job_name not in job_groups:
                    job_groups[job_name] = {
                        "csv": None,
                        "log": None,
                        "parquet": [],
                        "pid": None,
                    }
                job_groups[job_name]["log"] = file_info

            elif file_info["is_parquet"]:
                # This is a parquet file - extract job name from filename
                # Format: jobname_0_streaming_1_10000000_512.parquet
                job_name = filename.split("_0_streaming_")[0]
                if job_name not in job_groups:
                    job_groups[job_name] = {
                        "csv": None,
                        "log": None,
                        "parquet": [],
                        "pid": None,
                    }
                job_groups[job_name]["parquet"].append(file_info)

            elif file_info["is_pid"]:
                # This is a PID file
                job_name = filename[:-4]  # Remove .pid extension
                if job_name not in job_groups:
                    job_groups[job_name] = {
                        "csv": None,
                        "log": None,
                        "parquet": [],
                        "pid": None,
                    }
                job_groups[job_name]["pid"] = file_info

        # Categorize jobs
        unprocessed_jobs = []
        completed_jobs = []
        parquet_files_to_download = []
        log_files = []

        for job_name, job_files in job_groups.items():
            has_csv = job_files["csv"] is not None
            has_parquet = len(job_files["parquet"]) > 0
            has_log = job_files["log"] is not None

            if has_csv and not has_parquet:
                # Unprocessed job - has CSV but no parquet output
                # This could be a new job or a running job - the health check will determine if it's actually running
                unprocessed_jobs.append(
                    {
                        "job_name": job_name,
                        "csv_file": job_files["csv"]["name"],
                        "csv_size": job_files["csv"]["size"],
                        "has_log": has_log,
                        "log_file": job_files["log"]["name"] if has_log else None,
                    }
                )

            elif has_csv and has_parquet:
                # Job with parquet files - but we need to check if it's actually completed
                # We'll mark it as completed for now, but the download logic will verify completion
                completed_jobs.append(
                    {
                        "job_name": job_name,
                        "csv_file": job_files["csv"]["name"],
                        "parquet_files": [p["name"] for p in job_files["parquet"]],
                        "has_log": has_log,
                        "log_file": job_files["log"]["name"] if has_log else None,
                    }
                )

                # Add parquet files to download list (but they'll be verified for completion before download)
                for parquet_file in job_files["parquet"]:
                    parquet_files_to_download.append(
                        {
                            "filename": parquet_file["name"],
                            "size": parquet_file["size"],
                            "job_name": job_name,
                        }
                    )

            # Collect log files for health checking
            if has_log:
                log_files.append(
                    {
                        "job_name": job_name,
                        "log_file": job_files["log"]["name"],
                        "size": job_files["log"]["size"],
                    }
                )

        return {
            "unprocessed_jobs": unprocessed_jobs,
            "completed_jobs": completed_jobs,
            "parquet_files_to_download": parquet_files_to_download,
            "log_files": log_files,
        }

    async def ensure_available_jobfile_on_instance(self) -> bool:
        """Ensure there's an unprocessed job file on the instance."""
        try:
            # Check if instance is prepared first
            conn = await self.ssh.ensure_connection()
            prep_result = await conn.run(
                f"test -f {self.preparation_marker}", check=False
            )
            if prep_result.exit_status != 0:
                logger.debug(
                    f"Instance not prepared yet, skipping job file check for {self.host_port}"
                )
                return False

            # First check if there's already a running job
            job_running, job_info = await self._check_job_running()
            if job_running:
                # There's already a job running, don't start another one
                self.current_job = job_info.get("file")
                logger.debug(
                    f"Job already running on {self.host_port}: {self.current_job}"
                )
                return True

            # Get job status to find unprocessed jobs
            job_status = await self.get_instance_job_status()
            if not job_status["success"]:
                logger.error(f"Failed to get job status for {self.host_port}")
                return False

            unprocessed_jobs = job_status["unprocessed_jobs"]

            if unprocessed_jobs:
                logger.info(
                    f"Found {len(unprocessed_jobs)} unprocessed job(s) on {self.host_port}"
                )
                for job in unprocessed_jobs:
                    logger.info(f"  - {job['job_name']} (CSV: {job['csv_file']})")

                # Pick the first unprocessed job - it's already on the instance
                selected_job = unprocessed_jobs[0]
                job_name = selected_job["job_name"]

                # Set current job
                self.current_job = job_name

                # Check if this job exists in our local input queue
                local_job_file = self.job_manager.input_queue_dir / job_name
                if local_job_file.exists():
                    # Move job from pending to running in the job manager
                    self.job_manager.mark_job_running(job_name)
                    logger.info(f"📤 JOB ASSIGNED: {job_name} → {self.host_port}")

                    # Update statistics counter
                    if self.cluster_manager:
                        self.cluster_manager.update_stat("jobs_assigned")
                else:
                    # Job exists on remote but not in local queue - just start it
                    logger.info(
                        f"📤 JOB STARTING: {job_name} (already on instance) → {self.host_port}"
                    )

                return True

            # No unprocessed jobs found, try to assign one
            job_file = self.job_manager.assign_job()
            if not job_file:
                logger.info(f"No jobs available to assign to {self.host_port}")
                return False

            # Set current job
            self.current_job = job_file.name

            # Upload job file to instance
            sftp = await self.ssh.get_sftp()
            remote_path = f"{self.repo_dir}/{job_file.name}"
            await sftp.put(str(job_file), remote_path)

            # Enhanced logging
            host_port = f"{self.host}:{self.port}"
            logger.info(f"📤 JOB ASSIGNED: {job_file.name} → {host_port}")

            # Update statistics counter
            if self.cluster_manager:
                self.cluster_manager.update_stat("jobs_assigned")

            return True

        except Exception as e:
            logger.error(f"Failed to ensure job file on {self.host_port}: {e}")
            return False

    async def _cleanup_failed_job(self, job_name: str) -> None:
        """Clean up files from a failed job (both crashed and killed jobs)."""
        try:
            conn = await self.ssh.ensure_connection()

            # Remove CSV, log, and PID files for the failed job
            files_to_remove = [
                f"{self.repo_dir}/{job_name}.csv",
                f"{self.repo_dir}/{job_name}.csv.log",
                f"{self.repo_dir}/{job_name}.pid",
            ]

            for file_path in files_to_remove:
                await conn.run(f"rm -f {file_path}", check=False)

            logger.info(
                f"Cleaned up failed job files for {job_name} on {self.host_port}"
            )

        except Exception as e:
            logger.error(
                f"Failed to cleanup failed job {job_name} on {self.host_port}: {e}"
            )

    async def _cleanup_crashed_job(self, job_name: str) -> None:
        """Clean up files from a crashed job (alias for _cleanup_failed_job)."""
        await self._cleanup_failed_job(job_name)

    async def download_output_files(self, local_output_dir: str) -> bool:
        """Download parquet output files from completed jobs only."""
        try:
            # Check if instance is prepared first
            conn = await self.ssh.ensure_connection()
            prep_result = await conn.run(
                f"test -f {self.preparation_marker}", check=False
            )
            if prep_result.exit_status != 0:
                logger.debug(
                    f"Instance not prepared yet, skipping output download for {self.host_port}"
                )
                return True

            # Get job status to find parquet files to download
            job_status = await self.get_instance_job_status()
            if not job_status["success"]:
                logger.error(f"Failed to get job status for {self.host_port}")
                return False

            parquet_files = job_status["parquet_files_to_download"]
            if not parquet_files:
                logger.info(f"No parquet files to download from {self.host_port}")
                return True

            # Filter to only completed jobs (process not running)
            completed_parquet_files = []
            for parquet_info in parquet_files:
                job_name = parquet_info["job_name"]

                # Check if this specific job is still running
                is_job_still_running = await self._is_job_still_running(job_name)
                if is_job_still_running:
                    # This job is currently running, skip its parquet files
                    logger.debug(
                        f"Skipping {parquet_info['filename']} - job {job_name} is still running"
                    )
                    continue

                # Job is not running, safe to download
                completed_parquet_files.append(parquet_info)

            if not completed_parquet_files:
                logger.info(
                    f"No completed parquet files to download from {self.host_port}"
                )
                return True

            # Ensure local output directory exists
            Path(local_output_dir).mkdir(parents=True, exist_ok=True)

            sftp = await self.ssh.get_sftp()
            downloaded_count = 0
            skipped_count = 0

            for parquet_info in completed_parquet_files:
                remote_file = f"{self.repo_dir}/{parquet_info['filename']}"
                local_file = Path(local_output_dir) / parquet_info["filename"]

                # Check if file already exists locally
                if local_file.exists():
                    logger.info(
                        f"Skipping {parquet_info['filename']} - already exists locally"
                    )
                    skipped_count += 1
                    continue

                try:
                    # Download the file
                    await sftp.get(remote_file, str(local_file))
                    host_port = f"{self.host}:{self.port}"
                    logger.info(
                        f"📥 OUTPUT DOWNLOADED: {parquet_info['filename']} from {host_port}"
                    )
                    downloaded_count += 1

                    # Update statistics
                    if self.cluster_manager:
                        self.cluster_manager.update_stat("outputs_downloaded")

                except Exception as e:
                    logger.error(
                        f"Failed to download {parquet_info['filename']} from {self.host_port}: {e}"
                    )

            logger.info(
                f"Download summary for {self.host_port}: {downloaded_count} downloaded, {skipped_count} skipped"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to download output files from {self.host_port}: {e}")
            return False

    async def start_job_execution(self) -> bool:
        """Start job execution on the instance."""
        if not self.current_job:
            return False

        try:
            conn = await self.ssh.ensure_connection()

            # Execute job command using bash -lc to activate environment (like controller.py)
            # Handle job name with or without .csv extension
            if self.current_job.endswith(".csv"):
                job_name = self.current_job[:-4]  # Remove .csv extension
                job_path = f"{self.repo_dir}/{self.current_job}"
                log_path = f"{self.repo_dir}/{job_name}.csv.log"
            else:
                job_name = self.current_job
                job_path = f"{self.repo_dir}/{self.current_job}.csv"
                log_path = f"{self.repo_dir}/{self.current_job}.csv.log"
            cmd = (
                f"nohup bash -lc '"
                f"cd {self.repo_dir} && "
                f"PYTHONPATH={self.repo_dir} {self.venv_python} -u scripts/run_garch_jobs.py "
                f"{job_path} --num_sim 10000000 --num_quantiles 512 --stride 1 "
                f"> {log_path} 2>&1 &"
                f"' >/dev/null 2>&1 &"
            )

            result = await conn.run(cmd, check=False)
            if result.exit_status == 0:
                logger.info(f"Started job {self.current_job} on {self.host_port}")
                return True
            else:
                logger.error(
                    f"Failed to start job {self.current_job} on {self.host_port}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to start job execution on {self.host_port}: {e}")
            return False

    async def handle(self) -> None:
        """Main instance handle loop - adaptive timing with 10-second minimum interval."""
        try:
            # 1. Get instance status (SSH, preparation, job status, etc.)
            instance_status = await self.get_instance_status()

            if not instance_status["success"]:
                # Status collection failed - wait with exponential backoff
                await self._wait_with_exponential_backoff()
                return

            # Always handle output files first (regardless of job status)
            await self.handle_output_files(instance_status)

            if instance_status["job_running"]:
                # Job is running - check if it's healthy
                if not await self.is_job_healthy(instance_status):
                    # Check if job completed successfully (process stopped but output exists)
                    if await self._is_job_completed(instance_status):
                        await self._handle_job_completion(instance_status)
                    else:
                        # Job failed or stuck - handle the failure
                        await self.handle_job_failure(instance_status)
            else:
                # No job running - try to start a new one
                await self.try_start_new_job()

            # 3. Update next check time (adaptive timing)
            # If loop took < 10 seconds, wait for remainder. If > 10 seconds, start immediately
            await self._update_next_check()

        except Exception as e:
            # Status collection failed - wait with exponential backoff
            logger.error(f"Handle loop error for {self.host_port}: {e}")
            await self._wait_with_exponential_backoff()

    async def try_start_new_job(self) -> None:
        """Try to start a new job when none is running."""
        # 1. Ensure instance is prepared
        if not await self.ensure_prepared():
            return

        # 2. Ensure available job file on instance
        if not await self.ensure_available_jobfile_on_instance():
            return  # No jobs available or upload failed

        # 3. Start job execution
        await self.start_job_execution()

    async def _wait_with_exponential_backoff(self) -> None:
        """Wait with exponential backoff for failed operations."""
        backoff_times = [10, 30, 60]  # seconds
        backoff_index = min(self.error_count, len(backoff_times) - 1)
        wait_time = backoff_times[backoff_index]

        logger.info(f"Waiting {wait_time}s before retry for {self.host_port}")
        await asyncio.sleep(wait_time)

    async def _update_next_check(self) -> None:
        """Update next check time with adaptive timing."""
        now = time.time()
        loop_duration = now - self.last_check_time

        if loop_duration < 10:
            # Wait for remainder of 10-second minimum interval
            wait_time = 10 - loop_duration
            await asyncio.sleep(wait_time)

        self.last_check_time = time.time()
        self.next_check_time = self.last_check_time + 10

    async def close(self) -> None:
        """Close instance manager and cleanup resources."""
        if hasattr(self, "_closed") and self._closed:
            logger.debug(f"Instance manager already closed for {self.host_port}")
            return

        self._closed = True
        await self.ssh.close()
        logger.info(f"Closed instance manager for {self.host_port}")

    # ---------- Preparation Subfunctions ----------

    async def _verify_python_environment(self, conn: Any) -> bool:
        """Verify Python environment is available."""
        try:
            result = await conn.run(f"test -x {self.venv_python}", check=False)
            if result.exit_status != 0:
                logger.error(
                    f"Python not found at {self.venv_python} on {self.host_port}"
                )
                return False

            logger.info(f"Python environment verified for {self.host_port}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to verify Python environment for {self.host_port}: {e}"
            )
            return False

    async def _setup_repository(self, conn: Any) -> bool:
        """Setup the repository on the instance."""
        try:
            # Check if directory exists
            result = await conn.run(f"test -d {self.repo_dir}", check=False)
            dir_exists = result.exit_status == 0

            # Check if it's a git repo
            result = await conn.run(f"test -d {self.repo_dir}/.git", check=False)
            is_repo = result.exit_status == 0

            if not dir_exists:
                logger.info(f"Cloning repository for {self.host_port}")
                result = await conn.run(
                    f"git clone -q {self.github_repo} {self.repo_dir}", check=False
                )
                if result.exit_status != 0:
                    logger.error(f"Failed to clone repository for {self.host_port}")
                    return False
            elif not is_repo:
                logger.warning(
                    f"Directory exists but not a git repo for {self.host_port}, replacing"
                )
                timestamp = int(time.time())
                backup = f"{self.repo_dir}.backup.{timestamp}"
                await conn.run(f"mv {self.repo_dir} {backup}", check=False)
                result = await conn.run(
                    f"git clone -q {self.github_repo} {self.repo_dir}", check=False
                )
                if result.exit_status != 0:
                    logger.error(
                        f"Failed to clone repository after backup for {self.host_port}"
                    )
                    return False
            else:
                logger.info(f"Updating repository for {self.host_port}")
                result = await conn.run(
                    f"git -C {self.repo_dir} fetch -q --all && "
                    f"git -C {self.repo_dir} reset -q --hard origin/main",
                    check=False,
                )
                if result.exit_status != 0:
                    logger.warning(
                        f"Failed to update repository for {self.host_port}, trying fresh clone"
                    )
                    timestamp = int(time.time())
                    backup = f"{self.repo_dir}.backup.{timestamp}"
                    await conn.run(f"mv {self.repo_dir} {backup}", check=False)
                    result = await conn.run(
                        f"git clone -q {self.github_repo} {self.repo_dir}", check=False
                    )
                    if result.exit_status != 0:
                        logger.error(
                            f"Failed to clone repository after update failure for {self.host_port}"
                        )
                        return False

            # Verify required script exists
            result = await conn.run(
                f"test -f {self.repo_dir}/scripts/run_garch_jobs.py", check=False
            )
            if result.exit_status != 0:
                logger.error(f"Required script not found for {self.host_port}")
                return False

            logger.info(f"Repository setup completed for {self.host_port}")
            return True

        except Exception as e:
            logger.error(f"Failed to setup repository for {self.host_port}: {e}")
            return False

    async def _check_pytorch(self, conn: Any) -> bool:
        """Check if PyTorch is available (required, not installed)."""
        try:
            result = await conn.run(
                f"{self.venv_python} -c 'import torch; print(torch.__version__)'",
                check=False,
            )
            if result.exit_status != 0:
                logger.error(
                    f"PyTorch not available for {self.host_port}. Provision an image with PyTorch preinstalled."
                )
                return False

            logger.info(f"PyTorch verified for {self.host_port}")
            return True

        except Exception as e:
            logger.error(f"Failed to check PyTorch for {self.host_port}: {e}")
            return False

    async def _install_python_packages(self, conn: Any) -> bool:
        """Install required Python packages."""
        try:
            # Ensure pip is available
            result = await conn.run(f"{self.venv_python} -m pip --version", check=False)
            if result.exit_status != 0:
                logger.info(f"Installing pip for {self.host_port}")
                result = await conn.run(
                    f"{self.venv_python} -m ensurepip --upgrade", check=False
                )
                if result.exit_status != 0:
                    logger.error(f"Failed to install pip for {self.host_port}")
                    return False

            # Check which packages are missing
            missing_packages = await self._get_missing_packages(conn)
            if not missing_packages:
                logger.info(
                    f"All required packages already installed for {self.host_port}"
                )
                return True

            # Install missing packages
            logger.info(
                f"Installing missing packages for {self.host_port}: {missing_packages}"
            )
            packages_str = " ".join(missing_packages)
            result = await conn.run(
                f"bash -c 'export PIP_DISABLE_PIP_VERSION_CHECK=1; "
                f"export PIP_NO_CACHE_DIR=1; "
                f"{self.venv_python} -m pip install --no-input --quiet {packages_str}'",
                check=False,
            )
            if result.exit_status != 0:
                logger.error(f"Failed to install packages for {self.host_port}")
                logger.error(f"Installation stdout: {result.stdout}")
                logger.error(f"Installation stderr: {result.stderr}")
                return False

            # No need to install density_engine package - it's already available from the git clone
            logger.info(f"Repository code is ready for {self.host_port}")

            logger.info(f"Package installation completed for {self.host_port}")

            # Verify all packages are working
            if not await self._verify_packages(conn):
                return False

            logger.info(f"Python packages installed successfully for {self.host_port}")
            return True

        except Exception as e:
            logger.error(f"Failed to install Python packages for {self.host_port}: {e}")
            return False

    async def _get_missing_packages(self, conn: Any) -> list[str]:
        """Get list of missing required packages."""
        try:
            code = f"""
import importlib.util
required = {self.required_packages!r}
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
print(','.join(missing))
"""
            result = await conn.run(f"{self.venv_python} -c '{code}'", check=False)
            if result.exit_status != 0:
                return self.required_packages  # Assume all missing if check fails

            missing_str = result.stdout.strip()
            if not missing_str:
                return []

            return [pkg.strip() for pkg in missing_str.split(",") if pkg.strip()]

        except Exception:
            return self.required_packages  # Assume all missing if check fails

    async def _verify_packages(self, conn: Any) -> bool:
        """Verify all required packages can be imported."""
        try:
            # Use the same approach as controller.py - heredoc syntax to avoid f-string escaping issues
            code = (
                f"{self.venv_python} - <<'PY'\n"
                "import importlib, json\n"
                f"mods={self.required_packages!r}\n"
                "ok=[]; bad=[]\n"
                "for m in mods:\n"
                "    try:\n"
                "        importlib.import_module(m); ok.append(m)\n"
                "    except Exception as e:\n"
                "        bad.append([m, str(e)])\n"
                'print(json.dumps({"ok": ok, "bad": bad}))\n'
                "PY"
            )
            logger.debug(f"Running package verification code: {code}")
            result = await conn.run(code, check=False)
            if result.exit_status != 0:
                logger.error(f"Package verification failed for {self.host_port}")
                logger.error(
                    f"Python command failed with exit code {result.exit_status}"
                )
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False

            import json

            try:
                data = json.loads(result.stdout.strip())
                bad_packages = data.get("bad", [])
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse package verification JSON for {self.host_port}: {e}"
                )
                logger.error(f"Raw output: {result.stdout}")
                return False

            if bad_packages:
                problems = "; ".join([f"{pkg}: {error}" for pkg, error in bad_packages])
                logger.error(
                    f"Package import failures for {self.host_port}: {problems}"
                )
                return False

            logger.info(f"All packages verified for {self.host_port}")
            return True

        except Exception as e:
            logger.error(f"Failed to verify packages for {self.host_port}: {e}")
            return False

    async def _run_smoke_test(self, conn: Any) -> bool:
        """Run a smoke test to verify everything works."""
        try:
            # Create test CSV using heredoc (like controller.py approach)
            test_csv = f"{self.repo_dir}/__smoketest__.csv"
            test_content = (
                "eta,lam,var0,alpha,gamma,beta,p\n"
                "12.840664961397676,-0.5875573,3.6697478,0.006930725835263729,0.5710068283794048,0.620523473918438,0.98825073\n"
            )

            # Write test CSV using heredoc
            write_cmd = f"""
cat > {test_csv} << 'EOF'
{test_content}EOF
"""
            result = await conn.run(write_cmd, check=False)
            if result.exit_status != 0:
                logger.error(f"Failed to write test CSV for {self.host_port}")
                return False

            # Clean up any previous test files
            await conn.run(
                f"rm -f {self.repo_dir}/__smoketest__*.parquet {self.repo_dir}/__smoketest__.csv.log {self.repo_dir}/__smoketest__*.pid",
                check=False,
            )

            # Run smoke test with PYTHONPATH set to include the repo directory
            cmd = (
                f"cd {self.repo_dir} && "
                f"PYTHONPATH={self.repo_dir} {self.venv_python} -u scripts/run_garch_jobs.py "
                f"{test_csv} --num_sim 1000 --num_quantiles 512 --stride 1"
            )

            result = await conn.run(cmd, check=False, timeout=180)

            # Check if parquet file was created
            parquet_files = await self._get_parquet_files(conn, "__smoketest__")

            # Cleanup
            await conn.run(f"rm -f {test_csv}", check=False)
            for parquet_file in parquet_files:
                await conn.run(f"rm -f {self.repo_dir}/{parquet_file}", check=False)

            if result.exit_status != 0 or not parquet_files:
                logger.error(f"Smoke test failed for {self.host_port}")
                logger.error(f"Smoke test exit code: {result.exit_status}")
                logger.error(f"Smoke test stdout: {result.stdout}")
                logger.error(f"Smoke test stderr: {result.stderr}")
                logger.error(f"Parquet files found: {parquet_files}")
                return False

            logger.info(f"Smoke test passed for {self.host_port}")
            return True

        except Exception as e:
            logger.error(f"Smoke test failed for {self.host_port}: {e}")
            return False

    async def _get_parquet_files(self, conn: Any, prefix: str) -> list[str]:
        """Get list of parquet files with given prefix."""
        try:
            result = await conn.run(
                f"find {self.repo_dir} -name '{prefix}*.parquet' -type f", check=False
            )
            if result.exit_status == 0:
                files = [
                    line.strip() for line in result.stdout.splitlines() if line.strip()
                ]
                return [Path(f).name for f in files]
            return []
        except Exception:
            return []
