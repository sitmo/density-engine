"""
Result handling for the vast.ai automation system.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.ssh import SSHClient, create_ssh_connection, download_file, execute_command
from ..instances.discovery import InstanceInfo
from ..utils.exceptions import FileOperationError
from ..utils.logging import get_logger, log_function_call

logger = get_logger(__name__)


@dataclass
class ResultData:
    """Result data structure."""

    file_path: Path
    file_size: int
    creation_time: str
    content_type: str


@dataclass
class ProcessedResults:
    """Processed results."""

    job_file: str
    result_files: list[Path]
    total_size: int
    success: bool
    error_message: str | None = None


@log_function_call
def find_result_files(instance: InstanceInfo, pattern: str = "*.parquet") -> list[str]:
    """Find result files on an instance."""
    try:
        logger.debug(
            f"Finding result files with pattern '{pattern}' on instance {instance.contract_id}"
        )

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Find files in the density-engine directory
            find_cmd = f"find /root/density-engine -name '{pattern}' -type f"
            result = execute_command(ssh_client, find_cmd, timeout=30)

            if result.success and result.stdout.strip():
                files = [
                    line.strip()
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                ]
                logger.info(f"Found {len(files)} result files matching '{pattern}'")
                return files
            else:
                logger.info(f"No result files found matching '{pattern}'")
                return []

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to find result files: {e}")
        return []


@log_function_call
def download_result_file(
    instance: InstanceInfo, remote_path: str, local_path: Path
) -> bool:
    """Download a result file from an instance."""
    try:
        logger.info(f"Downloading result file {remote_path} to {local_path}")

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Ensure local directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download the file
            success = download_file(ssh_client, remote_path, str(local_path))

            if success:
                logger.info(f"✅ Result file downloaded successfully: {local_path}")
                return True
            else:
                logger.error(f"❌ Failed to download result file: {remote_path}")
                return False

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to download result file: {e}")
        return False


@log_function_call
def validate_result_file(file_path: Path) -> bool:
    """Validate a result file."""
    try:
        logger.debug(f"Validating result file: {file_path}")

        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Result file does not exist: {file_path}")
            return False

        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            logger.warning(f"Result file is empty: {file_path}")
            return False

        # Check file extension
        if file_path.suffix not in [".parquet", ".csv", ".json"]:
            logger.warning(f"Result file has unexpected extension: {file_path}")
            return False

        logger.debug(f"Result file validation passed: {file_path} ({file_size} bytes)")
        return True

    except Exception as e:
        logger.error(f"Failed to validate result file: {e}")
        return False


@log_function_call
def parse_result_file(file_path: Path) -> ResultData:
    """Parse a result file and extract metadata."""
    try:
        logger.debug(f"Parsing result file: {file_path}")

        if not file_path.exists():
            raise FileOperationError(f"Result file does not exist: {file_path}")

        # Get file metadata
        stat = file_path.stat()
        file_size = stat.st_size
        creation_time = str(stat.st_ctime)
        content_type = file_path.suffix

        result_data = ResultData(
            file_path=file_path,
            file_size=file_size,
            creation_time=creation_time,
            content_type=content_type,
        )

        logger.debug(f"Parsed result file: {file_size} bytes, type: {content_type}")
        return result_data

    except Exception as e:
        logger.error(f"Failed to parse result file: {e}")
        raise FileOperationError(f"Failed to parse result file: {e}")


@log_function_call
def collect_job_results(instance: InstanceInfo, job_file: str) -> list[Path]:
    """Collect all result files for a job."""
    try:
        logger.info(
            f"Collecting results for job {job_file} on instance {instance.contract_id}"
        )

        # Find result files
        result_files = find_result_files(instance, "*.parquet")

        if not result_files:
            logger.warning(f"No result files found for job {job_file}")
            return []

        # Create local results directory
        results_dir = Path("results") / job_file.replace(".csv", "")
        results_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = []

        # Download each result file
        for remote_file in result_files:
            filename = os.path.basename(remote_file)
            local_file = results_dir / filename

            if download_result_file(instance, remote_file, local_file):
                if validate_result_file(local_file):
                    downloaded_files.append(local_file)
                    logger.info(f"✅ Collected result file: {local_file}")
                else:
                    logger.warning(
                        f"❌ Downloaded file failed validation: {local_file}"
                    )
                    local_file.unlink()  # Remove invalid file
            else:
                logger.error(f"❌ Failed to download result file: {remote_file}")

        logger.info(
            f"✅ Collected {len(downloaded_files)} result files for job {job_file}"
        )
        return downloaded_files

    except Exception as e:
        logger.error(f"Failed to collect job results: {e}")
        return []


@log_function_call
def process_job_results(job_file: str, result_files: list[Path]) -> ProcessedResults:
    """Process job results."""
    try:
        logger.info(f"Processing results for job {job_file}")

        if not result_files:
            return ProcessedResults(
                job_file=job_file,
                result_files=[],
                total_size=0,
                success=False,
                error_message="No result files to process",
            )

        # Calculate total size
        total_size = sum(file_path.stat().st_size for file_path in result_files)

        # Validate all files
        valid_files = []
        for file_path in result_files:
            if validate_result_file(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"Invalid result file: {file_path}")

        success = len(valid_files) > 0

        processed_results = ProcessedResults(
            job_file=job_file,
            result_files=valid_files,
            total_size=total_size,
            success=success,
            error_message=None if success else "No valid result files found",
        )

        logger.info(
            f"✅ Processed {len(valid_files)} valid result files ({total_size} bytes)"
        )
        return processed_results

    except Exception as e:
        logger.error(f"Failed to process job results: {e}")
        return ProcessedResults(
            job_file=job_file,
            result_files=[],
            total_size=0,
            success=False,
            error_message=str(e),
        )


@log_function_call
def archive_completed_job(job_file: str, results: ProcessedResults) -> bool:
    """Archive a completed job and its results."""
    try:
        logger.info(f"Archiving completed job {job_file}")

        # Create archive directory
        archive_dir = Path("archives") / job_file.replace(".csv", "")
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Move job file to archive
        from ..core.files import move_file

        job_source = Path("jobs") / "completed" / job_file
        job_dest = archive_dir / job_file

        if job_source.exists():
            move_file(job_source, job_dest)
            logger.info(f"✅ Archived job file: {job_file}")

        # Move result files to archive
        for result_file in results.result_files:
            archive_result = archive_dir / result_file.name
            move_file(result_file, archive_result)
            logger.info(f"✅ Archived result file: {result_file.name}")

        # Create summary file
        summary_file = archive_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Job: {job_file}\n")
            f.write(f"Result files: {len(results.result_files)}\n")
            f.write(f"Total size: {results.total_size} bytes\n")
            f.write(f"Success: {results.success}\n")
            if results.error_message:
                f.write(f"Error: {results.error_message}\n")

        logger.info(f"✅ Job {job_file} archived successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to archive completed job: {e}")
        return False
