"""
Job operations for the vast.ai automation system.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List

from ..utils.exceptions import FileOperationError
from ..utils.logging import get_logger, log_function_call
from .files import JobState

logger = get_logger(__name__)


@log_function_call
def parse_job_file(job_file: Path) -> dict[str, Any]:
    """Parse a job file and return its contents."""
    try:
        if not job_file.exists():
            raise FileOperationError(f"Job file {job_file} does not exist")

        logger.debug(f"Parsing job file: {job_file}")

        with open(job_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise FileOperationError(f"Job file {job_file} is empty")

        # Extract job metadata from CSV content
        job_info = {
            "filename": job_file.name,
            "rows": rows,
            "row_count": len(rows),
            "job_type": "garch",  # Default type, could be determined from directory or other means
        }

        logger.debug(
            f"Parsed job file: {job_info['row_count']} rows, type: {job_info['job_type']}"
        )
        return job_info

    except Exception as e:
        logger.error(f"Failed to parse job file {job_file}: {e}")
        raise FileOperationError(f"Failed to parse job file: {e}")


@log_function_call
def validate_job_file(job_file: Path) -> bool:
    """Validate a job file."""
    try:
        job_info = parse_job_file(job_file)

        # Check if file has rows
        if job_info["row_count"] == 0:
            logger.warning(f"Job file {job_file} has no rows")
            return False

        # Check if all rows have required columns
        required_columns = ["omega", "alpha", "beta", "gamma", "eta", "lam"]
        rows = job_info["rows"]

        for i, row in enumerate(rows):
            for col in required_columns:
                if col not in row or not row[col]:
                    logger.warning(f"Job file {job_file} row {i} missing column {col}")
                    return False

        logger.debug(f"Job file {job_file} validation passed")
        return True

    except Exception as e:
        logger.error(f"Failed to validate job file {job_file}: {e}")
        return False


@log_function_call
def discover_job_files(directory: Path) -> list[Path]:
    """Discover job files in a directory."""
    try:
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return []

        job_files = list(directory.glob("*.csv"))
        logger.info(f"Discovered {len(job_files)} job files in {directory}")
        return job_files

    except Exception as e:
        logger.error(f"Failed to discover job files in {directory}: {e}")
        raise FileOperationError(f"Failed to discover job files: {e}")


@log_function_call
def prioritize_jobs(job_files: list[Path]) -> list[Path]:
    """Prioritize job files for execution."""
    try:
        # Sort by filename (which includes row numbers)
        # This ensures jobs are processed in order
        sorted_jobs = sorted(job_files, key=lambda x: x.name)

        logger.debug(f"Prioritized {len(sorted_jobs)} jobs")
        return sorted_jobs

    except Exception as e:
        logger.error(f"Failed to prioritize jobs: {e}")
        raise FileOperationError(f"Failed to prioritize jobs: {e}")
