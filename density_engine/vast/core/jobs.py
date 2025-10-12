"""
Job operations for the vast.ai automation system.
"""

import csv
from datetime import timedelta
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

        # Extract job metadata from filename
        filename_parts = job_file.stem.split("_")
        job_info = {
            "filename": job_file.name,
            "rows": rows,
            "row_count": len(rows),
            "job_type": filename_parts[0] if filename_parts else "unknown",
            "start_row": int(filename_parts[2]) if len(filename_parts) > 2 else 0,
            "end_row": (
                int(filename_parts[3]) if len(filename_parts) > 3 else len(rows) - 1
            ),
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
def get_job_arguments(job_file: Path) -> dict[str, Any]:
    """Get job arguments from a job file."""
    try:
        job_info = parse_job_file(job_file)

        # Default arguments
        args = {"num_sim": 1000000, "num_quantiles": 512, "stride": 1}

        # Extract arguments from filename if present
        filename_parts = job_file.stem.split("_")
        if len(filename_parts) > 5:
            try:
                args["num_sim"] = int(filename_parts[5])
            except (ValueError, IndexError):
                pass

        if len(filename_parts) > 6:
            try:
                args["num_quantiles"] = int(filename_parts[6])
            except (ValueError, IndexError):
                pass

        logger.debug(f"Job arguments for {job_file}: {args}")
        return args

    except Exception as e:
        logger.error(f"Failed to get job arguments for {job_file}: {e}")
        return {"num_sim": 1000000, "num_quantiles": 512, "stride": 1}


@log_function_call
def estimate_job_duration(job_file: Path) -> timedelta:
    """Estimate job duration based on job parameters."""
    try:
        job_info = parse_job_file(job_file)
        args = get_job_arguments(job_file)

        # Rough estimation based on number of simulations and rows
        num_sim = args.get("num_sim", 1000000)
        num_rows = job_info["row_count"]

        # Estimate: 1M simulations per row takes about 2 minutes
        base_time_per_row = 2  # minutes
        simulation_factor = num_sim / 1000000

        estimated_minutes = num_rows * base_time_per_row * simulation_factor
        estimated_duration = timedelta(minutes=estimated_minutes)

        logger.debug(f"Estimated duration for {job_file}: {estimated_duration}")
        return estimated_duration

    except Exception as e:
        logger.error(f"Failed to estimate job duration for {job_file}: {e}")
        # Return default estimate
        return timedelta(minutes=10)


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
def categorize_jobs_by_type(job_files: list[Path]) -> dict[str, list[Path]]:
    """Categorize job files by type."""
    try:
        categories = {}

        for job_file in job_files:
            job_info = parse_job_file(job_file)
            job_type = job_info["job_type"]

            if job_type not in categories:
                categories[job_type] = []

            categories[job_type].append(job_file)

        logger.info(f"Categorized {len(job_files)} jobs into {len(categories)} types")
        for job_type, files in categories.items():
            logger.debug(f"  {job_type}: {len(files)} files")

        return categories

    except Exception as e:
        logger.error(f"Failed to categorize jobs: {e}")
        raise FileOperationError(f"Failed to categorize jobs: {e}")


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
