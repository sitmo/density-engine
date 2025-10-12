"""
File operations for the vast.ai automation system.
"""

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

from ..utils.exceptions import FileOperationError
from ..utils.logging import get_logger, log_function_call

logger = get_logger(__name__)


class JobState(Enum):
    """Job states."""

    TODO = "todo"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def move_file(source: Path, destination: Path) -> bool:
    """Move a file from source to destination."""
    try:
        logger.debug(f"Moving file: {source} -> {destination}")

        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Use atomic move operation
        temp_file = destination.with_suffix(destination.suffix + ".tmp")

        # First copy to temp file
        shutil.copy2(str(source), str(temp_file))

        # Verify copy succeeded
        if not temp_file.exists():
            raise Exception("Copy to temp file failed")

        # Remove original file
        source.unlink()

        # Atomic rename temp to final destination
        temp_file.rename(destination)

        logger.info(f"Moved file successfully: {source} -> {destination}")
        return True

    except Exception as e:
        logger.error(f"Failed to move file {source} to {destination}: {e}")
        # Clean up temp file if it exists
        if "temp_file" in locals() and temp_file.exists():
            temp_file.unlink()
        raise FileOperationError(f"Failed to move file: {e}")


@log_function_call
def copy_file(source: Path, destination: Path) -> bool:
    """Copy a file from source to destination."""
    try:
        logger.debug(f"Copying file: {source} -> {destination}")

        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(str(source), str(destination))

        logger.info(f"Copied file successfully: {source} -> {destination}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy file {source} to {destination}: {e}")
        raise FileOperationError(f"Failed to copy file: {e}")


@log_function_call
def ensure_directory_exists(path: Path) -> bool:
    """Ensure a directory exists."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise FileOperationError(f"Failed to create directory: {e}")


@log_function_call
def list_files(directory: Path, pattern: str = "*") -> list[Path]:
    """List files in a directory matching a pattern."""
    try:
        if not directory.exists():
            return []

        files = list(directory.glob(pattern))
        logger.debug(f"Found {len(files)} files in {directory} matching {pattern}")
        return files

    except Exception as e:
        logger.error(f"Failed to list files in {directory}: {e}")
        raise FileOperationError(f"Failed to list files: {e}")


def move_job_file(job_file: str, from_state: JobState, to_state: JobState) -> bool:
    """Move a job file between state directories."""
    try:
        # Define directory structure
        base_dir = Path("jobs")
        from_dir = base_dir / from_state.value
        to_dir = base_dir / to_state.value

        source_file = from_dir / job_file
        dest_file = to_dir / job_file

        if not source_file.exists():
            logger.warning(
                f"Source file {source_file} does not exist for move from {from_state.value} to {to_state.value}"
            )
            return False

        # Ensure destination directory exists
        ensure_directory_exists(to_dir)

        # Move the file
        success = move_file(source_file, dest_file)

        if success:
            logger.info(f"Moved {job_file} from {from_state.value} to {to_state.value}")
            # Verify the move completed
            if dest_file.exists():
                logger.debug(f"File move verified: {dest_file} exists")
            else:
                logger.error(f"File move failed: {dest_file} does not exist after move")
                return False

        return success

    except Exception as e:
        logger.error(f"Failed to move job file {job_file}: {e}")
        raise FileOperationError(f"Failed to move job file: {e}")


def atomic_file_move(source: Path, destination: Path) -> bool:
    """Perform an atomic file move operation."""
    return move_file(source, destination)


@log_function_call
def cleanup_temp_files(directory: Path) -> int:
    """Clean up temporary files in a directory."""
    try:
        temp_files = list_files(directory, "*.tmp")
        cleaned_count = 0

        for temp_file in temp_files:
            try:
                temp_file.unlink()
                cleaned_count += 1
                logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

        logger.info(f"Cleaned up {cleaned_count} temporary files")
        return cleaned_count

    except Exception as e:
        logger.error(f"Failed to cleanup temp files in {directory}: {e}")
        raise FileOperationError(f"Failed to cleanup temp files: {e}")
