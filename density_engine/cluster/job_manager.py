"""
Job management for cluster instances.

This module handles job file management, job assignment, and job lifecycle.
"""

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("JOB")


class JobManager:
    """Manages job files and assignment for cluster instances."""

    def __init__(
        self,
        input_queue_dir: Path,
        output_dir: Path,
        running_dir: Path,
        failed_dir: Path,
    ) -> None:
        self.input_queue_dir = Path(input_queue_dir)
        self.output_dir = Path(output_dir)
        self.running_dir = Path(running_dir)
        self.failed_dir = Path(failed_dir)

        # Ensure directories exist
        for dir_path in [
            self.input_queue_dir,
            self.output_dir,
            self.running_dir,
            self.failed_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Track jobs that are currently being processed
        self._processing_jobs: set[str] = set()

    def get_available_jobs(self) -> list[Path]:
        """Get list of available job files in input queue."""
        return list(self.input_queue_dir.glob("*.csv"))

    def get_job_count(self) -> int:
        """Get count of available jobs."""
        return len(self.get_available_jobs())

    def assign_job(self) -> Path | None:
        """Randomly assign a job from input queue."""
        available_jobs = self.get_available_jobs()
        if not available_jobs:
            return None

        # Random selection
        job_file = random.choice(available_jobs)

        # Move to running directory
        running_job = self.running_dir / job_file.name
        try:
            job_file.rename(running_job)
            self._processing_jobs.add(job_file.name)

            # Create dispatch metadata
            dispatch_info = {
                "assigned_at": time.time(),
                "original_path": str(job_file),
            }
            dispatch_file = self.running_dir / f"{job_file.stem}.dispatch.json"
            dispatch_file.write_text(json.dumps(dispatch_info, indent=2))

            logger.info(f"Assigned job: {job_file.name}")
            return running_job

        except FileNotFoundError:
            # Job was taken by another process
            logger.debug(f"Job {job_file.name} was already taken")
            return None
        except Exception as e:
            logger.error(f"Failed to assign job {job_file.name}: {e}")
            return None

    def return_job_to_queue(self, job_name: str, reason: str = "failed") -> None:
        """Return a job back to the input queue."""
        running_job = self.running_dir / job_name
        dispatch_file = self.running_dir / f"{Path(job_name).stem}.dispatch.json"

        if running_job.exists():
            try:
                # Move back to input queue
                input_job = self.input_queue_dir / job_name
                running_job.rename(input_job)
                logger.info(f"Returned job {job_name} to queue: {reason}")
            except Exception as e:
                logger.error(f"Failed to return job {job_name} to queue: {e}")

        # Clean up dispatch file
        if dispatch_file.exists():
            try:
                dispatch_file.unlink()
            except Exception:
                pass

        self._processing_jobs.discard(job_name)

    def mark_job_running(self, job_name: str) -> None:
        """Mark a job as running (move from input queue to running directory)."""
        input_job = self.input_queue_dir / job_name
        running_job = self.running_dir / job_name

        if input_job.exists():
            try:
                # Move from input queue to running
                input_job.rename(running_job)
                self._processing_jobs.add(job_name)
                logger.info(f"Marked job {job_name} as running")
            except Exception as e:
                logger.error(f"Failed to mark job {job_name} as running: {e}")
        else:
            logger.warning(f"Job {job_name} not found in input queue directory")

    def mark_job_failed(self, job_name: str, reason: str) -> None:
        """Mark a job as failed."""
        running_job = self.running_dir / job_name
        dispatch_file = self.running_dir / f"{Path(job_name).stem}.dispatch.json"

        if running_job.exists():
            try:
                # Move to failed directory
                failed_job = self.failed_dir / job_name
                running_job.rename(failed_job)

                # Write failure reason
                reason_file = self.failed_dir / f"{Path(job_name).stem}.reason.txt"
                reason_file.write_text(reason)

                logger.info(f"Marked job {job_name} as failed: {reason}")
            except Exception as e:
                logger.error(f"Failed to mark job {job_name} as failed: {e}")

        # Clean up dispatch file
        if dispatch_file.exists():
            try:
                dispatch_file.unlink()
            except Exception:
                pass

        self._processing_jobs.discard(job_name)

    def mark_job_completed(self, job_name: str) -> None:
        """Mark a job as completed."""
        running_job = self.running_dir / job_name
        dispatch_file = self.running_dir / f"{Path(job_name).stem}.dispatch.json"

        if running_job.exists():
            try:
                # Move to done directory (if it exists)
                done_dir = self.input_queue_dir.parent / "done"
                done_dir.mkdir(exist_ok=True)
                done_job = done_dir / job_name
                running_job.rename(done_job)

                logger.info(f"Marked job {job_name} as completed")
            except Exception as e:
                logger.error(f"Failed to mark job {job_name} as completed: {e}")

        # Clean up dispatch file
        if dispatch_file.exists():
            try:
                dispatch_file.unlink()
            except Exception:
                pass

        self._processing_jobs.discard(job_name)

    def get_processing_jobs(self) -> set[str]:
        """Get set of currently processing job names."""
        return self._processing_jobs.copy()

    def get_job_stats(self) -> dict[str, int]:
        """Get job statistics."""
        return {
            "available": len(self.get_available_jobs()),
            "processing": len(self._processing_jobs),
            "running": len(list(self.running_dir.glob("*.csv"))),
            "failed": len(list(self.failed_dir.glob("*.csv"))),
        }
