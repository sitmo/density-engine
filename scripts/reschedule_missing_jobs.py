#!/usr/bin/env python3
"""
Script to reschedule missing jobs by copying them from todo to pending.

This script identifies jobs that are:
1. In the todo folder (all jobs)
2. NOT in the outputs folder (not completed)
3. NOT in the pending folder (not already scheduled)

And copies them to the pending folder for rescheduling.
"""

import os
import shutil
from pathlib import Path
from typing import Set, List


def get_job_basenames(folder_path: str, extension: str) -> Set[str]:
    """Extract job basenames from files in a folder."""
    basenames = set()
    folder = Path(folder_path)
    
    if not folder.exists():
        return basenames
    
    for file_path in folder.glob(f"*.{extension}"):
        # Extract the job basename (everything before the first underscore after the job name)
        filename = file_path.stem  # Remove extension
        
        # For outputs, we need to extract the base job name
        # Format: garch_test_job_0000_0099_0_streaming_1_10000000_512
        # We want: garch_test_job_0000_0099
        if extension == "parquet":
            # Split by underscore and take first 5 parts (garch_test_job_XXXX_YYYY)
            parts = filename.split("_")
            if len(parts) >= 5:
                basename = "_".join(parts[:5])
                basenames.add(basename)
        else:
            # For CSV files, the filename is already the basename
            basenames.add(filename)
    
    return basenames


def find_missing_jobs(todo_folder: str, pending_folder: str, outputs_folder: str) -> List[str]:
    """Find jobs that are in todo but not in pending or outputs."""
    print("Scanning folders...")
    
    # Get all job basenames from each folder
    todo_jobs = get_job_basenames(todo_folder, "csv")
    pending_jobs = get_job_basenames(pending_folder, "csv")
    completed_jobs = get_job_basenames(outputs_folder, "parquet")
    
    print(f"Found {len(todo_jobs)} jobs in todo folder")
    print(f"Found {len(pending_jobs)} jobs in pending folder")
    print(f"Found {len(completed_jobs)} jobs in outputs folder")
    
    # Find jobs that are in todo but not in pending or completed
    missing_jobs = todo_jobs - pending_jobs - completed_jobs
    
    # Convert to sorted list for consistent output
    missing_jobs_list = sorted(list(missing_jobs))
    
    return missing_jobs_list


def copy_jobs_to_pending(missing_jobs: List[str], todo_folder: str, pending_folder: str) -> None:
    """Copy missing jobs from todo to pending folder."""
    todo_path = Path(todo_folder)
    pending_path = Path(pending_folder)
    
    # Ensure pending folder exists
    pending_path.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    for job_basename in missing_jobs:
        source_file = todo_path / f"{job_basename}.csv"
        dest_file = pending_path / f"{job_basename}.csv"
        
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            copied_count += 1
            print(f"Copied: {job_basename}.csv")
        else:
            print(f"Warning: Source file not found: {source_file}")
    
    print(f"\nSuccessfully copied {copied_count} jobs to pending folder")


def main():
    """Main function to reschedule missing jobs."""
    # Define folder paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    todo_folder = project_root / "jobs" / "todo"
    pending_folder = project_root / "jobs" / "pending"
    outputs_folder = project_root / "outputs"
    
    print("=== Job Rescheduling Tool ===")
    print(f"Todo folder: {todo_folder}")
    print(f"Pending folder: {pending_folder}")
    print(f"Outputs folder: {outputs_folder}")
    print()
    
    # Find missing jobs
    missing_jobs = find_missing_jobs(str(todo_folder), str(pending_folder), str(outputs_folder))
    
    if not missing_jobs:
        print("No missing jobs found! All jobs are either pending or completed.")
        return
    
    print(f"\nFound {len(missing_jobs)} missing jobs:")
    print("=" * 50)
    
    # Display missing jobs (limit to first 20 for readability)
    display_jobs = missing_jobs[:20]
    for job in display_jobs:
        print(f"  {job}.csv")
    
    if len(missing_jobs) > 20:
        print(f"  ... and {len(missing_jobs) - 20} more jobs")
    
    print("=" * 50)
    
    # Ask for confirmation
    print(f"\nThese {len(missing_jobs)} jobs will be copied from todo to pending.")
    response = input("OK? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\nCopying jobs...")
        copy_jobs_to_pending(missing_jobs, str(todo_folder), str(pending_folder))
        print(f"\nDone! {len(missing_jobs)} jobs have been rescheduled.")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    main()
