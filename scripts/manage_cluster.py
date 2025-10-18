#!/usr/bin/env python3
"""
Cluster management script for density-engine.

This script manages a cluster of instances for processing GARCH jobs.
It imports generic functionality from density_engine.cluster and defines
project-specific configuration.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from density_engine.cluster import ClusterManager, JobManager

# ---------- Project-Specific Configuration ----------

# Repository and environment configuration
REPO_DIR = "/root/density-engine"
VENV_PYTHON = "/venv/main/bin/python"
PREPARATION_MARKER = f"{REPO_DIR}/.prepared"

# SSH configuration
SSH_KEY_PATHS = "~/.ssh/id_ed25519"  # comma-separated if multiple

# Job configuration
NUM_SIM = 10_000_000
NUM_QUANTILES = 512
STRIDE = 1

# Cluster configuration
DISCOVERY_INTERVAL = 30  # seconds
INSTANCE_HANDLE_INTERVAL = 10  # seconds

# Vast CLI command
VAST_CLI = ["python", "vast.py", "show", "instances", "--raw"]

# Local directories
JOBS_DIR = Path("jobs")
INPUT_QUEUE_DIR = JOBS_DIR / "pending"
OUTPUT_DIR = Path("outputs")
RUNNING_DIR = JOBS_DIR / "running"
FAILED_DIR = JOBS_DIR / "failed"

# Ensure directories exist
for dir_path in [INPUT_QUEUE_DIR, OUTPUT_DIR, RUNNING_DIR, FAILED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ---------- Logging Configuration ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logging.getLogger("asyncssh").setLevel(logging.WARNING)
logger = logging.getLogger("🧮 cluster-manager")

# ---------- Global State ----------
cluster_manager: ClusterManager = None
job_manager: JobManager = None
shutdown_in_progress = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_in_progress
    if shutdown_in_progress:
        return  # Already shutting down
    shutdown_in_progress = True
    logger.info(f"Received signal {signum}, shutting down...")
    if cluster_manager:
        asyncio.create_task(cluster_manager.stop())
    sys.exit(0)


async def main():
    """Main cluster management loop."""
    global cluster_manager, job_manager
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse SSH key files
    ssh_key_files = []
    for key_path in SSH_KEY_PATHS.split(","):
        key_path = os.path.expanduser(key_path.strip())
        if key_path and os.path.exists(key_path):
            ssh_key_files.append(key_path)
    
    if not ssh_key_files:
        logger.error("No SSH key files found. Please check SSH_KEY_PATHS configuration.")
        sys.exit(1)
    
    # Create job manager
    job_manager = JobManager(
        input_queue_dir=INPUT_QUEUE_DIR,
        output_dir=OUTPUT_DIR,
        running_dir=RUNNING_DIR,
        failed_dir=FAILED_DIR,
    )
    
    # Create cluster manager
    cluster_manager = ClusterManager(
        job_manager=job_manager,
        vast_cli_command=VAST_CLI,
        ssh_key_files=ssh_key_files,
        discovery_interval=DISCOVERY_INTERVAL,
    )
    
    # Print initial status
    logger.info("Starting density-engine cluster manager")
    logger.info(f"Input queue: {INPUT_QUEUE_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"SSH keys: {ssh_key_files}")
    logger.info(f"Discovery interval: {DISCOVERY_INTERVAL}s")
    
    # Check for available jobs
    available_jobs = job_manager.get_available_jobs()
    logger.info(f"Found {len(available_jobs)} jobs in input queue")
    
    if not available_jobs:
        logger.warning("No jobs found in input queue. Add CSV files to jobs/pending/ to start processing.")
    
    try:
        # Start cluster manager
        await cluster_manager.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Cluster manager error: {e}")
        raise
    finally:
        global shutdown_in_progress
        if not shutdown_in_progress and cluster_manager:
            shutdown_in_progress = True
            await cluster_manager.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
