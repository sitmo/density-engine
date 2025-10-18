#!/usr/bin/env python3
"""
Test script for cluster management functionality.

This script tests the basic functionality of the cluster management modules
without requiring actual SSH connections or vast.ai instances.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from density_engine.cluster import JobManager, SSHClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_job_manager():
    """Test job manager functionality."""
    logger.info("Testing JobManager...")
    
    # Create test directories
    test_dir = Path("test_cluster")
    input_dir = test_dir / "pending"
    output_dir = test_dir / "outputs"
    running_dir = test_dir / "running"
    failed_dir = test_dir / "failed"
    
    # Clean up any existing test data
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
    
    # Create job manager
    job_manager = JobManager(
        input_queue_dir=input_dir,
        output_dir=output_dir,
        running_dir=running_dir,
        failed_dir=failed_dir,
    )
    
    # Create some test job files
    test_jobs = ["job1.csv", "job2.csv", "job3.csv"]
    for job_name in test_jobs:
        (input_dir / job_name).write_text("test,data\n1,2\n")
    
    # Test job assignment
    logger.info(f"Available jobs: {len(job_manager.get_available_jobs())}")
    
    assigned_job = job_manager.assign_job()
    if assigned_job:
        logger.info(f"Assigned job: {assigned_job.name}")
    
    # Test job stats
    stats = job_manager.get_job_stats()
    logger.info(f"Job stats: {stats}")
    
    # Test job failure handling
    if assigned_job:
        job_manager.mark_job_failed(assigned_job.name, "test failure")
        logger.info("Marked job as failed")
    
    # Test final stats
    final_stats = job_manager.get_job_stats()
    logger.info(f"Final stats: {final_stats}")
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir)
    logger.info("JobManager test completed")


async def test_ssh_client():
    """Test SSH client functionality (without actual connection)."""
    logger.info("Testing SSHClient...")
    
    # Create SSH client (won't actually connect)
    ssh_client = SSHClient(
        host="localhost",
        port=22,
        username="test",
        key_files=[],
    )
    
    # Test that we can create the client
    logger.info(f"Created SSH client for {ssh_client.host}:{ssh_client.port}")
    
    # Test connection attempt (will fail, but that's expected)
    try:
        await ssh_client.ensure_connection()
        logger.info("SSH connection successful (unexpected)")
    except Exception as e:
        logger.info(f"SSH connection failed as expected: {e}")
    
    # Test cleanup
    await ssh_client.close()
    logger.info("SSHClient test completed")


async def main():
    """Run all tests."""
    logger.info("Starting cluster management tests...")
    
    try:
        await test_job_manager()
        await test_ssh_client()
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
