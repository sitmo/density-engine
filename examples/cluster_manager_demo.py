#!/usr/bin/env python3
"""
Demo script for the enhanced ClusterManager with automatic instance provisioning.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from density_engine.cluster.cluster_manager import ClusterManager
from density_engine.cluster.job_manager import JobManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Demo the enhanced ClusterManager."""
    logger.info("🚀 ClusterManager Demo with Automatic Instance Provisioning")
    logger.info("=" * 60)
    
    # Create a job manager with required directories
    job_manager = JobManager(
        input_queue_dir="jobs/pending",
        output_dir="jobs/completed", 
        running_dir="jobs/running",
        failed_dir="jobs/failed"
    )
    
    # Create ClusterManager with new parameters
    cluster_manager = ClusterManager(
        job_manager=job_manager,
        ssh_key_files=["~/.ssh/id_rsa"],  # Replace with your SSH key
        target_instances=3,               # Target 3 instances
        image_name="pytorch/pytorch",     # Use PyTorch image
        max_price_per_hour=0.05,          # Max $0.05/hour
        # allowed_gpus=None,              # Will default to RTX_30_PLUS and validate
        discovery_interval=30             # Check every 30 seconds
    )
    
    logger.info(f"📋 Configuration:")
    logger.info(f"  • Target instances: {cluster_manager.target_instances}")
    logger.info(f"  • Image: {cluster_manager.image_name}")
    logger.info(f"  • Max price: ${cluster_manager.max_price_per_hour}/hr")
    logger.info(f"  • Allowed GPUs: {len(cluster_manager.allowed_gpus)} validated GPUs")
    logger.info(f"    {cluster_manager.allowed_gpus[:5]}{'...' if len(cluster_manager.allowed_gpus) > 5 else ''}")
    
    # Example of using custom GPU list
    logger.info(f"\n💡 Example: Custom GPU list would be:")
    logger.info(f"    allowed_gpus=['RTX_4090', 'RTX_4080']  # Will be validated")
    
    try:
        # Start the cluster manager
        logger.info("\n🔄 Starting cluster manager...")
        await cluster_manager.start()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopping cluster manager...")
        await cluster_manager.stop()
        logger.info("✅ Cluster manager stopped")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        await cluster_manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
