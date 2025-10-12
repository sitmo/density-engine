#!/usr/bin/env python3
"""
Vast.ai Manager - Clean interface using the new modular design.

This script provides a clean interface to the vast.ai automation system
using the new modular architecture in density_engine/vast/.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from density_engine.vast.instances.discovery import discover_active_instances, InstanceInfo
from density_engine.vast.instances.preparation import prepare_instance_for_jobs
from density_engine.vast.instances.monitoring import get_instance_summary
from density_engine.vast.execution.job_runner import execute_job_on_instance
from density_engine.vast.execution.result_handler import collect_job_results
from density_engine.vast.core.ssh import create_ssh_connection
from density_engine.vast.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def find_machines(price_max: float = 0.2, gpu_memory_min: int = 8, limit: int = 10):
    """Find suitable machines on vast.ai."""
    try:
        logger.info(f"Finding machines: price_max=${price_max}, gpu_memory_min={gpu_memory_min}GB")
        
        instances = discover_active_instances()
        
        # Filter by criteria
        suitable = []
        for instance in instances:
            if instance.price_per_hour <= price_max and instance.ssh_host and instance.ssh_port:
                suitable.append(instance)
        
        # Sort by price
        suitable.sort(key=lambda x: x.price_per_hour)
        
        # Display results
        print(f"\nFound {len(suitable)} suitable machines:")
        print("-" * 80)
        print(f"{'ID':<8} {'GPU':<20} {'Price/hr':<10} {'SSH':<15} {'Status':<10}")
        print("-" * 80)
        
        for instance in suitable[:limit]:
            ssh_info = f"{instance.ssh_host}:{instance.ssh_port}" if instance.ssh_host else "N/A"
            print(f"{instance.contract_id:<8} {instance.gpu_name:<20} ${instance.price_per_hour:<9.3f} {ssh_info:<15} {instance.status:<10}")
        
        return suitable[:limit]
        
    except Exception as e:
        logger.error(f"Failed to find machines: {e}")
        return []


def rent_machine(machine_id: int, disk: int = 50):
    """Rent a machine using vast.ai CLI."""
    try:
        logger.info(f"Renting machine {machine_id} with {disk}GB disk")
        
        import subprocess
        
        command = [
            "python3", "vast.py", "create", "instance", str(machine_id),
            "--template_hash", "1b8178d5b54c33b04683af690b18b72b",
            "--disk", str(disk),
            "--ssh", "--direct"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully rented machine {machine_id}")
            return True
        else:
            logger.error(f"❌ Failed to rent machine: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to rent machine: {e}")
        return False


def show_instances():
    """Show all active instances."""
    try:
        logger.info("Fetching active instances")
        
        instances = discover_active_instances()
        
        if not instances:
            print("No active instances found")
            return []
        
        print(f"\nActive Instances ({len(instances)}):")
        print("-" * 100)
        print(f"{'Contract ID':<12} {'GPU':<20} {'Price/hr':<10} {'SSH':<20} {'Status':<10} {'Health':<10}")
        print("-" * 100)
        
        for instance in instances:
            ssh_info = f"{instance.ssh_host}:{instance.ssh_port}" if instance.ssh_host else "N/A"
            
            # Get health status
            try:
                summary = get_instance_summary(instance)
                health = summary.health.value
            except:
                health = "Unknown"
            
            print(f"{instance.contract_id:<12} {instance.gpu_name:<20} ${instance.price_per_hour:<9.3f} {ssh_info:<20} {instance.status:<10} {health:<10}")
        
        return instances
        
    except Exception as e:
        logger.error(f"Failed to show instances: {e}")
        return []


def prepare_instance(contract_id: str):
    """Prepare an instance for running jobs."""
    try:
        logger.info(f"Preparing instance {contract_id}")
        
        # Find the instance
        instances = discover_active_instances()
        instance = None
        
        for inst in instances:
            if str(inst.contract_id) == contract_id:
                instance = inst
                break
        
        if not instance:
            logger.error(f"Instance {contract_id} not found")
            return False
        
        # Prepare the instance
        success = prepare_instance_for_jobs(instance)
        
        if success:
            logger.info(f"✅ Instance {contract_id} prepared successfully")
        else:
            logger.error(f"❌ Failed to prepare instance {contract_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to prepare instance: {e}")
        return False


def run_jobs(contract_id: str, job_file: str):
    """Run jobs on an instance."""
    try:
        logger.info(f"Running job {job_file} on instance {contract_id}")
        
        # Find the instance
        instances = discover_active_instances()
        instance = None
        
        for inst in instances:
            if str(inst.contract_id) == contract_id:
                instance = inst
                break
        
        if not instance:
            logger.error(f"Instance {contract_id} not found")
            return False
        
        # Check if job file exists
        job_path = Path(f"jobs/todo/{job_file}")
        if not job_path.exists():
            logger.error(f"Job file {job_path} not found")
            return False
        
        # Execute the job
        success = execute_job_on_instance(instance, job_path)
        
        if success:
            logger.info(f"✅ Job {job_file} started on instance {contract_id}")
        else:
            logger.error(f"❌ Failed to start job {job_file} on instance {contract_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to run jobs: {e}")
        return False


def download_results(contract_id: str):
    """Download results from an instance."""
    try:
        logger.info(f"Downloading results from instance {contract_id}")
        
        # Find the instance
        instances = discover_active_instances()
        instance = None
        
        for inst in instances:
            if str(inst.contract_id) == contract_id:
                instance = inst
                break
        
        if not instance:
            logger.error(f"Instance {contract_id} not found")
            return False
        
        # Collect results (this will find all parquet files)
        result_files = collect_job_results(instance, "all_jobs")
        
        if result_files:
            logger.info(f"✅ Downloaded {len(result_files)} result files")
            return True
        else:
            logger.warning("No result files found")
            return False
        
    except Exception as e:
        logger.error(f"Failed to download results: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Vast.ai Manager - Clean interface using modular design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find suitable machines
  python manage.py --find --price-max 0.2 --gpu-memory 8
  
  # Rent a machine
  python manage.py --rent 35055 --disk 50
  
  # Show all instances
  python manage.py --list
  
  # Prepare an instance
  python manage.py --prepare 12345
  
  # Run jobs on an instance
  python manage.py --run-jobs 12345 garch_test_jobs.csv
  
  # Download results
  python manage.py --download 12345
        """
    )
    
    # Main commands
    parser.add_argument('--find', action='store_true', help='Find suitable machines')
    parser.add_argument('--rent', type=int, help='Rent a machine by ID')
    parser.add_argument('--list', action='store_true', help='List all instances')
    parser.add_argument('--prepare', type=str, help='Prepare an instance by contract ID')
    parser.add_argument('--run-jobs', nargs=2, metavar=('CONTRACT_ID', 'JOB_FILE'), help='Run jobs on an instance')
    parser.add_argument('--download', type=str, help='Download results from an instance')
    
    # Options
    parser.add_argument('--price-max', type=float, default=0.2, help='Maximum price per hour (default: 0.2)')
    parser.add_argument('--gpu-memory', type=int, default=8, help='Minimum GPU memory in GB (default: 8)')
    parser.add_argument('--disk', type=int, default=50, help='Disk size in GB for rental (default: 50)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        if args.find:
            find_machines(args.price_max, args.gpu_memory)
        
        elif args.rent:
            rent_machine(args.rent, args.disk)
        
        elif args.list:
            show_instances()
        
        elif args.prepare:
            prepare_instance(args.prepare)
        
        elif args.run_jobs:
            contract_id, job_file = args.run_jobs
            run_jobs(contract_id, job_file)
        
        elif args.download:
            download_results(args.download)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
