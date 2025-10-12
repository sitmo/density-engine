#!/usr/bin/env python3
"""
Rent a machine on vast.ai using the new modular design.
"""

import argparse
import ast
import subprocess
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from density_engine.vast.instances.discovery import discover_active_instances
from density_engine.vast.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def rent_machine_cli(machine_id: int, disk: int = 50, debug: bool = False, wait: bool = False):
    """Rent a machine using vast.ai CLI."""
    try:
        logger.info(f"Renting machine {machine_id} with {disk}GB disk")
        
        if debug:
            logger.info(f"Command: python3 vast.py create instance {machine_id} --template_hash 1b8178d5b54c33b04683af690b18b72b --disk {disk} --ssh --direct")
        
        # Use vast.py CLI to rent machine
        command = [
            "python3", "vast.py", "create", "instance", str(machine_id),
            "--template_hash", "1b8178d5b54c33b04683af690b18b72b",  # Official vast.ai PyTorch template
            "--disk", str(disk),
            "--ssh", "--direct",
            "--onstart-cmd", "echo 'Machine started'"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            stdout = result.stdout.strip()
            
            if debug:
                logger.info(f"Raw output: {stdout}")
            
            # Handle "Started. " prefix
            if stdout.startswith("Started. "):
                stdout = stdout[9:]
            
            # Check for error messages
            if stdout.startswith("failed with error"):
                logger.error(f"❌ Rental failed: {stdout}")
                return None
            
            try:
                # Parse the response
                response = ast.literal_eval(stdout)
                
                if response.get('success'):
                    contract_id = response.get('new_contract', response.get('contract_id'))
                    logger.info(f"✅ Successfully rented machine {machine_id}")
                    logger.info(f"Contract ID: {contract_id}")
                    
                    if wait:
                        logger.info("Waiting for instance to be ready...")
                        wait_for_instance_ready(contract_id, debug)
                    
                    return response
                else:
                    logger.error(f"❌ Rental failed: {response}")
                    return None
                    
            except (ValueError, SyntaxError) as e:
                logger.error(f"❌ Failed to parse response: {e}")
                logger.error(f"Raw response: {stdout}")
                return None
        
        else:
            logger.error(f"❌ Command failed with return code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Rental command timed out")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to rent machine: {e}")
        return None


def wait_for_instance_ready(contract_id: int, debug: bool = False, max_wait: int = 300):
    """Wait for an instance to be ready."""
    try:
        logger.info(f"Waiting for instance {contract_id} to be ready...")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Check if instance is in the active instances list
            instances = discover_active_instances()
            
            for instance in instances:
                if instance.contract_id == contract_id:
                    if instance.ssh_host and instance.ssh_port:
                        logger.info(f"✅ Instance {contract_id} is ready!")
                        logger.info(f"SSH: {instance.ssh_host}:{instance.ssh_port}")
                        return True
            
            if debug:
                logger.info(f"Instance {contract_id} not ready yet, waiting...")
            
            time.sleep(10)
        
        logger.warning(f"❌ Instance {contract_id} did not become ready within {max_wait} seconds")
        return False
        
    except Exception as e:
        logger.error(f"Failed to wait for instance: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Rent a machine on vast.ai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rent machine 35055 with default 50GB disk
  python rent_machine.py 35055
  
  # Rent machine with custom disk size
  python rent_machine.py 35055 --disk 100
  
  # Rent machine and wait for it to be ready
  python rent_machine.py 35055 --wait
  
  # Rent machine with debug output
  python rent_machine.py 35055 --debug
        """
    )
    
    parser.add_argument('machine_id', type=int, help='Machine ID to rent')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--wait', action='store_true', help='Wait for instance to be ready')
    parser.add_argument('--disk', type=int, default=50, help='Disk size in GB (default: 50)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        result = rent_machine_cli(
            machine_id=args.machine_id,
            disk=args.disk,
            debug=args.debug,
            wait=args.wait
        )
        
        if result:
            print(f"✅ Successfully rented machine {args.machine_id}")
            if 'new_contract' in result:
                print(f"Contract ID: {result['new_contract']}")
            sys.exit(0)
        else:
            print(f"❌ Failed to rent machine {args.machine_id}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Rental interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Rental failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()