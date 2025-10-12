#!/usr/bin/env python3
"""
Find suitable machines on vast.ai using the new modular design.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from density_engine.vast.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def find_suitable_machines(gpu_memory_min: int = 4, price_max: float = 0.2, limit: int = 20, debug: bool = False, json_output: bool = False):
    """Find suitable machines available for rent on vast.ai."""
    try:
        if debug:
            logger.info(f"Searching for available machines: gpu_memory_min={gpu_memory_min}GB, price_max=${price_max}")
        
        # Use vast.py CLI to search for available machines
        import subprocess
        
        # Build query string for vast.py search offers
        query_parts = [
            f"dph_total<={price_max}",
            f"gpu_ram>={gpu_memory_min}",
            "rentable=true",
            "verified=true"
        ]
        query = " ".join(query_parts)
        
        command = [
            "python3", "vast.py", "search", "offers",
            query,
            "--limit", str(limit),
            "--order", "dph",  # Sort by price ascending (cheapest first)
            "--raw"
        ]
        
        if debug:
            logger.info(f"Running command: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.error(f"Search command failed: {result.stderr}")
            if json_output:
                print(json.dumps({'error': result.stderr}))
            return []
        
        # Parse the output
        output = result.stdout.strip()
        if not output:
            logger.info("No machines found matching criteria")
            if json_output:
                print(json.dumps([]))
            return []
        
        try:
            # Parse JSON output
            machines_data = json.loads(output)
            if not isinstance(machines_data, list):
                machines_data = [machines_data]
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON output: {output}")
            if json_output:
                print(json.dumps({'error': 'Failed to parse API response'}))
            return []
        
        if debug:
            logger.info(f"Found {len(machines_data)} machines from API")
            if machines_data:
                logger.info(f"Sample machine data: {machines_data[0]}")
        
        # Filter and format results
        suitable_machines = []
        for machine in machines_data:
            # Extract machine information
            machine_id = machine.get('id', 'Unknown')
            gpu_name = machine.get('gpu_name', 'Unknown')
            price_per_hour = machine.get('dph_total', 0.0)  # dph_total = total dollars per hour
            gpu_ram_gb = machine.get('gpu_ram', 0)  # Already in GB
            cpu_cores = machine.get('cpu_cores', 0)
            cpu_ram_gb = machine.get('cpu_ram', 0)  # Already in GB
            location = machine.get('geolocation', 'Unknown')
            reliability = machine.get('reliability', 0)
            
            # Additional filtering if needed
            if price_per_hour <= price_max and gpu_ram_gb >= gpu_memory_min:
                suitable_machines.append({
                    'machine_id': machine_id,
                    'gpu_name': gpu_name,
                    'price_per_hour': price_per_hour,
                    'gpu_ram_gb': gpu_ram_gb,
                    'cpu_cores': cpu_cores,
                    'cpu_ram_gb': cpu_ram_gb,
                    'location': location,
                    'reliability': reliability
                })
        
        # Sort by price (cheapest first)
        suitable_machines.sort(key=lambda x: x['price_per_hour'])
        
        if json_output:
            # Output as JSON
            print(json.dumps(suitable_machines, indent=2))
        else:
            # Output as table
            print(f"\nFound {len(suitable_machines)} suitable machines:")
            print("-" * 120)
            print(f"{'Machine ID':<12} {'GPU':<25} {'Price/hr':<10} {'GPU RAM':<10} {'CPU':<8} {'RAM':<8} {'Location':<15} {'Reliability':<12}")
            print("-" * 120)
            
            for machine in suitable_machines:
                print(f"{machine['machine_id']:<12} {machine['gpu_name']:<25} ${machine['price_per_hour']:<9.3f} {machine['gpu_ram_gb']:<9.1f}GB {machine['cpu_cores']:<8} {machine['cpu_ram_gb']:<7.1f}GB {machine['location']:<15} {machine['reliability']:<12}")
        
        return suitable_machines
        
    except Exception as e:
        logger.error(f"Failed to find suitable machines: {e}")
        if json_output:
            print(json.dumps({'error': str(e)}))
        return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Find suitable machines on vast.ai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find machines under $0.2/hour with 8GB+ GPU memory
  python find_machines.py --price-max 0.2 --gpu-memory 8
  
  # Find machines and output as JSON
  python find_machines.py --price-max 0.3 --json
  
  # Find machines with debug output
  python find_machines.py --price-max 0.2 --debug
        """
    )
    
    parser.add_argument('--gpu-memory', type=int, default=8, help='Minimum GPU memory in GB (default: 8)')
    parser.add_argument('--price-max', type=float, default=0.2, help='Maximum price per hour in USD (default: 0.2)')
    parser.add_argument('--limit', type=int, default=20, help='Maximum number of machines to display (default: 20)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        find_suitable_machines(
            gpu_memory_min=args.gpu_memory,
            price_max=args.price_max,
            limit=args.limit,
            debug=args.debug,
            json_output=args.json
        )
    
    except KeyboardInterrupt:
        logger.info("Search interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()