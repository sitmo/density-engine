#!/usr/bin/env python3
"""
Minimal example: Rent RTX 30+ instances and monitor them.

This script demonstrates:
1. Renting 2 RTX 30+ instances for max $0.06/hr
2. Monitoring their status for 3 minutes
3. Canceling them after 3 minutes

Usage:
    python examples/rent_and_monitor_instances.py
"""

import time
import sys
from pathlib import Path
from typing import List, Dict, Any
from tabulate import tabulate

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from density_engine.cluster.vast_api import (
    find_best_offer,
    rent_instance,
    destroy_instance,
    destroy_all_instances,
    list_instances,
    validate_gpu_names
)
from density_engine.cluster.gpu_names import RTX_30_PLUS


def print_instances_table(instances: List[Dict[str, Any]]) -> None:
    """Print instances in a nice table format."""
    if not instances:
        print("No instances found.")
        return
    
    # Prepare table data
    table_data = []
    for instance in instances:
        table_data.append([
            instance["id"],
            instance["gpu_name"],
            f"${instance['price_per_hour']:.3f}/hr",
            f"{instance['reliability']:.1%}",
            instance["location"],
            instance["status"],
            f"{instance['cpu_cores']} cores",
            f"{instance['gpu_ram']}GB"
        ])
    
    headers = [
        "ID", "GPU", "Price", "Reliability", 
        "Location", "Status", "CPU", "RAM"
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def rent_instances(count: int = 2, max_price: float = 0.06) -> List[str]:
    """Rent the specified number of RTX 30+ instances."""
    print(f"🔍 Searching for {count} RTX 30+ instances at max ${max_price}/hr...")
    
    rented_instance_ids = []
    
    for i in range(count):
        print(f"\n📋 Renting instance {i+1}/{count}...")
        
        # Find best offer
        best_offer = find_best_offer(
            gpu_list=RTX_30_PLUS,
            max_price=max_price,
            min_reliability=0.90,
            min_gpu_ram=8,
            sort_by_price=True
        )
        
        if not best_offer:
            print(f"❌ No suitable RTX 30+ instances found for ${max_price}/hr")
            continue
        
        print(f"✅ Found: {best_offer['gpu_name']} at ${best_offer['price_per_hour']:.3f}/hr "
              f"(reliability: {best_offer['reliability']:.1%}, location: {best_offer['location']})")
        
        # Rent the instance
        rental_result = rent_instance(best_offer["id"])
        
        if rental_result:
            instance_id = rental_result["instance_id"]
            rented_instance_ids.append(instance_id)
            print(f"🚀 Successfully rented instance: {instance_id}")
        else:
            print(f"❌ Failed to rent instance {best_offer['id']}")
    
    return rented_instance_ids


def monitor_instances(instance_ids: List[str], duration_minutes: int = 3) -> None:
    """Monitor instances for the specified duration."""
    print(f"\n⏱️  Monitoring {len(instance_ids)} instances for {duration_minutes} minutes...")
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    while time.time() < end_time:
        remaining_time = int(end_time - time.time())
        print(f"\n⏰ Time remaining: {remaining_time}s")
        
        # Get current instances
        all_instances = list_instances()
        
        # Filter to our rented instances
        our_instances = [
            inst for inst in all_instances 
            if inst["id"] in instance_ids
        ]
        
        print(f"\n📊 Our rented instances ({len(our_instances)}/{len(instance_ids)}):")
        print_instances_table(our_instances)
        
        # Check if all instances are ready
        ready_count = sum(1 for inst in our_instances if inst["status"] == "running")
        print(f"\n✅ Ready instances: {ready_count}/{len(our_instances)}")
        
        # Wait 30 seconds before next check
        time.sleep(30)
    
    print(f"\n⏰ Monitoring period complete!")


def cancel_instances(instance_ids: List[str]) -> None:
    """Cancel all rented instances."""
    print(f"\n🗑️  Canceling {len(instance_ids)} instances...")
    
    for instance_id in instance_ids:
        print(f"🗑️  Canceling instance {instance_id}...")
        success = destroy_instance(instance_id)
        
        if success:
            print(f"✅ Successfully canceled instance {instance_id}")
        else:
            print(f"❌ Failed to cancel instance {instance_id}")


def main():
    """Main function."""
    print("🚀 RTX 30+ Instance Rental Demo")
    print("=" * 50)
    
    # Validate GPU names
    print("🔍 Validating RTX 30+ GPU names...")
    valid_gpus = validate_gpu_names(RTX_30_PLUS[:5])  # Check first 5
    print(f"✅ Found {len(valid_gpus)} valid RTX 30+ GPUs: {valid_gpus}")
    
    # Rent instances
    rented_ids = rent_instances(count=2, max_price=0.06)
    
    if not rented_ids:
        print("❌ No instances were rented. Exiting.")
        return
    
    print(f"\n🎉 Successfully rented {len(rented_ids)} instances!")
    
    # Monitor instances
    monitor_instances(rented_ids, duration_minutes=3)
    
    # Cancel instances
    cancel_instances(rented_ids)
    
    # Destroy all instances (cleanup)
    print("\n🧹 Cleaning up all remaining instances...")
    success = destroy_all_instances(max_wait_time=300)  # 5 minutes max
    
    if success:
        print("✅ All instances successfully destroyed!")
    else:
        print("⚠️  Some instances may still be running. Check manually.")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
