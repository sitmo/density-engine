"""
Cluster manager for coordinating multiple instances.

This module handles the global cluster management including instance discovery,
job assignment, and coordination of instance managers.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .gpu_names import RTX_30_PLUS
from .instance_manager import InstanceManager
from .job_manager import JobManager
from .vast_api import (
    _print_instances_table,
    destroy_all_instances,
    find_best_offer,
    list_instances,
    rent_instance,
    validate_gpu_names,
)

logger = logging.getLogger("CLU")


class ClusterManager:
    """Manages the entire cluster of instances."""

    def __init__(
        self,
        job_manager: JobManager,
        ssh_key_files: list[str],
        target_instances: int = 2,
        image_name: str = "pytorch/pytorch",
        max_price_per_hour: float = 0.06,
        allowed_gpus: list[str] | None = None,
        discovery_interval: int = 30,
    ):
        self.job_manager = job_manager
        self.ssh_key_files = ssh_key_files
        self.target_instances = target_instances
        self.image_name = image_name
        self.max_price_per_hour = max_price_per_hour
        self.allowed_gpus = validate_gpu_names(allowed_gpus or RTX_30_PLUS)
        self.discovery_interval = discovery_interval

        # Log validated GPU list
        logger.info(
            f"Validated {len(self.allowed_gpus)} allowed GPUs: {self.allowed_gpus[:5]}{'...' if len(self.allowed_gpus) > 5 else ''}"
        )

        # Instance management
        self.instance_managers: dict[str, InstanceManager] = {}
        self.instance_tasks: dict[str, asyncio.Task] = {}

        # Discovery state
        self.last_discovery_time: float = 0.0
        self.next_discovery_time: float = 0.0
        self._last_good_hosts: list[str] = []

        # Global state
        self.running = False

        # Statistics tracking
        self.stats = {
            "instances_added": 0,
            "instances_removed": 0,
            "instances_rented": 0,
            "jobs_assigned": 0,
            "jobs_returned": 0,
            "outputs_downloaded": 0,
        }

        # Status table timing
        self.last_status_display = 0.0

    def update_stat(self, stat_name: str, increment: int = 1) -> None:
        """Update a statistics counter."""
        if stat_name in self.stats:
            self.stats[stat_name] += increment

    async def destroy_all_instances(self, max_wait_time: int = 300) -> bool:
        """Destroy all rented instances using vast_api."""
        return destroy_all_instances(max_wait_time)

    async def rent_instances_if_needed(self) -> int:
        """Rent additional instances if we have fewer than target_instances."""
        current_count = len(self.instance_managers)
        needed = self.target_instances - current_count

        if needed <= 0:
            return 0

        logger.info(
            f"Need {needed} more instances (current: {current_count}, target: {self.target_instances})"
        )

        rented_count = 0
        for i in range(needed):
            try:
                logger.info(f"Renting instance {i+1}/{needed}...")

                # Find best offer
                best_offer = find_best_offer(
                    gpu_list=self.allowed_gpus,
                    max_price=self.max_price_per_hour,
                    min_gpu_ram=8,
                    sort_by_price=True,
                )

                if not best_offer:
                    logger.warning(
                        f"No suitable instances found for ${self.max_price_per_hour}/hr"
                    )
                    break

                logger.info(
                    f"Found offer: {best_offer['gpu_name']} at ${best_offer['price_per_hour']:.3f}/hr"
                )

                # Rent the instance
                rental_result = rent_instance(str(best_offer["id"]))
                if rental_result and rental_result.get("status") == "rented":
                    logger.info(
                        f"Successfully rented instance: {rental_result['instance_id']}"
                    )
                    rented_count += 1
                    self.stats["instances_rented"] += 1
                else:
                    logger.error(f"Failed to rent instance: {rental_result}")

            except Exception as e:
                logger.error(f"Error renting instance {i+1}: {e}")

        logger.info(f"Rented {rented_count} new instances")
        return rented_count

    def _print_instances_table(self, instances: dict[str, dict[str, Any]]) -> None:
        """Print instances using vast_api table function."""
        # Convert to the format expected by vast_api
        vast_instances = []
        for instance_id, instance_info in instances.items():
            vast_instances.append(
                {
                    "id": instance_id,
                    "gpu_name": instance_info.get("gpu_name", "Unknown"),
                    "price_per_hour": instance_info.get("price_per_hour", 0.0),
                    "reliability": instance_info.get("reliability", 0.0),
                    "location": instance_info.get("location", "Unknown"),
                    "status": instance_info.get("status", "unknown"),
                    "cpu_cores": instance_info.get("cpu_cores", 0),
                    "gpu_ram": instance_info.get("gpu_ram", 0),
                    "cpu_util": instance_info.get("cpu_util", 0.0),
                    "gpu_util": instance_info.get("gpu_util", 0.0),
                    "gpu_temp": instance_info.get("gpu_temp", 0.0),
                    "status_msg": instance_info.get("status_msg", ""),
                    "image_runtype": instance_info.get("image_runtype", ""),
                    "image_uuid": instance_info.get("image_uuid", ""),
                }
            )
        _print_instances_table(vast_instances)

    async def get_instance_list(self) -> dict[str, dict[str, Any]]:
        """Query API for all instances."""
        try:
            # Get instances using the clean API
            instances_data = list_instances()

            if not instances_data:
                logger.warning("No instances found, using last known hosts")
                return self._parse_hosts(self._last_good_hosts)

            # Convert to the expected format
            instances = {}
            parsed_count = 0
            for item in instances_data:
                instance_info = self._parse_instance_info(item)
                if instance_info:
                    instances[instance_info["id"]] = instance_info
                    parsed_count += 1

            logger.debug(f"Successfully parsed {parsed_count} instances from API")
            self._last_good_hosts = [info["host_spec"] for info in instances.values()]
            return instances

        except Exception as e:
            logger.error(f"Failed to query instance list: {e}")
            return self._parse_hosts(self._last_good_hosts)

    def _parse_instance_info(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Parse instance information from vast CLI output."""
        try:
            # Extract basic info
            ip = item.get("ssh_host") or item.get("public_ipaddr")
            port = item.get("ssh_port", 22)

            if not ip:
                return None

            # Check if instance is running
            status = item.get(
                "actual_status",
                item.get("cur_state", item.get("intended_status", "unknown")),
            )
            if status.lower() != "running":
                return None

            # Create instance info
            instance_id = str(item.get("id", f"{ip}:{port}"))
            host_spec = f"root@{ip}:{port}"

            return {
                "id": instance_id,
                "host": ip,
                "port": int(port),
                "host_spec": host_spec,
                "status": "running",
                "api_data": item,
            }

        except Exception as e:
            logger.error(f"Failed to parse instance info: {e}")
            return None

    def _parse_hosts(self, host_specs: list[str]) -> dict[str, dict[str, Any]]:
        """Parse host specifications into instance info."""
        instances = {}
        for i, host_spec in enumerate(host_specs):
            try:
                # Parse host_spec like "root@ip:port"
                if "@" in host_spec and ":" in host_spec:
                    user, rest = host_spec.split("@", 1)
                    host, port_str = rest.rsplit(":", 1)
                    port = int(port_str)
                else:
                    continue

                instance_id = f"instance_{i}_{host}_{port}"
                instances[instance_id] = {
                    "id": instance_id,
                    "host": host,
                    "port": port,
                    "host_spec": host_spec,
                    "status": "running",
                    "api_data": {},
                }
            except Exception:
                continue

        return instances

    async def add_instance_manager(
        self, instance_id: str, instance_info: dict[str, Any]
    ) -> None:
        """Create new InstanceManager for discovered instance."""
        if instance_id in self.instance_managers:
            return

        try:
            instance_manager = InstanceManager(
                instance_id=instance_id,
                host=instance_info["host"],
                port=instance_info["port"],
                job_manager=self.job_manager,
                ssh_key_files=self.ssh_key_files,
                cluster_manager=self,
            )

            # Update API state information
            instance_manager.update_api_state(instance_info.get("api_data", {}))

            self.instance_managers[instance_id] = instance_manager

            # Start handle loop task
            task = asyncio.create_task(self._instance_handle_loop(instance_manager))
            self.instance_tasks[instance_id] = task

            # Update statistics
            self.stats["instances_added"] += 1

            # Enhanced logging
            host_port = f"{instance_info['host']}:{instance_info['port']}"
            logger.info(f"🆕 NEW INSTANCE: {host_port}")
            logger.info(f"📊 Total instances: {len(self.instance_managers)}")

        except Exception as e:
            logger.error(f"Failed to add instance manager for {instance_id}: {e}")

    async def remove_instance_manager(self, instance_id: str) -> None:
        """Clean up InstanceManager for terminated instance."""
        if instance_id in self.instance_tasks:
            task = self.instance_tasks.pop(instance_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if instance_id in self.instance_managers:
            instance_manager = self.instance_managers.pop(instance_id)
            await instance_manager.close()

            # Update statistics
            self.stats["instances_removed"] += 1

            # Enhanced logging
            host_port = f"{instance_manager.host}:{instance_manager.port}"
            logger.info(f"🗑️ REMOVED INSTANCE: {host_port}")
            logger.info(f"📊 Total instances: {len(self.instance_managers)}")

    async def update_instance_manager(
        self, instance_id: str, instance_info: dict[str, Any]
    ) -> None:
        """Update InstanceManager with latest API data."""
        if instance_id in self.instance_managers:
            # Update API state information
            instance_manager = self.instance_managers[instance_id]
            instance_manager.update_api_state(instance_info.get("api_data", {}))
            logger.debug(f"Updated instance manager for {instance_id}")

    async def _instance_handle_loop(self, instance_manager: InstanceManager) -> None:
        """Handle loop for a single instance."""
        try:
            while self.running:
                await instance_manager.handle()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                f"Instance handle loop error for {instance_manager.instance_id}: {e}"
            )
        finally:
            await instance_manager.close()

    async def print_status_table(self) -> None:
        """Print cluster status table."""
        if not self.instance_managers:
            print("\n📊 CLUSTER STATUS: No instances")
            return

        print(f"\n📊 CLUSTER STATUS ({len(self.instance_managers)} instances)")

        # Print statistics
        print(f"\n📈 STATISTICS:")
        print(f"  • Instances added: {self.stats['instances_added']}")
        print(f"  • Instances removed: {self.stats['instances_removed']}")
        print(f"  • Instances rented: {self.stats['instances_rented']}")
        print(f"  • Jobs assigned: {self.stats['jobs_assigned']}")
        print(f"  • Jobs returned: {self.stats['jobs_returned']}")
        print(f"  • Outputs downloaded: {self.stats['outputs_downloaded']}")

    async def handle_cluster(self) -> None:
        """Global cluster manager - runs every 30 seconds."""
        try:
            # 1. Query API for all instances
            api_instances = await self.get_instance_list()

            # 2. Compare with current InstanceManagers
            current_instances = set(self.instance_managers.keys())
            api_instance_ids = set(api_instances.keys())

            # 3. Add new instances
            for instance_id in api_instance_ids - current_instances:
                await self.add_instance_manager(instance_id, api_instances[instance_id])

            # 4. Remove terminated instances
            for instance_id in current_instances - api_instance_ids:
                await self.remove_instance_manager(instance_id)

            # 5. Update existing instances
            for instance_id in current_instances & api_instance_ids:
                await self.update_instance_manager(
                    instance_id, api_instances[instance_id]
                )

            # 6. Rent additional instances if needed
            await self.rent_instances_if_needed()

            # 7. Update global state
            await self.update_global_state()

        except Exception as e:
            logger.error(f"Cluster handle error: {e}")

    async def update_global_state(self) -> None:
        """Update global cluster state."""
        # This could include metrics, logging, etc.
        active_instances = len(self.instance_managers)
        job_stats = self.job_manager.get_job_stats()

        logger.info(f"Cluster state: {active_instances} instances, {job_stats}")

    async def start(self) -> None:
        """Start the cluster manager."""
        self.running = True
        logger.info("Starting cluster manager")

        try:
            while self.running:
                await self.handle_cluster()

                # Print status table every 10 seconds
                current_time = time.time()
                if current_time - self.last_status_display >= 10.0:
                    await self.print_status_table()
                    self.last_status_display = current_time

                # Wait for next discovery cycle
                now = time.time()
                if now < self.next_discovery_time:
                    await asyncio.sleep(self.next_discovery_time - now)

                self.last_discovery_time = time.time()
                self.next_discovery_time = (
                    self.last_discovery_time + self.discovery_interval
                )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Cluster manager error: {e}")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the cluster manager and cleanup all resources."""
        if not self.running:
            logger.debug("Cluster manager already stopped, skipping")
            return

        self.running = False
        logger.info("Stopping cluster manager")

        # Cancel all instance tasks
        for task in self.instance_tasks.values():
            task.cancel()

        # Wait for all tasks to complete
        if self.instance_tasks:
            await asyncio.gather(*self.instance_tasks.values(), return_exceptions=True)

        # Close all instance managers
        for instance_manager in self.instance_managers.values():
            await instance_manager.close()

        self.instance_managers.clear()
        self.instance_tasks.clear()

        logger.info("Cluster manager stopped")

    def get_cluster_stats(self) -> dict[str, Any]:
        """Get cluster statistics."""
        return {
            "instances": len(self.instance_managers),
            "jobs": self.job_manager.get_job_stats(),
            "running": self.running,
        }
