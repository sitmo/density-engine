"""
Instance discovery for the vast.ai automation system.
"""

import ast
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.state import InstanceState, InstanceStatus
from ..utils.exceptions import InstanceError
from ..utils.logging import get_logger, log_function_call

logger = get_logger(__name__)


@dataclass
class InstanceInfo:
    """Instance information from vast.ai API."""

    contract_id: int
    machine_id: int
    gpu_name: str
    price_per_hour: float
    ssh_host: str
    ssh_port: int
    status: str
    public_ipaddr: str
    ports: dict[str, Any]
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@log_function_call
def fetch_instances_from_api() -> list[dict[str, Any]]:
    """Fetch instances from vast.ai API using CLI."""
    try:
        logger.debug("Fetching instances from vast.ai API")

        # Use vast.py CLI to get instances
        result = subprocess.run(
            ["python3", "vast.py", "show", "instances", "--raw"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.error(f"Failed to fetch instances: {result.stderr}")
            return []

        # Parse the output
        output = result.stdout.strip()
        if not output:
            logger.warning("No instances returned from API")
            return []

        try:
            # Try to parse as JSON first
            instances_data = json.loads(output)
            if isinstance(instances_data, dict) and "instances" in instances_data:
                instances: list[dict[str, Any]] = instances_data["instances"]
            elif isinstance(instances_data, list):
                instances: list[dict[str, Any]] = instances_data
            else:
                instances: list[dict[str, Any]] = [instances_data]
        except json.JSONDecodeError:
            try:
                # Try to parse as Python literal (ast.literal_eval)
                instances: list[dict[str, Any]] = ast.literal_eval(output)
                if not isinstance(instances, list):
                    instances = [instances]
            except (ValueError, SyntaxError) as e:
                logger.error(f"Failed to parse instances output: {e}")
                logger.debug(f"Raw output: {output}")
                return []

        logger.info(f"Fetched {len(instances)} instances from API")
        return instances  # type: ignore[return]

    except subprocess.TimeoutExpired:
        logger.error("Timeout while fetching instances from API")
        return []
    except Exception as e:
        logger.error(f"Failed to fetch instances from API: {e}")
        return []


@log_function_call
def parse_instance_data(raw_data: dict[str, Any]) -> InstanceInfo:
    """Parse raw instance data into InstanceInfo."""
    try:
        # Extract SSH connection details
        ssh_host = raw_data.get("public_ipaddr", "")
        ports = raw_data.get("ports", {})
        ssh_port = 0

        if "22/tcp" in ports and ports["22/tcp"]:
            ssh_port = int(ports["22/tcp"][0]["HostPort"])

        if not ssh_port:
            ssh_port = raw_data.get("ssh_port", 0)

        instance_info = InstanceInfo(
            contract_id=raw_data.get("id", 0),
            machine_id=raw_data.get("machine_id", 0),
            gpu_name=raw_data.get("gpu_name", "Unknown"),
            price_per_hour=raw_data.get("price", 0.0),
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            status=raw_data.get("status", "unknown"),
            public_ipaddr=raw_data.get("public_ipaddr", ""),
            ports=ports,
            metadata=raw_data,
        )

        logger.debug(
            f"Parsed instance {instance_info.contract_id}: {ssh_host}:{ssh_port}"
        )
        return instance_info

    except Exception as e:
        logger.error(f"Failed to parse instance data: {e}")
        raise InstanceError(f"Failed to parse instance data: {e}")


@log_function_call
def filter_instances_by_criteria(
    instances: list[InstanceInfo], criteria: dict[str, Any]
) -> list[InstanceInfo]:
    """Filter instances by criteria."""
    try:
        filtered_instances = []

        for instance in instances:
            # Check status
            if "status" in criteria and instance.status != criteria["status"]:
                continue

            # Check price
            if (
                "max_price" in criteria
                and instance.price_per_hour > criteria["max_price"]
            ):
                continue

            # Check GPU
            if "gpu_name" in criteria and criteria["gpu_name"] not in instance.gpu_name:
                continue

            # Check SSH availability
            if "require_ssh" in criteria and criteria["require_ssh"]:
                if not instance.ssh_host or not instance.ssh_port:
                    continue

            filtered_instances.append(instance)

        logger.debug(
            f"Filtered {len(instances)} instances to {len(filtered_instances)}"
        )
        return filtered_instances

    except Exception as e:
        logger.error(f"Failed to filter instances: {e}")
        raise InstanceError(f"Failed to filter instances: {e}")


@log_function_call
def discover_active_instances() -> list[InstanceInfo]:
    """Discover all active instances."""
    try:
        raw_instances = fetch_instances_from_api()
        instances: list[InstanceInfo] = []

        for raw_data in raw_instances:
            try:
                instance = parse_instance_data(raw_data)
                instances.append(instance)
            except Exception as e:
                logger.warning(f"Failed to parse instance data: {e}")
                continue

        logger.info(f"Discovered {len(instances)} active instances")
        return instances

    except Exception as e:
        logger.error(f"Failed to discover active instances: {e}")
        raise InstanceError(f"Failed to discover active instances: {e}")


@log_function_call
def discover_new_instances() -> list[InstanceInfo]:
    """Discover new instances (not yet in state)."""
    try:
        from ..core.state import get_state_manager

        active_instances = discover_active_instances()
        state_manager = get_state_manager()

        new_instances = []
        for instance in active_instances:
            instance_id = str(instance.contract_id)
            if instance_id not in state_manager.instances:
                new_instances.append(instance)
                logger.info(f"Discovered new instance: {instance_id}")

        logger.info(f"Discovered {len(new_instances)} new instances")
        return new_instances

    except Exception as e:
        logger.error(f"Failed to discover new instances: {e}")
        raise InstanceError(f"Failed to discover new instances: {e}")


@log_function_call
def get_instance_by_id(instance_id: str) -> InstanceInfo | None:
    """Get instance by ID."""
    try:
        from ..core.state import get_state_manager

        state_manager = get_state_manager()
        if instance_id in state_manager.instances:
            instance_state = state_manager.instances[instance_id]
            return InstanceInfo(
                contract_id=int(instance_id),
                machine_id=instance_state.metadata.get("machine_id", 0),
                gpu_name=instance_state.metadata.get("gpu_name", "Unknown"),
                price_per_hour=instance_state.metadata.get("price_per_hour", 0.0),
                ssh_host=instance_state.ssh_host,
                ssh_port=instance_state.ssh_port,
                status=instance_state.status.value,
                public_ipaddr=instance_state.ssh_host,
                ports={},
                metadata=instance_state.metadata,
            )

        return None

    except Exception as e:
        logger.error(f"Failed to get instance by ID {instance_id}: {e}")
        return None
