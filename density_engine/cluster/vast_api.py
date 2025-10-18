import json
import logging
import subprocess
import time
from typing import Any, Dict, List, Optional, Union

from .gpu_names import ALL_GPU_NAMES

logger = logging.getLogger(__name__)


def _run_vast_command(
    command: list[str], timeout: int = 30
) -> dict[str, Any] | list[Any] | None:
    """Execute a Vast.ai CLI command and return parsed JSON."""
    try:
        logger.debug(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            logger.error(f"Command failed: {' '.join(command)}")
            logger.error(f"Error: {result.stderr}")
            return None

        if not result.stdout.strip():
            return None

        return json.loads(result.stdout)  # type: ignore[no-any-return]

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {' '.join(command)}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.debug(f"Raw output: {result.stdout[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Command error: {e}")
        return None


def _build_gpu_query(gpu_list: list[str]) -> str:
    """Build GPU filter query string."""
    if not gpu_list:
        return ""

    gpu_names = [name.replace(" ", "_") for name in gpu_list]
    return (
        f"gpu_name in [{','.join(gpu_names)}]"
        if len(gpu_names) > 1
        else f"gpu_name={gpu_names[0]}"
    )


def _parse_instance_data(item: dict) -> dict[str, str | int | float]:
    """Parse raw instance data into standardized format."""
    return {
        "id": str(item.get("id", "")),
        "gpu_name": item.get("gpu_name", "Unknown"),
        "price_per_hour": float(item.get("dph_total", item.get("dph", 0.0))),
        "reliability": float(item.get("reliability", item.get("reliability2", 0.0))),
        "location": item.get("geolocation", "Unknown"),
        "ssh_host": item.get("ssh_host", ""),
        "ssh_port": int(item.get("ssh_port", 22)),
        "status": item.get(
            "actual_status",
            item.get("cur_state", item.get("intended_status", "unknown")),
        ),
        "cpu_cores": int(item.get("cpu_cores", 0)),
        "gpu_ram": int(item.get("gpu_ram", 0)),
        "disk_space": int(item.get("disk_space", 0)),
        "cpu_util": float(item.get("cpu_util", 0.0)),
        "gpu_util": float(item.get("gpu_util", 0.0)) if item.get("gpu_util") else 0.0,
        "gpu_temp": float(item.get("gpu_temp", 0.0)) if item.get("gpu_temp") else 0.0,
        "status_msg": item.get("status_msg", ""),
        "image_runtype": item.get("image_runtype", ""),
        "image_uuid": item.get("image_uuid", ""),
    }


def _print_instances_table(instances: list[dict[str, str | int | float]]) -> None:
    """Print instances in a nice table format."""
    if not instances:
        print("No instances found.")
        return

    # Prepare table data
    table_data = []
    for instance in instances:
        # Format status with additional info
        status_display = str(instance["status"])
        if instance["status_msg"]:
            status_display += f" ({instance['status_msg']})"

        # Format utilization info
        cpu_util_val = instance["cpu_util"]
        cpu_util = (
            f"{cpu_util_val:.1f}%"
            if isinstance(cpu_util_val, (int, float)) and cpu_util_val > 0
            else "N/A"
        )

        gpu_util_val = instance["gpu_util"]
        gpu_util = (
            f"{gpu_util_val:.1f}%"
            if isinstance(gpu_util_val, (int, float)) and gpu_util_val > 0
            else "N/A"
        )

        gpu_temp_val = instance["gpu_temp"]
        gpu_temp = (
            f"{gpu_temp_val:.0f}°C"
            if isinstance(gpu_temp_val, (int, float)) and gpu_temp_val > 0
            else "N/A"
        )

        table_data.append(
            [
                instance["id"],
                instance["gpu_name"],
                f"${instance['price_per_hour']:.3f}/hr",
                f"{instance['reliability']:.1%}",
                instance["location"],
                status_display,
                f"{instance['cpu_cores']} cores",
                f"{instance['gpu_ram']}GB",
                cpu_util,
                gpu_util,
                gpu_temp,
            ]
        )

    headers = [
        "ID",
        "GPU",
        "Price",
        "Reliability",
        "Location",
        "Status",
        "CPU",
        "RAM",
        "CPU%",
        "GPU%",
        "Temp",
    ]

    # Use tabulate if available, otherwise simple table
    from tabulate import tabulate  # type: ignore[import-untyped]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def search_offers(
    gpu_list: list[str],
    max_price: float = 0.06,
    min_reliability: float = 0.95,
    min_gpu_ram: int = 8,
    limit: int = 50,
) -> list[dict[str, str | int | float]]:
    """
    Search Vast.ai for machine offers.

    Args:
        gpu_list: List of GPU names to search for
        max_price: Maximum price per hour
        min_reliability: Minimum reliability score
        min_gpu_ram: Minimum GPU RAM in GB
        limit: Maximum number of results

    Returns:
        List of offer dictionaries
    """
    # Build query string
    query_parts = []

    # GPU filter
    gpu_query = _build_gpu_query(gpu_list)
    if gpu_query:
        query_parts.append(gpu_query)

    # Other filters
    query_parts.extend(
        [
            f"dph_total<={max_price}",
            f"reliability>={min_reliability}",
            f"gpu_ram>={min_gpu_ram}",
            "rented=False",
            "verified=True",
        ]
    )

    query = " ".join(query_parts)

    # Execute search
    command = ["python", "vast.py", "search", "offers", "--raw", query]
    data = _run_vast_command(command)

    if not data:
        return []

    # Parse results - data should be a list of offers
    if isinstance(data, list):
        offers_data = data
    else:  # data is dict
        offers_data = [data]

    offers = []
    for item in offers_data:
        if isinstance(item, dict):
            offer = _parse_instance_data(item)
            offers.append(offer)

    logger.info(f"Found {len(offers)} offers")
    return offers[:limit]


def find_best_offer(
    gpu_list: list[str],
    max_price: float = 0.06,
    min_reliability: float = 0.95,
    min_gpu_ram: int = 8,
    sort_by_price: bool = True,
) -> dict[str, str | int | float] | None:
    """
    Find the best offer from Vast.ai.

    Args:
        gpu_list: List of GPU names to search for
        max_price: Maximum price per hour
        min_reliability: Minimum reliability score
        min_gpu_ram: Minimum GPU RAM in GB
        sort_by_price: If True, sort by price; if False, sort by reliability

    Returns:
        Best offer dictionary or None if no offers found
    """
    offers = search_offers(
        gpu_list=gpu_list,
        max_price=max_price,
        min_reliability=min_reliability,
        min_gpu_ram=min_gpu_ram,
        limit=50,
    )

    if not offers:
        return None

    # Sort offers
    if sort_by_price:
        offers.sort(key=lambda x: x["price_per_hour"])
    else:
        offers.sort(key=lambda x: x["reliability"], reverse=True)

    return offers[0]


def rent_instance(
    offer_id: str, image: str = "vastai/pytorch", disk: int = 20
) -> dict[str, str] | None:
    """
    Rent a specific instance from Vast.ai.

    Args:
        offer_id: Vast.ai offer ID
        image: Docker image to use, default is vastai/pytorch
        disk: size of local disk partition in GB, default is 20GB
    Returns:
        Dict with rental details or None if failed
    """
    command = [
        "python",
        "vast.py",
        "create",
        "instance",
        str(offer_id),
        "--image",
        image,
        "--disk",
        str(disk),
        "--ssh",
        "--raw",
    ]

    logger.info(f"Renting instance {offer_id}")
    data = _run_vast_command(command, timeout=60)

    if not data:
        return None

    if isinstance(data, dict) and data.get("success") and "new_contract" in data:
        instance_id = str(data["new_contract"])
        logger.info(f"Successfully rented instance: {instance_id}")
        return {
            "instance_id": instance_id,
            "status": "rented",
            "message": f"Instance {instance_id} rented successfully",
        }
    else:
        logger.error(f"Rental failed: {data}")
        return None


def destroy_instance(instance_id: str) -> bool:
    """
    Destroy a rented instance.

    Args:
        instance_id: Vast.ai instance ID

    Returns:
        True if successful, False otherwise
    """
    command = ["python", "vast.py", "destroy", "instance", str(instance_id), "--raw"]

    logger.info(f"Destroying instance {instance_id}")
    data = _run_vast_command(command)

    if data is not None:
        logger.info(f"Successfully destroyed instance: {instance_id}")
        return True
    else:
        logger.error(f"Failed to destroy instance: {instance_id}")
        return False


def list_instances() -> list[dict[str, str | int | float]]:
    """
    List all rented instances from Vast.ai.

    Returns:
        List of instance dictionaries with standardized fields
    """
    command = ["python", "vast.py", "show", "instances", "--raw"]

    logger.info("Listing Vast.ai instances")
    data = _run_vast_command(command)

    if not data:
        return []

    # Parse results - data should be a list of instances
    if isinstance(data, list):
        instances_data = data
    else:  # data is dict
        instances_data = [data]

    instances = []
    for item in instances_data:
        if isinstance(item, dict):
            instance = _parse_instance_data(item)
            instances.append(instance)

    logger.info(f"Found {len(instances)} instances")
    return instances


def destroy_all_instances(max_wait_time: int = 300) -> bool:
    """
    Destroy all rented instances with monitoring.

    Args:
        max_wait_time: Maximum time to wait for all instances to be destroyed (seconds)

    Returns:
        True if all instances were successfully destroyed, False otherwise
    """
    logger.info("🗑️  Starting destruction of all instances...")

    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        instances = list_instances()

        if not instances:
            logger.info("✅ All instances have been destroyed!")
            return True

        logger.info(f"📊 Found {len(instances)} instances to destroy:")
        _print_instances_table(instances)

        # Destroy each instance
        for instance in instances:
            instance_id = str(instance["id"])
            logger.info(f"🗑️  Destroying instance {instance_id}...")

            success = destroy_instance(instance_id)
            if success:
                logger.info(f"✅ Successfully destroyed instance {instance_id}")
            else:
                logger.warning(f"❌ Failed to destroy instance {instance_id}")

        logger.info(f"⏳ Waiting 30 seconds before next check...")
        time.sleep(30)

    # Final check
    final_instances = list_instances()
    if not final_instances:
        logger.info("✅ All instances have been destroyed!")
        return True
    else:
        logger.warning(
            f"⚠️  {len(final_instances)} instances still remain after {max_wait_time}s timeout"
        )
        return False


def validate_gpu_names(gpu_list: list[str]) -> list[str]:
    """Validate GPU names against known Vast.ai GPU types."""
    validated = []
    for gpu in gpu_list:
        # Check both original format and underscore format
        if gpu in ALL_GPU_NAMES:
            validated.append(gpu)
        elif gpu.replace(" ", "_") in ALL_GPU_NAMES:
            validated.append(gpu.replace(" ", "_"))
    return validated
