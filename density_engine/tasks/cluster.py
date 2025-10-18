"""
Celery tasks for cluster lifecycle management.

This module defines Celery tasks for renting, destroying, and managing
the lifecycle of cluster instances on Vast.ai.
"""

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from celery import current_task
from celery.exceptions import Retry

if TYPE_CHECKING:
    from celery import Task

from ..types import ClusterResult, ClusterStatus, InstanceResult, TaskResult
from .celery_app import app
from .instances import prepare_instance

logger = logging.getLogger(__name__)


@app.task(bind=True, max_retries=3, default_retry_delay=120)  # type: ignore[misc]
def rent_new_instance(
    self: "Task",
    gpu_type: str | list[str] = "RTX 4090",
    max_price_per_hour: float = 1.0,
    min_gpu_ram: int = 24,
    min_disk_space: int = 50,
    min_reliability: float = 0.95,
    preferred_countries: list[str] | None = None,
    excluded_countries: list[str] | None = None,
    min_dlperf: float | None = None,
    sort_by_performance: bool = False,
) -> InstanceResult:
    """
    Rent a new instance from Vast.ai marketplace with sophisticated filtering.

    Args:
        gpu_type: Type of GPU required (e.g., 'RTX 4090', 'A100') or list of GPU types
        max_price_per_hour: Maximum price per hour willing to pay
        min_gpu_ram: Minimum GPU RAM required (GB)
        min_disk_space: Minimum disk space required (GB)
        min_reliability: Minimum reliability score (0-1)
        preferred_countries: List of preferred country codes (e.g., ['US', 'TW'])
        excluded_countries: List of excluded country codes (e.g., ['CN', 'VN'])
        min_dlperf: Minimum deep learning performance score
        sort_by_performance: If True, sort by performance; if False, sort by price

    Returns:
        InstanceRentalResult containing rental information
    """
    try:
        logger.info(
            f"Renting new instance: GPU={gpu_type}, max_price=${max_price_per_hour}/hr, "
            f"min_reliability={min_reliability}"
        )

        # Import the enhanced marketplace
        from ..cluster.marketplace import MachineCriteria, SortOrder, VastMarketplace

        # Build search criteria
        criteria = MachineCriteria(
            gpu_name=gpu_type,
            max_price_per_hour=max_price_per_hour,
            min_gpu_ram=min_gpu_ram,
            min_disk_space=min_disk_space,
            min_reliability=min_reliability,
            preferred_countries=preferred_countries,
            excluded_countries=excluded_countries,
            min_dlperf=min_dlperf,
            verified_only=True,
            rentable_only=True,
        )

        # Choose sort order
        sort_order = (
            SortOrder.PERFORMANCE_DESC if sort_by_performance else SortOrder.PRICE_ASC
        )

        # Search for best offer
        marketplace = VastMarketplace()
        best_offer = marketplace.find_best_offer(criteria, sort_order, limit=20)

        if not best_offer:
            logger.warning("No suitable machines found matching criteria")
            return InstanceResult(
                status="FAILED",
                instance_id=None,
                message="No suitable machines found matching criteria",
                error="No offers available",
            )

        # Extract offer details
        machine_id = best_offer.get("id", "unknown")
        gpu_name = best_offer.get("gpu_name", "Unknown")
        price = best_offer.get("dph_total", 0.0)
        reliability = best_offer.get("reliability", 0.0)
        location = best_offer.get("geolocation", "Unknown")

        logger.info(
            f"Selected offer: {gpu_name} ${price:.3f}/hr "
            f"(reliability={reliability:.3f}, location={location})"
        )

        # Here we would actually rent the instance using Vast.ai API
        # For now, simulate the rental process
        rental_result = {
            "instance_id": f"vast_{machine_id}",
            "machine_id": machine_id,
            "gpu_name": gpu_name,
            "price_per_hour": price,
            "reliability": reliability,
            "location": location,
            "rented_at": time.time(),
            "status": "rented",
            "ssh_host": "192.168.1.200",  # Would be provided by Vast.ai
            "ssh_port": 22,
        }

        logger.info(f"Instance rented: {rental_result}")

        # Schedule preparation for the new instance
        prepare_instance.delay(rental_result["instance_id"], rental_result)

        return InstanceResult(
            status="SUCCESS",
            instance_id=rental_result["instance_id"],
            message=f"Successfully rented instance {rental_result['instance_id']}",
            error=None,
        )

    except Exception as exc:
        logger.error(f"Failed to rent instance: {exc}")
        raise self.retry(exc=exc)


@app.task(bind=True, max_retries=3, default_retry_delay=60)  # type: ignore[misc]
def destroy_instance(
    self: "Task",
    instance_id: str,
    reason: str | None = None,
) -> InstanceResult:
    """
    Destroy an instance and clean up resources.

    Args:
        instance_id: Unique instance identifier
        reason: Reason for destruction (optional)

    Returns:
        Dict containing destruction result
    """
    try:
        logger.info(f"Destroying instance {instance_id}, reason: {reason}")

        # Import the Vast API
        from ..cluster.vast_api import destroy_instance as vast_destroy_instance

        # Destroy the instance
        success = vast_destroy_instance(instance_id)

        if not success:
            logger.error(f"Failed to destroy instance {instance_id}")
            return InstanceResult(
                status="FAILED",
                instance_id=instance_id,
                message=f"Failed to destroy instance {instance_id}",
                error="Destruction failed",
            )

        logger.info(f"Successfully destroyed instance {instance_id}")
        return InstanceResult(
            status="SUCCESS",
            instance_id=instance_id,
            message=f"Successfully destroyed instance {instance_id}",
            error=None,
        )

    except Exception as exc:
        logger.error(f"Failed to destroy instance {instance_id}: {exc}")
        raise self.retry(exc=exc)


@app.task  # type: ignore[misc]
def get_market_offers(
    gpu_type: str | list[str] = "RTX 4090",
    max_price: float = 1.0,
    min_gpu_ram: int = 24,
    min_reliability: float = 0.95,
    preferred_countries: list[str] | None = None,
    excluded_countries: list[str] | None = None,
    sort_by_performance: bool = False,
    limit: int = 50,
) -> TaskResult:
    """
    Query Vast.ai marketplace for available offers with sophisticated filtering.

    Args:
        gpu_type: Type of GPU to search for or list of GPU types
        max_price: Maximum price per hour
        min_gpu_ram: Minimum GPU RAM required
        min_reliability: Minimum reliability score
        preferred_countries: List of preferred country codes
        excluded_countries: List of excluded country codes
        sort_by_performance: If True, sort by performance; if False, sort by price
        limit: Maximum number of offers to return

    Returns:
        TaskResult containing available offers
    """
    try:
        logger.info(f"Querying market offers: GPU={gpu_type}, max_price=${max_price}")

        # Import the enhanced marketplace
        from ..cluster.marketplace import MachineCriteria, SortOrder, VastMarketplace

        # Build search criteria
        criteria = MachineCriteria(
            gpu_name=gpu_type,
            max_price_per_hour=max_price,
            min_gpu_ram=min_gpu_ram,
            verified_only=True,
            rentable_only=True,
        )

        # Search for offers
        marketplace = VastMarketplace()
        offers = marketplace.search_offers(criteria, SortOrder.PRICE_ASC, limit=50)

        # Format offers for response
        formatted_offers = []
        for offer in offers:
            formatted_offers.append(
                {
                    "offer_id": offer.get("id", "unknown"),
                    "gpu_type": offer.get("gpu_name", "Unknown"),
                    "price_per_hour": offer.get("dph_total", 0.0),
                    "gpu_ram": offer.get("gpu_ram", 0),
                    "disk_space": offer.get("disk_space", 0),
                    "availability": "available",
                    "reliability": offer.get("reliability", 0.0),
                    "location": offer.get("geolocation", "Unknown"),
                    "cpu_cores": offer.get("cpu_cores", 0),
                    "cpu_ram": offer.get("cpu_ram", 0),
                }
            )

        logger.info(f"Found {len(formatted_offers)} market offers")

        return TaskResult(
            status="SUCCESS",
            message=f"Found {len(formatted_offers)} offers",
            data={
                "offers": formatted_offers,
                "total_offers": len(formatted_offers),
                "gpu_type": gpu_type,
                "max_price": max_price,
                "min_gpu_ram": min_gpu_ram,
            },
            details={
                "offers": formatted_offers,
                "total_offers": len(formatted_offers),
                "gpu_type": gpu_type,
                "max_price": max_price,
                "min_gpu_ram": min_gpu_ram,
            },
            error=None,
        )

    except Exception as exc:
        logger.error(f"Failed to query market offers: {exc}")
        return TaskResult(
            status="FAILED",
            message="Failed to query market offers",
            data={
                "gpu_type": gpu_type,
                "max_price": max_price,
                "min_gpu_ram": min_gpu_ram,
            },
            details={
                "gpu_type": gpu_type,
                "max_price": max_price,
                "min_gpu_ram": min_gpu_ram,
            },
            error=str(exc),
        )


@app.task  # type: ignore[misc]
def rebalance_cluster(
    target_count: int,
    gpu_type: str = "RTX 4090",
    max_price_per_hour: float = 1.0,
) -> ClusterResult:
    """
    Rebalance cluster to target instance count.

    Args:
        target_count: Target number of instances
        gpu_type: Type of GPU for new instances
        max_price_per_hour: Maximum price for new instances

    Returns:
        Dict containing rebalancing result
    """
    try:
        logger.info(f"Rebalancing cluster to {target_count} instances")

        # Here we would:
        # 1. Get current instance count
        # 2. Calculate difference from target
        # 3. Rent new instances if needed
        # 4. Destroy excess instances if needed

        # For now, simulate rebalancing
        current_count = 2  # Mock current count
        difference = target_count - current_count

        if difference > 0:
            # Need more instances
            logger.info(f"Need to rent {difference} more instances")
            for i in range(difference):
                rent_new_instance.delay(gpu_type, max_price_per_hour)
        elif difference < 0:
            # Need to destroy instances
            logger.info(f"Need to destroy {-difference} instances")
            # Would get list of instances and destroy excess ones

        return ClusterResult(
            status="SUCCESS",
            message=f"Cluster rebalancing initiated: {difference:+d} instances",
            data={
                "target_count": target_count,
                "current_count": current_count,
                "difference": difference,
            },
            details={
                "target_count": target_count,
                "current_count": current_count,
                "difference": difference,
            },
            actions_taken=[
                (
                    f"Rented {difference} instances"
                    if difference > 0
                    else (
                        f"Destroyed {-difference} instances"
                        if difference < 0
                        else "No action needed"
                    )
                )
            ],
            cluster_status=None,
            error=None,
        )

    except Exception as exc:
        logger.error(f"Failed to rebalance cluster: {exc}")
        return ClusterResult(
            status="FAILED",
            message="Failed to rebalance cluster",
            data={},
            details={},
            cluster_status=None,
            actions_taken=[],
            error=str(exc),
        )


@app.task  # type: ignore[misc]
def monitor_cluster_health() -> ClusterResult:
    """
    Monitor overall cluster health (periodic task).

    Returns:
        Dict containing cluster health summary
    """
    try:
        logger.debug("Monitoring cluster health")

        # Here we would:
        # 1. Check all instances
        # 2. Monitor job queue
        # 3. Check resource usage
        # 4. Identify issues

        # For now, return mock health data
        health_summary: dict[str, int | str] = {
            "total_instances": 2,
            "healthy_instances": 2,
            "unhealthy_instances": 0,
            "jobs_running": 1,
            "jobs_pending": 5,
            "jobs_failed": 0,
            "cluster_status": "healthy",
        }

        logger.debug(f"Cluster health: {health_summary}")

        return ClusterResult(
            status="HEALTHY",
            message="Cluster health monitoring completed",
            data={
                "total_instances": health_summary["total_instances"],
                "healthy_instances": health_summary["healthy_instances"],
                "unhealthy_instances": health_summary["unhealthy_instances"],
                "jobs_running": health_summary["jobs_running"],
                "jobs_pending": health_summary["jobs_pending"],
                "jobs_failed": health_summary["jobs_failed"],
                "cluster_status": health_summary["cluster_status"],
            },
            details={
                "total_instances": health_summary["total_instances"],
                "healthy_instances": health_summary["healthy_instances"],
                "unhealthy_instances": health_summary["unhealthy_instances"],
                "jobs_running": health_summary["jobs_running"],
                "jobs_pending": health_summary["jobs_pending"],
                "jobs_failed": health_summary["jobs_failed"],
                "cluster_status": health_summary["cluster_status"],
            },
            cluster_status=None,
            actions_taken=[],
            error=None,
        )

    except Exception as exc:
        logger.error(f"Failed to monitor cluster health: {exc}")
        return ClusterResult(
            status="CRITICAL",
            message="Failed to monitor cluster health",
            data={"error": str(exc)},
            details={"error": str(exc)},
            cluster_status=None,
            actions_taken=[],
            error=str(exc),
        )


@app.task  # type: ignore[misc]
def cleanup_unused_instances() -> TaskResult:
    """
    Clean up instances that have been idle for too long.

    Returns:
        Dict containing cleanup result
    """
    try:
        logger.info("Cleaning up unused instances")

        # Here we would:
        # 1. Find instances idle for more than X hours
        # 2. Check if they have any pending jobs
        # 3. Destroy unused instances

        # For now, simulate cleanup
        cleaned_instances: list[str] = []

        logger.info(f"Cleaned up {len(cleaned_instances)} unused instances")

        return TaskResult(
            status="SUCCESS",
            message=f"Cleaned up {len(cleaned_instances)} unused instances",
            data={
                "cleaned_instances": cleaned_instances,
                "count": len(cleaned_instances),
            },
            details={
                "cleaned_instances": cleaned_instances,
                "count": len(cleaned_instances),
            },
            error=None,
        )

    except Exception as exc:
        logger.error(f"Failed to cleanup unused instances: {exc}")
        return TaskResult(
            status="FAILED",
            message="Failed to cleanup unused instances",
            data={},
            details={},
            error=str(exc),
        )


@app.task  # type: ignore[misc]
def get_cluster_status() -> ClusterResult:
    """
    Get current cluster status.

    Returns:
        Dict containing cluster status
    """
    try:
        logger.debug("Getting cluster status")

        # Here we would query Redis for cluster status
        # For now, return mock status
        cluster_status = {
            "total_instances": 2,
            "prepared_instances": 2,
            "running_jobs": 1,
            "pending_jobs": 5,
            "failed_jobs": 0,
            "cluster_load": 0.3,
            "last_updated": time.time(),
        }

        cluster_status_obj = ClusterStatus(
            total_instances=int(cluster_status["total_instances"]),
            available_instances=int(cluster_status["prepared_instances"]),
            running_instances=int(cluster_status["prepared_instances"]),
            pending_instances=0,
            errored_instances=0,
            total_jobs=int(
                cluster_status["running_jobs"]
                + cluster_status["pending_jobs"]
                + cluster_status["failed_jobs"]
            ),
            running_jobs=int(cluster_status["running_jobs"]),
            pending_jobs=int(cluster_status["pending_jobs"]),
            failed_jobs=int(cluster_status["failed_jobs"]),
            completed_jobs=0,
            message="Cluster status retrieved",
        )

        return ClusterResult(
            status="SUCCESS",
            message="Cluster status retrieved",
            data={k: v for k, v in cluster_status.items()},
            details={k: float(v) for k, v in cluster_status.items()},
            cluster_status=cluster_status_obj,
            actions_taken=[],
            error=None,
        )

    except Exception as exc:
        logger.error(f"Failed to get cluster status: {exc}")
        return ClusterResult(
            status="FAILED",
            message="Failed to get cluster status",
            data={},
            details={},
            cluster_status=None,
            actions_taken=[],
            error=str(exc),
        )


@app.task(bind=True, max_retries=3, default_retry_delay=120)  # type: ignore[misc]
def rent_instance(
    self: "Task",
    gpu_list: list[str],
    max_price_per_hour: float = 1.0,
    min_gpu_ram: int = 24,
    min_disk_space: int = 50,
    min_reliability: float = 0.95,
    preferred_countries: list[str] | None = None,
    excluded_countries: list[str] | None = None,
    min_dlperf: float | None = None,
    sort_by_performance: bool = False,
) -> InstanceResult:
    """
    Rent an instance with specified GPU list.

    Args:
        gpu_list: List of GPU names to search for
        max_price_per_hour: Maximum price per hour willing to pay
        min_gpu_ram: Minimum GPU RAM required (GB)
        min_disk_space: Minimum disk space required (GB)
        min_reliability: Minimum reliability score (0-1)
        preferred_countries: List of preferred country codes (e.g., ['US', 'TW'])
        excluded_countries: List of excluded country codes (e.g., ['CN', 'VN'])
        min_dlperf: Minimum deep learning performance score
        sort_by_performance: If True, sort by performance; if False, sort by price

    Returns:
        InstanceRentalResult containing rental information
    """
    try:
        logger.info(
            f"Renting instance with GPUs: {gpu_list[:3]}... (max_price=${max_price_per_hour}/hr, "
            f"min_reliability={min_reliability})"
        )

        # Import the Vast API
        from ..cluster.vast_api import find_best_offer, rent_instance

        # Use provided GPU list
        if not gpu_list:
            logger.warning("No GPUs specified in gpu_list")
            return InstanceResult(
                status="FAILED",
                instance_id=None,
                message="No GPUs specified in gpu_list",
                error="Empty GPU list",
            )

        logger.info(f"Searching for {len(gpu_list)} GPU types: {gpu_list}")

        # Find best offer
        best_offer: dict[str, str | int | float] | None = find_best_offer(
            gpu_list=gpu_list,
            max_price=max_price_per_hour,
            min_reliability=min_reliability,
            min_gpu_ram=min_gpu_ram,
            sort_by_price=not sort_by_performance,
        )

        if not best_offer:
            logger.warning(f"No suitable machines found for GPU list: {gpu_list}")
            return InstanceResult(
                status="FAILED",
                instance_id=None,
                message=f"No suitable machines found for GPU list: {gpu_list}",
                error="No offers available",
            )

        logger.info(
            f"Selected offer: {best_offer['gpu_name']} ${best_offer['price_per_hour']:.3f}/hr "
            f"(reliability={best_offer['reliability']:.3f}, location={best_offer['location']})"
        )

        # Rent the instance
        rental_result = rent_instance(str(best_offer["id"]))

        if not rental_result:
            logger.error(f"Failed to rent instance {best_offer['id']}")
            return InstanceResult(
                status="FAILED",
                instance_id=None,
                message=f"Failed to rent instance {best_offer['id']}",
                error="Rental failed",
            )

        logger.info(f"Instance rented: {rental_result}")

        # Schedule preparation for the new instance
        prepare_instance.delay(rental_result["instance_id"], rental_result)

        return InstanceResult(
            status="SUCCESS",
            instance_id=rental_result["instance_id"],
            message=f"Successfully rented instance {rental_result['instance_id']} with {best_offer['gpu_name']}",
            error=None,
        )

    except Exception as exc:
        logger.error(f"Failed to rent instance: {exc}")
        raise self.retry(exc=exc)
