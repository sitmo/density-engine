#!/usr/bin/env python3
"""
Enhanced Vast.ai marketplace integration with sophisticated filtering and sorting.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .gpu_names import ALL_GPU_NAMES, RTX_30_PLUS, RTX_40_PLUS, RTX_50_ONLY

logger = logging.getLogger(__name__)


def get_gpu_name_variants(gpu_name: str) -> list[str]:
    """Get possible variants of a GPU name for API compatibility."""
    variants = [gpu_name]
    # Add underscore variant
    if " " in gpu_name:
        variants.append(gpu_name.replace(" ", "_"))
    # Add space variant
    if "_" in gpu_name:
        variants.append(gpu_name.replace("_", " "))
    return variants


def validate_gpu_name(gpu_name: str) -> bool:
    """Validate if a GPU name is recognized."""
    return gpu_name.upper() in [name.upper() for name in ALL_GPU_NAMES]


def get_high_performance_gpus() -> list[str]:
    """Get list of high-performance GPU names."""
    return ["RTX 4090", "A100", "H100", "RTX 4080", "RTX 4070 Ti"]


def get_budget_gpus() -> list[str]:
    """Get list of budget-friendly GPU names."""
    return ["RTX 3060", "RTX 3070", "RTX 3080", "RTX 4060", "RTX 4070"]


def get_multi_gpu_candidates() -> list[str]:
    """Get list of GPUs suitable for multi-GPU setups."""
    return ["RTX 4090", "A100", "H100", "RTX 4080"]


def get_rtx_series_and_above(series: str | int) -> list[str]:
    """Get RTX GPUs of specified series and above."""
    series_num = int(series) if isinstance(series, str) else series
    if series_num >= 40:
        return RTX_40_PLUS
    elif series_num >= 30:
        return RTX_30_PLUS
    else:
        return ALL_GPU_NAMES


def get_rtx_performance_tier(tier: str) -> list[str]:
    """Get RTX GPUs by performance tier."""
    if tier.lower() == "high":
        return ["RTX 4090", "RTX 4080"]
    elif tier.lower() == "mid":
        return ["RTX 4070", "RTX 4070 Ti", "RTX 3080"]
    elif tier.lower() == "low":
        return ["RTX 4060", "RTX 3060", "RTX 3070"]
    else:
        return []


def get_rtx_by_memory(memory_gb: int, max_gb: int | None = None) -> list[str]:
    """Get RTX GPUs with at least specified memory."""
    gpus = []
    for gpu in ALL_GPU_NAMES:
        if "RTX" in gpu.upper():
            # Extract memory from GPU name (simplified)
            if "4090" in gpu or "4080" in gpu:
                gpus.append(gpu)
            elif "4070" in gpu and memory_gb <= 12:
                gpus.append(gpu)
            elif "4060" in gpu and memory_gb <= 8:
                gpus.append(gpu)
    return gpus


def get_rtx_series(series: str) -> list[str]:
    """Get GPUs from a specific RTX series."""
    series_num = int(series) if series.isdigit() else 40
    if series_num >= 50:
        return RTX_50_ONLY
    elif series_num >= 40:
        return RTX_40_PLUS
    elif series_num >= 30:
        return RTX_30_PLUS
    else:
        return []


class SortOrder(Enum):
    """Sort order options for machine search."""

    PRICE_ASC = "dph"  # Cheapest first
    PRICE_DESC = "dph-"  # Most expensive first
    RELIABILITY_DESC = "reliability-"  # Most reliable first
    PERFORMANCE_DESC = "dlperf-"  # Best performance first
    GPU_COUNT_DESC = "num_gpus-"  # Most GPUs first
    GPU_RAM_DESC = "gpu_ram-"  # Most GPU RAM first


@dataclass
class MachineCriteria:
    """Criteria for filtering machine offers."""

    # Hardware requirements
    gpu_name: str | list[str] | None = None
    min_gpu_ram: int = 8
    min_gpu_total_ram: int | None = None
    min_num_gpus: int = 1
    max_num_gpus: int | None = None
    min_cpu_cores: int | None = None
    min_cpu_ram: int | None = None
    min_disk_space: int | None = None

    # Performance requirements
    min_compute_cap: int | None = None
    min_total_flops: float | None = None
    min_dlperf: float | None = None

    # Cost constraints
    max_price_per_hour: float = 1.0
    min_dlperf_per_dollar: float | None = None

    # Reliability requirements
    min_reliability: float = 0.95
    verified_only: bool = True
    rentable_only: bool = True

    # Location preferences
    preferred_countries: list[str] | None = None
    excluded_countries: list[str] | None = None

    # System requirements
    min_driver_version: str | None = None
    min_cuda_version: float | None = None
    min_duration_days: float | None = None
    static_ip_required: bool = False

    # Network requirements
    min_download_speed: float | None = None
    min_upload_speed: float | None = None

    def __post_init__(self) -> None:
        """Validate GPU names after initialization."""
        if self.gpu_name:
            if isinstance(self.gpu_name, str):
                # Check if it's a valid GPU name (with or without underscores)
                variants = get_gpu_name_variants(self.gpu_name)
                if not any(validate_gpu_name(variant) for variant in variants):
                    logger.warning(
                        f"GPU name '{self.gpu_name}' may not be recognized by Vast.ai API"
                    )
            elif isinstance(self.gpu_name, list):
                # Validate each GPU name in the list
                for gpu in self.gpu_name:
                    variants = get_gpu_name_variants(gpu)
                    if not any(validate_gpu_name(variant) for variant in variants):
                        logger.warning(
                            f"GPU name '{gpu}' may not be recognized by Vast.ai API"
                        )


class VastMarketplace:
    """Enhanced Vast.ai marketplace integration."""

    def __init__(self, vast_py_path: str = "vast.py"):
        """Initialize marketplace client.

        Args:
            vast_py_path: Path to vast.py CLI tool
        """
        self.vast_py_path = vast_py_path

    def search_offers(
        self,
        criteria: MachineCriteria,
        sort_order: SortOrder = SortOrder.PRICE_ASC,
        limit: int = 50,
        timeout: int = 30,
    ) -> list[dict[str, Any]]:
        """Search for machine offers with sophisticated filtering.

        Args:
            criteria: Machine selection criteria
            sort_order: How to sort results
            limit: Maximum number of results
            timeout: Request timeout in seconds

        Returns:
            List of machine offers matching criteria
        """
        try:
            # Build query string
            query_parts = self._build_query(criteria)
            query = " ".join(query_parts)

            # Build command
            command = [
                "python3",
                self.vast_py_path,
                "search",
                "offers",
                query,
                "--limit",
                str(limit),
                "--order",
                sort_order.value,
                "--raw",
            ]

            logger.debug(f"Running command: {' '.join(command)}")

            # Execute search
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout
            )

            if result.returncode != 0:
                logger.error(f"Search command failed: {result.stderr}")
                return []

            # Parse results
            output = result.stdout.strip()
            if not output:
                logger.info("No machines found matching criteria")
                return []

            try:
                parsed_data = json.loads(output)
                if isinstance(parsed_data, list):
                    machines_data: list[dict[str, Any]] = parsed_data
                else:
                    machines_data = [parsed_data]
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON output: {output}")
                return []

            logger.info(f"Found {len(machines_data)} machines matching criteria")
            return machines_data

        except subprocess.TimeoutExpired:
            logger.error(f"Search command timed out after {timeout}s")
            return []
        except Exception as e:
            logger.error(f"Failed to search offers: {e}")
            return []

    def _build_query(self, criteria: MachineCriteria) -> list[str]:
        """Build query string from criteria."""
        query_parts = []

        # Hardware filters
        if criteria.gpu_name:
            if isinstance(criteria.gpu_name, str):
                # Single GPU name - replace spaces with underscores for Vast.ai query
                gpu_name = criteria.gpu_name.replace(" ", "_")
                query_parts.append(f"gpu_name={gpu_name}")
            elif isinstance(criteria.gpu_name, list):
                # Multiple GPU names - use 'in' operator
                gpu_names = [name.replace(" ", "_") for name in criteria.gpu_name]
                gpu_names_str = ",".join(gpu_names)
                query_parts.append(f"gpu_name in [{gpu_names_str}]")

        if criteria.min_gpu_ram:
            query_parts.append(f"gpu_ram>={criteria.min_gpu_ram}")

        if criteria.min_gpu_total_ram:
            query_parts.append(f"gpu_total_ram>={criteria.min_gpu_total_ram}")

        if criteria.min_num_gpus:
            query_parts.append(f"num_gpus>={criteria.min_num_gpus}")

        if criteria.max_num_gpus:
            query_parts.append(f"num_gpus<={criteria.max_num_gpus}")

        if criteria.min_cpu_cores:
            query_parts.append(f"cpu_cores>={criteria.min_cpu_cores}")

        if criteria.min_cpu_ram:
            query_parts.append(f"cpu_ram>={criteria.min_cpu_ram}")

        if criteria.min_disk_space:
            query_parts.append(f"disk_space>={criteria.min_disk_space}")

        # Performance filters
        if criteria.min_compute_cap:
            query_parts.append(f"compute_cap>={criteria.min_compute_cap}")

        if criteria.min_total_flops:
            query_parts.append(f"total_flops>={criteria.min_total_flops}")

        if criteria.min_dlperf:
            query_parts.append(f"dlperf>={criteria.min_dlperf}")

        # Cost filters
        if criteria.max_price_per_hour:
            query_parts.append(f"dph_total<={criteria.max_price_per_hour}")

        if criteria.min_dlperf_per_dollar:
            query_parts.append(f"dlperf_per_dphtotal>={criteria.min_dlperf_per_dollar}")

        # Reliability filters
        if criteria.min_reliability:
            query_parts.append(f"reliability>={criteria.min_reliability}")

        if criteria.verified_only:
            query_parts.append("verified=true")

        if criteria.rentable_only:
            query_parts.append("rentable=true")

        # Location filters
        if criteria.preferred_countries:
            countries_str = ",".join(criteria.preferred_countries)
            query_parts.append(f"geolocation in [{countries_str}]")

        if criteria.excluded_countries:
            countries_str = ",".join(criteria.excluded_countries)
            query_parts.append(f"geolocation notin [{countries_str}]")

        # System filters
        if criteria.min_driver_version:
            query_parts.append(f"driver_version>={criteria.min_driver_version}")

        if criteria.min_cuda_version:
            query_parts.append(f"cuda_vers>={criteria.min_cuda_version}")

        if criteria.min_duration_days:
            query_parts.append(f"duration>={criteria.min_duration_days}")

        if criteria.static_ip_required:
            query_parts.append("static_ip=true")

        # Network filters
        if criteria.min_download_speed:
            query_parts.append(f"inet_down>={criteria.min_download_speed}")

        if criteria.min_upload_speed:
            query_parts.append(f"inet_up>={criteria.min_upload_speed}")

        return query_parts

    def find_best_offer(
        self,
        criteria: MachineCriteria,
        sort_order: SortOrder = SortOrder.PRICE_ASC,
        limit: int = 20,
    ) -> dict | None:
        """Find the best machine offer matching criteria.

        Args:
            criteria: Machine selection criteria
            sort_order: How to sort results
            limit: Maximum number of results to consider

        Returns:
            Best machine offer or None if no matches
        """
        offers = self.search_offers(criteria, sort_order, limit)

        if not offers:
            return None

        # Return the first (best) offer based on sort order
        best_offer = offers[0]

        logger.info(
            f"Best offer: {best_offer.get('gpu_name', 'Unknown')} "
            f"${best_offer.get('dph_total', 0):.3f}/hr "
            f"reliability={best_offer.get('reliability', 0):.3f}"
        )

        return best_offer

    def get_market_summary(self, criteria: MachineCriteria, limit: int = 100) -> dict:
        """Get market summary for given criteria.

        Args:
            criteria: Machine selection criteria
            limit: Maximum number of results to analyze

        Returns:
            Market summary statistics
        """
        offers = self.search_offers(criteria, SortOrder.PRICE_ASC, limit)

        if not offers:
            return {
                "total_offers": 0,
                "price_range": {"min": 0, "max": 0, "avg": 0},
                "reliability_range": {"min": 0, "max": 0, "avg": 0},
                "gpu_types": {},
                "countries": {},
            }

        # Calculate statistics
        prices = [offer.get("dph_total", 0) for offer in offers]
        reliabilities = [offer.get("reliability", 0) for offer in offers]

        gpu_types: dict[str, int] = {}
        countries: dict[str, int] = {}

        for offer in offers:
            gpu_name = offer.get("gpu_name", "Unknown")
            gpu_types[gpu_name] = gpu_types.get(gpu_name, 0) + 1

            country = offer.get("geolocation", "Unknown")
            countries[country] = countries.get(country, 0) + 1

        return {
            "total_offers": len(offers),
            "price_range": {
                "min": min(prices) if prices else 0,
                "max": max(prices) if prices else 0,
                "avg": sum(prices) / len(prices) if prices else 0,
            },
            "reliability_range": {
                "min": min(reliabilities) if reliabilities else 0,
                "max": max(reliabilities) if reliabilities else 0,
                "avg": sum(reliabilities) / len(reliabilities) if reliabilities else 0,
            },
            "gpu_types": gpu_types,
            "countries": countries,
        }

    def find_high_performance_offers(
        self,
        max_price: float = 2.0,
        min_reliability: float = 0.98,
        preferred_countries: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Find high-performance GPU offers (RTX 4090, A100, H100, etc.)."""
        criteria = MachineCriteria(
            gpu_name=get_high_performance_gpus(),
            max_price_per_hour=max_price,
            min_reliability=min_reliability,
            preferred_countries=preferred_countries,
            min_gpu_ram=24,
        )
        return self.search_offers(criteria, SortOrder.PERFORMANCE_DESC, limit)

    def find_budget_offers(
        self,
        max_price: float = 0.5,
        min_reliability: float = 0.9,
        preferred_countries: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Find budget-friendly GPU offers."""
        criteria = MachineCriteria(
            gpu_name=get_budget_gpus(),
            max_price_per_hour=max_price,
            min_reliability=min_reliability,
            preferred_countries=preferred_countries,
            min_gpu_ram=8,
        )
        return self.search_offers(criteria, SortOrder.PRICE_ASC, limit)

    def find_multi_gpu_offers(
        self,
        gpu_type: str | list[str] | None = None,
        min_num_gpus: int = 2,
        max_price: float = 1.0,
        min_reliability: float = 0.95,
        limit: int = 10,
    ) -> list[dict]:
        """Find offers with multiple GPUs."""
        if gpu_type is None:
            gpu_type = get_multi_gpu_candidates()

        criteria = MachineCriteria(
            gpu_name=gpu_type,
            min_num_gpus=min_num_gpus,
            max_price_per_hour=max_price,
            min_reliability=min_reliability,
        )
        return self.search_offers(criteria, SortOrder.GPU_COUNT_DESC, limit)

    def find_rtx_series_offers(
        self,
        series: str = "40",  # "50", "40", "30", "20"
        max_price: float = 1.0,
        min_reliability: float = 0.95,
        limit: int = 20,
    ) -> list[dict]:
        """Find offers from a specific RTX series."""

        rtx_gpus = get_rtx_series(series)

        if not rtx_gpus:
            logger.warning(f"No RTX {series} series GPUs found")
            return []

        criteria = MachineCriteria(
            gpu_name=rtx_gpus,
            max_price_per_hour=max_price,
            min_reliability=min_reliability,
        )
        return self.search_offers(criteria, SortOrder.PRICE_ASC, limit)

    def find_rtx_series_and_above_offers(
        self,
        min_series: str = "30",
        max_price: float = 1.0,
        min_reliability: float = 0.95,
        preferred_countries: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Find RTX offers from a specific series and above.

        Args:
            min_series: Minimum RTX series ('20', '30', '40', '50')
            max_price: Maximum price per hour
            min_reliability: Minimum reliability score
            preferred_countries: List of preferred country codes
            limit: Maximum number of offers to return

        Returns:
            List of offers from RTX series and above

        Examples:
            find_rtx_series_and_above_offers('30')  # RTX 30, 40, 50 series
            find_rtx_series_and_above_offers('40')  # RTX 40, 50 series
        """
        rtx_gpus = get_rtx_series_and_above(min_series)

        if not rtx_gpus:
            logger.warning(f"No RTX {min_series}+ series GPUs found")
            return []

        criteria = MachineCriteria(
            gpu_name=rtx_gpus,
            max_price_per_hour=max_price,
            min_reliability=min_reliability,
            preferred_countries=preferred_countries,
        )
        return self.search_offers(criteria, SortOrder.PRICE_ASC, limit)

    def find_rtx_performance_tier_offers(
        self,
        tier: str = "high",
        max_price: float = 1.0,
        min_reliability: float = 0.95,
        preferred_countries: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Find RTX offers by performance tier.

        Args:
            tier: Performance tier ('budget', 'mid', 'high', 'enthusiast')
            max_price: Maximum price per hour
            min_reliability: Minimum reliability score
            preferred_countries: List of preferred country codes
            limit: Maximum number of offers to return

        Returns:
            List of offers from the specified performance tier
        """
        rtx_gpus = get_rtx_performance_tier(tier)

        if not rtx_gpus:
            logger.warning(f"No RTX {tier} tier GPUs found")
            return []

        criteria = MachineCriteria(
            gpu_name=rtx_gpus,
            max_price_per_hour=max_price,
            min_reliability=min_reliability,
            preferred_countries=preferred_countries,
        )
        return self.search_offers(criteria, SortOrder.PRICE_ASC, limit)

    def find_rtx_by_memory_offers(
        self,
        min_gb: int = 12,
        max_gb: int | None = None,
        max_price: float = 1.0,
        min_reliability: float = 0.95,
        preferred_countries: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Find RTX offers by memory range.

        Args:
            min_gb: Minimum memory in GB
            max_gb: Maximum memory in GB (optional)
            max_price: Maximum price per hour
            min_reliability: Minimum reliability score
            preferred_countries: List of preferred country codes
            limit: Maximum number of offers to return

        Returns:
            List of offers from RTX GPUs within memory range
        """
        rtx_gpus = get_rtx_by_memory(min_gb, max_gb)

        if not rtx_gpus:
            logger.warning(f"No RTX GPUs found with {min_gb}+ GB memory")
            return []

        criteria = MachineCriteria(
            gpu_name=rtx_gpus,
            max_price_per_hour=max_price,
            min_reliability=min_reliability,
            preferred_countries=preferred_countries,
        )
        return self.search_offers(criteria, SortOrder.PRICE_ASC, limit)


# Convenience functions for common use cases
def find_rtx4090_offers(
    max_price: float = 1.0,
    min_reliability: float = 0.95,
    preferred_countries: list[str] | None = None,
) -> list[dict]:
    """Find RTX 4090 offers with common criteria."""
    criteria = MachineCriteria(
        gpu_name="RTX_4090",
        max_price_per_hour=max_price,
        min_reliability=min_reliability,
        preferred_countries=preferred_countries,
    )

    marketplace = VastMarketplace()
    return marketplace.search_offers(criteria)


def find_rtx_offers(
    max_price: float = 1.0,
    min_reliability: float = 0.95,
    preferred_countries: list[str] | None = None,
) -> list[dict]:
    """Find RTX 30+ series offers with common criteria."""
    criteria = MachineCriteria(
        gpu_name=RTX_30_PLUS,
        max_price_per_hour=max_price,
        min_reliability=min_reliability,
        preferred_countries=preferred_countries,
    )

    marketplace = VastMarketplace()
    return marketplace.search_offers(criteria)


def find_high_end_gpu_offers(
    max_price: float = 2.0,
    min_reliability: float = 0.98,
    preferred_countries: list[str] | None = None,
) -> list[dict]:
    """Find high-end GPU offers (RTX 40+ series)."""
    criteria = MachineCriteria(
        gpu_name=RTX_40_PLUS,
        max_price_per_hour=max_price,
        min_reliability=min_reliability,
        preferred_countries=preferred_countries,
        min_gpu_ram=24,
    )

    marketplace = VastMarketplace()
    return marketplace.search_offers(criteria, SortOrder.PERFORMANCE_DESC)


def find_budget_gpu_offers(
    max_price: float = 0.5,
    min_reliability: float = 0.9,
    preferred_countries: list[str] | None = None,
) -> list[dict]:
    """Find budget-friendly GPU offers (RTX 3080, RTX 3090, RTX 4070)."""
    criteria = MachineCriteria(
        gpu_name=["RTX_3080", "RTX_3090", "RTX_4070"],
        max_price_per_hour=max_price,
        min_reliability=min_reliability,
        preferred_countries=preferred_countries,
        min_gpu_ram=8,
    )

    marketplace = VastMarketplace()
    return marketplace.search_offers(criteria, SortOrder.PRICE_ASC)
