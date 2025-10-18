"""
Clean, minimal type definitions for the density-engine project.

Consolidated to just the essential TypedDicts to avoid type proliferation.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


# --- Core Data Types ---
class InstanceData(TypedDict):
    """Standard instance information."""

    id: str
    ssh_host: str
    ssh_port: int
    gpu_name: str
    price_per_hour: float
    reliability: float
    location: str
    status: str
    cpu_cores: int
    gpu_ram: int
    disk_space: int


class JobData(TypedDict):
    """Standard job information."""

    id: str
    model_name: str
    instance_id: str
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]
    parameters: dict[str, float | int | str]
    output_path: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    error_message: str | None


class TaskResult(TypedDict):
    """Standard task result."""

    status: Literal["SUCCESS", "FAILED", "PENDING", "RETRY"]
    message: str
    data: dict[str, str | int | float | bool | list | dict] | None
    details: dict[str, str | int | float | bool | list | dict] | None
    error: str | None


# --- Specific Result Types ---
class InstanceResult(TypedDict):
    """Instance operation result."""

    status: Literal["SUCCESS", "FAILED"]
    instance_id: str | None
    message: str
    error: str | None


class JobResult(TypedDict):
    """Job operation result."""

    status: Literal["SUCCESS", "FAILED"]
    job_id: str
    message: str
    error: str | None


class ClusterResult(TypedDict):
    """Cluster operation result."""

    status: Literal["SUCCESS", "FAILED", "HEALTHY", "CRITICAL"]
    message: str
    data: dict[str, str | int | float | bool | list | dict] | None
    details: dict[str, str | int | float | bool | list | dict] | None
    cluster_status: ClusterStatus | None
    actions_taken: list[str] | None
    error: str | None


# --- Market Data ---
class MarketOffer(TypedDict):
    """Vast.ai market offer."""

    id: str
    gpu_name: str
    price_per_hour: float
    reliability: float
    location: str
    gpu_ram: int
    cpu_cores: int
    disk_space: int
    ssh_host: str
    ssh_port: int


# --- Health & Status ---
class HealthStatus(TypedDict):
    """Health check result."""

    status: Literal["HEALTHY", "UNHEALTHY", "UNKNOWN"]
    instance_id: str
    message: str
    error: str | None


class ClusterStatus(TypedDict):
    """Overall cluster status."""

    total_instances: int
    available_instances: int
    running_instances: int
    pending_instances: int
    errored_instances: int
    total_jobs: int
    running_jobs: int
    pending_jobs: int
    failed_jobs: int
    completed_jobs: int
    message: str


# --- Instance Management Types ---
class InstanceInfo(TypedDict):
    """Instance information for management tasks."""

    id: str
    ssh_host: str
    ssh_port: int
    gpu_name: str
    price_per_hour: float
    reliability: float
    location: str
    status: str
    cpu_cores: int
    gpu_ram: int
    disk_space: int


class InstanceHealthStatus(TypedDict):
    """Instance health status."""

    status: Literal["HEALTHY", "UNHEALTHY", "UNKNOWN"]
    instance_id: str
    message: str
    error: str | None


class InstanceHealthResult(TypedDict):
    """Instance health check result."""

    status: Literal["SUCCESS", "FAILED"]
    message: str
    data: list[InstanceHealthStatus] | None
    error: str | None


class InstancePreparationResult(TypedDict):
    """Instance preparation result."""

    status: Literal["SUCCESS", "FAILED"]
    message: str
    data: InstanceInfo | None
    error: str | None


# --- Job Management Types ---
class JobParams(TypedDict):
    """Job parameters."""

    model_name: str
    parameters: dict[str, float | int | str]
    output_path: str


class JobSubmissionResult(TypedDict):
    """Job submission result."""

    status: Literal["SUCCESS", "FAILED"]
    job_id: str
    message: str
    error: str | None


class JobExecutionResult(TypedDict):
    """Job execution result."""

    status: Literal["SUCCESS", "FAILED", "RUNNING"]
    job_id: str
    instance_id: str
    start_time: str | None
    end_time: str | None
    progress: float
    message: str
    error: str | None


class JobMonitoringResult(TypedDict):
    """Job monitoring result."""

    status: Literal["SUCCESS", "FAILED", "RUNNING", "COMPLETED"]
    job_id: str
    instance_id: str
    progress: float
    message: str
    error: str | None


class JobCollectionResult(TypedDict):
    """Job collection result."""

    status: Literal["SUCCESS", "FAILED"]
    job_id: str
    instance_id: str
    collected_files: list[str]
    message: str
    error: str | None


# --- Recovery Types ---
class TaskStatus(TypedDict):
    """Task status information."""

    task_id: str
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]
    result: Any | None
    traceback: str | None
    message: str | None
    error: str | None


class ValidationResults(TypedDict):
    """Validation results."""

    valid: bool
    errors: list[str]
    warnings: list[str]


class RecoverySummary(TypedDict):
    """Recovery operation summary."""

    status: Literal["SUCCESS", "FAILED", "PARTIAL"]
    message: str
    instances_recovered: int
    running_jobs: int
    jobs_recovered: int
    pending_tasks: int
    recovered_instances: int
    recovered_jobs: int
    errors: list[str]


class RepairSummary(TypedDict):
    """Repair operation summary."""

    status: Literal["SUCCESS", "FAILED", "PARTIAL"]
    message: str
    repaired_instances: int
    repaired_jobs: int
    errors: list[str]


class CleanupSummary(TypedDict):
    """Cleanup operation summary."""

    status: Literal["SUCCESS", "FAILED", "PARTIAL"]
    message: str
    cleaned_instances: int
    cleaned_jobs: int
    errors: list[str]
