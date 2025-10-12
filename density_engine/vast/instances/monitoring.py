"""
Instance monitoring for the vast.ai automation system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from ..core.ssh import SSHClient, create_ssh_connection, execute_command
from ..core.state import InstanceState
from ..instances.discovery import InstanceInfo
from ..utils.exceptions import InstanceError
from ..utils.logging import get_logger, log_function_call

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Instance health status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class UtilizationMetrics:
    """Instance utilization metrics."""

    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_usage: float


@dataclass
class DiskSpaceInfo:
    """Disk space information."""

    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float


@dataclass
class MemoryInfo:
    """Memory information."""

    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float


@dataclass
class Issue:
    """Instance issue."""

    severity: str
    description: str
    component: str


@dataclass
class InstanceSummary:
    """Instance summary."""

    instance_id: str
    status: str
    health: HealthStatus
    utilization: UtilizationMetrics
    issues: list[Issue]


@log_function_call
def check_instance_health(instance: InstanceInfo) -> HealthStatus:
    """Check the health of an instance."""
    try:
        logger.debug(f"Checking health of instance {instance.contract_id}")

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Check if SSH is responsive
            result = execute_command(ssh_client, "echo 'health_check'", timeout=10)
            if not result.success:
                logger.warning(f"Instance {instance.contract_id} SSH not responsive")
                return HealthStatus.CRITICAL

            # Check if critical processes are running
            check_processes_cmd = (
                "ps aux | grep -E '(python|torch)' | grep -v grep | wc -l"
            )
            result = execute_command(ssh_client, check_processes_cmd, timeout=10)

            if result.success:
                process_count = int(result.stdout.strip())
                if process_count > 0:
                    logger.debug(
                        f"Instance {instance.contract_id} has {process_count} Python processes"
                    )
                else:
                    logger.warning(
                        f"Instance {instance.contract_id} has no Python processes"
                    )

            logger.debug(f"Instance {instance.contract_id} health check passed")
            return HealthStatus.HEALTHY

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to check instance health: {e}")
        return HealthStatus.UNKNOWN


@log_function_call
def get_instance_utilization(instance: InstanceInfo) -> UtilizationMetrics:
    """Get instance utilization metrics."""
    try:
        logger.debug(f"Getting utilization for instance {instance.contract_id}")

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Get CPU usage
            cpu_cmd = "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | sed 's/%us,//'"
            cpu_result = execute_command(ssh_client, cpu_cmd, timeout=10)
            cpu_usage = float(cpu_result.stdout.strip()) if cpu_result.success else 0.0

            # Get memory usage
            mem_cmd = "free | grep Mem | awk '{printf \"%.1f\", $3/$2 * 100.0}'"
            mem_result = execute_command(ssh_client, mem_cmd, timeout=10)
            memory_usage = (
                float(mem_result.stdout.strip()) if mem_result.success else 0.0
            )

            # Get disk usage
            disk_cmd = "df -h / | awk 'NR==2{print $5}' | sed 's/%//'"
            disk_result = execute_command(ssh_client, disk_cmd, timeout=10)
            disk_usage = (
                float(disk_result.stdout.strip()) if disk_result.success else 0.0
            )

            # GPU usage (if available)
            gpu_cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo '0'"
            gpu_result = execute_command(ssh_client, gpu_cmd, timeout=10)
            gpu_usage = float(gpu_result.stdout.strip()) if gpu_result.success else 0.0

            metrics = UtilizationMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                disk_usage=disk_usage,
            )

            logger.debug(
                f"Instance {instance.contract_id} utilization: CPU={cpu_usage}%, Memory={memory_usage}%, GPU={gpu_usage}%, Disk={disk_usage}%"
            )
            return metrics

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to get instance utilization: {e}")
        return UtilizationMetrics(0.0, 0.0, 0.0, 0.0)


@log_function_call
def check_disk_space(instance: InstanceInfo) -> DiskSpaceInfo:
    """Check disk space on the instance."""
    try:
        logger.debug(f"Checking disk space for instance {instance.contract_id}")

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Get disk space information
            disk_cmd = "df -BG / | awk 'NR==2{print $2,$3,$4,$5}'"
            result = execute_command(ssh_client, disk_cmd, timeout=10)

            if result.success:
                parts = result.stdout.strip().split()
                if len(parts) >= 4:
                    total_gb = float(parts[0].replace("G", ""))
                    used_gb = float(parts[1].replace("G", ""))
                    available_gb = float(parts[2].replace("G", ""))
                    usage_percent = float(parts[3].replace("%", ""))

                    disk_info = DiskSpaceInfo(
                        total_gb=total_gb,
                        used_gb=used_gb,
                        available_gb=available_gb,
                        usage_percent=usage_percent,
                    )

                    logger.debug(
                        f"Instance {instance.contract_id} disk: {used_gb:.1f}GB/{total_gb:.1f}GB ({usage_percent:.1f}%)"
                    )
                    return disk_info

            # Fallback to default values
            return DiskSpaceInfo(0.0, 0.0, 0.0, 0.0)

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")
        return DiskSpaceInfo(0.0, 0.0, 0.0, 0.0)


@log_function_call
def check_memory_usage(instance: InstanceInfo) -> MemoryInfo:
    """Check memory usage on the instance."""
    try:
        logger.debug(f"Checking memory usage for instance {instance.contract_id}")

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Get memory information
            mem_cmd = "free -g | awk 'NR==2{print $2,$3,$7}'"
            result = execute_command(ssh_client, mem_cmd, timeout=10)

            if result.success:
                parts = result.stdout.strip().split()
                if len(parts) >= 3:
                    total_gb = float(parts[0])
                    used_gb = float(parts[1])
                    available_gb = float(parts[2])
                    usage_percent = (used_gb / total_gb) * 100 if total_gb > 0 else 0

                    memory_info = MemoryInfo(
                        total_gb=total_gb,
                        used_gb=used_gb,
                        available_gb=available_gb,
                        usage_percent=usage_percent,
                    )

                    logger.debug(
                        f"Instance {instance.contract_id} memory: {used_gb:.1f}GB/{total_gb:.1f}GB ({usage_percent:.1f}%)"
                    )
                    return memory_info

            # Fallback to default values
            return MemoryInfo(0.0, 0.0, 0.0, 0.0)

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to check memory usage: {e}")
        return MemoryInfo(0.0, 0.0, 0.0, 0.0)


@log_function_call
def monitor_instance_status(instance: InstanceInfo) -> str:
    """Monitor instance status."""
    try:
        logger.debug(f"Monitoring status of instance {instance.contract_id}")

        # Check health
        health = check_instance_health(instance)

        # Check utilization
        utilization = get_instance_utilization(instance)

        # Determine status based on health and utilization
        if health == HealthStatus.CRITICAL:
            return "critical"
        elif health == HealthStatus.WARNING:
            return "warning"
        elif utilization.disk_usage > 90:
            return "disk_full"
        elif utilization.memory_usage > 95:
            return "memory_full"
        else:
            return "healthy"

    except Exception as e:
        logger.error(f"Failed to monitor instance status: {e}")
        return "unknown"


@log_function_call
def detect_instance_issues(instance: InstanceInfo) -> list[Issue]:
    """Detect issues with an instance."""
    try:
        logger.debug(f"Detecting issues for instance {instance.contract_id}")

        issues = []

        # Check disk space
        disk_info = check_disk_space(instance)
        if disk_info.usage_percent > 90:
            issues.append(
                Issue(
                    severity="critical",
                    description=f"Disk usage is {disk_info.usage_percent:.1f}%",
                    component="disk",
                )
            )
        elif disk_info.usage_percent > 80:
            issues.append(
                Issue(
                    severity="warning",
                    description=f"Disk usage is {disk_info.usage_percent:.1f}%",
                    component="disk",
                )
            )

        # Check memory usage
        memory_info = check_memory_usage(instance)
        if memory_info.usage_percent > 95:
            issues.append(
                Issue(
                    severity="critical",
                    description=f"Memory usage is {memory_info.usage_percent:.1f}%",
                    component="memory",
                )
            )
        elif memory_info.usage_percent > 85:
            issues.append(
                Issue(
                    severity="warning",
                    description=f"Memory usage is {memory_info.usage_percent:.1f}%",
                    component="memory",
                )
            )

        # Check health
        health = check_instance_health(instance)
        if health == HealthStatus.CRITICAL:
            issues.append(
                Issue(
                    severity="critical",
                    description="Instance health check failed",
                    component="health",
                )
            )
        elif health == HealthStatus.WARNING:
            issues.append(
                Issue(
                    severity="warning",
                    description="Instance health check warning",
                    component="health",
                )
            )

        logger.debug(
            f"Detected {len(issues)} issues for instance {instance.contract_id}"
        )
        return issues

    except Exception as e:
        logger.error(f"Failed to detect instance issues: {e}")
        return []


@log_function_call
def get_instance_summary(instance: InstanceInfo) -> InstanceSummary:
    """Get a summary of an instance."""
    try:
        logger.debug(f"Getting summary for instance {instance.contract_id}")

        # Get status
        status = monitor_instance_status(instance)

        # Get health
        health = check_instance_health(instance)

        # Get utilization
        utilization = get_instance_utilization(instance)

        # Get issues
        issues = detect_instance_issues(instance)

        summary = InstanceSummary(
            instance_id=str(instance.contract_id),
            status=status,
            health=health,
            utilization=utilization,
            issues=issues,
        )

        logger.debug(
            f"Instance {instance.contract_id} summary: {status}, {health.value}, {len(issues)} issues"
        )
        return summary

    except Exception as e:
        logger.error(f"Failed to get instance summary: {e}")
        return InstanceSummary(
            instance_id=str(instance.contract_id),
            status="unknown",
            health=HealthStatus.UNKNOWN,
            utilization=UtilizationMetrics(0.0, 0.0, 0.0, 0.0),
            issues=[],
        )
