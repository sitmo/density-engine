"""
Instance lifecycle management for the vast.ai automation system.
"""

import subprocess
from typing import Optional

from ..core.state import InstanceState, InstanceStatus
from ..instances.discovery import InstanceInfo
from ..utils.exceptions import InstanceError
from ..utils.logging import get_logger, log_function_call

logger = get_logger(__name__)


@log_function_call
def start_instance(instance_id: str) -> bool:
    """Start an instance."""
    try:
        logger.info(f"Starting instance {instance_id}")

        # Use vast.py CLI to start instance
        result = subprocess.run(
            ["python3", "vast.py", "start", "instance", instance_id],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            logger.info(f"✅ Instance {instance_id} started successfully")
            return True
        else:
            logger.error(f"❌ Failed to start instance {instance_id}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while starting instance {instance_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to start instance {instance_id}: {e}")
        return False


@log_function_call
def stop_instance(instance_id: str) -> bool:
    """Stop an instance."""
    try:
        logger.info(f"Stopping instance {instance_id}")

        # Use vast.py CLI to stop instance
        result = subprocess.run(
            ["python3", "vast.py", "stop", "instance", instance_id],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            logger.info(f"✅ Instance {instance_id} stopped successfully")
            return True
        else:
            logger.error(f"❌ Failed to stop instance {instance_id}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while stopping instance {instance_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to stop instance {instance_id}: {e}")
        return False


@log_function_call
def restart_instance(instance_id: str) -> bool:
    """Restart an instance."""
    try:
        logger.info(f"Restarting instance {instance_id}")

        # Stop first
        if not stop_instance(instance_id):
            logger.warning(f"Failed to stop instance {instance_id} before restart")

        # Wait a bit
        import time

        time.sleep(5)

        # Start again
        if not start_instance(instance_id):
            logger.error(f"Failed to start instance {instance_id} after restart")
            return False

        logger.info(f"✅ Instance {instance_id} restarted successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to restart instance {instance_id}: {e}")
        return False


@log_function_call
def terminate_instance(instance_id: str) -> bool:
    """Terminate an instance."""
    try:
        logger.info(f"Terminating instance {instance_id}")

        # Use vast.py CLI to destroy instance
        result = subprocess.run(
            ["python3", "vast.py", "destroy", "instance", instance_id],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            logger.info(f"✅ Instance {instance_id} terminated successfully")
            return True
        else:
            logger.error(
                f"❌ Failed to terminate instance {instance_id}: {result.stderr}"
            )
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while terminating instance {instance_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to terminate instance {instance_id}: {e}")
        return False


@log_function_call
def manage_instance_lifecycle(
    instance: InstanceInfo, desired_state: InstanceStatus
) -> bool:
    """Manage instance lifecycle to reach desired state."""
    try:
        logger.info(
            f"Managing lifecycle for instance {instance.contract_id} to {desired_state.value}"
        )

        instance_id = str(instance.contract_id)

        if desired_state == InstanceStatus.RUNNING:
            if instance.status == "running":
                logger.info(f"Instance {instance_id} is already running")
                return True
            else:
                return start_instance(instance_id)

        elif desired_state == InstanceStatus.STOPPED:
            if instance.status == "stopped":
                logger.info(f"Instance {instance_id} is already stopped")
                return True
            else:
                return stop_instance(instance_id)

        elif desired_state == InstanceStatus.RESTARTING:
            return restart_instance(instance_id)

        elif desired_state == InstanceStatus.TERMINATED:
            return terminate_instance(instance_id)

        else:
            logger.warning(f"Unknown desired state: {desired_state}")
            return False

    except Exception as e:
        logger.error(f"Failed to manage instance lifecycle: {e}")
        return False


@log_function_call
def handle_instance_failure(instance: InstanceInfo) -> bool:
    """Handle instance failure by attempting recovery."""
    try:
        logger.warning(f"Handling failure for instance {instance.contract_id}")

        instance_id = str(instance.contract_id)

        # Try to restart the instance
        logger.info(f"Attempting to restart failed instance {instance_id}")
        if restart_instance(instance_id):
            logger.info(f"✅ Successfully restarted failed instance {instance_id}")
            return True
        else:
            logger.error(f"❌ Failed to restart instance {instance_id}")
            return False

    except Exception as e:
        logger.error(f"Failed to handle instance failure: {e}")
        return False


@log_function_call
def cleanup_failed_instance(instance: InstanceInfo) -> bool:
    """Clean up a failed instance."""
    try:
        logger.info(f"Cleaning up failed instance {instance.contract_id}")

        instance_id = str(instance.contract_id)

        # Terminate the instance
        if terminate_instance(instance_id):
            logger.info(f"✅ Successfully cleaned up failed instance {instance_id}")
            return True
        else:
            logger.error(f"❌ Failed to clean up instance {instance_id}")
            return False

    except Exception as e:
        logger.error(f"Failed to cleanup failed instance: {e}")
        return False
