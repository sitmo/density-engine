"""
Instance preparation for the vast.ai automation system.
"""

from typing import Any, Dict, List

from ..core.ssh import SSHClient, create_ssh_connection, execute_command
from ..instances.discovery import InstanceInfo
from ..utils.exceptions import InstanceError
from ..utils.logging import get_logger, log_function_call

logger = get_logger(__name__)


@log_function_call
def check_torch_installation(ssh_client: SSHClient) -> bool:
    """Check if torch is installed on the instance."""
    try:
        logger.debug("Checking torch installation")

        # Try multiple commands to find torch
        torch_commands = [
            "source /venv/main/bin/activate && python3 -c 'import torch; print(\"torch_ready\")'",
            'bash -c \'source /venv/main/bin/activate && python3 -c "import torch; print(\\"torch_ready\\")"\'',
            "/venv/main/bin/python3 -c 'import torch; print(\"torch_ready\")'",
            "python3 -c 'import torch; print(\"torch_ready\")'",
            "python -c 'import torch; print(\"torch_ready\")'",
        ]

        for cmd in torch_commands:
            result = execute_command(ssh_client, cmd, timeout=10)
            if result.success and "torch_ready" in result.stdout:
                logger.info("✅ Torch is installed and working")
                return True

        logger.warning("❌ Torch is not installed or not working")
        return False

    except Exception as e:
        logger.error(f"Failed to check torch installation: {e}")
        return False


@log_function_call
def check_density_engine_installation(ssh_client: SSHClient) -> bool:
    """Check if density_engine is installed on the instance."""
    try:
        logger.debug("Checking density_engine installation")

        cmd = "source /venv/main/bin/activate && python3 -c 'import density_engine; print(\"density_engine_ready\")'"
        result = execute_command(ssh_client, cmd, timeout=30)

        if result.success and "density_engine_ready" in result.stdout:
            logger.info("✅ density_engine is installed and working")
            return True
        else:
            logger.warning("❌ density_engine is not installed or not working")
            return False

    except Exception as e:
        logger.error(f"Failed to check density_engine installation: {e}")
        return False


@log_function_call
def install_missing_dependencies(
    ssh_client: SSHClient, dependencies: list[str]
) -> bool:
    """Install missing dependencies on the instance."""
    try:
        logger.debug(f"Installing missing dependencies: {dependencies}")

        if not dependencies:
            logger.info("No dependencies to install")
            return True

        # Install dependencies using virtual environment
        install_cmd = f"source /venv/main/bin/activate && pip3 install --user {' '.join(dependencies)}"
        result = execute_command(ssh_client, install_cmd, timeout=300)

        if result.success:
            logger.info(f"✅ Successfully installed dependencies: {dependencies}")
            return True
        else:
            logger.error(f"❌ Failed to install dependencies: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


@log_function_call
def clone_repository(ssh_client: SSHClient, repo_url: str, target_dir: str) -> bool:
    """Clone the repository on the instance."""
    try:
        logger.debug(f"Cloning repository {repo_url} to {target_dir}")

        # First, check if the directory exists and if it's a git repository
        check_cmd = f"if [ -d {target_dir} ]; then cd {target_dir} && git status > /dev/null 2>&1 && echo 'git_repo' || echo 'not_git_repo'; else echo 'no_dir'; fi"
        result = execute_command(ssh_client, check_cmd, timeout=10)

        if result.success and "not_git_repo" in result.stdout:
            # Directory exists but is not a git repository, remove it
            logger.info(
                f"Directory {target_dir} exists but is not a git repository. Removing it..."
            )
            remove_cmd = f"rm -rf {target_dir}"
            remove_result = execute_command(ssh_client, remove_cmd, timeout=30)
            if not remove_result.success:
                logger.error(
                    f"Failed to remove existing directory: {remove_result.stderr}"
                )
                return False

        # Clone or update repository
        if result.success and "git_repo" in result.stdout:
            # It's a git repository, try to pull
            logger.info(f"Updating existing git repository at {target_dir}")
            clone_cmd = f"cd {target_dir} && git pull"
        else:
            # Clone fresh repository
            logger.info(f"Cloning fresh repository to {target_dir}")
            clone_cmd = f"cd /root && git clone {repo_url} {target_dir}"

        result = execute_command(ssh_client, clone_cmd, timeout=60)

        if result.success:
            logger.info(f"✅ Repository cloned/updated successfully")
            return True
        else:
            logger.error(f"❌ Failed to clone repository: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Failed to clone repository: {e}")
        return False


@log_function_call
def prepare_instance_for_jobs(instance: InstanceInfo) -> bool:
    """Prepare an instance for running jobs."""
    try:
        logger.info(f"Preparing instance {instance.contract_id} for jobs")

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Check if torch is installed
            if not check_torch_installation(ssh_client):
                logger.error(
                    f"❌ Instance {instance.contract_id} FAILED: torch is missing!"
                )
                logger.error(
                    "❌ This instance was supposed to have torch pre-installed."
                )
                logger.error(
                    "❌ The vast.ai PyTorch template is not working correctly."
                )
                logger.error(
                    "❌ Please rent a different machine or check the template."
                )
                return False

            # Clone repository
            if not clone_repository(
                ssh_client,
                "https://github.com/sitmo/density-engine.git",
                "/root/density-engine",
            ):
                logger.error(
                    f"❌ Failed to clone repository on instance {instance.contract_id}"
                )
                return False

            # Check if density_engine is installed
            if not check_density_engine_installation(ssh_client):
                logger.info("Installing density_engine package...")

                # Install the local package
                install_cmd = "cd /root/density-engine && source /venv/main/bin/activate && pip3 install --user -e ."
                result = execute_command(ssh_client, install_cmd, timeout=300)

                if not result.success:
                    logger.error(
                        f"❌ Failed to install density_engine: {result.stderr}"
                    )
                    return False

                # Verify installation
                if not check_density_engine_installation(ssh_client):
                    logger.error(f"❌ density_engine installation verification failed")
                    return False

            logger.info(f"✅ Instance {instance.contract_id} is prepared")
            return True

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to prepare instance {instance.contract_id}: {e}")
        return False


@log_function_call
def verify_instance_readiness(instance: InstanceInfo) -> bool:
    """Verify that an instance is ready to run jobs."""
    try:
        logger.debug(f"Verifying instance {instance.contract_id} readiness")

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Check torch
            if not check_torch_installation(ssh_client):
                return False

            # Check density_engine
            if not check_density_engine_installation(ssh_client):
                return False

            # Check if repository exists
            check_repo_cmd = (
                "test -d /root/density-engine && echo 'exists' || echo 'missing'"
            )
            result = execute_command(ssh_client, check_repo_cmd, timeout=10)

            if not result.success or "exists" not in result.stdout:
                logger.warning(
                    f"Repository not found on instance {instance.contract_id}"
                )
                return False

            logger.info(f"✅ Instance {instance.contract_id} is ready")
            return True

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to verify instance readiness: {e}")
        return False


@log_function_call
def setup_development_environment(instance: InstanceInfo) -> bool:
    """Set up a complete development environment on the instance."""
    try:
        logger.info(
            f"Setting up development environment on instance {instance.contract_id}"
        )

        # Create SSH connection
        ssh_client = create_ssh_connection(instance.ssh_host, instance.ssh_port)
        ssh_client.connect()

        try:
            # Clone repository
            if not clone_repository(
                ssh_client,
                "https://github.com/sitmo/density-engine.git",
                "/root/density-engine",
            ):
                return False

            # Install missing dependencies
            missing_deps = ["pandas", "pyarrow", "scipy", "numpy"]
            if not install_missing_dependencies(ssh_client, missing_deps):
                return False

            # Install local package
            install_cmd = "cd /root/density-engine && source /venv/main/bin/activate && pip3 install --user -e ."
            result = execute_command(ssh_client, install_cmd, timeout=300)

            if not result.success:
                logger.error(f"❌ Failed to install local package: {result.stderr}")
                return False

            logger.info(
                f"✅ Development environment set up on instance {instance.contract_id}"
            )
            return True

        finally:
            ssh_client.close()

    except Exception as e:
        logger.error(f"Failed to setup development environment: {e}")
        return False
