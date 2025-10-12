"""
SSH operations for the vast.ai automation system.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import paramiko

from ..utils.exceptions import SSHConnectionError
from ..utils.logging import get_logger, log_execution_time, log_function_call

logger = get_logger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int


class SSHClient:
    """SSH client wrapper with connection management."""

    def __init__(self, host: str, port: int, username: str = "root"):
        self.host = host
        self.port = port
        self.username = username
        self.client: paramiko.SSHClient | None = None
        self.sftp: paramiko.SFTPClient | None = None
        self.connected = False

    def connect(
        self, max_retries: int = 3, retry_delay: int = 5
    ) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
        """Connect to SSH server."""
        if self.connected and self.client and self.sftp:
            return self.client, self.sftp

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Connecting to SSH host {self.host}:{self.port} (attempt {attempt + 1})"
                )

                self.client = paramiko.SSHClient()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                # Try SSH key authentication first
                logger.info("Trying default SSH keys")
                self.client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    timeout=30,
                )

                transport = self.client.get_transport()
                if transport:
                    try:
                        version = transport.get_version()
                        local_version = transport.get_local_version()
                        logger.info(
                            f"Connected (version {version}, client {local_version})"
                        )
                    except AttributeError:
                        logger.info("Connected (version info not available)")
                else:
                    logger.info("Connected (transport not available)")

                # Get auth banner
                transport = self.client.get_transport()
                if transport:
                    banner = transport.get_banner()
                    if banner:
                        logger.info(f"Auth banner: {banner}")

                logger.info("Authentication (publickey) successful!")
                logger.info("SSH connection established after 1 attempts")

                # Open SFTP connection
                self.sftp = self.client.open_sftp()
                try:
                    sftp_version = self.sftp.get_channel().get_transport().get_version()
                    logger.info(
                        f"[chan 0] Opened sftp connection (server version {sftp_version})"
                    )
                except AttributeError:
                    logger.info(
                        "[chan 0] Opened sftp connection (server version not available)"
                    )
                logger.info("SFTP connection established")

                self.connected = True
                return self.client, self.sftp

            except Exception as e:
                logger.warning(f"SSH connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise SSHConnectionError(
                        f"Failed to connect to {self.host}:{self.port} after {max_retries} attempts: {e}"
                    )

        # This should never be reached, but mypy needs it
        raise SSHConnectionError("Unexpected error in connect method")

    def close(self) -> None:
        """Close SSH connections."""
        if self.sftp:
            self.sftp.close()
            logger.info("[chan 0] sftp session closed.")
        if self.client:
            self.client.close()
        logger.info("SSH connections closed")
        self.connected = False

    def execute_command(self, command: str, timeout: int = 30) -> CommandResult:
        """Execute a command on the remote server."""
        if not self.connected or not self.client:
            raise SSHConnectionError("Not connected to SSH server")

        try:
            logger.info(f"Executing command: {command}")
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)

            # Wait for command to complete
            exit_code = stdout.channel.recv_exit_status()

            # Read output
            stdout_text = stdout.read().decode("utf-8").strip()
            stderr_text = stderr.read().decode("utf-8").strip()

            success = exit_code == 0
            logger.info(
                f"Command completed successfully"
                if success
                else f"Command failed with exit code {exit_code}"
            )

            return CommandResult(
                success=success,
                stdout=stdout_text,
                stderr=stderr_text,
                exit_code=exit_code,
            )

        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return CommandResult(success=False, stdout="", stderr=str(e), exit_code=-1)

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload a file to the remote server."""
        if not self.connected or not self.sftp:
            raise SSHConnectionError("Not connected to SSH server")

        try:
            logger.info(f"Uploading {local_path} to {remote_path}")
            self.sftp.put(local_path, remote_path)
            logger.info(f"File uploaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download a file from the remote server."""
        if not self.connected or not self.sftp:
            raise SSHConnectionError("Not connected to SSH server")

        try:
            logger.info(f"Downloading {remote_path} to {local_path}")
            self.sftp.get(remote_path, local_path)
            logger.info(f"File downloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False


@log_function_call
def create_ssh_connection(host: str, port: int, username: str = "root") -> SSHClient:
    """Create a new SSH connection."""
    return SSHClient(host, port, username)


@log_function_call
def execute_command(
    client: SSHClient, command: str, timeout: int = 30
) -> CommandResult:
    """Execute a command on the remote server."""
    return client.execute_command(command, timeout)


@log_function_call
def upload_file(client: SSHClient, local_path: str, remote_path: str) -> bool:
    """Upload a file to the remote server."""
    return client.upload_file(local_path, remote_path)


@log_function_call
def download_file(client: SSHClient, remote_path: str, local_path: str) -> bool:
    """Download a file from the remote server."""
    return client.download_file(remote_path, local_path)


@log_function_call
def test_ssh_connectivity(host: str, port: int) -> bool:
    """Test SSH connectivity without full connection."""
    try:
        client = create_ssh_connection(host, port)
        client.connect(max_retries=1, retry_delay=1)
        client.close()
        return True
    except Exception:
        return False


@log_function_call
def with_ssh_connection(host: str, port: int, func: Callable) -> Any:
    """Execute a function with an SSH connection."""
    client = create_ssh_connection(host, port)
    try:
        client.connect()
        return func(client)
    finally:
        client.close()


@log_function_call
def execute_with_retry(
    client: SSHClient, command: str, max_retries: int = 3
) -> CommandResult:
    """Execute a command with retry logic."""
    for attempt in range(max_retries):
        result = client.execute_command(command)
        if result.success:
            return result

        if attempt < max_retries - 1:
            logger.warning(
                f"Command failed, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(2)

    return result
