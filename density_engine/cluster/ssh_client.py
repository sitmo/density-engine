"""
SSH client wrapper with connection management and retry logic.

This module provides a persistent SSH connection wrapper that handles
connection failures, reconnection, and retry logic.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncssh

logger = logging.getLogger("SSH")


class SSHClient:
    """Persistent SSH client with automatic reconnection and retry logic."""

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: str = "root",
        key_files: list[str] | None = None,
        connect_timeout: int = 10,
        keepalive_interval: int = 30,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.key_files = key_files or []
        self.connect_timeout = connect_timeout
        self.keepalive_interval = keepalive_interval

        self.conn: asyncssh.SSHClientConnection | None = None
        self.sftp: asyncssh.SFTPClient | None = None
        self._next_connect_ts: float = 0.0
        self._conn_backoff = 1.0
        self._conn_backoff_max = 60.0
        self._last_conn_error_logged = 0.0
        self._last_conn_test = 0.0
        self._conn_test_interval = 30.0  # Test connection every 30 seconds

    async def ensure_connection(self) -> asyncssh.SSHClientConnection:
        """Ensure SSH connection is established with retry logic."""
        if self.conn is not None:
            # Only test connection periodically, not on every call
            now = time.time()
            if now - self._last_conn_test < self._conn_test_interval:
                return self.conn

            # Test connection with a simple command
            try:
                await self.conn.run("true", check=False, timeout=3)
                self._last_conn_test = now
                return self.conn
            except Exception:
                # Connection is dead, clean up and reconnect
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None

        now = time.time()
        if now < self._next_connect_ts:
            await asyncio.sleep(self._next_connect_ts - now)

        try:
            self.conn = await asyncssh.connect(
                self.host,
                username=self.username,
                port=self.port,
                client_keys=self.key_files or None,
                agent_path=None,  # Will be set by caller if needed
                known_hosts=None,
                keepalive_interval=self.keepalive_interval,
                connect_timeout=self.connect_timeout,
            )
            self._next_connect_ts = 0.0
            self._conn_backoff = 1.0
            self._last_conn_test = time.time()
            logger.info(f"SSH connected to {self.host}:{self.port}")
            return self.conn
        except Exception as e:
            now = time.time()
            if now - self._last_conn_error_logged >= self._conn_backoff - 0.5:
                logger.warning(f"SSH connection failed to {self.host}:{self.port}: {e}")
                self._last_conn_error_logged = now
            self._next_connect_ts = max(self._next_connect_ts, now + self._conn_backoff)
            self._conn_backoff = min(self._conn_backoff * 2.0, self._conn_backoff_max)
            raise

    async def run_command(
        self, command: str, timeout: int = 30
    ) -> asyncssh.SSHCompletedProcess:
        """Run a command with automatic reconnection on failure."""
        try:
            conn = await self.ensure_connection()
            return await conn.run(command, check=False, timeout=timeout)
        except Exception:
            # Connection failed, try to reconnect and retry once
            self.conn = None
            try:
                conn = await self.ensure_connection()
                return await conn.run(command, check=False, timeout=timeout)
            except Exception as e:
                logger.error(f"SSH command failed after retry on {self.host}: {e}")
                raise

    async def get_sftp(self) -> asyncssh.SFTPClient:
        """Get SFTP client with automatic reconnection on failure."""
        if self.sftp is not None:
            try:
                # Test SFTP connection with a simple operation
                await self.sftp.stat(".")
                return self.sftp
            except Exception:
                # SFTP connection is dead, clean up
                try:
                    if self.sftp is not None:
                        # SFTP client doesn't have close method, just set to None
                        pass
                except Exception:
                    pass
                self.sftp = None

        try:
            conn = await self.ensure_connection()
            self.sftp = await conn.start_sftp_client()
            return self.sftp
        except Exception:
            # Connection failed, try to reconnect and retry once
            self.conn = None
            self.sftp = None
            try:
                conn = await self.ensure_connection()
                self.sftp = await conn.start_sftp_client()
                return self.sftp
            except Exception as e:
                logger.error(f"SFTP connection failed after retry on {self.host}: {e}")
                raise

    async def close(self) -> None:
        """Close SSH and SFTP connections."""
        if self.sftp is not None:
            try:
                # SFTP client doesn't have close method, just set to None
                pass
            except Exception:
                pass
            self.sftp = None

        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if self.conn is not None:
            try:
                asyncio.create_task(self.close())
            except Exception:
                pass
