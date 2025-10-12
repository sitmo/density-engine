"""
Core modules for the vast.ai automation system.
"""

from .files import JobState, copy_file, move_file, move_job_file
from .jobs import (
    discover_job_files,
    get_job_arguments,
    parse_job_file,
    validate_job_file,
)
from .ssh import (
    SSHClient,
    create_ssh_connection,
    download_file,
    execute_command,
    upload_file,
)
from .state import (
    InstanceState,
    InstanceStatus,
    JobStateInfo,
    StateManager,
    get_state_manager,
)

__all__ = [
    "SSHClient",
    "create_ssh_connection",
    "execute_command",
    "upload_file",
    "download_file",
    "move_file",
    "copy_file",
    "move_job_file",
    "JobState",
    "InstanceState",
    "InstanceStatus",
    "JobStateInfo",
    "StateManager",
    "get_state_manager",
    "parse_job_file",
    "validate_job_file",
    "get_job_arguments",
    "discover_job_files",
]
