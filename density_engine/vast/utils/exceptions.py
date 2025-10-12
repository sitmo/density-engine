"""
Custom exceptions for the vast.ai automation system.
"""


class VastAIError(Exception):
    """Base exception for all vast.ai automation errors."""

    pass


class InstanceError(VastAIError):
    """Exception raised for instance-related errors."""

    pass


class JobExecutionError(VastAIError):
    """Exception raised for job execution errors."""

    pass


class SSHConnectionError(VastAIError):
    """Exception raised for SSH connection errors."""

    pass


class FileOperationError(VastAIError):
    """Exception raised for file operation errors."""

    pass


class StateManagementError(VastAIError):
    """Exception raised for state management errors."""

    pass


class ConfigurationError(VastAIError):
    """Exception raised for configuration errors."""

    pass
