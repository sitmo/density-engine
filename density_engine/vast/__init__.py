"""
Vast.ai automation system.

This package provides a clean, modular interface for automating vast.ai operations.
"""

__version__ = "1.0.0"
__author__ = "Density Engine Team"

from .core.state import get_state_manager
from .orchestration.coordinator import get_coordinator
from .orchestration.scheduler import get_scheduler

# Import main components for easy access
from .orchestration.workflow import get_workflow_manager
from .utils.config import load_config
from .utils.logging import setup_logging

__all__ = [
    "get_workflow_manager",
    "get_coordinator",
    "get_scheduler",
    "get_state_manager",
    "load_config",
    "setup_logging",
]
