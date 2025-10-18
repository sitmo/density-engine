# Cluster Management Module

This module provides generic cluster management functionality for density-engine.

## Structure

- **`cluster_manager.py`** - Global cluster coordination and instance discovery
- **`instance_manager.py`** - Per-instance management and job execution
- **`job_manager.py`** - Job file management and assignment
- **`ssh_client.py`** - SSH connection management with retry logic

## Usage

The cluster module is designed to be imported by specific cluster management scripts.
See `scripts/manage_cluster.py` for an example implementation.

## Key Features

- **Stateless Design** - Queries facts and takes actions based on current reality
- **Adaptive Timing** - 10-second minimum interval between instance checks
- **Automatic Reconnection** - SSH connections are automatically recreated on failure
- **Job Assignment** - Random job assignment from input queue
- **Error Recovery** - Failed jobs are returned to queue for retry
- **Output Management** - Automatic download of completed job outputs

## Configuration

Project-specific configuration is defined in the management script:
- SSH key paths
- Job parameters
- Directory paths
- API commands
- Timeouts and intervals
