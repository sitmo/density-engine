# Vast.ai Automation System Design

## Overview

This document outlines the software architecture for the vast.ai automation system. The design follows a layered, functional approach with clear separation of concerns and reusable primitive functions.

## Design Principles

1. **Layered Architecture**: Primitive functions → Composite functions → Orchestration
2. **Single Responsibility**: Each function has one clear purpose
3. **Composability**: Functions can be combined to build more complex behaviors
4. **Testability**: Each function can be tested independently
5. **Reusability**: Functions are designed to be reused across different contexts

## Directory Structure

```
density_engine/vast/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── ssh.py          # SSH connection and command execution
│   ├── files.py        # File operations (upload, download, move)
│   ├── state.py        # State management and persistence
│   └── jobs.py         # Job file operations and parsing
├── instances/
│   ├── __init__.py
│   ├── discovery.py    # Instance discovery from vast.ai API
│   ├── preparation.py  # Instance preparation and dependency checking
│   ├── monitoring.py   # Instance health and status monitoring
│   └── lifecycle.py    # Instance lifecycle management
├── execution/
│   ├── __init__.py
│   ├── job_runner.py   # Job execution and monitoring
│   ├── process_monitor.py  # Process detection and monitoring
│   └── result_handler.py   # Result collection and processing
├── orchestration/
│   ├── __init__.py
│   ├── scheduler.py    # Task scheduling and queue management
│   ├── coordinator.py  # High-level orchestration logic
│   └── workflow.py     # Complete workflow definitions
└── utils/
    ├── __init__.py
    ├── config.py       # Configuration management
    ├── logging.py      # Logging utilities
    └── exceptions.py   # Custom exceptions
```

## Core Module (`density_engine/vast/core/`)

### `ssh.py` - SSH Operations
```python
# Primitive SSH functions
def create_ssh_connection(host: str, port: int, username: str = "root") -> SSHClient
def execute_command(client: SSHClient, command: str, timeout: int = 30) -> CommandResult
def upload_file(client: SSHClient, local_path: str, remote_path: str) -> bool
def download_file(client: SSHClient, remote_path: str, local_path: str) -> bool
def test_ssh_connectivity(host: str, port: int) -> bool

# Composite SSH functions
def with_ssh_connection(host: str, port: int, func: Callable) -> Any
def execute_with_retry(client: SSHClient, command: str, max_retries: int = 3) -> CommandResult
```

### `files.py` - File Operations
```python
# Primitive file functions
def move_file(source: Path, destination: Path) -> bool
def copy_file(source: Path, destination: Path) -> bool
def ensure_directory_exists(path: Path) -> bool
def list_files(directory: Path, pattern: str = "*") -> List[Path]

# Composite file functions
def move_job_file(job_file: str, from_state: JobState, to_state: JobState) -> bool
def atomic_file_move(source: Path, destination: Path) -> bool
def cleanup_temp_files(directory: Path) -> int
```

### `state.py` - State Management
```python
# State data structures
@dataclass
class InstanceState:
    contract_id: str
    ssh_host: str
    ssh_port: int
    status: InstanceStatus
    last_updated: datetime
    job_state: JobState
    current_job: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class JobState:
    job_file: str
    state: JobState
    assigned_instance: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    metadata: Dict[str, Any]

# Primitive state functions
def load_state(file_path: Path) -> Dict[str, Any]
def save_state(state: Dict[str, Any], file_path: Path) -> bool
def update_instance_state(instance_id: str, updates: Dict[str, Any]) -> bool
def update_job_state(job_file: str, updates: Dict[str, Any]) -> bool

# Composite state functions
def get_idle_instances() -> List[InstanceState]
def get_available_jobs() -> List[JobState]
def mark_job_assigned(job_file: str, instance_id: str) -> bool
def mark_job_completed(job_file: str, success: bool) -> bool
```

### `jobs.py` - Job Operations
```python
# Primitive job functions
def parse_job_file(job_file: Path) -> Dict[str, Any]
def validate_job_file(job_file: Path) -> bool
def get_job_arguments(job_file: Path) -> Dict[str, Any]
def estimate_job_duration(job_file: Path) -> timedelta

# Composite job functions
def discover_job_files(directory: Path) -> List[Path]
def categorize_jobs_by_type(job_files: List[Path]) -> Dict[str, List[Path]]
def prioritize_jobs(job_files: List[Path]) -> List[Path]
```

## Instances Module (`density_engine/vast/instances/`)

### `discovery.py` - Instance Discovery
```python
# Primitive discovery functions
def fetch_instances_from_api() -> List[Dict[str, Any]]
def parse_instance_data(raw_data: Dict[str, Any]) -> InstanceInfo
def filter_instances_by_criteria(instances: List[InstanceInfo], criteria: Dict[str, Any]) -> List[InstanceInfo]

# Composite discovery functions
def discover_active_instances() -> List[InstanceInfo]
def discover_new_instances() -> List[InstanceInfo]
def get_instance_by_id(instance_id: str) -> Optional[InstanceInfo]
```

### `preparation.py` - Instance Preparation
```python
# Primitive preparation functions
def check_torch_installation(ssh_client: SSHClient) -> bool
def check_density_engine_installation(ssh_client: SSHClient) -> bool
def install_missing_dependencies(ssh_client: SSHClient, dependencies: List[str]) -> bool
def clone_repository(ssh_client: SSHClient, repo_url: str, target_dir: str) -> bool

# Composite preparation functions
def prepare_instance_for_jobs(instance: InstanceInfo) -> bool
def verify_instance_readiness(instance: InstanceInfo) -> bool
def setup_development_environment(instance: InstanceInfo) -> bool
```

### `monitoring.py` - Instance Monitoring
```python
# Primitive monitoring functions
def check_instance_health(instance: InstanceInfo) -> HealthStatus
def get_instance_utilization(instance: InstanceInfo) -> UtilizationMetrics
def check_disk_space(instance: InstanceInfo) -> DiskSpaceInfo
def check_memory_usage(instance: InstanceInfo) -> MemoryInfo

# Composite monitoring functions
def monitor_instance_status(instance: InstanceInfo) -> InstanceStatus
def detect_instance_issues(instance: InstanceInfo) -> List[Issue]
def get_instance_summary(instance: InstanceInfo) -> InstanceSummary
```

### `lifecycle.py` - Instance Lifecycle
```python
# Primitive lifecycle functions
def start_instance(instance_id: str) -> bool
def stop_instance(instance_id: str) -> bool
def restart_instance(instance_id: str) -> bool
def terminate_instance(instance_id: str) -> bool

# Composite lifecycle functions
def manage_instance_lifecycle(instance: InstanceInfo, desired_state: InstanceStatus) -> bool
def handle_instance_failure(instance: InstanceInfo) -> bool
def cleanup_failed_instance(instance: InstanceInfo) -> bool
```

## Execution Module (`density_engine/vast/execution/`)

### `job_runner.py` - Job Execution
```python
# Primitive job execution functions
def upload_job_file(instance: InstanceInfo, job_file: Path) -> bool
def start_job_process(instance: InstanceInfo, job_file: str, args: Dict[str, Any]) -> ProcessInfo
def generate_job_command(job_file: str, args: Dict[str, Any]) -> str
def create_job_log_file(instance: InstanceInfo, job_file: str) -> str

# Composite job execution functions
def execute_job_on_instance(instance: InstanceInfo, job_file: Path) -> bool
def run_job_with_monitoring(instance: InstanceInfo, job_file: Path) -> JobExecutionResult
def handle_job_execution_error(instance: InstanceInfo, job_file: str, error: Exception) -> bool
```

### `process_monitor.py` - Process Monitoring
```python
# Primitive process monitoring functions
def find_process_by_name(instance: InstanceInfo, process_name: str) -> List[ProcessInfo]
def check_process_running(instance: InstanceInfo, process_id: str) -> bool
def get_process_output(instance: InstanceInfo, log_file: str) -> str
def parse_job_log(log_content: str) -> JobLogInfo

# Composite process monitoring functions
def monitor_job_execution(instance: InstanceInfo, job_file: str) -> JobStatus
def detect_job_completion(instance: InstanceInfo, job_file: str) -> CompletionStatus
def handle_job_timeout(instance: InstanceInfo, job_file: str) -> bool
```

### `result_handler.py` - Result Processing
```python
# Primitive result functions
def find_result_files(instance: InstanceInfo, pattern: str = "*.parquet") -> List[str]
def download_result_file(instance: InstanceInfo, remote_path: str, local_path: Path) -> bool
def validate_result_file(file_path: Path) -> bool
def parse_result_file(file_path: Path) -> ResultData

# Composite result functions
def collect_job_results(instance: InstanceInfo, job_file: str) -> List[Path]
def process_job_results(job_file: str, result_files: List[Path]) -> ProcessedResults
def archive_completed_job(job_file: str, results: ProcessedResults) -> bool
```

## Orchestration Module (`density_engine/vast/orchestration/`)

### `scheduler.py` - Task Scheduling
```python
# Primitive scheduling functions
def create_task(task_type: TaskType, **kwargs) -> Task
def schedule_task(task: Task, priority: int = 0) -> bool
def get_next_task() -> Optional[Task]
def mark_task_completed(task_id: str) -> bool

# Composite scheduling functions
def schedule_instance_preparation(instance: InstanceInfo) -> bool
def schedule_job_execution(instance: InstanceInfo, job_file: str) -> bool
def schedule_result_collection(instance: InstanceInfo, job_file: str) -> bool
def reschedule_failed_task(task: Task, delay: timedelta) -> bool
```

### `coordinator.py` - High-level Coordination
```python
# Primitive coordination functions
def assign_job_to_instance(job_file: str, instance: InstanceInfo) -> bool
def find_best_instance_for_job(job_file: str) -> Optional[InstanceInfo]
def balance_workload_across_instances() -> Dict[str, int]
def detect_stuck_jobs() -> List[str]

# Composite coordination functions
def coordinate_job_distribution() -> int
def coordinate_instance_preparation() -> int
def coordinate_result_collection() -> int
def handle_system_failures() -> int
```

### `workflow.py` - Complete Workflows
```python
# Complete workflow functions
def run_complete_job_workflow(job_file: str) -> WorkflowResult
def run_instance_preparation_workflow(instance: InstanceInfo) -> WorkflowResult
def run_result_collection_workflow(instance: InstanceInfo, job_file: str) -> WorkflowResult
def run_system_maintenance_workflow() -> WorkflowResult

# High-level orchestration
def orchestrate_job_processing() -> OrchestrationResult
def orchestrate_instance_management() -> OrchestrationResult
def orchestrate_system_health() -> OrchestrationResult
```

## Utils Module (`density_engine/vast/utils/`)

### `config.py` - Configuration Management
```python
# Configuration functions
def load_config(config_file: Path) -> Dict[str, Any]
def get_config_value(key: str, default: Any = None) -> Any
def update_config(key: str, value: Any) -> bool
def validate_config(config: Dict[str, Any]) -> List[str]
```

### `logging.py` - Logging Utilities
```python
# Logging functions
def setup_logging(log_level: str = "INFO") -> None
def get_logger(name: str) -> Logger
def log_function_call(func: Callable) -> Callable
def log_execution_time(func: Callable) -> Callable
```

### `exceptions.py` - Custom Exceptions
```python
# Custom exception classes
class VastAIError(Exception): pass
class InstanceError(VastAIError): pass
class JobExecutionError(VastAIError): pass
class SSHConnectionError(VastAIError): pass
class FileOperationError(VastAIError): pass
class StateManagementError(VastAIError): pass
```

## Usage Examples

### Simple Job Execution
```python
from density_engine.vast.orchestration.workflow import run_complete_job_workflow

# Execute a single job
result = run_complete_job_workflow("garch_test_job_0000_0099.csv")
```

### Instance Management
```python
from density_engine.vast.instances.discovery import discover_active_instances
from density_engine.vast.instances.preparation import prepare_instance_for_jobs

# Discover and prepare instances
instances = discover_active_instances()
for instance in instances:
    prepare_instance_for_jobs(instance)
```

### Custom Orchestration
```python
from density_engine.vast.orchestration.coordinator import coordinate_job_distribution
from density_engine.vast.orchestration.scheduler import schedule_task

# Custom orchestration logic
jobs_assigned = coordinate_job_distribution()
schedule_task(create_task(TaskType.COLLECT_RESULTS))
```

## Benefits of This Design

1. **Modularity**: Each module has a clear, focused responsibility
2. **Testability**: Functions can be tested independently
3. **Reusability**: Primitive functions can be composed into complex behaviors
4. **Maintainability**: Clear separation of concerns makes code easier to understand and modify
5. **Extensibility**: New functionality can be added without affecting existing code
6. **Debugging**: Issues can be isolated to specific modules and functions

## Migration Strategy

1. **Phase 1**: Extract core primitive functions from existing orchestrator
2. **Phase 2**: Implement instance management functions
3. **Phase 3**: Implement job execution and monitoring functions
4. **Phase 4**: Implement orchestration layer
5. **Phase 5**: Replace existing orchestrator with new modular system
6. **Phase 6**: Add comprehensive testing and documentation

This design provides a solid foundation for a maintainable, scalable vast.ai automation system.
