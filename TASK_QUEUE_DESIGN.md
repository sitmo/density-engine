# Task-Based State Management Design

## Overview
This document outlines a task-based approach for managing 100+ instances efficiently using a job queue system with timer-based state inspection and state transition tasks.

## Core Concept

Instead of having each instance check its own state periodically, we use a centralized task queue where:

1. **Timer-based tasks** inspect instance states at regular intervals
2. **State transition tasks** move instances between states based on inspection results
3. **Action tasks** perform specific operations (prepare, start job, collect results)
4. **Priority-based scheduling** ensures critical operations happen first

## Task Types

### 1. Inspection Tasks (Timer-based)
- **DISCOVER_INSTANCES**: Every 60s - Find new instances via API
- **CHECK_CONNECTIVITY**: Every 30s - Test SSH connections for CONNECTING instances
- **CHECK_PREPARATION**: Every 60s - Verify preparation status for CONNECTED instances
- **MONITOR_JOBS**: Every 10s - Check job status for RUNNING instances
- **CHECK_RESULTS**: Every 30s - Look for completed results for COMPLETING instances

### 2. State Transition Tasks (Event-driven)
- **TRANSITION_TO_CONNECTED**: When SSH connection succeeds
- **TRANSITION_TO_PREPARING**: When instance needs preparation
- **TRANSITION_TO_READY**: When preparation completes
- **TRANSITION_TO_RUNNING**: When job starts successfully
- **TRANSITION_TO_COMPLETING**: When job finishes
- **TRANSITION_TO_IDLE**: When results are collected
- **TRANSITION_TO_ERROR**: When any operation fails

### 3. Action Tasks (State-specific)
- **PREPARE_INSTANCE**: Install software and dependencies
- **ASSIGN_JOB**: Find and assign available job to idle instance
- **START_JOB**: Upload job file and start execution
- **COLLECT_RESULTS**: Download parquet files and clean up
- **RECOVER_INSTANCE**: Attempt to recover from error state

## Task Queue Architecture

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import heapq
import threading
import time

class TaskType(Enum):
    # Inspection tasks
    DISCOVER_INSTANCES = "discover_instances"
    CHECK_CONNECTIVITY = "check_connectivity"
    CHECK_PREPARATION = "check_preparation"
    MONITOR_JOBS = "monitor_jobs"
    CHECK_RESULTS = "check_results"
    
    # State transition tasks
    TRANSITION_TO_CONNECTED = "transition_to_connected"
    TRANSITION_TO_PREPARING = "transition_to_preparing"
    TRANSITION_TO_READY = "transition_to_ready"
    TRANSITION_TO_RUNNING = "transition_to_running"
    TRANSITION_TO_COMPLETING = "transition_to_completing"
    TRANSITION_TO_IDLE = "transition_to_idle"
    TRANSITION_TO_ERROR = "transition_to_error"
    
    # Action tasks
    PREPARE_INSTANCE = "prepare_instance"
    ASSIGN_JOB = "assign_job"
    START_JOB = "start_job"
    COLLECT_RESULTS = "collect_results"
    RECOVER_INSTANCE = "recover_instance"

class TaskPriority(Enum):
    CRITICAL = 1    # Error recovery, job completion
    HIGH = 2       # Job assignment, state transitions
    NORMAL = 3      # Regular inspections
    LOW = 4         # Discovery, cleanup

@dataclass
class Task:
    task_type: TaskType
    instance_id: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    scheduled_time: datetime = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.scheduled_time is None:
            self.scheduled_time = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class TaskQueue:
    def __init__(self):
        self.queue = []  # Priority queue (heap)
        self.lock = threading.Lock()
        self.task_handlers: Dict[TaskType, callable] = {}
        self.running = False
        self.worker_threads = []
        
    def add_task(self, task: Task):
        """Add a task to the queue."""
        with self.lock:
            # Use negative priority for min-heap behavior
            heapq.heappush(self.queue, (
                task.priority.value,
                task.scheduled_time.timestamp(),
                task
            ))
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute."""
        with self.lock:
            if not self.queue:
                return None
                
            priority, timestamp, task = heapq.heappop(self.queue)
            
            # Check if task is ready to execute
            if task.scheduled_time <= datetime.now():
                return task
            else:
                # Put task back in queue
                heapq.heappush(self.queue, (priority, timestamp, task))
                return None
    
    def schedule_periodic_tasks(self):
        """Schedule periodic inspection tasks."""
        current_time = datetime.now()
        
        # Schedule discovery task every 60 seconds
        if not self._has_recent_task(TaskType.DISCOVER_INSTANCES, 60):
            self.add_task(Task(
                task_type=TaskType.DISCOVER_INSTANCES,
                priority=TaskPriority.NORMAL,
                scheduled_time=current_time
            ))
        
        # Schedule connectivity checks every 30 seconds
        if not self._has_recent_task(TaskType.CHECK_CONNECTIVITY, 30):
            self.add_task(Task(
                task_type=TaskType.CHECK_CONNECTIVITY,
                priority=TaskPriority.NORMAL,
                scheduled_time=current_time
            ))
        
        # Schedule job monitoring every 10 seconds
        if not self._has_recent_task(TaskType.MONITOR_JOBS, 10):
            self.add_task(Task(
                task_type=TaskType.MONITOR_JOBS,
                priority=TaskPriority.HIGH,
                scheduled_time=current_time
            ))
    
    def _has_recent_task(self, task_type: TaskType, seconds: int) -> bool:
        """Check if a task of this type was scheduled recently."""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        
        with self.lock:
            for priority, timestamp, task in self.queue:
                if (task.task_type == task_type and 
                    task.scheduled_time > cutoff_time):
                    return True
        return False
    
    def register_handler(self, task_type: TaskType, handler: callable):
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
    
    def execute_task(self, task: Task) -> bool:
        """Execute a task using its registered handler."""
        handler = self.task_handlers.get(task.task_type)
        if not handler:
            return False
        
        try:
            return handler(task)
        except Exception as e:
            # Handle task failure
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.scheduled_time = datetime.now() + timedelta(seconds=30)
                self.add_task(task)
            return False
```

## Instance State Management

```python
class InstanceStateManager:
    def __init__(self):
        self.instances: Dict[str, InstanceStateInfo] = {}
        self.task_queue = TaskQueue()
        self._setup_task_handlers()
    
    def _setup_task_handlers(self):
        """Register handlers for all task types."""
        # Inspection task handlers
        self.task_queue.register_handler(
            TaskType.DISCOVER_INSTANCES, 
            self._handle_discover_instances
        )
        self.task_queue.register_handler(
            TaskType.CHECK_CONNECTIVITY, 
            self._handle_check_connectivity
        )
        self.task_queue.register_handler(
            TaskType.MONITOR_JOBS, 
            self._handle_monitor_jobs
        )
        
        # State transition handlers
        self.task_queue.register_handler(
            TaskType.TRANSITION_TO_CONNECTED, 
            self._handle_transition_to_connected
        )
        self.task_queue.register_handler(
            TaskType.TRANSITION_TO_PREPARING, 
            self._handle_transition_to_preparing
        )
        
        # Action handlers
        self.task_queue.register_handler(
            TaskType.PREPARE_INSTANCE, 
            self._handle_prepare_instance
        )
        self.task_queue.register_handler(
            TaskType.ASSIGN_JOB, 
            self._handle_assign_job
        )
    
    def _handle_discover_instances(self, task: Task) -> bool:
        """Discover new instances and add them to state management."""
        try:
            # Call existing discovery function
            active_instances = discover_active_instances()
            
            for instance in active_instances:
                instance_id = str(instance.contract_id)
                
                if instance_id not in self.instances:
                    # New instance - add to state management
                    self.instances[instance_id] = InstanceStateInfo(
                        instance_id=instance_id,
                        state=InstanceState.DISCOVERED,
                        metadata={
                            "ssh_host": instance.ssh_host,
                            "ssh_port": instance.ssh_port,
                            "last_discovered": datetime.now().isoformat()
                        }
                    )
                    
                    # Schedule connectivity check
                    self.task_queue.add_task(Task(
                        task_type=TaskType.CHECK_CONNECTIVITY,
                        instance_id=instance_id,
                        priority=TaskPriority.HIGH,
                        scheduled_time=datetime.now()
                    ))
                    
                    logger.info(f"Discovered new instance {instance_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to discover instances: {e}")
            return False
    
    def _handle_check_connectivity(self, task: Task) -> bool:
        """Check SSH connectivity for instances in CONNECTING state."""
        try:
            if not task.instance_id:
                # Check all instances in CONNECTING state
                connecting_instances = [
                    instance_id for instance_id, info in self.instances.items()
                    if info.state == InstanceState.CONNECTING
                ]
                
                for instance_id in connecting_instances:
                    self.task_queue.add_task(Task(
                        task_type=TaskType.CHECK_CONNECTIVITY,
                        instance_id=instance_id,
                        priority=TaskPriority.HIGH,
                        scheduled_time=datetime.now()
                    ))
                
                return True
            
            # Check specific instance
            instance_info = self.instances.get(task.instance_id)
            if not instance_info:
                return False
            
            # Attempt SSH connection
            ssh_client = create_ssh_connection(
                instance_info.metadata["ssh_host"],
                instance_info.metadata["ssh_port"]
            )
            ssh_client.connect()
            
            # Test connection
            result = execute_command(ssh_client, "echo 'test'", timeout=5)
            ssh_client.close()
            
            if result.success:
                # Schedule transition to CONNECTED
                self.task_queue.add_task(Task(
                    task_type=TaskType.TRANSITION_TO_CONNECTED,
                    instance_id=task.instance_id,
                    priority=TaskPriority.HIGH,
                    scheduled_time=datetime.now()
                ))
                return True
            else:
                # Stay in CONNECTING state, will retry
                return False
                
        except Exception as e:
            logger.error(f"Connectivity check failed for {task.instance_id}: {e}")
            return False
    
    def _handle_monitor_jobs(self, task: Task) -> bool:
        """Monitor jobs for instances in RUNNING state."""
        try:
            running_instances = [
                instance_id for instance_id, info in self.instances.items()
                if info.state == InstanceState.RUNNING
            ]
            
            for instance_id in running_instances:
                instance_info = self.instances[instance_id]
                current_job = instance_info.metadata.get("current_job")
                
                if not current_job:
                    continue
                
                # Check job completion
                instance_data = InstanceInfo(
                    contract_id=int(instance_id),
                    machine_id=0,
                    gpu_name="Unknown",
                    price_per_hour=0.0,
                    ssh_host=instance_info.metadata["ssh_host"],
                    ssh_port=instance_info.metadata["ssh_port"],
                    status="running",
                    public_ipaddr=instance_info.metadata["ssh_host"],
                    ports={},
                )
                
                completion_status = detect_job_completion(instance_data, current_job)
                
                if completion_status.completed:
                    # Schedule transition to COMPLETING
                    self.task_queue.add_task(Task(
                        task_type=TaskType.TRANSITION_TO_COMPLETING,
                        instance_id=instance_id,
                        priority=TaskPriority.CRITICAL,
                        scheduled_time=datetime.now()
                    ))
            
            return True
            
        except Exception as e:
            logger.error(f"Job monitoring failed: {e}")
            return False
    
    def _handle_transition_to_connected(self, task: Task) -> bool:
        """Transition instance to CONNECTED state."""
        try:
            instance_info = self.instances.get(task.instance_id)
            if not instance_info:
                return False
            
            instance_info.state = InstanceState.CONNECTED
            instance_info.last_updated = datetime.now()
            
            # Schedule preparation check
            self.task_queue.add_task(Task(
                task_type=TaskType.CHECK_PREPARATION,
                instance_id=task.instance_id,
                priority=TaskPriority.HIGH,
                scheduled_time=datetime.now()
            ))
            
            logger.info(f"Instance {task.instance_id} transitioned to CONNECTED")
            return True
            
        except Exception as e:
            logger.error(f"Transition to CONNECTED failed for {task.instance_id}: {e}")
            return False
    
    def _handle_assign_job(self, task: Task) -> bool:
        """Assign available job to idle instance."""
        try:
            # Find idle instances
            idle_instances = [
                instance_id for instance_id, info in self.instances.items()
                if info.state == InstanceState.IDLE
            ]
            
            if not idle_instances:
                return True  # No idle instances
            
            # Find available jobs
            available_jobs = self.state_manager.get_available_jobs()
            if not available_jobs:
                return True  # No available jobs
            
            # Assign job to first idle instance
            instance_id = idle_instances[0]
            job = available_jobs[0]
            
            # Update instance state
            instance_info = self.instances[instance_id]
            instance_info.state = InstanceState.ASSIGNED
            instance_info.metadata["current_job"] = job.job_file
            
            # Schedule job start
            self.task_queue.add_task(Task(
                task_type=TaskType.START_JOB,
                instance_id=instance_id,
                priority=TaskPriority.HIGH,
                scheduled_time=datetime.now()
            ))
            
            logger.info(f"Assigned job {job.job_file} to instance {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Job assignment failed: {e}")
            return False
    
    def start_worker_loop(self):
        """Start the task processing loop."""
        self.running = True
        
        # Start periodic task scheduler
        scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        scheduler_thread.start()
        
        # Start task workers
        for i in range(3):  # 3 worker threads
            worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            worker_thread.start()
            self.worker_threads.append(worker_thread)
    
    def _scheduler_loop(self):
        """Periodically schedule inspection tasks."""
        while self.running:
            try:
                self.task_queue.schedule_periodic_tasks()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)
    
    def _worker_loop(self):
        """Process tasks from the queue."""
        while self.running:
            try:
                task = self.task_queue.get_next_task()
                if task:
                    self.task_queue.execute_task(task)
                else:
                    time.sleep(0.1)  # No tasks ready, sleep briefly
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(1)
```

## Benefits of Task Queue Approach

1. **Efficient Resource Usage**: Only check instances that need checking
2. **Priority-based Processing**: Critical tasks (job completion) get priority
3. **Scalable**: Can handle 100+ instances with fixed worker threads
4. **Fault Tolerant**: Failed tasks can be retried automatically
5. **Easy Monitoring**: All tasks are logged and can be monitored
6. **Flexible Scheduling**: Different intervals for different types of checks

## Task Scheduling Intervals

- **DISCOVER_INSTANCES**: Every 60s
- **CHECK_CONNECTIVITY**: Every 30s (for CONNECTING instances)
- **CHECK_PREPARATION**: Every 60s (for CONNECTED instances)
- **MONITOR_JOBS**: Every 10s (for RUNNING instances)
- **CHECK_RESULTS**: Every 30s (for COMPLETING instances)
- **ASSIGN_JOB**: Every 5s (when jobs available)

## Implementation Benefits

1. **No Exponential Backoff**: Simple 30s retry for failed tasks
2. **Centralized Control**: All instance management through task queue
3. **Easy Testing**: Each task handler can be tested independently
4. **Clear Separation**: Inspection, transition, and action tasks are separate
5. **Efficient Scaling**: Worker threads can process tasks in parallel
