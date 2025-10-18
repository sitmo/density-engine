# Density Engine - Professional Computational Finance Platform

A professional, production-ready platform for Monte Carlo simulation of financial models with distributed computing capabilities and automated dataset generation.

## 🚀 Features

- **Multiple Stochastic Volatility Models**: GARCH, Rough Heston, Rough Bergomi
- **Professional Task Queue**: Redis + Celery with persistence and automatic recovery
- **Distributed Computing**: Full lifecycle management of cloud GPU instances (Vast.ai)
- **Automated Dataset Generation**: Professional HuggingFace datasets with proper schemas
- **Robust Error Handling**: Automatic retry, failure recovery, and state management
- **Scalable Architecture**: Support for 100+ concurrent instances

## 📁 Project Structure

```
density-engine/
├── density_engine/           # Core package
│   ├── models/              # Monte Carlo models (protocol-based)
│   │   ├── base.py          # MonteCarloModelProtocol definition
│   │   ├── registry.py      # Model registration system
│   │   ├── garch.py         # GARCH model implementation
│   │   ├── rough_heston.py  # Rough Heston model
│   │   └── rough_bergomi.py # Rough Bergomi model
│   ├── tasks/               # Celery task definitions
│   │   ├── celery_app.py    # Celery configuration
│   │   ├── jobs.py          # Job management tasks
│   │   ├── instances.py     # Instance management tasks
│   │   ├── cluster.py       # Cluster lifecycle tasks
│   │   └── recovery.py      # State recovery mechanism
│   ├── cluster/             # Enhanced cluster management
│   │   ├── cluster_manager.py
│   │   ├── instance_manager.py
│   │   ├── ssh_client.py
│   │   └── lifecycle.py     # Instance lifecycle management
│   └── torch.py             # PyTorch utilities
├── datasets/                # Multi-dataset structure
│   ├── garch/
│   │   ├── config/          # Model configuration
│   │   ├── raw_outputs/     # Raw parquet files
│   │   ├── hf/              # HuggingFace dataset
│   │   └── logs/            # Job logs
│   ├── rough_heston/        # Same structure
│   └── rough_bergomi/       # Same structure
├── scripts/                 # Utility scripts
│   ├── run_simulation_worker.py    # Generic simulation worker
│   ├── merge_outputs_enhanced.py  # Multi-model dataset merger
│   └── deploy/              # Deployment scripts
├── config/                  # Configuration files
└── notebooks/               # Jupyter notebooks
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Redis server
- Poetry (for dependency management)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd density-engine
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Start Redis server**:
   ```bash
   redis-server --appendonly yes --appendfsync everysec
   ```

4. **Set environment variables**:
   ```bash
   export VAST_API_KEY="your-vast-api-key"
   export REDIS_HOST="localhost"
   export REDIS_PORT="6379"
   ```

## 🚀 Quick Start

### 1. Start the System

```bash
# Start all components
./scripts/deploy/start_system.sh

# Or start components individually:
# Start Celery workers
celery -A density_engine.tasks.celery_app worker --loglevel=info

# Start Celery beat scheduler
celery -A density_engine.tasks.celery_app beat --loglevel=info

# Start cluster manager
python scripts/manage_cluster.py
```

### 2. Submit a Simulation Job

```python
from density_engine.tasks.jobs import submit_simulation_job

# Submit a GARCH simulation job
result = submit_simulation_job.delay(
    model_name="garch",
    job_csv_path="datasets/garch/jobs/garch_job_001.csv",
    instance_id="instance_001",
    job_params={
        "num_sim": 10_000_000,
        "num_quantiles": 512,
    }
)

print(f"Job submitted: {result.get()}")
```

### 3. Run Local Simulation

```python
from density_engine.models import get_model_class

# Get GARCH model
GARCH = get_model_class("gjrgarch_normalized")

# Create model instance
model = GARCH(
    alpha=0.05,
    gamma=0.1,
    beta=0.9,
    eta=10.0,
    lam=0.0,
    device="cuda"  # Use GPU if available
)

# Run simulation
model.reset(num_paths=1_000_000)
quantiles = model.path_quantiles([1, 5, 10, 20, 50, 100])

print(f"Generated quantiles shape: {quantiles.shape}")
```

### 4. Create HuggingFace Dataset

```bash
# Process GARCH outputs
python scripts/merge_outputs_enhanced.py garch

# Process Rough Heston outputs
python scripts/merge_outputs_enhanced.py rough_heston

# Process Rough Bergomi outputs
python scripts/merge_outputs_enhanced.py rough_bergomi
```

## 📊 Models

### GARCH (Generalized Autoregressive Conditional Heteroskedasticity)

```python
from density_engine.models import GJRGARCHNormalized

model = GJRGARCHNormalized(
    alpha=0.05,      # GARCH alpha parameter
    gamma=0.1,       # Leverage effect parameter
    beta=0.9,        # GARCH beta parameter
    sigma0_sq=1.0,   # Initial variance
    eta=10.0,        # Degrees of freedom
    lam=0.0,         # Skewness parameter
    device="cuda"    # Use GPU
)
```

### Rough Heston

```python
from density_engine.models import RoughHeston

model = RoughHeston(
    kappa=2.0,       # Mean reversion speed
    theta=0.04,      # Long-term variance
    sigma=0.3,       # Volatility of volatility
    rho=-0.7,        # Correlation
    alpha=0.5,       # Roughness parameter
    v0=0.04,         # Initial variance
    device="cuda"    # Use GPU
)
```

### Rough Bergomi

```python
from density_engine.models import RoughBergomi

model = RoughBergomi(
    eta=1.9,         # Volatility of volatility
    rho=-0.9,        # Correlation
    hurst=0.1,       # Hurst parameter
    xi0=0.04,        # Initial forward variance
    device="cuda"    # Use GPU
)
```

## 🔧 Configuration

### Model Configuration

Each model has its own configuration file in `datasets/{model}/config/model_config.json`:

```json
{
  "model_name": "garch",
  "description": "GARCH model description",
  "default_parameters": {
    "num_sim": 10000000,
    "num_quantiles": 512
  },
  "model_parameters": {
    "alpha": {
      "description": "GARCH alpha parameter",
      "type": "float",
      "range": [0.001, 0.1]
    }
  },
  "huggingface": {
    "dataset_name": "garch-densities",
    "description": "GARCH density dataset",
    "tags": ["garch", "finance", "density-estimation"]
  }
}
```

### System Configuration

Edit `config/production.yaml`:

```yaml
redis:
  host: localhost
  port: 6379

celery:
  worker_concurrency: 4
  max_tasks_per_child: 100

cluster:
  max_instances: 100
  target_instances: 50
```

## 📈 Monitoring

### Celery Monitoring

```bash
# Monitor Celery workers
celery -A density_engine.tasks.celery_app inspect active

# Monitor task queues
celery -A density_engine.tasks.celery_app inspect stats

# Monitor scheduled tasks
celery -A density_engine.tasks.celery_app inspect scheduled
```

### Redis Monitoring

```bash
# Check Redis status
redis-cli ping

# Monitor Redis operations
redis-cli monitor

# Check memory usage
redis-cli info memory
```

### Cluster Status

```python
from density_engine.tasks.cluster import get_cluster_status

status = get_cluster_status.delay()
print(status.get())
```

## 🔄 Task Queue System

### Job Management Tasks

- `submit_simulation_job`: Submit new simulation jobs
- `execute_job_on_instance`: Execute jobs on instances
- `monitor_job_progress`: Monitor running jobs
- `collect_job_results`: Collect completed job results
- `handle_job_failure`: Handle job failures

### Instance Management Tasks

- `discover_instances`: Discover available instances
- `prepare_instance`: Prepare instances for jobs
- `health_check_instance`: Check instance health
- `cleanup_instance`: Clean up instances

### Cluster Lifecycle Tasks

- `rent_new_instance`: Rent new instances from marketplace
- `destroy_instance`: Destroy instances
- `rebalance_cluster`: Rebalance cluster size
- `monitor_cluster_health`: Monitor overall cluster health

## 🛡️ Error Handling & Recovery

### Automatic Recovery

The system includes comprehensive recovery mechanisms:

1. **State Recovery**: Reconciles Redis state with actual instance state on restart
2. **Task Retry**: Automatic retry for transient failures
3. **Instance Health Monitoring**: Continuous health checks and automatic replacement
4. **Job Failure Handling**: Automatic cleanup and rescheduling

### Manual Recovery

```python
from density_engine.tasks.recovery import recover_system_state

# Recover system state after restart
recovery_result = recover_system_state.delay()
print(recovery_result.get())
```

## 📚 API Reference

### Model Protocol

All models implement the `MonteCarloModelProtocol`:

```python
class MonteCarloModelProtocol(Protocol):
    def __init__(self, device: Union[str, torch.device, None] = None, **params: Union[float, int]) -> None: ...
    def reset(self, num_paths: int) -> None: ...
    def path_quantiles(self, t: List[int], **kwargs) -> torch.Tensor: ...
    @property
    def parameter_dict(self) -> Dict[str, float]: ...
    @property
    def model_name(self) -> str: ...
```

### Task API

```python
# Submit job
result = submit_simulation_job.delay(
    model_name="garch",
    job_csv_path="path/to/job.csv",
    instance_id="instance_001"
)

# Monitor job
status = monitor_job_progress.delay(result["job_id"], "instance_001")

# Collect results
results = collect_job_results.delay(result["job_id"], "instance_001")
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=density_engine tests/
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Support

- **Issues**: GitHub Issues
- **Documentation**: [Wiki](link-to-wiki)
- **Discussions**: GitHub Discussions

## 🏆 Acknowledgments

- Built with PyTorch for high-performance computing
- Uses Celery for distributed task processing
- Integrates with Vast.ai for cloud GPU access
- Publishes datasets to HuggingFace Hub
