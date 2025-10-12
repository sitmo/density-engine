# density-engine
Pretrained Simulation based Density Models for Finance

## Quick Start

### Local Usage
```bash
# Install dependencies
poetry install

# Run example notebook
jupyter notebook notebooks/example_garch.ipynb
```

### Cloud Computing (Vast.ai)
For large-scale GARCH simulations, use our automated vast.ai integration:

```bash
# Setup vast.ai automation
cd scripts/vast && ./setup_vast.sh

# Run jobs on cloud GPU
python run_vast.py --args "jobs/garch_test_jobs.csv --start 0 --end 100 --stride 10"
```

See [scripts/vast/README.md](scripts/vast/README.md) for detailed documentation.
