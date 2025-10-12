# Vast.ai Automation Scripts

This directory contains clean, minimal scripts for automating vast.ai operations using the new modular design.

## Scripts

### Core Scripts

- **`find_machines.py`** - Find suitable machines on vast.ai
- **`rent_machine.py`** - Rent a machine using vast.ai CLI  
- **`manage.py`** - Unified management interface
- **`orchestrator.py`** - Automated orchestrator using the new modular design

## Usage

### Find Machines
```bash
# Find machines under $0.2/hour with 8GB+ GPU memory
poetry run python scripts/vast/find_machines.py --price-max 0.2 --gpu-memory 8

# Find machines and output as JSON
poetry run python scripts/vast/find_machines.py --price-max 0.3 --json

# Find machines with debug output
poetry run python scripts/vast/find_machines.py --price-max 0.2 --debug
```

### Rent Machine
```bash
# Rent machine 35055 with default 50GB disk
poetry run python scripts/vast/rent_machine.py 35055

# Rent machine with custom disk size
poetry run python scripts/vast/rent_machine.py 35055 --disk 100

# Rent machine and wait for it to be ready
poetry run python scripts/vast/rent_machine.py 35055 --wait

# Rent machine with debug output
poetry run python scripts/vast/rent_machine.py 35055 --debug
```

### Manage Instances
```bash
# Find suitable machines
poetry run python scripts/vast/manage.py --find --price-max 0.2 --gpu-memory 8

# Rent a machine
poetry run python scripts/vast/manage.py --rent 35055 --disk 50

# Show all instances
poetry run python scripts/vast/manage.py --list

# Prepare an instance
poetry run python scripts/vast/manage.py --prepare 12345

# Run jobs on an instance
poetry run python scripts/vast/manage.py --run-jobs 12345 garch_test_jobs.csv

# Download results
poetry run python scripts/vast/manage.py --download 12345
```

### Run Orchestrator
```bash
# Run for 10 iterations
poetry run python scripts/vast/orchestrator.py --run --max-iterations 10

# Show status
poetry run python scripts/vast/orchestrator.py --status

# Run indefinitely
poetry run python scripts/vast/orchestrator.py --run
```

## Design Principles

All scripts follow these clean design principles:

1. **Minimal and Focused**: Each script has a single, clear purpose
2. **Modular**: Uses the new `density_engine/vast/` modular architecture
3. **No Wrapper Functions**: Direct use of primitive functions from the modules
4. **Clean Error Handling**: Proper exception handling and logging
5. **Consistent Interface**: Similar command-line patterns across all scripts
6. **No Legacy Code**: Completely rebuilt using the new design

## Architecture

The scripts use the clean, modular architecture located in `density_engine/vast/`:

- **Core modules**: SSH operations, file management, state management, job operations
- **Instance modules**: Discovery, preparation, monitoring, lifecycle management  
- **Execution modules**: Job running, process monitoring, result handling
- **Orchestration modules**: Task scheduling, coordination, workflow management
- **Utility modules**: Configuration, logging, exceptions

See `VAST.md` for the complete design documentation.

## Requirements

- Poetry environment with dependencies installed
- `VAST_API_KEY` environment variable set
- SSH keys configured for passwordless access
- `vast.py` CLI tool installed and configured