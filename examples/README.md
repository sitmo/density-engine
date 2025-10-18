# Density Engine Examples

This directory contains example scripts demonstrating how to use the density-engine system.

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r examples/requirements.txt
   ```

2. **Set up Vast.ai API**:
   - Make sure you have `vast.py` CLI installed and configured
   - Ensure you have sufficient credits in your Vast.ai account

3. **Configure SSH keys** (if needed):
   - Make sure your SSH keys are set up for Vast.ai instances

## Quick Start

1. **Test the API first** (recommended):
   ```bash
   python examples/test_vast_api.py
   ```

2. **Run the rental demo**:
   ```bash
   python examples/rent_and_monitor_instances.py
   ```

## Examples

### 1. Test Vast API

**File**: `test_vast_api.py`

**Description**: Test script to verify Vast API functions work correctly without renting instances.

**Features**:
- Tests GPU name validation
- Tests offer search functionality
- Tests best offer finding
- Tests instance listing
- No actual instance rental (safe to run)

**Usage**:
```bash
python examples/test_vast_api.py
```

### 2. Rent and Monitor Instances

**File**: `rent_and_monitor_instances.py`

**Description**: Demonstrates renting RTX 30+ instances and monitoring their status.

**Features**:
- Rents 2 RTX 30+ instances for max $0.06/hr
- Monitors instances for 3 minutes with status updates
- Displays instances in a nice table format
- Automatically cancels instances after monitoring period

**Usage**:
```bash
python examples/rent_and_monitor_instances.py
```

**Expected Output**:
```
🚀 RTX 30+ Instance Rental Demo
==================================================
🔍 Validating RTX 30+ GPU names...
✅ Found 5 valid RTX 30+ GPUs: ['RTX_3060', 'RTX_3060_Ti', 'RTX_3070', 'RTX_3070_Ti', 'RTX_3080']

🔍 Searching for 2 RTX 30+ instances at max $0.06/hr...

📋 Renting instance 1/2...
✅ Found: RTX 3060 at $0.045/hr (reliability: 95%, location: US)
🚀 Successfully rented instance: 12345

📋 Renting instance 2/2...
✅ Found: RTX 3070 at $0.052/hr (reliability: 98%, location: US)
🚀 Successfully rented instance: 12346

🎉 Successfully rented 2 instances!

⏱️  Monitoring 2 instances for 3 minutes...

⏰ Time remaining: 180s

📊 Our rented instances (2/2):
+-------+----------+-----------+-------------+----------+---------+----------+------+
| ID    | GPU      | Price     | Reliability | Location | Status  | CPU      | RAM  |
+=======+==========+===========+=============+==========+=========+==========+======+
| 12345 | RTX 3060 | $0.045/hr | 95%         | US       | running | 8 cores  | 12GB |
+-------+----------+-----------+-------------+----------+---------+----------+------+
| 12346 | RTX 3070 | $0.052/hr | 98%         | US       | running | 8 cores  | 8GB  |
+-------+----------+-----------+-------------+----------+---------+----------+------+

✅ Ready instances: 2/2

🗑️  Canceling 2 instances...
🗑️  Canceling instance 12345...
✅ Successfully canceled instance 12345
🗑️  Canceling instance 12346...
✅ Successfully canceled instance 12346

✅ Demo complete!
```

## Notes

- **Cost**: The script will rent instances for about 3 minutes. At $0.06/hr, this costs approximately $0.003 per instance.
- **GPU Selection**: The script uses the `RTX_30_PLUS` list from `density_engine.cluster.gpu_names`
- **Error Handling**: The script includes basic error handling for failed rentals
- **Monitoring**: Updates every 30 seconds during the monitoring period

## Customization

You can modify the script to:
- Change the number of instances to rent
- Adjust the maximum price per hour
- Modify the monitoring duration
- Use different GPU types
- Add more detailed monitoring (CPU usage, memory, etc.)
