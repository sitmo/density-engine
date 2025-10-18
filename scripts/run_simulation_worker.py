#!/usr/bin/env python3
"""
Generic simulation worker script for all Monte Carlo models.

This script can run simulations for any model that implements the MonteCarloModel protocol.
It reads job parameters from CSV files and generates quantile outputs.

Usage:
    python run_simulation_worker.py <model_name> <job_csv_path> [options]

Arguments:
    model_name: Name of the model to use (gjrgarch_normalized, gjrgarch, rough_heston, rough_bergomi)
    job_csv_path: Path to CSV file containing job parameters
    --num_sim: Number of simulations (default from model config)
    --num_quantiles: Number of quantiles (default 512)
    --stride: Stride for processing rows (default 1)
    --output_dir: Output directory (default: datasets/{model}/raw_outputs/)
"""

import argparse
import csv
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from density_engine.models import get_model_class, get_model_names
from density_engine.torch import get_best_device


def load_model_config(model_name: str) -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    config_path = project_root / "datasets" / model_name / "config" / "model_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def make_times() -> List[int]:
    """Generate time points for simulation."""
    # Random points in time (same as original script)
    times = set(np.concatenate([
        np.arange(20),
        np.random.randint(20, 100, size=20),
        np.random.randint(100, 1000, size=20)
    ]) + 1)
    
    times = sorted(times)
    times = [int(t) for t in times]
    times[-1] = 1000
    return times


def parse_job_csv(csv_path: Path, model_name: str) -> List[Dict[str, Any]]:
    """Parse job CSV file and return list of parameter dictionaries."""
    config = load_model_config(model_name)
    model_params = config["model_parameters"]
    
    # Get expected parameter names
    expected_params = list(model_params.keys())
    
    jobs = []
    
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row_num, row in enumerate(reader):
            # Convert string values to appropriate types
            job_params = {}
            for param_name in expected_params:
                if param_name in row:
                    param_type = model_params[param_name]["type"]
                    if param_type == "float":
                        job_params[param_name] = float(row[param_name])
                    elif param_type == "int":
                        job_params[param_name] = int(row[param_name])
                    else:
                        job_params[param_name] = row[param_name]
                else:
                    # Use default value if available
                    if "default" in model_params[param_name]:
                        job_params[param_name] = model_params[param_name]["default"]
                    else:
                        raise ValueError(f"Missing required parameter: {param_name}")
            
            # Add job metadata
            job_params["job_row"] = row_num
            job_params["p"] = float(row.get("p", np.random.random()))  # Random splitting parameter
            
            jobs.append(job_params)
    
    return jobs


def run_simulation(
    model_name: str,
    job_params: Dict[str, Any],
    num_sim: int,
    num_quantiles: int,
    times: List[int],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run simulation for a single job."""
    # Get model class
    model_class = get_model_class(model_name)
    if not model_class:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create model instance
    model = model_class(device=device, **job_params)
    
    # Reset simulation state
    model.reset(num_sim)
    
    # Run simulation
    quantiles = model.path_quantiles(times, size=num_quantiles, normalize=True)
    
    # Convert to CPU numpy for output
    quantiles_np = quantiles.cpu().numpy()
    
    # Create output records
    records = []
    for j, t in enumerate(times):
        record = {
            **model.parameter_dict,  # Model parameters
            "ti": t,
            "p": job_params["p"],
            "x": quantiles_np[j, :].astype(np.float32).tolist(),
        }
        records.append(record)
    
    return records


def save_results(
    records: List[Dict[str, Any]],
    model_name: str,
    job_row: int,
    output_dir: Path,
    job_params: Dict[str, Any],
) -> Path:
    """Save simulation results to parquet file."""
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Convert float64 columns to float32 to save space
    float64_cols = df.select_dtypes(include=[np.float64]).columns
    df[float64_cols] = df[float64_cols].astype(np.float32)
    
    # Generate filename
    timestamp = int(time.time())
    filename = f"{model_name}_job_{job_row}_{timestamp}_{len(records)}_{512}.parquet"
    output_path = output_dir / filename
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    
    return output_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generic simulation worker for Monte Carlo models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'model_name',
        type=str,
        help='Name of the model to use (gjrgarch_normalized, gjrgarch, rough_heston, rough_bergomi)'
    )
    
    parser.add_argument(
        'job_csv_path',
        type=str,
        help='Path to CSV file containing job parameters'
    )
    
    parser.add_argument(
        '--num_sim',
        type=int,
        default=None,
        help='Number of simulations (default from model config)'
    )
    
    parser.add_argument(
        '--num_quantiles',
        type=int,
        default=512,
        help='Number of quantiles (default: 512)'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Stride for processing rows (default: 1)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: datasets/{model}/raw_outputs/)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='PyTorch device (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Validate model name
    available_models = get_model_names()
    if args.model_name not in available_models:
        print(f"Error: Unknown model '{args.model_name}'", file=sys.stderr)
        print(f"Available models: {', '.join(available_models)}", file=sys.stderr)
        sys.exit(1)
    
    # Load model configuration
    try:
        config = load_model_config(args.model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Set defaults from config
    if args.num_sim is None:
        args.num_sim = config["default_parameters"]["num_sim"]
    
    if args.output_dir is None:
        args.output_dir = project_root / "datasets" / args.model_name / "raw_outputs"
    else:
        args.output_dir = Path(args.output_dir)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_best_device()
    
    print(f"Model: {args.model_name}")
    print(f"Job CSV: {args.job_csv_path}")
    print(f"Number of simulations: {args.num_sim:,}")
    print(f"Number of quantiles: {args.num_quantiles}")
    print(f"Stride: {args.stride}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print("-" * 50)
    
    # Parse job CSV
    try:
        jobs = parse_job_csv(Path(args.job_csv_path), args.model_name)
    except Exception as e:
        print(f"Error parsing job CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(jobs)} jobs to process")
    
    # Generate time points
    times = make_times()
    print(f"Time points: {times}")
    
    # Process jobs
    processed_count = 0
    all_records = []
    
    for i, job_params in enumerate(jobs):
        if i % args.stride != 0:
            continue  # Skip based on stride
        
        print(f"Processing job {i}: {job_params}")
        
        try:
            # Run simulation
            records = run_simulation(
                args.model_name,
                job_params,
                args.num_sim,
                args.num_quantiles,
                times,
                device,
            )
            
            all_records.extend(records)
            processed_count += 1
            
            print(f"  Generated {len(records)} records")
            
        except Exception as e:
            print(f"  Error processing job {i}: {e}", file=sys.stderr)
            continue
    
    # Save all results to single file
    if all_records:
        output_path = save_results(
            all_records,
            args.model_name,
            0,  # Single file for all jobs
            args.output_dir,
            {"p": 0.5},  # Dummy p value
        )
        
        print(f"Saved {len(all_records)} records to {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    print("-" * 50)
    print(f"Processed {processed_count} jobs")
    print(f"Total records: {len(all_records)}")


if __name__ == "__main__":
    main()
