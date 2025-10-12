#!/usr/bin/env python3
"""
Script to run jobs from a CSV file with configurable row ranges.

Usage:
    python run_jobs.py <jobfile> [--start START] [--end END] [--stride STRIDE] [--num_sim NUM_SIM] [--num_quantiles NUM_QUANTILES]

Arguments:
    jobfile: Path to the CSV job file
    --start: Starting row number (default: 0)
    --end: Ending row number (default: infinity, processes all remaining rows)
    --stride: Step size between rows to process (default: 1024)
    --num_sim: Number of simulations (default: 100000)
    --num_quantiles: Number of quantiles (default: 512)
"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from density_engine.garch import GJRGARCHReduced_torch
from density_engine.skew_student_t import HansenSkewedT_torch


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run jobs from a CSV file with configurable row ranges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'jobfile',
        type=str,
        help='Path to the CSV job file'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Starting row number (default: 0)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Ending row number (default: None for streaming through entire file)'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Number of rows to process at a time (default: 1)'
    )
    
    parser.add_argument(
        '--num_sim',
        type=int,
        default=1_000_000,
        help='Number of simulations (default: 1_000_000)'
    )
    
    parser.add_argument(
        '--num_quantiles',
        type=int,
        default=512,
        help='Number of quantiles (default: 512)'
    )
    
    return parser.parse_args()

def make_times():
    # random points in time
    times = set(np.concatenate([
        np.arange(20),
        np.random.randint(20, 100, size=20),
        np.random.randint(100, 1000, size=20)
    ])+1)
    
    times = sorted(times)
    times = [int(t) for t in times]
    times[-1] = 1000
    return times


def process_job_file(jobfile_path, start_row, end_row, stride, num_sim, num_quantiles):
    """Process the job file and print specified rows."""
    jobfile_path = Path(jobfile_path)
    
    if not jobfile_path.exists():
        print(f"Error: Job file '{jobfile_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing job file: {jobfile_path}")
    print(f"DEBUG: jobfile_path.parent = {jobfile_path.parent}")
    print(f"DEBUG: jobfile_path.parent.absolute() = {jobfile_path.parent.absolute()}")
    print(f"DEBUG: Current working directory = {Path.cwd()}")
    print(f"Start row: {start_row}")
    print(f"End row: {end_row if end_row is not None else 'infinity'}")
    print(f"Stride: {stride}")
    print(f"Number of simulations: {num_sim}")
    print(f"Number of quantiles: {num_quantiles}")
    print("-" * 50)
    

    try:
        with open(jobfile_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            
            # Read header
            header = next(reader)
            print(f"Header: {header}")
            print("-" * 50)
            
            # Calculate which rows to process based on stride
            rows_to_process = []
            current_row = start_row
            
            # If no end_row specified, process all rows (streaming)
            if end_row is None:
                print(f"No end row specified, processing all rows starting from {start_row} with stride {stride}")
                # We'll process rows as we encounter them in the CSV
                rows_to_process = None  # Signal to process all rows
            else:
                # Calculate specific rows to process
                while current_row <= end_row:
                    rows_to_process.append(current_row)
                    current_row += stride
            
            if rows_to_process is not None:
                print(f"Rows to process: {rows_to_process}")
            else:
                print("Streaming mode: processing all rows")
            print("-" * 50)
            
            # Process rows
            current_row = 0
            processed_count = 0

            data = []
            
            for row in reader:
                # Check if we should process this row
                should_process = False
                if rows_to_process is None:
                    # Streaming mode: process every stride-th row starting from start_row
                    if current_row >= start_row and (current_row - start_row) % stride == 0:
                        should_process = True
                else:
                    # Specific rows mode
                    if current_row in rows_to_process:
                        should_process = True
                
                if should_process:
                    print(f"Row {current_row}: {row}")

                    times = make_times()

                    # Parse CSV parameters: eta,lam,var0,alpha,gamma,beta,p
                    eta = float(row[0])
                    lam = float(row[1])
                    var0 = float(row[2])
                    alpha = float(row[3])
                    gamma = float(row[4])
                    beta = float(row[5])
                    p = float(row[6])
                    
                    # Create Hansen skewed t-distribution
                    dist = HansenSkewedT_torch(eta=eta, lam=lam)
                    
                    # do a reduced garch quantile simulation
                    reduced_garch = GJRGARCHReduced_torch(
                        alpha=alpha,
                        gamma=gamma,
                        beta=beta,
                        sigma0_sq=var0,
                        dist=dist
                    )
                    reduced_garch.reset(num_sim)
                    q_reduced = reduced_garch.path_quantiles(times, size=num_quantiles, normalize=True).cpu().numpy()

                    for j in range(len(times)):
                        record = {
                            'alpha': alpha,
                            'gamma': gamma,
                            'beta': beta,
                            'var0': var0,
                            'eta': eta,
                            'lam': lam,                       
                            'num_sim_paths': num_sim,
                            'job_row': current_row,
                            'ti': times[j],
                            'p': p,
                            'x': q_reduced[j, :].astype(np.float32)
                        }
                        data.append(record)


                    processed_count += 1
                
                current_row += 1
            
            print("-" * 50)
            print(f"Total rows processed: {processed_count}")
            if rows_to_process is not None:
                print(f"Expected rows: {len(rows_to_process)}")
            else:
                print(f"Streaming completed: processed {processed_count} rows")
            
            if end_row is None:
                filename = str(jobfile_path.parent.absolute() / (jobfile_path.stem +f'_{start_row}_streaming_{stride}_{num_sim}_{num_quantiles}' + '.parquet'))
            else:
                filename = str(jobfile_path.parent.absolute() / (jobfile_path.stem +f'_{start_row}_{end_row}_{stride}_{num_sim}_{num_quantiles}' + '.parquet'))
            df = pd.DataFrame(data)
            df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
            print('saving', filename)
            df.to_parquet(filename, index=False)   


            
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate arguments
    if args.start < 0:
        print("Error: Start row must be non-negative.", file=sys.stderr)
        sys.exit(1)
    
    if args.end is not None and args.end < args.start:
        print("Error: End row must be greater than or equal to start row.", file=sys.stderr)
        sys.exit(1)
    
    if args.stride <= 0:
        print("Error: Stride must be positive.", file=sys.stderr)
        sys.exit(1)
    
    process_job_file(args.jobfile, args.start, args.end, args.stride, args.num_sim, args.num_quantiles)


if __name__ == "__main__":
    main()
