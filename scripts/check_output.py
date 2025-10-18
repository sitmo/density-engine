#!/usr/bin/env python3
"""
Script to check parquet files in the outputs directory.

Checks:
1. Files are readable
2. Row count matches expected (100 rows for range YYYY-XXXX+1)
3. Row number ranges are contiguous
4. Reports missing files (gaps in ranges)
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


def parse_filename(filename: str) -> Optional[Tuple[int, int, str, str]]:
    """
    Parse filename to extract start, end row numbers, file type, and full path.
    
    Expected format: garch_(test|training)_job_XXXX_YYYY_0_streaming_1_ZZZZZZZ_512.parquet
    
    Returns:
        (start_row, end_row, file_type, full_path) or None if parsing fails
    """
    pattern = r'garch_(test|training)_job_(\d+)_(\d+)_0_streaming_1_\d+_512\.parquet'
    match = re.match(pattern, filename)
    
    if match:
        file_type = match.group(1)  # 'test' or 'training'
        start_row = int(match.group(2))
        end_row = int(match.group(3))
        return start_row, end_row, file_type, filename
    return None


def check_parquet_file(file_path: Path) -> Tuple[bool, int, str]:
    """
    Check if a parquet file is readable and return unique row count (ignoring ti and x columns).
    
    Returns:
        (is_readable, unique_row_count, error_message)
    """
    try:
        df = pd.read_parquet(file_path)
        
        # Get unique rows by dropping ti and x columns if they exist
        columns_to_drop = []
        if 'ti' in df.columns:
            columns_to_drop.append('ti')
        if 'x' in df.columns:
            columns_to_drop.append('x')
        
        if columns_to_drop:
            unique_df = df.drop(columns=columns_to_drop).drop_duplicates()
            unique_count = len(unique_df)
        else:
            # If ti and x columns don't exist, just count total rows
            unique_count = len(df)
            
        return True, unique_count, ""
    except Exception as e:
        return False, 0, str(e)


def find_missing_ranges(found_ranges: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Find missing ranges in the sequence, separated by file type.
    
    Args:
        found_ranges: List of (start, end, file_type) tuples sorted by start
        
    Returns:
        List of missing (start, end, file_type) ranges
    """
    if not found_ranges:
        return []
    
    # Group ranges by file type
    test_ranges = [(start, end) for start, end, file_type in found_ranges if file_type == 'test']
    training_ranges = [(start, end) for start, end, file_type in found_ranges if file_type == 'training']
    
    missing = []
    
    # Find missing ranges for test files
    if test_ranges:
        test_ranges.sort(key=lambda x: x[0])
        current_expected = test_ranges[0][0]
        for start, end in test_ranges:
            if start > current_expected:
                missing.append((current_expected, start - 1, 'test'))
            current_expected = end + 1
    
    # Find missing ranges for training files
    if training_ranges:
        training_ranges.sort(key=lambda x: x[0])
        current_expected = training_ranges[0][0]
        for start, end in training_ranges:
            if start > current_expected:
                missing.append((current_expected, start - 1, 'training'))
            current_expected = end + 1
    
    return missing


def main():
    """Main function to check all parquet files."""
    outputs_dir = Path("/Users/thijs/Projects/density-engine/outputs")
    
    if not outputs_dir.exists():
        print(f"ERROR: Outputs directory not found: {outputs_dir}")
        sys.exit(1)
    
    print("Checking parquet files in outputs directory...")
    print("=" * 60)
    
    # Find all parquet files and parse their names
    parquet_files = []
    for file_path in outputs_dir.glob("*.parquet"):
        parsed = parse_filename(file_path.name)
        if parsed:
            start_row, end_row, file_type, filename = parsed
            parquet_files.append((start_row, end_row, file_type, file_path))
        else:
            print(f"WARNING: Could not parse filename: {file_path.name}")
    
    if not parquet_files:
        print("No valid parquet files found.")
        return
    
    # Sort by start row
    parquet_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(parquet_files)} parquet files")
    print()
    
    # Check each file
    corrupt_files = []
    wrong_row_count = []
    found_ranges = []
    
    for start_row, end_row, file_type, file_path in parquet_files:
        expected_rows = end_row - start_row + 1
        
        # Check if file is readable
        is_readable, actual_rows, error_msg = check_parquet_file(file_path)
        
        if not is_readable:
            corrupt_files.append((file_path.name, error_msg))
            print(f"❌ CORRUPT: {file_path.name} - {error_msg}")
        else:
            found_ranges.append((start_row, end_row, file_type))
            
            if actual_rows != expected_rows:
                wrong_row_count.append((file_path.name, expected_rows, actual_rows))
                print(f"⚠️  WRONG UNIQUE ROW COUNT: {file_path.name} - Expected {expected_rows}, got {actual_rows}")
            else:
                print(f"✅ OK: {file_path.name} - {actual_rows} unique rows")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Report corrupt files
    if corrupt_files:
        print(f"\n❌ CORRUPT FILES ({len(corrupt_files)}):")
        for filename, error in corrupt_files:
            print(f"  - {filename}: {error}")
    else:
        print("\n✅ No corrupt files found")
    
    # Report wrong unique row count files
    if wrong_row_count:
        print(f"\n⚠️  WRONG UNIQUE ROW COUNT ({len(wrong_row_count)}):")
        for filename, expected, actual in wrong_row_count:
            print(f"  - {filename}: Expected {expected}, got {actual}")
    else:
        print("\n✅ All files have correct unique row counts")
    
    # Check for missing ranges
    missing_ranges = find_missing_ranges(found_ranges)
    if missing_ranges:
        # Separate missing ranges by file type
        test_missing = [(start, end) for start, end, file_type in missing_ranges if file_type == 'test']
        training_missing = [(start, end) for start, end, file_type in missing_ranges if file_type == 'training']
        
        print(f"\n🔍 MISSING RANGES ({len(missing_ranges)} total):")
        
        if test_missing:
            print(f"\n  📊 TEST FILES ({len(test_missing)} missing ranges):")
            for start, end in test_missing:
                print(f"    - Rows {start:05d} to {end:05d} (missing {end - start + 1} rows)")
        
        if training_missing:
            print(f"\n  🏋️  TRAINING FILES ({len(training_missing)} missing ranges):")
            for start, end in training_missing:
                print(f"    - Rows {start:05d} to {end:05d} (missing {end - start + 1} rows)")
    else:
        print("\n✅ No missing ranges found - all ranges are contiguous")
    
    # Overall summary
    total_issues = len(corrupt_files) + len(wrong_row_count) + len(missing_ranges)
    if total_issues == 0:
        print("\n🎉 All checks passed! No issues found.")
    else:
        print(f"\n📊 Total issues found: {total_issues}")
        print(f"   - Corrupt files: {len(corrupt_files)}")
        print(f"   - Wrong unique row counts: {len(wrong_row_count)}")
        print(f"   - Missing ranges: {len(missing_ranges)}")


if __name__ == "__main__":
    main()
