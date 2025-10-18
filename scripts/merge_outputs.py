#!/usr/bin/env python3
"""
Script to create a professional HuggingFace dataset from parquet files in outputs directory.

Reads all garch_test_job_* and garch_training_job_* parquet files, sorts them by row range,
and creates a sharded HuggingFace dataset structure with 65k rows per shard.
"""

import os
import re
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Iterator
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def stream_parquet_files(file_list: List[Tuple[int, int, str, Path]]) -> Iterator[pd.DataFrame]:
    """
    Stream parquet files one by one to avoid loading everything into memory.
    
    Args:
        file_list: List of (start_row, end_row, file_type, file_path) tuples
    
    Yields:
        pd.DataFrame: One dataframe at a time
    """
    for start_row, end_row, file_type, file_path in file_list:
        try:
            logger.debug(f"Streaming {file_path.name} (rows {start_row}-{end_row})")
            df = pd.read_parquet(file_path)
            
            # Drop unnecessary columns and convert ti to float32
            df = df.drop(columns=['job_row', 'num_sim_paths'])
            df['ti'] = df['ti'].astype('float32')
            
            yield df
            
        except Exception as e:
            logger.error(f"Failed to read {file_path.name}: {e}")
            continue


def create_hf_dataset_structure(outputs_dir: Path) -> Path:
    """
    Create the HuggingFace dataset directory structure.
    
    Args:
        outputs_dir: Base outputs directory
    
    Returns:
        Path to the HF dataset directory
    """
    hf_repo_dir = outputs_dir / "hf-repo"
    hf_repo_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (hf_repo_dir / "data" / "train").mkdir(parents=True, exist_ok=True)
    (hf_repo_dir / "data" / "test").mkdir(parents=True, exist_ok=True)
    
    return hf_repo_dir


def get_parquet_files(outputs_dir: Path) -> Tuple[List[Tuple[int, int, str, Path]], List[Tuple[int, int, str, Path]]]:
    """
    Get all parquet files and separate them by type.
    
    Returns:
        (test_files, training_files) where each is a list of (start_row, end_row, file_type, file_path)
    """
    test_files = []
    training_files = []
    
    for file_path in outputs_dir.glob("*.parquet"):
        parsed = parse_filename(file_path.name)
        if parsed:
            start_row, end_row, file_type, filename = parsed
            if file_type == 'test':
                test_files.append((start_row, end_row, file_type, file_path))
            elif file_type == 'training':
                training_files.append((start_row, end_row, file_type, file_path))
        else:
            logger.warning(f"Could not parse filename: {file_path.name}")
    
    # Sort by start row
    test_files.sort(key=lambda x: x[0])
    training_files.sort(key=lambda x: x[0])
    
    return test_files, training_files


def create_sharded_dataset(file_list: List[Tuple[int, int, str, Path]], 
                          output_dir: Path, 
                          split_name: str, 
                          shard_size: int = 65000) -> Tuple[bool, int]:
    """
    Create sharded HuggingFace dataset from parquet files.
    
    Args:
        file_list: List of (start_row, end_row, file_type, file_path) tuples
        output_dir: Directory to write shards to
        split_name: Name of the split (train/test)
        shard_size: Number of rows per shard
    
    Returns:
        (success, total_shards_created)
    """
    if not file_list:
        logger.warning(f"No {split_name} files found to process")
        return False, 0
    
    logger.info(f"Creating sharded {split_name} dataset from {len(file_list)} files...")
    logger.info(f"Shard size: {shard_size:,} rows per shard")
    
    try:
        current_shard = 0
        current_rows = 0
        total_rows_processed = 0
        current_dataframes = []
        
        # Stream through files and create shards
        for df in stream_parquet_files(file_list):
            current_dataframes.append(df)
            current_rows += len(df)
            total_rows_processed += len(df)
            
            logger.debug(f"Accumulated {current_rows:,} rows in current shard")
            
            # When we reach shard size, write the shard
            if current_rows >= shard_size:
                shard_filename = f"{split_name}-{current_shard:05d}-of-XXXXX.parquet"
                shard_path = output_dir / shard_filename
                
                logger.info(f"Writing shard {current_shard} with {current_rows:,} rows to {shard_filename}")
                
                # Concatenate current dataframes
                shard_df = pd.concat(current_dataframes, ignore_index=True)
                
                # Sort by ti to ensure proper ordering
                shard_df = shard_df.sort_values('ti').reset_index(drop=True)
                
                # Write shard
                shard_df.to_parquet(shard_path, index=False)
                
                logger.info(f"  ✅ Shard {current_shard} written: {len(shard_df):,} rows, "
                          f"{shard_path.stat().st_size / (1024*1024):.1f} MB")
                
                # Reset for next shard
                current_shard += 1
                current_rows = 0
                current_dataframes = []
        
        # Write final shard if there are remaining rows
        if current_dataframes:
            shard_filename = f"{split_name}-{current_shard:05d}-of-XXXXX.parquet"
            shard_path = output_dir / shard_filename
            
            logger.info(f"Writing final shard {current_shard} with {current_rows:,} rows to {shard_filename}")
            
            # Concatenate remaining dataframes
            shard_df = pd.concat(current_dataframes, ignore_index=True)
            shard_df = shard_df.sort_values('ti').reset_index(drop=True)
            
            # Write final shard
            shard_df.to_parquet(shard_path, index=False)
            
            logger.info(f"  ✅ Final shard {current_shard} written: {len(shard_df):,} rows, "
                      f"{shard_path.stat().st_size / (1024*1024):.1f} MB")
            
            current_shard += 1
        
        total_shards = current_shard
        
        # Rename files with correct total count
        logger.info(f"Renaming shard files with correct total count ({total_shards})...")
        for i in range(total_shards):
            old_name = f"{split_name}-{i:05d}-of-XXXXX.parquet"
            new_name = f"{split_name}-{i:05d}-of-{total_shards:05d}.parquet"
            
            old_path = output_dir / old_name
            new_path = output_dir / new_name
            
            if old_path.exists():
                old_path.rename(new_path)
                logger.debug(f"Renamed {old_name} -> {new_name}")
        
        logger.info(f"✅ Successfully created {split_name} dataset:")
        logger.info(f"   - Total files processed: {len(file_list)}")
        logger.info(f"   - Total rows processed: {total_rows_processed:,}")
        logger.info(f"   - Total shards created: {total_shards}")
        logger.info(f"   - Average rows per shard: {total_rows_processed // total_shards:,}")
        
        return True, total_shards
        
    except Exception as e:
        logger.error(f"Failed to create sharded {split_name} dataset: {e}")
        return False, 0


def create_dataset_info(hf_repo_dir: Path, train_shards: int, test_shards: int) -> None:
    """
    Create dataset_info.json for HuggingFace dataset.
    
    Args:
        hf_repo_dir: Path to HF dataset directory
        train_shards: Number of training shards
        test_shards: Number of test shards
    """
    dataset_info = {
        "dataset_name": "garch-densities",
        "description": "GARCH model density estimation dataset with train/test splits",
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "features": {
            "alpha": {"dtype": "float64", "description": "GARCH alpha parameter"},
            "gamma": {"dtype": "float64", "description": "GARCH gamma parameter"},
            "beta": {"dtype": "float64", "description": "GARCH beta parameter"},
            "var0": {"dtype": "float64", "description": "Initial variance"},
            "eta": {"dtype": "float64", "description": "Eta parameter"},
            "lam": {"dtype": "float64", "description": "Lambda parameter"},
            "ti": {"dtype": "float32", "description": "Time index"},
            "p": {"dtype": "float64", "description": "Uniform [0,1] random number for splitting while preserving low-discrepancy"},
            "x": {"dtype": "list", "description": "Quantile / inverse CDF values array"}
        },
        "splits": {
            "train": {
                "num_examples": "unknown",  # Will be updated after processing
                "num_shards": train_shards
            },
            "test": {
                "num_examples": "unknown",  # Will be updated after processing
                "num_shards": test_shards
            }
        },
        "download_size": "unknown",
        "dataset_size": "unknown"
    }
    
    dataset_info_path = hf_repo_dir / "dataset_info.json"
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"Created dataset_info.json at {dataset_info_path}")


def create_readme(hf_repo_dir: Path) -> None:
    """
    Create README.md for HuggingFace dataset.
    
    Args:
        hf_repo_dir: Path to HF dataset directory
    """
    readme_content = """---
license: mit
task_categories:
- other
language:
- en
tags:
- garch
- finance
- density-estimation
- time-series
size_categories:
- 10M<n<100M
---

# GARCH Densities Dataset

This dataset contains GARCH model density estimation results with train/test splits.

## Dataset Description

The dataset includes:
- **alpha**: GARCH alpha parameter
- **gamma**: GARCH gamma parameter  
- **beta**: GARCH beta parameter
- **var0**: Initial variance
- **eta**: Eta parameter
- **lam**: Lambda parameter
- **num_sim_paths**: Number of simulation paths
- **job_row**: Job row identifier
- **ti**: Time index
- **p**: Uniform [0,1] random number for splitting while preserving low-discrepancy
- **x**: Quantile / inverse CDF values array

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/garch-densities")

# Access train and test splits
train_data = dataset["train"]
test_data = dataset["test"]

# Create data loader for training
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
```

## Dataset Statistics

- **Train split**: Multiple shards with ~65k rows each
- **Test split**: Multiple shards with ~65k rows each
- **Format**: Parquet files optimized for efficient loading

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{garch_densities_2024,
  title={GARCH Densities Dataset},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/datasets/your-username/garch-densities}
}
```
"""
    
    readme_path = hf_repo_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Created README.md at {readme_path}")


def create_gitattributes(hf_repo_dir: Path) -> None:
    """
    Create .gitattributes for Git LFS configuration.
    
    Args:
        hf_repo_dir: Path to HF dataset directory
    """
    gitattributes_content = """*.parquet filter=lfs diff=lfs merge=lfs -text
*.json filter=lfs diff=lfs merge=lfs -text
"""
    
    gitattributes_path = hf_repo_dir / ".gitattributes"
    with open(gitattributes_path, 'w') as f:
        f.write(gitattributes_content)
    
    logger.info(f"Created .gitattributes at {gitattributes_path}")


def main():
    """Main function to create HuggingFace dataset from parquet files."""
    outputs_dir = Path("/Users/thijs/Projects/density-engine/outputs")
    
    if not outputs_dir.exists():
        logger.error(f"Outputs directory not found: {outputs_dir}")
        sys.exit(1)
    
    logger.info("Starting HuggingFace dataset creation process...")
    logger.info("=" * 60)
    
    # Create HF dataset structure
    hf_repo_dir = create_hf_dataset_structure(outputs_dir)
    logger.info(f"Created HF dataset structure at: {hf_repo_dir}")
    
    # Get all parquet files (already separated by train/test)
    test_files, training_files = get_parquet_files(outputs_dir)
    
    logger.info(f"Found {len(test_files)} test files and {len(training_files)} training files")
    
    if not test_files and not training_files:
        logger.warning("No parquet files found to process")
        return
    
    success_count = 0
    train_shards = 0
    test_shards = 0
    
    # Process test files (maintain existing test split)
    if test_files:
        logger.info("\n" + "=" * 40)
        logger.info("PROCESSING TEST FILES")
        logger.info("=" * 40)
        
        test_output_dir = hf_repo_dir / "data" / "test"
        success, shards = create_sharded_dataset(test_files, test_output_dir, "test")
        if success:
            success_count += 1
            test_shards = shards
    else:
        logger.info("No test files found to process")
    
    # Process training files (maintain existing train split)
    if training_files:
        logger.info("\n" + "=" * 40)
        logger.info("PROCESSING TRAINING FILES")
        logger.info("=" * 40)
        
        train_output_dir = hf_repo_dir / "data" / "train"
        success, shards = create_sharded_dataset(training_files, train_output_dir, "train")
        if success:
            success_count += 1
            train_shards = shards
    else:
        logger.info("No training files found to process")
    
    # Create HF metadata files
    if success_count > 0:
        logger.info("\n" + "=" * 40)
        logger.info("CREATING HF METADATA")
        logger.info("=" * 40)
        
        create_dataset_info(hf_repo_dir, train_shards, test_shards)
        create_readme(hf_repo_dir)
        create_gitattributes(hf_repo_dir)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("HF DATASET CREATION SUMMARY")
    logger.info("=" * 60)
    
    if success_count == 2:
        logger.info("🎉 HuggingFace dataset created successfully!")
    elif success_count == 1:
        logger.info("⚠️  Partial success - some splits processed")
    else:
        logger.error("❌ No dataset splits were processed successfully")
        sys.exit(1)
    
    logger.info(f"Dataset location: {hf_repo_dir}")
    logger.info(f"Train shards: {train_shards}")
    logger.info(f"Test shards: {test_shards}")
    logger.info(f"Structure:")
    logger.info(f"  📁 {hf_repo_dir}/")
    logger.info(f"  ├── 📄 README.md")
    logger.info(f"  ├── 📄 dataset_info.json")
    logger.info(f"  ├── 📄 .gitattributes")
    logger.info(f"  └── 📁 data/")
    logger.info(f"      ├── 📁 train/ ({train_shards} shards)")
    logger.info(f"      └── 📁 test/ ({test_shards} shards)")


if __name__ == "__main__":
    main()
