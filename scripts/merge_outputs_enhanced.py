#!/usr/bin/env python3
"""
Enhanced script to create professional HuggingFace datasets from parquet files for multiple models.

This script can process parquet files for any model (GARCH, Rough Heston, Rough Bergomi)
and create model-specific HuggingFace datasets with proper schemas and metadata.

Usage:
    python merge_outputs.py <model_name> [options]

Arguments:
    model_name: Name of the model (gjrgarch_normalized, gjrgarch, rough_heston, rough_bergomi)
    --input_dir: Input directory containing parquet files (default: datasets/{model}/raw_outputs/)
    --output_dir: Output directory for HF dataset (default: datasets/{model}/hf/)
    --shard_size: Number of rows per shard (default: 65000)
    --split_ratio: Train/test split ratio (default: 0.8)
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
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from density_engine.models import get_model_names

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_config(model_name: str) -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    config_path = project_root / "datasets" / model_name / "config" / "model_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def parse_filename(filename: str, model_name: str) -> Optional[Tuple[int, int, str, str]]:
    """
    Parse filename to extract start, end row numbers, file type, and full path.
    
    Expected format: {model_name}_job_XXXX_YYYY_timestamp_count_512.parquet
    
    Returns:
        (start_row, end_row, file_type, full_path) or None if parsing fails
    """
    # Try different patterns for different filename formats
    patterns = [
        rf'{model_name}_job_(\d+)_(\d+)_\d+_\d+_512\.parquet',
        rf'{model_name}_job_(\d+)_(\d+)_\d+_streaming_\d+_\d+_512\.parquet',
        rf'{model_name}_job_(\d+)_(\d+)_\d+\.parquet',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            start_row = int(match.group(1))
            end_row = int(match.group(2))
            # Determine file type based on row range (simple heuristic)
            file_type = "test" if start_row % 10 == 0 else "training"
            return start_row, end_row, file_type, filename
    
    return None


def stream_parquet_files(file_list: List[Tuple[int, int, str, Path]], model_config: Dict[str, Any]) -> Iterator[pd.DataFrame]:
    """
    Stream parquet files one by one to avoid loading everything into memory.
    
    Args:
        file_list: List of (start_row, end_row, file_type, file_path) tuples
        model_config: Model configuration dictionary
    
    Yields:
        pd.DataFrame: One dataframe at a time
    """
    for start_row, end_row, file_type, file_path in file_list:
        try:
            logger.debug(f"Streaming {file_path.name} (rows {start_row}-{end_row})")
            df = pd.read_parquet(file_path)
            
            # Drop unnecessary columns if they exist
            columns_to_drop = ['job_row', 'num_sim_paths']
            for col in columns_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            # Convert ti to float32
            if 'ti' in df.columns:
                df['ti'] = df['ti'].astype('float32')
            
            yield df
            
        except Exception as e:
            logger.error(f"Failed to read {file_path.name}: {e}")
            continue


def create_hf_dataset_structure(output_dir: Path) -> Path:
    """
    Create the HuggingFace dataset directory structure.
    
    Args:
        output_dir: Base output directory
    
    Returns:
        Path to the HF dataset directory
    """
    hf_repo_dir = output_dir
    hf_repo_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (hf_repo_dir / "data" / "train").mkdir(parents=True, exist_ok=True)
    (hf_repo_dir / "data" / "test").mkdir(parents=True, exist_ok=True)
    
    return hf_repo_dir


def get_parquet_files(input_dir: Path, model_name: str) -> Tuple[List[Tuple[int, int, str, Path]], List[Tuple[int, int, str, Path]]]:
    """
    Get all parquet files and separate them by type.
    
    Returns:
        (test_files, training_files) where each is a list of (start_row, end_row, file_type, file_path)
    """
    test_files = []
    training_files = []
    
    for file_path in input_dir.glob("*.parquet"):
        parsed = parse_filename(file_path.name, model_name)
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


def create_sharded_dataset(
    file_list: List[Tuple[int, int, str, Path]], 
    output_dir: Path, 
    split_name: str, 
    model_config: Dict[str, Any],
    shard_size: int = 65000
) -> Tuple[bool, int]:
    """
    Create sharded HuggingFace dataset from parquet files.
    
    Args:
        file_list: List of (start_row, end_row, file_type, file_path) tuples
        output_dir: Directory to write shards to
        split_name: Name of the split (train/test)
        model_config: Model configuration dictionary
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
        for df in stream_parquet_files(file_list, model_config):
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
                if 'ti' in shard_df.columns:
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
            if 'ti' in shard_df.columns:
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
        if total_shards > 0:
            logger.info(f"   - Average rows per shard: {total_rows_processed // total_shards:,}")
        
        return True, total_shards
        
    except Exception as e:
        logger.error(f"Failed to create sharded {split_name} dataset: {e}")
        return False, 0


def create_dataset_info(hf_repo_dir: Path, model_config: Dict[str, Any], train_shards: int, test_shards: int) -> None:
    """
    Create dataset_info.json for HuggingFace dataset.
    
    Args:
        hf_repo_dir: Path to HF dataset directory
        model_config: Model configuration dictionary
        train_shards: Number of training shards
        test_shards: Number of test shards
    """
    hf_config = model_config["huggingface"]
    
    dataset_info = {
        "dataset_name": hf_config["dataset_name"],
        "description": hf_config["description"],
        "version": model_config["version"],
        "created": datetime.now().isoformat(),
        "features": model_config["output_schema"],
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


def create_readme(hf_repo_dir: Path, model_config: Dict[str, Any]) -> None:
    """
    Create README.md for HuggingFace dataset.
    
    Args:
        hf_repo_dir: Path to HF dataset directory
        model_config: Model configuration dictionary
    """
    hf_config = model_config["huggingface"]
    model_name = model_config["model_name"]
    
    # Create feature descriptions
    feature_descriptions = []
    for param_name, param_info in model_config["model_parameters"].items():
        feature_descriptions.append(f"- **{param_name}**: {param_info['description']}")
    
    readme_content = f"""---
license: {hf_config["license"]}
task_categories:
- other
language:
- en
tags:
{chr(10).join([f"- {tag}" for tag in hf_config["tags"]])}
size_categories:
- 10M<n<100M
---

# {hf_config["dataset_name"].replace("-", " ").title()}

{hf_config["description"]}

## Dataset Description

The dataset includes:
{chr(10).join(feature_descriptions)}
- **ti**: Time index
- **p**: Uniform [0,1] random number for splitting while preserving low-discrepancy
- **x**: Quantile / inverse CDF values array

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/{hf_config['dataset_name']}")

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
@dataset{{{model_name}_densities_2024,
  title={{{hf_config['dataset_name'].replace("-", " ").title()} Dataset}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/datasets/your-username/{hf_config['dataset_name']}}}
}}
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
    parser = argparse.ArgumentParser(
        description="Create HuggingFace dataset from parquet files for any model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'model_name',
        type=str,
        help='Name of the model (gjrgarch_normalized, gjrgarch, rough_heston, rough_bergomi)'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Input directory containing parquet files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for HF dataset'
    )
    
    parser.add_argument(
        '--shard_size',
        type=int,
        default=65000,
        help='Number of rows per shard (default: 65000)'
    )
    
    parser.add_argument(
        '--split_ratio',
        type=float,
        default=0.8,
        help='Train/test split ratio (default: 0.8)'
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
        model_config = load_model_config(args.model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Set default directories
    if args.input_dir is None:
        args.input_dir = project_root / "datasets" / args.model_name / "raw_outputs"
    else:
        args.input_dir = Path(args.input_dir)
    
    if args.output_dir is None:
        args.output_dir = project_root / "datasets" / args.model_name / "hf"
    else:
        args.output_dir = Path(args.output_dir)
    
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Shard size: {args.shard_size:,}")
    logger.info(f"Split ratio: {args.split_ratio}")
    logger.info("=" * 60)
    
    # Create HF dataset structure
    hf_repo_dir = create_hf_dataset_structure(args.output_dir)
    logger.info(f"Created HF dataset structure at: {hf_repo_dir}")
    
    # Get all parquet files
    test_files, training_files = get_parquet_files(args.input_dir, args.model_name)
    
    logger.info(f"Found {len(test_files)} test files and {len(training_files)} training files")
    
    if not test_files and not training_files:
        logger.warning("No parquet files found to process")
        return
    
    success_count = 0
    train_shards = 0
    test_shards = 0
    
    # Process test files
    if test_files:
        logger.info("\n" + "=" * 40)
        logger.info("PROCESSING TEST FILES")
        logger.info("=" * 40)
        
        test_output_dir = hf_repo_dir / "data" / "test"
        success, shards = create_sharded_dataset(test_files, test_output_dir, "test", model_config, args.shard_size)
        if success:
            success_count += 1
            test_shards = shards
    else:
        logger.info("No test files found to process")
    
    # Process training files
    if training_files:
        logger.info("\n" + "=" * 40)
        logger.info("PROCESSING TRAINING FILES")
        logger.info("=" * 40)
        
        train_output_dir = hf_repo_dir / "data" / "train"
        success, shards = create_sharded_dataset(training_files, train_output_dir, "train", model_config, args.shard_size)
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
        
        create_dataset_info(hf_repo_dir, model_config, train_shards, test_shards)
        create_readme(hf_repo_dir, model_config)
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
