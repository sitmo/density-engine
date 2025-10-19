#!/usr/bin/env python3
# scripts/train.py

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch
from typing import List
import matplotlib.pyplot as plt
import os
import random

# Columns
PARAM_COLS = ["alpha","gamma","beta","var0","eta","lam","ti","ti_indicator"]  # 'ti' will be log(ti), 'ti_indicator' is 1 if original ti was 1
ORIGINAL_COLS = ["alpha","gamma","beta","var0","eta","lam","ti","p","x"]  # Original dataset columns
ALL_COLS   = PARAM_COLS + ["x"]

# Import MDN model
from density_engine.models.mdn import MixtureDensityNetwork

# Import outlier tracking utilities
from density_engine.utils.quantiles import WindowQuantiles
from density_engine.utils.ringbuffer import SampleRingBuffer

# ---- top-level helpers (picklable) ----

class GarchDataset(Dataset):
    """PyTorch Dataset for GARCH data."""
    
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Apply log transforms
        ti_original = row["ti"]
        ti_transformed = np.log(row["ti"] - 0.8) - 1.0
        var0_transformed = np.log(row["var0"])
        eta_transformed = np.log(row["eta"]) - 1.0
        ti_indicator = 1.0 if ti_original == 1.0 else 0.0
        
        return {
            "alpha": row["alpha"],
            "gamma": row["gamma"],
            "beta": row["beta"],
            "var0": var0_transformed,
            "eta": eta_transformed,
            "lam": row["lam"],
            "ti": ti_transformed,
            "ti_indicator": ti_indicator,
            "x": row["x"]
        }

def transform_params_single(example):
    """Apply log transforms to a single example."""
    # Store original ti value before transformation
    ti_original = float(example["ti"])
    
    # Apply log transforms (ensure ti - 0.8 > 0)
    ti_transformed = np.log(ti_original - 0.8) - 1.0
    var0_transformed = np.log(float(example["var0"]))
    eta_transformed = np.log(float(example["eta"])) - 1.0
    
    # Add ti indicator: 1 if original ti was 1, else 0
    ti_indicator = 1.0 if ti_original == 1.0 else 0.0
    
    # Return transformed example
    return {
        "alpha": float(example["alpha"]),
        "gamma": float(example["gamma"]),
        "beta": float(example["beta"]),
        "var0": var0_transformed,
        "eta": eta_transformed,
        "lam": float(example["lam"]),
        "ti": ti_transformed,
        "ti_indicator": ti_indicator,
        "x": example["x"]  # Keep as list
    }

def collate_with_transforms(batch):
    """Collate function that applies transforms during batching."""
    # Extract raw data
    alpha = np.array([ex["alpha"] for ex in batch])
    gamma = np.array([ex["gamma"] for ex in batch])
    beta = np.array([ex["beta"] for ex in batch])
    var0 = np.array([ex["var0"] for ex in batch])
    eta = np.array([ex["eta"] for ex in batch])
    lam = np.array([ex["lam"] for ex in batch])
    ti = np.array([ex["ti"] for ex in batch])
    x = np.array([ex["x"] for ex in batch])
    
    # Apply log transforms
    ti_original = ti.copy()
    ti_transformed = np.log(ti - 0.8) - 1.0
    var0_transformed = np.log(var0)
    eta_transformed = np.log(eta) - 1.0
    ti_indicator = (ti_original == 1.0).astype(np.float32)
    
    # Stack parameters
    params = torch.stack([
        torch.from_numpy(alpha),
        torch.from_numpy(gamma),
        torch.from_numpy(beta),
        torch.from_numpy(var0_transformed),
        torch.from_numpy(eta_transformed),
        torch.from_numpy(lam),
        torch.from_numpy(ti_transformed),
        torch.from_numpy(ti_indicator)
    ], dim=1).float()
    
    # Stack targets
    targets = torch.from_numpy(x).float()
    
    return params, targets

def build_params_on_device(batch, device):
    """Build parameters on GPU from raw batch data with transforms applied on device."""
    # Move all tensors to device first (non-blocking for efficiency)
    for k in batch:
        batch[k] = batch[k].to(device, non_blocking=True)
    
    # Apply transforms on GPU
    ti_original = batch["ti"]
    
    params = torch.stack([
        batch["alpha"],
        batch["gamma"],
        batch["beta"],
        torch.log(batch["var0"]),                    # log transform on GPU
        torch.log(batch["eta"]) - 1.0,               # log transform on GPU
        batch["lam"],
        torch.log(ti_original - 0.8) - 1.0,          # log transform on GPU
        (ti_original == 1.0).float(),                # ti_indicator on GPU
    ], dim=1)
    
    targets = batch["x"]  # already [B, 512]
    return params, targets

def compute_cdf_metrics(model, params, targets, quantile_levels):
    """
    Compute CDF-based metrics efficiently (single CDF call).
    
    Args:
        model: MDN model
        params: Input parameters [B, 8]
        targets: Target quantiles [B, 512]
        quantile_levels: Reference quantile levels [512]
    
    Returns:
        row_losses: RMSE loss per row [B] for importance sampling
        mean_max_diff: Mean of max absolute differences (scalar)
    """
    # Compute CDF values once
    cdf_values = model.cdf(params, targets)  # [B, 512]
    
    # Reference quantile levels - expand to [B, 512]
    batch_size = params.size(0)
    ref_quantiles = quantile_levels.unsqueeze(0).expand(batch_size, -1)
    
    # Compute differences
    diff = cdf_values - ref_quantiles  # [B, 512]
    
    # RMSE per row (for loss and outlier detection)
    row_losses = torch.sqrt(torch.mean(diff ** 2, dim=1))  # [B]
    
    # Max absolute difference per row, then mean across batch (for monitoring)
    max_diff_per_row = torch.abs(diff).amax(dim=1)  # [B]
    mean_max_diff = max_diff_per_row.mean()  # scalar
    
    return row_losses, mean_max_diff

def add_noise_to_quantiles(q: torch.Tensor, p: torch.Tensor, N: float = 1e6) -> torch.Tensor:
    """
    q: [B, M] precise quantiles at fixed p = linspace(0.001, 0.999, M)
    returns q_noisy: [B, M] with delta-method Brownian-bridge CDF noise
    """
    B, M = q.shape
    device, dtype = q.device, q.dtype

    # Q'(p): centered differences (ends one-sided)
    Qp = torch.empty_like(q)
    denom_mid = (p[2:] - p[:-2])
    Qp[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / denom_mid
    Qp[:, 0]    = (q[:, 1] - q[:, 0]) / (p[1] - p[0])
    Qp[:, -1]   = (q[:, -1] - q[:, -2]) / (p[-1] - p[-2])

    # Brownian bridge Δp on the p-grid (simulate W at p plus W at 1.0)
    p_full = torch.cat([p, torch.tensor([1.0], device=device, dtype=dtype)])
    dp = torch.diff(torch.nn.functional.pad(p_full, (1, 0), value=0.0))  # [M+1]
    std = dp.sqrt().expand(B, -1)
    eps = torch.randn(B, M + 1, device=device, dtype=dtype) * std
    W = torch.cumsum(eps, dim=1)                   # Brownian motion
    W1 = W[:, -1: ]                                # value at t=1
    Bbridge = W[:, :-1] - p.unsqueeze(0) * W1      # bridge at p
    dP = Bbridge / N**0.5                          # Δp with 1/√N scale

    # Delta method: q_noisy = q + Q'(p) * Δp
    return q + Qp * dP

# ---- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default="sitmo/garch_densities")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--samples", type=int, default=10_000_000,
                   help="Total number of samples to process (default: 10,000,000)")
    ap.add_argument("--workers", type=int, default=8)
    
    # MDN architecture arguments
    ap.add_argument("--layers", type=str, default="64,64", 
                   help="Comma-separated list of hidden layer sizes")
    ap.add_argument("--components", type=int, default=128,
                   help="Number of mixture components")
    ap.add_argument("--activation", type=str, default="gelu",
                   help="Activation function (selu, relu, tanh, gelu, swish)")
    ap.add_argument("--center", action="store_true", default=True,
                   help="Center the mixture mean to zero (default: True)")
    ap.add_argument("--no-center", dest="center", action="store_false",
                   help="Disable centering of mixture mean")
    
    # Learning rate arguments
    ap.add_argument("--lrs", type=float, default=5e-4,
                   help="Learning rate start")
    ap.add_argument("--lre", type=float, default=1e-7,
                   help="Learning rate end")
    ap.add_argument("--lrd", type=float, default=0.99,
                   help="Learning rate decay factor (default: 0.99)")
    ap.add_argument("--lrds", type=int, default=10_000,
                   help="Learning rate decay samples - number of samples to process before LR decay step (default: 10_000)")
    ap.add_argument("--log-steps", type=int, default=16384,
                   help="Logging steps - number of samples to process before logging metrics (default: 16384)")
    
    # Outlier buffer arguments
    ap.add_argument("--outlier-buffer", type=int, default=0,
                   help="Size of outlier buffer (default: 0, set to e.g. 10_000)")
    ap.add_argument("--outlier-mix", type=float, default=0.5,
                   help="Fraction of batches that should be outlier batches (default: 0.5 for 50%)")
    
    # Training noise arguments
    ap.add_argument("--train-noise", type=int, default=None,
                   help="Add noise to training quantiles with N samples (default: None, no noise)")

    if torch.backends.mps.is_available():
        default_device = "mps"
    elif torch.cuda.is_available():
        default_device = "cuda"
    else:
        default_device = "cpu"
    ap.add_argument("--device", type=str, default=default_device)

    args = ap.parse_args()
    print(f"Arguments: {args}")

    device = torch.device(args.device)
    
    # Print device information
    print(f"\n=== Device Information ===")
    print(f"Selected device: {device}")
    print(f"Device type: {device.type}")
    
    if device.type == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        print(f"CUDA capability: {torch.cuda.get_device_capability(device)}")
    elif device.type == "mps":
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
    elif device.type == "cpu":
        print(f"CPU threads: {torch.get_num_threads()}")
        print(f"CPU interop threads: {torch.get_num_interop_threads()}")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"========================\n")
    
    # Parse layers argument
    hidden_layers = [int(x.strip()) for x in args.layers.split(',')]
    
    print(f"Loading dataset: {args.repo}")
    t0 = time.perf_counter()
    ds = load_dataset(args.repo, token=False)
    t1 = time.perf_counter()
    print(f"Loaded splits: {list(ds.keys())} in {(t1 - t0):.2f}s")

    print(f"Building loaders (batch_size={args.batch_size}, workers={args.workers})")
    
    # Use PyTorch format for efficient GPU pipeline - transforms will be done on GPU
    train_dataset = ds["train"].with_format("torch", columns=ORIGINAL_COLS)
    test_dataset = ds["test"].with_format("torch", columns=ORIGINAL_COLS)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.workers),
        prefetch_factor=4 if args.workers else None,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.workers),
        prefetch_factor=4 if args.workers else None,
        drop_last=False,
    )
    
    # Print dataset statistics
    train_samples = len(train_loader.dataset)
    test_samples = len(test_loader.dataset)
    train_batches = len(train_loader)
    test_batches = len(test_loader)
    
    print(f"Dataset statistics:")
    print(f"  Train: {train_samples:,} samples, {train_batches:,} batches")
    print(f"  Test:  {test_samples:,} samples, {test_batches:,} batches")
    print(f"  Total: {train_samples + test_samples:,} samples")
    
    # Set target number of samples
    max_samples = args.samples
    estimated_epochs = max_samples / train_samples
    print(f"Target samples: {max_samples:,}")
    print(f"Estimated epochs: {estimated_epochs:.2f}")

    # Create quantile levels for loss computation
    quantile_levels = torch.linspace(0.001, 0.999, 512, device=device)
    
    # Initialize MDN model
    print(f"Initializing MDN with layers={hidden_layers}, components={args.components}, activation={args.activation}, center={args.center}")
    model = MixtureDensityNetwork(
        input_dim=len(PARAM_COLS),  # 8 parameters (7 original + ti_indicator)
        hidden_layers=hidden_layers,
        num_components=args.components,
        activation=args.activation,
        center=args.center,
        device=device
    )
    
    # Calculate batches per learning rate decay step
    batches_per_lr_step = max(1, args.lrds // args.batch_size)
    print(f"Learning rate decay: every {batches_per_lr_step} batches (every {args.lrds} samples)")
    
    # Calculate batches per logging step
    batches_per_log_step = max(1, args.log_steps // args.batch_size)
    print(f"Logging: every {batches_per_log_step} batches (every {args.log_steps} samples)")
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lrs)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lrd)
    
    # Enable mixed precision training for CUDA
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("Mixed precision training (AMP) enabled")
    
    # Enable CUDA optimizations
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        print("CUDA optimizations enabled (TF32, matmul precision, cuDNN benchmark)")
    
    # Initialize outlier tracking infrastructure
    quantile_tracker = WindowQuantiles(eps=1e-2, window_step=5000)
    outlier_buffer = None
    
    if args.outlier_buffer > 0:
        print(f"Initializing outlier buffer with capacity {args.outlier_buffer}")
        outlier_buffer = SampleRingBuffer(
            capacity=args.outlier_buffer,
            x_shape=(len(PARAM_COLS),),  # 8 parameters
            y_shape=(512,),  # 512 quantiles
            device=device
        )
    else:
        print("Outlier buffer disabled (--outlier-buffer=0)")
    
    # Start MLflow run
    with mlflow.start_run():
        # Log all arguments except device
        mlflow.log_params({
            "repo": args.repo,
            "batch_size": args.batch_size,
            "samples": max_samples,
            "estimated_epochs": estimated_epochs,
            "workers": args.workers,
            "layers": args.layers,
            "components": args.components,
            "activation": args.activation,
            "center": args.center,
            "lrs": args.lrs,
            "lre": args.lre,
            "lrd": args.lrd,
            "lrds": args.lrds,
            "log_steps": args.log_steps,
            "outlier_buffer": args.outlier_buffer,
            "outlier_mix": args.outlier_mix,
            "train_noise": args.train_noise,
        })
        
        print(f"Starting training for {max_samples:,} samples...")
        
        # Initialize counters and tracking
        total_samples_processed = 0
        batch_idx = 0
        
        # Initialize tracking lists
        train_losses = []
        train_max_diffs = []
        batch_losses = []
        batch_max_diffs = []
        regular_batch_losses = []
        outlier_batch_losses = []
        
        # Create iterator for training data
        train_iter = iter(train_loader)
        
        while total_samples_processed < max_samples:
            try:
                # Probabilistically determine batch type
                use_outlier_batch = False
                if outlier_buffer is not None and outlier_buffer.size >= args.batch_size:
                    use_outlier_batch = random.random() < args.outlier_mix
                
                if use_outlier_batch:
                    # 100% from outlier buffer
                    params, targets, _ = outlier_buffer.sample(args.batch_size, out_device=device)
                    is_outlier_batch = True
                else:
                    # Regular batch from loader - build params on GPU with transforms
                    batch = next(train_iter)
                    params, targets = build_params_on_device(batch, device)
                    is_outlier_batch = False
                
                # Apply noise to training quantiles if --train-noise is specified
                if args.train_noise is not None and not is_outlier_batch:
                    targets = add_noise_to_quantiles(targets, quantile_levels, N=args.train_noise)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Compute loss with mixed precision
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    row_losses, max_diff = compute_cdf_metrics(model, params, targets, quantile_levels)
                    loss = row_losses.mean()
                
                if not is_outlier_batch:
                    # Regular batch: update quantiles and buffer
                    for loss_val in row_losses:
                        quantile_tracker.insert(loss_val.item())
                    
                    threshold = quantile_tracker.query(0.98)
                    outlier_mask = row_losses > threshold
                        
                    if outlier_mask.any() and outlier_buffer is not None:
                        outlier_buffer.append(params[outlier_mask], targets[outlier_mask])
                    
                    train_losses.append(loss.item())
                    train_max_diffs.append(max_diff.item())
                    batch_losses.append(loss.item())
                    batch_max_diffs.append(max_diff.item())
                    regular_batch_losses.append(loss.item())
                else:
                    # Outlier batch: just track loss
                    outlier_batch_losses.append(loss.item())
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                batch_idx += 1
                # Count actual samples in this batch (last batch might be smaller)
                actual_batch_size = params.size(0)
                total_samples_processed += actual_batch_size
                
                # Check if we've reached the target number of samples
                if total_samples_processed >= max_samples:
                    print(f"Reached target of {max_samples:,} samples. Stopping training.")
                    break
                
                # Update learning rate every batches_per_lr_step batches
                if batch_idx % batches_per_lr_step == 0:
                    # Update learning rate
                    if optimizer.param_groups[0]['lr'] > args.lre:
                        scheduler.step()
                        # if learning rate is less than lre, set it to lre and reset scheduler
                        if optimizer.param_groups[0]['lr'] < args.lre:
                            optimizer.param_groups[0]['lr'] = args.lre
                
                # Log metrics every batches_per_log_step batches
                if batch_idx % batches_per_log_step == 0:
                    # Calculate rolling averages for the logging period
                    log_period = min(batches_per_log_step, len(batch_losses))
                    avg_loss = np.mean(batch_losses[-log_period:]) if len(batch_losses) >= log_period else np.mean(batch_losses)
                    avg_max_diff = np.mean(batch_max_diffs[-log_period:]) if len(batch_max_diffs) >= log_period else np.mean(batch_max_diffs)
                    
                    # Compute test set metrics on 10 batches
                    model.eval()
                    test_losses = []
                    test_max_diffs = []
                    
                    with torch.inference_mode():
                        for test_batch_idx, test_batch in enumerate(test_loader):
                            if test_batch_idx >= 10:  # Limit to 10 batches
                                break
                            test_params, test_targets = build_params_on_device(test_batch, device)
                            
                            # Compute test loss and max_diff (no AMP in eval)
                            test_row_losses, test_max_diff = compute_cdf_metrics(model, test_params, test_targets, quantile_levels)
                            test_loss = test_row_losses.mean()
                            
                            test_losses.append(test_loss.item())
                            test_max_diffs.append(test_max_diff.item())
                    
                    # Average test metrics
                    avg_test_loss = np.mean(test_losses)
                    avg_test_max_diff = np.mean(test_max_diffs)
                    
                    # Set model back to training mode
                    model.train()
                    
                    # Compute outlier metrics
                    avg_outlier_loss = np.mean(outlier_batch_losses[-log_period:]) if len(outlier_batch_losses) >= log_period else (np.mean(outlier_batch_losses) if outlier_batch_losses else 0.0)
                    outlier_buffer_size = outlier_buffer.size if outlier_buffer is not None else 0
                    loss_quantile_98 = quantile_tracker.query(0.98)
                    
                    # Calculate current epoch based on total batches processed
                    current_epoch = total_samples_processed // train_samples + 1
                    
                    # Console output
                    print(f"Epoch: {current_epoch}/{int(estimated_epochs)+1}: "
                        f"Batch: {batch_idx:05d}, "
                        f"Samples: {total_samples_processed:08d}/{max_samples:08d} | "
                        f"Loss: {avg_loss:.6f}, "
                        f"Max Diff: {avg_max_diff:.6f}, "
                        f"Outlier Loss: {avg_outlier_loss:.6f}, "
                        f"Test Loss: {avg_test_loss:.6f}, "
                        f"Test Max Diff: {avg_test_max_diff:.6f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                    
                    # MLflow logging
                    metrics = {
                        "train_loss": avg_loss,
                        "train_max_diff": avg_max_diff,
                        "test_loss": avg_test_loss,
                        "test_max_diff": avg_test_max_diff,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "outlier_buffer_size": outlier_buffer_size,
                        "loss_quantile_98": loss_quantile_98,
                    }
                    
                    if outlier_batch_losses:
                        metrics["outlier_loss"] = avg_outlier_loss
                    
                    mlflow.log_metrics(metrics, step=total_samples_processed)
            
            except StopIteration:
                # End of dataset, recreate loader with fresh shuffling
                train_dataset = ds["train"].with_format("torch", columns=ORIGINAL_COLS)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=bool(args.workers),
                    prefetch_factor=4 if args.workers else None,
                    drop_last=False,
                )
                train_iter = iter(train_loader)
                print("Restarting dataset iterator with fresh shuffling...")
        
        # Final validation phase
        model.eval()
        test_losses = []
        test_max_diffs = []
        
        with torch.inference_mode():
            for test_batch in test_loader:
                test_params, test_targets = build_params_on_device(test_batch, device)
                
                row_losses, max_diff = compute_cdf_metrics(model, test_params, test_targets, quantile_levels)
                loss = row_losses.mean()
                
                test_losses.append(loss.item())
                test_max_diffs.append(max_diff.item())
        
        # Compute final averages
        avg_train_loss = np.mean(regular_batch_losses) if regular_batch_losses else 0.0
        avg_train_max_diff = np.mean(train_max_diffs) if train_max_diffs else 0.0
        avg_test_loss = np.mean(test_losses)
        avg_test_max_diff = np.mean(test_max_diffs)
        avg_outlier_loss_final = np.mean(outlier_batch_losses) if outlier_batch_losses else 0.0
        
        # Log final metrics
        final_metrics = {
            "final_train_loss": avg_train_loss,
            "final_train_max_diff": avg_train_max_diff,
            "final_test_loss": avg_test_loss,
            "final_test_max_diff": avg_test_max_diff,
            "final_learning_rate": optimizer.param_groups[0]['lr'],
            "total_samples_processed": total_samples_processed,
        }
        
        if outlier_batch_losses:
            final_metrics["final_outlier_loss"] = avg_outlier_loss_final
        
        mlflow.log_metrics(final_metrics, step=total_samples_processed)
        
        # Save the fully trained model to MLflow
        print("Saving fully trained model to MLflow...")
        mlflow.pytorch.log_model(
            model, 
            "model",
            extra_files={
                "model_config.txt": f"""Model Configuration:
- Architecture: Mixture Density Network (MDN)
- Input dimensions: {len(PARAM_COLS)} parameters
- Hidden layers: {hidden_layers}
- Number of components: {args.components}
- Activation function: {args.activation}
- Center mixture mean: {args.center}
- Device: {device}
- Total samples processed: {total_samples_processed:,}
- Final train loss: {avg_train_loss:.6f}
- Final test loss: {avg_test_loss:.6f}
- Final learning rate: {optimizer.param_groups[0]['lr']:.6f}
"""
            }
        )
        
        # Log model artifacts
        mlflow.log_artifact("model_config.txt")
        
        print("Training completed!")
        print(f"Total samples processed: {total_samples_processed:,}")
        print(f"Final train loss: {avg_train_loss:.6f}")
        print(f"Final test loss: {avg_test_loss:.6f}")
        print(f"Model saved to MLflow with artifact name: 'model'")

if __name__ == "__main__":
    main()
