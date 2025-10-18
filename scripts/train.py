#!/usr/bin/env python3
# scripts/train.py

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
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

def transform_params(example):
    """Replace ti, var0, and eta with their log transforms and add ti indicator lazily (applied on access)."""
    # Store original ti value before transformation
    ti_original = example["ti"]
    

    # Apply log transforms (ensure ti - 0.8 > 0)
    example["ti"] = np.log(example["ti"] - 0.8) - 1.0
    example["var0"] = np.log(example["var0"])
    example["eta"] = np.log(example["eta"]) - 1.0
    
    # Add ti indicator: 1 if original ti was 1, else 0
    example["ti_indicator"] = 1.0 if ti_original == 1.0 else 0.0
    return example

def collate(batch):
    """Stack a list of dicts (NumPy) into batched torch tensors."""
    b = {k: np.stack([ex[k] for ex in batch]) for k in ALL_COLS}
    params  = torch.from_numpy(np.stack([b[c] for c in PARAM_COLS], axis=1))  # [B,8]
    targets = torch.from_numpy(b["x"])                                        # [B,512]
    return params.float(), targets.float()

class TransformedDataset:
    """Wrapper that applies transform and provides all columns including ti_indicator."""
    
    def __init__(self, dataset, transform_func):
        self.dataset = dataset
        self.transform_func = transform_func
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform_func(item)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

def make_loader(ds, split, batch_size=1024, workers=4, shuffle=False):
    # Create transformed dataset that includes ti_indicator
    base_dataset = ds[split].with_format("numpy", columns=ORIGINAL_COLS)
    transformed_dataset = TransformedDataset(base_dataset, transform_params)
    
    pin = torch.cuda.is_available()
    return DataLoader(
        transformed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=bool(workers),
        prefetch_factor=(2 if workers else None),
        collate_fn=collate,
        drop_last=False,
    )

def to_device(t1, t2, device):
    non_block = device.type == "cuda"
    return t1.to(device, non_blocking=non_block), t2.to(device, non_blocking=non_block)

def compute_cdf_loss(model, params, targets, quantile_levels):
    """
    Compute RMSE loss per row between CDF(targets) and quantile_levels.
    
    Args:
        model: MDN model
        params: Input parameters [B, 8]
        targets: Target quantiles [B, 512]
        quantile_levels: Reference quantile levels [512]
    
    Returns:
        RMSE loss per row [B] - for importance sampling preparation
    """
    # Compute CDF values for targets
    cdf_values = model.cdf(params, targets)  # [B, 512]
    
    # Reference quantile levels should be [512] - expand to [B, 512]
    batch_size = params.size(0)
    ref_quantiles = quantile_levels.unsqueeze(0).expand(batch_size, -1)
    
    # Compute RMSE per row (across the 512 quantile differences)
    row_rmse = torch.sqrt(torch.mean((cdf_values - ref_quantiles) ** 2, dim=1))
    
    return row_rmse

def compute_max_abs_diff(model, params, targets, quantile_levels):
    """
    Compute mean of max absolute differences per row (batch-size independent).
    
    Args:
        model: MDN model
        params: Input parameters [B, 8]
        targets: Target quantiles [B, 512]
        quantile_levels: Reference quantile levels [512]
    
    Returns:
        Mean of max absolute differences per row (scalar)
    """
    cdf_values = model.cdf(params, targets)  # [B, 512]
    batch_size = params.size(0)
    ref_quantiles = quantile_levels.unsqueeze(0).expand(batch_size, -1)
    
    # Compute absolute differences [B, 512]
    abs_diff = torch.abs(cdf_values - ref_quantiles)
    
    # Compute max per row (across 512 quantiles), then mean across batch
    max_diff_per_row = torch.max(abs_diff, dim=1).values.detach()  # [B] - max for each row
    mean_max_diff = torch.mean(max_diff_per_row)   # scalar - mean across batch
    return mean_max_diff

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
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--samples", type=int, default=10_000_000,
                   help="Total number of samples to process (default: 10,000,000)")
    ap.add_argument("--workers", type=int, default=4)
    
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
    ap.add_argument("--log-steps", type=int, default=20_000,
                   help="Logging steps - number of samples to process before logging metrics (default: 20_000)")
    
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
    
    # Parse layers argument
    hidden_layers = [int(x.strip()) for x in args.layers.split(',')]
    
    print(f"Loading dataset: {args.repo}")
    t0 = time.perf_counter()
    ds = load_dataset(args.repo, token=False)
    t1 = time.perf_counter()
    print(f"Loaded splits: {list(ds.keys())} in {(t1 - t0):.2f}s")

    print(f"Building loaders (batch_size={args.batch_size}, workers={args.workers})")
    train_loader = make_loader(ds, "train", batch_size=args.batch_size, workers=args.workers, shuffle=True)
    test_loader  = make_loader(ds, "test",  batch_size=args.batch_size, workers=args.workers, shuffle=False)
    
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
                    # Regular batch from loader
                    params, targets = next(train_iter)
                    params, targets = to_device(params, targets, device)
                    is_outlier_batch = False
                
                # Apply noise to training quantiles if --train-noise is specified
                if args.train_noise is not None and not is_outlier_batch:
                    targets = add_noise_to_quantiles(targets, quantile_levels, N=args.train_noise)
                
                optimizer.zero_grad()
                
                # Compute loss per row, then average across batch
                row_losses = compute_cdf_loss(model, params, targets, quantile_levels)
                loss = torch.mean(row_losses)  # Average across batch rows
                
                if not is_outlier_batch:
                    # Regular batch: update quantiles and buffer
                    for loss_val in row_losses:
                        quantile_tracker.insert(loss_val.item())
                    
                    threshold = quantile_tracker.query(0.98)
                    outlier_mask = row_losses > threshold
                        
                    if outlier_mask.any() and outlier_buffer is not None:
                        outlier_buffer.append(params[outlier_mask], targets[outlier_mask])
                    
                    max_diff = compute_max_abs_diff(model, params, targets, quantile_levels)
                    train_losses.append(loss.item())
                    train_max_diffs.append(max_diff.item())
                    batch_losses.append(loss.item())
                    batch_max_diffs.append(max_diff.item())
                    regular_batch_losses.append(loss.item())
                else:
                    # Outlier batch: just track loss
                    outlier_batch_losses.append(loss.item())
                
                loss.backward()
                optimizer.step()
                
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
                    
                    with torch.no_grad():
                        for test_batch_idx, (test_params, test_targets) in enumerate(test_loader):
                            if test_batch_idx >= 10:  # Limit to 10 batches
                                break
                            test_params, test_targets = to_device(test_params, test_targets, device)
                            
                            # Compute test loss and max_diff
                            test_row_losses = compute_cdf_loss(model, test_params, test_targets, quantile_levels)
                            test_loss = torch.mean(test_row_losses)
                            test_max_diff = compute_max_abs_diff(model, test_params, test_targets, quantile_levels)
                            
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
                train_loader = make_loader(ds, "train", batch_size=args.batch_size, workers=args.workers, shuffle=True)
                train_iter = iter(train_loader)
                print("Restarting dataset iterator with fresh shuffling...")
        
        # Final validation phase
        model.eval()
        test_losses = []
        test_max_diffs = []
        
        with torch.no_grad():
            for params, targets in test_loader:
                params, targets = to_device(params, targets, device)
                
                row_losses = compute_cdf_loss(model, params, targets, quantile_levels)
                loss = torch.mean(row_losses)  # Average across batch rows
                max_diff = compute_max_abs_diff(model, params, targets, quantile_levels)
                
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
        
        # Log the final model
        mlflow.pytorch.log_model(model, "model")
        
        print("Training completed!")
        print(f"Total samples processed: {total_samples_processed:,}")
        print(f"Final train loss: {avg_train_loss:.6f}")
        print(f"Final test loss: {avg_test_loss:.6f}")

if __name__ == "__main__":
    main()
