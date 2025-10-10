import os
import sys
import numpy as np
from scipy.stats import qmc
import pandas as pd
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from density_engine.skew_student_t import HansenSkewedT_torch

def sample_alpha_effgamma_beta(points):
    """
    Sample uniformly inside the simplex volume {0 <= x,y,z <=1, 0 <= x+y+z <=1}
    using the sorting trick.
    """

    n_samples = points.shape[0]
    # Augment with a fake fourth coordinate = 1
    ones = np.ones((n_samples, 1))
    points_aug = np.hstack([points, ones])

    # Sort along each row
    sorted_points = np.sort(points_aug, axis=1)

    # Differences give barycentric coordinates
    x = sorted_points[:, 0]
    y = sorted_points[:, 1] - sorted_points[:, 0]
    z = sorted_points[:, 2] - sorted_points[:, 1]

    return x, y, z

def uniform_to_theta(u):
    # GJR-GARCH model parameters Theta
    u0 = u[:, 0]
    eta = 4 + np.exp(np.log(96)*u0) # 4 ... 100 logarithmic interpolated

    u1 = u[:, 1]
    lam = 0.95*(2 * u1 - 1) # -0.95 ... +0.95 uniform, but to close too -1 and +1 makes the dist break down

    u2 = u[:, 2]
    var0 = 10**(2 * u2 - 1) # approx 1/10 ... 10, logarithmic interpolated

    # 0 < alpha + eff_gamma + beta < 0.99
    alpha, eff_gamma, beta = sample_alpha_effgamma_beta(u[:, 3:6])
    alpha *= 0.99
    eff_gamma *= 0.99
    beta *= 0.99

    # compute gamma from eff_gamma, eff_gamma = gamma * F0
    P0 = []
    for eta_i, lam_i in zip(eta, lam):
        dist = HansenSkewedT_torch(eta=eta_i, lam=lam_i, device='cpu')
        P0.append(dist.second_moment_left())
    P0 = np.array(P0)

    gamma = eff_gamma / P0
    return eta, lam, var0, alpha, gamma, beta


def generate_jobs_csv(seed, num_jobs_pow, output_filename):
    """
    Generate a single CSV file with all job parameters.
    
    Args:
        seed: Random seed for reproducibility
        num_jobs_pow: Number of jobs as 2^num_jobs_pow
        output_filename: Name of the output CSV file
    """
    num_jobs = 2**num_jobs_pow

    bits = int(0.5 + np.log2(num_jobs))

    sobol_engine = qmc.Sobol(d=7, scramble=True, optimization='random-cd', bits=bits, seed=seed)
    sobol_points = sobol_engine.random(n=num_jobs).astype(np.float32)

    # enhance surface/boundary coverage to reduce extrapolation
    sobol_points[:,:6] = np.clip(sobol_points[:,:6] * 1.1 - 0.05, 0, 1)

    # Train-test split dimension
    p = sobol_points[:, 6]

    # Convert Sobol points to GJR-GARCH Parameters (Theta)
    eta, lam, var0, alpha, gamma, beta = uniform_to_theta(sobol_points)

    df = pd.DataFrame({
        'eta': eta,
        'lam': lam,
        'var0': var0,
        'alpha': alpha,
        'gamma': gamma,
        'beta': beta,
        'p': p
    })

    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"✅ Saved {num_jobs:,} jobs to {output_filename}")
    
    return df


if __name__ == "__main__":
    print("🚀 Generating GARCH job parameter files")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("jobs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate training set jobs (130k parameter cases)
    print("📊 Generating training set jobs (130k parameter cases)...")
    train_df = generate_jobs_csv(
        seed=0, 
        num_jobs_pow=17, 
        output_filename=output_dir / "garch_training_jobs.csv"
    )
    
    # Generate test set jobs (32k parameter cases)
    print("📊 Generating test set jobs (32k parameter cases)...")
    test_df = generate_jobs_csv(
        seed=1, 
        num_jobs_pow=15, 
        output_filename=output_dir / "garch_test_jobs.csv"
    )
    
    print()
    print("🎉 Job generation complete!")
    print(f"📁 Training jobs: {len(train_df):,} rows saved to jobs/garch_training_jobs.csv")
    print(f"📁 Test jobs: {len(test_df):,} rows saved to jobs/garch_test_jobs.csv")
    print()
    print("💡 You can now orchestrate job processing and result collection separately.")
