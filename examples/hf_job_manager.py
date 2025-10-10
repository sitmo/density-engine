"""
Example of using Hugging Face API with dotenv for credential management.

This module demonstrates how to:
1. Load HF credentials from .env file
2. Use HF API for dataset operations
3. Manage job definitions and results
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, Repository
from huggingface_hub.utils import RepositoryNotFoundError

# Load environment variables from .env file
load_dotenv()


class HFJobManager:
    """Manages job definitions and results on Hugging Face datasets using dotenv credentials."""
    
    def __init__(self, dataset_name: str = "density-engine-jobs", organization: Optional[str] = None):
        """Initialize HF job manager.
        
        Args:
            dataset_name: Name of the HF dataset
            organization: HF organization name (if None, uses HF_USERNAME from .env)
        """
        self.dataset_name = dataset_name
        self.organization = organization or os.getenv("HF_USERNAME")
        
        if not self.organization:
            raise ValueError("HF_USERNAME not found in environment variables")
        
        self.dataset_path = f"{self.organization}/{self.dataset_name}"
        self.api = HfApi(token=os.getenv("HF_TOKEN"))
    
    def ensure_dataset_exists(self) -> None:
        """Ensure the dataset exists on Hugging Face."""
        try:
            self.api.repo_info(repo_id=self.dataset_path, repo_type="dataset")
        except RepositoryNotFoundError:
            self.api.create_repo(
                repo_id=self.dataset_path,
                repo_type="dataset",
                private=True,
                exist_ok=True
            )
    
    def upload_job_definitions(self, csv_file: Path, status: str = "todo") -> None:
        """Upload job definitions CSV to the dataset.
        
        Args:
            csv_file: Path to CSV file with job definitions
            status: Job status (todo, in-progress, done)
        """
        self.ensure_dataset_exists()
        
        remote_path = f"job-definitions/{status}/{csv_file.name}"
        
        self.api.upload_file(
            path_or_fileobj=str(csv_file),
            path_in_repo=remote_path,
            repo_id=self.dataset_path,
            repo_type="dataset",
            commit_message=f"Add job definitions: {csv_file.name}"
        )
    
    def list_jobs(self, status: Optional[str] = None) -> list[dict]:
        """List jobs in the dataset.
        
        Args:
            status: Filter by job status (todo, in-progress, done)
            
        Returns:
            List of job information dictionaries
        """
        self.ensure_dataset_exists()
        
        jobs = []
        try:
            files = self.api.list_repo_files(
                repo_id=self.dataset_path,
                repo_type="dataset"
            )
            
            job_files = [f for f in files if f.startswith("job-definitions/") and f.endswith(".csv")]
            
            for file_path in job_files:
                path_parts = file_path.split("/")
                if len(path_parts) >= 3:
                    file_status = path_parts[2]
                    filename = path_parts[-1]
                    
                    if status and file_status != status:
                        continue
                    
                    jobs.append({
                        'filename': filename,
                        'status': file_status,
                        'path': file_path
                    })
        
        except Exception as e:
            print(f"Error listing jobs: {e}")
        
        return jobs


def create_example_job_csv(output_path: Path) -> None:
    """Create an example job definition CSV.
    
    Args:
        output_path: Path to save the CSV file
    """
    jobs = [
        {
            'job_id': 'garch_example_001',
            'simulation_type': 'GJRGARCH',
            'parameters': '{"alpha": 0.1, "gamma": 0.05, "beta": 0.85, "sigma0_sq": 1.0, "eta": 10.0, "lam": 0.0}',
            'output_config': '{"n_simulations": 1000, "n_periods": 252, "save_plots": true}',
            'priority': 'medium',
            'created_at': '2024-01-15T10:00:00Z',
            'created_by': 'example_user'
        }
    ]
    
    df = pd.DataFrame(jobs)
    df.to_csv(output_path, index=False)
    print(f"Created example job CSV: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("HF Job Manager Example")
    print("=" * 50)
    
    # Check if credentials are loaded
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    
    if not hf_token:
        print("❌ HF_TOKEN not found in environment variables")
        print("Please create a .env file with:")
        print("HF_TOKEN=your_token_here")
        print("HF_USERNAME=your_username")
        exit(1)
    
    if not hf_username:
        print("❌ HF_USERNAME not found in environment variables")
        print("Please add HF_USERNAME=your_username to your .env file")
        exit(1)
    
    print(f"✅ Credentials loaded for user: {hf_username}")
    
    # Create example job CSV
    example_csv = Path("example_jobs.csv")
    create_example_job_csv(example_csv)
    
    # Initialize job manager
    try:
        job_manager = HFJobManager()
        print(f"✅ Connected to dataset: {job_manager.dataset_path}")
        
        # List existing jobs
        jobs = job_manager.list_jobs()
        print(f"📋 Found {len(jobs)} job files")
        
        # Upload example job
        job_manager.upload_job_definitions(example_csv)
        print("✅ Uploaded example job definitions")
        
        # Clean up
        example_csv.unlink()
        print("🧹 Cleaned up example file")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Clean up on error
        if example_csv.exists():
            example_csv.unlink()
