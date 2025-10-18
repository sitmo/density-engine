#!/usr/bin/env python3
"""
Test script for GPU validation in ClusterManager.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from density_engine.cluster.gpu_names import RTX_30_PLUS
from density_engine.cluster.vast_api import validate_gpu_names

def test_gpu_validation():
    """Test GPU validation functionality."""
    print("🧪 Testing GPU Validation")
    print("=" * 40)
    
    # Test 1: Default RTX_30_PLUS
    print(f"\n1. RTX_30_PLUS list ({len(RTX_30_PLUS)} GPUs):")
    print(f"   {RTX_30_PLUS[:5]}{'...' if len(RTX_30_PLUS) > 5 else ''}")
    
    # Test 2: Validate RTX_30_PLUS
    validated_rtx30 = validate_gpu_names(RTX_30_PLUS)
    print(f"\n2. Validated RTX_30_PLUS ({len(validated_rtx30)} GPUs):")
    print(f"   {validated_rtx30[:5]}{'...' if len(validated_rtx30) > 5 else ''}")
    
    # Test 3: Custom GPU list (using correct Vast.ai format)
    custom_gpus = ["RTX 4090", "RTX 4080", "RTX 3070", "INVALID_GPU", "RTX 3060"]
    print(f"\n3. Custom GPU list: {custom_gpus}")
    validated_custom = validate_gpu_names(custom_gpus)
    print(f"   Validated ({len(validated_custom)} GPUs): {validated_custom}")
    
    # Test 4: Empty list
    empty_list = []
    validated_empty = validate_gpu_names(empty_list)
    print(f"\n4. Empty list: {validated_empty}")
    
    # Test 5: None (should use RTX_30_PLUS)
    validated_none = validate_gpu_names(None or RTX_30_PLUS)
    print(f"\n5. None -> RTX_30_PLUS ({len(validated_none)} GPUs):")
    print(f"   {validated_none[:5]}{'...' if len(validated_none) > 5 else ''}")
    
    print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    test_gpu_validation()
