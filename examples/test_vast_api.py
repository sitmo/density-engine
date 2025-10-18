#!/usr/bin/env python3
"""
Test script to verify Vast API functions work correctly.

This script tests the core Vast API functions without renting any instances.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from density_engine.cluster.vast_api import (
    search_offers,
    find_best_offer,
    list_instances,
    validate_gpu_names
)
from density_engine.cluster.gpu_names import RTX_30_PLUS


def test_gpu_validation():
    """Test GPU name validation."""
    print("🔍 Testing GPU name validation...")
    
    valid_gpus = validate_gpu_names(RTX_30_PLUS[:5])
    print(f"✅ Valid RTX 30+ GPUs: {valid_gpus}")
    
    invalid_gpus = validate_gpu_names(["INVALID_GPU", "FAKE_RTX"])
    print(f"❌ Invalid GPUs: {invalid_gpus}")
    
    return len(valid_gpus) > 0


def test_search_offers():
    """Test searching for offers."""
    print("\n🔍 Testing offer search...")
    
    offers = search_offers(
        gpu_list=RTX_30_PLUS[:3],  # Test with first 3 RTX 30+ GPUs
        max_price=0.10,
        min_reliability=0.90,
        min_gpu_ram=8,
        limit=5
    )
    
    print(f"✅ Found {len(offers)} offers")
    
    if offers:
        print("📋 Sample offers:")
        for i, offer in enumerate(offers[:3]):
            print(f"  {i+1}. {offer['gpu_name']} - ${offer['price_per_hour']:.3f}/hr "
                  f"(reliability: {offer['reliability']:.1%})")
    
    return len(offers) > 0


def test_find_best_offer():
    """Test finding the best offer."""
    print("\n🔍 Testing best offer search...")
    
    best_offer = find_best_offer(
        gpu_list=RTX_30_PLUS[:3],
        max_price=0.10,
        min_reliability=0.90,
        min_gpu_ram=8,
        sort_by_price=True
    )
    
    if best_offer:
        print(f"✅ Best offer: {best_offer['gpu_name']} at ${best_offer['price_per_hour']:.3f}/hr")
        print(f"   Reliability: {best_offer['reliability']:.1%}")
        print(f"   Location: {best_offer['location']}")
        return True
    else:
        print("❌ No offers found")
        return False


def test_list_instances():
    """Test listing current instances."""
    print("\n🔍 Testing instance listing...")
    
    instances = list_instances()
    print(f"✅ Found {len(instances)} current instances")
    
    if instances:
        print("📋 Current instances:")
        for instance in instances[:3]:  # Show first 3
            print(f"  - {instance['id']}: {instance['gpu_name']} "
                  f"({instance['status']}) - ${instance['price_per_hour']:.3f}/hr")
    
    return True  # This should always work


def main():
    """Run all tests."""
    print("🧪 Vast API Test Suite")
    print("=" * 40)
    
    tests = [
        ("GPU Validation", test_gpu_validation),
        ("Search Offers", test_search_offers),
        ("Find Best Offer", test_find_best_offer),
        ("List Instances", test_list_instances),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    print("\n📊 Test Results:")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Vast API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check your Vast.ai configuration.")


if __name__ == "__main__":
    main()
