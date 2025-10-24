#!/usr/bin/env python3
"""
Test script to verify that .txt EXPLAIN ANALYZE files can be loaded for training.

This script demonstrates:
1. Loading raw EXPLAIN ANALYZE .txt files directly
2. Using the unified AbstractPlanParser interface
3. Creating training datasets from .txt files

Usage:
    python test_txt_loading.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from cross_db_benchmark.benchmark_tools.trino.parse_plan import TrinoPlanParser
from training.dataset.dataset_creation import read_workload_runs


def test_direct_parsing():
    """Test 1: Direct parsing using TrinoPlanParser"""
    print("=" * 80)
    print("Test 1: Direct parsing using TrinoPlanParser.parse_explain_analyze_file()")
    print("=" * 80)
    
    # Path to the explain_analyze_results directory
    txt_file = Path("/Users/an/query_engine/explain_analyze_results/accidents_complex_workload_200k_s1_explain_analyze.txt")
    
    if not txt_file.exists():
        print(f"❌ File not found: {txt_file}")
        return False
    
    parser = TrinoPlanParser()
    
    try:
        parsed_plans, runtimes = parser.parse_explain_analyze_file(
            str(txt_file),
            min_runtime=100,
            max_runtime=30000
        )
        
        print(f"✅ Successfully parsed {len(parsed_plans)} plans")
        print(f"   Runtime range: {min(runtimes):.2f}ms - {max(runtimes):.2f}ms")
        print(f"   Average runtime: {sum(runtimes)/len(runtimes):.2f}ms")
        
        # Show first plan details
        if parsed_plans:
            plan = parsed_plans[0]
            print(f"\n📊 First plan details:")
            print(f"   Operator: {plan.op_name}")
            print(f"   Est card: {plan.est_card}")
            print(f"   Act card: {plan.act_card}")
            print(f"   Database type: {plan.database_type}")
            print(f"   Runtime: {plan.plan_runtime}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline():
    """Test 2: Loading through training pipeline"""
    print("\n" + "=" * 80)
    print("Test 2: Loading through training pipeline (read_workload_runs)")
    print("=" * 80)
    
    # List of .txt files
    txt_files = [
        Path("/Users/an/query_engine/explain_analyze_results/accidents_complex_workload_200k_s1_explain_analyze.txt"),
        Path("/Users/an/query_engine/explain_analyze_results/airline_complex_workload_200k_s1_explain_analyze.txt"),
    ]
    
    # Filter to existing files
    existing_files = [f for f in txt_files if f.exists()]
    
    if not existing_files:
        print(f"❌ No files found in: {txt_files[0].parent}")
        return False
    
    print(f"Loading {len(existing_files)} files:")
    for f in existing_files:
        print(f"  - {f.name}")
    
    try:
        plans, database_statistics = read_workload_runs(
            workload_run_paths=existing_files,
            execution_mode=None,  # Not used for .txt files
            limit_queries=None,
            limit_queries_affected_wl=None
        )
        
        print(f"\n✅ Successfully loaded {len(plans)} plans from {len(existing_files)} files")
        print(f"   Database statistics keys: {list(database_statistics.keys())}")
        
        # Show plan distribution
        from collections import Counter
        op_counts = Counter()
        for plan in plans:
            op_counts[plan.op_name] += 1
        
        print(f"\n📊 Top operators:")
        for op, count in op_counts.most_common(5):
            print(f"   {op:30s}: {count:4d}")
        
        # Show database_id distribution
        db_id_counts = Counter(plan.database_id for plan in plans)
        print(f"\n📊 Plans per database:")
        for db_id, count in sorted(db_id_counts.items()):
            print(f"   Database {db_id}: {count} plans")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_creation():
    """Test 3: Creating training dataset"""
    print("\n" + "=" * 80)
    print("Test 3: Creating training dataset from .txt files")
    print("=" * 80)
    
    from training.dataset.dataset_creation import create_datasets
    from classes.classes import ZeroShotModelConfig
    
    # List of .txt files
    txt_files = [
        Path("/Users/an/query_engine/explain_analyze_results/accidents_complex_workload_200k_s1_explain_analyze.txt"),
        Path("/Users/an/query_engine/explain_analyze_results/airline_complex_workload_200k_s1_explain_analyze.txt"),
    ]
    
    # Filter to existing files
    existing_files = [f for f in txt_files if f.exists()]
    
    if not existing_files:
        print(f"❌ No files found")
        return False
    
    try:
        # Create a minimal config
        config = ZeroShotModelConfig(
            name=type('obj', (object,), {'NAME': 'test_model'}),
            limit_queries=None,
            limit_queries_affected_wl=None,
            cap_training_samples=None
        )
        
        label_norm, train_dataset, val_dataset, database_statistics = create_datasets(
            workload_run_paths=existing_files,
            model_config=config,
            val_ratio=0.15,
            shuffle_before_split=False
        )
        
        print(f"✅ Successfully created datasets:")
        print(f"   Training samples: {len(train_dataset) if train_dataset else 0}")
        print(f"   Validation samples: {len(val_dataset) if val_dataset else 0}")
        print(f"   Label normalizer: {label_norm}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 Testing .txt file loading for zero-shot training\n")
    
    results = []
    
    # Test 1: Direct parsing
    results.append(("Direct parsing", test_direct_parsing()))
    
    # Test 2: Training pipeline
    results.append(("Training pipeline", test_training_pipeline()))
    
    # Test 3: Dataset creation
    results.append(("Dataset creation", test_dataset_creation()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\n💡 You can now use .txt files directly in your training pipeline:")
        print("   1. Place .txt files in explain_analyze_results/")
        print("   2. Add them to your workload_runs list")
        print("   3. The training pipeline will automatically detect and parse them")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

