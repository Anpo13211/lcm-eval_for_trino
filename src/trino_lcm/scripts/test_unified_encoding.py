#!/usr/bin/env python3
"""
Test if PostgreSQL featurization works with Trino plan operators.

This test verifies that:
1. TrinoPlanOperator maps Trino-specific fields to PostgreSQL-compatible names
2. postgres_plan_collator can process Trino plans
3. Missing features (e.g., est_cost) are handled correctly
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from cross_db_benchmark.benchmark_tools.trino.parse_plan import TrinoPlanParser
from training.featurizations import PostgresTrueCardDetail
from models.zeroshot.postgres_plan_batching import postgres_plan_collator
import numpy as np


def test_feature_mapping():
    """Test 1: Verify that Trino features are mapped to PostgreSQL names"""
    print("=" * 80)
    print("Test 1: Feature mapping (Trino → PostgreSQL compatible names)")
    print("=" * 80)
    
    txt_file = Path("/Users/an/query_engine/explain_analyze_results/accidents_complex_workload_200k_s1_explain_analyze.txt")
    
    if not txt_file.exists():
        print(f"❌ File not found: {txt_file}")
        return False
    
    parser = TrinoPlanParser()
    parsed_plans, runtimes = parser.parse_explain_analyze_file(
        str(txt_file),
        min_runtime=100,
        max_runtime=30000
    )
    
    if not parsed_plans:
        print("❌ No plans parsed")
        return False
    
    plan = parsed_plans[0]
    
    # Check PostgreSQL-compatible feature names
    postgres_features = ['op_name', 'act_card', 'est_card', 'est_width', 'workers_planned', 
                         'act_children_card', 'est_children_card']
    
    print(f"\n📋 Checking plan_parameters for PostgreSQL-compatible names:")
    all_present = True
    
    for feature in postgres_features:
        if feature in plan.plan_parameters:
            value = plan.plan_parameters[feature]
            print(f"   ✅ {feature:25s}: {value}")
        else:
            print(f"   ❌ {feature:25s}: MISSING")
            all_present = False
    
    # Check Trino-specific names are also present (for reference)
    print(f"\n📋 Trino-specific names (should also exist):")
    trino_features = ['act_output_rows', 'est_rows']
    for feature in trino_features:
        if feature in plan.plan_parameters:
            value = plan.plan_parameters[feature]
            print(f"   ✅ {feature:25s}: {value}")
    
    # Check features that don't exist in Trino
    print(f"\n📋 PostgreSQL-only features (should be None or 0):")
    pg_only_features = ['est_cost', 'est_startup_cost']
    for feature in pg_only_features:
        value = plan.plan_parameters.get(feature, 'NOT SET')
        print(f"   ℹ️  {feature:25s}: {value}")
    
    return all_present


def test_postgres_featurization_on_trino():
    """Test 2: Use PostgreSQL featurization on Trino plans"""
    print("\n" + "=" * 80)
    print("Test 2: PostgreSQL featurization on Trino plans")
    print("=" * 80)
    
    txt_file = Path("/Users/an/query_engine/explain_analyze_results/accidents_complex_workload_200k_s1_explain_analyze.txt")
    
    if not txt_file.exists():
        print(f"❌ File not found: {txt_file}")
        return False
    
    parser = TrinoPlanParser()
    parsed_plans, runtimes = parser.parse_explain_analyze_file(
        str(txt_file),
        min_runtime=100,
        max_runtime=30000
    )
    
    if not parsed_plans:
        print("❌ No plans parsed")
        return False
    
    # Use PostgreSQL featurization
    featurization = PostgresTrueCardDetail()
    
    print(f"\n📊 PostgresTrueCardDetail features:")
    print(f"   PLAN_FEATURES: {featurization.PLAN_FEATURES}")
    print(f"   FILTER_FEATURES: {featurization.FILTER_FEATURES}")
    print(f"   COLUMN_FEATURES: {featurization.COLUMN_FEATURES}")
    
    # Check if all required features are present in Trino plans
    plan = parsed_plans[0]
    missing_features = []
    
    for feature in featurization.PLAN_FEATURES:
        if feature not in plan.plan_parameters:
            missing_features.append(feature)
    
    if missing_features:
        print(f"\n⚠️  Missing features in Trino plan:")
        for feature in missing_features:
            print(f"   - {feature}")
        print(f"\n💡 These should be set to 0 or None by TrinoPlanOperator")
        return False
    else:
        print(f"\n✅ All PostgreSQL features are available in Trino plan!")
        
        # Show values
        print(f"\n📋 Feature values:")
        for feature in featurization.PLAN_FEATURES:
            value = plan.plan_parameters[feature]
            print(f"   {feature:25s}: {value}")
    
    return True


def test_postgres_collator_on_trino():
    """Test 3: Use postgres_plan_collator on Trino plans"""
    print("\n" + "=" * 80)
    print("Test 3: postgres_plan_collator on Trino plans")
    print("=" * 80)
    
    txt_file = Path("/Users/an/query_engine/explain_analyze_results/accidents_complex_workload_200k_s1_explain_analyze.txt")
    
    if not txt_file.exists():
        print(f"❌ File not found: {txt_file}")
        return False
    
    parser = TrinoPlanParser()
    parsed_plans, runtimes = parser.parse_explain_analyze_file(
        str(txt_file),
        min_runtime=100,
        max_runtime=30000
    )
    
    if not parsed_plans:
        print("❌ No plans parsed")
        return False
    
    # Prepare minimal feature_statistics (normally from gather_feature_statistics)
    featurization = PostgresTrueCardDetail()
    
    # Create dummy feature statistics
    from training.preprocessing.feature_statistics import FeatureType
    from sklearn.preprocessing import RobustScaler
    
    feature_statistics = {}
    
    # Categorical features
    for feature in ['op_name', 'operator', 'aggregation', 'data_type']:
        feature_statistics[feature] = {
            'type': str(FeatureType.categorical),
            'value_dict': {}  # Will be populated
        }
    
    # Numerical features
    numerical_features = ['act_card', 'est_width', 'workers_planned', 'act_children_card',
                         'literal_feature', 'avg_width', 'correlation', 'n_distinct', 
                         'null_frac', 'reltuples', 'relpages']
    for feature in numerical_features:
        scaler = RobustScaler()
        scaler.fit(np.array([[0], [1], [100]]))
        feature_statistics[feature] = {
            'type': str(FeatureType.numeric),
            'scaler': scaler,
            'center': 1.0,
            'scale': 1.0
        }
    
    # Collect actual values for categorical features
    for plan in parsed_plans:
        op_name = plan.plan_parameters.get('op_name')
        if op_name and op_name not in feature_statistics['op_name']['value_dict']:
            idx = len(feature_statistics['op_name']['value_dict'])
            feature_statistics['op_name']['value_dict'][str(op_name)] = idx
    
    # Dummy database statistics
    from types import SimpleNamespace
    db_statistics = {
        0: SimpleNamespace(
            table_stats=[],
            column_stats=[]
        )
    }
    
    print(f"\n📦 Attempting to collate {len(parsed_plans)} Trino plans...")
    
    try:
        # Call postgres_plan_collator with Trino plans
        graph, features, labels, sample_idxs = postgres_plan_collator(
            plans=parsed_plans[:3],  # Test with first 3 plans
            feature_statistics=feature_statistics,
            db_statistics=db_statistics,
            plan_featurization=featurization
        )
        
        print(f"\n✅ Successfully created graph!")
        print(f"   Graph type: {type(graph)}")
        print(f"   Number of node types: {len(graph.ntypes)}")
        print(f"   Node types: {graph.ntypes}")
        print(f"   Number of edge types: {len(graph.etypes)}")
        print(f"   Labels shape: {len(labels)}")
        print(f"   Sample indices: {sample_idxs}")
        
        # Show feature dimensions
        print(f"\n📊 Feature shapes:")
        for node_type, feat in features.items():
            if hasattr(feat, 'shape'):
                print(f"   {node_type:20s}: {feat.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_coverage():
    """Test 4: Check feature coverage across multiple Trino datasets"""
    print("\n" + "=" * 80)
    print("Test 4: Feature coverage across multiple datasets")
    print("=" * 80)
    
    results_dir = Path("/Users/an/query_engine/explain_analyze_results")
    txt_files = sorted(results_dir.glob("*_explain_analyze.txt"))[:5]
    
    if not txt_files:
        print(f"❌ No files found")
        return False
    
    parser = TrinoPlanParser()
    featurization = PostgresTrueCardDetail()
    
    all_features_present = True
    
    for txt_file in txt_files:
        parsed_plans, _ = parser.parse_explain_analyze_file(
            str(txt_file),
            min_runtime=100,
            max_runtime=30000
        )
        
        if not parsed_plans:
            continue
        
        # Check first plan
        plan = parsed_plans[0]
        missing = []
        
        for feature in featurization.PLAN_FEATURES:
            if feature not in plan.plan_parameters:
                missing.append(feature)
        
        status = "✅" if not missing else "❌"
        dataset_name = txt_file.stem.split('_')[0]
        print(f"   {status} {dataset_name:15s}: {len(parsed_plans):3d} plans", end="")
        
        if missing:
            print(f" (missing: {missing})")
            all_features_present = False
        else:
            print()
    
    return all_features_present


def main():
    """Run all tests"""
    print("🚀 Testing PostgreSQL featurization on Trino plans\n")
    
    results = []
    
    # Test 1: Feature mapping
    results.append(("Feature mapping", test_feature_mapping()))
    
    # Test 2: PostgreSQL featurization
    results.append(("PostgreSQL featurization", test_postgres_featurization_on_trino()))
    
    # Test 3: postgres_plan_collator
    results.append(("postgres_plan_collator", test_postgres_collator_on_trino()))
    
    # Test 4: Feature coverage
    results.append(("Feature coverage", test_feature_coverage()))
    
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
        print("\n💡 PostgreSQL準拠のモデルでTrinoデータが使えます！")
        print("   - Trino固有の特徴量はPostgreSQL互換名にマッピング済み")
        print("   - 欠損特徴量（est_cost等）は自動的に0/Noneで埋められます")
        print("   - postgres_plan_collatorがそのまま使えます")
    else:
        print("\n⚠️  Some tests failed.")
        print("   必要に応じてTrinoPlanOperatorでadhocな補正を追加してください。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

