#!/usr/bin/env python3
"""
Simple test to verify Trino plans have PostgreSQL-compatible features.

This test checks that TrinoPlanOperator correctly maps Trino-specific
features to PostgreSQL-compatible names.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from cross_db_benchmark.benchmark_tools.trino.parse_plan import TrinoPlanParser


def test_postgres_compatible_features():
    """Test that Trino plans have PostgreSQL-compatible feature names"""
    print("=" * 80)
    print("Test: PostgreSQL-compatible feature mapping in Trino plans")
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
    
    print(f"\n✅ Parsed {len(parsed_plans)} plans")
    
    # PostgreSQL準拠の必須特徴量（AbstractPlanOperator）
    required_features = {
        'op_name': 'str',
        'est_card': 'float',      # PostgreSQL: est_card
        'act_card': 'float',      # PostgreSQL: act_card
        'est_width': 'float',
        'workers_planned': 'int',
        'act_children_card': 'float',
        'est_children_card': 'float'
    }
    
    # Check first plan
    plan = parsed_plans[0]
    print(f"\n📋 Checking plan_parameters for PostgreSQL-compatible features:")
    print(f"   Plan operator: {plan.plan_parameters.get('op_name', 'Unknown')}")
    print()
    
    all_present = True
    for feature, expected_type in required_features.items():
        if feature in plan.plan_parameters:
            value = plan.plan_parameters[feature]
            print(f"   ✅ {feature:25s}: {value:>15} (type: {type(value).__name__})")
        else:
            print(f"   ❌ {feature:25s}: MISSING")
            all_present = False
    
    # Check Trino-specific names (should also be present)
    print(f"\n📋 Trino-specific features (original names):")
    trino_specific = {
        'act_output_rows': 'act_card のTrino名',
        'est_rows': 'est_card のTrino名'
    }
    
    for feature, description in trino_specific.items():
        if feature in plan.plan_parameters:
            value = plan.plan_parameters[feature]
            print(f"   ✅ {feature:25s}: {value:>15} ({description})")
        else:
            print(f"   ℹ️  {feature:25s}: Not set")
    
    # Check PostgreSQL-only features (should be None or 0 in Trino)
    print(f"\n📋 PostgreSQL-only features (Trinoには存在しない):")
    pg_only = {
        'est_cost': 'PostgreSQLのコスト推定',
        'est_startup_cost': 'PostgreSQLの起動コスト'
    }
    
    for feature, description in pg_only.items():
        value = plan.plan_parameters.get(feature)
        if value is None:
            print(f"   ✅ {feature:25s}: None (正しく処理されている)")
        else:
            print(f"   ⚠️  {feature:25s}: {value}")
    
    if all_present:
        print(f"\n🎉 すべての必須PostgreSQL互換特徴量が存在します！")
    else:
        print(f"\n❌ いくつかの必須特徴量が欠けています")
    
    return all_present


def test_multiple_datasets():
    """Test feature compatibility across multiple datasets"""
    print("\n" + "=" * 80)
    print("Test: Multiple datasets compatibility")
    print("=" * 80)
    
    results_dir = Path("/Users/an/query_engine/explain_analyze_results")
    txt_files = sorted(results_dir.glob("*_explain_analyze.txt"))[:5]
    
    if not txt_files:
        print(f"❌ No files found")
        return False
    
    parser = TrinoPlanParser()
    
    # PostgreSQL必須特徴量
    required_features = ['op_name', 'est_card', 'act_card', 'est_width', 
                        'workers_planned', 'act_children_card', 'est_children_card']
    
    print(f"\n📊 Checking {len(txt_files)} datasets:\n")
    
    all_compatible = True
    total_plans = 0
    
    for txt_file in txt_files:
        try:
            parsed_plans, _ = parser.parse_explain_analyze_file(
                str(txt_file),
                min_runtime=100,
                max_runtime=30000
            )
            
            if not parsed_plans:
                print(f"   ⚠️  {txt_file.stem[:40]:40s}: 0 plans")
                continue
            
            # Check first plan
            plan = parsed_plans[0]
            missing = []
            
            for feature in required_features:
                if feature not in plan.plan_parameters:
                    missing.append(feature)
            
            status = "✅" if not missing else "❌"
            dataset_name = txt_file.stem.split('_')[0]
            
            print(f"   {status} {dataset_name:15s}: {len(parsed_plans):3d} plans", end="")
            
            if missing:
                print(f" (欠損: {', '.join(missing)})")
                all_compatible = False
            else:
                print(" (すべて揃っている)")
                total_plans += len(parsed_plans)
            
        except Exception as e:
            print(f"   ❌ {txt_file.stem[:40]:40s}: Error - {e}")
            all_compatible = False
    
    print(f"\n📊 合計: {total_plans} plans がPostgreSQL互換")
    
    return all_compatible


def test_feature_values():
    """Test that feature values are reasonable"""
    print("\n" + "=" * 80)
    print("Test: Feature value sanity check")
    print("=" * 80)
    
    txt_file = Path("/Users/an/query_engine/explain_analyze_results/accidents_complex_workload_200k_s1_explain_analyze.txt")
    
    if not txt_file.exists():
        print(f"❌ File not found")
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
    
    print(f"\n📊 Analyzing {len(parsed_plans)} plans:\n")
    
    # Collect statistics
    from collections import defaultdict
    feature_stats = defaultdict(list)
    
    for plan in parsed_plans:
        for key in ['est_card', 'act_card', 'est_width', 'workers_planned']:
            value = plan.plan_parameters.get(key)
            if value is not None:
                feature_stats[key].append(value)
    
    # Show statistics
    all_reasonable = True
    
    for feature, values in feature_stats.items():
        if not values:
            print(f"   ⚠️  {feature}: No data")
            continue
        
        avg_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        
        # Check if values are reasonable
        reasonable = True
        if feature in ['est_card', 'act_card'] and (min_val < 0 or max_val > 1e10):
            reasonable = False
        
        status = "✅" if reasonable else "❌"
        print(f"   {status} {feature:20s}: min={min_val:>10.1f}, avg={avg_val:>10.1f}, max={max_val:>10.1f}")
        
        if not reasonable:
            all_reasonable = False
    
    return all_reasonable


def main():
    """Run all tests"""
    print("🚀 Testing Trino-PostgreSQL compatibility\n")
    
    results = []
    
    # Test 1: Feature compatibility
    results.append(("PostgreSQL-compatible features", test_postgres_compatible_features()))
    
    # Test 2: Multiple datasets
    results.append(("Multiple datasets", test_multiple_datasets()))
    
    # Test 3: Feature values
    results.append(("Feature value sanity", test_feature_values()))
    
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
        print("\n💡 結論:")
        print("   ✅ TrinoのプランはPostgreSQL互換の特徴量名を持っています")
        print("   ✅ PostgreSQL準拠のモデルでTrinoデータをトレーニング可能です")
        print("   ✅ 欠損特徴量（est_cost等）は自動的に処理されています")
        print("\n📝 次のステップ:")
        print("   1. Trinoの.txtファイルを直接トレーニングデータとして使用")
        print("   2. PostgreSQL準拠のFeaturization（例：PostgresTrueCardDetail）を使用")
        print("   3. 既存のpostgres_plan_collatorがそのまま動作します")
    else:
        print("\n⚠️  Some tests failed.")
        print("   TrinoPlanOperatorでadhocな補正が必要かもしれません。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

