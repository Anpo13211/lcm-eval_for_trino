#!/usr/bin/env python3
"""
Simple test script to verify .txt EXPLAIN ANALYZE file parsing.

This script tests only the core parsing functionality without
requiring the full training pipeline setup.

Usage:
    python test_txt_loading_simple.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from cross_db_benchmark.benchmark_tools.trino.parse_plan import TrinoPlanParser


def test_single_file():
    """Test parsing a single .txt file"""
    print("=" * 80)
    print("Test: Parsing single .txt file with TrinoPlanParser")
    print("=" * 80)
    
    # Path to the explain_analyze_results directory
    txt_file = Path("/Users/an/query_engine/explain_analyze_results/accidents_complex_workload_200k_s1_explain_analyze.txt")
    
    if not txt_file.exists():
        print(f"❌ File not found: {txt_file}")
        print(f"\n💡 Please ensure explain_analyze_results directory exists with .txt files")
        return False
    
    parser = TrinoPlanParser()
    
    try:
        print(f"\n📂 Parsing: {txt_file.name}")
        
        parsed_plans, runtimes = parser.parse_explain_analyze_file(
            str(txt_file),
            min_runtime=100,
            max_runtime=30000
        )
        
        print(f"\n✅ Successfully parsed {len(parsed_plans)} plans")
        
        if not parsed_plans:
            print("⚠️  No plans passed the runtime filter (100ms - 30000ms)")
            return False
        
        print(f"\n📊 Runtime Statistics:")
        print(f"   Min:     {min(runtimes):8.2f} ms")
        print(f"   Max:     {max(runtimes):8.2f} ms")
        print(f"   Average: {sum(runtimes)/len(runtimes):8.2f} ms")
        print(f"   Median:  {sorted(runtimes)[len(runtimes)//2]:8.2f} ms")
        
        # Show first plan details
        plan = parsed_plans[0]
        print(f"\n📋 Sample Plan (first):")
        print(f"   Operator:      {plan.op_name}")
        print(f"   Est card:      {plan.est_card:,.0f}")
        print(f"   Act card:      {plan.act_card:,.0f}")
        print(f"   Est width:     {plan.est_width:.0f} bytes")
        print(f"   Database type: {plan.database_type}")
        print(f"   Database ID:   {plan.database_id}")
        print(f"   Runtime:       {plan.plan_runtime:.2f} ms")
        print(f"   Children:      {len(plan.children)}")
        
        # Operator distribution
        from collections import Counter
        op_counts = Counter()
        
        def count_ops(node):
            op_counts[node.op_name] += 1
            for child in node.children:
                count_ops(child)
        
        for plan in parsed_plans:
            count_ops(plan)
        
        print(f"\n📊 Top 10 Operators:")
        for op, count in op_counts.most_common(10):
            print(f"   {op:35s}: {count:4d}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_files():
    """Test parsing multiple .txt files"""
    print("\n" + "=" * 80)
    print("Test: Parsing multiple .txt files")
    print("=" * 80)
    
    results_dir = Path("/Users/an/query_engine/explain_analyze_results")
    
    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        return False
    
    txt_files = sorted(results_dir.glob("*_explain_analyze.txt"))
    
    if not txt_files:
        print(f"❌ No .txt files found in {results_dir}")
        return False
    
    print(f"\n📂 Found {len(txt_files)} .txt files")
    
    parser = TrinoPlanParser()
    
    all_plans = []
    all_runtimes = []
    file_results = []
    
    for txt_file in txt_files[:5]:  # Test first 5 files
        try:
            parsed_plans, runtimes = parser.parse_explain_analyze_file(
                str(txt_file),
                min_runtime=100,
                max_runtime=30000
            )
            
            all_plans.extend(parsed_plans)
            all_runtimes.extend(runtimes)
            
            file_results.append({
                'name': txt_file.stem,
                'plans': len(parsed_plans),
                'avg_runtime': sum(runtimes) / len(runtimes) if runtimes else 0
            })
            
            print(f"   ✅ {txt_file.stem[:40]:40s}: {len(parsed_plans):3d} plans")
            
        except Exception as e:
            print(f"   ❌ {txt_file.stem[:40]:40s}: {e}")
            file_results.append({
                'name': txt_file.stem,
                'plans': 0,
                'avg_runtime': 0
            })
    
    if all_plans:
        print(f"\n📊 Overall Statistics:")
        print(f"   Total plans:    {len(all_plans)}")
        print(f"   Avg runtime:    {sum(all_runtimes)/len(all_runtimes):.2f} ms")
        print(f"   Files processed: {len([r for r in file_results if r['plans'] > 0])}/{len(file_results)}")
        
        return True
    else:
        print("\n❌ No plans were successfully parsed")
        return False


def test_raw_plan_interface():
    """Test the parse_raw_plan unified interface"""
    print("\n" + "=" * 80)
    print("Test: Unified parse_raw_plan interface")
    print("=" * 80)
    
    # Sample Trino EXPLAIN ANALYZE output
    sample_plan = '''Trino version: 475
Queued: 800.00us, Analysis: 1.23s, Planning: 155.02ms, Execution: 678.41ms
Fragment 1 [SINGLE]
    CPU: 2.80ms, Scheduled: 2.90ms, Blocked 2.87s (Input: 2.55s, Output: 0.00ns), Input: 1 row (9B); per task: avg.: 1.00 std.dev.: 0.00, Output: 1 row (9B)
    Peak Memory: 528B, Tasks count: 1; per task: max: 528B
    Output layout: [sum]
    Output partitioning: SINGLE []
    Aggregate[type = FINAL]
    │   Layout: [sum:double]
    │   Estimates: {rows: 1 (9B), cpu: ?, memory: 9B, network: 0B}
    │   CPU: 1.00ms (0.43%), Scheduled: 1.00ms (0.29%), Blocked: 0.00ns (0.00%), Output: 1 row (9B)
    │   Input avg.: 1.00 rows, Input std.dev.: 0.00%
    │   sum := sum(sum_8)
    └─ LocalExchange[partitioning = SINGLE]
       │   Layout: [sum_8:double]
       │   Estimates: {rows: ? (?), cpu: 0, memory: 0B, network: 0B}
'''
    
    parser = TrinoPlanParser()
    
    try:
        # Test the unified interface
        root_operator, execution_time, planning_time = parser.parse_raw_plan(
            sample_plan,
            analyze=True,
            parse=True
        )
        
        print(f"✅ parse_raw_plan() works correctly")
        print(f"\n📋 Parsed Results:")
        print(f"   Execution time:  {execution_time:.2f} ms")
        print(f"   Planning time:   {planning_time:.2f} ms")
        print(f"   Root operator:   {root_operator.op_name if root_operator else 'None'}")
        
        if root_operator:
            print(f"   Est card:        {root_operator.est_card}")
            print(f"   Act card:        {root_operator.act_card}")
            print(f"   Database type:   {root_operator.database_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 Testing .txt file loading for Trino\n")
    
    results = []
    
    # Test 1: Single file
    results.append(("Single file parsing", test_single_file()))
    
    # Test 2: Multiple files
    results.append(("Multiple files parsing", test_multiple_files()))
    
    # Test 3: Unified interface
    results.append(("Unified interface", test_raw_plan_interface()))
    
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
        print("\n💡 The .txt file parsing is working correctly.")
        print("   You can now use these files directly in your zero-shot training!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

