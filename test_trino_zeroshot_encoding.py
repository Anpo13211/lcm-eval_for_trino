"""
Trino版zero-shotモデルのエンコーディング機能のテスト
"""
import sys
import os
from pathlib import Path

# 環境変数の設定（classes.pyのimport前に実行）
for i in range(11):
    env_key = f'NODE{i:02d}'
    if os.environ.get(env_key) in (None, '', 'None'):
        os.environ[env_key] = '[]'

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pathlib import Path
from types import SimpleNamespace

def test_trino_zeroshot_encoding():
    """Trino版zero-shotモデルのエンコーディングをテスト"""
    
    print("=" * 60)
    print("Trino版Zero-Shotモデルのエンコーディングテスト")
    print("=" * 60)
    
    # 1. テストデータの準備
    txt_file = Path("/Users/an/query_engine/explain_analyze_results/accidents_combined_workloads_explain_analyze.txt")
    if not txt_file.exists():
        print(f"❌ テストファイルが見つかりません: {txt_file}")
        return False
    
    print(f"\n1. テストファイル読み込み: {txt_file}")
    
    # 2. クエリプランの読み込み
    try:
        from training.dataset.dataset_creation import read_explain_analyze_txt
        
        plans, database_stats = read_explain_analyze_txt(
            txt_file,
            path_index=0,
            limit_per_ds=2  # 最初の2クエリのみテスト
        )
        
        print(f"✅ {len(plans)} プランを読み込みました")
        print(f"   - database_stats.table_stats: {len(database_stats.table_stats)} テーブル")
        print(f"   - database_stats.column_stats: {len(database_stats.column_stats)} カラム")
        
    except Exception as e:
        print(f"❌ プラン読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 統計情報の確認
    if len(database_stats.table_stats) == 0:
        print("❌ テーブル統計が空です")
        return False
    
    if len(database_stats.column_stats) == 0:
        print("❌ カラム統計が空です")
        return False
    
    # 4. feature_statisticsの準備（ダミー）
    # 実際には学習時に生成されるが、テスト用に最小限の構造を作成
    try:
        import numpy as np
        feature_statistics = {
            'op_name': {
                'type': 'categorical',
                'value_dict': {'ScanFilterProject': 0, 'FilterProject': 1, 'Output': 2},
                'no_vals': 3
            },
            'act_card': {
                'type': 'numeric',
                'center': 1000.0,
                'scale': 500.0,
            },
            'act_output_rows': {
                'type': 'numeric',
                'center': 1000.0,
                'scale': 500.0,
            },
            'est_card': {
                'type': 'numeric',
                'center': 1000.0,
                'scale': 500.0,
            },
            'est_rows': {
                'type': 'numeric',
                'center': 1000.0,
                'scale': 500.0,
            },
            'est_width': {
                'type': 'numeric',
                'center': 8.0,
                'scale': 4.0,
            },
            'workers_planned': {
                'type': 'numeric',
                'center': 1.0,
                'scale': 0.5,
            },
            'act_children_card': {
                'type': 'numeric',
                'center': 1000.0,
                'scale': 500.0,
            },
            'null_frac': {
                'type': 'numeric',
                'center': 0.1,
                'scale': 0.2,
            },
            'n_distinct': {
                'type': 'numeric',
                'center': 100.0,
                'scale': 50.0,
            },
            'avg_width': {
                'type': 'numeric',
                'center': 8.0,
                'scale': 4.0,
            },
            'correlation': {
                'type': 'numeric',
                'center': 0.0,
                'scale': 0.5,
            },
            'data_type': {
                'type': 'categorical',
                'value_dict': {'int': 0, 'varchar': 1, 'misc': 2},
                'no_vals': 3
            },
            'reltuples': {
                'type': 'numeric',
                'center': 10000.0,
                'scale': 5000.0,
            },
            'relpages': {
                'type': 'numeric',
                'center': 100.0,
                'scale': 50.0,
            },
            'operator': {
                'type': 'categorical',
                'value_dict': {'$eq': 0, '$gt': 1, '$lt': 2},
                'no_vals': 3
            },
            'aggregation': {
                'type': 'categorical',
                'value_dict': {'Aggregator.COUNT': 0, 'Aggregator.SUM': 1, None: 2},
                'no_vals': 3
            }
        }
        
        # RobustScalerを追加（add_numerical_scalersが必要）
        from sklearn.preprocessing import RobustScaler
        from training.preprocessing.feature_statistics import FeatureType
        
        for k, v in feature_statistics.items():
            if v.get('type') == str(FeatureType.numeric):
                scaler = RobustScaler()
                scaler.center_ = np.array([v['center']])
                scaler.scale_ = np.array([v['scale']])
                feature_statistics[k]['scaler'] = scaler
        
        print("\n✅ feature_statisticsを準備しました")
        
    except Exception as e:
        print(f"❌ feature_statistics準備エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. plan_featurizationの準備
    try:
        from training.featurizations import PostgresTrueCardDetail
        
        plan_featurization = PostgresTrueCardDetail()
        print("\n✅ plan_featurizationを準備しました")
        
    except Exception as e:
        print(f"❌ plan_featurization準備エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. db_statisticsの準備（Postgres形式）
    try:
        db_statistics = {
            0: database_stats  # database_id=0に対応する統計情報
        }
        print("\n✅ db_statisticsを準備しました")
        print(f"   - table_stats: {len(db_statistics[0].table_stats)} テーブル")
        print(f"   - column_stats: {len(db_statistics[0].column_stats)} カラム")
        
    except Exception as e:
        print(f"❌ db_statistics準備エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. trino_plan_collatorの呼び出し
    try:
        from models.zeroshot.trino_plan_batching import trino_plan_collator
        import numpy as np
        
        print("\n7. trino_plan_collatorを呼び出し中...")
        
        # plansを(sample_idx, plan)のタプルリストに変換
        plans_with_idx = [(i, plan) for i, plan in enumerate(plans)]
        
        graph, features, labels, sample_idxs = trino_plan_collator(
            plans=plans_with_idx,
            feature_statistics=feature_statistics,
            db_statistics=db_statistics,
            plan_featurization=plan_featurization
        )
        
        print(f"✅ trino_plan_collator成功")
        print(f"   - graph: {graph}")
        print(f"   - features keys: {list(features.keys())}")
        print(f"   - labels shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
        print(f"   - sample_idxs: {sample_idxs}")
        
        # 特徴量の確認
        for feat_name, feat_tensor in features.items():
            if hasattr(feat_tensor, 'shape'):
                print(f"   - {feat_name}: shape={feat_tensor.shape}")
            else:
                print(f"   - {feat_name}: len={len(feat_tensor)}")
        
        return True
        
    except Exception as e:
        print(f"❌ trino_plan_collatorエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        import numpy as np
        from sklearn.preprocessing import RobustScaler
        from training.preprocessing.feature_statistics import FeatureType
    except ImportError as e:
        print(f"❌ 必要なモジュールがインストールされていません: {e}")
        print("   pip install numpy scikit-learn を実行してください")
        sys.exit(1)
    
    success = test_trino_zeroshot_encoding()
    if success:
        print("\n" + "=" * 60)
        print("✅ テスト成功")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ テスト失敗")
        print("=" * 60)
        sys.exit(1)

