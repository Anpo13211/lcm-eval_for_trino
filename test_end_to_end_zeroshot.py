"""
Trino Zero-Shotモデルのエンドツーエンドテスト
"""
import sys
import os
from pathlib import Path

# 環境変数の設定（必須 - import前に実行）
for i in range(11):
    env_key = f'NODE{i:02d}'
    if os.environ.get(env_key) in (None, '', 'None'):
        os.environ[env_key] = '[]'

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace

def test_end_to_end():
    """エンドツーエンドテスト"""
    
    print("=" * 80)
    print("Trino Zero-Shotモデル エンドツーエンドテスト")
    print("=" * 80)
    
    # 1. データファイルの確認
    test_file = Path("/Users/an/query_engine/explain_analyze_results/accidents_combined_workloads_explain_analyze.txt")
    if not test_file.exists():
        print(f"❌ テストファイルが見つかりません: {test_file}")
        return False
    
    print(f"\n✅ テストファイル: {test_file}")
    
    # 2. プランの読み込み
    print("\n📂 ステップ1: プランの読み込み")
    try:
        from training.dataset.dataset_creation import read_explain_analyze_txt
        
        plans, database_stats = read_explain_analyze_txt(
            test_file,
            path_index=0,
            limit_per_ds=5  # 5プランでテスト
        )
        
        print(f"✅ {len(plans)} プランを読み込みました")
        print(f"   - database_stats.table_stats: {len(database_stats.table_stats)} テーブル")
        print(f"   - database_stats.column_stats: {len(database_stats.column_stats)} カラム")
        
        if len(plans) == 0:
            print("❌ プランが0個です")
            return False
            
    except Exception as e:
        print(f"❌ プラン読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 統計情報の準備
    print("\n📊 ステップ2: 統計情報の準備")
    try:
        # db_statisticsをPostgres形式で準備
        db_statistics = {
            0: database_stats
        }
        print(f"✅ db_statistics準備完了")
        print(f"   - table_stats: {len(db_statistics[0].table_stats)} テーブル")
        print(f"   - column_stats: {len(db_statistics[0].column_stats)} カラム")
    except Exception as e:
        print(f"❌ 統計情報準備エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 特徴量統計の準備
    print("\n🔧 ステップ3: 特徴量統計の準備")
    try:
        from training.featurizations import TrinoTrueCardDetail
        from sklearn.preprocessing import RobustScaler
        from training.preprocessing.feature_statistics import FeatureType
        import numpy as np
        
        plan_featurization = TrinoTrueCardDetail()
        
        # 必要な特徴量をすべて含むfeature_statisticsを作成
        feature_statistics = {}
        
        # PLAN_FEATURES
        for feat in plan_featurization.PLAN_FEATURES:
            if feat == 'op_name':
                feature_statistics[feat] = {
                    'type': str(FeatureType.categorical),
                    'value_dict': {'ScanFilterProject': 0, 'FilterProject': 1, 'Output': 2, 'Aggregate': 3},
                    'no_vals': 100
                }
            else:
                feature_statistics[feat] = {
                    'type': str(FeatureType.numeric),
                    'center': 1000.0,
                    'scale': 500.0,
                }
        
        # FILTER_FEATURES
        for feat in plan_featurization.FILTER_FEATURES:
            if feat == 'operator':
                feature_statistics[feat] = {
                    'type': str(FeatureType.categorical),
                    'value_dict': {'=': 0, '$eq': 1, '$gt': 2, '$lt': 3},
                    'no_vals': 50
                }
            else:
                feature_statistics[feat] = {
                    'type': str(FeatureType.numeric),
                    'center': 0.0,
                    'scale': 1.0,
                }
        
        # COLUMN_FEATURES
        for feat in plan_featurization.COLUMN_FEATURES:
            feature_statistics[feat] = {
                'type': str(FeatureType.numeric),
                'center': 0.0,
                'scale': 1.0,
            }
        
        # TABLE_FEATURES
        for feat in plan_featurization.TABLE_FEATURES:
            feature_statistics[feat] = {
                'type': str(FeatureType.numeric),
                'center': 10000.0,
                'scale': 5000.0,
            }
        
        # OUTPUT_COLUMN_FEATURES
        for feat in plan_featurization.OUTPUT_COLUMN_FEATURES:
            if feat == 'aggregation':
                feature_statistics[feat] = {
                    'type': str(FeatureType.categorical),
                    'value_dict': {'Aggregator.COUNT': 0, 'Aggregator.SUM': 1, None: 2},
                    'no_vals': 10
                }
            else:
                feature_statistics[feat] = {
                    'type': str(FeatureType.numeric),
                    'center': 0.0,
                    'scale': 1.0,
                }
        
        # RobustScalerを追加
        for k, v in feature_statistics.items():
            if v.get('type') == str(FeatureType.numeric):
                scaler = RobustScaler()
                scaler.center_ = np.array([v['center']])
                scaler.scale_ = np.array([v['scale']])
                feature_statistics[k]['scaler'] = scaler
        
        print(f"✅ feature_statistics準備完了: {len(feature_statistics)} 特徴量")
        
    except Exception as e:
        print(f"❌ 特徴量統計準備エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. trino_plan_collatorのテスト
    print("\n📦 ステップ4: trino_plan_collatorのテスト")
    try:
        from models.zeroshot.trino_plan_batching import trino_plan_collator
        
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
        print(f"   - labels: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
        print(f"   - sample_idxs: {sample_idxs}")
        
        # 特徴量の確認
        for feat_name, feat_tensor in features.items():
            if hasattr(feat_tensor, 'shape'):
                print(f"   - {feat_name}: shape={feat_tensor.shape}")
            else:
                print(f"   - {feat_name}: len={len(feat_tensor)}")
        
    except Exception as e:
        print(f"❌ trino_plan_collatorエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. モデルの初期化とフォワードパステスト
    print("\n🤖 ステップ5: モデルの初期化とフォワードパス")
    try:
        from models.zeroshot.specific_models.trino_zero_shot import TrinoZeroShotModel
        from classes.classes import ZeroShotModelConfig
        
        model_config = ZeroShotModelConfig(
            hidden_dim=64,  # テスト用に小さめ
            hidden_dim_plan=64,
            hidden_dim_pred=64,
            p_dropout=0.1,
            featurization=plan_featurization,
            output_dim=1,
            batch_size=2
        )
        
        encoders = [
            ('plan', plan_featurization.PLAN_FEATURES),
            ('logical_pred', plan_featurization.FILTER_FEATURES),
            ('column', plan_featurization.COLUMN_FEATURES),
            ('table', plan_featurization.TABLE_FEATURES),
            ('filter_column', plan_featurization.FILTER_FEATURES + plan_featurization.COLUMN_FEATURES),
            ('output_column', plan_featurization.OUTPUT_COLUMN_FEATURES)
        ]
        
        # prepassesは、グラフにエッジが存在する場合のみ設定
        # column_to_output_column_edgesが空の場合はprepassesを空にする
        prepasses = []  # テスト用に空にする（col_output_colエッジがない場合があるため）
        tree_model_types = []  # prepassesが空の場合はtree_model_typesも空にする
        
        model = TrinoZeroShotModel(
            model_config=model_config,
            device='cpu',
            feature_statistics=feature_statistics,
            plan_featurization=plan_featurization,
            prepasses=prepasses,
            add_tree_model_types=tree_model_types,
            encoders=encoders
        )
        
        print(f"✅ モデル初期化成功")
        print(f"   - パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
        
        # フォワードパステスト
        model.eval()
        with torch.no_grad():
            predictions = model((graph, features))
            print(f"✅ フォワードパス成功")
            print(f"   - predictions shape: {predictions.shape}")
            print(f"   - predictions sample: {predictions[:3].flatten().tolist()}")
        
    except Exception as e:
        print(f"❌ モデルテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. train_zeroshot.pyの主要機能のテスト
    print("\n🔍 ステップ6: train_zeroshot.pyの主要機能テスト")
    try:
        # train_zeroshot.pyの関数をテスト
        from trino_lcm.scripts.train_zeroshot import (
            load_plans_from_files,
            create_dummy_feature_statistics,
            TrinoPlanDataset
        )
        
        # プランの読み込みテスト
        test_files = [test_file]
        loaded_plans = load_plans_from_files(test_files, max_plans_per_file=3)
        print(f"✅ load_plans_from_files成功: {len(loaded_plans)} プラン")
        
        # 特徴量統計の作成テスト
        dummy_stats = create_dummy_feature_statistics(plan_featurization)
        print(f"✅ create_dummy_feature_statistics成功: {len(dummy_stats)} 特徴量")
        
        # データセットの作成テスト
        dataset = TrinoPlanDataset(plans[:3])
        print(f"✅ TrinoPlanDataset作成成功: {len(dataset)} サンプル")
        
    except Exception as e:
        print(f"❌ train_zeroshot.py機能テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ エンドツーエンドテスト成功")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_end_to_end()
    if not success:
        sys.exit(1)

