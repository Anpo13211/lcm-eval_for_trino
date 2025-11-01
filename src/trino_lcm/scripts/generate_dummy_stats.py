#!/usr/bin/env python3
"""
ダミー統計情報を生成するスクリプト
Trinoデータベースにアクセスできない場合の開発・テスト用

Usage:
    # 単一データセット
    python src/trino_lcm/scripts/generate_dummy_stats.py \
        --template iceberg_imdb \
        --output iceberg_accidents \
        --tables accidents,severity,conditions
    
    # すべてのデータセット
    python src/trino_lcm/scripts/generate_dummy_stats.py \
        --template iceberg_imdb \
        --generate-all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


DATASETS = [
    'accidents', 'airline', 'baseball', 'basketball',
    'carcinogenesis', 'consumer', 'credit', 'employee',
    'fhnk', 'financial', 'geneea', 'genome', 'hepatitis',
    'imdb', 'movielens', 'seznam', 'ssb', 'tournament',
    'tpc_h', 'walmart'
]


def load_template(template_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """
    テンプレート統計情報を読み込む
    
    Returns:
        tuple: (table_stats, column_stats, metadata)
    """
    table_stats_file = template_dir / 'table_stats.json'
    column_stats_file = template_dir / 'column_stats.json'
    metadata_file = template_dir / 'metadata.json'
    
    if not table_stats_file.exists():
        raise FileNotFoundError(f"テンプレートが見つかりません: {table_stats_file}")
    
    with open(table_stats_file, 'r', encoding='utf-8') as f:
        table_stats = json.load(f)
    
    with open(column_stats_file, 'r', encoding='utf-8') as f:
        column_stats = json.load(f)
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return table_stats, column_stats, metadata


def generate_dummy_table_stats(table_names: List[str], template: Dict) -> Dict:
    """
    ダミーのテーブル統計を生成
    
    Args:
        table_names: テーブル名のリスト
        template: テンプレート統計情報
    
    Returns:
        Dict: ダミーテーブル統計
    """
    dummy_stats = {}
    
    # テンプレートから1つのテーブル統計を取得
    if template:
        template_table = next(iter(template.values()))
    else:
        # デフォルトテンプレート
        template_table = {
            'reltuples': 10000.0,
            'relpages': 100,
            'table_name': 'template',
            'row_count': 10000.0,
            'num_columns': 5
        }
    
    for table_name in table_names:
        dummy_stats[table_name] = {
            'reltuples': template_table.get('reltuples', 10000.0),
            'relpages': template_table.get('relpages', 100),
            'table_name': table_name,
            'row_count': template_table.get('row_count', 10000.0),
            'num_columns': template_table.get('num_columns', 5)
        }
    
    return dummy_stats


def generate_dummy_column_stats(
    table_names: List[str],
    columns_per_table: int,
    template: Dict
) -> Dict:
    """
    ダミーのカラム統計を生成
    
    Args:
        table_names: テーブル名のリスト
        columns_per_table: テーブルあたりのカラム数
        template: テンプレート統計情報
    
    Returns:
        Dict: ダミーカラム統計
    """
    dummy_stats = {}
    
    # テンプレートから1つのカラム統計を取得
    if template:
        template_column = next(iter(template.values()))
    else:
        # デフォルトテンプレート
        template_column = {
            'n_distinct': 100,
            'null_frac': 0.0,
            'avg_width': 10,
            'correlation': 0.0,
            'data_type': 'integer'
        }
    
    for table_name in table_names:
        for col_idx in range(columns_per_table):
            col_name = f"col_{col_idx}"
            key = f"('{table_name}', '{col_name}')"
            
            dummy_stats[key] = {
                'n_distinct': template_column.get('n_distinct', 100),
                'null_frac': template_column.get('null_frac', 0.0),
                'avg_width': template_column.get('avg_width', 10),
                'correlation': template_column.get('correlation', 0.0),
                'data_type': template_column.get('data_type', 'integer'),
                'table': table_name,
                'column': col_name
            }
    
    return dummy_stats


def generate_from_template(
    template_dir: Path,
    output_dir: Path,
    dataset_name: str,
    table_names: List[str] = None
):
    """
    テンプレートからダミー統計情報を生成
    
    Args:
        template_dir: テンプレートディレクトリ
        output_dir: 出力ディレクトリ
        dataset_name: データセット名
        table_names: テーブル名リスト（省略時はテンプレートと同じ構造）
    """
    print(f"📋 テンプレート読み込み中: {template_dir}")
    
    try:
        template_table_stats, template_column_stats, template_metadata = load_template(template_dir)
    except Exception as e:
        print(f"❌ テンプレート読み込みエラー: {e}", file=sys.stderr)
        return False
    
    # テーブル名が指定されていない場合、テンプレートと同じ数のダミーテーブルを生成
    if not table_names:
        num_tables = len(template_table_stats)
        table_names = [f"table_{i}" for i in range(num_tables)]
    
    print(f"📊 ダミー統計情報を生成中...")
    print(f"   テーブル数: {len(table_names)}")
    print(f"   テーブル名: {', '.join(table_names)}")
    
    # ダミー統計を生成
    dummy_table_stats = generate_dummy_table_stats(table_names, template_table_stats)
    
    # カラム数をテンプレートから推定
    avg_columns = (
        sum(t.get('num_columns', 5) for t in template_table_stats.values()) 
        / len(template_table_stats)
        if template_table_stats else 5
    )
    
    dummy_column_stats = generate_dummy_column_stats(
        table_names,
        int(avg_columns),
        template_column_stats
    )
    
    # メタデータを更新
    dummy_metadata = {
        'catalog': 'iceberg',
        'schema': dataset_name,
        'num_tables': len(dummy_table_stats),
        'num_columns': len(dummy_column_stats),
        'is_dummy': True,
        'template': str(template_dir)
    }
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルに保存
    table_stats_file = output_dir / 'table_stats.json'
    column_stats_file = output_dir / 'column_stats.json'
    metadata_file = output_dir / 'metadata.json'
    
    with open(table_stats_file, 'w', encoding='utf-8') as f:
        json.dump(dummy_table_stats, f, indent=2, ensure_ascii=False)
    
    with open(column_stats_file, 'w', encoding='utf-8') as f:
        json.dump(dummy_column_stats, f, indent=2, ensure_ascii=False)
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(dummy_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ ダミー統計情報を保存しました:")
    print(f"   📄 {table_stats_file}")
    print(f"   📄 {column_stats_file}")
    print(f"   📄 {metadata_file}")
    print()
    print(f"⚠️  注意: これはダミーデータです")
    print(f"   実際の統計情報とは異なるため、モデルの精度は保証されません")
    print(f"   開発・テスト目的でのみ使用してください")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='ダミー統計情報を生成'
    )
    parser.add_argument(
        '--template',
        default='iceberg_imdb',
        help='テンプレートディレクトリ名（デフォルト: iceberg_imdb）'
    )
    parser.add_argument(
        '--output',
        help='出力ディレクトリ名（例: iceberg_accidents）'
    )
    parser.add_argument(
        '--tables',
        help='テーブル名のカンマ区切りリスト（省略時は自動生成）'
    )
    parser.add_argument(
        '--generate-all',
        action='store_true',
        help='すべてのデータセットのダミー統計を生成'
    )
    parser.add_argument(
        '--base-dir',
        default='datasets_statistics',
        help='統計情報ベースディレクトリ（デフォルト: datasets_statistics）'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    template_dir = base_dir / args.template
    
    if not template_dir.exists():
        print(f"❌ テンプレートディレクトリが存在しません: {template_dir}", file=sys.stderr)
        sys.exit(1)
    
    # テーブル名リスト
    table_names = [t.strip() for t in args.tables.split(',')] if args.tables else None
    
    if args.generate_all:
        # すべてのデータセットに対してダミー統計を生成
        print(f"🚀 すべてのデータセットのダミー統計を生成中...")
        print(f"   テンプレート: {template_dir}")
        print(f"   データセット数: {len(DATASETS)}")
        print()
        
        success_count = 0
        for dataset in DATASETS:
            output_dir = base_dir / f"iceberg_{dataset}"
            
            # 既に存在する場合はスキップ
            if output_dir.exists() and (output_dir / 'metadata.json').exists():
                print(f"⏭️  {dataset}: 既に存在します、スキップ")
                continue
            
            print(f"📊 {dataset} のダミー統計を生成中...")
            
            if generate_from_template(
                template_dir=template_dir,
                output_dir=output_dir,
                dataset_name=dataset,
                table_names=table_names
            ):
                success_count += 1
        
        print(f"\n✅ {success_count}/{len(DATASETS)} データセットのダミー統計を生成しました")
        
    elif args.output:
        # 単一データセットのダミー統計を生成
        output_dir = base_dir / args.output
        dataset_name = args.output.replace('iceberg_', '')
        
        generate_from_template(
            template_dir=template_dir,
            output_dir=output_dir,
            dataset_name=dataset_name,
            table_names=table_names
        )
    
    else:
        print("❌ --output または --generate-all を指定してください", file=sys.stderr)
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()



