import argparse

from cross_db_benchmark.meta_tools.dataset_stats import generate_dataset_statistics
from cross_db_benchmark.meta_tools.derive import derive_from_relational_fit
from cross_db_benchmark.meta_tools.replace_aliases import replace_workload_alias

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default=None)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--source', default=None)
    parser.add_argument('--target', default=None)
    parser.add_argument('--relational_fit_dataset_name', default=None)
    parser.add_argument('--replace_workload_alias', action='store_true')
    parser.add_argument('--generate_dataset_statistics', action='store_true')
    parser.add_argument('--derive_from_relational_fit', action='store_true')

    args = parser.parse_args()

    if args.derive_from_relational_fit:
        derive_from_relational_fit(args.relational_fit_dataset_name, args.dataset_name)

    if args.replace_workload_alias:
        replace_workload_alias(args.dataset_name, args.source, args.target)

    if args.generate_dataset_statistics:
        generate_dataset_statistics(args.target, args.data_dir)