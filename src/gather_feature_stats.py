import argparse
import os
import glob

from training.preprocessing.feature_statistics import gather_feature_statistics


def get_all_json_files(directory):
    # Use os.path.join to make the path OS-independent
    if os.path.isfile(directory) and directory.endswith('.json'):
        json_files = [directory]
    else:
        path = os.path.join(directory, '**', '*.json')
        # Use glob.glob with the recursive flag set to True to find all .json files in the directory and its subdirectories
        json_files = glob.glob(path, recursive=True)
        json_files = [file for file in json_files if "feature_statistics" not in file]
    print(json_files)
    return json_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default=None)
    parser.add_argument('--workload', default=None, nargs="*")
    parser.add_argument('--target', default=None)
    args = parser.parse_args()
    json_files = []
    for workload in args.workload:
        json_files += get_all_json_files(f'../data/runs/json/{args.database}/{workload}')
    print(len(json_files))
    gather_feature_statistics(json_files, args.target)
