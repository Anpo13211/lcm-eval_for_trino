import argparse
import multiprocessing
import multiprocessing as mp
import os
import time
from types import SimpleNamespace

from osfclient.cli import clone

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.generate_workload import generate_workload
from cross_db_benchmark.benchmark_tools.load_database import load_database
from cross_db_benchmark.datasets.datasets import source_dataset_list, database_list, ext_database_list
from cross_db_benchmark.meta_tools.scale_dataset import scale_up_dataset
from experiments.evaluation_workloads.generated.workload_defs import generate_workload_defs
from run_benchmark import StoreDictKeyPair


def workload_gen(input):
    source_dataset, workload_path, max_no_joins, workload_args = input
    generate_workload(source_dataset, workload_path, max_no_joins=max_no_joins, **workload_args)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../zero-shot-data/datasets')
    parser.add_argument('--workload_dir', default='../zero-shot-data/workloads')
    parser.add_argument('--osf_username', required=True)
    parser.add_argument('--osf_project', required=True)
    parser.add_argument('--osf_password', required=True)
    parser.add_argument("--database_conn", dest='database_conn_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument("--database_kwargs", dest='database_kwarg_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument('--autoscale', default=5000000, type=int)
    args = parser.parse_args()

    if args.database_conn_dict is None:
        args.database_conn_dict = dict(user='postgres', password='postgres', host='localhost')
    if args.database_kwarg_dict is None:
        args.database_kwarg_dict = dict()

    os.makedirs(args.data_dir, exist_ok=True)

    # check whether files are missing
    download_required = False
    for dataset in source_dataset_list:
        zip_file = os.path.join(args.data_dir, 'osfstorage', f'{dataset.name}.zip')
        if not os.path.exists(zip_file):
            print(f"Zip file {zip_file} does not exist. Download from osf required.")
            download_required = True

    # download from osf
    if download_required:
        print(f"Downloading files")

        # clone project from osf
        osf_args = SimpleNamespace(**dict(username=args.osf_username, project=args.osf_project,
                                          output=args.data_dir, update=True))
        os.environ["OSF_PASSWORD"] = args.osf_password
        clone(osf_args)

    # unzip if required
    for dataset in source_dataset_list:
        zip_file = os.path.join(args.data_dir, 'osfstorage/datasets', f'{dataset.name}.zip')
        if not os.path.exists(os.path.join(args.data_dir, dataset.name)):
            print(f"Unzipping {dataset.name}")
            os.system(f'unzip {zip_file} -d {args.data_dir}')

    # scale if required
    for dataset in database_list:
        if dataset.scale == 1:
            continue

        assert dataset.data_folder != dataset.source_dataset, "For scaling a new folder is required"
        print(f"Scaling dataset {dataset.db_name}")
        curr_source_dir = os.path.join(args.data_dir, dataset.source_dataset)
        curr_target_dir = os.path.join(args.data_dir, dataset.data_folder)
        if not os.path.exists(curr_target_dir):
            scale_up_dataset(dataset.source_dataset, curr_source_dir, curr_target_dir, scale=dataset.scale)
            # scale_up_dataset(dataset.source_dataset, curr_source_dir, curr_target_dir, autoscale_tuples=args.autoscale)
            # scale_up_dataset(dataset.source_dataset, curr_source_dir, curr_target_dir, autoscale_gb=0.5)

    # load databases
    # also load imdb full dataset to be able to run the full job benchmark
    for dataset in ext_database_list:
        # Use string instead of enum (supports plugin registry)
        for database in ['postgres']:  # Changed from [DatabaseSystem.POSTGRES]
            curr_data_dir = os.path.join(args.data_dir, dataset.data_folder)
            print(f"Loading database {dataset.db_name} from {curr_data_dir}")
            load_database(curr_data_dir, dataset.source_dataset, database, dataset.db_name, args.database_conn_dict,
                          args.database_kwarg_dict)

    # make sure required workloads are generated (do this in parallel)
    workload_gen_setups = generate_workload_defs(args.workload_dir)
    start_t = time.perf_counter()
    proc = multiprocessing.cpu_count() - 2
    p = mp.Pool(initargs=('arg',), processes=proc)
    p.map(workload_gen, workload_gen_setups)
    print(f"Generated workloads in {time.perf_counter() - start_t:.2f} secs")
