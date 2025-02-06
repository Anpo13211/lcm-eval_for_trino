from experiments.setup.utils import strip_commands

def gen_tune_commands(study_name=None,
                      workload_runs=None,
                      statistics_file='../zero-shot-data/runs/parsed_plans/statistics_workload_10k_s0_c8220.json',
                      n_trials=10,
                      n_workers=16,
                      db_host='c05.lab',
                      db_user='postgres',
                      db_password='postgres',
                      cardinalities='actual',
                      database=None,
                      max_epoch_tuples=10000,
                      ):
    workload_runs = ' '.join(workload_runs)
    assert study_name is not None
    exp_commands = [f"""python3 tune.py
                  --workload_runs {workload_runs}
                  --statistics_file {statistics_file}
                  --target ../zero-shot-data/tuning/{study_name}
                  [device_placeholder]
                  --database {str(database)}
                  --num_workers {n_workers}
                  --db_user {db_user}
                  --db_password {db_password}
                  --db_host {db_host}
                  --study_name {study_name}
                  --cardinalities {cardinalities}
                  --max_epoch_tuples {max_epoch_tuples}
                  --n_trials 1
                  --setup distributed
                  --seed {i}
            """ for i in range(n_trials)]

    exp_commands = strip_commands(exp_commands)
    return exp_commands
