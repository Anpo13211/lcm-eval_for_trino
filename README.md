# How Good Are Learned Cost Models, Really? Insights from Query Optimization Tasks. 

This repository contains the evaluation **source code** of the SIGMOD Paper:
"[How Good Are Learned Cost Models, Really? Insights from Query Optimization Tasks](https://dl.acm.org/doi/10.1145/3725309)" from
Roman Heinrich, Manisha Luthra, Johannes Wehrstein, Harald Kornmayer and Carsten Binnig.


## 1. Access to Training and Evaluation Data
<details>
  <summary>Click to Expand</summary>

**The training data, trained models and model predictions are available at [OSF](https://osf.io/rb5tn/).**
</details>

## 2. Source Code Origins
<details>
  <summary>Click to Expand</summary>

We thank all authors of the following repositories for providing their code. 
We modified some parts of the code to fit our evaluation setup, but the core functionality remains the same.

- **FlatVector** Predicting Multiple Metrics for Queries: Better Decisions Enabled by Machine Learning (ICDE 2009)
  -Authors: Archana Ganapathi; Harumi Kuno; Umeshwar Dayal; Janet L. Wiener; Armando Fox; Michael Jordan
  - Paper: https://ieeexplore.ieee.org/document/4812438
  - Code: Not available, but easy to implement (we used the simple flat vector model of this approach) 
    Each query is expressed as a feature vector that contains out of the operator types and their summed, intermediate cardinalities
  - Modifications: We applied a state-of-the-art regression on top (LightGBM) to predict the cost of the query. 
    The code is available at `src/models/tabular/train_tabular_baseline.py`.


- **MSCN**: Learned Cardinalities: Estimating Correlated Joins with Deep Learning (CIDR 2019)
  - Authors: Andreas Kipf, Thomas Kipf, Bernhard Radke, Viktor Leis, Peter Boncz, Alfons Kemper
  - Paper: https://www.cidrdb.org/cidr2019/papers/p101-kipf-cidr19.pdf
  - Code: https://github.com/andreaskipf/learnedcardinalities
  - Modifications: While originally designed for cost estimation, we use MSCN to predict execution costs (i.e. runtime) of queries.
    Most importantly, MSCN only learns from SQL strings and for sample bitmaps.
    Thus, we augment all plans with sample bitmaps, which are in the corresponding `augmented_plan_baseline` files.
    Moreover, as MSCN relies on one-hot-Encoding, it is built dynamically fo a given dataset based on the extracted feature-statistics.


- **QPPNet**: Plan-Structured Deep Neural Network Models for Query Performance Prediction (VLDB 2019)
  - Authors: Ryan Marcus, Olga Papaemmanouil
  - Paper: https://dl.acm.org/doi/abs/10.14778/3342263.3342646
  - Code: https://github.com/rabbit721/QPPNet
  - Modifications: While the original QPPNet-Implementation was tailored towards three datasets (TPC-H), we adapted the code to work with our datasets.
    We dynamically read out feature statistics to directly use the correct model size according to the required encoding.
    In addition, we executed training and evaluation queries in `JSON` mode for this particular model (i.e. `EXPLAIN ANALYZE FORMAT JSON`), 
    as it requires more features that are not available in the standard query plans (i.e. `EXPLAIN ANALYZE`).
    The code is available at `src/models/qppnet/`.
  

- **End-To-End**: End-To-End Learning for Cost Estimation of Query Execution Plans (VLDB 2019)
  - Authors: Ji Sun, Guoliang Li
  - Paper: https://dl.acm.org/doi/abs/10.14778/3368289.3368296
  - Code: https://github.com/greatji/Learning-based-cost-estimator
  - Modifications: Similar to MSCN, this model requires sample bitmaps. We augmented the query plans with sample bitmaps and trained the model.
    We also need to adapt the source code to be applicable for different datasets.
    The main implementation is available at: `src/models/workload_driven/model/mscn_model.py
  

- **QueryFormer**: QueryFormer: A Tree Transformer Model for Query Plan Representation (VLDB 2022)
  - Authors: Yue Zhao, Gao Cong, Jiachen Shi, Chunyan Miao
  - Paper: https://www.vldb.org/pvldb/vol15/p1658-zhao.pdf
  - Code: https://github.com/zhaoyue-ntu/QueryFormer
  - Modifications: Query Former was hard-coded to the IMDB dataset and often used non-parametric variables. T
    Thus, we deciced to extract the PyTorch model and ingested it in our existing training pipeline.
    We also adapted the code to work with our datasets and to be more flexible in terms of the input data.
    The implementation is available at: `src/models/workload_driven/model/e2e_model.py`
  
- **ZeroShot**: Zero-shot Cost Estimation with Deep Learning (VLDB 2022)
  - Authors: Benjamin Hilprecht and Carsten Binnig
  - Paper: https://www.vldb.org/pvldb/vol15/p2361-hilprecht.pdf
  - Code: https://github.com/DataManagementLab/zero-shot-cost-estimation
  - Modifications: Little modifications were applied to this model, as its source code was used and extended as basis for our paper.
    However, various improvements were made to the code to make it more efficient and to adapt it to our datasets.
    Moreover, we implemented different model variants and an optimized training pipeline.
    The implementation is available at: `src/models/zeroshot/zero_shot_model.py` 


- **DACE**: DACE: A Database-Agnostic Cost Estimator (ICDE 2024)
  - Authors: Zibo Liang, Xu Chen, Yuyang Xia, Runfan Ye, Haitian Chen, Jiandong Xie, Kai Zheng 
  - Paper: https://ieeexplore.ieee.org/document/10598079
  - Code: https://github.com/liang-zibo/DACE
  - Modifications: As DACE builds up on the Zero-Shot code base, its adaption was straight forward and little changes were required to make it running.
    Like in the other models, we adapted the code to work with our datasets and to be more flexible in terms of the input data.
    The impelementation is available at: `src/models/dace/dace_model.py`

</details>


## 3. Experimental Setup
<details>
  <summary>Click to Expand</summary>
In the following, we describe our experimental set-up. For details, please refer to the paper.
We disinguish between **local** and **remote** operations in the following.

- **Local**: Models are trained and evaluated on the local host, i.e. the machine where the code is executed.
While useful for debugging and development, this is impractical, as datasets are large and the training of the models is time-consuming.
- **Remote**: Models are trained and evaluated on remote machines. This includes the execution of training and evaluation queries, which we conducat on Cloudlab machines.
CLoudlab is a research platform that provides access to a variety of hardware resources for research purposes.
Moreover, we use a set of training machines that are equipped with GPUs to accelerate the training process.

### Hardware Instances
This code distinguishes between different type of machines.
- **Localhost**: Can be used to train and evaluate models in local mode. In remote mode, it is used as central coordinator to distribute code and evaluate results.
- **Cloudlab-Instances**: The machines of type [c8220](https://www.clemson.cloudlab.us/portal/show-nodetype.php?type=c8220&_gl=1*1goap78*_ga*MTIyNzE5ODc0My4xNzIxOTIwMTMw*_ga_6W2Y02FJX6*MTcyMzE4MzIwNi42LjEuMTcyMzE4MzIyMy4wLjAuMA..) 
  used in the Cloudlab environment to run training and test queries. These are the target instances, i.e. for these machines and database environments, LCMs are trained.
- **Training-Instances**: In order to accelerate the training LCMs, we use a set of cluster nodes that especially have GPU support (CUDA).
Moreover, we integrated and recommend [Weights & Biases](https://wandb.ai/site) to monitor the training process and evaluate the results.

### Environment Variables
To make the scripts running for local and remote operation, the user has to provide the following information by setting the corresponding environment variables:
- `LOCAL_ROOT_PATH`: The path to the root directory of the repository on the local host.
- `LOCAL_KNOWN_HOSTS_PATH`: The path to the known_hosts file on the local host.
- `CLOUDLAB_ROOT_PATH`: The path to the root directory of the repository on the cloudlab machines.
- `CLOUDLAB_SSH_USERNAME`: The username to log in to the cloudlab machines.
- `CLOUDLAB_SSH_KEY_PATH`: The path to the ssh key to log in to the cloudlab machines.
- `CLOUDLAB_SSH_PASSPHRASE`: The passphrase of the ssh key to log in to the cloudlab machines.
- `CLUSTER_ROOT_PATH`: The path to the root directory of the repository on the training machines.
- `CLUSTER_STORAGE_PATH`: The path to the storage directory on the training machines.
- `CLUSTER_SSH_USERNAME`: The username to login to the training machines.
- `CLUSTER_SSH_KEY_PATH`: The path to the ssh key to login to the training machines.
- `CLUSTER_SSH_PASSPHRASE`: The passphrase of the ssh key to login to the training machines.
- `WANDB_USER`: The username of the Weights & Biases account.
- `WANDB_PROJECT`: The project name of the Weights & Biases account.
- `OSF_USERNAME`: The username of the OSF account where the data is stored.
- `OSF_PASSWORD`: The password of the OSF account where the data is stored.
- `OSF_PROJECT`: The project name of the OSF account where the data is stored.
- `NODE00`: Node Information of the first training machine in the form of: `{'hostname': 'hostname.com', 'python': '3.9'}`.
- Add more nodes if necessary.
</details>


## 4. Local Training & Evaluation Pipeline
<details>
  <summary>Click to Expand</summary>

To locally train and evaluate the LCMs, we provide a set of scripts that can be executed on the local host.
Those scripts are similarly executed in the remote mode on the target machines.

```
python main.py 
--mode train 
--wandb_project lcm                   
--wandb_entity learneddb 
--model_type zeroshot
--device cuda:0 
--model_dir /path/where/to/save/model 
--target_dir /path/where/to/save/predictions
--statistics_file /path/to/feature_statistics.json
--seed 3
 --wandb_name my_model_training_run
 --workload_runs /path/to/training/run/imdb_scaled1/workload_200k_s1.json /path/ /path/to/training/run/baseball_scaled1/workload_200k_s1.json  
 --test_workload_runs /path/to/test/run/imdb_scaled1/join_order_full/job_light_36.json
 ```

This command trains the LCMs on the local host with the specified parameters.
It uses the following parameters:
- `mode`: The mode of the script, i.e. `train` , `retrain` or `predict`.
- `wandb_project`: The name of the Weights & Biases project to log the training process.
- `wandb_entity`: The name of the Weights & Biases entity to log the training process.
- `model_type`: The type of the model to train, i.e. `zeroshot`, `qppnet`, `mscn`, `e2e`, `queryformer`, `dace`.
- `device`: The device to use for training, i.e. `cuda:0` for GPU or `cpu` for CPU.
- `model_dir`: The directory where the trained model is saved.
- `target_dir`: The directory where the predictions are saved.
- `statistics_file`: The path to the feature statistics file that is used to train the model.
- `seed`: The random seed to use for training.
- `wandb_name`: The name of the Weights & Biases run to log the training process.
- `workload_runs`: The path to the training workload runs that are used to train the model.
- `test_workload_runs`: The path to the test workload runs that are used to evaluate the model.

## Additional Scripts
In addition to the main training and evaluation script, we provide a set of additional scripts that can be used to perform specific tasks:
- `gather_feature_statistics.py`: This script gathers feature statistics from the training workload runs and stores them in a JSON file.
  It is used to prepare the data for training the LCMs.
- `parse_all.py`: This script parses the query plans from the training workload runs and stores them in a specific format.
  It is used to prepare the data for training the LCMs.
- `run_benchmark.py`: This script is used to run queries that are used for training and evaluatio of the LCMs.
  It executes the queries on the database and stores the results in a specific format.
</details>

## 5. Remote Training & Evaluation Pipeline
<details>
  <summary>Click to Expand</summary>

The repository contains a set of scripts to automate the evaluation process.
They are located at `src/scripts/exp_runner` and realize the following tasks:
- **Setup**: Install all necessary dependencies and download the required data on a set of given cloudlab machines.
- **Run Training Workload**: Execute the training workloads on the target machines.
- **Run Evaluation Workload**: Execute selected evaluation workloads on the target machines.
- **Train Models**: Train the LCMs on the training machines with the training data.
- **Predict Models**: Predict the cost of selected evaluation workloads with the trained LCMs.
- **Remove Data**: Remove selected workloads data from localhost and training machines.

### 1. Setup Target Machines
Make sure that the target machines (i.e. Cloudlab Instances) can be reached over SSH.
Also make sure to put an `.env` file in the root directory of the repository with the environment variables.
The following command installs all necessary dependencies and downloads the required data from a OSF repository.
Moreover, it scales the data (if tables are too small), installs PostgreSQL (with hinting extension) and ingests 
the data to postgres. Moreover, training queries (SQL-Strings) are generated. 
Note that this script needs to be executed **twice**, as the hardware requires a reboot.
The node names (e.g. `clnode032.clemson.cloudlab.us`) are read out from a file.
This file and the SSH-configuration can be extracted from Cloudlabs `rspec` file with `parse_cloudlab_manifest.py`
To set-up the target machines, please run:

```
python3 exp_setup.py --task setup deliver start monitor
```

### 2. Run Training Workload
The following command executes the training workloads on the target machines.
```
python3 exp_run_training_workload.py --task setup deliver start monitor
```
Note the different data formats of the query plans that are generated:
- `raw`: This is the output of Postgres `EXPLAIN ANALYZE`command. Moreover, database and column statistics are stored.
- `parsed_plan`: This is a parsed version of the query plan that is fed into the LCMs.
- `parsed_plan_baseline`: These query plans are also parsed and additionally contain sample bitmaps which are required by some LCM.
- `json`: As identified later, Postgres also directly provides the query plan in JSON format.
This format contains more features than the standard plans and is required particularly by `QPPNet`.

### 3. Pick-Up Training Workload
The following command picks up the training workloads from the target machines and stores them on the local host.
```
python3 exp_run_training_workload.py --task pickup
```


### 4. Run Evaluation Workload
Similar as before, the following command executes the evaluation workloads on the target machines.
```
python3 exp_run_evaluation_workloads.py --task setup deliver start monitor
```
They can be fetched with:
```
python3 exp_run_training_workload.py --task pickup
```

### 5. Extract Feature Statistics
In order to train models, feature statistics are required to describe the data distribution and 
range of features. They are derived from the training workloads.
```
python3 gather_feature_statistics.py 
--database imdb 
--target ./data/runs/json/tpc_h_pk/feature_statistics.json 
--workload workload_100k_s1.json 
```
Note that QPPNet requires a different format of the feature statistics

### 5. Train Models
The following command trains the LCMs on the training machines with the training data.
Note that you require feature statistics and training data (parsed query plans) to train the models.
```
python3 exp_train_model.py --task setup deliver start monitor
```
In the script you can select the corresponding LCMs and databases that should be trained.
See the paper for more details on the training data selection for the LCMs.
The training script automatically evaluates the LCM against a test set as written in the paper and stores
its results under `/data/evaluation`
The models are stored (remotely) under `/data/models`.

### 6. Predict Workloads
To predict unseen workloads, please call:
```
python3 exp_predict_all.py --task setup deliver start monitor
```
Make sure to configure the desired evaluation workload and target database in the script.
The evaluation workloads are stored under `/data/evaluation`.

### 7. Fetch Predictions
The following command fetches the predictions from the target machines and stores them on the localhost under `/data/evaluation`.
```
python3 exp_predict_all.py --task pickup
```
</details>

## 6. Plots
<details>
  <summary>Click to Expand</summary>
The evaluation strategy is described in the paper and the corresponding jupyter notebooks can be found at:
`/src/evaluation/plots`. The notebooks are used to evaluate the results of the LCMs and compare them to the ground truth.
</details>

## 7. References
<details>
  <summary>Click to Expand</summary>
You can find the paper in the ACM Digital Library or on arXiv:
- **Paper (ACM DL): https://dl.acm.org/doi/10.1145/3725309**
- **Paper (arXiv): https://arxiv.org/abs/2502.01229**

Please cite the paper as follows:

```bibtex
  @article{10.1145/3725309,
    author = {Heinrich, Roman and Luthra, Manisha and Wehrstein, Johannes and Kornmayer, Harald and Binnig, Carsten},
    title = {How Good are Learned Cost Models, Really? Insights from Query Optimization Tasks},
    year = {2025},
    issue_date = {June 2025},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {3},
    number = {3},
    url = {https://doi.org/10.1145/3725309},
    doi = {10.1145/3725309},
    journal = {Proc. ACM Manag. Data},
    month = jun,
    articleno = {172},
    numpages = {27},
}
```
</details>