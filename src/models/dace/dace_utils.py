import json
import numpy as np
import torch


def load_json(path):
    with open(path) as json_file:
        json_obj = json.load(json_file)
    return json_obj


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
