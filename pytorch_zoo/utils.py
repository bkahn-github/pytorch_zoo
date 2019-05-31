import os
import random

import numpy as np

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_environment(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# from https://github.com/floydhub/save-and-resume
def save(model, fold):
    """Save checkpoint if a new best is achieved"""
    filename = f"./checkpoint-{fold}.pt"
    torch.save(model, filename)


def load(model, fold):
    model.load_state_dict(torch.load(f"./checkpoint-{fold}.pt"))

    return model


def gpu_usage(device=device, digits=4):
    print(
        f"GPU Usage: {round((torch.cuda.memory_allocated(device=device) / 1e9), digits)} GB\n"
    )


def n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
