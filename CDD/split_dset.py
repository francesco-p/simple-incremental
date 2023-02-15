from cProfile import label
import matplotlib.pyplot as plt
import torch
import torchvision
import seaborn as sns
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch import optim
import torch.nn as nn
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

import utils
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
import random
from opt import OPT
import copy
import matplotlib.pyplot as plt
import strategies as st
from models.resnet32 import resnet32
import dset
from trainer import Trainer
import save_features
import pretrain
import argparse
import CDD.main_single_temp as main_single_temp
from args import make_args

def main(n_run, seed):

    print("###########################################")
    print("########### DATASET PREPARATION ###########")
    train_data = dset.get_dset_data(OPT.DATASET, OPT.DATA_FOLDER, train=True)
    (fh_train_loader, fh_val_loader), (sh_train_loader, sh_val_loader), tasks, subsets = dset.prepare_tasks(train_data, OPT.NUM_TASKS, OPT.BATCH_SIZE)

    test_data = dset.get_dset_data(OPT.DATASET, OPT.DATA_FOLDER, train=False)
    _, small_test_loader = dset.split_train_val(test_data, OPT.BATCH_SIZE)

    #loss_fn = nn.CrossEntropyLoss()
    #args = make_args("0000")
    #save_features.main(args, fh_train_loader.dataset)
    #pretrain.main(args, fh_train_loader.dataset)
    #main_single_temp.main(args, fh_train_loader.dataset, fh_val_loader.dataset)
    for task_id, (task_train_sbs, task_val_sbs) in enumerate(subsets[0:]):
        t = task_id + 0
        print(f"---Beginning {t}---")
        args = make_args(t)
        #save_features.main(args, task_train_sbs)
        pretrain.main(args, task_train_sbs)
        main_single_temp.main(args, task_train_sbs, task_val_sbs)
        print(f"---Ending {t}---")
        



if __name__ == "__main__":
    
    # Set seeds for multiple runs
    for n, seed in enumerate(OPT.SEEDS):
        main(n, seed)
