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
import CDD.main_single as main_single
from opt import OPT

def main():
    args = OPT.opts

    utils.set_seeds(args.seed)
    print("###########################################")
    print("########### DATASET PREPARATION ###########")

    if OPT.DATASET == 'Core50':
        # Core50 does not support warmup
        tasks, validation, subsets = dset.gen_core50_tasks()
    else:
        train_data = dset.get_dset_data(OPT.DATASET, train=True)
        test_data = dset.get_dset_data(OPT.DATASET, train=False)
        #_, small_test_loader = dset.split_train_val(test_data, OPT.BATCH_SIZE)

        if OPT.DO_WARMUP:
            (fh_train_loader, fh_val_loader), (sh_train_loader, sh_val_loader), tasks, subsets = dset.get_tasks(train_data, OPT.NUM_TASKS, OPT.BATCH_SIZE)
        else:
            tasks, subsets = dset.get_tasks(train_data, OPT.NUM_TASKS, OPT.BATCH_SIZE)


    for task_id, (task_train_sbs, task_val_sbs) in enumerate(subsets[0:]):
        t = task_id + 0
        print(f"---Beginning {t}---")
        args.cdd_name_folder = f"_{t}"
        pretrain.main(args, task_train_sbs)
        main_single.main(args, task_train_sbs, task_val_sbs)
        print(f"---Ending {t}---")




if __name__ == "__main__":
   main()
