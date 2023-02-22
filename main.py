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
import timm
import utils
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from opt import OPT
import copy
import matplotlib.pyplot as plt
import strategies as st
from models.resnet32 import resnet32
import dset
from trainer import Trainer
import os


def MethodFactory(method, **kwargs):
    """Factory method for creating a strategy object."""

    if method == 'finetuning':
        return st.Finetuning(**kwargs)
    elif method == 'ema':
        return st.Ema(**kwargs)
    elif method == 'lessforg':
        return st.LessForg(**kwargs)
    elif method == 'surgicalft':
        return st.SurgicalFT(**kwargs)
    elif method == 'finetuningfc':
        return st.FinetuningFC(**kwargs)
    elif method == 'ojkd':
        return st.OJKD(**kwargs)
    elif method == 'CDD':
        return st.CDD(**kwargs)
    elif method == 'icarl':
        return st.iCaRL(**kwargs)
    elif method == 'boundary':
        return st.Boundary(**kwargs)
    elif method == 'replay':
        return st.Replay(**kwargs)
    else:
        raise NotImplementedError(f"Unknown method {method}")



def main(seed):
    utils.set_seeds(seed)

    if OPT.TENSORBOARD:
        writer = SummaryWriter()
    else:
        writer = None

    print("###########################################")
    print("########### DATASET PREPARATION ###########")

    if OPT.DATASET == 'Core50':
        # Core50 does not support warmup
        tasks, validation = dset.gen_core50_tasks()
    else:
        train_data = dset.get_dset_data(OPT.DATASET, train=True)
        test_data = dset.get_dset_data(OPT.DATASET, train=False)
        #_, small_test_loader = dset.split_train_val(test_data, OPT.BATCH_SIZE)
        
        if OPT.DO_WARMUP:
            (fh_train_loader, fh_val_loader), (sh_train_loader, sh_val_loader), tasks, subsets = dset.get_tasks(train_data, OPT.NUM_TASKS, OPT.BATCH_SIZE)
        else:
            tasks, subsets = dset.get_tasks(train_data, OPT.NUM_TASKS, OPT.BATCH_SIZE)


    if OPT.DO_WARMUP:
        if OPT.LOAD_FISRT_SECOND_HALF_MODELS:
            print("###########################################")
            print("######### MODELS ALREADY TRAINED ##########")
            # [TODO] load models with dynamic epoch number (not hardcoded)
            at_epoch=9999
            at_seed = 0
            model_fh = utils.load_model(OPT.MODEL, OPT.DATASET, OPT.NUM_CLASSES, at_epoch, at_seed, 'fh')

            print(' this is an evil hack and you syould not do it check it please'*100)
            # we load fh model even for sh because we compare with CDD in the server
            model_sh = utils.load_model(OPT.MODEL, OPT.DATASET, OPT.NUM_CLASSES, at_epoch, at_seed, 'fh')

            fh = Trainer(model_fh, OPT.DEVICE, OPT.NUM_CLASSES, writer, f'{OPT.DATASET}_{OPT.MODEL}_fh')
            sh = Trainer(model_sh, OPT.DEVICE, OPT.NUM_CLASSES, writer, f'{OPT.DATASET}_{OPT.MODEL}_sh')


        else:
            print("###########################################")
            print("############# FIRST / SECOND ##############")
            print("FIRST HALF")
            loss_fn = nn.CrossEntropyLoss()
            model_fh = utils.get_model(OPT.MODEL, OPT.NUM_CLASSES, OPT.PRETRAINED)
            optimizer = optim.AdamW(model_fh.parameters(), lr=OPT.LR_FH, weight_decay=OPT.WD_FH)
            fh = Trainer(model_fh, OPT.DEVICE, OPT.NUM_CLASSES, writer, tag=f'{OPT.DATASET}_{OPT.MODEL}_fh')
            fh.train_eval(optimizer, loss_fn, OPT.EPOCHS_FH, fh_train_loader, fh_val_loader)

            print("SECOND HALF")
            model_sh = copy.deepcopy(model_fh)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model_sh.parameters(), lr=OPT.LR_SH, weight_decay=OPT.WD_SH)
            sh = Trainer(model_sh, OPT.DEVICE, OPT.NUM_CLASSES, writer, tag=f'{OPT.DATASET}_{OPT.MODEL}_sh')
            sh.train_eval(optimizer, loss_fn, OPT.EPOCHS_SH, sh_train_loader, sh_val_loader)

    print("###########################################")
    print("############## CONTINUAL STEP #############")

    # Copy the model to avoid overwriting
    if OPT.DO_WARMUP:
        model_c = copy.deepcopy(model_fh)
    else:
        model_c = utils.get_model(OPT.MODEL, OPT.NUM_CLASSES, OPT.PRETRAINED)

    # Create the approach and train
    continual_metrics = []
    OPT.ARGS_CONT['model'] = model_c
    strategy = MethodFactory(OPT.METHOD_CONT, **OPT.ARGS_CONT)
    print(f"Continual learning with {OPT.METHOD_CONT} strategy")
    for task_id, (task_train_loader, task_val_loader) in enumerate(tasks):
        print(f"---Task {task_id}---")
        tag = f'{task_id}'
        strategy.train(task_train_loader, task_val_loader, writer, tag)
        if OPT.DO_WARMUP:
            loss, acc = strategy.eval(sh_val_loader, writer, 'sh')
        else:
            if OPT.DATASET == 'Core50':
                loss, acc = strategy.eval(validation, writer, 'sh')
            else:
                loss, acc = strategy.eval(sh_val_loader, writer, 'sh')

        continual_metrics.append((loss, acc))

    
    print("###########################################")
    print("############## WRITE TO CSV #############")
    if OPT.DO_WARMUP:
        loss_fn = nn.CrossEntropyLoss()
        _, fh_acc = fh.eval(sh_val_loader, loss_fn)
        print(f"model_first_half accuracy @ eval_second_half: {fh_acc:.5f}")
        _, sh_acc = sh.eval(sh_val_loader, loss_fn)
        print(f"model_second_half accuracy @ eval_second_half: {sh_acc:.5f}")

        data = [seed] + [a for l, a in continual_metrics] + [fh_acc, sh_acc]
    else:
        data = [seed] + [a for l, a in continual_metrics] + [0, 0]
    
    row = ",".join(str(value) for value in data)
    utils.write_line_to_csv(row, strategy.get_csv_name(), OPT.APPEND)

    if OPT.TENSORBOARD:
        writer.close()


if __name__ == "__main__":
    
    main(OPT.SEED)
