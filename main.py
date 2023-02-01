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

    if method == 'Finetuning':
        return st.Finetuning(**kwargs)
    elif method == 'Ema':
        return st.Ema(**kwargs)
    elif method == 'LessForg':
        return st.LessForg(**kwargs)
    elif method == 'SurgicalFT':
        return st.SurgicalFT(**kwargs)
    elif method == 'FinetuningFC':
        return st.FinetuningFC(**kwargs)
    elif method == 'OJKD':
        return st.OJKD(**kwargs)
    else:
        raise NotImplementedError(f"Unknown method {method}")



def main(n_run, seed):
    utils.set_seeds(seed)

    if OPT.TENSORBOARD:
        writer = SummaryWriter()
    else:
        writer = None

    print("###########################################")
    print("########### DATASET PREPARATION ###########")
    train_data = dset.get_dset_data(OPT.DATASET, OPT.DATA_FOLDER, train=True)
    (fh_train_loader, fh_val_loader), (sh_train_loader, sh_val_loader), tasks = dset.prepare_tasks(train_data, OPT.NUM_TASKS, OPT.BATCH_SIZE)

    test_data = dset.get_dset_data(OPT.DATASET, OPT.DATA_FOLDER, train=False)
    _, small_test_loader = dset.split_train_val(test_data, OPT.BATCH_SIZE)

    loss_fn = nn.CrossEntropyLoss()


    if OPT.ALL:
        print("###########################################")
        print("########### TRAIN ON ALL DATASET ##########")
        model_all = utils.get_model(OPT.MODEL, OPT.NUM_CLASSES, OPT.PRETRAINED)
        optimizer = optim.Adam(model_all.parameters(), lr=OPT.LR_ALL, weight_decay=OPT.WD_ALL)
        train_loader, val_loader = dset.split_train_val(train_data, OPT.BATCH_SIZE)
        fh = Trainer(model_all, OPT.DEVICE, OPT.NUM_CLASSES, writer, tag=f'{OPT.DATASET}_{OPT.MODEL}_all')
        fh.train_eval(optimizer, loss_fn, OPT.EPOCHS_ALL, train_loader, val_loader)

    if OPT.LOAD_FISRT_SECOND_HALF_MODELS:
        print("###########################################")
        print("######### MODELS ALREADY TRAINED ##########")
        # [TODO] load models with dynamic epoch number (not hardcoded)
        at_epoch=0
        model_fh, model_sh = utils.load_models(OPT.MODEL, OPT.DATASET, OPT.NUM_CLASSES, at_epoch)
        fh = Trainer(model_fh, OPT.DEVICE, OPT.NUM_CLASSES, writer, f'{OPT.DATASET}_{OPT.MODEL}_fh')
        sh = Trainer(model_sh, OPT.DEVICE, OPT.NUM_CLASSES, writer, f'{OPT.DATASET}_{OPT.MODEL}_sh')


    else:
        print("###########################################")
        print("############# FIRST / SECOND ##############")
        print("FIRST HALF")
        model_fh = utils.get_model(OPT.MODEL, OPT.NUM_CLASSES, OPT.PRETRAINED)
        optimizer = optim.AdamW(model_fh.parameters(), lr=OPT.LR_FH, weight_decay=OPT.WD_FH)
        fh = Trainer(model_fh, OPT.DEVICE, OPT.NUM_CLASSES, writer, tag=f'{OPT.DATASET}_{OPT.MODEL}_fh')
        #fh.train_eval(optimizer, loss_fn, OPT.EPOCHS_FH, fh_train_loader, fh_val_loader)

        print("SECOND HALF")
        model_sh = copy.deepcopy(model_fh)
        optimizer = optim.AdamW(model_sh.parameters(), lr=OPT.LR_SH, weight_decay=OPT.WD_SH)
        sh = Trainer(model_sh, OPT.DEVICE, OPT.NUM_CLASSES, writer, tag=f'{OPT.DATASET}_{OPT.MODEL}_sh')
        #sh.train_eval(optimizer, loss_fn, OPT.EPOCHS_SH, sh_train_loader, sh_val_loader)

    print("###########################################")
    print("############## CONTINUAL STEP #############")

    # Copy the model to avoid overwriting
    model_c = copy.deepcopy(model_fh)

    # Create the approach and train
    continual_metrics = []
    OPT.ARGS_CONT['model'] = model_c
    strategy = MethodFactory(OPT.METHOD_CONT, **OPT.ARGS_CONT)
    for task_id, (task_train_loader, task_val_loader) in enumerate(tasks):
       print(f"---Task {task_id}---")
       tag = f't{task_id}'
       strategy.train(task_train_loader, task_val_loader, writer, tag)
       sh_loss, sh_acc = strategy.eval(sh_val_loader, writer, 'sh')
       continual_metrics.append((sh_loss, sh_acc))

    
    # Print accuracies
    
    # print("m total @ val2:")
    # all_l, all_a = utils.inference(model_all, loss_fn, sh_val_loader) 
    
    _, fh_acc = fh.eval(sh_val_loader, loss_fn)
    print(f"model_first_half accuracy @ eval_second_half: {fh_acc:.5f}")
    _, sh_acc = sh.eval(sh_val_loader, loss_fn)
    print(f"model_second_half accuracy @ eval_second_half: {sh_acc:.5f}")

    # Write continual metrics to csv
    append = True if n_run > 0 else False
    data = [seed] + [a for l, a in continual_metrics] + [fh_acc, sh_acc]
    row = ",".join(str(value) for value in data)

    fname = os.path.join(OPT.CSV_FOLDER, f"{OPT.DATASET}_{OPT.NUM_TASKS}tasks_{strategy.name.replace('_','')}_{OPT.MODEL.replace('_','')}.csv")
    utils.write_line_to_csv(row, fname, append)

    if OPT.TENSORBOARD:
        writer.close()


if __name__ == "__main__":
    
    # Set seeds for multiple runs
    for n, seed in enumerate(OPT.SEEDS):
        if n > 0:
            OPT.ALL = False
            OPT.LOAD_FISRT_SECOND_HALF_MODELS = True
        else:
            OPT.LOAD_FISRT_SECOND_HALF_MODELS = False

        main(n, seed)
