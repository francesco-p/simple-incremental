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
import seaborn


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
    elif method == 'Soup':
        return st.Soup(**kwargs)
    elif method == 'ICARL':
        return st.ICARL(**kwargs)
    elif method == 'boundary':
        return st.Boundary(**kwargs)
    elif method == 'replay':
        return st.Replay(**kwargs)
    else:
        raise NotImplementedError(f"Unknown method {method}")



def main(seed):

    utils.set_seeds(seed)

    if OPT.tboard:
        writer = SummaryWriter()
    else:
        writer = None

    print("###########################################")
    print("########### DATASET PREPARATION ###########")

    if OPT.dataset == 'Core50':
        if OPT.split_core:
            tasks, validation, _ = dset.gen_core50_tasks()
        else:
            tasks, _ = dset.gen_all_core50_tasks()

    else:
        train_data = dset.get_dset_data(OPT.dataset, train=True)
        test_data = dset.get_dset_data(OPT.dataset, train=False)
        #_, small_test_loader = dset.split_train_val(test_data, OPT.batch_size)
        
        if OPT.do_warmup:
            (fh_train_loader, fh_val_loader), (sh_train_loader, sh_val_loader), tasks, subsets = dset.get_tasks(train_data, OPT.num_tasks, OPT.batch_size)
        else:
            tasks, subsets = dset.get_tasks(train_data, OPT.num_tasks, OPT.batch_size)



    if OPT.do_warmup:
        if OPT.LOAD_FISRT_SECOND_HALF_MODELS:
            print("###########################################")
            print("######### MODELS ALREADY TRAINED ##########")
            # [TODO] load models with dynamic epoch number (not hardcoded)
            at_epoch=9999
            at_seed = 0
            model_fh = utils.load_model(OPT.model, OPT.dataset, OPT.num_classes, at_epoch, at_seed, 'fh')

            print(' this is an evil hack and you syould not do it check it please'*100)
            # we load fh model even for sh because we compare with CDD in the server
            model_sh = utils.load_model(OPT.model, OPT.dataset, OPT.num_classes, at_epoch, at_seed, 'fh')


            fh = Trainer(model_fh, OPT.device, OPT.num_classes, writer, f'{OPT.dataset}_{OPT.model}_fh')
            sh = Trainer(model_sh, OPT.device, OPT.num_classes, writer, f'{OPT.dataset}_{OPT.model}_sh')


        else:
            print("###########################################")
            print("############# FIRST / SECOND ##############")
            print("FIRST HALF")
            loss_fn = nn.CrossEntropyLoss()
            model_fh = utils.get_model(OPT.model, OPT.num_classes, OPT.pretrained)
            optimizer = optim.AdamW(model_fh.parameters(), lr=OPT.LR_FH, weight_decay=OPT.WD_FH)
            fh = Trainer(model_fh, OPT.device, OPT.num_classes, writer, tag=f'{OPT.dataset}_{OPT.model}_fh')
            fh.train_eval(optimizer, loss_fn, OPT.EPOCHS_FH, fh_train_loader, fh_val_loader)

            print("SECOND HALF")
            model_sh = copy.deepcopy(model_fh)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model_sh.parameters(), lr=OPT.LR_SH, weight_decay=OPT.WD_SH)
            sh = Trainer(model_sh, OPT.device, OPT.num_classes, writer, tag=f'{OPT.dataset}_{OPT.model}_sh')
            sh.train_eval(optimizer, loss_fn, OPT.EPOCHS_SH, sh_train_loader, sh_val_loader)

    print("###########################################")
    print("############## CONTINUAL STEP #############")

    # Copy the model to avoid overwriting
    if OPT.do_warmup:
        model_c = copy.deepcopy(model_fh)
    else:
        model_c = utils.get_model(OPT.model, OPT.num_classes, OPT.pretrained)

    # Create the approach and train
    continual_metrics = []
    OPT.args_cont['model'] = model_c
    strategy = MethodFactory(OPT.strategy, **OPT.args_cont)
    print(f"Continual learning with {OPT.strategy} strategy")
    if OPT.dataset == 'Core50' and OPT.split_core:
        matrix = torch.zeros((10, 11))
    else:
        matrix = torch.zeros((OPT.num_tasks, OPT.num_tasks)) ### not safe
         
    for task_id, (task_train_loader, task_val_loader) in enumerate(tasks):

        print(f"---Task {task_id}---")
        tag = f'{task_id}'
        strategy.train(task_train_loader, task_val_loader, writer, tag)
        if OPT.do_warmup:
            loss, acc = strategy.eval(sh_val_loader, writer, 'sh')
        else:
            if OPT.dataset == 'Core50':
                for eval_task_id , (eval_task_train_loader, eval_task_val_loader) in enumerate(tasks):
                    loss, acc = strategy.eval(eval_task_val_loader, None, 'sh')
                    matrix[task_id, eval_task_id] = acc
                
                if OPT.split_core:
                    loss, acc = strategy.eval(validation, writer, 'sh')
                    matrix[task_id, -1] = acc
                else:
                    matrix[task_id, -1] = 0.
                    
            else:
                for eval_task_id , (eval_task_train_loader, eval_task_val_loader) in enumerate(tasks):
                    loss, acc = strategy.eval(eval_task_val_loader, None, 'sh')
                    matrix[task_id, eval_task_id] = acc
                loss, acc = strategy.eval(sh_val_loader, writer, 'sh')

        continual_metrics.append((loss, acc))



    
    print("###########################################")
    print("############## WRITE TO CSV #############")
    if OPT.do_warmup:
        loss_fn = nn.CrossEntropyLoss()
        _, fh_acc = fh.eval(sh_val_loader, loss_fn)
        print(f"model_first_half accuracy @ eval_second_half: {fh_acc:.5f}")
        _, sh_acc = sh.eval(sh_val_loader, loss_fn)
        print(f"model_second_half accuracy @ eval_second_half: {sh_acc:.5f}")

        data = [seed] + [a for l, a in continual_metrics] + [fh_acc, sh_acc]
    else:
        data = [seed] + [a for l, a in continual_metrics] + [0, 0]
    
    row = ",".join(str(value) for value in data)
    utils.write_line_to_csv(row, strategy.get_csv_name(), OPT.append)
    if OPT.dataset == "Core50":
        seaborn.heatmap(matrix, vmin = 0, vmax = 1, annot = True)
        plt.savefig(f"{OPTOPT.project_folder}/matrices/{strategy.get_csv_name()[:-4].split('/')[-1]}_{OPT.seed}.pdf")

    if OPT.tboard:
        writer.close()


if __name__ == "__main__":   

    main(OPT.seed)

