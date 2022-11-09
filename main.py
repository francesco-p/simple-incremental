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
from strategies.less_forg import LessForg
from strategies.surgicalft import SurgicalFT
import utils
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from opt import OPT
import copy
import matplotlib.pyplot as plt
from strategies.finetuning import Finetuning
from strategies.ema import Ema
from models.resnet32 import resnet32
import dset



def main():
   

    if OPT.LOG:
        writer = SummaryWriter()
    else:
        writer = None

    #############################
    #### DATASET PREPARATION ####
    #############################
    train_data = dset.get_dset_data(OPT.DATASET, OPT.DATA_FOLDER, train=True)
    (fh_train_loader, fh_val_loader), (sh_train_loader, sh_val_loader), tasks = dset.prepare_tasks(train_data, OPT.NUM_TASKS)

    test_data = dset.get_dset_data(OPT.DATASET, OPT.DATA_FOLDER, train=False)
    _, small_test_loader = dset.split_train_val(test_data)

    loss_fn = nn.CrossEntropyLoss()

    ##############################
    #### TRAIN ON ALL DATASET ####
    ##############################
    if OPT.TRAIN_ALL:
        model_all = utils.get_model(OPT.MODEL, OPT.NUM_CLASSES, OPT.PRETRAINED)
        optimizer = optim.Adam(model_all.parameters(), lr=OPT.LR, weight_decay=OPT.WD)
        train_loader, val_loader = dset.split_train_val(train_data)
        utils.train_loop(optimizer, model_all, loss_fn, train_loader, val_loader, writer, 'all', scheduler=True)


    #######################
    #### FIRST / SECOND ###
    #######################
    if OPT.LOAD_FISRT_SECOND_HALF_MODELS:
        model_fh, model_sh = utils.load_models(OPT.MODEL, OPT.NUM_CLASSES)

    else:
        print("FIRST HALF")
        model_fh = utils.get_model(OPT.MODEL, OPT.NUM_CLASSES, OPT.PRETRAINED)
        optimizer = optim.Adam(model_fh.parameters(), lr=OPT.LR, weight_decay=OPT.WD)
        utils.train_loop(optimizer, model_fh, loss_fn, fh_train_loader, fh_val_loader, writer, 'fh')

        print("SECOND HALF")
        model_sh = copy.deepcopy(model_fh)
        optimizer = optim.Adam(model_sh.parameters(), lr=OPT.CONTINUAL_LR, weight_decay=OPT.CONTINUAL_WD)
        utils.train_loop(optimizer, model_sh, loss_fn, sh_train_loader, sh_val_loader, writer, 'sh')

    ##################
    #### CONTINUAL ###
    ##################

    model_c = copy.deepcopy(model_fh)

    print("CONTINUAL")
    c_a = []
    approach = SurgicalFT(model_c, layer=1)
    for t, (tr, val) in enumerate(tasks):
        print(f"---{t}---")
        approach.train(tr, val, writer, f't{t}')
        loss, acc = utils.test(model_c, approach.loss_fn, sh_val_loader)
        c_a.append((loss, acc))

    # print("m_a @ val2:")
    # all_l, all_a = utils.test(model_all, loss_fn, sh_val_loader) 
    print("m1 @ val2:")
    fh_l, fh_a = utils.test(model_fh, loss_fn, sh_val_loader)
    print("m2 @ val2:")
    sh_l, sh_a = utils.test(model_sh, loss_fn, sh_val_loader)

    #utils.plot(fh_a, sh_a, c_a, approach.name)
    with open(f"values_layer2.csv","a") as f:
        values = [OPT.SEED] + [x for y,x in c_a]
        row = ",".join(str(x) for x in values)
        f.write(row + "\n")

    ### TEST 
    #
    #utils.test(model_1, loss_fn, test_loader)

    if OPT.LOG:
        writer.close()


if __name__ == "__main__":
    for s in range(10):
        OPT.SEED = s
        utils.set_seeds(OPT.SEED)
        main()
