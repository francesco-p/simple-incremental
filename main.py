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

utils.set_seeds()

train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# Prepare datasets and tasks
train_cifar_data = datasets.CIFAR10(OPT.DATA_FOLDER, train=True, download=True, transform=train_tfms)



##############################
### PREPARE DATASET SPLITS ###
##############################

first_half, second_half, tasks = utils.prepare_tasks(train_cifar_data, 10)

##################
### first half ###
##################

# Model definition
model_fh =  timm.create_model(OPT.MODEL, pretrained=False, num_classes=OPT.NUM_CLASSES)
model_fh.to(OPT.DEVICE)

# Define loss function and optimizer
optimizer = optim.Adam(model_1.parameters(), lr=OPT.LR, weight_decay=OPT.WD)
loss_fn = nn.BCEWithLogitsLoss()

# Prepare half of CIFAR as warmup
fh_train_loader, fh_val_loader = first_half

# Train
writer = SummaryWriter()
utils.train_loop(optimizer, model_fh, loss_fn, fh_train_loader, fh_val_loader, writer, 'fh')
writer.close()

##################
### secnd half ###
##################

# Model definition
model_sh =  timm.create_model(OPT.MODEL, pretrained=False, num_classes=OPT.NUM_CLASSES)
model_sh.to(OPT.DEVICE)

# Define loss function and optimizer
optimizer = optim.Adam(model_sh.parameters(), lr=OPT.LR, weight_decay=OPT.WD)
loss_fn = nn.BCEWithLogitsLoss()

sh_train_loader, sh_val_loader = second_half

# Train
writer = SummaryWriter()
utils.train_loop(optimizer, model_sh, loss_fn, sh_train_loader, sh_val_loader, writer, 'sh')
writer.close()


##################
#### CONTINUAL ###
##################

# Model definition
#model_1 =  timm.create_model(OPT.MODEL, pretrained=False, num_classes=OPT.NUM_CLASSES)
#model_1.to(OPT.DEVICE)

# Define loss function and optimizer
optimizer = optim.Adam(model_1.parameters(), lr=OPT.LR)
loss_fn = nn.BCEWithLogitsLoss()

model_fh.load_state_dict(torch.load('/home/francesco/Documents/single_task/chk/half_1/0040.pt'))

for t, (tr, val) in enumerate(tasks):

    writer = SummaryWriter()
    utils.train_loop(optimizer, model_fh, loss_fn, tr, val, writer, f't{t}')
    writer.close()



utils.test(model_fh, loss_fn, sh_val_loader) # p1: should be the worst
utils.test(model_sh, loss_fn, sh_val_loader) # p2: should be the best
utils.test(model_c, loss_fn, sh_val_loader) # pc: p1<pc<p2



##############
#### TEST ####
##############
#test_cifar_data = datasets.CIFAR10(OPT.DATA_FOLDER, train=False, download=True, transform=train_tfms)
#test_loader = DataLoader(test_cifar_data, batch_size=OPT.BATCH_SIZE, drop_last=False)
#
#utils.test(model_1, loss_fn, test_loader)
