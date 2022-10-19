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
from utils import get_indices, OPT, train_loop
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter


# Set seeds
torch.manual_seed(OPT.SEED)
random.seed(OPT.SEED)
np.random.seed(OPT.SEED)

trans = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_cifar_data = datasets.CIFAR10(OPT.DATA_FOLDER, train=True, download=False, transform=trans)
test_cifar_data = datasets.CIFAR10(OPT.DATA_FOLDER, train=False, download=False, transform=trans)
len(train_cifar_data), len(test_cifar_data)


# Extract example indices of two classes
d12_idx = get_indices(train_cifar_data, 0, 1)

# Split indices in D1 and D2 
d1_idx = d12_idx[:5000]
d2_idx = d12_idx[5000:]

# Creates D1,D2 train and validation
d1_tr, d1_val = Subset(train_cifar_data, d1_idx[:4500]), Subset(train_cifar_data, d1_idx[4500:])
d2_tr, d2_val = Subset(train_cifar_data, d2_idx[:4500]), Subset(train_cifar_data, d2_idx[4500:])


# Model definition
model =  timm.create_model(OPT.MODEL, pretrained=False, num_classes=OPT.NUM_CLASSES)
model.to(OPT.NUM_CLASSES)
model.train()

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=OPT.LR)
loss_fn = nn.BCEWithLogitsLoss()

# Train loop
train_loader = DataLoader(d1_tr, batch_size=OPT.TRAINING_BATCH_SIZE, drop_last=False)
val_loader = DataLoader(d1_val, batch_size=OPT.VAL_BATCH_SIZE, drop_last=False)

writer = SummaryWriter()
train_loop(optimizer, model, loss_fn, train_loader, val_loader, writer)
writer.close()