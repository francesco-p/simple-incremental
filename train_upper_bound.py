from cProfile import label
from torch import optim
import torch.nn as nn
import timm
import utils
from torch.utils.tensorboard import SummaryWriter
import dset
from trainer import Trainer

class UB:
    SEED = 1
    DATASET = 'CIFAR100'
    MODEL = 'resnet18'
    DEVICE = 'cuda'
    PRETRAINED = False
    TENSORBOARD = True
    LR = 1e-3
    WD = 1e-4
    EPOCHS = 50
    BATCH_SIZE = 64
    NUM_CLASSES = 100

def main():

    print("###########################################")
    print("########### TRAIN ON ALL DATASET ##########")

    # Logging
    writer = SummaryWriter(log_dir=f'runs/{UB.DATASET}_{UB.MODEL}_seed{UB.SEED}_ub') if UB.TENSORBOARD else None

    # Seeds
    utils.set_seeds(UB.SEED)

    # Create data
    train_data = dset.get_dset_data(UB.DATASET, train=True)
    train_loader, val_loader = dset.split_train_val(train_data, UB.BATCH_SIZE)

    # Define optimizer model and loss function
    loss_fn = nn.CrossEntropyLoss()
    model = utils.get_model(UB.MODEL, UB.NUM_CLASSES, UB.PRETRAINED)
    optimizer = optim.Adam(model.parameters(), lr=UB.LR, weight_decay=UB.WD)

    fh = Trainer(model, UB.DEVICE, UB.NUM_CLASSES, writer, tag=f'{UB.DATASET}_{UB.MODEL}_seed{UB.SEED}_ub')
    fh.train_eval(optimizer, loss_fn, UB.EPOCHS, train_loader, val_loader, scheduler=True)


if __name__ == '__main__':
    print(UB.__dict__)
    main()