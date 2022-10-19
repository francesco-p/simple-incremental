
from torch.utils.data import Subset
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


# Params
class OPT:

    EPOCHS = 20
    TRAINING_BATCH_SIZE = 64
    VAL_BATCH_SIZE = 32
    LR = 1e-3
    DATA_FOLDER = '~/data'
    SEED = 42
    NUM_CLASSES = 2
    LOG_EVERY = 1
    DEVICE = 'cuda'
    MODEL = 'resnet18'


def get_indices(data, x, y):
    return [i for i in range(len(data)) if (data[i][1] == x) or (data[i][1] == y)]


def get_train_val_subsets(data, subset, train_percentage=0.9):
    n_train = int(len(subset) * train_percentage)
    train = Subset(data, subset[:n_train])
    test = Subset(data, subset[n_train:])
    return train, test


def log_metrics(loader, loss, acc, epoch, session, writer):
    
    if session == 'train':
        bsize = OPT.TRAINING_BATCH_SIZE
    elif session == 'val':
        bsize = OPT.VAL_BATCH_SIZE
    else:
        raise NotImplementedError 

    examples_seen = (bsize * len(loader))
    
    loss /= examples_seen
    print(f'        loss_{session}:{loss:.5f}')
    writer.add_scalar(f'loss/{session}', loss, epoch)

    acc /= examples_seen
    print(f'        acc_{session}:{acc:.5f}')
    writer.add_scalar(f'acc/{session}', acc, epoch)


def train_loop(optimizer, model, loss_fn, train_loader, val_loader, writer):
    
    for epoch in range(0, OPT.EPOCHS):
        print(f'    EPOCH {epoch} ')
        
        ##################
        #### Training ####
        ##################
        cumul_loss_train = 0
        cumul_acc_train = 0
        model.train()
        for n, (x, y) in enumerate(train_loader):
            # Move to GPU
            x = x.to(OPT.DEVICE)
            y = y.to(OPT.DEVICE)

            # Forward data to model and compute loss
            y_hat = model(x).to(torch.float32)
            y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
            loss_train = loss_fn(y_hat, y_onehot)
            
            # Backward
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Compute measures
            cumul_loss_train += loss_train.item()
            cumul_acc_train += (y_hat.argmax(dim=1) == y).sum().item()

        # Print measures
        log_metrics(train_loader, cumul_loss_train, cumul_acc_train, epoch, 'train', writer)


        ####################
        #### Validation ####
        ####################
        if (epoch == 0) or ((epoch % OPT.LOG_EVERY) == 0):
                with torch.no_grad():
                    cumul_loss_val = 0
                    cumul_acc_val = 0
                    model.eval()
                    for x, y in val_loader:
                        
                        # Move to GPU
                        x = x.to(OPT.DEVICE)
                        y = y.to(OPT.DEVICE)

                        # Forward to model
                        y_hat = model(x).to(torch.float32)
                        y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
                        loss_val = loss_fn(y_hat, y_onehot)

                        # Compute measures
                        cumul_loss_val += loss_val.item()
                        cumul_acc_val += (y_hat.argmax(dim=1) == y).sum().item()
                    
                    # Print measures
                    log_metrics(val_loader, cumul_loss_val, cumul_acc_val, epoch, 'val', writer)

    
