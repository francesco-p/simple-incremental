
from ssl import CertificateError
from torch.utils.data import Subset
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from opt import OPT
# import chnklib
import random
import numpy as np


def set_seeds():
    """ Set reproducibility seeds """
    torch.manual_seed(OPT.SEED)
    random.seed(OPT.SEED)
    np.random.seed(OPT.SEED)


def prepare_tasks(data, num_tasks):
    """ Split a dataset in a warmup task + a sequence of tasks. It selects half
    of the dataset to be used as a warmup dataset and then returns a number of
    continual tasks. Each task (both warmup and continual) are a train, val tuple
    :returns tuple(tuple, list(tuples))
    """

    ### 1. Select half dataset
    even_indices = [x for x in range(0, len(data), 2)]

    # Select 90% to train and 10% to val
    wrmp_split = int((len(even_indices)) * 0.9)
    wrmp_train_indices =  even_indices[:wrmp_split]
    wrmp_val_indices =  even_indices[wrmp_split:]

    # Prepare loaders for wrmp
    wrmp_train_sbs = torch.utils.data.Subset(data, wrmp_train_indices)
    wrmp_val_sbs = torch.utils.data.Subset(data, wrmp_val_indices)
    wrmp_train_loader = torch.utils.data.DataLoader(wrmp_train_sbs, batch_size=OPT.BATCH_SIZE,shuffle=True)
    wrmp_val_loader = torch.utils.data.DataLoader(wrmp_val_sbs, batch_size=OPT.BATCH_SIZE,shuffle=True)
    wrmp_task = (wrmp_train_loader, wrmp_val_loader)

    ### 2. Select half dataset
    odd_indices = [x for x in range(1, len(data), 2)]

    # Select 90% to train and 10% to val
    n_wrmp_split = int((len(odd_indices)) * 0.9)
    n_wrmp_train_indices =  odd_indices[:n_wrmp_split]
    n_wrmp_val_indices =  odd_indices[n_wrmp_split:]

    # Prepare loaders for wrmp
    n_wrmp_train_sbs = torch.utils.data.Subset(data, n_wrmp_train_indices)
    n_wrmp_val_sbs = torch.utils.data.Subset(data, n_wrmp_val_indices)
    n_wrmp_train_loader = torch.utils.data.DataLoader(n_wrmp_train_sbs, batch_size=OPT.BATCH_SIZE,shuffle=True)
    n_wrmp_val_loader = torch.utils.data.DataLoader(n_wrmp_val_sbs, batch_size=OPT.BATCH_SIZE,shuffle=True)
    n_wrmp_task = (n_wrmp_train_loader, n_wrmp_val_loader)

    ### 3. Calculate a subtask split length
    task_len = len(odd_indices) // num_tasks
    task_split = int(task_len * 0.9)

    # Construct continual tasks
    tasks = []
    for t in range(num_tasks):
        # Selects subindices
        start_idx = task_len * t
        end_idx = start_idx + task_len
        task_indices = odd_indices[start_idx:end_idx]
        task_train_indices = task_indices[:task_split]
        task_val_indices = task_indices[task_split:]

        # Prepare loaders for each task
        task_train_sbs = torch.utils.data.Subset(data, task_train_indices)
        task_val_sbs = torch.utils.data.Subset(data, task_val_indices)
        task_train_loader = torch.utils.data.DataLoader(task_train_sbs, batch_size=OPT.BATCH_SIZE,shuffle=True)
        task_val_loader = torch.utils.data.DataLoader(task_val_sbs, batch_size=OPT.BATCH_SIZE,shuffle=True)
        tasks.append((task_train_loader, task_val_loader))

    return (wrmp_task, n_wrmp_task, tasks)


def log_metrics(loader, seen, loss, acc, epoch, session, writer, tag):
    """ Prints metrics to screen and logs to tensorboard """
    loss /= seen
    acc /= seen
    print(f'        {session:<6} - l:{loss:.5f}  a:{acc:.5f}')
    writer.add_scalar(f'{tag}/loss/{session}', loss, epoch)
    writer.add_scalar(f'{tag}/acc/{session}', acc, epoch)


def train_loop(optimizer, model, loss_fn, train_loader, val_loader, writer, tag, scheduler=False):

    if scheduler:
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, epochs=OPT.EPOCHS, steps_per_epoch=len(train_loader))

    for epoch in range(0, OPT.EPOCHS):
        print(f'    EPOCH {epoch} ')

        ##################
        #### Training ####
        cumul_loss_train = 0
        cumul_acc_train = 0
        seen = 0
        model.train()
        for x, y in train_loader:
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

            # Record & update learning rate
            if scheduler:
                sched.step()

            # Compute measures
            cumul_loss_train += loss_train.item()
            cumul_acc_train += (y_hat.argmax(dim=1) == y).sum().item()
            seen += len(y)

        # Print measures
        log_metrics(train_loader, seen, cumul_loss_train, cumul_acc_train, epoch, 'train', writer, tag)

        ####################
        #### Validation ####
        if (epoch == 0) or ((epoch % OPT.LOG_EVERY) == 0):
                with torch.no_grad():
                    cumul_loss_val = 0
                    cumul_acc_val = 0
                    seen = 0
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
                        seen += len(y)

                    # Print measures
                    log_metrics(val_loader, seen, cumul_loss_val, cumul_acc_val, epoch, 'val', writer, tag)

        # Save the model
        if (epoch % OPT.CHK_EVERY) == 0:
            torch.save(model.state_dict(), OPT.CHK_FOLDER+f'/{tag}_{epoch:04}.pt')


def test(model, loss_fn, test_loader):

    with torch.no_grad():
        cumul_loss = 0
        cumul_acc = 0
        seen = 0
        model.eval()
        for x, y in test_loader:

            # Move to GPU
            x = x.to(OPT.DEVICE)
            y = y.to(OPT.DEVICE)

            # Forward to model
            y_hat = model(x).to(torch.float32)
            y_onehot = F.one_hot(y, num_classes=OPT.NUM_CLASSES).to(torch.float32)
            loss_val = loss_fn(y_hat, y_onehot)

            # Compute measures
            cumul_loss += loss_val.item()
            cumul_acc += (y_hat.argmax(dim=1) == y).sum().item()
            seen += len(y)

        # Prints metrics
        cumul_loss /= seen
        print(f'----------')
        print(f'        loss_test:{cumul_loss:.5f}')

        cumul_acc /= seen
        print(f'        acc_test:{cumul_acc:.5f}')



def train_all_dset(train_cifar_data):
    # Model definition
    model_1 =  timm.create_model(OPT.MODEL, pretrained=False, num_classes=OPT.NUM_CLASSES)
    model_1.to(OPT.DEVICE)

    # Define loss function and optimizer
    optimizer = optim.Adam(model_1.parameters(), lr=OPT.LR, weight_decay=OPT.WD)
    loss_fn = nn.BCEWithLogitsLoss()

    # Split train val
    cifar_len = len(train_cifar_data)
    train_split = int(cifar_len * 0.9)
    train_sbs = torch.utils.data.Subset(train_cifar_data, range(train_split))
    val_sbs = torch.utils.data.Subset(train_cifar_data, range(train_split, cifar_len))
    train_loader = torch.utils.data.DataLoader(train_sbs, batch_size=OPT.BATCH_SIZE,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_sbs, batch_size=OPT.BATCH_SIZE,shuffle=True)

    # Train
    writer = SummaryWriter()
    utils.train_loop(optimizer, model_1, loss_fn, train_loader, val_loader, writer, 'wrm')
    writer.close()

