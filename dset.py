from opt import OPT
import torch
from torchvision import datasets
from torchvision import transforms as trs

TRANSFOMATIONS = {

    "CIFAR10": trs.Compose([trs.RandomCrop(32, padding=4, padding_mode='reflect'),
                trs.RandomHorizontalFlip(),
                trs.ToTensor(),
                trs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),

    "CIFAR100": trs.Compose([trs.RandomCrop(32, padding=4, padding_mode='reflect'),
                trs.RandomHorizontalFlip(),
                trs.ToTensor(),
                trs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),

    "SVHN": trs.Compose([trs.RandomHorizontalFlip(),
                trs.ToTensor()])
}


def get_dset_data(dataset, data_folder, train):
    if dataset == 'CIFAR10':
        data = datasets.CIFAR10(data_folder, train=train, download=True, transform=TRANSFOMATIONS[dataset])
    elif dataset == 'CIFAR100':
        data = datasets.CIFAR100(data_folder, train=train, download=True, transform=TRANSFOMATIONS[dataset])
    elif dataset == 'SVHN':
        split = 'train' if train else 'test'
        data = datasets.SVHN(data_folder, split=split, download=True, transform=TRANSFOMATIONS[dataset])
    else:
        raise NotImplementedError

    return data


def split_train_val(data, split=0.9):
    train_split = int(len(data) * split)
    train_sbs = torch.utils.data.Subset(data, range(train_split))
    val_sbs = torch.utils.data.Subset(data, range(train_split, len(data)))
    train_loader = torch.utils.data.DataLoader(train_sbs, batch_size=OPT.BATCH_SIZE,shuffle=False, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_sbs, batch_size=OPT.BATCH_SIZE,shuffle=False, num_workers=8)
    return train_loader, val_loader


# [TODO] to fix the split quantity in OPT
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
    wrmp_train_loader = torch.utils.data.DataLoader(wrmp_train_sbs, batch_size=OPT.BATCH_SIZE,shuffle=False, num_workers=8)
    wrmp_val_loader = torch.utils.data.DataLoader(wrmp_val_sbs, batch_size=OPT.BATCH_SIZE,shuffle=False, num_workers=8)
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
    n_wrmp_train_loader = torch.utils.data.DataLoader(n_wrmp_train_sbs, batch_size=OPT.BATCH_SIZE,shuffle=False, num_workers=8)
    n_wrmp_val_loader = torch.utils.data.DataLoader(n_wrmp_val_sbs, batch_size=OPT.BATCH_SIZE,shuffle=False, num_workers=8)
    n_wrmp_task = (n_wrmp_train_loader, n_wrmp_val_loader)

    ### 3. Calculate a subtask split length
    tr_task_len = len(n_wrmp_train_indices) // num_tasks
    val_task_len = len(n_wrmp_val_indices) // num_tasks

    # Construct continual tasks
    tasks = []
    for t in range(num_tasks):
        # Selects subindices
        c_tr_indices = n_wrmp_train_indices[(tr_task_len * t):(tr_task_len * t)+tr_task_len]
        c_val_indices = n_wrmp_val_indices[(val_task_len * t):(val_task_len * t)+val_task_len]

        # Prepare loaders for each task
        task_train_sbs = torch.utils.data.Subset(data, c_tr_indices)
        task_val_sbs = torch.utils.data.Subset(data, c_val_indices)
        task_train_loader = torch.utils.data.DataLoader(task_train_sbs, batch_size=OPT.BATCH_SIZE,shuffle=False, num_workers=8)
        task_val_loader = torch.utils.data.DataLoader(task_val_sbs, batch_size=OPT.BATCH_SIZE,shuffle=False, num_workers=8)
        tasks.append((task_train_loader, task_val_loader))

    return (wrmp_task, n_wrmp_task, tasks)
