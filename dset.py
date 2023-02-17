from opt import OPT
import torch
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import DataLoader, Subset, Dataset
from stats import DSET_TRANSF


class TaskDataset(Dataset):
    """ Dataset for a task. It is a subset of a dataset. """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def convert_to_task_dataset(task_loader):

    return TaskDataset(task_loader.dataset, task_loader.dataset.indices)


def get_dset_data(dataset, data_folder=OPT.DATA_FOLDER, train = True):
    """ Return a dataset object """

    if dataset == 'CIFAR10':
        data = CIFAR10(data_folder, train=train, download=True, transform=DSET_TRANSF[dataset])
    elif dataset == 'CIFAR100':
        data = CIFAR100(data_folder, train=train, download=True, transform=DSET_TRANSF[dataset])
    elif dataset == 'SVHN':
        split = 'train' if train else 'test'
        data = SVHN(data_folder, split=split, download=True, transform=DSET_TRANSF[dataset])
    else:
        raise NotImplementedError

    return data


def split_train_val(data, bsize, workers=8, split=0.9):
    """ Split a dataset in train and val. It returns a train and val tuple"""
    
    train_split = int(len(data) * split)
    train_sbs = Subset(data, range(train_split))
    val_sbs = Subset(data, range(train_split, len(data)))
    train_loader = DataLoader(train_sbs, batch_size=bsize, shuffle=False, num_workers=workers)
    val_loader = DataLoader(val_sbs, batch_size=bsize, shuffle=False, num_workers=workers)

    return train_loader, val_loader


# [TODO] to fix the split quantity in OPT
def prepare_tasks(data, num_tasks, bsize, workers=8):
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
    wrmp_train_sbs = Subset(data, wrmp_train_indices)
    wrmp_val_sbs = Subset(data, wrmp_val_indices)
    wrmp_train_loader = DataLoader(wrmp_train_sbs, batch_size=bsize,shuffle=False, num_workers=workers)
    wrmp_val_loader = DataLoader(wrmp_val_sbs, batch_size=bsize,shuffle=False, num_workers=workers)
    wrmp_task = (wrmp_train_loader, wrmp_val_loader)

    ### 2. Select half dataset
    odd_indices = [x for x in range(1, len(data), 2)]

    # Select 90% to train and 10% to val
    n_wrmp_split = int((len(odd_indices)) * 0.9)
    n_wrmp_train_indices =  odd_indices[:n_wrmp_split]
    n_wrmp_val_indices =  odd_indices[n_wrmp_split:]

    # Prepare loaders for wrmp
    n_wrmp_train_sbs = Subset(data, n_wrmp_train_indices)
    n_wrmp_val_sbs = Subset(data, n_wrmp_val_indices)
    n_wrmp_train_loader = DataLoader(n_wrmp_train_sbs, batch_size=bsize,shuffle=False, num_workers=workers)
    n_wrmp_val_loader = DataLoader(n_wrmp_val_sbs, batch_size=bsize,shuffle=False, num_workers=workers)
    n_wrmp_task = (n_wrmp_train_loader, n_wrmp_val_loader)

    ### 3. Calculate a subtask split length
    tr_task_len = len(n_wrmp_train_indices) // num_tasks
    val_task_len = len(n_wrmp_val_indices) // num_tasks

    # Construct continual tasks
    tasks = []
    subsets = []
    for t in range(num_tasks):
        # Selects subindices
        c_tr_indices = n_wrmp_train_indices[(tr_task_len * t):(tr_task_len * t)+tr_task_len]
        c_val_indices = n_wrmp_val_indices[(val_task_len * t):(val_task_len * t)+val_task_len]

        # Prepare loaders for each task
        task_train_sbs = Subset(data, c_tr_indices)
        task_val_sbs = Subset(data, c_val_indices)
        
        task_train_loader = DataLoader(task_train_sbs, batch_size=bsize,shuffle=False, num_workers=workers)
        task_val_loader = DataLoader(task_val_sbs, batch_size=bsize,shuffle=False, num_workers=workers)
        tasks.append((task_train_loader, task_val_loader))
        subsets.append((task_train_sbs, task_val_sbs))

    return (wrmp_task, n_wrmp_task, tasks, subsets)



if __name__ == "__main__":

    # Prepare data
    data = get_dset_data('CIFAR10', OPT.DATA_FOLDER, train=True)
    train_loader, val_loader = split_train_val(data, bsize=1)
    (fh_train_loader, fh_val_loader), (sh_train_loader, sh_val_loader), tasks, subsets = prepare_tasks(data, num_tasks=10, bsize=1)

    # Print some info
    print('Train loader len: {}'.format(len(train_loader)))
    print('Val loader len: {}'.format(len(val_loader)))
    print('First half train loader len: {}'.format(len(fh_train_loader)))
    print('First half val loader len: {}'.format(len(fh_val_loader)))
    print('Second half train loader len: {}'.format(len(sh_train_loader)))
    print('Second half val loader len: {}'.format(len(sh_val_loader)))
    print('Tasks len: {}'.format(len(tasks)))
    print('Task 0 train loader len: {}'.format(len(tasks[0][0])))
    print('Task 0 val loader len: {}'.format(len(tasks[0][1])))
    print('Task 1 train loader len: {}'.format(len(tasks[1][0])))
    print('Task 1 val loader len: {}'.format(len(tasks[1][1])))
    
    # Prints the first 5 indices for each task (should be different)
    for t in range(len(tasks)):
        print('Task {} train indices: {}'.format(t, tasks[t][0].dataset.indices[:5]))
        print('Task {} val indices: {}'.format(t, tasks[t][1].dataset.indices[:5]))

    # Asserts the indices are correct (no overlap)
    assert len(set(fh_train_loader.dataset.indices).intersection(sh_train_loader.dataset.indices)) == 0

    # asserts the indices for each task are correct (no overlap)
    for t in range(len(tasks)):
        for t2 in range(len(tasks)):
            if t != t2:
                assert len(set(tasks[t][0].dataset.indices).intersection(tasks[t2][0].dataset.indices)) == 0
                assert len(set(tasks[t][1].dataset.indices).intersection(tasks[t2][1].dataset.indices)) == 0
    



    
