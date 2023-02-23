from opt import OPT
import torch
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from stats import DSET_TRANSF
import glob
import random
import cv2
from torchvision import transforms
import stats
import numpy as np
from tqdm import tqdm

class Core50Dataset(Dataset):
    """ Scenario Dataset for Core50 it requires a scenario number  """
    
    def __init__(self, data_path, scenario_n, preload=False, load_data_in_memory=True, transform=None):
        self.data_path = data_path+'/core50_128x128'
        self.transform = transform
        self.scenario_n = scenario_n
        self._set_data_and_targets()
        
        if preload:
            self._preloaded_data()
        else:
            if load_data_in_memory:
                self._load_data()
            else:
                raise NotImplementedError('we need to check this case...for icarl')

    def _preloaded_data(self):
        self.data = torch.load(self.data_path+f'_scenarios/core50_scenario_{self.scenario_n}')

    def _load_data(self):
        self.data = np.zeros((len(self.paths), 128, 128, 3)).astype(np.uint8)
        for i, path in tqdm(enumerate(self.paths)):
            x = cv2.imread(path).astype(np.uint8)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.uint8)
            self.data[i] = x

    def _set_data_and_targets(self):
        """ Retrieve all paths and targets and shuffle them"""

        # Retrieve all paths of the specified shenario
        self.paths = glob.glob(self.data_path+'/'+f's{self.scenario_n}/*/*.png')
        self.targets = self._extract_targets_from_paths(self.paths)
        
        # Shuffle the lists in unison
        combined = list(zip(self.paths, self.targets))
        random.shuffle(combined)
        self.paths, self.targets = zip(*combined)

        # Retrieve all
        #self.paths[-1] = glob.glob(self.data_path+'/*/*/*.png')        
    
    def reset_task_to(self, scenario_n):
        """ Reset the dataset to a new scenario"""
        self.scenario_n = scenario_n
        self._set_data_and_targets(scenario_n)

    def _extract_targets_from_paths(self, paths):
        targets = []
        for path in paths:
            # Corrects targets starting from 0 to 49
            targets.append(int(path.split('/')[-2][1:])-1)
        return targets
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        x = cv2.imread(self.paths[index])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        y = self.targets[index]
        if self.transform:
            x = self.transform(x)

        return x, y


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
    elif dataset == 'Core50':
        data = Core50Dataset(data_folder, scenario_number=-1)
    elif dataset == 'CIFAR100':
        data = CIFAR100(data_folder, train=train, download=True, transform=DSET_TRANSF[dataset])
    elif dataset == 'SVHN':
        split = 'train' if train else 'test'
        data = SVHN(data_folder, split=split, download=True, transform=DSET_TRANSF[dataset])
    else:
        raise NotImplementedError(f'Dataset not implemented, found {dataset}')

    return data


def split_train_val(data, batch_size, split=0.9, return_subsets=False, shuffle=True, num_workers=8):
    """ Split a dataset in train and val. It returns a train and val tuple"""
    
    train_split = int(len(data) * split)
    train_sbs = Subset(data, range(train_split))
    val_sbs = Subset(data, range(train_split, len(data)))
    train_loader = DataLoader(train_sbs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_sbs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    if return_subsets:
        return (train_sbs, val_sbs), (train_loader, val_loader)
    else:
        return train_loader, val_loader


def get_tasks(data, num_tasks, bsize, do_warmup=False, workers=8):
    if do_warmup:
        return split_data_in_tasks_with_warmup(data, num_tasks, bsize, workers)
    else:
        return split_data_in_tasks(data, num_tasks, bsize, workers)


# [TODO] to fix the split quantity in OPT
def split_data_in_tasks(data, num_tasks, bsize, workers=8):
    """ Split a dataset in num_tasks tasks. It returns a list of tuples, each tuple is a train, val tuple"""
    
    # Select 90% to train and 10% to val
    train_len = int((len(data)) * 0.9)
    val_len =  len(data) - train_len

    # Calculate the split for each task
    length_of_train_task = train_len // num_tasks
    length_of_val_task = val_len // num_tasks
    
    # Prepare loaders for each task
    tasks = []
    subsets = []
    for i in range(num_tasks):
        init, end = i*length_of_train_task, (i+1)*length_of_train_task
        train_sbs = Subset(data, range(init, end))

        init, end = train_len+(i*length_of_val_task), train_len+((i+1)*length_of_val_task)
        val_sbs = Subset(data, range(init, end))
        
        train_loader = DataLoader(train_sbs, batch_size=bsize, shuffle=True, num_workers=workers)
        val_loader = DataLoader(val_sbs, batch_size=bsize, shuffle=True, num_workers=workers)
        tasks.append((train_loader, val_loader))
        subsets.append((train_sbs, val_sbs))
        
    return tasks, subsets




# [TODO] to fix the split quantity in OPT
def split_data_in_tasks_with_warmup(data, num_tasks, bsize, workers=8):
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


def gen_core50_tasks():
    """ Generate a list of tasks for Core50 dataset """
    task_id = [x for x in range(1,12)]
    random.shuffle(task_id)

    tasks = []
    val_dsets = []    
    for i in task_id[:-1]:
        dataset = Core50Dataset(OPT.DATA_FOLDER, scenario_n=i, transform=stats.DSET_TRANSF['Core50'])
        (tr_sbs, val_sbs), (train, val) = split_train_val(dataset, batch_size=OPT.BATCH_SIZE, return_subsets=True)
        tasks.append((train, val))
        val_dsets.append(val_sbs)

    # Append the last unseen scenario
    val_dsets.append(Core50Dataset(OPT.DATA_FOLDER, scenario_n=task_id[-1], transform=stats.DSET_TRANSF['Core50']))
    val_dset = ConcatDataset(val_dsets)
    val_loader = DataLoader(val_dset, batch_size=OPT.BATCH_SIZE, shuffle=True, num_workers=OPT.NUM_WORKERS)

    return tasks, val_loader


