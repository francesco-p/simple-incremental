import torch
from torch.utils.data import Dataset


def cls_to_idx(train_set):
    """ Given a dataset it returns the indices of the elements of the classes
    train_set: pytorch dataset
    """
    n_classes = len(train_set.classes)
    cls_indices = []
    for cid in range(n_classes):
        cls_indices.append(torch.where(torch.tensor(train_set.targets, dtype=torch.long) == cid)[0])
    return cls_indices

def gen_subset(dset, mapping, indices):
    subset = None
    for cid in indices:
        cls_idx = mapping[cid.item()]
        if subset == None:
            subset = cls_idx
        else:
            subset = torch.cat((subset, cls_idx), dim=0)
    return torch.utils.data.Subset(dset, subset)

def task_gen(train_set, increment):
    """ Generates a continual setup class incremental
    train_set: torch.dataset (with two fields .classes=name of classes and .target=the labels)
    increment: the number of classes per task
    returns: a generator dataset with the sub
    """
    n_classes = len(train_set.classes)
    perm = torch.randperm(n_classes)
    init = 0
    if (n_classes % increment) != 0:
        raise ValueError(f"Incremental step must divide the total number of classes {n_classes}")
    else:
        n_tasks = n_classes // increment

    mapping = cls_to_idx(train_set)

    for i in range(n_tasks):
        init = i * increment
        end = init + increment
        indices = perm[init:end]
        names = [train_set.classes[x] for x in indices]
        yield gen_subset(train_set, mapping, indices), names, indices, perm



def create_subset_dset(dset, subset, task_classes_name, task_classes_indices):
    X = None
    Y = None
    for idx in subset:
        tmp_x = dset[idx][0].unsqueeze(0)
        tmp_y = torch.tensor(task_classes_indices.tolist().index(dset[idx][1])).unsqueeze(0)

        if X == None:
            X = tmp_x
            Y = tmp_y
        else:
            X = torch.cat((X, tmp_x), dim=0)
            Y = torch.cat((Y, tmp_y), dim=-1)

    return MySubsetDataset(X, Y , task_classes_indices, task_classes_name)



class MySubsetDataset(Dataset):
    def __init__(self, x, y, original_y, y_names):
        self.x = x
        self.y = y
        self.original_y = original_y
        self.y_names = y_names

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


###############################################################################

def get_indices(data, class_indices):
    """ Given a dataset, it selects the indices of the examples
    in class indices
    :returns: list(int)
    """
    return [i for i in range(len(data)) if data[i][1] in class_indices]


def get_train_val_subsets(data, subset, train_percentage=0.9):
    n_train = int(len(subset) * train_percentage)
    train = Subset(data, subset[:n_train])
    val = Subset(data, subset[n_train:])
    return train, val


# def split_dataset():
#     # Extract example indices of two classes
#     d12_idx = get_indices(train_cifar_data, [0, 1])

# def split_dataset():
#     # Extract example indices of two classes
#     d12_idx = get_indices(train_cifar_data, [0, 1])

#     # Split indices in D1 and D2 
#     d1_idx = d12_idx[:5000]
#     d2_idx = d12_idx[5000:]

#     # Creates D1,D2 train and validation
#     d12_idx = Subset(train_cifar_data, d12_idx[:8000]), Subset(train_cifar_data, d12_idx[8000:])
#     d1_tr, d1_val = Subset(train_cifar_data, d1_idx[:4500]), Subset(train_cifar_data, d1_idx[4500:])
#     d2_tr, d2_val = Subset(train_cifar_data, d2_idx[:4500]), Subset(train_cifar_data, d2_idx[4500:])




# # Model definition
# model_all =  timm.create_model(OPT.MODEL, pretrained=False, num_classes=OPT.NUM_CLASSES)
# model_all.to(OPT.DEVICE)
