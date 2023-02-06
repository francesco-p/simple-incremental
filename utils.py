
import torch
import torch.nn.functional as F
from opt import OPT
import random
import numpy as np
import timm
import matplotlib.pyplot as plt
from models.resnet32 import resnet32
import pandas as pd

def plot_csv(csv_file):

    # Split name of file to get dataset name
    dataset = csv_file.split('/')[-1].split('_')[0]

    # Split name of file to get method name
    method = csv_file.split('/')[-1].split('_')[2]

    # Split name of file to get model name
    model = csv_file.split('/')[-1].split('_')[-1].split('.')[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    df = pd.read_csv(csv_file)

    # number of tasks
    num_tasks = df.shape[1] - 3

    all_results = df.loc[:, df.columns[1:-2]].values
    mean = all_results.mean(axis=0)
    std = all_results.std(axis=0)

    ax.plot(mean, label=method, color='orange', ls='-')
    ax.scatter(range(all_results.shape[1]), mean, color='orange', marker='*')
    ax.fill_between(range(len(mean)), mean-std, mean+std, color='orange', linewidth=1, alpha=0.3)

    ax.set_xticks(range(num_tasks))
    ax.set_xticklabels(range(1, num_tasks+1))
    ax.set_xlabel('Tasks')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Performance on {dataset} dataset - model {model}')
    ax.legend()
    plt.show()
    plt.savefig("plot.png")

def load_models(model, dataset, num_classes, epoch, device=OPT.DEVICE, chk_folder=OPT.CHK_FOLDER):
    """ Load models from checkpoint of a given epoch 
    the epoch number is padded with zeros to 4 digits"""
    
    model_fh =  get_model(model, num_classes, False)
    name = f'{chk_folder}/{dataset}_{model}_fh_epoch{epoch:04}.pt'
    model_fh.load_state_dict(torch.load(name))
    
    model_sh =  get_model(model, num_classes, False)
    name = f'{chk_folder}/{dataset}_{model}_sh_epoch{epoch:04}.pt'
    model_sh.load_state_dict(torch.load(name))
    
    model_fh = model_fh.to(device)
    model_sh = model_sh.to(device)

    return model_fh, model_sh


def write_line_to_csv(data, name, append=False, log=False):
    """ Write a line to a csv file. If append is False, the file is overwritten. """
    # Write header
    header = 'seed,'+','.join([f'task{n+1}' for n in range(OPT.NUM_TASKS)])+',first_half,second_half'
    if not append:
        with open(name, 'w') as f:
            f.write(header+'\n')

    # Write data to file    
    with open(name, 'a') as f:
        f.write(data+'\n')
        if log:
            print(data)


def get_model(model_name, num_classes, pretrained):

    if model_name == 'resnet18':
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'dla46x_c':
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'mobilenetv2_035':
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'resnet32':
        if pretrained:
            raise NotImplementedError('Pretrained resnet32 is not implemented')
        model = resnet32(num_classes=num_classes, pretrained=pretrained)
    else:
        raise NotImplementedError(f"Unknown model {model_name}")

    return model


def set_seeds(seed):
    """ Set reproducibility seeds """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_output(out):
    """ Checks if output is a tuple and returns a dictionary"""
    out_dict = {}
    if type(out) == tuple:
        if out[0].shape == out[1].shape: #accomodates ojkd
            out_dict['bkb'] = out[0]
            out_dict['fr'] = out[1]
        else:
            out_dict['y_hat'], out_dict['fts'] = out[0], out[1]
    else:
        out_dict['y_hat'] = out
    return out_dict



if __name__ == '__main__':
    
    # Some tests
    set_seeds(0)
    x = torch.randn(10, 3, 32, 32)
    model = get_model('resnet18', 10, pretrained=False)
    out = model(x)
    assert out.shape == (10, 10)

    # Test check_output
    out = model(x)
    out_dict = check_output(out)
    assert out_dict['y_hat'].shape == (10, 10)

    # create fake csv file for testing
    csv_file = 'csv/CIFAR100_10tasks_CDD_dla46xc.csv'
    data = f'1000,'+','.join([str(random.random()) for _ in range(10)])+',0.5,0.6'
    write_line_to_csv(data, csv_file, append=False)
        
    # append to csv file some fake data
    for i in range(10):
        data = f'{i},'+','.join([str(random.random()) for _ in range(10)])+',0.5,0.6'
        write_line_to_csv(data, csv_file, append=True)   

    # Test plot csv
    plot_csv(csv_file)
