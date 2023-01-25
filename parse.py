import argparse

dataset_stats = {

    'CIFAR10': (10, (3,32,32)),

    'CIFAR100': (100, (3,32,32))

}

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    # Changing options -- Apart from these arguments, we do not mess with other arguments
    parser.add_argument('--data_dir', type=str, default='~/data', help='Directory where all datasets are stored')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='Directory where all logs are stored')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility of class-setting etc')
    parser.add_argument('--exp_name', type=str, default='test', help='Experiment name')

    parser.add_argument('--dataset', type=str, required=True, help='Name of dataset', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'TinyImagenet', 'ImageNet100', 'ImageNet'])

    parser.add_argument('--num_tasks', type=int, required=True, help='Number of tasks')

    parser.add_argument('--model', type=str, default='MLP', choices=['MLP', 'ResNet', 'DenseNet', 'NIN'], help='Model architecture')

    # Approaches
    parser.add_argument('--approach', type=str, required=True, choices=['surgical', 'less_forg', 'ojkd'])

    # Default experiment options
    parser.add_argument('--maxlr', type=float, default=0.05, help='Starting Learning rate')
    parser.add_argument('--minlr', type=float, default=0.0005, help='Ending Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size to be used in training')
    parser.add_argument('--clip', type=float, default=10.0, help='Gradient Clipped if val >= clip')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel worker threads')

    opt = parser.parse_args()


    return opt


if __name__ == '__main__':
    opt = parse_args()
    print(dataset_stats[opt.dataset])

