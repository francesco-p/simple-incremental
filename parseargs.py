import argparse
from terminaltables import SingleTable
import os

def beautify_args(args):
    # Create a list of lists containing argument names and values
    arg_lists = [['Argument', 'Value']]
    for arg in vars(args):
        arg_lists.append([arg, getattr(args, arg)])

    # Create a table object with the argument list
    table = SingleTable(arg_lists)

    # Format the table
    table.inner_row_border = True
    table.inner_column_border = False

    return table.table

# DEVICE = f'cuda:{opts.device}' if opts.device != '-1' else 'cpu'


# Params
def parse_args():

    #####################
    ###### GENERAL ######
    parser = argparse.ArgumentParser(description='Single task training')

    parser.add_argument('--data_path', type=str, default=f'{os.environ["DATASET_ROOT"]}', help='Path where data is stored')
    parser.add_argument('--project_path', type=str, default=f'{os.environ["NI_PROJECT"]}', help='Path of current folder')
    parser.add_argument('--device', type=str, default='0', help='Gpu to use, -1 for cpu')
    
    #####################
    ###### EXPERIM ######
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use')
    parser.add_argument('--pretrained', default=False, action=argparse.BooleanOptionalAction, help='Use pretrained model')
    
    parser.add_argument('--dataset', type=str, default='Core50', help='Dataset to use')
    parser.add_argument('--num_tasks', type=int, default=11, help='Number of tasks')
    parser.add_argument('--seed', type=int, default=0, help='Seed')

    parser.add_argument('--new_csv', default=False, action=argparse.BooleanOptionalAction, help='Delete current csv file andcreate a new one')
    parser.add_argument('--tboard', default=False, action=argparse.BooleanOptionalAction, help='Tensorboard')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaaluation task every x epochs [use 1 for now]')

    
    #####################
    ###### WARMPUP ######
    parser.add_argument('--do_warmup', default=False, action=argparse.BooleanOptionalAction, help='Do warmup')
    parser.add_argument('--load_checkpoints', default=False, action=argparse.BooleanOptionalAction,help='Load Checkpoint')
    parser.add_argument('--load_epoch', type=int, default=9999, help='Load Checkpoint at epoch')
    parser.add_argument('--load_seed', type=int, default=0, help='Load checkpoint with seed')

    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=0.0001, help='Warmup learning rate')
    parser.add_argument('--warmup_momentum', type=float, default=0.9, help='Warmup momentum')
    parser.add_argument('--warmup_weight_decay', type=float, default=0.0005, help='Warmup weight decay')
    parser.add_argument('--warmup_batch_size', type=int, default=64, help='Warmup batch size')


    ################################
    ###### CONTINUAL TRAINING ######
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs for each strategy task')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    
    ######################
    ###### STRATEGY ######
    parser.add_argument('--strategy', type=str, default='finetuning', help='Strategy to be used')
    # SURGICAL
    parser.add_argument('--surgical_layer', type=int, default=3, help='Surgical layer')
    # REPLAY / CDD
    parser.add_argument('--buffer_size', type=int, default=500, help='Buffer size of replay strategy')
    # CDD
    parser.add_argument('--cdd_iterations', type=int, default=1000, help='Buffer size of replay strategy')

    opts = parser.parse_args(['--strategy icarl', '--epochs 1'])
    table = beautify_args(opts)
    print(table)
    return opts