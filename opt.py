from stats import DSET_CLASSES, DSET_IMG_SHAPE
import os
import argparse
from terminaltables import SingleTable

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



# Params
class OPT:

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
    parser.add_argument('--strategy', required=True, type=str, default='finetuning', help='Strategy to be used')
    # SURGICAL
    parser.add_argument('--surgical_layer', type=int, default=3, help='Surgical layer')
    # REPLAY / CDD
    parser.add_argument('--buffer_size', type=int, default=500, help='Buffer size of replay strategy')
    # CDD
    parser.add_argument('--cdd_iterations', type=int, default=1000, help='Buffer size of replay strategy')

    opts = parser.parse_args()
    table = beautify_args(opts)
    print(table)

    # Folders
    PROJECT_FOLDER = opts.project_path
    DATA_FOLDER = opts.data_path
    CSV_FOLDER = f'{PROJECT_FOLDER}/csv/'
    CHK_FOLDER = f'{PROJECT_FOLDER}/chk/'
    
    # Set up folders
    if not os.path.exists(CHK_FOLDER):
        os.mkdir(CHK_FOLDER)
    
    if not os.path.exists(CSV_FOLDER):
        os.mkdir(CSV_FOLDER)

    # Set multiple seeds for multiple runs
    SEED = opts.seed 

    # Datset parameters
    DO_WARMUP = opts.do_warmup
    
    DATASET = opts.dataset
    NUM_TASKS = opts.num_tasks
    NUM_CLASSES = DSET_CLASSES[DATASET]
    BATCH_SIZE = opts.batch_size
    NUM_WORKERS = opts.num_workers

    # Model parameters
    MODEL = opts.model
    PRETRAINED = opts.pretrained
    DEVICE = f'cuda:{opts.device}' if opts.device != '-1' else 'cpu'

    # Load pretrained models on first and second half
    LOAD_FISRT_SECOND_HALF_MODELS = True

    # Tensorboard logging
    TENSORBOARD = opts.tboard

    # Append to .csv ?
    APPEND = opts.new_csv

    ############ FH PARAMS ############
    LR_FH = 1e-3
    WD_FH = 1e-4
    EPOCHS_FH = 25
    ############ SH PARAMS ############
    LR_SH = 1e-3
    WD_SH = 1e-4
    EPOCHS_SH = 25
    ############ CONT PARAMS ############
    METHOD_CONT = opts.strategy
    # Approach params, if no params, leave empty dict
    
    #ARGS_CONT = {'original_impl':True}
    if METHOD_CONT == 'surgicalft':
        ARGS_CONT = {'layer':opts.surgical_layer}
    elif METHOD_CONT == 'replay':
        ARGS_CONT = {'buffer_size':opts.buffer_size}
    else:
        ARGS_CONT = {}

    LR_CONT = opts.lr
    WD_CONT = opts.wd
    EPOCHS_CONT = opts.epochs
    EVAL_EVERY_CONT = opts.eval_every
    #####################################

    #######CDD#######
    CDD_ITERATIONS = opts.cdd_iterations

    ######ICARL#######
    KEEP = 15

    ######ICARL#######
    IMG_SHAPE = DSET_IMG_SHAPE[opts.dataset]
    MEMORY_SIZE = opts.buffer_size
    EMB_SIZE = 64 # 64 for resnet32



    
