from stats import DSET_CLASSES
import os
import argparse


# Params
class OPT:
    parser = argparse.ArgumentParser(description='Single task training')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset to use')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks')
    parser.add_argument('--strategy', type=str, default='replay', help='Strategy to use')
    
    
    parser.add_argument('--surgical_layer', type=int, default=3, help='Surgical layer')
    
    parser.add_argument('--buffer_size', type=int, default=500, help='Buffer size of replay strategy')
    
    opts = parser.parse_args()



    # Folders
    HOME_FOLDER = "/home/francesco/Documents/single_task"
    CSV_FOLDER = f'{HOME_FOLDER}/csv/'
    DATA_FOLDER = f'{HOME_FOLDER}/../../data/'
    CHK_FOLDER = f'{HOME_FOLDER}/chk/'
    
    if not os.path.exists(CHK_FOLDER):
        os.mkdir(CHK_FOLDER)
    
    if not os.path.exists(CSV_FOLDER):
        os.mkdir(CSV_FOLDER)

    # Set multiple seeds for multiple runs
    SEEDS = [0,1,2,3,4,5,6,7,8,9] #[69,1337]

    # Datset parameters
    DATASET = opts.dataset
    NUM_TASKS = opts.num_tasks
    NUM_CLASSES = DSET_CLASSES[DATASET]
    BATCH_SIZE = 64
    NUM_WORKERS = 8

    # Model parameters
    MODEL = opts.model
    PRETRAINED = False
    DEVICE = 'cuda:0'

    # Load pretrained models on first and second half
    LOAD_FISRT_SECOND_HALF_MODELS = True

    # Tensorboard logging
    TENSORBOARD = True

    # Append to .csv ?
    APPEND = True

    ############ ALL PARAMS ############
    # Train also on all dataet?
    ALL = False
    LR_ALL = 1e-3
    WD_ALL = 1e-4
    EPOCHS_ALL = 50
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

    LR_CONT = 1e-4
    WD_CONT = 1e-5
    EPOCHS_CONT = 10

    EVAL_EVERY_CONT = 1
    #####################################

    #######CDD#######
    CDD_ITERATIONS = 1000

    ######ICARL#######
    KEEP = 15

    ######ICARL#######
    IMG_SHAPE = (3, 32, 32)
    MEMORY_SIZE = 500
    EMB_SIZE = 64 # 64 for resnet32



    
