from stats import DSET_CLASSES
import os


# Params
class OPT:

    # Folders
    HOME_FOLDER = "/home/leonardolabs/Documents/simple_incremental"
    CSV_FOLDER = f'{HOME_FOLDER}/csv/'
    DATA_FOLDER = f'{HOME_FOLDER}/../../data/'
    CHK_FOLDER = f'{HOME_FOLDER}/chk/'
    
    if not os.path.exists(CHK_FOLDER):
        os.mkdir(CHK_FOLDER)
    
    if not os.path.exists(CSV_FOLDER):
        os.mkdir(CSV_FOLDER)

    # Set multiple seeds for multiple runs
    SEEDS = [0,1,2,3,4,5,6,7,8,9]#[69, 1337, 444, 555, 666]

    # Datset parameters
    DATASET = 'CIFAR100'
    NUM_TASKS = 5
    NUM_CLASSES = DSET_CLASSES[DATASET]
    BATCH_SIZE = 64

    # Model parameters
    MODEL = 'resnet18'
    PRETRAINED = True
    DEVICE = 'cuda:0'

    # Load pretrained models on first and second half
    LOAD_FISRT_SECOND_HALF_MODELS = True

    # Tensorboard logging
    TENSORBOARD = False

    ############ ALL PARAMS ############
    # Train also on all dataet?
    ALL = False
    LR_ALL = 1e-3
    WD_ALL = 1e-4
    EPOCHS_ALL = 1
    ############ FH PARAMS ############
    LR_FH = 1e-3
    WD_FH = 1e-4
    EPOCHS_FH = 25
    ############ SH PARAMS ############
    LR_SH = 1e-4
    WD_SH = 1e-5
    EPOCHS_SH = 25
    ############ CONT PARAMS ############
    METHOD_CONT = 'CDD'
    # Approach params, if no params, leave empty dict
    #ARGS = {'layer':2}
    ARGS_CONT = {}

    LR_CONT = 1e-4
    WD_CONT = 1e-5
    EPOCHS_CONT = 10
    

    EVAL_EVERY_CONT = 1
    #####################################

    #######CDD#######
    CDD_ITERATIONS = 5000
    BUFFER_SIZE = 500

    
