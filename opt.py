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
    SEEDS = [69]#[69, 1337, 444, 555, 666]

    # Datset parameters
    DATASET = 'CIFAR100'
    NUM_TASKS = 10
    NUM_CLASSES = DSET_CLASSES[DATASET]
    BATCH_SIZE = 128

    # Model parameters
    MODEL = 'dla46x_c'
    PRETRAINED = True
    DEVICE = 'cuda:0'

    # Load pretrained models on first and second half
    LOAD_FISRT_SECOND_HALF_MODELS = False

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
    EPOCHS_FH = 20
    ############ SH PARAMS ############
    LR_SH = 1e-3
    WD_SH = 1e-4
    EPOCHS_SH = 20
    ############ CONT PARAMS ############
    METHOD_CONT = 'Finetuning'
    # Approach params, if no params, leave empty dict
    #ARGS = {'layer':2}
    ARGS_CONT = {}

    LR_CONT = 1e-4
    WD_CONT = 1e-5
    EPOCHS_CONT = 20

    EVAL_EVERY_CONT = 1
    #####################################

    #######CDD#######
    CDD_ITERATIONS = 20000


    
