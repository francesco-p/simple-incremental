from stats import DSET_CLASSES
import os


# Params
class OPT:

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
    SEEDS = [69] #[69,1337]

    # Datset parameters
    DATASET = 'CIFAR10'
    NUM_TASKS = 5
    NUM_CLASSES = DSET_CLASSES[DATASET]
    BATCH_SIZE = 256
    NUM_WORKERS = 8

    # Model parameters
    MODEL = 'resnet32'
    PRETRAINED = False
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
    EPOCHS_FH = 4
    ############ SH PARAMS ############
    LR_SH = 1e-3
    WD_SH = 1e-4
    EPOCHS_SH = 4
    ############ CONT PARAMS ############
    METHOD_CONT = 'CDD'
    # Approach params, if no params, leave empty dict
    #ARGS_CONT = {'layer':3}
    #ARGS_CONT = {'original_impl':True}
    ARGS_CONT = {}

    LR_CONT = 1e-2
    WD_CONT = 1e-5
    EPOCHS_CONT = 5

    EVAL_EVERY_CONT = 1
    #####################################

    #######CDD#######
    CDD_ITERATIONS = 1000


    ######ICARL#######
    IMG_SHAPE = (3, 32, 32)
    MEMORY_SIZE = 2000
    EMB_SIZE = 64 # 64 for resnet32



    
