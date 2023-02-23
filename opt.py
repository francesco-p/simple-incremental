from stats import DSET_CLASSES, DSET_IMG_SHAPE
import os


# Params
class OPT:

  
    # Folders
    PROJECT_FOLDER = os.environ["NI_PROJECT"]
    DATA_FOLDER = os.environ["DATASET_ROOT"]
    CSV_FOLDER = f'{PROJECT_FOLDER}/csv/'
    CHK_FOLDER = f'{PROJECT_FOLDER}/chk/'
    
    # Set up folders
    if not os.path.exists(CHK_FOLDER):
        os.mkdir(CHK_FOLDER)
    
    if not os.path.exists(CSV_FOLDER):
        os.mkdir(CSV_FOLDER)

    # Set multiple seeds for multiple runs
    SEED = 0

    # Datset parameters
    DO_WARMUP = False
    
    DATASET = 'Core50'
    NUM_TASKS = 11
    NUM_CLASSES = DSET_CLASSES[DATASET]
    BATCH_SIZE = 64
    NUM_WORKERS = 8

    # Model parameters
    MODEL = 'resnet18'
    PRETRAINED = False
    FEATURES_ONLY = True
    DEVICE = 'cuda:0'

    # Load pretrained models on first and second half
    LOAD_FISRT_SECOND_HALF_MODELS = True

    # Tensorboard logging
    TENSORBOARD = False

    # Append to .csv ?
    APPEND = True

    ############ FH PARAMS ############
    LR_FH = 1e-3
    WD_FH = 1e-4
    EPOCHS_FH = 25
    ############ SH PARAMS ############
    LR_SH = 1e-3
    WD_SH = 1e-4
    EPOCHS_SH = 25
    ############ CONT PARAMS ############
    METHOD_CONT = 'icarl'
    # Approach params, if no params, leave empty dict
    
    #ARGS_CONT = {'original_impl':True}
    if METHOD_CONT == 'surgicalft':
        ARGS_CONT = {'layer':3}
    elif METHOD_CONT == 'replay':
        ARGS_CONT = {'buffer_size':500}
    else:
        ARGS_CONT = {}

    LR_CONT = 1e-3
    WD_CONT = 1e-5
    EPOCHS_CONT = 1
    EVAL_EVERY_CONT = 1
    #####################################

    #######CDD#######
    CDD_ITERATIONS = 1000

    ######ICARL#######
    KEEP = 15

    ######ICARL#######
    IMG_SHAPE = DSET_IMG_SHAPE[DATASET]
    MEMORY_SIZE = 500
    EMB_SIZE = 64 # 64 for resnet32



    
