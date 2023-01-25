import strategies

DSET_CLASSES = {
        'CIFAR100':100,
        'CIFAR10':10,
        'SVHN':10
    }


def approach_constructor(approach, params):
    if approach == 'surgical':
        if len(params) <= 2:
            raise ValueError(f'{approach=} must contain 2 params found {len(params)} instead')
        else:
            model, layer = params
            strategies.surgicalft.SurgicalFT(model, 
            )





# Params
class OPT:

    SEED = 42

    # Approach
    APPROACH = 'surgical'

    # Approach param
    SURGICAL_LAYER = 2

    # Datset parameters
    DATASET = 'CIFAR100'
    DATA_FOLDER = '~/data'
    NUM_TASKS = 10
    NUM_CLASSES = DSET_CLASSES[DATASET]

    # Model parameters
    PRETRAINED = False
    BATCH_SIZE = 64
    MODEL = 'resnet18' #'dla60x_c'
    DEVICE = 'cuda'

    # Train also on all dataet?
    TRAIN_ALL = False

    # First half parameters
    EPOCHS = 30
    LR = 1e-3
    WD = 1e-4

    # Continual parameters
    CONTINUAL_EPOCHS = 7
    CONTINUAL_LR = 1e-4
    CONTINUAL_WD = 1e-5

    # Load pretrained models on first and second half
    LOAD_FISRT_SECOND_HALF_MODELS = True

    # Model checkpointing
    CHK_FOLDER = '/home/francesco/Documents/single_task/chk/'
    CHK_EVERY = 5
    CHK = True

    # Tensorboard logging
    LOG = False
    LOG_EVERY = 1


    # SURGICAL LAYER
    SURGICAL_LAYER = 3
