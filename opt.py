
DSET_CLASSES = {
        'CIFAR100':100,
        'CIFAR10':10,
        'SVHN':10
    }

# Params
class OPT:

    SEED = 42

    # Datset parameters
    DATASET = 'CIFAR100'
    DATA_FOLDER = '~/data'
    NUM_TASKS = 6
    NUM_CLASSES = DSET_CLASSES[DATASET]

    # Model parameters
    PRETRAINED = False
    BATCH_SIZE = 256
    MODEL = 'resnet18'
    DEVICE = 'cuda:1'

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
    CHK_FOLDER = '/home/leonardolabs/Documents/simple-incremental/chk'
    CHK_EVERY = 1
    CHK = True

    # Tensorboard logging
    LOG = False
    LOG_EVERY = 1


    # SURGICAL LAYER
    SURGICAL_LAYER = 2
