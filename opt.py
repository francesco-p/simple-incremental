# Params
class OPT:

    EPOCHS = 100
    BATCH_SIZE = 256
    LR = 1e-3
    WD = 1e-4
    DATA_FOLDER = '~/data'
    CHK_FOLDER = '/home/francesco/Documents/single_task/chk'
    SEED = 42
    NUM_CLASSES = 10
    LOG_EVERY = 1
    CHK_EVERY = 2
    DEVICE = 'cuda'
    MODEL = 'resnet50'
    NUM_TASKS = 10