from stats import DSET_CLASSES, DSET_IMG_SHAPE
import os
import argparse
from terminaltables import AsciiTable

def beautify_args(args):
    # Create a list of lists containing argument names and values

    # Print essential info separately
    essential = ['ni_project_path', 'data_path']
    for v in essential:
        print(f'{v}: '+vars(args)[v])

    arg_lists = []
    row = []
    seen = 0
    for arg in sorted(vars(args)):
        if arg in essential:
            continue
        row.append(f'{arg}: {getattr(args, arg)}')
        if seen % 3 == 0:
            arg_lists.append(row)
            row = []
        seen += 1

    # Create a table object with the argument list
    table = AsciiTable(arg_lists)
    table.title = 'Arguments'

    # Format the table
    #table.inner_row_border = True
    table.inner_heading_row_border = False
    table.inner_column_border = False

    return table.table


def get_opts():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    #####################
    ###### GENERAL ######
    parser = argparse.ArgumentParser(description='Single task training')

    parser.add_argument('--data_path', type=str, default=f'{os.environ["DATA_PATH"]}', help='Path where data is stored')
    parser.add_argument('--ni_project_path', type=str, default=f'{os.environ["NI_PROJECT_PATH"]}', help='Path of current folder')
    parser.add_argument('--device', type=str, default='0', help='Gpu to use, -1 for cpu')
    # diocane
    parser.add_argument('--gpu_id', type=int, default=1)
    
    #####################
    ###### EXPERIM ######
    parser.add_argument('--model', type=str, default='resnet18', help='Model to use')
    # [TODO] add automatic emb size detection
    parser.add_argument('--emb_size', type=int, default=512, help='Embedding size')
    parser.add_argument('--pretrained', default=False, action=argparse.BooleanOptionalAction, help='Use pretrained model')
    
    parser.add_argument('--dataset', type=str, default='Core50', help='Dataset to use')
    parser.add_argument('--split_core', action=argparse.BooleanOptionalAction, default=False, help='split into scenarios')
    parser.add_argument('--num_tasks', type=int, default=11, help='Number of tasks')
    parser.add_argument('--seed', type=int, default=0, help='Seed')

    parser.add_argument('--append', default=True, action=argparse.BooleanOptionalAction, help='Delete current csv file andcreate a new one')
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
    parser.add_argument('--strategy', required=False, type=str, default='CDD', help='Strategy to be used')
    # SURGICAL
    parser.add_argument('--surgical_layer', type=int, default=3, help='Surgical layer')
    # REPLAY / CDD
    parser.add_argument('--buffer_size', type=int, default=500, help='Buffer size of replay strategy')

    #####################
    #### CDD PARAMS #####

    #cdd specific    
    parser.add_argument('--cdd_model', type=str, default='ConvNet', help='model')
    parser.add_argument('--cdd_iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--cdd_start_iteration', type=int, default=0, help='training iterations')
    parser.add_argument('--cdd_half', action='store_true')
    parser.add_argument('--cdd_batch', type=int, default=-1)
    parser.add_argument('--cdd_RP_hid', type=int, default=128)
    parser.add_argument('--cdd_name_folder', type=str, default="")
    parser.add_argument('--cdd_no_init', action = "store_true")
    parser.add_argument('--cdd_save_folder', type=str, default="CDD/features_final")

    # hparms for ae
    parser.add_argument('--cdd_ae_iteration', type=int, default=1000)
    parser.add_argument('--cdd_lr_ae', type=float, default=1e-2)
    parser.add_argument('--cdd_ipc', type=int, default=1)
    parser.add_argument('--cdd_hdims', type=list, default=[6,9,12])
    parser.add_argument('--cdd_num_seed_vec', type=int, default=5)
    parser.add_argument('--cdd_num_decoder', type=int, default=4)

    # Adds custom params
    parser.add_argument('--cdd_stride', type=int, default=2)
    parser.add_argument('--cdd_kernel_size', type=int, default=2)
    parser.add_argument('--cdd_padding', type=int, default=0)

    # data
    parser.add_argument('--cdd_feature_path', type=str, default='CDD/features_final')
    # save
    parser.add_argument('--cdd_save_path', type=str, default='CDD/results')
    parser.add_argument('--cdd_exp_name', type=str, default=None)
    # repeat
    parser.add_argument('--cdd_num_eval', type=int, default=3)

    # hparms for ours
    parser.add_argument('--cdd_lr_seed_vec', type=float, default=1e-1)
    parser.add_argument('--cdd_lr_iteration', type=list, default=[2000, 3000, 4000])
    parser.add_argument('--cdd_lr_decoder', type=float, default=1e-2)
    parser.add_argument('--cdd_linear_schedule', action='store_true')


    # image syn evaluation
    parser.add_argument('--cdd_model_eval_pool', type=str, default="ConvNet")
    parser.add_argument('--cdd_epoch', type=int, default=200)
    parser.add_argument('--cdd_lr', type=float, default=0.01)
    parser.add_argument('--cdd_print_every', type=int, default=100)
    parser.add_argument('--cdd_eval_every', type=int, default=500)
    parser.add_argument('--cdd_buffer_every', type=int, default=100)
    parser.add_argument('--cdd_not_eval', action='store_true')

    opts = parser.parse_args()

    # glue the args together without modifying the original kfs code
    opts.cdd_dataset = opts.dataset
    opts.cdd_data_path = opts.data_path
    opts.cdd_seed = opts.seed
    
    # we need to set the device to cuda:0 or cpu or change all the 
    # occurences of opts.device to opts.gpu_id in kfs code
    opts.gpu_id = opts.device # 0 1 -1
    opts.cdd_gpu_id = opts.gpu_id 
    opts.device = f'cuda:{opts.gpu_id}' if int(opts.gpu_id) >= 0 else 'cpu'
    opts.cdd_device = opts.device

    opts.img_shape = DSET_IMG_SHAPE[opts.dataset]

    # Manages other args that are not in the parser
    opts.csv_folder = f'{opts.ni_project_path}/csv/'
    opts.chk_folder = f'{opts.ni_project_path}/chk/'

    opts.num_classes = DSET_CLASSES[opts.dataset]

    # These are the paremeter that are passed to the strategy
    # need to be checked for each strategys
    #ARGS_CONT = {'original_impl':True}
    if opts.strategy == 'surgicalft':
        opts.args_cont = {'layer':opts.surgical_layer}
    elif opts.strategy == 'replay' or opts.strategy == "CDD":
        opts.args_cont = {'buffer_size':opts.buffer_size}
    else:
        opts.args_cont = {}

    return opts

    
OPT = get_opts()

table = beautify_args(OPT)
print(table)
