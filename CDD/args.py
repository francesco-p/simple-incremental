import torch
import argparse
import os
import sys
#current = os.path.dirname(os.path.realpath(__file__))
#parent_directory = os.path.dirname(current)
#sys.path.append(parent_directory)

from opt import OPT
def make_args(task_id):

    parser = argparse.ArgumentParser(description='Parameter Processing')

    #general    
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--data_path', type=str, default='/home/leonardolabs/data', help='dataset path')
    parser.add_argument('--iteration', type=int, default=1, help='training iterations')
    parser.add_argument('--start_iteration', type=int, default=0, help='training iterations')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--batch', type=int, default=-1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--RP_hid', type=int, default=128)
    parser.add_argument('--name_folder', type=str, default="")
    parser.add_argument('--save_folder', type=str, default="CDD/features_final")
    parser.add_argument('--no_init', action = "store_true")

    # hparms for ae
    parser.add_argument('--ae_iteration', type=int, default=2000)
    parser.add_argument('--lr_ae', type=float, default=1e-2)
    parser.add_argument('--ipc', type=int, default=1)
    parser.add_argument('--hdims', type=list, default=[6,9,12])
    parser.add_argument('--num_seed_vec', type=int, default=16)
    parser.add_argument('--num_decoder', type=int, default=8)

    # Adds custom params
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--padding', type=int, default=0)


    # data
    parser.add_argument('--feature_path', type=str, default='CDD/features_final')
    # save
    parser.add_argument('--save_path', type=str, default='CDD/results')
    parser.add_argument('--exp_name', type=str, default=None)
    # repeat
    parser.add_argument('--num_eval', type=int, default=3)

    # hparms for ours
    parser.add_argument('--lr_seed_vec', type=float, default=1e-1)
    parser.add_argument('--lr_iteration', type=list, default=[50, 100, 150])
    parser.add_argument('--lr_decoder', type=float, default=1e-2)
    parser.add_argument('--linear_schedule', action='store_true')


    # image syn evaluation
    parser.add_argument('--model_eval_pool', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=250)
    parser.add_argument('--buffer_every', type=int, default=100)
    parser.add_argument('--not_eval', action='store_true')

    args = parser.parse_args()
    args.dataset = f"{OPT.DATASET}"
    args.iteration = OPT.CDD_ITERATIONS
    args.name_folder = f"_{task_id}"
    args.device = torch.device(f"cuda:{args.gpu_id}")

    return args
