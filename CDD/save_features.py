import os
import time
import argparse
from tqdm import trange

import torch
from torch.utils.data import DataLoader

from data import get_dataset, DiffAugment, ParamDiffAug, get_all_images, TensorDataset
from kfs_models.wrapper import get_model
import torchvision.transforms as transforms
import timm

def main(args, dset):
    args.cdd_device = torch.device(f"cuda:{args.device}")
    args.cdd_dsa_param = ParamDiffAug()
    if args.cdd_dataset == 'SVHN':
        args.cdd_dsa_strategy = 'color_crop_cutout_scale_rotate'
    else:
        args.cdd_dsa_strategy = 'color_crop_cutout_flip_scale_rotate'

    save_path = f'{args.cdd_save_folder}/{args.cdd_dataset}/{args.cdd_model}{args.cdd_name_folder}'
    os.makedirs(save_path, exist_ok=True)

    channel, im_size, num_classes, normalize, images_all, indices_class, _ = get_dataset(args.cdd_dataset, args.cdd_data_path, dset)
    normalize = lambda x: x 

    for it in trange(args.cdd_start_iteration, args.cdd_iteration+1):
        net = get_model(args, args.cdd_model, channel, num_classes, im_size).to(args.cdd_device) # get a random model
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False
        if args.cdd_model == "efficientnet":
            embed = timm.create_model('efficientnet_b0', num_classes = 0, pretrained = False).to(f"cuda:{args.device}")
        elif args.cdd_model == "dla46x_c":
            embed = timm.create_model('dla46x_c', num_classes = 0, pretrained = False).to(f"cuda:{args.device}")
        else:
            embed = net.embed

        features_dict = {}

        with torch.no_grad():
            for c in range(num_classes):
                seed = int(time.time() * 1000) % 100000
                all_img_real = get_all_images(images_all, indices_class, c)
                if args.cdd_batch <= 0:
                    all_img_real = all_img_real.to(args.cdd_device)
                    if args.cdd_half:
                        half_index = int(0.5*len(all_img_real))
                        output_real_mean = 0.0

                        img_real = all_img_real[:half_index]
                        img_real = DiffAugment(normalize(img_real), args.cdd_dsa_strategy, seed=seed, param=args.cdd_dsa_param)
                        output_real_mean += embed(img_real).sum(dim=0)

                        img_real = all_img_real[half_index:]
                        img_real = DiffAugment(normalize(img_real), args.cdd_dsa_strategy, seed=seed, param=args.cdd_dsa_param)
                        output_real_mean += embed(img_real).sum(dim=0)

                        output_real_mean /= len(all_img_real)
                    else:
                        all_img_real = DiffAugment(normalize(all_img_real), args.cdd_dsa_strategy, seed=seed, param=args.cdd_dsa_param)
                        output_real_mean = embed(all_img_real).mean(dim=0)
                else:
                    loader = DataLoader(
                        TensorDataset(all_img_real),
                        batch_size=args.cdd_batch,
                        shuffle=False,
                        num_workers=0
                    )
                    output_real_mean = 0.0
                    for img_real in loader:
                        img_real = img_real.to(args.cdd_device)
                        img_real = DiffAugment(normalize(img_real), args.cdd_dsa_strategy, seed=seed, param=args.cdd_dsa_param)
                        output_real_mean += embed(img_real).sum(dim=0)
                    output_real_mean /= len(all_img_real)

                output_real_mean = output_real_mean.detach().cpu()
                f_dict = {'seed': seed, 'mean': output_real_mean}
                features_dict[c] = f_dict

        net = net.to(torch.device('cpu'))
        features_dict['state_dict'] = net.state_dict()
        save_name = f'dsa_{it}.pth'
        torch.save(features_dict, f'{save_path}/{save_name}')
