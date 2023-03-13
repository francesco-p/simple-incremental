import argparse
from tqdm import trange

import torch

from data import get_dataset, TensorDataset
from generator import SyntheticImageGenerator

import os
from opt import OPT
def main(args, dset):
    ''' data set '''
    channel, im_size, num_classes, normalize, images_all, indices_class, testloader = get_dataset(OPT.DATASET, args.cdd_data_path, dset)

    #import ipdb; ipdb.set_trace()
    ''' initialize '''
    print(OPT.DATASET, im_size)
    generator = SyntheticImageGenerator(
            num_classes, im_size, args.cdd_num_seed_vec, args.cdd_num_decoder, args.cdd_hdims,
            args.cdd_kernel_size, args.cdd_stride, args.cdd_padding).to(args.cdd_device)

    optimizer_ae = torch.optim.Adam(generator.parameters(), lr=args.cdd_lr_ae)
    scheduler_ae = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_ae, milestones=[int(0.5*args.cdd_ae_iteration)], gamma=0.1)
    img_real_dataloader = torch.utils.data.DataLoader(
        TensorDataset(images_all.detach().cpu()), batch_size=256, shuffle=True, num_workers=8, drop_last=True)
    img_real_iter = iter(img_real_dataloader)
    for i in trange(1, args.cdd_ae_iteration+1):
        try:
            img_real = next(img_real_iter)
        except StopIteration:
            img_real_iter = iter(img_real_dataloader)
            img_real = next(img_real_iter)
        
        img_real = img_real.to(args.cdd_device)
        loss = generator.autoencoder(img_real)
        optimizer_ae.zero_grad()
        loss.backward()
        optimizer_ae.step()
        scheduler_ae.step()

        if i % 1000 == 0:
            print(f'pretrain step {i}: {loss.item()}')

    os.makedirs("./pretrained_ae/", exist_ok=True)
    save_name = f'CDD/pretrained_ae/{OPT.DATASET}_{args.cdd_ipc}_{args.cdd_num_seed_vec}_{args.cdd_num_decoder}_seed_{OPT.SEED}.pth'
    torch.save(generator.state_dict(), save_name)

