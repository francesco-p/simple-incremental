import os
from absl import logging
from datetime import datetime
import numpy as np
import wandb
from tqdm import trange

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR

from opt import OPT
from data import TensorDataset, DiffAugment

def default_args(args):
    if args.cdd_dataset == "SVHN" or args.cdd_dataset == "CIFAR10":
        args.cdd_kernel_size = 2
        args.cdd_stride = 2
        args.cdd_padding = 0
        if args.cdd_ipc == 1: # 1.0046875          
            args.cdd_hdims = [6,9,12]
            args.cdd_num_seed_vec = 13
            args.cdd_num_decoder = 8
        elif args.cdd_ipc == 10: # 10.28828125
            args.cdd_hdims = [6,9,12]
            args.cdd_num_seed_vec = 160
            args.cdd_num_decoder = 12
        elif args.cdd_ipc == 50: # 50.01921875
            args.cdd_hdims = [6,12]
            args.cdd_num_seed_vec = 200
            args.cdd_num_decoder = 16
        else:
            raise NotImplementedError

    elif args.cdd_dataset == "CIFAR100":
        args.cdd_kernel_size = 2
        args.cdd_stride = 2
        args.cdd_padding = 0
        if args.cdd_ipc == 1: # 1.01921875      
            args.cdd_hdims = [6,9,12]
            args.cdd_num_seed_vec = 16
            args.cdd_num_decoder = 8
        elif args.cdd_ipc == 10: # 10.028828125
            args.cdd_hdims = [6,9,12]
            args.cdd_num_seed_vec = 160
            args.cdd_num_decoder = 12
        else:
            raise NotImplementedError

    elif args.cdd_dataset == "ImageNet10":  
        args.cdd_kernel_size = 4
        args.cdd_stride = 2
        args.cdd_padding = 1
        if args.cdd_ipc == 1: # 1.0041015625         
            args.cdd_hdims = [3,3,3]
            args.cdd_num_seed_vec = 64
            args.cdd_num_decoder = 14
        elif args.cdd_ipc == 10: # 10.005422247023809
            args.cdd_hdims = [4,6]
            args.cdd_num_seed_vec = 80
            args.cdd_num_decoder = 14
        else:
            raise NotImplementedError

def sum_params(input):
    tensors = torch.cat([x.view(-1) for x in input])

    dist.all_reduce(tensors)
    #tensors /= dist.get_world_size()

    idx = 0
    for x in input:
        numel = x.numel()
        x.data.copy_(tensors[idx : idx + numel].view(x.size()))
        idx += numel

def get_linear_schedule_warmup_constant(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            #return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def evaluate(args, net, image_syn, label_syn, testloader, normalize):

    trainloader = DataLoader(
        TensorDataset(image_syn, label_syn),
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=OPT.lr, weight_decay=OPT.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.cdd_epoch // 3, 5 * args.cdd_epoch // 6], gamma=0.2)

    # train
    net.train()
    for _ in trange(args.cdd_epoch):        
        for x_tr, y_tr in trainloader:
            # data
            x_tr, y_tr = x_tr.to(args.cdd_device), y_tr.to(args.cdd_device)
            x_tr = DiffAugment(normalize(x_tr), args.cdd_dsa_strategy, param=args.cdd_dsa_param)
            
            # update
            optimizer.zero_grad()
            loss_tr = F.cross_entropy(net(x_tr), y_tr)
            loss_tr.backward()
            optimizer.step()
        
        # scheulder update
        scheduler.step()
    
    # test
    count = 0.0
    loss_te = 0.0
    accuracy_te = 0.0
    net.eval()
    with torch.no_grad():
        for x_te, y_te in testloader:
            # data
            x_te, y_te = x_te.to(args.cdd_device), y_te.to(args.cdd_device)
            x_te = normalize(x_te)

            # prediction
            y_te_pred = net(x_te)
            loss_te += F.cross_entropy(y_te_pred, y_te, reduction='sum')
            accuracy_te += torch.eq(y_te_pred.argmax(dim=-1), y_te).sum().float()
            count += x_te.shape[0]
    
    del net

    return loss_te.item()/count, accuracy_te.item()*100/count

def evaluate_debug(args, net, image_syn, label_syn, testloader, normalize):
   
    
    trainloader = DataLoader(
        TensorDataset(image_syn, label_syn),
        batch_size=args.cdd_batch,
        shuffle=True,
        num_workers=0
    )
    logger = Logger(
        exp_name=f"{args.cdd_eval_opt}_{args.cdd_batch}_{args.cdd_lr}_{args.cdd_wd}_mixup_p=1.0,beta=0.2",
        save_dir=None,
        print_every=1,
        save_every=1,
        total_step=args.cdd_epoch*len(trainloader),
        print_to_stdout=True,
        wandb_project_name=f"syn_debug",
        wandb_tags=[],
        wandb_config=args,
    )
    if args.cdd_eval_opt == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.cdd_lr, momentum=0.9, weight_decay=args.cdd_wd)
    elif args.cdd_eval_opt== "adam":
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.cdd_lr, weight_decay=args.cdd_wd)
    else:
        raise NotImplementedError
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.cdd_epoch // 3, 5 * args.cdd_epoch // 6], gamma=0.2)

    logger.start()

    # train
    for _ in trange(args.cdd_epoch):        
        for i, (x_tr, y_tr) in enumerate(trainloader):
            net.train()
            # data
            x_tr, y_tr = x_tr.to(args.cdd_device), y_tr.to(args.cdd_device)
            x_tr = DiffAugment(normalize(x_tr), args.cdd_dsa_strategy, param=args.cdd_dsa_param)

            # mixup
            r = np.random.rand(1)
            if r < 1.0:
                lam = np.random.beta(0.2, 0.2)
                rand_index = random_indices(y_tr, nclass=10)

                y_tr_b = y_tr[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(x_tr.size(), lam)
                x_tr[:, :, bbx1:bbx2, bby1:bby2] = x_tr[rand_index, :, bbx1:bbx2, bby1:bby2]
                ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_tr.size()[-1] * x_tr.size()[-2]))

                l_tr = net(x_tr)
                loss_tr = F.cross_entropy(l_tr, y_tr) * ratio + F.cross_entropy(l_tr, y_tr_b) * (1. - ratio)
            
            else:
                loss_tr = F.cross_entropy(net(x_tr), y_tr)     
            
            # update
            optimizer.zero_grad()
            loss_tr = F.cross_entropy(net(x_tr), y_tr)
            loss_tr.backward()
            optimizer.step()

            logger.meter("loss", "train", loss_tr)
            if i == len(trainloader) - 1:
                net.eval()
                # test
                count = 0.0
                loss_te = 0.0
                accuracy_te = 0.0
                with torch.no_grad():
                    for x_te, y_te in testloader:
                        # data
                        x_te, y_te = x_te.to(args.cdd_device), y_te.to(args.cdd_device)
                        x_te = normalize(x_te)

                        # prediction
                        y_te_pred = net(x_te)
                        loss_te += F.cross_entropy(y_te_pred, y_te, reduction='sum')
                        accuracy_te += torch.eq(y_te_pred.argmax(dim=-1), y_te).sum().float()
                        count += x_te.shape[0]
            
                logger.meter("loss", "test", loss_te/count)
                logger.meter("accuracy", "test", accuracy_te*100/count)

            logger.step()
        
        # scheulder update
        scheduler.step()

    logger.finish()

    # test
    count = 0.0
    loss_te = 0.0
    accuracy_te = 0.0
    net.eval()
    with torch.no_grad():
        for x_te, y_te in testloader:
            # data
            x_te, y_te = x_te.to(args.cdd_device), y_te.to(args.cdd_device)
            x_te = normalize(x_te)

            # prediction
            y_te_pred = net(x_te)
            loss_te += F.cross_entropy(y_te_pred, y_te, reduction='sum')
            accuracy_te += torch.eq(y_te_pred.argmax(dim=-1), y_te).sum().float()
            count += x_te.shape[0]
    
    del net

    return loss_te.item()/count, accuracy_te.item()*100/count


def evaluate_imagenet(args, net, image_syn, label_syn, testloader, normalize):
    
    trainloader = DataLoader(
        TensorDataset(image_syn, label_syn),
        batch_size=args.cdd_batch,
        shuffle=True,
        num_workers=0
    )
    resize = transforms.Resize(224)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.cdd_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.cdd_epoch // 3, 5 * args.cdd_epoch // 6], gamma=0.2)

    # train
    for _ in trange(args.cdd_epoch):        
        for x_tr, y_tr in trainloader:
            # data
            x_tr, y_tr = x_tr.to(args.cdd_device), y_tr.to(args.cdd_device)
            x_tr = DiffAugment(normalize(resize(x_tr)), args.cdd_dsa_strategy, param=args.cdd_dsa_param)
            
            # update
            optimizer.zero_grad()
            loss_tr = F.cross_entropy(net(x_tr), y_tr)
            loss_tr.backward()
            optimizer.step()
        
        # scheulder update
        scheduler.step()
    
    # test
    count = 0.0
    loss_te = 0.0
    accuracy_te = 0.0
    with torch.no_grad():
        for x_te, y_te in testloader:
            # data
            x_te, y_te = x_te.to(args.cdd_device), y_te.to(args.cdd_device)
            x_te = normalize(resize(x_te))

            # prediction
            y_te_pred = net(x_te)
            loss_te += F.cross_entropy(y_te_pred, y_te, reduction='sum')
            accuracy_te += torch.eq(y_te_pred.argmax(dim=-1), y_te).sum().float()
            count += x_te.shape[0]
    
    del net

    return loss_te.item()/count, accuracy_te.item()*100/count

def random_indices(y, nclass=10, intraclass=False, device='cuda'):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def evaluate_snu(args, model_eval, net, image_syn, label_syn, testloader, normalize):
    
    logger = Logger(
        exp_name=f"mixup_snu_{args.cdd_beta1}_{args.cdd_beta2}",
        save_dir=None,
        print_every=1,
        save_every=1,
        total_step=args.cdd_epoch,
        print_to_stdout=True,
        wandb_project_name=f"{model_eval}_debug_repeat",
        wandb_tags=[],
        wandb_config=args,
    )
    
    trainloader = DataLoader(
        TensorDataset(image_syn, label_syn),
        batch_size=args.cdd_batch,
        shuffle=True,
        num_workers=0
    )

    optimizer = torch.optim.SGD(net.parameters(), lr=args.cdd_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.cdd_epoch // 3, 5 * args.cdd_epoch // 6], gamma=0.2)

    # train
    logger.start()
    for epoch in range(1, args.cdd_epoch+1):
        for x_tr, y_tr in trainloader:
            optimizer.zero_grad()

            # data
            x_tr, y_tr = x_tr.to(args.cdd_device), y_tr.to(args.cdd_device)
            x_tr = DiffAugment(normalize(x_tr), args.cdd_dsa_strategy, param=args.cdd_dsa_param)

            # mixup
            r = np.random.rand(1)
            if r < 0.5:
                lam = np.random.beta(args.cdd_beta1, args.cdd_beta2)
                rand_index = random_indices(y_tr, nclass=args.cdd_num_classes)

                y_tr_b = y_tr[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(x_tr.size(), lam)
                x_tr[:, :, bbx1:bbx2, bby1:bby2] = x_tr[rand_index, :, bbx1:bbx2, bby1:bby2]
                ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_tr.size()[-1] * x_tr.size()[-2]))

                l_tr = net(x_tr)
                loss_tr = F.cross_entropy(l_tr, y_tr) * ratio + F.cross_entropy(l_tr, y_tr_b) * (1. - ratio)
            
            else:
                loss_tr = F.cross_entropy(net(x_tr), y_tr)     

            loss_tr.backward()
            optimizer.step()

            logger.meter('loss', 'train', loss_tr.item())

        if epoch % 20 == 0:
            # test
            count = 0.0
            loss_te = 0.0
            accuracy_te = 0.0
            with torch.no_grad():
                for x_te, y_te in testloader:
                    # data
                    x_te, y_te = x_te.to(args.cdd_device), y_te.to(args.cdd_device)
                    x_te = normalize(x_te)

                    # prediction
                    y_te_pred = net(x_te)
                    loss_te += F.cross_entropy(y_te_pred, y_te, reduction='sum')
                    accuracy_te += torch.eq(y_te_pred.argmax(dim=-1), y_te).sum().float()
                    count += x_te.shape[0]
            
            logger.meter('loss', 'test', loss_te.item()/count)
            logger.meter('accuracy', 'test', accuracy_te.item()*100/count)
        
        # scheulder update
        scheduler.step()

        logger.step()
    logger.finish()
    
    del net

    return loss_te.item()/count, accuracy_te.item()*100/count

def evaluate_intra_code(args, model_eval, net, generator, testloader, normalize):
    
    logger = Logger(
        exp_name=f"mixup_intra_code_{args.cdd_beta1}_{args.cdd_beta2}",
        save_dir=None,
        print_every=1,
        save_every=1,
        total_step=args.cdd_epoch,
        print_to_stdout=True,
        wandb_project_name=f"{model_eval}_debug_repeat",
        wandb_tags=[],
        wandb_config=args,
    )
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.cdd_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.cdd_epoch // 3, 5 * args.cdd_epoch // 6], gamma=0.2)

    import math
    iteration = math.ceil(args.cdd_num_classes*args.cdd_ipc*args.cdd_augment_factor*args.cdd_num_decoder / args.cdd_batch)

    # train
    logger.start()
    for epoch in range(1, args.cdd_epoch+1):
        for _ in range(iteration):
            optimizer.zero_grad()
            
            r = np.random.rand(1)
            with torch.no_grad():
                x_tr_list, y_tr_list = [], []
                for c in range(args.cdd_num_classes):
                    seed_vec_idx = torch.randperm(args.cdd_ipc*args.cdd_augment_factor)[:5]
                    seed_vec = generator.seed_vec[c][seed_vec_idx].clone().detach()
                    decoder_idx = torch.randperm(args.cdd_num_decoder)[:5]
                    decoders = []
                    for idx in decoder_idx:
                        decoders.append(generator.decoders[idx])

                    for decoder in decoders:
                        # mixup
                        if r < 0.0:
                            lam = np.random.beta(args.cdd_beta1, args.cdd_beta2)
                            seed_vec_ = seed_vec[torch.randperm(len(seed_vec))]
                            seed_vec = lam*seed_vec + (1.-lam)*seed_vec_
                            x_tr = decoder(seed_vec)
                            x_tr_list.append(x_tr)
                            y_tr = torch.LongTensor([c]*x_tr.shape[0]).to(x_tr.device)
                            y_tr_list.append(y_tr)
                        # no
                        else:
                            x_tr = decoder(seed_vec)
                            x_tr_list.append(x_tr)
                            y_tr = torch.LongTensor([c]*x_tr.shape[0]).to(x_tr.device)
                            y_tr_list.append(y_tr)
                x_tr, y_tr = torch.cat(x_tr_list, dim=0), torch.cat(y_tr_list, dim=0)

            # data
            x_tr = DiffAugment(normalize(x_tr), args.cdd_dsa_strategy, param=args.cdd_dsa_param)

            # update
            loss_tr = F.cross_entropy(net(x_tr), y_tr)
            loss_tr.backward() 
            optimizer.step()

            logger.meter('loss', 'train', loss_tr.item())

        if epoch % 20 == 0:
            # test
            count = 0.0
            loss_te = 0.0
            accuracy_te = 0.0
            with torch.no_grad():
                for x_te, y_te in testloader:
                    # data
                    x_te, y_te = x_te.to(args.cdd_device), y_te.to(args.cdd_device)
                    x_te = normalize(x_te)

                    # prediction
                    y_te_pred = net(x_te)
                    loss_te += F.cross_entropy(y_te_pred, y_te, reduction='sum')
                    accuracy_te += torch.eq(y_te_pred.argmax(dim=-1), y_te).sum().float()
                    count += x_te.shape[0]
            
            logger.meter('loss', 'test', loss_te.item()/count)
            logger.meter('accuracy', 'test', accuracy_te.item()*100/count)
        
        # scheulder update
        scheduler.step()

        logger.step()
    logger.finish()
    
    del net

    return loss_te.item()/count, accuracy_te.item()*100/count

def evaluate_inter_code(args, model_eval, net, generator, testloader, normalize):
    
    logger = Logger(
        exp_name=f"mixup_inter_code_{args.cdd_beta1}_{args.cdd_beta2}",
        save_dir=None,
        print_every=1,
        save_every=1,
        total_step=args.cdd_epoch,
        print_to_stdout=True,
        wandb_project_name=f"{model_eval}_debug_repeat",
        wandb_tags=[],
        wandb_config=args,
    )
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.cdd_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.cdd_epoch // 3, 5 * args.cdd_epoch // 6], gamma=0.2)

    import math
    iteration = math.ceil(args.cdd_num_classes*args.cdd_ipc*args.cdd_augment_factor*args.cdd_num_decoder / args.cdd_batch)

    # train
    logger.start()
    for epoch in range(1, args.cdd_epoch+1):
        for _ in range(iteration):
            optimizer.zero_grad()
            
            r = np.random.rand(1)
            with torch.no_grad():
                x_tr_list, y_tr_list, y_tr_b_list = [], [], []
                
                decoder_idx = torch.randperm(args.cdd_num_decoder)[:5]
                decoders = []
                for idx in decoder_idx:
                    decoders.append(generator.decoders[idx])

                seed_vec_list = []
                seed_y_list = []
                for c in range(args.cdd_num_classes):
                    seed_vec_idx = torch.randperm(args.cdd_ipc*args.cdd_augment_factor)[:5]
                    seed_vec = generator.seed_vec[c][seed_vec_idx].clone().detach()
                    seed_vec_list.append(seed_vec)
                    seed_y = torch.LongTensor([c]*seed_vec.shape[0]).to(seed_vec.device)
                    seed_y_list.append(seed_y)
                seed_vec = torch.cat(seed_vec_list, dim=0)
                seed_y = torch.cat(seed_y_list, dim=0)

                # mixup
                if r < 0.0:
                    lam = np.random.beta(args.cdd_beta1, args.cdd_beta2)
                    permutation = torch.randperm(len(seed_vec))

                    seed_vec_ = seed_vec[permutation]
                    seed_vec = lam*seed_vec + (1.-lam)*seed_vec_
                    for decoder in decoders:
                        x_tr = decoder(seed_vec)
                        x_tr_list.append(x_tr)
                        y_tr_list.append(seed_y)
                        y_tr_b_list.append(seed_y[permutation])
                    x_tr, y_tr, y_tr_b = torch.cat(x_tr_list, dim=0), torch.cat(y_tr_list, dim=0), torch.cat(y_tr_b_list, dim=0)
                # no
                else:
                    for decoder in decoders:
                        x_tr = decoder(seed_vec)
                        x_tr_list.append(x_tr)
                        y_tr_list.append(seed_y)
                    x_tr, y_tr = torch.cat(x_tr_list, dim=0), torch.cat(y_tr_list, dim=0)

            # data
            x_tr = DiffAugment(normalize(x_tr), args.cdd_dsa_strategy, param=args.cdd_dsa_param)

            if r < 0.0:
                l_tr = net(x_tr)
                loss_tr = F.cross_entropy(l_tr, y_tr) * lam + F.cross_entropy(l_tr, y_tr_b) * (1. - lam)
            else:
                loss_tr = F.cross_entropy(net(x_tr), y_tr)

            # update           
            loss_tr.backward() 
            optimizer.step()

            logger.meter('loss', 'train', loss_tr.item())

        if epoch % 20 == 0:
            # test
            count = 0.0
            loss_te = 0.0
            accuracy_te = 0.0
            with torch.no_grad():
                for x_te, y_te in testloader:
                    # data
                    x_te, y_te = x_te.to(args.cdd_device), y_te.to(args.cdd_device)
                    x_te = normalize(x_te)

                    # prediction
                    y_te_pred = net(x_te)
                    loss_te += F.cross_entropy(y_te_pred, y_te, reduction='sum')
                    accuracy_te += torch.eq(y_te_pred.argmax(dim=-1), y_te).sum().float()
                    count += x_te.shape[0]
            
            logger.meter('loss', 'test', loss_te.item()/count)
            logger.meter('accuracy', 'test', accuracy_te.item()*100/count)
        
        # scheulder update
        scheduler.step()

        logger.step()
    logger.finish()
    
    del net

    return loss_te.item()/count, accuracy_te.item()*100/count

def evaluate_intra_decoder(args, model_eval, net, generator, testloader, normalize):
    
    logger = Logger(
        exp_name=f"mixup_intra_decoder_{args.cdd_beta1}_{args.cdd_beta2}",
        save_dir=None,
        print_every=1,
        save_every=1,
        total_step=args.cdd_epoch,
        print_to_stdout=True,
        wandb_project_name=f"{model_eval}_debug_repeat",
        wandb_tags=[],
        wandb_config=args,
    )
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.cdd_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.cdd_epoch // 3, 5 * args.cdd_epoch // 6], gamma=0.2)

    import math
    iteration = math.ceil(args.cdd_num_classes*args.cdd_ipc*args.cdd_augment_factor*args.cdd_num_decoder / args.cdd_batch)
    from copy import deepcopy
    # train
    logger.start()
    for epoch in range(1, args.cdd_epoch+1):
        for _ in range(iteration):
            optimizer.zero_grad()
            
            r = np.random.rand(1)
            with torch.no_grad():
                x_tr_list, y_tr_list = [], []
                for c in range(args.cdd_num_classes):
                    seed_vec_idx = torch.randperm(args.cdd_ipc*args.cdd_augment_factor)[:5]
                    seed_vec = generator.seed_vec[c][seed_vec_idx]
                    decoder_idx = torch.randperm(args.cdd_num_decoder)[:5]
                    decoders = []
                    for idx in decoder_idx:
                        decoders.append(generator.decoders[idx])

                    for i in range(len(decoders)):
                        # mixup
                        if r < 0.5:
                            lam = np.random.beta(args.cdd_beta1, args.cdd_beta2)
                            decoder = deepcopy(decoders[0]).to(args.cdd_device)
                            j = i if i == (len(decoders) - 1) else 0
                            for w, w1, w2 in zip(decoder.parameters(), decoders[i].parameters(), decoders[j].parameters()):
                                w.data.copy_(lam*w1.data+(1.-lam)*w2.data)
                            x_tr = decoder(seed_vec)
                            x_tr_list.append(x_tr)
                            y_tr = torch.LongTensor([c]*x_tr.shape[0]).to(x_tr.device)
                            y_tr_list.append(y_tr)
                        # no
                        else:
                            x_tr = decoders[i](seed_vec)
                            x_tr_list.append(x_tr)
                            y_tr = torch.LongTensor([c]*x_tr.shape[0]).to(x_tr.device)
                            y_tr_list.append(y_tr)
                x_tr, y_tr = torch.cat(x_tr_list, dim=0), torch.cat(y_tr_list, dim=0)

            # data
            x_tr = DiffAugment(normalize(x_tr), args.cdd_dsa_strategy, param=args.cdd_dsa_param)

            # update
            loss_tr = F.cross_entropy(net(x_tr), y_tr) 
            loss_tr.backward() 
            optimizer.step()

            logger.meter('loss', 'train', loss_tr.item())

        if epoch % 20 == 0:
            # test
            count = 0.0
            loss_te = 0.0
            accuracy_te = 0.0
            with torch.no_grad():
                for x_te, y_te in testloader:
                    # data
                    x_te, y_te = x_te.to(args.cdd_device), y_te.to(args.cdd_device)
                    x_te = normalize(x_te)

                    # prediction
                    y_te_pred = net(x_te)
                    loss_te += F.cross_entropy(y_te_pred, y_te, reduction='sum')
                    accuracy_te += torch.eq(y_te_pred.argmax(dim=-1), y_te).sum().float()
                    count += x_te.shape[0]
            
            logger.meter('loss', 'test', loss_te.item()/count)
            logger.meter('accuracy', 'test', accuracy_te.item()*100/count)
        
        # scheulder update
        scheduler.step()

        logger.step()
    logger.finish()
    
    del net

    return loss_te.item()/count, accuracy_te.item()*100/count

def evaluate_orig(args, model_eval, net, image_syn, label_syn, testloader, normalize):
    
    logger = Logger(
        exp_name="orig",
        save_dir=None,
        print_every=1,
        save_every=1,
        total_step=args.cdd_epoch,
        print_to_stdout=True,
        wandb_project_name=f"{model_eval}_debug_repeat",
        wandb_tags=[],
        wandb_config=args,
    )
    
    trainloader = DataLoader(
        TensorDataset(image_syn, label_syn),
        batch_size=args.cdd_batch,
        shuffle=True,
        num_workers=0
    )

    optimizer = torch.optim.SGD(net.parameters(), lr=args.cdd_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.cdd_epoch // 3, 5 * args.cdd_epoch // 6], gamma=0.2)

    # train
    logger.start()
    for epoch in range(1, args.cdd_epoch+1):
        for x_tr, y_tr in trainloader:
            # data
            x_tr, y_tr = x_tr.to(args.cdd_device), y_tr.to(args.cdd_device)
            x_tr = DiffAugment(normalize(x_tr), args.cdd_dsa_strategy, param=args.cdd_dsa_param)
            
            # update
            optimizer.zero_grad()
            loss_tr = F.cross_entropy(net(x_tr), y_tr)            
            loss_tr.backward()
            optimizer.step()

            #print("here")
            #print(loss_tr)
            logger.meter('loss', 'train', loss_tr.item())
    
        if epoch % 20 == 0:
            # test
            count = 0.0
            loss_te = 0.0
            accuracy_te = 0.0
            with torch.no_grad():
                for x_te, y_te in testloader:
                    # data
                    x_te, y_te = x_te.to(args.cdd_device), y_te.to(args.cdd_device)
                    x_te = normalize(x_te)

                    # prediction
                    y_te_pred = net(x_te)
                    loss_te += F.cross_entropy(y_te_pred, y_te, reduction='sum')
                    accuracy_te += torch.eq(y_te_pred.argmax(dim=-1), y_te).sum().float()
                    count += x_te.shape[0]
            
            logger.meter('loss', 'test', loss_te.item()/count)
            logger.meter('accuracy', 'test', accuracy_te.item()*100/count)
        
        # scheulder update
        scheduler.step()

        logger.step()
    logger.finish()
    
    del net

    return loss_te.item()/count, accuracy_te.item()*100/count

class Logger:
    def __init__(
        self,
        exp_name,
        exp_suffix="",
        save_dir=None,
        print_every=100,
        save_every=100,
        total_step=0,
        print_to_stdout=True,
        wandb_project_name=None,
        wandb_tags=[],
        wandb_config=None,
    ):
        if save_dir is not None:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

        self.print_every = print_every
        self.save_every = save_every
        self.step_count = 0
        self.total_step = total_step
        self.print_to_stdout = print_to_stdout

        self.writer = None
        self.start_time = None
        self.groups = dict()
        self.models_to_save = dict()
        self.objects_to_save = dict()
        if "/" in exp_suffix:
            exp_suffix = "_".join(exp_suffix.split("/")[:-1])
        wandb.init(entity="distill-me-all", project=wandb_project_name, name=exp_name + "_" + exp_suffix, tags=wandb_tags, reinit=True)
        wandb.config.update(wandb_config)

    def register_model_to_save(self, model, name):
        assert name not in self.models_to_save.keys(), "Name is already registered."

        self.models_to_save[name] = model

    def register_object_to_save(self, object, name):
        assert name not in self.objects_to_save.keys(), "Name is already registered."

        self.objects_to_save[name] = object

    def step(self):
        self.step_count += 1
        if self.step_count % self.print_every == 0:
            if self.print_to_stdout:
                self.print_log(self.step_count, self.total_step, elapsed_time=datetime.now() - self.start_time)
            self.write_log(self.step_count)

        if self.step_count % self.save_every == 0:
            #self.save_models(self.step_count)
            #self.save_objects(self.step_count)
            self.save_models()
            self.save_objects()

    def meter(self, group_name, log_name, value):
        if group_name not in self.groups.keys():
            self.groups[group_name] = dict()

        if log_name not in self.groups[group_name].keys():
            self.groups[group_name][log_name] = Accumulator()

        self.groups[group_name][log_name].update_state(value)

    def reset_state(self):
        for _, group in self.groups.items():
            for _, log in group.items():
                log.reset_state()

    def print_log(self, step, total_step, elapsed_time=None):
        print(f"[Step {step:5d}/{total_step}]", end="  ")

        for name, group in self.groups.items():
            print(f"({name})", end="  ")
            for log_name, log in group.items():
                res = log.result()
                if res is None:
                    continue

                if "acc" in log_name.lower():
                    print(f"{log_name} {res:.2f}", end=" | ")
                else:
                    print(f"{log_name} {res:.4f}", end=" | ")

        if elapsed_time is not None:
            print(f"(Elapsed time) {elapsed_time}")
        else:
            print()

    def write_log(self, step):
        log_dict = {}
        for group_name, group in self.groups.items():
            for log_name, log in group.items():
                res = log.result()
                if res is None:
                    continue
                log_dict["{}/{}".format(group_name, log_name)] = res
        wandb.log(log_dict, step=step)

        self.reset_state()

    def write_log_individually(self, name, value, step):
        if self.use_wandb:
            wandb.log({name: value}, step=step)
        else:
            self.writer.add_scalar(name, value, step=step)

    def save_models(self, suffix=None):
        if self.save_dir is None:
            return

        for name, model in self.models_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"{_name}.pth"))

            if self.print_to_stdout:
                logging.info(f"{name} is saved to {self.save_dir}")

    def save_objects(self, suffix=None):
        if self.save_dir is None:
            return

        for name, obj in self.objects_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(obj, os.path.join(self.save_dir, f"{_name}.pth"))

            if self.print_to_stdout:
                logging.info(f"{name} is saved to {self.save_dir}")

    def start(self):
        if self.print_to_stdout:
            logging.info("Training starts!")
        #self.save_models("init")
        #self.save_objects("init")
        self.start_time = datetime.now()

    def finish(self):
        if self.step_count % self.save_every != 0:
            self.save_models(self.step_count)
            self.save_objects(self.step_count)

        if self.print_to_stdout:
            logging.info("Training is finished!")
        wandb.join()

class Accumulator:
    def __init__(self):
        self.data = 0
        self.num_data = 0

    def reset_state(self):
        self.data = 0
        self.num_data = 0

    def update_state(self, tensor):
        with torch.no_grad():
            self.data += tensor
            self.num_data += 1

    def result(self):
        if self.num_data == 0:
            return None        
        data = self.data.item() if hasattr(self.data, 'item') else self.data
        return float(data) / self.num_data
