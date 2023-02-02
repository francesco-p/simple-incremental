import os
import time
import random
import argparse
import numpy as np
from tqdm import trange

import torch
from torchvision.utils import save_image, make_grid

from data import get_dataset, DiffAugment, ParamDiffAug
from kfs_models.wrapper import get_model
from kfs_utils import evaluate, Logger, get_linear_schedule_warmup_constant, default_args
from generator import SyntheticImageGenerator


def main(args, dset):
    args.device = torch.device(f"cuda:{args.gpu_id}")
    args.ae_path = f'CDD/pretrained_ae/{args.dataset}_{args.ipc}_{args.num_seed_vec}_{args.num_decoder}_default.pth'

    if args.exp_name is None:
        args.exp_name = f'{args.model}_{args.ipc}_{args.num_seed_vec}_{args.num_decoder}'
        if args.linear_schedule:
            args.exp_name += "_linear_schedule"
    print(args.exp_name)

    args.dsa_param = ParamDiffAug()
    if args.dataset == 'SVHN':
        args.dsa_strategy = 'color_crop_cutout_scale_rotate'
    else:
        args.dsa_strategy = 'color_crop_cutout_flip_scale_rotate'

    ''' feature path'''
    args.feature_path = f'{args.feature_path}/{args.dataset}/{args.model}{args.name_folder}'
    print(args.feature_path, args.model)
    ''' data set '''
    channel, im_size, num_classes, normalize, _, _, testloader = get_dataset(args.dataset, args.data_path, dset)

    ''' model eval pool'''
    if args.model_eval_pool is None:
        args.model_eval_pool = [args.model]
    else:
        args.model_eval_pool = args.model_eval_pool.split("_")
    accs_all_exps = dict() # record performances of all experiments
    for key in args.model_eval_pool:
        accs_all_exps[key] = []
    data_save = []

    ''' save path '''
    save_path = f'{args.save_path}/{args.dataset}/{args.exp_name}/task{args.name_folder}'
    os.makedirs(save_path, exist_ok=True)

    ''' initialize '''
    generator = SyntheticImageGenerator(
            num_classes, im_size, args.num_seed_vec, args.num_decoder, args.hdims,
            args.kernel_size, args.stride, args.padding).to(args.device)
    optimizer_gen = torch.optim.Adam(
        [{'params': generator.seed_vec, 'lr': args.lr_seed_vec},
        {'params': generator.decoders.parameters(), 'lr': args.lr_decoder}])
    if args.linear_schedule:
        scheduler_gen = get_linear_schedule_warmup_constant(
            optimizer_gen, 2000, args.iteration)
    else:
        scheduler_gen = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_gen, milestones=args.lr_iteration, gamma=0.2)

    ''' load autoencoder '''
    generator.load_state_dict(torch.load(args.ae_path))
    
    del generator.encoders
    generator.broadcast_decoder()

    buffer = []
    for it in range(1, args.iteration+1):
        ''' buffer '''
        if len(buffer) == 0:
            print("-------------------------Buffer Loading-------------------------")
            for i in trange(args.buffer_every):
                buffer_it = it + i
                buffer.append(torch.load(f"{args.feature_path}/dsa_{buffer_it}.pth", map_location='cpu'))
            print(f"#########################################\nstep {it}/{args.iteration}")
                
        ''' get feature dict '''
        features_dict = buffer.pop(0)

        ''' get model '''
        net = get_model(args, args.model, channel, num_classes, im_size)
        if hasattr(net, "classifier"):
            del net.classifier
        if hasattr(net, 'fc'):
            del net.fc
        #print(f"features_dict path: {args.feature_path}{args.name_folder}/dsa_{buffer_it}.pth")
        net.load_state_dict(features_dict['state_dict'], strict=False)
        net = net.to(args.device)
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False
        if args.model != "efficientnet":
            embed = net.embed
        else:
            embed = torch.nn.Sequential(*[l for l in list(net.children())[:6]])
        ''' update synthetic data '''
        loss = 0.0
        for c in range(num_classes): 
            seed = features_dict[c]['seed']

            # real
            output_real_mean = features_dict[c]['mean'].to(args.device)

            # syn
            img_syn = generator.get_sample(c)[0]
            img_syn = DiffAugment(normalize(img_syn), args.dsa_strategy, seed=seed, param=args.dsa_param)
            
            output_syn_mean = torch.mean(embed(img_syn), dim=0)

            # compute loss
            loss += torch.sum((output_real_mean - output_syn_mean)**2)

        # update
        optimizer_gen.zero_grad()
        loss.backward()
        optimizer_gen.step()
        scheduler_gen.step()
        loss_avg = loss.item() / num_classes  



        ''' Evaluate synthetic data '''
        if it % args.eval_every == 0:
            image_syn, label_syn = generator.get_all_cpu()
            image_syn, label_syn = image_syn.detach(), label_syn.detach()
            print(image_syn.shape, "SHAAAAAAPPEEEE")
            ''' Evaluate all model_eval '''
            if not args.not_eval:
                for model_eval in args.model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    accs = []
                    num_eval = args.num_eval if it == args.iteration else 1
                    for _ in range(num_eval):
                        net_eval = get_model(args, model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        _, acc = evaluate(args, net_eval, image_syn, label_syn, testloader, normalize)
                        accs.append(acc)

                    if it == args.iteration: # record the final results
                        accs_all_exps[model_eval] += accs


                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

            ''' visualize and save '''
            save_name = os.path.join(save_path, f'{it}.png')
            grid = make_grid(image_syn, nrow=args.num_seed_vec*args.num_decoder)
            save_image(image_syn, save_name, nrow=args.num_seed_vec*args.num_decoder)

        if it == args.iteration: # only record the final results
            image_syn, label_syn = generator.get_all_cpu()
            image_syn, label_syn = image_syn.detach(), label_syn.detach()

            data_save.append([image_syn, label_syn])
            torch.save({'data': data_save, 'generator': generator, 'accs_all_exps': accs_all_exps, }, os.path.join(save_path, f'res.pth'))


    print('\n==================== Final Results ====================\n')
    for key in args.model_eval_pool:
        accs = accs_all_exps[key]
        print('Train on %s, Evaluate on %s for %d: mean  = %.2f%%  std = %.2f%%'%(args.model, key, len(accs), np.mean(accs), np.std(accs)))
        with open(f'{args.save_path}/{args.dataset}/{args.exp_name}/{key}_final_results.txt', 'w') as f:
            f.write(f'mean = {np.mean(accs)}, std = {np.std(accs)}')

