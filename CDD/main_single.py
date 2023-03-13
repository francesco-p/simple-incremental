import os 
import time
import random
import argparse
import numpy as np
from tqdm import trange
import time
import torch
from torchvision.utils import save_image, make_grid

from kfs_models.wrapper import get_model
from kfs_utils import evaluate, Logger, get_linear_schedule_warmup_constant, default_args
from generator import SyntheticImageGenerator

from torch.utils.data import DataLoader
from data import get_dataset, DiffAugment, ParamDiffAug, get_all_images, TensorDataset
import timm
from opt import OPT


def main(args, dset_train, dset_test):
    args.cdd_device = torch.device(f"cuda:{args.cdd_gpu_id}")
    args.cdd_ae_path = f'CDD/pretrained_ae/{OPT.dataset}_{args.cdd_ipc}_{args.cdd_num_seed_vec}_{args.cdd_num_decoder}_seed_{OPT.seed}.pth'
    #args.cdd_ae_path = f'CDD/pretrained_ae/CIFAR100_1_10_10_default.pth'
    if args.cdd_exp_name is None:
        args.cdd_exp_name = f'{args.cdd_model}_{args.cdd_ipc}_{args.cdd_num_seed_vec}_{args.cdd_num_decoder}'
        if args.cdd_linear_schedule:
            args.cdd_exp_name += "_linear_schedule"
    print(args.cdd_exp_name)

    args.cdd_dsa_param = ParamDiffAug()
    if args.cdd_dataset == 'SVHN':
        args.cdd_dsa_strategy = 'color_crop_cutout_scale_rotate'
    else:
        args.cdd_dsa_strategy = 'color_crop_cutout_flip_scale_rotate'

    ''' data set '''
    channel, im_size, num_classes, normalize, images_all, indices_class, testloader = get_dataset(args.cdd_dataset, args.cdd_data_path, dset_train, dset_test)
    
    ''' model eval pool'''
    if args.cdd_model_eval_pool is None:
        args.cdd_model_eval_pool = [args.cdd_model]
    else:
        args.cdd_model_eval_pool = args.cdd_model_eval_pool.split("/")
    accs_all_exps = dict() # record performances of all experiments
    for key in args.cdd_model_eval_pool:
        accs_all_exps[key] = []
    data_save = []

    ''' save path '''
    save_path = f'{args.cdd_save_path}/{args.cdd_dataset}_{OPT.seed}/{args.cdd_exp_name}/task{args.cdd_name_folder}'
    os.makedirs(save_path, exist_ok=True)

    ''' initialize '''
    generator = SyntheticImageGenerator(
            num_classes, im_size, args.cdd_num_seed_vec, args.cdd_num_decoder, args.cdd_hdims,
            args.cdd_kernel_size, args.cdd_stride, args.cdd_padding).to(args.cdd_device)
    optimizer_gen = torch.optim.Adam(
        [{'params': generator.seed_vec, 'lr': args.cdd_lr_seed_vec},
        {'params': generator.decoders.parameters(), 'lr': args.cdd_lr_decoder}])
    if args.cdd_linear_schedule:
        scheduler_gen = get_linear_schedule_warmup_constant(
            optimizer_gen, 2000, args.cdd_iteration)
    else:
        scheduler_gen = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_gen, milestones=args.cdd_lr_iteration, gamma=0.2)

    ''' load autoencoder '''
    generator.load_state_dict(torch.load(args.cdd_ae_path))
    
    del generator.encoders
    generator.broadcast_decoder()

    buffer = []
    
    #normalize = lambda x: x
    print(f"Beginning training")
    for it in trange(1, args.cdd_iteration+1):
        if it % 100 == 0:
            print(f"Iteration {it}/{args.cdd_iteration}")
            print(f"Loss AVG class: {loss_avg}")
            
        #print(f"iteration n {it}")
        net = get_model(args, args.cdd_model, channel, num_classes, im_size).to(f"cuda:{args.cdd_gpu_id}") # get a random model
        net = net.to(f"cuda:{args.cdd_gpu_id}")
        net.eval()
        for param in list(net.parameters()):
            param.requires_grad = False

        if args.cdd_model == "efficientnet":
            embed = timm.create_model('efficientnet_b0', num_classes = 0, pretrained = False).to(f"cuda:{args.cdd_gpu_id}")
        elif args.cdd_model == "dla46x_c":
            embed = timm.create_model('dla46x_c', num_classes = 0, pretrained = False).to(f"cuda:{args.cdd_gpu_id}")
            for param in list(embed.parameters()):
                param.requires_grad = False
        else:
            embed = net.embed        

        if hasattr(net, "classifier"):
            del net.classifier
        if hasattr(net, 'fc'):
            del net.fc

        ''' update synthetic data '''

        #print("A")
        n_step = 1
        #for index in range(n_step):
        loss = 0.0
        for c in range(num_classes):
            seed = int(time.time() * 1000) % 100000
        
            all_img_real = get_all_images(images_all, indices_class, c)
            step_size = len(all_img_real)//n_step
            #all_img_real = all_img_real[index*step_size : (index+1)*step_size]
            all_img_real = DiffAugment(normalize(all_img_real), args.cdd_dsa_strategy, seed=seed, param=args.cdd_dsa_param)
            all_img_real = all_img_real.to(f"cuda:{args.cdd_gpu_id}")
            #all_img_real = torch.rand(3,3,128,128).to(f"cuda:{args.cdd_gpu_id}")
            output_real_mean = embed(all_img_real).mean(dim=0)
            
            # syn
            img_syn = generator.get_sample(c)[0]
            img_syn = DiffAugment(normalize(img_syn), args.cdd_dsa_strategy, seed=seed, param=args.cdd_dsa_param)
            #img_syn = torch.rand(3,3,128,128).to(f"cuda:{args.cdd_gpu_id}")
            output_syn_mean = torch.mean(embed(img_syn), dim=0)
        
            # compute loss
            loss += torch.sum((output_real_mean - output_syn_mean)**2)
        #print("B")
        # update
        optimizer_gen.zero_grad()
        loss.backward()
        optimizer_gen.step()
        scheduler_gen.step()
        loss_avg = loss.item() / num_classes  



        ''' Evaluate synthetic data '''
        if it % args.cdd_eval_every == 0:
            image_syn, label_syn = generator.get_all_cpu()
            image_syn, label_syn = image_syn.detach(), label_syn.detach()
            print(image_syn.shape, "SHAAAAAAPPEEEE")
            ''' Evaluate all model_eval '''
            if not args.cdd_not_eval:
                for model_eval in args.cdd_model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.cdd_model, model_eval, it))
                    accs = []
                    num_eval = args.cdd_num_eval if it == args.cdd_iteration else 1
                    for _ in range(num_eval):
                        net_eval = get_model(args, model_eval, channel, num_classes, im_size).to(args.cdd_device) # get a random model
                        _, acc = evaluate(args, net_eval, image_syn, label_syn, testloader, normalize)
                        accs.append(acc)

                    if it == args.cdd_iteration: # record the final results
                        accs_all_exps[model_eval] += accs


                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

            ''' visualize and save '''
            save_name = os.path.join(save_path, f'{it}.png')
            grid = make_grid(image_syn, nrow=args.cdd_num_seed_vec*args.cdd_num_decoder)
            save_image(image_syn, save_name, nrow=args.cdd_num_seed_vec*args.cdd_num_decoder)

            #data_save.append([image_syn, label_syn])
            #torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(save_path, f'res_{it}.pth'))
            torch.save(generator, os.path.join(save_path, f'generator_{it}.pth'))
            
        if it == args.cdd_iteration: # only record the final results
            image_syn, label_syn = generator.get_all_cpu()
            image_syn, label_syn = image_syn.detach(), label_syn.detach()

            data_save.append([image_syn, label_syn])
            torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(save_path, f'res.pth'))
            torch.save(generator, os.path.join(save_path, f'generator.pth'))


    print('\n==================== Final Results ====================\n')
    for key in args.cdd_model_eval_pool:
        accs = accs_all_exps[key]
        print('Train on %s, Evaluate on %s for %d: mean  = %.2f%%  std = %.2f%%'%(args.cdd_model, key, len(accs), np.mean(accs), np.std(accs)))
        with open(f'{save_path}/{key}_final_results.txt', 'w') as f:
            f.write(f'mean = {np.mean(accs)}, std = {np.std(accs)}')

