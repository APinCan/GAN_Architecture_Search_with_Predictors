import logging
import operator
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from torchvision.utils import make_grid
from tqdm import tqdm
import torch.nn.functional as F

from utils.fid_score import calculate_fid_given_paths
from utils.inception_score import get_inception_score

logger = logging.getLogger(__name__)
device = torch.device("cuda:0")


def train_qin(args, gen_net: nn.Module, dis_net: nn.Module, g_loss_history, d_loss_history, gen_optimizer
                 , dis_optimizer, train_loader, cur_stage, smooth=False, WARMUP=False):
    dynamic_reset = False
    step = 0
    gen_step = 0
    # train mode
    gen_net.train()
    dis_net.train()

    datasetsize = len(train_loader)
    if cur_stage == 0:
        shared_epoch = 1
    elif cur_stage == 1:
        shared_epoch = 1
    else:
        shared_epoch = 1
    for epoch in range(shared_epoch):
        for iter_idx, (imgs, _) in enumerate(train_loader):
            progress = (epoch * datasetsize + iter_idx + 1) / (datasetsize * args.shared_epoch)
            
            
            dis_net.cur_stage = cur_stage
            # Adversarial ground truths
            real_imgs = imgs.type(torch.cuda.FloatTensor)

            # Sample noise as generator input
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            dis_optimizer.zero_grad()

            real_validity = dis_net(real_imgs)
            if smooth:
                fake_imgs = gen_net(z, progress).detach()
            else:
                fake_imgs = gen_net(z, 1.).detach()
            assert fake_imgs.size() == real_imgs.size(), print(f'fake image size is {fake_imgs.size()}, '
                                                               f'while real image size is {real_imgs.size()}')

            fake_validity = dis_net(fake_imgs)

            # cal loss
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            d_loss.backward()
            dis_optimizer.step()

            # add to window
            d_loss_history.push(d_loss.item())

            # -----------------
            #  Train Generator
            # -----------------
            if step % args.n_critic == 0:
                gen_optimizer.zero_grad()

                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
                # Smooth is disabled for our final submission. We leave it here for you to play with. 
                if smooth:
                    gen_imgs = gen_net(gen_z, progress)
                else:
                    gen_imgs = gen_net(gen_z, 1.)
                fake_validity = dis_net(gen_imgs)

                # cal loss
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                gen_optimizer.step()

                # add to window
                g_loss_history.push(g_loss.item())
                gen_step += 1

            # verbose
            """
            if gen_step and iter_idx % args.print_freq == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                    (epoch, args.shared_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(),
                     g_loss.item()))
            """
            # check window
            if g_loss_history.is_full():
                if g_loss_history.get_var() < args.dynamic_reset_threshold \
                        or d_loss_history.get_var() < args.dynamic_reset_threshold:
                    dynamic_reset = True
                    logger.info("=> dynamic resetting triggered")
                    g_loss_history.clear()
                    d_loss_history.clear()
                    return dynamic_reset

            step += 1


    return dynamic_reset


def only_get_is(args, gen_net: nn.Module, num_img,z_numpy=None, get_is_score=True, default=True):
    # eval mode
    gen_net = gen_net.eval()
    eval_iter = num_img // args.eval_batch_size 
    img_list = list()
    state_list = list()
    for i in range(eval_iter):
        # We use a fixed set of random seeds for the reward and progressive states in the search stage to stabalize the training
        np.random.seed(i)
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
        # Generate a batch of images
        gen_imgs, gen_states = gen_net(z, eval=True) # when cnn torch.Size([100, 128, 1, 1])
        gen_imgs2 = gen_imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1)
        # gen_imgs2 = gen_imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0)
        img_list.extend(list(gen_imgs2.to('cpu',torch.uint8).numpy()))

        if default:
            # vector settting
            state_list.extend(list(gen_states.to('cpu').numpy()))
        else:
            # cnn setting
            state_list.extend(list(gen_states.to('cpu').detach().numpy()))


    if default:
        # vector setting
        state = list(np.mean(state_list, axis=0).flatten())
    else:
        # cnn setting
        state = list(np.mean(state_list, axis=0))
        state = [list(s) for s in state]

    if not get_is_score:
        return state
    # get inception score
    logger.info('calculate Inception score...')
    is_time = time.time()
    mean, std, scores = get_inception_score(img_list)
    is_time = time.time() - is_time

    return mean


def get_is(args, gen_net: nn.Module, num_img,z_numpy=None, get_is_score=True, default=True):
    # eval mode
    gen_net = gen_net.eval()
    eval_iter = num_img // args.eval_batch_size 
    img_list = list()
    state_list = list()
    for i in range(eval_iter):
        # We use a fixed set of random seeds for the reward and progressive states in the search stage to stabalize the training
        np.random.seed(i)
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
        # Generate a batch of images
        gen_imgs, gen_states = gen_net(z, eval=True) # when cnn torch.Size([100, 128, 1, 1])
        gen_imgs2 = gen_imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1)
        # gen_imgs2 = gen_imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0)
        img_list.extend(list(gen_imgs2.to('cpu',torch.uint8).numpy()))

        if default:
            # vector settting
            state_list.extend(list(gen_states.to('cpu').numpy()))
        else:
            # cnn setting
            state_list.extend(list(gen_states.to('cpu').detach().numpy()))


    if default:
        # vector setting
        state = list(np.mean(state_list, axis=0).flatten())
    else:
        # cnn setting
        state = list(np.mean(state_list, axis=0))
        state = [list(s) for s in state]


    if not get_is_score:
        return state
    # get inception score
    logger.info('calculate Inception score...')
    mean, std, scores = get_inception_score(img_list)
    logger.info('=> calculate fid score')
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.system('rm -rf {}'.format(fid_buffer_dir))
    os.makedirs(fid_buffer_dir, exist_ok=True)
    for img_idx, img in enumerate(img_list):
        if img_idx < 5000:
            file_name = os.path.join(fid_buffer_dir, f'iter0_b{img_idx}.png')
            imsave(file_name, img)
    fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    assert os.path.exists(fid_stat)
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    return mean, fid_score, state


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict, clean_dir=True):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    if clean_dir:
        os.system('rm -r {}'.format(fid_buffer_dir))
    else:
        logger.info(f'=> sampled images are saved to {fid_buffer_dir}')

    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten



def arch2matrix(arch, cur_stage):
    OPERATION_INDEX = {"input": 0, "bn_batch":1, "in_batch": 2, "bilinear_up": 3, "nearest_up": 4,
                "deconv_up":5, "conv":6, "output":7}

    # arch = arch.cpu().numpy()
    # cell1
    if cur_stage==0:
        # input(output), 3, 2, output
        operation_matrix = np.zeros((7, 8))
        adjacent_matrix = np.zeros((7, 7))

        operation_matrix[0, OPERATION_INDEX["input"]] = 1
        operation_matrix[6, OPERATION_INDEX["output"]] = 1
        if arch[0]==0: # post conv
            if arch[1]==0: # no batch
                pass
            elif arch[1]==1: # bn
                operation_matrix[3, OPERATION_INDEX["bn_batch"]] = 1
                operation_matrix[5, OPERATION_INDEX["bn_batch"]] = 1
            elif arch[1]==2: # in
                operation_matrix[3, OPERATION_INDEX["in_batch"]] = 1
                operation_matrix[5, OPERATION_INDEX["in_batch"]] = 1
            if arch[2]==0: # bilinear
                operation_matrix[1, OPERATION_INDEX["bilinear_up"]] = 1
            elif arch[2]==1: # nearest
                operation_matrix[1, OPERATION_INDEX["nearest_up"]] = 1
            elif arch[2]==2: # deconv
                operation_matrix[1, OPERATION_INDEX["deconv_up"]] = 1
            operation_matrix[2, OPERATION_INDEX["conv"]] = 1
            operation_matrix[4, OPERATION_INDEX["conv"]] = 1
        elif arch[0]==1: # pre conv
            if arch[1]==0: # no batch
                pass
            elif arch[1]==1: # bn
                operation_matrix[1, OPERATION_INDEX["bn_batch"]] = 1
                operation_matrix[4, OPERATION_INDEX["bn_batch"]] = 1
            elif arch[1]==2: # in
                operation_matrix[1, OPERATION_INDEX["in_batch"]] = 1
                operation_matrix[4, OPERATION_INDEX["in_batch"]] = 1

            if arch[2]==0: # bilinear
                operation_matrix[2, OPERATION_INDEX["bilinear_up"]] = 1
            elif arch[2]==1: # nearest
                operation_matrix[2, OPERATION_INDEX["nearest_up"]] = 1
            elif arch[2]==2: # deconv
                operation_matrix[2, OPERATION_INDEX["deconv_up"]] = 1
            operation_matrix[3, OPERATION_INDEX["conv"]] = 1
            operation_matrix[5, OPERATION_INDEX["conv"]] = 1
        adjacent_matrix[0, 1] = 1
        adjacent_matrix[1, 2] = 1
        adjacent_matrix[2, 3] = 1
        adjacent_matrix[3, 4] = 1
        adjacent_matrix[4, 5] = 1
        adjacent_matrix[5, 6] = 1
        if arch[3]==1: # shortcut true
            adjacent_matrix[0, 6] = 1

    # cell2
    elif cur_stage==1:
        # input(output), input(skip), 3, 2, output
        operation_matrix = np.zeros((8, 8))
        adjacent_matrix = np.zeros((8, 8))

        operation_matrix[0:2, OPERATION_INDEX["input"]] = 1
        operation_matrix[7, OPERATION_INDEX["output"]] = 1
        if arch[0]==0: # post conv
            if arch[1]==0: # no batch
                pass
            elif arch[1]==1: # bn
                operation_matrix[4, OPERATION_INDEX["bn_batch"]] = 1
                operation_matrix[6, OPERATION_INDEX["bn_batch"]] = 1
            elif arch[1]==2: # in
                operation_matrix[4, OPERATION_INDEX["in_batch"]] = 1
                operation_matrix[6, OPERATION_INDEX["in_batch"]] = 1
            if arch[2]==0: # bilinear
                operation_matrix[2, OPERATION_INDEX["bilinear_up"]] = 1
            elif arch[2]==1: # nearest
                operation_matrix[2, OPERATION_INDEX["nearest_up"]] = 1
            elif arch[2]==2: # deconv
                operation_matrix[2, OPERATION_INDEX["deconv_up"]] = 1
            operation_matrix[3, OPERATION_INDEX["conv"]] = 1
            operation_matrix[5, OPERATION_INDEX["conv"]] = 1
        elif arch[0]==1: # pre conv
            if arch[1]==0: # no batch
                pass
            elif arch[1]==1: # bn
                operation_matrix[2, OPERATION_INDEX["bn_batch"]] = 1
                operation_matrix[5, OPERATION_INDEX["bn_batch"]] = 1
            elif arch[1]==2: # in
                operation_matrix[2, OPERATION_INDEX["in_batch"]] = 1
                operation_matrix[5, OPERATION_INDEX["in_batch"]] = 1

            if arch[2]==0: # bilinear
                operation_matrix[3, OPERATION_INDEX["bilinear_up"]] = 1
            elif arch[2]==1: # nearest
                operation_matrix[3, OPERATION_INDEX["nearest_up"]] = 1
            elif arch[2]==2: # deconv
                operation_matrix[3, OPERATION_INDEX["deconv_up"]] = 1
            operation_matrix[4, OPERATION_INDEX["conv"]] = 1
            operation_matrix[6, OPERATION_INDEX["conv"]] = 1
        adjacent_matrix[0, 2] = 1
        adjacent_matrix[2, 3] = 1
        adjacent_matrix[3, 4] = 1
        adjacent_matrix[4, 5] = 1
        adjacent_matrix[5, 6] = 1
        adjacent_matrix[6, 7] = 1
        if arch[3]==1: # shortcut true
            adjacent_matrix[0, 7] = 1
        if arch[4]==1: # skip true
            adjacent_matrix[1, 5] = 1

    # cell3
    elif cur_stage==2:
        # input(output), input(skip), input(skip), 3, 2, output
        operation_matrix = np.zeros((9, 8))
        adjacent_matrix = np.zeros((9, 9))

        operation_matrix[0:3, OPERATION_INDEX["input"]] = 1
        operation_matrix[8, OPERATION_INDEX["output"]] = 1
        if arch[0]==0: # post conv
            if arch[1]==0: # no batch
                pass
            elif arch[1]==1: # bn
                operation_matrix[5, OPERATION_INDEX["bn_batch"]] = 1
                operation_matrix[7, OPERATION_INDEX["bn_batch"]] = 1
            elif arch[1]==2: # in
                operation_matrix[5, OPERATION_INDEX["in_batch"]] = 1
                operation_matrix[7, OPERATION_INDEX["in_batch"]] = 1
            if arch[2]==0: # bilinear
                operation_matrix[3, OPERATION_INDEX["bilinear_up"]] = 1
            elif arch[2]==1: # nearest
                operation_matrix[3, OPERATION_INDEX["nearest_up"]] = 1
            elif arch[2]==2: # deconv
                operation_matrix[3, OPERATION_INDEX["deconv_up"]] = 1
            operation_matrix[4, OPERATION_INDEX["conv"]] = 1
            operation_matrix[6, OPERATION_INDEX["conv"]] = 1
        elif arch[0]==1: # pre conv
            if arch[1]==0: # no batch
                pass
            elif arch[1]==1: # bn
                operation_matrix[3, OPERATION_INDEX["bn_batch"]] = 1
                operation_matrix[6, OPERATION_INDEX["bn_batch"]] = 1
            elif arch[1]==2: # in
                operation_matrix[3, OPERATION_INDEX["in_batch"]] = 1
                operation_matrix[6, OPERATION_INDEX["in_batch"]] = 1

            if arch[2]==0: # bilinear
                operation_matrix[4, OPERATION_INDEX["bilinear_up"]] = 1
            elif arch[2]==1: # nearest
                operation_matrix[4, OPERATION_INDEX["nearest_up"]] = 1
            elif arch[2]==2: # deconv
                operation_matrix[4, OPERATION_INDEX["deconv_up"]] = 1
            operation_matrix[5, OPERATION_INDEX["conv"]] = 1
            operation_matrix[7, OPERATION_INDEX["conv"]] = 1
        adjacent_matrix[0, 3] = 1
        adjacent_matrix[3, 4] = 1
        adjacent_matrix[4, 5] = 1
        adjacent_matrix[5, 6] = 1
        adjacent_matrix[6, 7] = 1
        adjacent_matrix[7, 8] = 1
        if arch[3]==1: # shortcut true
            adjacent_matrix[0, 8] = 1
        if arch[4]==3: # skip true
            adjacent_matrix[1, 6] = 1
            adjacent_matrix[2, 6] = 1
        elif arch[4]==2:
            adjacent_matrix[2, 6] = 1
            # adjacent_matrix[1, 6] = 1
        elif arch[4]==1:
            adjacent_matrix[1, 6] = 1

    return operation_matrix, adjacent_matrix