from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import models_search
from functions import validate
from utils.utils import set_log_dir, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
import numpy as np
from tensorboardX import SummaryWriter

torch.multiprocessing.set_start_method('spawn')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    assert args.load_path.endswith('.pth')
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir('logs_eval', args.exp_name)
    logger = create_logger(args.path_helper['log_path'], phase='test')

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    gen_net = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
    gen_net.set_arch(args.arch, cur_stage=2)
    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.dataset.lower() =='cifar100':
        fid_stat = 'fid_stat/fid_stats_cifar100_train.npz'
    elif args.dataset.lower()=='imagenet32':
        fid_stat = 'fid_stat/fid_stats_imagenet32.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    
    # set writer
    logger.info(f'=> resuming from {args.load_path}')
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
  
    if 'avg_gen_state_dict' in checkpoint:
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        epoch = checkpoint['epoch'] - 1
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    else:
        gen_net.load_state_dict(checkpoint)
        logger.info(f'=> loaded checkpoint {checkpoint_file}')

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': 0,
    }
    inception_score, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict, clean_dir=False)
    logger.info(f'Inception score: {inception_score}, FID score: {fid_score}.')


if __name__ == '__main__':
    main()