from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
import pandas as pd
import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm


import cfg
import models_search
import datasets
from utils.utils import set_log_dir, save_checkpoint, create_logger, RunningStats
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from functions import *
from sac import SAC
from replay_memory import ReplayMemory
from predictor_model import NeuralPredictor
from predictor_model import NeuralPredictorFID

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")


def get_arch(layer, cur_arch):
    if layer==0:
        cur_arch = cur_arch[0:4]
    elif layer==1:
        cur_arch = cur_arch[0:5]
    elif layer==2:
        if cur_arch[4]+cur_arch[5]==2:
            cur_arch = cur_arch[0:4]+ [3]
        elif cur_arch[4]+cur_arch[5]==0:
            cur_arch = cur_arch[0:4]+ [0]
        elif cur_arch[4]==1 and cur_arch[5]==0:
            cur_arch = cur_arch[0:4]+ [1]
        else:
            cur_arch = cur_arch[0:4] +[2]

    return cur_arch

def create_shared_gan(args, weights_init):
    gen_net = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
    dis_net = eval('models_search.'+args.dis_model+'.Discriminator')(args=args).cuda()
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    return gen_net, dis_net, gen_optimizer, dis_optimizer


def main():
    args = cfg.parse_args()
    
    torch.cuda.manual_seed(args.random_seed)
    print(args)    
    

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init)

    # initial
    start_search_iter = 0

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        cur_stage = checkpoint['cur_stage']

        start_search_iter = checkpoint['search_iter']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        prev_archs = checkpoint['prev_archs']
        prev_hiddens = checkpoint['prev_hiddens']

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (search iteration {start_search_iter})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        prev_archs = None
        prev_hiddens = None

        # set controller && its optimizer
        cur_stage = 0

    # set up data_loader
    dataset = datasets.ImageDataset(args, 2**(cur_stage+3))
    train_loader = dataset.train
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'controller_steps': start_search_iter * args.ctrl_step
    }

    g_loss_history = RunningStats(args.dynamic_reset_window)
    d_loss_history = RunningStats(args.dynamic_reset_window)

    # train loop
    Agent=SAC(131)
    
    memory = ReplayMemory(5000)
    predictor_memory = ReplayMemory(5000)
    updates=0
    outinfo = {'rewards': [],
                'a_loss': [],
                'critic_error': [],
                }
    Best=False
    Z_NUMPY=None
    WARMUP=True
    update_time=1
    writer = writer_dict['writer']

    predictor = NeuralPredictor().to(device)
    criterion = nn.MSELoss()
    fid_predictor = NeuralPredictorFID().to(device)
    fid_criterion = nn.MSELoss()
    lr = 0.003
    is_optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    fid_optimizer = torch.optim.Adam(fid_predictor.parameters(), lr=lr)
    train_flag = True
    predictor_epoch = 5


    for search_iter in tqdm(range(int(start_search_iter), 100), desc='search progress'):
        logger.info(f"<start search iteration {search_iter}>")
        if search_iter>=1:
            WARMUP=False

        total_layer_num = 3
        ds = [datasets.ImageDataset(args, 2 ** (k + 3)) for k in range(total_layer_num)]
        train_loaders = [d.train for d in ds]
        last_R = 0. # Initial reward
        last_fid=10000  # Inital reward
        last_arch = [] 

        # Set exploration 
        if search_iter > 69: 
            update_time=10
            Best=True
        else:
            Best=False
        if search_iter > 79:
            train_flag = False

        gen_net.set_stage(-1)
        last_R,last_fid,last_state = get_is(args, gen_net, args.rl_num_eval_img, get_is_score=True)

        prev_hiddens = [torch.zeros(1, 144).to(device), torch.zeros(1, 144).to(device)]
        fid_prev_hiddens = [torch.zeros(1, 144).to(device), torch.zeros(1, 144).to(device)]
        tmp_is, tmp_fid = -1.0, -1.0

        for layer in range(3):
            cur_stage = layer # This defines which layer to use as output, for example, if cur_stage==0, then the output will be the first layer output. Set it to 2 if you want the output of the last layer.
            action, return_eval = Agent.select_action_evaliter([layer, last_R,0.01*last_fid] + last_state,Best, eval_iter=args.number_eval_arches)

            training_action = [action[0][0],action[0][1],action[1][0],action[1][1],action[1][2],action[2][0],action[2][1],action[2][2],action[3][0],action[3][1],action[4][0],action[4][1],action[5][0],action[5][1]]
            cur_arch = [np.argmax(k) for k in action]
            cur_arch = get_arch(layer, cur_arch)
            if train_flag:
                predictor.train()
                fid_predictor.train()
                eval_archs = []
                for eval_action in return_eval: # iteration=5
                    eval_archs.append([np.argmax(k) for k in eval_action])
                eval_archs = [get_arch(layer, arch) for arch in eval_archs]

                eval_original_islist = []
                eval_original_fidlist = []
                predict_islist = []
                fid_predict_islist = []
                vertics_list = []
                adjacent_matrix_list = []
                operation_matrix_list = []
                # eval predictor
                tensor_output_list = []
                fid_tensor_output_list = []

                for arch in eval_archs:
                    if layer != 0:
                        gen_net.set_arch(last_arch+arch, layer)
                    else:
                        gen_net.set_arch(arch, layer)

                    # training fid
                    if search_iter > 20:
                        eval_original_is, eval_original_fid, _ = get_is(args, gen_net, args.rl_num_eval_img, get_is_score=True)
                        eval_original_islist.append(eval_original_is)
                        eval_original_fidlist.append(eval_original_fid*0.01)
                    else:
                        # trainig only
                        eval_original_is = only_get_is(args, gen_net, args.rl_num_eval_img, get_is_score=True)
                        eval_original_islist.append(eval_original_is)

                    operation_matrix, adjacent_matrix = arch2matrix(arch, cur_stage)
                    adjacent_matrix = torch.tensor(adjacent_matrix, dtype=torch.float, device=device)
                    operation_matrix = torch.tensor(operation_matrix, dtype=torch.float, device=device)
                    if cur_stage==0:
                        n_vertices=7
                    elif cur_stage==1:
                        n_vertices=8
                    elif cur_stage==2:
                        n_vertices=9

                    output, hidden = predictor(n_vertices , adjacent_matrix, operation_matrix, prev_hiddens, layer)
                    tensor_output_list.append(output)
                    predict_islist.append(output.item())
                    if search_iter > 20:
                        fid_output, fid_hidden = fid_predictor(output.clone().detach(), n_vertices, adjacent_matrix, operation_matrix, fid_prev_hiddens, layer)
                        fid_tensor_output_list.append(fid_output)
                        fid_predict_islist.append(fid_output.item())

                # for next cell
                gen_net.set_arch(last_arch+cur_arch, layer)
                tmp_is, tmp_fid, state = get_is(args, gen_net, args.rl_num_eval_img, Z_NUMPY, True, True)

                eval_original_islist.append(tmp_is)
                eval_original_fidlist.append(tmp_fid*0.01)
                operation_matrix, adjacent_matrix = arch2matrix(cur_arch, layer)
                adjacent_matrix = torch.tensor(adjacent_matrix, dtype=torch.float, device=device)
                operation_matrix = torch.tensor(operation_matrix, dtype=torch.float, device=device)
                if cur_stage==0:
                    n_vertices=7
                elif cur_stage==1:
                    n_vertices=8
                elif cur_stage==2:
                    n_vertices=9

                output, hidden = predictor(n_vertices , adjacent_matrix, operation_matrix, prev_hiddens, layer)
                tensor_output_list.append(output)
                loss = criterion(torch.tensor(eval_original_islist, dtype=torch.float, device=device).unsqueeze(1), torch.stack(tensor_output_list).float())
                if search_iter > 20:
                    fid_output, fid_hidden = fid_predictor(torch.tensor(tmp_is,device=device), n_vertices , adjacent_matrix, operation_matrix, fid_prev_hiddens, layer)
                    fid_tensor_output_list.append(fid_output)
                    fid_loss = fid_criterion(torch.tensor(eval_original_fidlist, dtype=torch.float, device=device).unsqueeze(1), torch.stack(fid_tensor_output_list).float())
                else:
                    fid_output, fid_hidden = fid_predictor(torch.tensor(tmp_is, device=device), n_vertices , adjacent_matrix, operation_matrix, fid_prev_hiddens, layer)
                    fid_loss = fid_criterion(torch.tensor(tmp_fid*0.01, device=device), fid_output)

                is_optimizer.zero_grad()
                loss.backward()
                is_optimizer.step()
                fid_optimizer.zero_grad()
                fid_loss.backward()
                fid_optimizer.step()

                prev_hiddens = (hidden[0].detach(), hidden[1].detach())
                fid_prev_hiddens = (fid_hidden[0].detach(), fid_hidden[1].detach())

                R = tmp_is
                fid = tmp_fid

            else:
                # return_eval is none
                predictor.eval()
                fid_predictor.eval()
                gen_net.set_arch(last_arch+cur_arch, layer)
                R, fid, state  = get_is(args, gen_net, args.rl_num_eval_img, Z_NUMPY, True, True)

                operation_matrix, adjacent_matrix = arch2matrix(cur_arch, layer)
                adjacent_matrix = torch.tensor(adjacent_matrix, dtype=torch.float, device=device)
                operation_matrix = torch.tensor(operation_matrix, dtype=torch.float, device=device)
                if cur_stage==0:
                    n_vertices=7
                elif cur_stage==1:
                    n_vertices=8
                elif cur_stage==2:
                    n_vertices=9
                output, hidden = predictor(n_vertices , adjacent_matrix, operation_matrix, prev_hiddens, layer)
                fid_output, fid_hidden = fid_predictor(output.clone().detach(), n_vertices , adjacent_matrix, operation_matrix, fid_prev_hiddens, layer)
                
                loss = criterion(torch.tensor(R, device=device), torch.reshape(output, (1,)))
                fid_loss = fid_criterion(torch.tensor(fid*0.01, device=device), torch.reshape(fid_output, (1,)))

                R = output.item()
                fid = fid_output.item()*100
                prev_hiddens = (hidden[0].detach(), hidden[1].detach())
                fid_prev_hiddens = (fid_hidden[0].detach(), fid_hidden[1].detach())


            last_arch += cur_arch

            # if layer !=2:
            dynamic_reset = train_qin(args, gen_net, dis_net, g_loss_history, d_loss_history, gen_optimizer,
                                    dis_optimizer, train_loaders[layer], cur_stage, smooth=False, WARMUP=WARMUP) 
            mask = 0 if layer == total_layer_num-1 else 1
            if search_iter >=0: # warm up
                memory.push([layer,last_R,0.01*last_fid]+last_state, training_action, R-last_R+0.01*(last_fid-fid), [layer+1,R,0.01*fid] + state, mask)  # Append transition to memory

            if len(memory) >= 64:
                # Number of updates per step in environment
                for i in range(update_time):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = Agent.update_parameters(memory, min(len(memory),256), updates)
       
                    updates += 1 
                    outinfo['critic_error']=min(critic_1_loss, critic_2_loss)
                    outinfo['entropy']=ent_loss
                    outinfo['a_loss']=policy_loss
            last_R = R # next step
            last_fid = fid
            last_state = state

        outinfo['rewards']=R
        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = Agent.update_parameters(memory, len(memory), updates)
        updates += 1 
        outinfo['critic_error']=min(critic_1_loss, critic_2_loss)
        outinfo['entropy']=ent_loss
        outinfo['a_loss']=policy_loss

        writer.add_scalar('optimizing/rewards', outinfo['rewards'], search_iter)
        writer.add_scalar('optimizing/critic_error', outinfo['critic_error'], search_iter)
        writer.add_scalar('optimizing/entropy', outinfo['entropy'], search_iter)
        writer.add_scalar('optimizing/a_loss', outinfo['a_loss'], search_iter)

        # Clean up and start a new trajectory from scratch
        del gen_net, dis_net, gen_optimizer, dis_optimizer
        gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(args, weights_init)
        WARMUP=False



if __name__ == '__main__':
    main()