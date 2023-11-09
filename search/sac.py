
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sac_utils import soft_update, hard_update
from sac_model import GaussianPolicy, QNetwork
from models_search.building_blocks_search import CONV_TYPE, NORM_TYPE, UP_TYPE, SHORT_CUT_TYPE, SKIP_TYPE


from torch import autograd
import logging

import numpy as np

def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class SAC(object):
    def __init__(self,num_inputs):

        self.gamma = 1
        self.tau = 0.005

        self.alpha = 0.1


        self.policy_type = "Gaussian"
        self.target_update_interval = 1
        self.automatic_entropy_tuning = False
        self.hid_size = 128
        self.target_entropy=-3
        self.device = torch.device("cuda:0")

        num_inputs = num_inputs
        
        tokens = [len(CONV_TYPE), len(NORM_TYPE), len(UP_TYPE), len(SHORT_CUT_TYPE), len(SKIP_TYPE),len(SKIP_TYPE)]
        action_space=sum(tokens)
       

        self.critic = QNetwork(num_inputs, action_space, self.hid_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.0003)

        self.critic_target = QNetwork(num_inputs, action_space, self.hid_size)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(3).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=0.0003)

            self.policy = GaussianPolicy(num_inputs, action_space, self.hid_size, action_space).to(self.device)


            self.policy_optim = Adam(self.policy.parameters(), lr=0.0003)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space, self.hid_size, action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=0.0003)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate is False:
            action1,action2,action3,action4, action5,action6,_,_,_,_,_, _ ,_,_, _, _ ,_,_,= self.policy.sample(state)
        else:
            _,_, _, _ ,_,_,_,_,_,_,_,_,action1,action2,action3,action4,action5,action6 = self.policy.sample(state)
        return action1.detach().cpu().numpy()[0],action2.detach().cpu().numpy()[0],action3.detach().cpu().numpy()[0],action4.detach().cpu().numpy()[0],action5.detach().cpu().numpy()[0],action6.detach().cpu().numpy()[0]


    def select_action_evaliter(self, state, evaluate=False, eval_iter=0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate is False:
            action1,action2,action3,action4, action5,action6,_,_,_,_,_, _ ,_,_, _, _ ,_,_, return_eval= self.policy.sample(state, max_iter=eval_iter)
        else:
            _,_, _, _ ,_,_,_,_,_,_,_,_,action1,action2,action3,action4,action5,action6, return_eval = self.policy.sample(state, max_iter=eval_iter)
        
        return_eval2 = []
        for one_eval in return_eval:
            tmp = []
            for action in one_eval:
                tmp.append(action.detach().cpu().numpy()[0])
            return_eval2.append(tmp)

        return [action1.detach().cpu().numpy()[0],action2.detach().cpu().numpy()[0],action3.detach().cpu().numpy()[0],action4.detach().cpu().numpy()[0],action5.detach().cpu().numpy()[0],action6.detach().cpu().numpy()[0]], return_eval2

        if not evaluate:
            # trainining
            action1,action2,action3,action4, action5,action6,_,_,_,_,_, _ ,_,_, _, _ ,_,_, return_eval= self.policy.sample(state, max_iter=eval_iter)
            return_eval2 = []
            for one_eval in return_eval:
                tmp = []
                for action in one_eval:
                    tmp.append(action.detach().cpu().numpy()[0])
                return_eval2.append(tmp)
            return [action1.detach().cpu().numpy()[0],action2.detach().cpu().numpy()[0],action3.detach().cpu().numpy()[0],action4.detach().cpu().numpy()[0],action5.detach().cpu().numpy()[0],action6.detach().cpu().numpy()[0]], return_eval2
        else:
            # evaluation
            _,_, _, _ ,_,_,_,_,_,_,_,_,action1,action2,action3,action4,action5,action6, return_eval = self.policy.sample(state, max_iter=1)
            return [action1.detach().cpu().numpy()[0],action2.detach().cpu().numpy()[0],action3.detach().cpu().numpy()[0],action4.detach().cpu().numpy()[0],action5.detach().cpu().numpy()[0],action6.detach().cpu().numpy()[0]], _



    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_state_action_1,next_state_action_2,next_state_action_3,next_state_action_4,next_state_action_5,next_state_action_6,next_state_log_pi_1,next_state_log_pi_2,next_state_log_pi_3,next_state_log_pi_4,next_state_log_pi_5,next_state_log_pi_6,_, _,_,_,_,_,= self.policy.sample(next_state_batch)
            next_state_action=torch.cat((next_state_action_1,next_state_action_2,next_state_action_3,next_state_action_4,next_state_action_5,next_state_action_6),1)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * (next_state_log_pi_1+next_state_log_pi_2+next_state_log_pi_3+next_state_log_pi_4+next_state_log_pi_5+next_state_log_pi_6)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi_1,pi_2,pi_3,pi_4,pi_5,pi_6,log_pi_1,log_pi_2,log_pi_3,log_pi_4,log_pi_5,log_pi_6,_,_,_,_,_,_, = self.policy.sample(state_batch)
        pi=torch.cat((pi_1,pi_2,pi_3,pi_4,pi_5,pi_6),1)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * (log_pi_1+log_pi_2+log_pi_3+log_pi_4+log_pi_5+log_pi_6)) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:

            alpha_loss = -(self.log_alpha * ((log_pi_1+log_pi_2+log_pi_3+log_pi_4+log_pi_5+log_pi_6))).detach().mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), ((log_pi_1+log_pi_2+log_pi_3+log_pi_4+log_pi_5+log_pi_6).mean()).item(), alpha_tlogs.item()


    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
