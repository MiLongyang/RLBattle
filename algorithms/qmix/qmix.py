# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .rnn_agent import RNNAgent
from .q_mixer import QMixer
from common.episode_replay_buffer import EpisodeReplayBuffer
from ..base_algorithm import BaseAlgorithm

class QMIX(BaseAlgorithm):
    """
    QMIX 算法主类
    """
    def __init__(self, obs_dims, action_dims, num_agents, state_dim, args, device):
        self.args = args
        self.device = device
        self.num_agents = num_agents
        self.state_dim = state_dim
        
        # 检查并确认所有智能体是否同构 (QMIX的典型假设)
        assert all(d == obs_dims[0] for d in obs_dims), "QMIX 要求所有智能体的观测维度相同"
        assert all(d == action_dims[0] for d in action_dims), "QMIX 要求所有智能体的动作维度相同"
        self.obs_dim = obs_dims[0]
        self.action_dim = action_dims[0]

        # 初始化网络
        self.agent_net = RNNAgent(self.obs_dim, self.action_dim, args).to(device)
        self.mixer_net = QMixer(num_agents, state_dim, args.mixer_hidden_dim, device).to(device)

        # 初始化目标网络
        self.target_agent_net = RNNAgent(self.obs_dim, self.action_dim, args).to(device)
        self.target_mixer_net = QMixer(num_agents, state_dim, args.mixer_hidden_dim, device).to(device)
        
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())

        # 设置优化器
        network_params = list(self.agent_net.parameters()) + list(self.mixer_net.parameters())
        self.optimizer = optim.Adam(params=network_params, lr=args.lr_qmix)

        # 经验回放池
        self.replay_buffer = EpisodeReplayBuffer(args.buffer_size_qmix, args.episode_limit, num_agents, obs_dims, action_dims, state_dim)
        
        self.train_step = 0

    def select_actions(self, observations, hidden_states, epsilon):
        actions = []
        new_hidden_states = []
        for i, obs in enumerate(observations):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q_values, h_out = self.agent_net(obs_tensor, hidden_states[i])
            
            if np.random.rand() < epsilon:
                action = np.random.randint(0, self.action_dim)
            else:
                action = q_values.argmax().item()
            actions.append(action)
            new_hidden_states.append(h_out)
        
        return actions, new_hidden_states

    def learn(self):
        if self.replay_buffer.size() < self.args.batch_size:
            return

        batch, episode_lengths = self.replay_buffer.sample(self.args.batch_size, self.device)
        
        max_len = int(episode_lengths.max())
        if max_len == 0:
            return # 批次中最长回合长度为0, 无需学习
        
        # 将所有采样到的数据裁剪到当前batch中最长episode的长度
        for key in batch.keys():
            batch[key] = batch[key][:, :max_len]
            
        # 1. 计算当前Q_tot
        q_evals, q_targets = [], []
        hidden_eval = self.agent_net.init_hidden().unsqueeze(0).expand(self.args.batch_size, self.num_agents, -1)
        hidden_target = self.target_agent_net.init_hidden().unsqueeze(0).expand(self.args.batch_size, self.num_agents, -1)

        for t in range(max_len):
            obs_t = batch['obs'][:, t]
            action_t = batch['actions'][:, t].long()
            
            # (batch_size, num_agents, q_values)
            q_eval_t, h_eval_next = self.agent_net(obs_t.reshape(-1, self.obs_dim), hidden_eval.reshape(-1, self.args.agent_hidden_dim))
            q_eval_t = q_eval_t.reshape(self.args.batch_size, self.num_agents, -1)
            hidden_eval = h_eval_next.reshape(self.args.batch_size, self.num_agents, -1) # 恢复维度
            # 把采取的动作的Q值拿出来
            q_evals_t_taken = torch.gather(q_eval_t, dim=2, index=action_t).squeeze(2) # (batch_size, num_agents)
            q_evals.append(q_evals_t_taken)

            # 2. 计算目标Q_tot
            with torch.no_grad():
                next_obs_t = batch['next_obs'][:, t]
                q_target_t, h_target_next = self.target_agent_net(next_obs_t.reshape(-1, self.obs_dim), hidden_target.reshape(-1, self.args.agent_hidden_dim))
                q_target_t = q_target_t.reshape(self.args.batch_size, self.num_agents, -1)
                hidden_target = h_target_next.reshape(self.args.batch_size, self.num_agents, -1) # 恢复维度
                # 采用 Double-DQN 的思想, 动作由 eval_net 决定
                action_next = q_eval_t.argmax(dim=2, keepdim=True)
                q_target_t_taken = torch.gather(q_target_t, dim=2, index=action_next).squeeze(2)
                q_targets.append(q_target_t_taken)

        q_evals = torch.stack(q_evals, dim=1) # (batch_size, max_len, num_agents)
        q_targets = torch.stack(q_targets, dim=1) # (batch_size, max_len, num_agents)

        q_total_eval = self.mixer_net(q_evals, batch['state'][:, :max_len])
        q_total_target = self.target_mixer_net(q_targets, batch['state'][:, :max_len])
        
        rewards = batch['rewards'][:, :max_len].squeeze(2)
        dones = batch['dones'][:, :max_len].squeeze(2)
        
        # 计算 TD-target
        td_target = rewards + self.args.gamma * (1 - dones) * q_total_target
        
        # 损失计算
        mask = (torch.ones_like(dones) - dones).float() # 创建mask, 忽略填充部分
        loss = F.mse_loss(q_total_eval * mask, td_target.detach() * mask)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.agent_net.parameters()) + list(self.mixer_net.parameters()), 1.0)
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.args.target_update_interval == 0:
            self.update_target_networks()
            
    def update_target_networks(self):
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())

    def add_experience(self, obs, actions, rewards, next_obs, dones):
        self.replay_buffer.add(obs, actions, rewards, next_obs, dones)

    def save_models(self, path, episode_num):
        """
        保存 QMIX 的模型
        """
        torch.save(self.agent_net.state_dict(), f"{path}/agent_net_{episode_num}.pth")
        torch.save(self.mixer_net.state_dict(), f"{path}/mixer_net_{episode_num}.pth")

    def load_models(self, path, episode_num):
        """
        加载 QMIX 的模型
        """
        self.agent_net.load_state_dict(torch.load(f"{path}/agent_net_{episode_num}.pth"))
        self.mixer_net.load_state_dict(torch.load(f"{path}/mixer_net_{episode_num}.pth"))
