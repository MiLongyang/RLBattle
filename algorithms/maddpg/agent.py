# -*- coding: utf-8 -*-

import torch
import numpy as np
from .model import Actor, Critic

class Agent:
    """
    MADDPG 智能体
    """
    def __init__(self, agent_id, obs_dim, action_dim, num_agents, state_dim, actor_lr, critic_lr, device):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Actor 网络
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.target_actor = Actor(obs_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic 网络
        self.critic = Critic(state_dim, action_dim, num_agents).to(device)
        self.target_critic = Critic(state_dim, action_dim, num_agents).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs, add_noise=False, noise_std=0.1):
        """
        根据观测选择动作
        """
        obs = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        action = self.actor(obs).squeeze(0).cpu().detach().numpy()
        
        if add_noise:
            noise = np.random.randn(self.action_dim) * noise_std
            action += noise
            action = np.clip(action, -1.0, 1.0) # 假设动作空间是[-1, 1]
            
        return action

    def soft_update(self, target_net, source_net, tau):
        """
        软更新目标网络参数
        """
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update_target_networks(self, tau):
        """
        更新 Actor 和 Critic 的目标网络
        """
        self.soft_update(self.target_actor, self.actor, tau)
        self.soft_update(self.target_critic, self.critic, tau)
