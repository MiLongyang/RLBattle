# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor (策略) 网络
    输入: 自身观测 (observation)
    输出: 确定性动作 (action)
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        # 使用 tanh 将输出限制在 [-1, 1] 之间
        action = torch.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    """
    Critic (价值) 网络
    输入: 所有智能体的观测 (state) 和动作 (action)
    输出: Q值 (Q-value)
    """
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=64):
        super(Critic, self).__init__()
        # Critic的输入是全局状态和所有智能体的动作
        input_dim = state_dim + action_dim * num_agents
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # 将 state 和 action 在最后一个维度上拼接
        # action 是一个包含多个智能体动作的列表, 需要先拼接
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
