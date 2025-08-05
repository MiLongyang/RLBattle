# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class Actor(nn.Module):
    """
    MAPPO Actor网络（策略网络）
    支持连续和离散动作空间
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, 
                 action_type: str = 'discrete'):
        super(Actor, self).__init__()
        self.action_type = action_type
        self.action_dim = action_dim
        
        # 共享特征提取层
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if action_type == 'discrete':
            # 离散动作空间：输出动作概率分布
            self.action_head = nn.Linear(hidden_dim, action_dim)
        else:
            # 连续动作空间：输出均值和标准差
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            obs: 观测输入 (batch_size, obs_dim)
            
        Returns:
            如果是离散动作：(action_logits, None)
            如果是连续动作：(action_mean, action_log_std)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        
        if self.action_type == 'discrete':
            action_logits = self.action_head(x)
            return action_logits, None
        else:
            action_mean = self.mean_head(x)
            action_log_std = self.log_std_head(x)
            # 限制标准差的范围
            action_log_std = torch.clamp(action_log_std, -20, 2)
            return action_mean, action_log_std
    
    def get_action_and_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取动作和对数概率
        
        Args:
            obs: 观测输入
            
        Returns:
            (action, log_prob): 动作和对数概率
        """
        if self.action_type == 'discrete':
            action_logits, _ = self.forward(obs)
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action, log_prob
        else:
            action_mean, action_log_std = self.forward(obs)
            action_std = torch.exp(action_log_std)
            action_dist = torch.distributions.Normal(action_mean, action_std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            return action, log_prob
    
    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算给定动作的对数概率
        
        Args:
            obs: 观测输入
            action: 动作
            
        Returns:
            log_prob: 对数概率
        """
        if self.action_type == 'discrete':
            action_logits, _ = self.forward(obs)
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            log_prob = action_dist.log_prob(action.squeeze(-1))
            return log_prob
        else:
            action_mean, action_log_std = self.forward(obs)
            action_std = torch.exp(action_log_std)
            action_dist = torch.distributions.Normal(action_mean, action_std)
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            return log_prob


class Critic(nn.Module):
    """
    MAPPO Critic网络（价值网络）
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            obs: 观测输入 (batch_size, obs_dim)
            
        Returns:
            value: 状态价值 (batch_size, 1)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value


class CentralizedCritic(nn.Module):
    """
    MAPPO 中心化Critic网络
    使用全局状态信息进行价值估计
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(CentralizedCritic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 全局状态输入 (batch_size, state_dim)
            
        Returns:
            value: 全局状态价值 (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value