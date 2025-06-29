# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixer(nn.Module):
    """
    QMIX 混合网络
    输入:
        - 所有智能体各自的 Q 值 (q_values)
        - 全局状态 (state)
    输出:
        - 混合后的总 Q 值 (q_total)
    """
    def __init__(self, num_agents, state_dim, mixing_embed_dim, device):
        super(QMixer, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.device = device

        # 超网络(Hypernetworks) - 根据 state 生成混合网络的权重和偏置
        # 第1层权重
        self.hyper_w1 = nn.Linear(self.state_dim, self.mixing_embed_dim * self.num_agents)
        # 第1层偏置
        self.hyper_b1 = nn.Linear(self.state_dim, self.mixing_embed_dim)

        # 第2层权重
        self.hyper_w2 = nn.Linear(self.state_dim, self.mixing_embed_dim)
        # 第2层偏置
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, self.mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(self.mixing_embed_dim, 1)
        )

    def forward(self, q_values, states):
        # q_values: (batch_size, num_agents)
        # states: (batch_size, state_dim)
        batch_size = q_values.size(0)
        q_values = q_values.view(-1, 1, self.num_agents) # (batch_size, 1, num_agents)

        # 生成第1层网络的权重和偏置
        w1 = torch.abs(self.hyper_w1(states)) # 保证权重非负
        w1 = w1.view(-1, self.num_agents, self.mixing_embed_dim) # (batch_size, num_agents, embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.mixing_embed_dim) # (batch_size, 1, embed_dim)

        # 第1层混合
        hidden = F.elu(torch.bmm(q_values, w1) + b1) # (batch_size, 1, embed_dim)

        # 生成第2层网络的权重和偏置
        w2 = torch.abs(self.hyper_w2(states)) # 保证权重非负
        w2 = w2.view(-1, self.mixing_embed_dim, 1) # (batch_size, embed_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1) # (batch_size, 1, 1)

        # 第2层混合
        q_total = torch.bmm(hidden, w2) + b2 # (batch_size, 1, 1)
        q_total = q_total.view(batch_size, -1) # (batch_size, 1)
        
        return q_total
