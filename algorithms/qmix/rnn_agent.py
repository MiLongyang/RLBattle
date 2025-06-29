# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    """
    QMIX 中的智能体, 使用 RNN 来处理局部观测历史
    """
    def __init__(self, input_shape, n_actions, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.agent_hidden_dim)
        # 使用 GRU 作为 RNN 层
        self.rnn = nn.GRUCell(args.agent_hidden_dim, args.agent_hidden_dim)
        self.fc2 = nn.Linear(args.agent_hidden_dim, n_actions)

    def init_hidden(self):
        # 创建一个初始的隐藏状态, a new variable on the device
        return self.fc1.weight.new(1, self.args.agent_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        输入:
            inputs (torch.Tensor): 智能体的当前观测
            hidden_state (torch.Tensor): GRU 的上一个隐藏状态
        输出:
            q_values (torch.Tensor): 每个动作的 Q 值
            h (torch.Tensor): GRU 的当前隐藏状态
        """
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q, h
