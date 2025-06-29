# -*- coding: utf-8 -*-

import numpy as np
import torch

class EpisodeReplayBuffer:
    """
    为QMIX设计的、存储完整Episode的回放池
    """
    def __init__(self, capacity, episode_limit, num_agents, obs_dims, action_dims, state_dim):
        # 检查并确认所有智能体是否同构 (QMIX的典型假设)
        assert all(d == obs_dims[0] for d in obs_dims), "EpisodeReplayBuffer 要求所有智能体的观测维度相同"
        assert all(d == action_dims[0] for d in action_dims), "EpisodeReplayBuffer 要求所有智能体的动作维度相同"

        self.capacity = capacity
        self.counter = 0
        
        # 为每个数据点分配空间
        # 我们需要存储 (obs, action, reward, done, state, next_obs)
        self.buffers = {
            'obs': np.zeros((capacity, episode_limit, num_agents, obs_dims[0])),
            'actions': np.zeros((capacity, episode_limit, num_agents, action_dims[0])),
            'rewards': np.zeros((capacity, episode_limit, 1)),
            'dones': np.zeros((capacity, episode_limit, 1)),
            'state': np.zeros((capacity, episode_limit, state_dim)),
            'next_obs': np.zeros((capacity, episode_limit, num_agents, obs_dims[0])),
            'next_state': np.zeros((capacity, episode_limit, state_dim))
        }
        # 记录每个episode的实际长度
        self.episode_lengths = np.zeros(capacity)

    def add_episode(self, episode_batch, episode_length):
        """
        向回放池中添加一个完整的episode

        Args:
            episode_batch (dict): 包含一个回合数据的字典,
                                  key为 'obs', 'actions' 等,
                                  value为 (episode_len, num_agents, dim) 的numpy数组.
            episode_length (int): 该回合的实际长度.
        """
        index = self.counter % self.capacity
        
        for key in self.buffers.keys():
            self.buffers[key][index, :episode_length] = episode_batch[key]
        self.episode_lengths[index] = episode_length
        
        self.counter += 1

    def sample(self, batch_size, device):
        """
        从回放池中随机采样一个批次的episodes
        """
        max_index = min(self.counter, self.capacity)
        batch_indices = np.random.choice(max_index, batch_size, replace=False)

        # 根据索引获取数据
        batch = {}
        for key in self.buffers.keys():
            batch[key] = self.buffers[key][batch_indices]
        
        episode_lengths = self.episode_lengths[batch_indices]

        # 将数据转换为tensor并移动到指定设备
        for key in batch.keys():
            batch[key] = torch.from_numpy(batch[key]).float().to(device)

        return batch, torch.from_numpy(episode_lengths).int().to(device)

    def size(self):
        """
        返回当前存储的episode数量
        """
        return self.counter 