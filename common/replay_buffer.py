# -*- coding: utf-8 -*-

import numpy as np

class ReplayBuffer:
    """
    一个简单的多智能体经验回放池 (用于MADDPG)
    """
    def __init__(self, capacity, num_agents, obs_dims, action_dims, state_dim):
        self.capacity = capacity
        self.num_agents = num_agents
        self.counter = 0

        # 为每个智能体的数据分配空间
        self.obs_buffer = [np.zeros((self.capacity, obs_dims[i])) for i in range(num_agents)]
        self.action_buffer = [np.zeros((self.capacity, action_dims[i])) for i in range(num_agents)]
        self.reward_buffer = np.zeros((self.capacity, 1))
        self.next_obs_buffer = [np.zeros((self.capacity, obs_dims[i])) for i in range(num_agents)]
        self.done_buffer = np.zeros((self.capacity, 1))

        # 增加专门用于存储全局状态的空间
        self.state_buffer = np.zeros((self.capacity, state_dim))
        self.next_state_buffer = np.zeros((self.capacity, state_dim))

    def add(self, obs, state, actions, rewards, next_obs, next_state, dones):
        """
        向回放池中添加一条经验 (一个时间步)

        Args:
            obs (list of np.array): 每个智能体的观测
            state (np.array): 当前的全局状态
            actions (list of np.array): 每个智能体的动作
            rewards (float): 全局奖励
            next_obs (list of np.array): 每个智能体的下一个观测
            next_state (np.array): 下一个全局状态
            dones (bool): 是否结束
        """
        index = self.counter % self.capacity
        
        for i in range(self.num_agents):
            self.obs_buffer[i][index] = obs[i]
            self.action_buffer[i][index] = actions[i]
            self.next_obs_buffer[i][index] = next_obs[i]
        
        self.state_buffer[index] = state
        self.next_state_buffer[index] = next_state
        self.reward_buffer[index] = rewards
        self.done_buffer[index] = dones
        
        self.counter += 1

    def sample(self, batch_size):
        """
        从回放池中随机采样一个批次的数据
        """
        max_index = min(self.counter, self.capacity)
        batch_indices = np.random.choice(max_index, batch_size, replace=False)

        obs_batch = [self.obs_buffer[i][batch_indices] for i in range(self.num_agents)]
        action_batch = [self.action_buffer[i][batch_indices] for i in range(self.num_agents)]
        reward_batch = self.reward_buffer[batch_indices]
        next_obs_batch = [self.next_obs_buffer[i][batch_indices] for i in range(self.num_agents)]
        done_batch = self.done_buffer[batch_indices]

        # 从专门的buffer中获取真实的全局状态
        state_batch = self.state_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, state_batch, next_state_batch

    def size(self):
        """
        返回当前存储的经验数量
        """
        return self.counter
