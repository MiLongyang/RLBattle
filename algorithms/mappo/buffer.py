# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import Dict, List, Tuple

class MAPPOBuffer:
    """
    MAPPO算法的经验缓冲区
    用于存储轨迹数据并计算优势函数
    """
    def __init__(self, buffer_size: int, num_agents: int, obs_dims: List[int], 
                 action_dims: List[int], state_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.state_dim = state_dim
        self.device = device
        
        # 初始化缓冲区
        self.reset()
    
    def reset(self):
        """重置缓冲区"""
        self.observations = [[] for _ in range(self.num_agents)]
        self.actions = [[] for _ in range(self.num_agents)]
        self.log_probs = [[] for _ in range(self.num_agents)]
        self.values = [[] for _ in range(self.num_agents)]
        self.rewards = []
        self.dones = []
        self.states = []
        self.step_count = 0
    
    def add(self, observations: List[np.ndarray], actions: List[np.ndarray], 
            log_probs: List[float], values: List[float], reward: float, 
            done: bool, state: np.ndarray):
        """
        添加一步经验
        
        Args:
            observations: 各智能体观测
            actions: 各智能体动作
            log_probs: 各智能体动作对数概率
            values: 各智能体价值估计
            reward: 全局奖励
            done: 是否结束
            state: 全局状态
        """
        for i in range(self.num_agents):
            self.observations[i].append(observations[i])
            self.actions[i].append(actions[i])
            self.log_probs[i].append(log_probs[i])
            self.values[i].append(values[i])
        
        self.rewards.append(reward)
        self.dones.append(done)
        self.states.append(state)
        self.step_count += 1
    
    def compute_advantages(self, next_values: List[float], gamma: float = 0.99, 
                          gae_lambda: float = 0.95) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        计算优势函数和回报
        
        Args:
            next_values: 下一状态的价值估计
            gamma: 折扣因子
            gae_lambda: GAE参数
            
        Returns:
            (advantages, returns): 优势函数和回报列表
        """
        advantages = [[] for _ in range(self.num_agents)]
        returns = [[] for _ in range(self.num_agents)]
        
        for agent_id in range(self.num_agents):
            agent_advantages = []
            agent_returns = []
            
            # 从后往前计算
            gae = 0
            for step in reversed(range(self.step_count)):
                if step == self.step_count - 1:
                    next_value = next_values[agent_id]
                    next_non_terminal = 1.0 - self.dones[step]
                else:
                    next_value = self.values[agent_id][step + 1]
                    next_non_terminal = 1.0 - self.dones[step]
                
                delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[agent_id][step]
                gae = delta + gamma * gae_lambda * next_non_terminal * gae
                
                agent_advantages.insert(0, gae)
                agent_returns.insert(0, gae + self.values[agent_id][step])
            
            advantages[agent_id] = torch.tensor(agent_advantages, dtype=torch.float32, device=self.device)
            returns[agent_id] = torch.tensor(agent_returns, dtype=torch.float32, device=self.device)
        
        return advantages, returns
    
    def get_batch(self, advantages: List[torch.Tensor], returns: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        获取训练批次数据
        
        Args:
            advantages: 优势函数
            returns: 回报
            
        Returns:
            batch: 批次数据字典
        """
        batch = {}
        
        # 转换为tensor
        batch['observations'] = []
        batch['actions'] = []
        batch['log_probs'] = []
        batch['values'] = []
        batch['advantages'] = []
        batch['returns'] = []
        
        for agent_id in range(self.num_agents):
            obs_tensor = torch.tensor(np.array(self.observations[agent_id]), 
                                    dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(np.array(self.actions[agent_id]), 
                                       dtype=torch.float32, device=self.device)
            log_prob_tensor = torch.tensor(self.log_probs[agent_id], 
                                         dtype=torch.float32, device=self.device)
            value_tensor = torch.tensor(self.values[agent_id], 
                                      dtype=torch.float32, device=self.device)
            
            batch['observations'].append(obs_tensor)
            batch['actions'].append(action_tensor)
            batch['log_probs'].append(log_prob_tensor)
            batch['values'].append(value_tensor)
            batch['advantages'].append(advantages[agent_id])
            batch['returns'].append(returns[agent_id])
        
        # 全局信息
        batch['rewards'] = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        batch['dones'] = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        batch['states'] = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        
        return batch
    
    def size(self) -> int:
        """返回缓冲区大小"""
        return self.step_count
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.step_count >= self.buffer_size