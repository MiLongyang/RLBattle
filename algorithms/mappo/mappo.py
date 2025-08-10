# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Union

from .networks import Actor, Critic, CentralizedCritic
from .buffer import MAPPOBuffer
from ..base_algorithm import BaseAlgorithm

logger = logging.getLogger(__name__)

class MAPPO(BaseAlgorithm):
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) 算法实现
    """
    def __init__(self, obs_dims: List[int], action_dims: List[int], num_agents: int,
                 state_dim: int, args, device: torch.device):
        # 调用父类初始化
        super().__init__(obs_dims, action_dims, num_agents, state_dim, args, device)
        
        # MAPPO特定参数
        self.clip_ratio = getattr(args, 'clip_ratio', 0.2)
        self.value_loss_coef = getattr(args, 'value_loss_coef', 0.5)
        self.entropy_coef = getattr(args, 'entropy_coef', 0.01)
        self.max_grad_norm = getattr(args, 'max_grad_norm', 0.5)
        self.ppo_epochs = getattr(args, 'ppo_epochs', 4)
        self.mini_batch_size = getattr(args, 'mini_batch_size', 64)
        self.buffer_size = getattr(args, 'buffer_size_mappo', 2048)
        self.gamma = getattr(args, 'gamma', 0.99)
        self.gae_lambda = getattr(args, 'gae_lambda', 0.95)
        self.use_centralized_critic = getattr(args, 'use_centralized_critic', True)
        
        # 确定动作类型
        self.action_type = getattr(args, 'action_type', 'continuous')
        
        # 初始化网络
        self.actors = []
        self.critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        actor_lr = getattr(args, 'actor_lr_mappo', 3e-4)
        critic_lr = getattr(args, 'critic_lr_mappo', 1e-3)
        
        # 对于MAPPO，全局状态通常是所有智能体观测的拼接
        actual_state_dim = sum(obs_dims)  # 实际的全局状态维度
        
        for i in range(num_agents):
            # Actor网络
            actor = Actor(obs_dims[i], action_dims[i], 
                         getattr(args, 'actor_hidden_dim', 64), self.action_type).to(device)
            self.actors.append(actor)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=actor_lr))
            
            # Critic网络 - 使用实际的状态维度
            if self.use_centralized_critic:
                critic = CentralizedCritic(actual_state_dim, 
                                         getattr(args, 'critic_hidden_dim', 64)).to(device)
            else:
                critic = Critic(obs_dims[i], 
                              getattr(args, 'critic_hidden_dim', 64)).to(device)
            self.critics.append(critic)
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=critic_lr))
        
        # 经验缓冲区
        self.buffer = MAPPOBuffer(self.buffer_size, num_agents, obs_dims, 
                                 action_dims, actual_state_dim, device)
        
        # 训练指标
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy_loss = 0.0
        self.update_count = 0

    def select_actions(self, observations: List[np.ndarray], 
                      **kwargs) -> Tuple[List[np.ndarray], Optional[None]]:
        """
        为所有智能体选择动作
        
        Args:
            observations: 所有智能体的观测列表
            **kwargs: 可选参数
            
        Returns:
            Tuple[actions, None]: 动作列表和None
        """
        # 验证观测数据
        if not self.validate_observations(observations):
            raise ValueError("Invalid observations provided")
        
        actions = []
        log_probs = []
        values = []
        
        # 获取全局状态（如果需要）
        state = kwargs.get('state', np.concatenate(observations))
        
        with torch.no_grad():
            for i, obs in enumerate(observations):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # 获取动作和对数概率
                if self.training:
                    action, log_prob = self.actors[i].get_action_and_log_prob(obs_tensor)
                else:
                    # 评估模式：选择最优动作
                    if self.action_type == 'discrete':
                        action_logits, _ = self.actors[i].forward(obs_tensor)
                        action = torch.argmax(action_logits, dim=-1)
                        log_prob = torch.zeros(1, device=self.device)
                    else:
                        action_mean, _ = self.actors[i].forward(obs_tensor)
                        action = action_mean
                        log_prob = torch.zeros(1, device=self.device)
                
                # 获取价值估计
                if self.use_centralized_critic:
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    value = self.critics[i](state_tensor)
                else:
                    value = self.critics[i](obs_tensor)
                
                # 确保动作输出格式正确
                action_np = action.cpu().numpy()
                if action_np.ndim == 0:
                    # 0维数组转换为1维
                    action_np = np.array([action_np])
                elif action_np.ndim > 1:
                    # 多维数组压缩为1维
                    action_np = action_np.squeeze()
                    if action_np.ndim == 0:
                        action_np = np.array([action_np])
                
                actions.append(action_np)
                log_probs.append(log_prob.cpu().numpy().item())
                values.append(value.cpu().numpy().item())
        
        # 存储额外信息用于训练
        if hasattr(self, '_temp_log_probs'):
            self._temp_log_probs = log_probs
            self._temp_values = values
        
        return actions, None
    
    def add_experience(self, observations: List[np.ndarray], actions: List[np.ndarray],
                      log_probs: List[float], values: List[float], reward: float,
                      done: bool, state: np.ndarray):
        """
        添加经验到缓冲区
        
        Args:
            observations: 观测列表
            actions: 动作列表
            log_probs: 对数概率列表
            values: 价值估计列表
            reward: 奖励
            done: 是否结束
            state: 全局状态
        """
        self.buffer.add(observations, actions, log_probs, values, reward, done, state)

    def learn(self, **kwargs) -> Dict[str, float]:
        """
        执行PPO学习更新
        
        Returns:
            Dict[str, float]: 训练指标字典
        """
        if self.buffer.size() < self.mini_batch_size:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'buffer_size': self.buffer.size()
            }
        
        # 计算下一状态的价值估计（用于GAE计算）
        next_values = []
        if self.buffer.size() > 0:
            # 使用最后一个状态估计下一状态价值
            last_obs = [self.buffer.observations[i][-1] for i in range(self.num_agents)]
            last_state = self.buffer.states[-1]
            
            with torch.no_grad():
                for i in range(self.num_agents):
                    if self.use_centralized_critic:
                        state_tensor = torch.tensor(last_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                        next_value = self.critics[i](state_tensor).cpu().numpy().item()
                    else:
                        obs_tensor = torch.tensor(last_obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                        next_value = self.critics[i](obs_tensor).cpu().numpy().item()
                    next_values.append(next_value)
        else:
            next_values = [0.0] * self.num_agents
        
        # 计算优势函数和回报
        advantages, returns = self.buffer.compute_advantages(next_values, self.gamma, self.gae_lambda)
        
        # 获取批次数据
        batch = self.buffer.get_batch(advantages, returns)
        
        # 执行多轮PPO更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        
        for epoch in range(self.ppo_epochs):
            policy_loss, value_loss, entropy_loss = self._update_networks(batch)
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy_loss += entropy_loss
        
        # 清空缓冲区
        self.buffer.reset()
        
        # 更新训练步数和指标
        self.update_training_step()
        self.update_count += 1
        
        # 计算平均损失
        avg_policy_loss = total_policy_loss / self.ppo_epochs
        avg_value_loss = total_value_loss / self.ppo_epochs
        avg_entropy_loss = total_entropy_loss / self.ppo_epochs
        
        self.last_policy_loss = avg_policy_loss
        self.last_value_loss = avg_value_loss
        self.last_entropy_loss = avg_entropy_loss
        
        metrics = {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_policy_loss + avg_value_loss + avg_entropy_loss,
            'training_step': self.training_step,
            'update_count': self.update_count,
            'buffer_size': self.buffer.size()
        }
        
        self.update_metrics(metrics)
        return metrics
    
    def _update_networks(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
        """
        更新Actor和Critic网络
        
        Args:
            batch: 批次数据
            
        Returns:
            (policy_loss, value_loss, entropy_loss): 各项损失
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        
        for agent_id in range(self.num_agents):
            # 获取当前智能体的数据
            obs = batch['observations'][agent_id]
            actions = batch['actions'][agent_id]
            old_log_probs = batch['log_probs'][agent_id]
            old_values = batch['values'][agent_id]
            advantages = batch['advantages'][agent_id]
            returns = batch['returns'][agent_id]
            
            # 标准化优势函数
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算新的对数概率和价值
            new_log_probs = self.actors[agent_id].get_log_prob(obs, actions)
            
            if self.use_centralized_critic:
                states = batch['states']
                new_values = self.critics[agent_id](states).squeeze(-1)
            else:
                new_values = self.critics[agent_id](obs).squeeze(-1)
            
            # 计算策略损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = F.mse_loss(new_values, returns)
            
            # 计算熵损失
            if self.action_type == 'discrete':
                action_logits, _ = self.actors[agent_id].forward(obs)
                action_probs = F.softmax(action_logits, dim=-1)
                entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
            else:
                action_mean, action_log_std = self.actors[agent_id].forward(obs)
                entropy = (action_log_std + 0.5 * np.log(2 * np.pi * np.e)).sum(dim=-1).mean()
            
            entropy_loss = -self.entropy_coef * entropy
            
            # 更新Actor
            actor_loss = policy_loss + entropy_loss
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actor_optimizers[agent_id].step()
            
            # 更新Critic
            critic_loss = self.value_loss_coef * value_loss
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.max_grad_norm)
            self.critic_optimizers[agent_id].step()
            
            # 累积损失
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        # 返回平均损失
        return (total_policy_loss / self.num_agents, 
                total_value_loss / self.num_agents, 
                total_entropy_loss / self.num_agents)

    def save_models(self, save_dir: str, episode: Union[str, int]) -> None:
        """
        保存MAPPO模型
        
        Args:
            save_dir: 保存目录路径
            episode: 回合数或标识符
        """
        start_time = time.time()
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            for i in range(self.num_agents):
                actor_path = os.path.join(save_dir, f"actor_agent_{i}_{episode}.pth")
                critic_path = os.path.join(save_dir, f"critic_agent_{i}_{episode}.pth")
                
                torch.save(self.actors[i].state_dict(), actor_path)
                torch.save(self.critics[i].state_dict(), critic_path)
            
            # 保存算法元数据
            metadata = {
                'algorithm': 'MAPPO',
                'episode': str(episode),
                'training_step': self.training_step,
                'episode_count': self.episode_count,
                'num_agents': self.num_agents,
                'obs_dims': self.obs_dims,
                'action_dims': self.action_dims,
                'action_type': self.action_type,
                'use_centralized_critic': self.use_centralized_critic,
                'update_count': self.update_count,
                'save_time': time.time()
            }
            
            metadata_path = os.path.join(save_dir, f"metadata_{episode}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            save_time = time.time() - start_time
            logger.info(f"MAPPO models saved successfully in {save_time:.2f}s to {save_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save MAPPO models: {e}")
            raise

    def load_models(self, load_dir: str, episode: Union[str, int]) -> None:
        """
        加载MAPPO模型
        
        Args:
            load_dir: 加载目录路径
            episode: 回合数或标识符
        """
        start_time = time.time()
        
        try:
            for i in range(self.num_agents):
                actor_path = os.path.join(load_dir, f"actor_agent_{i}_{episode}.pth")
                critic_path = os.path.join(load_dir, f"critic_agent_{i}_{episode}.pth")
                
                if not os.path.exists(actor_path) or not os.path.exists(critic_path):
                    raise FileNotFoundError(f"Model files not found for agent {i} at episode {episode}")
                
                self.actors[i].load_state_dict(torch.load(actor_path, map_location=self.device))
                self.critics[i].load_state_dict(torch.load(critic_path, map_location=self.device))
            
            load_time = time.time() - start_time
            logger.info(f"MAPPO models loaded successfully in {load_time:.2f}s from {load_dir}")
            
            if load_time > 4.0:
                logger.warning(f"Model loading took {load_time:.2f}s, exceeding 4s requirement")
                
        except Exception as e:
            logger.error(f"Failed to load MAPPO models: {e}")
            raise

    def get_training_metrics(self) -> Dict[str, float]:
        """
        获取当前训练指标
        
        Returns:
            Dict[str, float]: 训练指标字典
        """
        return {
            'policy_loss': self.last_policy_loss,
            'value_loss': self.last_value_loss,
            'entropy_loss': self.last_entropy_loss,
            'total_loss': self.last_policy_loss + self.last_value_loss + self.last_entropy_loss,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'buffer_size': self.buffer.size()
        }