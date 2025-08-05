# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Union
from .agent import Agent
from common.replay_buffer import ReplayBuffer
from ..base_algorithm import BaseAlgorithm

logger = logging.getLogger(__name__)

class MADDPG(BaseAlgorithm):
    """
    MADDPG 算法主类
    """
    def __init__(self, obs_dims: List[int], action_dims: List[int], num_agents: int, 
                 state_dim: int, args, device: torch.device, 
                 action_space_low: np.ndarray, action_space_high: np.ndarray):
        # 调用父类初始化
        super().__init__(obs_dims, action_dims, num_agents, state_dim, args, device)
        
        # MADDPG特定属性
        self.action_space_low = action_space_low
        self.action_space_high = action_space_high
        
        # 初始化智能体
        # 对于MADDPG，全局状态通常是所有智能体观测的拼接
        actual_state_dim = sum(obs_dims)  # 实际的全局状态维度
        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(
                agent_id=i,
                obs_dim=obs_dims[i],
                action_dim=action_dims[i],
                num_agents=num_agents,
                state_dim=actual_state_dim,  # 使用实际的状态维度
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                device=device,
                action_low=action_space_low,
                action_high=action_space_high,
                action_dims=action_dims  # 传递所有智能体的动作维度
            ))
        
        # 初始化经验回放池
        self.replay_buffer = ReplayBuffer(args.buffer_size_maddpg, self.num_agents, obs_dims, action_dims, actual_state_dim)
        
        # 训练指标
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0

    def select_actions(self, observations: List[np.ndarray], 
                      **kwargs) -> Tuple[List[np.ndarray], Optional[None]]:
        """
        为所有智能体选择动作
        
        Args:
            observations: 所有智能体的观测列表
            **kwargs: 可选参数，包括add_noise等
            
        Returns:
            Tuple[actions, None]: 动作列表和None（MADDPG不需要额外信息）
        """
        # 验证观测数据
        if not self.validate_observations(observations):
            raise ValueError("Invalid observations provided")
        
        # 获取参数
        add_noise = kwargs.get('add_noise', self.training)
        
        actions = []
        for agent, obs in zip(self.agents, observations):
            action = agent.select_action(obs, add_noise, self.args.noise_std)
            actions.append(action)
        
        return actions, None

    def learn(self, **kwargs) -> Dict[str, float]:
        """
        从经验回放池中采样, 训练所有智能体

        Args:
            **kwargs: 可选参数，包括gamma,batch_size等,如果没有提供则使用默认值

        Returns:
            Dict[str, float]: 训练指标字典
        """
        if self.replay_buffer.size() < self.args.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'buffer_size': self.replay_buffer.size()}

        # 采样经验
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, state_batch, next_state_batch = self.replay_buffer.sample(self.args.batch_size)

        # 将numpy数据转换为torch tensor并移动到指定设备
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        done_batch = torch.from_numpy(done_batch).float().to(self.device)
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        
        # 将list of numpy arrays 转换为 list of tensors
        obs_batch_tensor = [torch.from_numpy(obs).float().to(self.device) for obs in obs_batch]
        action_batch_tensor = [torch.from_numpy(act).float().to(self.device) for act in action_batch]
        next_obs_batch_tensor = [torch.from_numpy(obs).float().to(self.device) for obs in next_obs_batch]

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for agent in self.agents:
            # --- 更新 Critic ---
            # 1. 计算目标Q值 (y_i)
            with torch.no_grad():
                next_actions_target = [self.agents[i].target_actor(next_obs_batch_tensor[i]) for i in range(self.num_agents)]
                q_next_target = agent.target_critic(next_state_batch, next_actions_target)
                y = reward_batch + self.args.gamma * (1 - done_batch) * q_next_target
            
            # 2. 计算当前Q值
            q_current = agent.critic(state_batch, action_batch_tensor)

            # 3. 计算Critic损失并更新
            critic_loss = F.mse_loss(q_current, y)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
            agent.critic_optimizer.step()

            # --- 更新 Actor ---
            # 1. 根据当前策略计算动作
            actions_pred = [self.agents[i].actor(obs_batch_tensor[i]) for i in range(self.num_agents)]
            # 2. 计算 Actor 损失
            actor_loss = -agent.critic(state_batch, actions_pred).mean()
            # 3. 更新Actor网络
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()
            
            # 累积损失
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            
        # 所有智能体训练完一个batch后，软更新目标网络
        for agent in self.agents:
            agent.update_target_networks(self.args.tau)
        
        # 更新训练步数
        self.update_training_step()
        
        # 计算平均损失
        avg_actor_loss = total_actor_loss / self.num_agents
        avg_critic_loss = total_critic_loss / self.num_agents
        
        # 更新指标
        metrics = {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'buffer_size': self.replay_buffer.size(),
            'training_step': self.training_step
        }
        
        self.last_actor_loss = avg_actor_loss
        self.last_critic_loss = avg_critic_loss
        self.update_metrics(metrics)
        
        return metrics


    def add_experience(self, obs: List[np.ndarray], state: np.ndarray, actions: List[np.ndarray], 
                      rewards: float, next_obs: List[np.ndarray], next_state: np.ndarray, dones: bool):
        """
        向经验回放池中添加一条经验
        
        Args:
            obs: 当前观测列表
            state: 当前全局状态
            actions: 动作列表
            rewards: 奖励
            next_obs: 下一步观测列表
            next_state: 下一步全局状态
            dones: 是否结束
        """
        self.replay_buffer.add(obs, state, actions, rewards, next_obs, next_state, dones)

    def save_models(self, save_dir: str, episode: Union[str, int]) -> None:
        """
        保存所有智能体的模型
        
        Args:
            save_dir: 保存目录路径
            episode: 回合数或标识符
        """
        start_time = time.time()
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            for i, agent in enumerate(self.agents):
                actor_path = os.path.join(save_dir, f"actor_agent_{i}_{episode}.pth")
                critic_path = os.path.join(save_dir, f"critic_agent_{i}_{episode}.pth")
                
                torch.save(agent.actor.state_dict(), actor_path)
                torch.save(agent.critic.state_dict(), critic_path)
            
            # 保存算法元数据
            metadata = {
                'algorithm': 'MADDPG',
                'episode': str(episode),
                'training_step': self.training_step,
                'episode_count': self.episode_count,
                'num_agents': self.num_agents,
                'obs_dims': self.obs_dims,
                'action_dims': self.action_dims,
                'save_time': time.time()
            }
            
            import json
            metadata_path = os.path.join(save_dir, f"metadata_{episode}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            save_time = time.time() - start_time
            logger.info(f"MADDPG models saved successfully in {save_time:.2f}s to {save_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save MADDPG models: {e}")
            raise

    def load_models(self, load_dir: str, episode: Union[str, int]) -> None:
        """
        加载所有智能体的模型
        
        Args:
            load_dir: 加载目录路径
            episode: 回合数或标识符
        """
        start_time = time.time()
        
        try:
            for i, agent in enumerate(self.agents):
                actor_path = os.path.join(load_dir, f"actor_agent_{i}_{episode}.pth")
                critic_path = os.path.join(load_dir, f"critic_agent_{i}_{episode}.pth")
                
                if not os.path.exists(actor_path) or not os.path.exists(critic_path):
                    raise FileNotFoundError(f"Model files not found for agent {i} at episode {episode}")
                
                agent.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                agent.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                
                # 同步目标网络
                agent.target_actor.load_state_dict(agent.actor.state_dict())
                agent.target_critic.load_state_dict(agent.critic.state_dict())
            
            load_time = time.time() - start_time
            logger.info(f"MADDPG models loaded successfully in {load_time:.2f}s from {load_dir}")
            
            if load_time > 4.0:
                logger.warning(f"Model loading took {load_time:.2f}s, exceeding 4s requirement")
                
        except Exception as e:
            logger.error(f"Failed to load MADDPG models: {e}")
            raise

    def get_training_metrics(self) -> Dict[str, float]:
        """
        获取当前训练指标
        
        Returns:
            Dict[str, float]: 训练指标字典
        """
        return {
            'actor_loss': self.last_actor_loss,
            'critic_loss': self.last_critic_loss,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'buffer_size': self.replay_buffer.size(),
            'buffer_capacity': self.replay_buffer.capacity
        }
