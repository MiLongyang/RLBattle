# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from .rnn_agent import RNNAgent
from .q_mixer import QMixer
from common.episode_replay_buffer import EpisodeReplayBuffer
from ..base_algorithm import BaseAlgorithm

logger = logging.getLogger(__name__)

class QMIX(BaseAlgorithm):
    """
    QMIX 算法主类
    """
    def __init__(self, obs_dims: List[int], action_dims: List[int], num_agents: int, 
                 state_dim: int, args, device: torch.device):
        # 调用父类初始化
        super().__init__(obs_dims, action_dims, num_agents, state_dim, args, device)
        
        # 检查并确认所有智能体是否同构 (QMIX的典型假设)
        assert all(d == obs_dims[0] for d in obs_dims), "QMIX 要求所有智能体的观测维度相同"
        assert all(d == action_dims[0] for d in action_dims), "QMIX 要求所有智能体的动作维度相同"
        self.obs_dim = obs_dims[0]
        self.action_dim = action_dims[0]

        # 初始化网络
        self.agent_net = RNNAgent(self.obs_dim, self.action_dim, args).to(device)
        self.mixer_net = QMixer(num_agents, state_dim, args.mixer_hidden_dim, device).to(device)

        # 初始化目标网络
        self.target_agent_net = RNNAgent(self.obs_dim, self.action_dim, args).to(device)
        self.target_mixer_net = QMixer(num_agents, state_dim, args.mixer_hidden_dim, device).to(device)
        
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())

        # 设置优化器
        network_params = list(self.agent_net.parameters()) + list(self.mixer_net.parameters())
        self.optimizer = optim.Adam(params=network_params, lr=args.lr_qmix)

        # 经验回放池
        # 对于QMIX，全局状态通常是所有智能体观测的拼接
        actual_state_dim = sum(obs_dims)  # 实际的全局状态维度
        self.replay_buffer = EpisodeReplayBuffer(args.buffer_size_qmix, args.episode_limit, num_agents, obs_dims, action_dims, actual_state_dim)
        
        # 训练指标
        self.last_loss = 0.0
        self.target_update_count = 0

    def select_actions(self, observations: List[np.ndarray], 
                      **kwargs) -> Tuple[List[np.ndarray], Optional[List[torch.Tensor]]]:
        """
        为所有智能体选择动作
        
        Args:
            observations: 所有智能体的观测列表
            **kwargs: 可选参数，包括hidden_states, epsilon等
            
        Returns:
            Tuple[actions, new_hidden_states]: 动作列表和新的隐藏状态列表
        """
        # 验证观测数据
        if not self.validate_observations(observations):
            raise ValueError("Invalid observations provided")
        
        # 获取参数
        hidden_states = kwargs.get('hidden_states', None)
        epsilon = kwargs.get('epsilon', 0.0 if not self.training else 0.1)
        
        if hidden_states is None:
            hidden_states = [self.agent_net.init_hidden().to(self.device) for _ in range(self.num_agents)]
        
        actions = []
        new_hidden_states = []
        
        for i, obs in enumerate(observations):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            q_values, h_out = self.agent_net(obs_tensor, hidden_states[i])
            
            if np.random.rand() < epsilon:
                action = np.random.randint(0, self.action_dim)
            else:
                action = q_values.argmax().item()
            
            actions.append(np.array([action]))  # 转换为numpy数组保持一致性
            new_hidden_states.append(h_out)
        
        return actions, new_hidden_states

    def learn(self, **kwargs) -> Dict[str, float]:
        """
        从经验回放池中采样，训练QMIX网络
        
        Returns:
            Dict[str, float]: 训练指标字典
        """
        if self.replay_buffer.size() < self.args.batch_size:
            return {
                'loss': 0.0, 
                'buffer_size': self.replay_buffer.size(),
                'training_step': self.training_step
            }

        batch, episode_lengths = self.replay_buffer.sample(self.args.batch_size, self.device)
        
        max_len = int(episode_lengths.max())
        if max_len == 0:
            return {'loss': 0.0, 'buffer_size': self.replay_buffer.size()}
        
        # 将所有采样到的数据裁剪到当前batch中最长episode的长度
        for key in batch.keys():
            batch[key] = batch[key][:, :max_len]
            
        # 1. 计算当前Q_tot
        q_evals, q_targets = [], []
        hidden_eval = self.agent_net.init_hidden().unsqueeze(0).expand(self.args.batch_size, self.num_agents, -1)
        hidden_target = self.target_agent_net.init_hidden().unsqueeze(0).expand(self.args.batch_size, self.num_agents, -1)

        for t in range(max_len):
            obs_t = batch['obs'][:, t]
            action_t = batch['actions'][:, t].long()
            
            # (batch_size, num_agents, q_values)
            q_eval_t, h_eval_next = self.agent_net(obs_t.reshape(-1, self.obs_dim), hidden_eval.reshape(-1, self.args.agent_hidden_dim))
            q_eval_t = q_eval_t.reshape(self.args.batch_size, self.num_agents, -1)
            hidden_eval = h_eval_next.reshape(self.args.batch_size, self.num_agents, -1)
            # 把采取的动作的Q值拿出来
            q_evals_t_taken = torch.gather(q_eval_t, dim=2, index=action_t).squeeze(2)
            q_evals.append(q_evals_t_taken)

            # 2. 计算目标Q_tot
            with torch.no_grad():
                next_obs_t = batch['next_obs'][:, t]
                q_target_t, h_target_next = self.target_agent_net(next_obs_t.reshape(-1, self.obs_dim), hidden_target.reshape(-1, self.args.agent_hidden_dim))
                q_target_t = q_target_t.reshape(self.args.batch_size, self.num_agents, -1)
                hidden_target = h_target_next.reshape(self.args.batch_size, self.num_agents, -1)
                # 采用 Double-DQN 的思想, 动作由 eval_net 决定
                action_next = q_eval_t.argmax(dim=2, keepdim=True)
                q_target_t_taken = torch.gather(q_target_t, dim=2, index=action_next).squeeze(2)
                q_targets.append(q_target_t_taken)

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)

        q_total_eval = self.mixer_net(q_evals, batch['state'][:, :max_len])
        q_total_target = self.target_mixer_net(q_targets, batch['state'][:, :max_len])
        
        rewards = batch['rewards'][:, :max_len].squeeze(2)
        dones = batch['dones'][:, :max_len].squeeze(2)
        
        # 计算 TD-target
        td_target = rewards + self.args.gamma * (1 - dones) * q_total_target
        
        # 损失计算
        mask = (torch.ones_like(dones) - dones).float()
        loss = F.mse_loss(q_total_eval * mask, td_target.detach() * mask)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.agent_net.parameters()) + list(self.mixer_net.parameters()), 1.0)
        self.optimizer.step()

        # 更新训练步数
        self.update_training_step()
        
        # 更新目标网络
        if self.training_step % self.args.target_update_interval == 0:
            self.update_target_networks()
            self.target_update_count += 1
        
        # 更新指标
        self.last_loss = loss.item()
        metrics = {
            'loss': self.last_loss,
            'buffer_size': self.replay_buffer.size(),
            'training_step': self.training_step,
            'target_updates': self.target_update_count
        }
        
        self.update_metrics(metrics)
        return metrics
            
    def update_target_networks(self):
        """更新目标网络"""
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())
        logger.debug(f"Target networks updated at step {self.training_step}")

    def add_episode(self, episode_batch: Dict, episode_length: int):
        """
        向经验回放池中添加一个完整的episode
        
        Args:
            episode_batch: 包含一个回合数据的字典
            episode_length: 该回合的实际长度
        """
        self.replay_buffer.add_episode(episode_batch, episode_length)

    def save_models(self, save_dir: str, episode: Union[str, int]) -> None:
        """
        保存QMIX的模型
        
        Args:
            save_dir: 保存目录路径
            episode: 回合数或标识符
        """
        start_time = time.time()
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            agent_path = os.path.join(save_dir, f"agent_net_{episode}.pth")
            mixer_path = os.path.join(save_dir, f"mixer_net_{episode}.pth")
            
            torch.save(self.agent_net.state_dict(), agent_path)
            torch.save(self.mixer_net.state_dict(), mixer_path)
            
            # 保存算法元数据
            metadata = {
                'algorithm': 'QMIX',
                'episode': str(episode),
                'training_step': self.training_step,
                'episode_count': self.episode_count,
                'num_agents': self.num_agents,
                'obs_dims': self.obs_dims,
                'action_dims': self.action_dims,
                'target_update_count': self.target_update_count,
                'save_time': time.time()
            }
            
            metadata_path = os.path.join(save_dir, f"metadata_{episode}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            save_time = time.time() - start_time
            logger.info(f"QMIX models saved successfully in {save_time:.2f}s to {save_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save QMIX models: {e}")
            raise

    def load_models(self, load_dir: str, episode: Union[str, int]) -> None:
        """
        加载QMIX的模型
        
        Args:
            load_dir: 加载目录路径
            episode: 回合数或标识符
        """
        start_time = time.time()
        
        try:
            agent_path = os.path.join(load_dir, f"agent_net_{episode}.pth")
            mixer_path = os.path.join(load_dir, f"mixer_net_{episode}.pth")
            
            if not os.path.exists(agent_path) or not os.path.exists(mixer_path):
                raise FileNotFoundError(f"Model files not found at episode {episode}")
            
            self.agent_net.load_state_dict(torch.load(agent_path, map_location=self.device))
            self.mixer_net.load_state_dict(torch.load(mixer_path, map_location=self.device))
            
            # 同步目标网络
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())
            
            load_time = time.time() - start_time
            logger.info(f"QMIX models loaded successfully in {load_time:.2f}s from {load_dir}")
            
            if load_time > 4.0:
                logger.warning(f"Model loading took {load_time:.2f}s, exceeding 4s requirement")
                
        except Exception as e:
            logger.error(f"Failed to load QMIX models: {e}")
            raise

    def get_training_metrics(self) -> Dict[str, float]:
        """
        获取当前训练指标
        
        Returns:
            Dict[str, float]: 训练指标字典
        """
        return {
            'loss': self.last_loss,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'buffer_size': self.replay_buffer.size(),
            'buffer_capacity': self.replay_buffer.capacity,
            'target_update_count': self.target_update_count
        }
