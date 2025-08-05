# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union
import torch
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class BaseAlgorithm(ABC):
    """
    所有强化学习算法的抽象基类。
    定义了所有算法必须实现的通用接口，以确保框架的可扩展性。
    任何新的算法都应该继承这个类，并实现其所有抽象方法。
    """
    
    def __init__(self, obs_dims: List[int], action_dims: List[int], 
                 num_agents: int, state_dim: int, args: Any, device: torch.device):
        """
        初始化算法基础属性
        
        Args:
            obs_dims: 每个智能体的观测维度列表
            action_dims: 每个智能体的动作维度列表  
            num_agents: 智能体数量
            state_dim: 全局状态维度
            args: 算法配置参数
            device: 计算设备
        """
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.args = args
        self.device = device
        
        # 训练状态管理
        self.training_step = 0
        self.episode_count = 0
        self.training = True
        
        # 性能指标
        self.training_metrics = {}
        self.last_metrics_update = time.time()

    @abstractmethod
    def select_actions(self, observations: List[np.ndarray], 
                      **kwargs) -> Tuple[List[np.ndarray], Optional[Any]]:
        """
        根据当前状态为所有智能体选择动作
        
        Args:
            observations: 所有智能体的观测列表
            **kwargs: 算法特定的额外参数（如hidden_states, epsilon, add_noise等）
            
        Returns:
            Tuple[actions, additional_info]: 
                - actions: 所有智能体的动作列表
                - additional_info: 算法特定的额外信息（如新的hidden_states）
        """
        pass

    @abstractmethod
    def learn(self, **kwargs) -> Dict[str, float]:
        """
        执行一次学习/更新步骤
        
        Args:
            **kwargs: 算法特定的学习参数
            
        Returns:
            Dict[str, float]: 训练指标字典（如loss, actor_loss, critic_loss等）
        """
        pass

    @abstractmethod
    def save_models(self, save_dir: str, episode: Union[str, int]) -> None:
        """
        保存模型权重和相关信息
        
        Args:
            save_dir: 保存目录路径
            episode: 回合数或标识符
        """
        pass

    @abstractmethod
    def load_models(self, load_dir: str, episode: Union[str, int]) -> None:
        """
        加载模型权重和相关信息
        
        Args:
            load_dir: 加载目录路径
            episode: 回合数或标识符
        """
        pass

    @abstractmethod
    def get_training_metrics(self) -> Dict[str, float]:
        """
        获取当前训练指标
        
        Returns:
            Dict[str, float]: 训练指标字典
        """
        pass

    def set_training_mode(self, training: bool = True) -> None:
        """
        设置训练/评估模式
        
        Args:
            training: True为训练模式，False为评估模式
        """
        self.training = training
        logger.debug(f"Algorithm mode set to {'training' if training else 'evaluation'}")

    def update_training_step(self) -> None:
        """更新训练步数"""
        self.training_step += 1

    def update_episode_count(self) -> None:
        """更新回合数"""
        self.episode_count += 1
        logger.debug(f"Episode count updated to {self.episode_count}")

    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        获取算法基本信息
        
        Returns:
            Dict[str, Any]: 算法信息字典
        """
        return {
            'algorithm_name': self.__class__.__name__,
            'num_agents': self.num_agents,
            'obs_dims': self.obs_dims,
            'action_dims': self.action_dims,
            'state_dim': self.state_dim,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'training_mode': self.training,
            'device': str(self.device)
        }

    def validate_observations(self, observations: List[np.ndarray]) -> bool:
        """
        验证观测数据的有效性
        
        Args:
            observations: 观测数据列表
            
        Returns:
            bool: 验证是否通过
        """
        if len(observations) != self.num_agents:
            logger.error(f"Expected {self.num_agents} observations, got {len(observations)}")
            return False
            
        for i, obs in enumerate(observations):
            if obs.shape[-1] != self.obs_dims[i]:
                logger.error(f"Agent {i} observation dimension mismatch: "
                           f"expected {self.obs_dims[i]}, got {obs.shape[-1]}")
                return False
                
        return True

    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """
        更新训练指标
        
        Args:
            new_metrics: 新的指标字典
        """
        self.training_metrics.update(new_metrics)
        self.last_metrics_update = time.time()

    def reset_metrics(self) -> None:
        """重置训练指标"""
        self.training_metrics.clear()
        self.last_metrics_update = time.time()

    def __str__(self) -> str:
        """返回算法的字符串表示"""
        return f"{self.__class__.__name__}(agents={self.num_agents}, step={self.training_step})"

    def __repr__(self) -> str:
        """返回算法的详细字符串表示"""
        return (f"{self.__class__.__name__}("
                f"num_agents={self.num_agents}, "
                f"obs_dims={self.obs_dims}, "
                f"action_dims={self.action_dims}, "
                f"training_step={self.training_step})") 