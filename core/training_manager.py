# -*- coding: utf-8 -*-

import torch
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .config_manager import ConfigManager
from algorithms.algorithm_factory import AlgorithmFactory, AlgorithmRegistry
from algorithms.base_algorithm import BaseAlgorithm

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """训练指标数据模型"""
    episode: int
    step: int
    reward: float
    loss: Optional[float] = None
    actor_loss: Optional[float] = None
    critic_loss: Optional[float] = None
    q_loss: Optional[float] = None
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy_loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class TrainingManager:
    """
    统一训练管理器
    负责管理不同算法的训练流程
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.algorithm: Optional[BaseAlgorithm] = None
        self.env = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练状态
        self.is_training = False
        self.current_episode = 0
        self.total_episodes = 0
        self.training_start_time = None
        
        # 指标记录
        self.episode_rewards = []
        self.training_metrics_history = []
        self.best_reward = float('-inf')
        self.best_episode = 0
        
        logger.info(f"TrainingManager initialized with device: {self.device}")
    
    def initialize_training(self, algorithm_name: str, task_type: str, env, 
                          custom_config: Optional[Dict] = None) -> None:
        """
        初始化训练环境
        
        Args:
            algorithm_name: 算法名称
            task_type: 任务类型
            env: 环境实例
            custom_config: 自定义配置
        """
        try:
            logger.info(f"Initializing training: {algorithm_name} for {task_type} task")
            
            # 设置环境
            self.env = env
            env_info = env.get_env_info()
            
            # 获取配置
            config = self.config_manager.get_config(algorithm_name, task_type, custom_config)
            config['device'] = self.device
            
            # 验证配置
            if not self.config_manager.validate_config(config):
                raise ValueError("Invalid configuration")
            
            # 创建算法实例
            self.algorithm = AlgorithmFactory.create_algorithm(
                algorithm_name, env_info, config
            )
            
            # 设置训练参数
            self.total_episodes = config.get('num_episodes', 10000)
            self.current_episode = 0
            
            # 重置指标
            self.episode_rewards.clear()
            self.training_metrics_history.clear()
            self.best_reward = float('-inf')
            self.best_episode = 0
            
            logger.info(f"Training initialized successfully for {algorithm_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize training: {e}")
            raise
    
    def run_training(self, save_dir: str = "./models") -> Dict[str, Any]:
        """
        执行训练主循环
        
        Args:
            save_dir: 模型保存目录
            
        Returns:
            训练结果字典
        """
        if not self.algorithm or not self.env:
            raise RuntimeError("Training not initialized. Call initialize_training first.")
        
        self.is_training = True
        self.training_start_time = time.time()
        
        logger.info(f"Starting training for {self.total_episodes} episodes")
        
        try:
            training_results = {
                'episodes_completed': 0,
                'total_reward': 0,
                'average_reward': 0,
                'best_reward': self.best_reward,
                'best_episode': self.best_episode,
                'training_time': 0,
                'final_metrics': {}
            }
            
            for episode in range(self.total_episodes):
                if not self.is_training:
                    logger.info("Training stopped by user")
                    break
                
                self.current_episode = episode
                episode_result = self._run_episode(episode)
                
                # 更新结果
                episode_reward = episode_result['episode_reward']
                self.episode_rewards.append(episode_reward)
                training_results['total_reward'] += episode_reward
                
                # 更新最佳记录
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.best_episode = episode
                    training_results['best_reward'] = self.best_reward
                    training_results['best_episode'] = self.best_episode
                
                # 记录训练指标
                if 'metrics' in episode_result:
                    self.training_metrics_history.append(episode_result['metrics'])
                
                # 定期保存模型
                save_interval = getattr(self.algorithm.args, 'save_interval', 50)
                if (episode + 1) % save_interval == 0:
                    self._save_model(save_dir, episode + 1)
                
                # 定期输出日志
                log_interval = getattr(self.algorithm.args, 'log_interval', 10)
                if (episode + 1) % log_interval == 0:
                    self._log_training_progress(episode + 1, episode_reward)
                
                training_results['episodes_completed'] = episode + 1
            
            # 训练完成
            self.is_training = False
            training_time = time.time() - self.training_start_time
            training_results['training_time'] = training_time
            training_results['average_reward'] = training_results['total_reward'] / training_results['episodes_completed']
            
            # 最终保存
            self._save_model(save_dir, "final")
            
            # 获取最终指标
            if self.algorithm:
                training_results['final_metrics'] = self.algorithm.get_training_metrics()
            
            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Average reward: {training_results['average_reward']:.2f}")
            logger.info(f"Best reward: {training_results['best_reward']:.2f} at episode {training_results['best_episode']}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.is_training = False
            raise
        finally:
            if self.env:
                self.env.close()
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """
        执行单个训练回合
        
        Args:
            episode: 回合数
            
        Returns:
            回合结果字典
        """
        obs = self.env.reset()
        episode_reward = 0
        step_count = 0
        episode_limit = getattr(self.algorithm.args, 'episode_limit', 500)
        
        # 算法特定的初始化
        algorithm_state = self._initialize_algorithm_state()
        
        while step_count < episode_limit:
            # 选择动作
            actions, new_state = self.algorithm.select_actions(obs, **algorithm_state)
            
            # 环境交互
            next_obs, reward, done, info = self.env.step(actions)
            
            # 算法学习
            learn_info = self._algorithm_learn_step(
                obs, actions, reward, next_obs, done, info, algorithm_state
            )
            
            # 更新状态
            obs = next_obs
            episode_reward += reward
            step_count += 1
            
            # 更新算法状态
            if new_state is not None:
                algorithm_state.update(new_state if isinstance(new_state, dict) else {})
            
            if done:
                break
        
        # 更新回合计数
        self.algorithm.update_episode_count()
        
        return {
            'episode_reward': episode_reward,
            'steps': step_count,
            'metrics': learn_info
        }
    
    def _initialize_algorithm_state(self) -> Dict[str, Any]:
        """
        初始化算法特定状态
        
        Returns:
            算法状态字典
        """
        algorithm_name = self.algorithm.__class__.__name__.upper()
        
        if algorithm_name == "QMIX":
            # QMIX需要hidden_states和epsilon
            hidden_states = [self.algorithm.agent_net.init_hidden().to(self.device) 
                           for _ in range(self.algorithm.num_agents)]
            epsilon = self._get_epsilon()
            return {'hidden_states': hidden_states, 'epsilon': epsilon}
        
        elif algorithm_name == "MADDPG":
            # MADDPG需要add_noise参数
            return {'add_noise': self.algorithm.training}
        
        elif algorithm_name == "MAPPO":
            # MAPPO可能需要全局状态
            return {}
        
        return {}
    
    def _algorithm_learn_step(self, obs: List[np.ndarray], actions: List[np.ndarray],
                            reward: float, next_obs: List[np.ndarray], done: bool,
                            info: Dict[str, Any], algorithm_state: Dict[str, Any]) -> Dict[str, float]:
        """
        执行算法学习步骤
        
        Args:
            obs: 当前观测
            actions: 动作
            reward: 奖励
            next_obs: 下一观测
            done: 是否结束
            info: 环境信息
            algorithm_state: 算法状态
            
        Returns:
            学习指标字典
        """
        algorithm_name = self.algorithm.__class__.__name__.upper()
        
        # 获取全局状态
        state = info.get('state', np.concatenate(obs))
        next_state = info.get('next_state', np.concatenate(next_obs))
        
        if algorithm_name == "MADDPG":
            # MADDPG添加经验并学习
            self.algorithm.add_experience(obs, state, actions, reward, next_obs, next_state, done)
            return self.algorithm.learn()
        
        elif algorithm_name == "QMIX":
            # QMIX需要特殊处理episode buffer
            if not hasattr(self, '_qmix_episode_buffer'):
                self._qmix_episode_buffer = {
                    'obs': [], 'actions': [], 'rewards': [], 'dones': [], 
                    'state': [], 'next_obs': [], 'next_state': []
                }
            
            # 存储当前step的数据
            self._qmix_episode_buffer['obs'].append(obs)
            self._qmix_episode_buffer['actions'].append(actions)
            self._qmix_episode_buffer['rewards'].append([reward])
            self._qmix_episode_buffer['dones'].append([done])
            self._qmix_episode_buffer['state'].append(state)
            self._qmix_episode_buffer['next_obs'].append(next_obs)
            self._qmix_episode_buffer['next_state'].append(next_state)
            
            # 如果回合结束，添加到replay buffer并学习
            if done:
                episode_len = len(self._qmix_episode_buffer['obs'])
                episode_batch = {}
                for key in self._qmix_episode_buffer.keys():
                    episode_batch[key] = np.array(self._qmix_episode_buffer[key])
                
                self.algorithm.add_episode(episode_batch, episode_len)
                
                # 重置episode buffer
                self._qmix_episode_buffer = {
                    'obs': [], 'actions': [], 'rewards': [], 'dones': [], 
                    'state': [], 'next_obs': [], 'next_state': []
                }
                
                # 学习
                if self.algorithm.replay_buffer.size() >= self.algorithm.args.batch_size:
                    return self.algorithm.learn()
            
            return {'loss': 0.0, 'buffer_size': self.algorithm.replay_buffer.size()}
        
        elif algorithm_name == "MAPPO":
            # MAPPO需要特殊处理
            if hasattr(self.algorithm, 'select_actions'):
                # 获取log_probs和values（需要重新计算或从之前存储）
                log_probs = [0.0] * self.algorithm.num_agents  # 简化处理
                values = [0.0] * self.algorithm.num_agents
                
                self.algorithm.add_experience(obs, actions, log_probs, values, reward, done, state)
                
                # 如果buffer满了或回合结束，进行学习
                if self.algorithm.buffer.is_full() or done:
                    return self.algorithm.learn()
            
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        return {}
    
    def _get_epsilon(self) -> float:
        """
        获取当前epsilon值（用于QMIX）
        
        Returns:
            epsilon值
        """
        if not hasattr(self, '_epsilon'):
            self._epsilon = 1.0
            self._min_epsilon = 0.05
            self._epsilon_decay = (self._epsilon - self._min_epsilon) / getattr(
                self.algorithm.args, 'epsilon_decay_steps', 50000
            )
        
        # 更新epsilon
        self._epsilon = max(self._min_epsilon, self._epsilon - self._epsilon_decay)
        return self._epsilon
    
    def _save_model(self, save_dir: str, episode: int) -> None:
        """
        保存模型
        
        Args:
            save_dir: 保存目录
            episode: 回合数
        """
        try:
            algorithm_name = self.algorithm.__class__.__name__
            task_type = getattr(self.algorithm.args, 'task_type', 'unknown')
            model_dir = f"{save_dir}/{algorithm_name}_{task_type}"
            
            self.algorithm.save_models(model_dir, str(episode))
            logger.debug(f"Model saved at episode {episode}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _log_training_progress(self, episode: int, episode_reward: float) -> None:
        """
        记录训练进度
        
        Args:
            episode: 回合数
            episode_reward: 回合奖励
        """
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
        else:
            recent_avg = np.mean(self.episode_rewards)
        
        elapsed_time = time.time() - self.training_start_time
        
        logger.info(f"Episode {episode}/{self.total_episodes}, "
                   f"Reward: {episode_reward:.2f}, "
                   f"Recent Avg: {recent_avg:.2f}, "
                   f"Best: {self.best_reward:.2f}, "
                   f"Time: {elapsed_time:.1f}s")
    
    def stop_training(self) -> None:
        """停止训练"""
        self.is_training = False
        logger.info("Training stop requested")
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        获取训练状态
        
        Returns:
            训练状态字典
        """
        return {
            'is_training': self.is_training,
            'current_episode': self.current_episode,
            'total_episodes': self.total_episodes,
            'progress': self.current_episode / self.total_episodes if self.total_episodes > 0 else 0,
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'recent_rewards': self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards,
            'algorithm': self.algorithm.__class__.__name__ if self.algorithm else None,
            'device': str(self.device)
        }
    
    def get_training_history(self) -> Dict[str, List]:
        """
        获取训练历史
        
        Returns:
            训练历史字典
        """
        return {
            'episode_rewards': self.episode_rewards.copy(),
            'training_metrics': self.training_metrics_history.copy()
        }
    
    def load_checkpoint(self, checkpoint_path: str, episode: str) -> None:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            episode: 回合数
        """
        if not self.algorithm:
            raise RuntimeError("Algorithm not initialized")
        
        try:
            self.algorithm.load_models(checkpoint_path, episode)
            logger.info(f"Checkpoint loaded from {checkpoint_path}/{episode}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise