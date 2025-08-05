# -*- coding: utf-8 -*-

import json
import yaml
import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

@dataclass
class AlgorithmConfig:
    """算法配置数据模型"""
    algorithm_name: str
    task_type: str
    num_episodes: int
    batch_size: int
    gamma: float
    device: str
    save_interval: int
    
    # 算法特定参数
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """验证配置有效性"""
        required_fields = ['algorithm_name', 'task_type', 'num_episodes']
        return all(hasattr(self, field) and getattr(self, field) is not None 
                  for field in required_fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class ConfigManager:
    """
    配置管理器
    负责管理算法配置、任务映射和参数验证
    """
    
    def __init__(self):
        self.default_configs = self._load_default_configs()
        self.task_algorithm_mapping = {
            'recon': 'MADDPG',      # 侦察任务推荐MADDPG
            'feint': 'MADDPG',      # 佯攻任务推荐MADDPG（可选QMIX）
            'strike': 'QMIX'        # 协同打击任务推荐QMIX
        }
        self.algorithm_task_compatibility = {
            'MADDPG': ['recon', 'feint', 'strike'],
            'QMIX': ['feint', 'strike'],
            'MAPPO': ['recon', 'feint', 'strike']  # MAPPO作为通用算法
        }
    
    def get_config(self, algorithm_name: str, task_type: str, 
                   custom_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        获取算法配置
        
        Args:
            algorithm_name: 算法名称
            task_type: 任务类型
            custom_config: 自定义配置
            
        Returns:
            最终配置字典
        """
        algorithm_name = algorithm_name.upper()
        
        # 验证算法和任务兼容性
        if not self._is_compatible(algorithm_name, task_type):
            logger.warning(f"Algorithm {algorithm_name} may not be optimal for task {task_type}")
        
        # 基础配置
        base_config = self.default_configs.get(algorithm_name, {}).copy()
        
        # 任务特定配置
        task_config = self._get_task_specific_config(algorithm_name, task_type)
        
        # 合并配置（优先级：custom_config > task_config > base_config）
        final_config = {**base_config, **task_config}
        
        if custom_config:
            final_config.update(custom_config)
        
        # 添加基本信息
        final_config.update({
            'algorithm_name': algorithm_name,
            'task_type': task_type
        })
        
        return final_config
    
    def get_recommended_algorithm(self, task_type: str) -> str:
        """
        获取推荐算法
        
        Args:
            task_type: 任务类型
            
        Returns:
            推荐的算法名称
        """
        return self.task_algorithm_mapping.get(task_type, 'MADDPG')
    
    def get_alternative_algorithms(self, task_type: str) -> list:
        """
        获取任务的备选算法
        
        Args:
            task_type: 任务类型
            
        Returns:
            备选算法列表
        """
        alternatives = []
        for algorithm, compatible_tasks in self.algorithm_task_compatibility.items():
            if task_type in compatible_tasks:
                alternatives.append(algorithm)
        return alternatives
    
    def _is_compatible(self, algorithm_name: str, task_type: str) -> bool:
        """
        检查算法和任务的兼容性
        
        Args:
            algorithm_name: 算法名称
            task_type: 任务类型
            
        Returns:
            是否兼容
        """
        compatible_tasks = self.algorithm_task_compatibility.get(algorithm_name, [])
        return task_type in compatible_tasks
    
    def _get_task_specific_config(self, algorithm_name: str, task_type: str) -> Dict[str, Any]:
        """
        获取任务特定配置
        
        Args:
            algorithm_name: 算法名称
            task_type: 任务类型
            
        Returns:
            任务特定配置字典
        """
        task_configs = {
            'recon': {  # 侦察任务配置
                'MADDPG': {
                    'actor_lr': 1e-4,
                    'critic_lr': 1e-3,
                    'noise_std': 0.05,  # 侦察任务需要更小的探索噪声
                    'tau': 1e-3,
                    'buffer_size_maddpg': int(5e5),
                    'description': '侦察任务优化：降低探索噪声，提高隐蔽性'
                },
                'MAPPO': {
                    'actor_lr_mappo': 2e-4,
                    'critic_lr_mappo': 5e-4,
                    'entropy_coef': 0.005,  # 降低熵系数，减少随机性
                    'clip_ratio': 0.15,
                    'description': '侦察任务优化：降低随机性，提高策略稳定性'
                }
            },
            'feint': {  # 佯攻任务配置
                'MADDPG': {
                    'actor_lr': 2e-4,
                    'critic_lr': 2e-3,
                    'noise_std': 0.15,  # 佯攻任务需要更多探索
                    'tau': 2e-3,
                    'buffer_size_maddpg': int(8e5),
                    'description': '佯攻任务优化：增加探索性，提高欺骗效果'
                },
                'QMIX': {
                    'lr_qmix': 1e-3,
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.1,
                    'epsilon_decay_steps': 30000,
                    'target_update_interval': 150,
                    'description': '佯攻任务优化：平衡探索与利用，提高协调性'
                },
                'MAPPO': {
                    'actor_lr_mappo': 3e-4,
                    'critic_lr_mappo': 1e-3,
                    'entropy_coef': 0.02,  # 增加熵系数，提高探索性
                    'clip_ratio': 0.25,
                    'description': '佯攻任务优化：增加探索性和策略多样性'
                }
            },
            'strike': {  # 协同打击任务配置
                'QMIX': {
                    'lr_qmix': 5e-4,
                    'target_update_interval': 200,
                    'mixer_hidden_dim': 64,  # 协同打击需要更大的混合网络
                    'agent_hidden_dim': 128,
                    'buffer_size_qmix': 8000,
                    'description': '协同打击优化：增强网络容量，提高协同效果'
                },
                'MAPPO': {
                    'actor_lr_mappo': 3e-4,
                    'critic_lr_mappo': 1e-3,
                    'use_centralized_critic': True,  # 协同任务使用中心化Critic
                    'ppo_epochs': 6,  # 增加更新轮数
                    'description': '协同打击优化：使用中心化Critic，增强协调能力'
                },
                'MADDPG': {
                    'actor_lr': 1e-4,
                    'critic_lr': 1e-3,
                    'noise_std': 0.1,
                    'tau': 1e-3,
                    'description': '协同打击备选：连续动作空间的协同控制'
                }
            }
        }
        
        return task_configs.get(task_type, {}).get(algorithm_name, {})
    
    def _load_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        加载默认配置
        
        Returns:
            默认配置字典
        """
        return {
            'MADDPG': {
                'actor_lr': 1e-4,
                'critic_lr': 1e-3,
                'tau': 1e-3,
                'noise_std': 0.1,
                'buffer_size_maddpg': int(1e6),
                'batch_size': 256,
                'gamma': 0.99
            },
            'QMIX': {
                'lr_qmix': 5e-4,
                'target_update_interval': 200,
                'mixer_hidden_dim': 32,
                'agent_hidden_dim': 64,
                'buffer_size_qmix': 5000,
                'batch_size': 32,
                'gamma': 0.99,
                'epsilon_decay_steps': 50000
            },
            'MAPPO': {
                'actor_lr_mappo': 3e-4,
                'critic_lr_mappo': 1e-3,
                'clip_ratio': 0.2,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5,
                'ppo_epochs': 4,
                'mini_batch_size': 64,
                'buffer_size_mappo': 2048,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'use_centralized_critic': True,
                'action_type': 'discrete'
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置有效性
        
        Args:
            config: 配置字典
            
        Returns:
            是否有效
        """
        try:
            # 检查必要字段
            required_fields = ['algorithm_name', 'task_type']
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required config field: {field}")
                    return False
            
            # 检查算法特定参数
            algorithm_name = config['algorithm_name'].upper()
            if algorithm_name == 'MADDPG':
                required_params = ['actor_lr', 'critic_lr', 'tau', 'noise_std']
            elif algorithm_name == 'QMIX':
                required_params = ['lr_qmix', 'target_update_interval']
            elif algorithm_name == 'MAPPO':
                required_params = ['actor_lr_mappo', 'critic_lr_mappo', 'clip_ratio']
            else:
                logger.warning(f"Unknown algorithm: {algorithm_name}")
                return True  # 对未知算法宽松处理
            
            # 检查参数类型和范围
            for param in required_params:
                if param not in config:
                    logger.warning(f"Missing algorithm parameter: {param}")
                    continue
                
                value = config[param]
                if param.endswith('_lr') or param in ['tau', 'noise_std', 'clip_ratio']:
                    if not isinstance(value, (int, float)) or value <= 0:
                        logger.error(f"Invalid value for {param}: {value}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False
    
    def save_config(self, config: Dict[str, Any], file_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            file_path: 文件路径
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    json.dump(config, f, indent=2, ensure_ascii=False)
                elif file_path.endswith(('.yaml', '.yml')):
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Config saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    def load_config(self, file_path: str) -> Dict[str, Any]:
        """
        从文件加载配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            配置字典
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Config file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    config = json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Config loaded from: {file_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def get_config_template(self, algorithm_name: str, task_type: str) -> Dict[str, Any]:
        """
        获取配置模板
        
        Args:
            algorithm_name: 算法名称
            task_type: 任务类型
            
        Returns:
            配置模板
        """
        template = self.get_config(algorithm_name, task_type)
        
        # 添加注释信息
        template['_info'] = {
            'algorithm': algorithm_name,
            'task': task_type,
            'description': f"Configuration template for {algorithm_name} on {task_type} task",
            'recommended': algorithm_name == self.get_recommended_algorithm(task_type),
            'alternatives': self.get_alternative_algorithms(task_type)
        }
        
        return template
    
    def compare_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较两个配置的差异
        
        Args:
            config1: 配置1
            config2: 配置2
            
        Returns:
            差异报告
        """
        differences = {
            'only_in_config1': {},
            'only_in_config2': {},
            'different_values': {}
        }
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            if key not in config2:
                differences['only_in_config1'][key] = config1[key]
            elif key not in config1:
                differences['only_in_config2'][key] = config2[key]
            elif config1[key] != config2[key]:
                differences['different_values'][key] = {
                    'config1': config1[key],
                    'config2': config2[key]
                }
        
        return differences