# -*- coding: utf-8 -*-

import torch
import logging
from typing import Dict, Any, Type, List
from .base_algorithm import BaseAlgorithm

logger = logging.getLogger(__name__)

class ConfigObject:
    """
    将字典转换为对象，支持属性访问
    """
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)

class AlgorithmRegistry:
    """
    算法注册系统
    支持动态注册和发现算法
    """
    _algorithms: Dict[str, Type[BaseAlgorithm]] = {}
    _algorithm_info: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, algorithm_class: Type[BaseAlgorithm], 
                 description: str = "", **kwargs) -> None:
        """
        注册算法
        
        Args:
            name: 算法名称
            algorithm_class: 算法类
            description: 算法描述
            **kwargs: 额外信息
        """
        if not issubclass(algorithm_class, BaseAlgorithm):
            raise ValueError(f"Algorithm class {algorithm_class.__name__} must inherit from BaseAlgorithm")
        
        # 验证算法类是否实现了所有必要的抽象方法
        cls._validate_algorithm_interface(algorithm_class)
        
        name_upper = name.upper()
        cls._algorithms[name_upper] = algorithm_class
        cls._algorithm_info[name_upper] = {
            'name': name,
            'class': algorithm_class.__name__,
            'description': description,
            'module': algorithm_class.__module__,
            **kwargs
        }
        
        logger.info(f"Algorithm '{name}' registered successfully")
    
    @classmethod
    def _validate_algorithm_interface(cls, algorithm_class: Type[BaseAlgorithm]) -> None:
        """
        验证算法接口完整性
        
        Args:
            algorithm_class: 算法类
        """
        required_methods = [
            'select_actions', 'learn', 'save_models', 
            'load_models', 'get_training_metrics'
        ]
        
        for method_name in required_methods:
            if not hasattr(algorithm_class, method_name):
                raise ValueError(f"Algorithm class {algorithm_class.__name__} missing required method: {method_name}")
            
            method = getattr(algorithm_class, method_name)
            if not callable(method):
                raise ValueError(f"Algorithm class {algorithm_class.__name__} method {method_name} is not callable")
    
    @classmethod
    def get_algorithm(cls, name: str) -> Type[BaseAlgorithm]:
        """
        获取算法类
        
        Args:
            name: 算法名称
            
        Returns:
            算法类
        """
        name_upper = name.upper()
        if name_upper not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Algorithm '{name}' not found. Available algorithms: {available}")
        
        return cls._algorithms[name_upper]
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """
        列出所有已注册的算法
        
        Returns:
            算法名称列表
        """
        return list(cls._algorithms.keys())
    
    @classmethod
    def get_algorithm_info(cls, name: str = None) -> Dict[str, Any]:
        """
        获取算法信息
        
        Args:
            name: 算法名称，如果为None则返回所有算法信息
            
        Returns:
            算法信息字典
        """
        if name is None:
            return cls._algorithm_info.copy()
        
        name_upper = name.upper()
        if name_upper not in cls._algorithm_info:
            raise ValueError(f"Algorithm '{name}' not found")
        
        return cls._algorithm_info[name_upper].copy()
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        检查算法是否已注册
        
        Args:
            name: 算法名称
            
        Returns:
            是否已注册
        """
        return name.upper() in cls._algorithms
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        取消注册算法
        
        Args:
            name: 算法名称
        """
        name_upper = name.upper()
        if name_upper in cls._algorithms:
            del cls._algorithms[name_upper]
            del cls._algorithm_info[name_upper]
            logger.info(f"Algorithm '{name}' unregistered")
        else:
            logger.warning(f"Algorithm '{name}' not found for unregistration")


class AlgorithmFactory:
    """
    算法工厂类
    负责创建算法实例
    """
    
    @staticmethod
    def create_algorithm(algorithm_name: str, env_info: Dict[str, Any], 
                        config: Dict[str, Any]) -> BaseAlgorithm:
        """
        创建算法实例
        
        Args:
            algorithm_name: 算法名称
            env_info: 环境信息
            config: 配置参数
            
        Returns:
            算法实例
        """
        try:
            # 获取算法类
            algorithm_class = AlgorithmRegistry.get_algorithm(algorithm_name)
            
            # 准备初始化参数
            init_params = AlgorithmFactory._prepare_init_params(
                algorithm_name, algorithm_class, env_info, config
            )
            
            # 创建实例
            algorithm_instance = algorithm_class(**init_params)
            
            logger.info(f"Algorithm '{algorithm_name}' created successfully")
            return algorithm_instance
            
        except Exception as e:
            logger.error(f"Failed to create algorithm '{algorithm_name}': {e}")
            raise
    
    @staticmethod
    def _prepare_init_params(algorithm_name: str, algorithm_class: Type[BaseAlgorithm],
                           env_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备算法初始化参数
        
        Args:
            algorithm_name: 算法名称
            algorithm_class: 算法类
            env_info: 环境信息
            config: 配置参数
            
        Returns:
            初始化参数字典
        """
        # 将配置字典转换为对象，支持属性访问
        config_obj = ConfigObject(config)
        
        # 基础参数
        base_params = {
            'obs_dims': env_info["obs_dims"],
            'action_dims': env_info["action_dims"],
            'num_agents': env_info["num_agents"],
            'state_dim': env_info["state_dim"],
            'args': config_obj,
            'device': config.get('device', torch.device('cpu'))
        }
        
        # 算法特定参数
        algorithm_name_upper = algorithm_name.upper()
        
        if algorithm_name_upper == "MADDPG":
            # MADDPG需要动作空间范围
            base_params.update({
                'action_space_low': env_info.get("action_space_low", -1.0),
                'action_space_high': env_info.get("action_space_high", 1.0)
            })
        
        return base_params
    
    @staticmethod
    def get_supported_algorithms() -> List[str]:
        """
        获取支持的算法列表
        
        Returns:
            支持的算法名称列表
        """
        return AlgorithmRegistry.list_algorithms()
    
    @staticmethod
    def validate_algorithm_config(algorithm_name: str, config: Dict[str, Any]) -> bool:
        """
        验证算法配置
        
        Args:
            algorithm_name: 算法名称
            config: 配置参数
            
        Returns:
            配置是否有效
        """
        try:
            algorithm_class = AlgorithmRegistry.get_algorithm(algorithm_name)
            
            # 检查必要的配置参数
            required_keys = ['device']
            for key in required_keys:
                if key not in config:
                    logger.error(f"Missing required config key: {key}")
                    return False
            
            # 算法特定验证
            algorithm_name_upper = algorithm_name.upper()
            
            if algorithm_name_upper == "MADDPG":
                maddpg_keys = ['actor_lr', 'critic_lr', 'buffer_size_maddpg', 'tau', 'noise_std']
                for key in maddpg_keys:
                    if key not in config:
                        logger.warning(f"MADDPG config missing optional key: {key}")
            
            elif algorithm_name_upper == "QMIX":
                qmix_keys = ['lr_qmix', 'buffer_size_qmix', 'target_update_interval', 
                           'mixer_hidden_dim', 'agent_hidden_dim']
                for key in qmix_keys:
                    if key not in config:
                        logger.warning(f"QMIX config missing optional key: {key}")
            
            elif algorithm_name_upper == "MAPPO":
                mappo_keys = ['actor_lr_mappo', 'critic_lr_mappo', 'buffer_size_mappo', 
                            'clip_ratio', 'ppo_epochs']
                for key in mappo_keys:
                    if key not in config:
                        logger.warning(f"MAPPO config missing optional key: {key}")
            
            return True
            
        except Exception as e:
            logger.error(f"Algorithm config validation failed: {e}")
            return False


# 自动注册内置算法
def _register_builtin_algorithms():
    """注册内置算法"""
    try:
        from .maddpg.maddpg import MADDPG
        AlgorithmRegistry.register(
            "MADDPG", 
            MADDPG, 
            "Multi-Agent Deep Deterministic Policy Gradient",
            action_space="continuous",
            paper="https://arxiv.org/abs/1706.02275"
        )
    except ImportError as e:
        logger.warning(f"Failed to register MADDPG: {e}")
    
    try:
        from .qmix.qmix import QMIX
        AlgorithmRegistry.register(
            "QMIX", 
            QMIX, 
            "Q-Mix: Monotonic Value Function Factorisation",
            action_space="discrete",
            paper="https://arxiv.org/abs/1803.11485"
        )
    except ImportError as e:
        logger.warning(f"Failed to register QMIX: {e}")
    
    try:
        from .mappo.mappo import MAPPO
        AlgorithmRegistry.register(
            "MAPPO", 
            MAPPO, 
            "Multi-Agent Proximal Policy Optimization",
            action_space="both",
            paper="https://arxiv.org/abs/2103.01955"
        )
    except ImportError as e:
        logger.warning(f"Failed to register MAPPO: {e}")


# 在模块加载时自动注册内置算法
_register_builtin_algorithms()


class AlgorithmNotFoundError(Exception):
    """算法未找到异常"""
    pass


class AlgorithmInitializationError(Exception):
    """算法初始化异常"""
    pass


class IncompatibleAlgorithmError(Exception):
    """算法不兼容异常"""
    pass