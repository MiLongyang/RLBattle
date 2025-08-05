# -*- coding: utf-8 -*-

"""
算法模块
提供统一的强化学习算法接口和工厂模式
"""

from .base_algorithm import BaseAlgorithm
from .algorithm_factory import AlgorithmFactory, AlgorithmRegistry
from .algorithm_factory import AlgorithmNotFoundError, AlgorithmInitializationError, IncompatibleAlgorithmError

# 导入具体算法实现
try:
    from .maddpg.maddpg import MADDPG
    MADDPG_AVAILABLE = True
except ImportError:
    MADDPG_AVAILABLE = False

try:
    from .qmix.qmix import QMIX
    QMIX_AVAILABLE = True
except ImportError:
    QMIX_AVAILABLE = False

try:
    from .mappo.mappo import MAPPO
    MAPPO_AVAILABLE = True
except ImportError:
    MAPPO_AVAILABLE = False

# 算法可用性信息
ALGORITHM_AVAILABILITY = {
    'MADDPG': MADDPG_AVAILABLE,
    'QMIX': QMIX_AVAILABLE,
    'MAPPO': MAPPO_AVAILABLE
}

def get_available_algorithms():
    """
    获取可用算法列表
    
    Returns:
        List[str]: 可用算法名称列表
    """
    return [name for name, available in ALGORITHM_AVAILABILITY.items() if available]

def is_algorithm_available(algorithm_name: str) -> bool:
    """
    检查算法是否可用
    
    Args:
        algorithm_name: 算法名称
        
    Returns:
        bool: 是否可用
    """
    return ALGORITHM_AVAILABILITY.get(algorithm_name.upper(), False)

def create_algorithm(algorithm_name: str, env_info: dict, config: dict) -> BaseAlgorithm:
    """
    创建算法实例（便捷函数）
    
    Args:
        algorithm_name: 算法名称
        env_info: 环境信息
        config: 配置参数
        
    Returns:
        BaseAlgorithm: 算法实例
    """
    return AlgorithmFactory.create_algorithm(algorithm_name, env_info, config)

def register_algorithm(name: str, algorithm_class, description: str = "", **kwargs):
    """
    注册算法（便捷函数）
    
    Args:
        name: 算法名称
        algorithm_class: 算法类
        description: 算法描述
        **kwargs: 额外信息
    """
    AlgorithmRegistry.register(name, algorithm_class, description, **kwargs)

def list_registered_algorithms():
    """
    列出已注册的算法
    
    Returns:
        List[str]: 已注册算法名称列表
    """
    return AlgorithmRegistry.list_algorithms()

def get_algorithm_info(algorithm_name: str = None):
    """
    获取算法信息
    
    Args:
        algorithm_name: 算法名称，如果为None则返回所有算法信息
        
    Returns:
        dict: 算法信息
    """
    return AlgorithmRegistry.get_algorithm_info(algorithm_name)

# 导出的公共接口
__all__ = [
    # 基础类和工厂
    'BaseAlgorithm',
    'AlgorithmFactory', 
    'AlgorithmRegistry',
    
    # 异常类
    'AlgorithmNotFoundError',
    'AlgorithmInitializationError', 
    'IncompatibleAlgorithmError',
    
    # 具体算法（如果可用）
    'MADDPG' if MADDPG_AVAILABLE else None,
    'QMIX' if QMIX_AVAILABLE else None,
    'MAPPO' if MAPPO_AVAILABLE else None,
    
    # 工具函数
    'get_available_algorithms',
    'is_algorithm_available',
    'create_algorithm',
    'register_algorithm',
    'list_registered_algorithms',
    'get_algorithm_info',
    
    # 可用性信息
    'ALGORITHM_AVAILABILITY'
]

# 移除None值
__all__ = [item for item in __all__ if item is not None]

# 模块级别的信息
__version__ = "1.0.0"
__author__ = "Algorithm Framework Team"
__description__ = "Multi-Agent Reinforcement Learning Algorithm Framework"

# 在模块加载时显示可用算法信息
import logging
logger = logging.getLogger(__name__)

available_algorithms = get_available_algorithms()
if available_algorithms:
    logger.info(f"Available algorithms: {', '.join(available_algorithms)}")
else:
    logger.warning("No algorithms are available. Please check your installation.")

# 显示注册的算法信息
registered_algorithms = list_registered_algorithms()
if registered_algorithms:
    logger.info(f"Registered algorithms: {', '.join(registered_algorithms)}")

def print_algorithm_summary():
    """打印算法摘要信息"""
    print("=" * 60)
    print("ALGORITHM FRAMEWORK SUMMARY")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Available Algorithms: {', '.join(get_available_algorithms())}")
    print(f"Registered Algorithms: {', '.join(list_registered_algorithms())}")
    print("=" * 60)
    
    # 显示每个算法的详细信息
    for alg_name in get_available_algorithms():
        try:
            info = get_algorithm_info(alg_name)
            print(f"\n{alg_name}:")
            print(f"  Description: {info.get('description', 'N/A')}")
            print(f"  Module: {info.get('module', 'N/A')}")
            print(f"  Action Space: {info.get('action_space', 'N/A')}")
        except Exception as e:
            print(f"\n{alg_name}: Error getting info - {e}")
    
    print("=" * 60)