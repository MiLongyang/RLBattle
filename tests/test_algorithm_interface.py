# -*- coding: utf-8 -*-

"""
算法接口一致性测试
验证所有算法都正确实现了BaseAlgorithm接口
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

from algorithms.base_algorithm import BaseAlgorithm
from algorithms.algorithm_factory import AlgorithmFactory, AlgorithmRegistry
from algorithms import get_available_algorithms

class TestAlgorithmInterface(unittest.TestCase):
    """测试算法接口一致性"""
    
    def setUp(self):
        """设置测试环境"""
        self.device = torch.device('cpu')
        self.mock_env_info = {
            'obs_dims': [10, 10, 10],
            'action_dims': [2, 2, 2],
            'num_agents': 3,
            'state_dim': 30,
            'action_space_low': np.array([-1.0, -1.0]),
            'action_space_high': np.array([1.0, 1.0])
        }
        
        # 创建模拟配置
        self.mock_config = {
            'device': self.device,
            'batch_size': 32,
            'gamma': 0.99,
            'num_episodes': 100,
            'episode_limit': 50,
            'task_type': 'test'
        }
        
        # 为每个算法添加特定配置
        self.algorithm_configs = {
            'MADDPG': {
                **self.mock_config,
                'actor_lr': 1e-4,
                'critic_lr': 1e-3,
                'tau': 1e-3,
                'noise_std': 0.1,
                'buffer_size_maddpg': 1000
            },
            'QMIX': {
                **self.mock_config,
                'lr_qmix': 5e-4,
                'target_update_interval': 200,
                'mixer_hidden_dim': 32,
                'agent_hidden_dim': 64,
                'buffer_size_qmix': 100,
                'epsilon_decay_steps': 1000
            },
            'MAPPO': {
                **self.mock_config,
                'actor_lr_mappo': 3e-4,
                'critic_lr_mappo': 1e-3,
                'clip_ratio': 0.2,
                'ppo_epochs': 4,
                'buffer_size_mappo': 128,
                'action_type': 'discrete'
            }
        }
    
    def test_all_algorithms_inherit_base_algorithm(self):
        """测试所有算法都继承BaseAlgorithm"""
        available_algorithms = get_available_algorithms()
        
        for algorithm_name in available_algorithms:
            with self.subTest(algorithm=algorithm_name):
                try:
                    config = self.algorithm_configs.get(algorithm_name, self.mock_config)
                    algorithm = AlgorithmFactory.create_algorithm(
                        algorithm_name, self.mock_env_info, config
                    )
                    
                    self.assertIsInstance(algorithm, BaseAlgorithm,
                                        f"{algorithm_name} should inherit from BaseAlgorithm")
                    
                except Exception as e:
                    self.fail(f"Failed to create {algorithm_name}: {e}")
    
    def test_select_actions_interface(self):
        """测试select_actions方法接口一致性"""
        available_algorithms = get_available_algorithms()
        
        for algorithm_name in available_algorithms:
            with self.subTest(algorithm=algorithm_name):
                try:
                    config = self.algorithm_configs.get(algorithm_name, self.mock_config)
                    algorithm = AlgorithmFactory.create_algorithm(
                        algorithm_name, self.mock_env_info, config
                    )
                    
                    # 创建测试观测
                    observations = [
                        np.random.randn(obs_dim) 
                        for obs_dim in self.mock_env_info['obs_dims']
                    ]
                    
                    # 测试方法存在
                    self.assertTrue(hasattr(algorithm, 'select_actions'),
                                  f"{algorithm_name} should have select_actions method")
                    
                    # 测试方法调用
                    if algorithm_name == 'QMIX':
                        # QMIX需要hidden_states
                        hidden_states = [algorithm.agent_net.init_hidden().to(self.device) 
                                       for _ in range(self.mock_env_info['num_agents'])]
                        actions, new_state = algorithm.select_actions(
                            observations, hidden_states=hidden_states, epsilon=0.1
                        )
                    elif algorithm_name == 'MADDPG':
                        # MADDPG需要add_noise参数
                        actions, new_state = algorithm.select_actions(
                            observations, add_noise=False
                        )
                    else:
                        # MAPPO或其他算法
                        actions, new_state = algorithm.select_actions(observations)
                    
                    # 验证返回值格式
                    self.assertIsInstance(actions, list,
                                        f"{algorithm_name} should return list of actions")
                    self.assertEqual(len(actions), self.mock_env_info['num_agents'],
                                   f"{algorithm_name} should return actions for all agents")
                    
                    for i, action in enumerate(actions):
                        self.assertIsInstance(action, np.ndarray,
                                            f"{algorithm_name} action {i} should be numpy array")
                    
                except Exception as e:
                    self.fail(f"select_actions test failed for {algorithm_name}: {e}")
    
    def test_learn_interface(self):
        """测试learn方法接口一致性"""
        available_algorithms = get_available_algorithms()
        
        for algorithm_name in available_algorithms:
            with self.subTest(algorithm=algorithm_name):
                try:
                    config = self.algorithm_configs.get(algorithm_name, self.mock_config)
                    algorithm = AlgorithmFactory.create_algorithm(
                        algorithm_name, self.mock_env_info, config
                    )
                    
                    # 测试方法存在
                    self.assertTrue(hasattr(algorithm, 'learn'),
                                  f"{algorithm_name} should have learn method")
                    
                    # 测试方法调用（可能返回空结果，因为没有足够的经验）
                    metrics = algorithm.learn()
                    
                    # 验证返回值格式
                    self.assertIsInstance(metrics, dict,
                                        f"{algorithm_name} learn should return dict")
                    
                    # 验证返回值包含数值
                    for key, value in metrics.items():
                        self.assertIsInstance(value, (int, float),
                                            f"{algorithm_name} metric {key} should be numeric")
                    
                except Exception as e:
                    self.fail(f"learn test failed for {algorithm_name}: {e}")
    
    def test_save_load_models_interface(self):
        """测试模型保存/加载接口一致性"""
        available_algorithms = get_available_algorithms()
        
        for algorithm_name in available_algorithms:
            with self.subTest(algorithm=algorithm_name):
                try:
                    config = self.algorithm_configs.get(algorithm_name, self.mock_config)
                    algorithm = AlgorithmFactory.create_algorithm(
                        algorithm_name, self.mock_env_info, config
                    )
                    
                    # 测试方法存在
                    self.assertTrue(hasattr(algorithm, 'save_models'),
                                  f"{algorithm_name} should have save_models method")
                    self.assertTrue(hasattr(algorithm, 'load_models'),
                                  f"{algorithm_name} should have load_models method")
                    
                    # 测试保存（使用临时目录）
                    import tempfile
                    import os
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        save_dir = os.path.join(temp_dir, f"test_{algorithm_name}")
                        
                        # 测试保存
                        algorithm.save_models(save_dir, "test")
                        
                        # 验证文件被创建
                        self.assertTrue(os.path.exists(save_dir),
                                      f"{algorithm_name} should create save directory")
                        
                        # 测试加载
                        algorithm.load_models(save_dir, "test")
                    
                except Exception as e:
                    self.fail(f"save/load test failed for {algorithm_name}: {e}")
    
    def test_get_training_metrics_interface(self):
        """测试get_training_metrics方法接口一致性"""
        available_algorithms = get_available_algorithms()
        
        for algorithm_name in available_algorithms:
            with self.subTest(algorithm=algorithm_name):
                try:
                    config = self.algorithm_configs.get(algorithm_name, self.mock_config)
                    algorithm = AlgorithmFactory.create_algorithm(
                        algorithm_name, self.mock_env_info, config
                    )
                    
                    # 测试方法存在
                    self.assertTrue(hasattr(algorithm, 'get_training_metrics'),
                                  f"{algorithm_name} should have get_training_metrics method")
                    
                    # 测试方法调用
                    metrics = algorithm.get_training_metrics()
                    
                    # 验证返回值格式
                    self.assertIsInstance(metrics, dict,
                                        f"{algorithm_name} get_training_metrics should return dict")
                    
                    # 验证返回值包含数值
                    for key, value in metrics.items():
                        self.assertIsInstance(value, (int, float),
                                            f"{algorithm_name} metric {key} should be numeric")
                    
                except Exception as e:
                    self.fail(f"get_training_metrics test failed for {algorithm_name}: {e}")
    
    def test_training_mode_interface(self):
        """测试训练模式设置接口"""
        available_algorithms = get_available_algorithms()
        
        for algorithm_name in available_algorithms:
            with self.subTest(algorithm=algorithm_name):
                try:
                    config = self.algorithm_configs.get(algorithm_name, self.mock_config)
                    algorithm = AlgorithmFactory.create_algorithm(
                        algorithm_name, self.mock_env_info, config
                    )
                    
                    # 测试方法存在
                    self.assertTrue(hasattr(algorithm, 'set_training_mode'),
                                  f"{algorithm_name} should have set_training_mode method")
                    
                    # 测试设置训练模式
                    algorithm.set_training_mode(True)
                    self.assertTrue(algorithm.training,
                                  f"{algorithm_name} should be in training mode")
                    
                    # 测试设置评估模式
                    algorithm.set_training_mode(False)
                    self.assertFalse(algorithm.training,
                                   f"{algorithm_name} should be in evaluation mode")
                    
                except Exception as e:
                    self.fail(f"training mode test failed for {algorithm_name}: {e}")
    
    def test_algorithm_info_interface(self):
        """测试算法信息接口"""
        available_algorithms = get_available_algorithms()
        
        for algorithm_name in available_algorithms:
            with self.subTest(algorithm=algorithm_name):
                try:
                    config = self.algorithm_configs.get(algorithm_name, self.mock_config)
                    algorithm = AlgorithmFactory.create_algorithm(
                        algorithm_name, self.mock_env_info, config
                    )
                    
                    # 测试方法存在
                    self.assertTrue(hasattr(algorithm, 'get_algorithm_info'),
                                  f"{algorithm_name} should have get_algorithm_info method")
                    
                    # 测试方法调用
                    info = algorithm.get_algorithm_info()
                    
                    # 验证返回值格式
                    self.assertIsInstance(info, dict,
                                        f"{algorithm_name} get_algorithm_info should return dict")
                    
                    # 验证必要字段
                    required_fields = ['algorithm_name', 'num_agents', 'obs_dims', 'action_dims']
                    for field in required_fields:
                        self.assertIn(field, info,
                                    f"{algorithm_name} info should contain {field}")
                    
                except Exception as e:
                    self.fail(f"algorithm info test failed for {algorithm_name}: {e}")
    
    def test_observation_validation(self):
        """测试观测数据验证"""
        available_algorithms = get_available_algorithms()
        
        for algorithm_name in available_algorithms:
            with self.subTest(algorithm=algorithm_name):
                try:
                    config = self.algorithm_configs.get(algorithm_name, self.mock_config)
                    algorithm = AlgorithmFactory.create_algorithm(
                        algorithm_name, self.mock_env_info, config
                    )
                    
                    # 测试有效观测
                    valid_observations = [
                        np.random.randn(obs_dim) 
                        for obs_dim in self.mock_env_info['obs_dims']
                    ]
                    self.assertTrue(algorithm.validate_observations(valid_observations),
                                  f"{algorithm_name} should validate correct observations")
                    
                    # 测试无效观测（数量不匹配）
                    invalid_observations = [np.random.randn(10)]  # 只有一个观测
                    self.assertFalse(algorithm.validate_observations(invalid_observations),
                                   f"{algorithm_name} should reject incorrect number of observations")
                    
                    # 测试无效观测（维度不匹配）
                    invalid_dim_observations = [
                        np.random.randn(5) for _ in range(self.mock_env_info['num_agents'])
                    ]
                    self.assertFalse(algorithm.validate_observations(invalid_dim_observations),
                                   f"{algorithm_name} should reject incorrect observation dimensions")
                    
                except Exception as e:
                    self.fail(f"observation validation test failed for {algorithm_name}: {e}")


class TestAlgorithmRegistry(unittest.TestCase):
    """测试算法注册系统"""
    
    def test_algorithm_registration(self):
        """测试算法注册功能"""
        # 获取已注册的算法
        registered_algorithms = AlgorithmRegistry.list_algorithms()
        
        # 验证内置算法已注册
        expected_algorithms = ['MADDPG', 'QMIX', 'MAPPO']
        for algorithm in expected_algorithms:
            self.assertIn(algorithm, registered_algorithms,
                         f"{algorithm} should be registered")
    
    def test_algorithm_info_retrieval(self):
        """测试算法信息获取"""
        registered_algorithms = AlgorithmRegistry.list_algorithms()
        
        for algorithm_name in registered_algorithms:
            with self.subTest(algorithm=algorithm_name):
                info = AlgorithmRegistry.get_algorithm_info(algorithm_name)
                
                self.assertIsInstance(info, dict,
                                    f"{algorithm_name} info should be dict")
                self.assertIn('name', info,
                            f"{algorithm_name} info should contain name")
                self.assertIn('description', info,
                            f"{algorithm_name} info should contain description")


if __name__ == '__main__':
    unittest.main()