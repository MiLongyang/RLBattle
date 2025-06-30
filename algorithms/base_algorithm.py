# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    所有强化学习算法的抽象基类。
    定义了所有算法必须实现的通用接口，以确保框架的可扩展性。
    任何新的算法都应该继承这个类，并实现其所有抽象方法。
    """
    @abstractmethod
    def __init__(self, obs_dims, action_dims, num_agents, state_dim, args, device):
        """
        初始化算法，包括网络、优化器、经验回放池等。
        """
        pass

    @abstractmethod
    def select_actions(self, *args, **kwargs):
        """
        根据当前状态为所有智能体选择动作。
        方法的具体参数由子类自行定义。
        """
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        """
        执行一次学习/更新步骤。
        通常涉及从经验回放池中采样数据并更新网络。
        """
        pass 