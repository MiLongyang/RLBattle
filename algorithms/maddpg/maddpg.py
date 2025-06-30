# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
from .agent import Agent
from common.replay_buffer import ReplayBuffer
from ..base_algorithm import BaseAlgorithm

class MADDPG(BaseAlgorithm):
    """
    MADDPG 算法主类
    """
    def __init__(self, obs_dims, action_dims, num_agents, state_dim, args, device, action_space_low, action_space_high):
        self.num_agents = num_agents
        self.args = args
        self.device = device
        
        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(
                agent_id=i,
                obs_dim=obs_dims[i],
                action_dim=action_dims[i],
                num_agents=num_agents,
                state_dim=state_dim,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                device=device,
                action_low=action_space_low,
                action_high=action_space_high
            ))
        
        self.replay_buffer = ReplayBuffer(args.buffer_size_maddpg, self.num_agents, obs_dims, action_dims, state_dim)

    def select_actions(self, observations, add_noise=True):
        """
        为所有智能体选择动作
        """
        actions = []
        for agent, obs in zip(self.agents, observations):
            action = agent.select_action(obs, add_noise, self.args.noise_std)
            actions.append(action)
        return actions

    def learn(self):
        """
        从经验回放池中采样, 训练所有智能体
        """
        if self.replay_buffer.size() < self.args.batch_size:
            return

        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, state_batch, next_state_batch = self.replay_buffer.sample(self.args.batch_size)

        # 将numpy数据转换为torch tensor并移动到指定设备
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        done_batch = torch.from_numpy(done_batch).float().to(self.device)
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        
        # 增加维度检查, 确保全局状态维度正确
        assert state_batch.shape[1] == self.args.state_dim_env
        assert next_state_batch.shape[1] == self.args.state_dim_env

        # 将list of numpy arrays 转换为 list of tensors
        obs_batch_tensor = [torch.from_numpy(obs).float().to(self.device) for obs in obs_batch]
        action_batch_tensor = [torch.from_numpy(act).float().to(self.device) for act in action_batch]
        next_obs_batch_tensor = [torch.from_numpy(obs).float().to(self.device) for obs in next_obs_batch]


        for agent in self.agents:
            # --- 更新 Critic ---
            # 1. 计算目标Q值 (y_i)
            # 首先, 计算下一个状态的目标Actor网络输出的动作
            with torch.no_grad():
                next_actions_target = [self.agents[i].target_actor(next_obs_batch_tensor[i]) for i in range(self.num_agents)]
                # 然后, 计算下一个状态的目标Critic网络输出的Q值
                q_next_target = agent.target_critic(next_state_batch, next_actions_target)
                # 最后的 y_i
                y = reward_batch + self.args.gamma * (1 - done_batch) * q_next_target
            
            # 2. 计算当前Q值
            q_current = agent.critic(state_batch, action_batch_tensor)

            # 3. 计算Critic损失并更新
            critic_loss = F.mse_loss(q_current, y)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0) # 梯度裁剪
            agent.critic_optimizer.step()

            # --- 更新 Actor ---
            # 1. 根据当前策略计算动作
            actions_pred = [self.agents[i].actor(obs_batch_tensor[i]) for i in range(self.num_agents)]
            # 2. 计算 Actor 损失
            actor_loss = -agent.critic(state_batch, actions_pred).mean()
            # 3. 更新Actor网络
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0) # 梯度裁剪
            agent.actor_optimizer.step()
            
        # 所有智能体训练完一个batch后，软更新目标网络
        for agent in self.agents:
            agent.update_target_networks(self.args.tau)


    def add_experience(self, obs, state, actions, rewards, next_obs, next_state, dones):
        """
        向经验回放池中添加一条经验
        """
        self.replay_buffer.add(obs, state, actions, rewards, next_obs, next_state, dones)

    def save_models(self, path, episode_num):
        """
        保存所有智能体的模型
        """
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f"{path}/actor_agent_{i}_{episode_num}.pth")
            torch.save(agent.critic.state_dict(), f"{path}/critic_agent_{i}_{episode_num}.pth")

    def load_models(self, path, episode_num):
        """
        加载所有智能体的模型
        """
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(f"{path}/actor_agent_{i}_{episode_num}.pth"))
            agent.critic.load_state_dict(torch.load(f"{path}/critic_agent_{i}_{episode_num}.pth"))
