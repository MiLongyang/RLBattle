# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
from arguments import get_args
from envs.battle_env import BattleEnv
from algorithms.maddpg.maddpg import MADDPG
from algorithms.qmix.qmix import QMIX

def main():
    """
    统一训练入口
    """
    # 0. 获取参数及设置设备
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 1. 初始化环境
    # 将完整的args对象传递给环境, 以便环境可以根据task_type等参数进行配置
    env = BattleEnv(args)
    env_info = env.get_env_info()

    num_agents = env_info["num_agents"]
    state_dim = env_info["state_dim"]
    obs_dims = env_info["obs_dims"]
    action_dims = env_info["action_dims"]
    
    # 为保存模型创建目录
    save_dir = f"./models/{args.algorithm.upper()}_{args.task_type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 2. 根据配置选择并初始化算法
    algorithm_name = args.algorithm.upper()
    print(f"选择算法: {algorithm_name}")

    if algorithm_name == "MADDPG":
        agent_group = MADDPG(obs_dims, action_dims, num_agents, state_dim, args, device,
                             env_info["action_space_low"], env_info["action_space_high"])
    elif algorithm_name == "QMIX":
        agent_group = QMIX(obs_dims, action_dims, num_agents, state_dim, args, device)
    else:
        raise ValueError(f"未知算法: '{args.algorithm}'. 支持的算法包括: ['MADDPG', 'QMIX']")

    # 动态调整epsilon
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = (epsilon - min_epsilon) / args.epsilon_decay_steps

    # 3. 训练主循环
    for episode in range(args.num_episodes):
        # 重置环境
        obs = env.reset()
        episode_reward = 0
        
        # QMIX 相关初始化
        if algorithm_name == "QMIX":
            hidden_states = [agent_group.agent_net.init_hidden().to(device) for _ in range(num_agents)]
            episode_buffer = {
                'obs': [], 'actions': [], 'rewards': [], 'dones': [], 'state': [], 'next_obs': [], 'next_state': []
            }

        for step in range(env_info["episode_limit"]):
            actions, new_hidden_states = agent_group.select_actions(obs, hidden_states, epsilon) if algorithm_name == "QMIX" else (agent_group.select_actions(obs), None)
            
            next_obs, reward, done, info = env.step(actions)
            # 假设env返回全局状态, 如果没有, 我们用obs拼接来模拟
            state = info.get('state', np.concatenate(obs))
            next_state = info.get('next_state', np.concatenate(next_obs))

            if algorithm_name == "MADDPG":
                agent_group.add_experience(obs, state, actions, reward, next_obs, next_state, done)
                agent_group.learn()
            elif algorithm_name == "QMIX":
                # 存储当前step的数据到回合缓存
                episode_buffer['obs'].append(obs)
                episode_buffer['actions'].append(actions)
                episode_buffer['rewards'].append([reward])
                episode_buffer['dones'].append([done])
                episode_buffer['state'].append(state)
                episode_buffer['next_obs'].append(next_obs)
                episode_buffer['next_state'].append(next_state)
                hidden_states = new_hidden_states
            
            obs = next_obs
            episode_reward += reward
            epsilon = max(min_epsilon, epsilon - epsilon_decay)

            if done:
                break
        
        # QMIX 在回合结束后, 将整个episode存入buffer并学习
        if algorithm_name == "QMIX":
            episode_len = step + 1
            for key in episode_buffer.keys():
                episode_buffer[key] = np.array(episode_buffer[key])
            agent_group.replay_buffer.add_episode(episode_buffer, episode_len)
            
            # 当经验池里的数据量足够一个批次时才开始学习
            if agent_group.replay_buffer.size() >= args.batch_size:
                agent_group.learn()

        print(f"Episode {episode + 1}/{args.num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")

        # 每隔50个回合保存一次模型
        if (episode + 1) % 50 == 0:
            agent_group.save_models(save_dir, episode + 1)

    # 训练结束后最后保存一次
    agent_group.save_models(save_dir, "final")
    env.close()

if __name__ == '__main__':
    main()

