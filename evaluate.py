# -*- coding: utf-8 -*-

import torch
import numpy as np
from arguments import get_args
from envs.battle_env import BattleEnv
from algorithms.maddpg.maddpg import MADDPG
from algorithms.qmix.qmix import QMIX

def evaluate():
    """
    模型评估函数
    """
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化环境
    env = BattleEnv(args)
    env_info = env.get_env_info()

    # 2. 初始化算法
    algorithm_name = args.algorithm.upper()
    if algorithm_name == "MADDPG":
        agent_group = MADDPG(env_info["obs_dims"], env_info["action_dims"], env_info["num_agents"], env_info["state_dim"], args, device)
    elif algorithm_name == "QMIX":
        agent_group = QMIX(env_info["obs_dims"], env_info["action_dims"], env_info["num_agents"], env_info["state_dim"], args, device)
    else:
        raise ValueError(f"未知算法: '{args.algorithm}'.")

    # 3. 加载模型
    model_path = f"./models/{algorithm_name}_{args.task_type}"
    episode_to_load = args.load_model_episode
    
    try:
        agent_group.load_models(model_path, episode_to_load)
        print(f"成功加载模型 from {model_path}/{episode_to_load}")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 at {model_path}/{episode_to_load}. 请先训练模型。")
        return

    # 4. 评估主循环
    total_eval_episodes = 100 # 评估100个回合
    all_rewards = []
    
    print("开始评估...")
    for _ in range(total_eval_episodes):
        obs = env.reset()
        episode_reward = 0
        
        # QMIX 相关初始化
        if algorithm_name == "QMIX":
            hidden_states = [agent_group.agent_net.init_hidden().to(device) for _ in range(env_info["num_agents"])]

        while True:
            # 在评估时, 不使用探索噪声 (epsilon=0)
            if algorithm_name == "MADDPG":
                actions = agent_group.select_actions(obs, add_noise=False)
            elif algorithm_name == "QMIX":
                actions, hidden_states = agent_group.select_actions(obs, hidden_states, epsilon=0)
            
            # (这部分逻辑依赖于环境实现)
            next_obs, reward, done, _ = env.step(actions)
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        all_rewards.append(episode_reward)

    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print("评估完成.")
    print(f"平均奖励: {avg_reward:.2f} +/- {std_reward:.2f}")

    env.close()

if __name__ == '__main__':
    evaluate() 