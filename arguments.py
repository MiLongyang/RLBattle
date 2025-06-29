# -*- coding: utf-8 -*-

import argparse

def get_args():
    """
    定义并返回所有训练参数
    """
    parser = argparse.ArgumentParser(description="强化学习对战平台参数配置")

    # --- 通用参数 ---
    parser.add_argument("--algorithm", type=str, default="MADDPG", help="选择算法: 'MADDPG' 或 'QMIX'")
    parser.add_argument("--num_episodes", type=int, default=10000, help="总训练回合数")
    parser.add_argument("--episode_limit", type=int, default=500, help="每回合最大步数")
    parser.add_argument("--batch_size", type=int, default=256, help="训练批次大小")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--seed", type=int, default=123, help="随机种子")
    parser.add_argument("--epsilon_decay_steps", type=int, default=50000, help="QMIX中Epsilon衰减的总步数")

    # --- MADDPG 专属参数 ---
    parser.add_argument("--buffer_size_maddpg", type=int, default=int(1e6), help="MADDPG的经验回放池大小")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="MADDPG的Actor学习率")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="MADDPG的Critic学习率")
    parser.add_argument("--tau", type=float, default=1e-3, help="MADDPG目标网络软更新系数")
    parser.add_argument("--noise_std", type=float, default=0.1, help="MADDPG动作噪音标准差")

    # --- QMIX 专属参数 ---
    parser.add_argument("--buffer_size_qmix", type=int, default=5000, help="QMIX的经验回放池大小")
    parser.add_argument("--lr_qmix", type=float, default=5e-4, help="QMIX的学习率")
    parser.add_argument("--target_update_interval", type=int, default=200, help="QMIX目标网络更新频率")
    parser.add_argument("--mixer_hidden_dim", type=int, default=32, help="QMIX混合网络隐藏层维度")
    parser.add_argument("--agent_hidden_dim", type=int, default=64, help="QMIX智能体RNN隐藏层维度")
    
    # --- 环境参数 (占位) ---
    parser.add_argument("--num_agents", type=int, default=3, help="智能体数量")
    parser.add_argument("--state_dim_env", type=int, default=10, help="环境状态维度 (示例)")
    parser.add_argument("--action_dim_env", type=int, default=2, help="环境动作维度 (示例)")

    args = parser.parse_args()
    return args 