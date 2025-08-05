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
    parser.add_argument("--task_type", type=str, default="strike", choices=['recon', 'feint', 'strike'], help="任务类型: 'recon'(侦察), 'feint'(佯攻), 'strike'(协同打击)")

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
    
    # --- 配置文件支持 ---
    parser.add_argument("--config_file", type=str, default=None, help="配置文件路径 (JSON/YAML格式)")
    parser.add_argument("--save_config", type=str, default=None, help="保存当前配置到文件")
    
    # --- 训练管理参数 ---
    parser.add_argument("--save_interval", type=int, default=50, help="模型保存间隔（回合数）")
    parser.add_argument("--log_interval", type=int, default=10, help="日志输出间隔（回合数）")
    parser.add_argument("--eval_interval", type=int, default=100, help="评估间隔（回合数）")
    parser.add_argument("--eval_episodes", type=int, default=10, help="评估回合数")
    
    # --- 环境参数 (占位) ---
    parser.add_argument("--num_agents", type=int, default=3, help="智能体数量")
    parser.add_argument("--state_dim_env", type=int, default=10, help="环境状态维度 (示例)")
    parser.add_argument("--action_dim_env", type=int, default=2, help="环境动作维度 (示例)")

    # --- 模型加载参数 ---
    parser.add_argument("--load_model_episode", type=str, default="final", help="指定要加载进行评估的模型的回合数 (例如, '50', '100', 或者 'final')")

    args = parser.parse_args()
    
    # 如果指定了配置文件，则加载配置
    if args.config_file:
        args = _load_config_file(args, args.config_file)
    
    # 如果指定了保存配置，则保存当前配置
    if args.save_config:
        _save_config_file(args, args.save_config)
    
    return args

def _load_config_file(args, config_file: str):
    """
    从配置文件加载参数
    
    Args:
        args: 当前参数对象
        config_file: 配置文件路径
        
    Returns:
        更新后的参数对象
    """
    import json
    import yaml
    import os
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                config_data = json.load(f)
            elif config_file.endswith(('.yaml', '.yml')):
                config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file}")
        
        # 更新参数
        for key, value in config_data.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")
        
        print(f"Config loaded from: {config_file}")
        return args
        
    except Exception as e:
        raise RuntimeError(f"Failed to load config file {config_file}: {e}")

def _save_config_file(args, config_file: str):
    """
    保存当前配置到文件
    
    Args:
        args: 参数对象
        config_file: 配置文件路径
    """
    import json
    import yaml
    import os
    
    # 转换为字典
    config_data = vars(args).copy()
    
    # 移除不需要保存的参数
    exclude_keys = ['config_file', 'save_config']
    for key in exclude_keys:
        config_data.pop(key, None)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            elif config_file.endswith(('.yaml', '.yml')):
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported config file format: {config_file}")
        
        print(f"Config saved to: {config_file}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save config file {config_file}: {e}") 