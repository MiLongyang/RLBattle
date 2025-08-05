# -*- coding: utf-8 -*-

import torch
import logging
import sys
import os
from arguments import get_args
from envs.battle_env import BattleEnv
from core.config_manager import ConfigManager
from core.training_manager import TrainingManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    统一训练入口 - 使用新的训练管理框架
    """
    try:
        # 1. 获取参数
        args = get_args()
        logger.info(f"Arguments loaded: algorithm={args.algorithm}, task_type={args.task_type}")
        
        # 2. 初始化环境
        env = BattleEnv(args)
        env_info = env.get_env_info()
        logger.info(f"Environment initialized: {env_info['num_agents']} agents, "
                   f"obs_dims={env_info['obs_dims']}, action_dims={env_info['action_dims']}")
        
        # 3. 初始化配置管理器和训练管理器
        config_manager = ConfigManager()
        training_manager = TrainingManager(config_manager)
        
        # 4. 检查算法推荐
        recommended_algorithm = config_manager.get_recommended_algorithm(args.task_type)
        if args.algorithm.upper() != recommended_algorithm:
            logger.warning(f"Algorithm {args.algorithm} is not the recommended algorithm for task {args.task_type}. "
                         f"Recommended: {recommended_algorithm}")
            alternatives = config_manager.get_alternative_algorithms(args.task_type)
            logger.info(f"Available algorithms for {args.task_type}: {alternatives}")
        
        # 5. 准备自定义配置
        custom_config = vars(args)  # 将args转换为字典
        
        # 6. 初始化训练
        training_manager.initialize_training(
            algorithm_name=args.algorithm,
            task_type=args.task_type,
            env=env,
            custom_config=custom_config
        )
        
        # 7. 设置保存目录
        save_dir = f"./models/{args.algorithm.upper()}_{args.task_type}"
        os.makedirs(save_dir, exist_ok=True)
        
        # 8. 开始训练
        logger.info("Starting training...")
        training_results = training_manager.run_training(save_dir)
        
        # 9. 输出训练结果
        logger.info("Training completed successfully!")
        logger.info(f"Episodes completed: {training_results['episodes_completed']}")
        logger.info(f"Total reward: {training_results['total_reward']:.2f}")
        logger.info(f"Average reward: {training_results['average_reward']:.2f}")
        logger.info(f"Best reward: {training_results['best_reward']:.2f} at episode {training_results['best_episode']}")
        logger.info(f"Training time: {training_results['training_time']:.2f}s")
        
        # 10. 保存训练配置和结果
        final_config = config_manager.get_config(args.algorithm, args.task_type, custom_config)
        config_path = os.path.join(save_dir, "training_config.json")
        config_manager.save_config(final_config, config_path)
        
        # 保存训练结果
        import json
        results_path = os.path.join(save_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
        logger.info(f"Results saved to: {results_path}")
        
        return training_results
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'training_manager' in locals():
            training_manager.stop_training()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()
            logger.info("Environment closed")

def run_evaluation():
    """
    运行评估模式
    """
    try:
        args = get_args()
        logger.info(f"Starting evaluation: algorithm={args.algorithm}, task_type={args.task_type}")
        
        # 初始化环境
        env = BattleEnv(args)
        env_info = env.get_env_info()
        
        # 初始化配置管理器
        config_manager = ConfigManager()
        custom_config = vars(args)
        config = config_manager.get_config(args.algorithm, args.task_type, custom_config)
        config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建算法实例
        from algorithms.algorithm_factory import AlgorithmFactory
        algorithm = AlgorithmFactory.create_algorithm(args.algorithm, env_info, config)
        
        # 加载模型
        model_path = f"./models/{args.algorithm.upper()}_{args.task_type}"
        algorithm.load_models(model_path, args.load_model_episode)
        algorithm.set_training_mode(False)  # 设置为评估模式
        
        # 运行评估
        total_eval_episodes = getattr(args, 'eval_episodes', 100)
        all_rewards = []
        
        logger.info(f"Running evaluation for {total_eval_episodes} episodes...")
        
        for episode in range(total_eval_episodes):
            obs = env.reset()
            episode_reward = 0
            
            # 算法特定的初始化
            algorithm_state = {}
            if args.algorithm.upper() == "QMIX":
                algorithm_state['hidden_states'] = [algorithm.agent_net.init_hidden().to(config['device']) 
                                                  for _ in range(env_info["num_agents"])]
                algorithm_state['epsilon'] = 0.0  # 评估时不探索
            elif args.algorithm.upper() == "MADDPG":
                algorithm_state['add_noise'] = False  # 评估时不添加噪声
            
            step_count = 0
            while step_count < env_info.get("episode_limit", 500):
                actions, new_state = algorithm.select_actions(obs, **algorithm_state)
                next_obs, reward, done, info = env.step(actions)
                
                episode_reward += reward
                obs = next_obs
                step_count += 1
                
                # 更新算法状态
                if new_state is not None and args.algorithm.upper() == "QMIX":
                    algorithm_state['hidden_states'] = new_state
                
                if done:
                    break
            
            all_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                logger.info(f"Evaluation episode {episode + 1}/{total_eval_episodes}, "
                           f"Reward: {episode_reward:.2f}")
        
        # 计算评估结果
        import numpy as np
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        max_reward = np.max(all_rewards)
        min_reward = np.min(all_rewards)
        
        logger.info("Evaluation completed!")
        logger.info(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"Max reward: {max_reward:.2f}")
        logger.info(f"Min reward: {min_reward:.2f}")
        
        # 保存评估结果
        eval_results = {
            'algorithm': args.algorithm,
            'task_type': args.task_type,
            'episodes': total_eval_episodes,
            'average_reward': avg_reward,
            'std_reward': std_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'all_rewards': all_rewards
        }
        
        results_path = f"./models/{args.algorithm.upper()}_{args.task_type}/evaluation_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        import json
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {results_path}")
        
        return eval_results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()

if __name__ == '__main__':
    # 检查是否是评估模式
    if len(sys.argv) > 1 and sys.argv[1] == '--eval':
        run_evaluation()
    else:
        main()

