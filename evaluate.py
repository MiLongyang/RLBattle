# -*- coding: utf-8 -*-

import torch
import numpy as np
import logging
import json
import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from arguments import get_args
from envs.battle_env import BattleEnv
from core.config_manager import ConfigManager
from algorithms.algorithm_factory import AlgorithmFactory

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """评估指标数据模型"""
    algorithm: str
    task_type: str
    episode: str
    total_episodes: int
    average_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    success_rate: float
    average_steps: float
    evaluation_time: float
    all_rewards: List[float]
    all_steps: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

class UnifiedEvaluator:
    """
    统一评估器
    支持所有算法的标准化评估
    """
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"UnifiedEvaluator initialized with device: {self.device}")
    
    def evaluate_model(self, algorithm_name: str, task_type: str, model_episode: str,
                      eval_episodes: int = 100, model_path: Optional[str] = None,
                      custom_config: Optional[Dict] = None) -> EvaluationMetrics:
        """
        评估模型
        
        Args:
            algorithm_name: 算法名称
            task_type: 任务类型
            model_episode: 模型回合数
            eval_episodes: 评估回合数
            model_path: 模型路径（可选）
            custom_config: 自定义配置（可选）
            
        Returns:
            评估指标
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting evaluation: {algorithm_name} on {task_type} task")
            
            # 创建临时args对象用于环境初始化
            class TempArgs:
                def __init__(self, task_type):
                    self.task_type = task_type
                    self.algorithm = algorithm_name
                    self.episode_limit = 500
            
            temp_args = TempArgs(task_type)
            
            # 初始化环境
            env = BattleEnv(temp_args)
            env_info = env.get_env_info()
            
            # 获取配置
            config = self.config_manager.get_config(algorithm_name, task_type, custom_config)
            config['device'] = self.device
            
            # 创建算法实例
            algorithm = AlgorithmFactory.create_algorithm(algorithm_name, env_info, config)
            
            # 加载模型
            if model_path is None:
                model_path = f"./models/{algorithm_name.upper()}_{task_type}"
            
            algorithm.load_models(model_path, model_episode)
            algorithm.set_training_mode(False)  # 设置为评估模式
            
            logger.info(f"Model loaded from {model_path}/{model_episode}")
            
            # 运行评估
            all_rewards = []
            all_steps = []
            success_count = 0
            
            logger.info(f"Running evaluation for {eval_episodes} episodes...")
            
            for episode in range(eval_episodes):
                episode_result = self._run_evaluation_episode(
                    algorithm, env, env_info, algorithm_name
                )
                
                all_rewards.append(episode_result['reward'])
                all_steps.append(episode_result['steps'])
                
                if episode_result['success']:
                    success_count += 1
                
                # 定期输出进度
                if (episode + 1) % 10 == 0:
                    logger.info(f"Evaluation progress: {episode + 1}/{eval_episodes} episodes completed")
            
            # 计算统计指标
            avg_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            max_reward = np.max(all_rewards)
            min_reward = np.min(all_rewards)
            success_rate = success_count / eval_episodes
            avg_steps = np.mean(all_steps)
            evaluation_time = time.time() - start_time
            
            # 创建评估指标
            metrics = EvaluationMetrics(
                algorithm=algorithm_name,
                task_type=task_type,
                episode=model_episode,
                total_episodes=eval_episodes,
                average_reward=avg_reward,
                std_reward=std_reward,
                max_reward=max_reward,
                min_reward=min_reward,
                success_rate=success_rate,
                average_steps=avg_steps,
                evaluation_time=evaluation_time,
                all_rewards=all_rewards,
                all_steps=all_steps
            )
            
            logger.info("Evaluation completed!")
            logger.info(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
            logger.info(f"Success rate: {success_rate:.2%}")
            logger.info(f"Evaluation time: {evaluation_time:.2f}s")
            
            env.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _run_evaluation_episode(self, algorithm, env, env_info: Dict[str, Any], 
                              algorithm_name: str) -> Dict[str, Any]:
        """
        运行单个评估回合
        
        Args:
            algorithm: 算法实例
            env: 环境实例
            env_info: 环境信息
            algorithm_name: 算法名称
            
        Returns:
            回合结果字典
        """
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        episode_limit = env_info.get("episode_limit", 500)
        
        # 算法特定的初始化
        algorithm_state = self._initialize_algorithm_state(algorithm, env_info, algorithm_name)
        
        while step_count < episode_limit:
            # 选择动作（评估模式，不使用探索）
            actions, new_state = algorithm.select_actions(obs, **algorithm_state)
            
            # 环境交互
            next_obs, reward, done, info = env.step(actions)
            
            episode_reward += reward
            obs = next_obs
            step_count += 1
            
            # 更新算法状态
            if new_state is not None:
                if algorithm_name.upper() == "QMIX":
                    algorithm_state['hidden_states'] = new_state
            
            if done:
                break
        
        # 判断成功条件（可根据具体任务调整）
        success = self._evaluate_success(episode_reward, step_count, env_info)
        
        return {
            'reward': episode_reward,
            'steps': step_count,
            'success': success
        }
    
    def _initialize_algorithm_state(self, algorithm, env_info: Dict[str, Any], 
                                  algorithm_name: str) -> Dict[str, Any]:
        """
        初始化算法特定状态（评估模式）
        
        Args:
            algorithm: 算法实例
            env_info: 环境信息
            algorithm_name: 算法名称
            
        Returns:
            算法状态字典
        """
        algorithm_name = algorithm_name.upper()
        
        if algorithm_name == "QMIX":
            # QMIX需要hidden_states，评估时epsilon=0
            hidden_states = [algorithm.agent_net.init_hidden().to(self.device) 
                           for _ in range(env_info["num_agents"])]
            return {'hidden_states': hidden_states, 'epsilon': 0.0}
        
        elif algorithm_name == "MADDPG":
            # MADDPG评估时不添加噪声
            return {'add_noise': False}
        
        elif algorithm_name == "MAPPO":
            # MAPPO评估模式
            return {}
        
        return {}
    
    def _evaluate_success(self, reward: float, steps: int, env_info: Dict[str, Any]) -> bool:
        """
        评估回合是否成功
        
        Args:
            reward: 回合奖励
            steps: 回合步数
            env_info: 环境信息
            
        Returns:
            是否成功
        """
        # 简单的成功判断逻辑，可根据具体任务调整
        # 这里假设奖励大于0且没有超时就算成功
        episode_limit = env_info.get("episode_limit", 500)
        return reward > 0 and steps < episode_limit
    
    def compare_algorithms(self, algorithms: List[str], task_type: str, 
                          model_episode: str = "final", eval_episodes: int = 100) -> Dict[str, Any]:
        """
        比较多个算法的性能
        
        Args:
            algorithms: 算法名称列表
            task_type: 任务类型
            model_episode: 模型回合数
            eval_episodes: 评估回合数
            
        Returns:
            比较结果字典
        """
        logger.info(f"Comparing algorithms {algorithms} on {task_type} task")
        
        results = {}
        comparison_data = {
            'task_type': task_type,
            'model_episode': model_episode,
            'eval_episodes': eval_episodes,
            'algorithms': {},
            'comparison_time': time.time()
        }
        
        for algorithm in algorithms:
            try:
                logger.info(f"Evaluating {algorithm}...")
                metrics = self.evaluate_model(algorithm, task_type, model_episode, eval_episodes)
                results[algorithm] = metrics
                comparison_data['algorithms'][algorithm] = metrics.to_dict()
                
            except Exception as e:
                logger.error(f"Failed to evaluate {algorithm}: {e}")
                results[algorithm] = None
        
        # 生成比较分析
        analysis = self._analyze_comparison(results)
        comparison_data['analysis'] = analysis
        
        return comparison_data
    
    def _analyze_comparison(self, results: Dict[str, Optional[EvaluationMetrics]]) -> Dict[str, Any]:
        """
        分析比较结果
        
        Args:
            results: 评估结果字典
            
        Returns:
            分析结果字典
        """
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return {'error': 'No valid results to analyze'}
        
        analysis = {
            'best_average_reward': None,
            'best_success_rate': None,
            'most_efficient': None,
            'rankings': {
                'by_reward': [],
                'by_success_rate': [],
                'by_efficiency': []
            }
        }
        
        # 按平均奖励排序
        reward_ranking = sorted(valid_results.items(), 
                              key=lambda x: x[1].average_reward, reverse=True)
        analysis['rankings']['by_reward'] = [(alg, metrics.average_reward) 
                                           for alg, metrics in reward_ranking]
        analysis['best_average_reward'] = reward_ranking[0][0]
        
        # 按成功率排序
        success_ranking = sorted(valid_results.items(), 
                               key=lambda x: x[1].success_rate, reverse=True)
        analysis['rankings']['by_success_rate'] = [(alg, metrics.success_rate) 
                                                 for alg, metrics in success_ranking]
        analysis['best_success_rate'] = success_ranking[0][0]
        
        # 按效率排序（奖励/平均步数）
        efficiency_ranking = sorted(valid_results.items(), 
                                  key=lambda x: x[1].average_reward / max(x[1].average_steps, 1), 
                                  reverse=True)
        analysis['rankings']['by_efficiency'] = [(alg, metrics.average_reward / max(metrics.average_steps, 1)) 
                                                for alg, metrics in efficiency_ranking]
        analysis['most_efficient'] = efficiency_ranking[0][0]
        
        return analysis
    
    def save_evaluation_results(self, metrics: EvaluationMetrics, 
                              save_path: Optional[str] = None) -> str:
        """
        保存评估结果
        
        Args:
            metrics: 评估指标
            save_path: 保存路径（可选）
            
        Returns:
            保存路径
        """
        if save_path is None:
            save_dir = f"./models/{metrics.algorithm.upper()}_{metrics.task_type}"
            save_path = os.path.join(save_dir, f"evaluation_results_{metrics.episode}.json")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        logger.info(f"Evaluation results saved to: {save_path}")
        return save_path

def evaluate():
    """
    主评估函数（保持向后兼容）
    """
    args = get_args()
    
    evaluator = UnifiedEvaluator()
    
    try:
        # 运行评估
        metrics = evaluator.evaluate_model(
            algorithm_name=args.algorithm,
            task_type=args.task_type,
            model_episode=args.load_model_episode,
            eval_episodes=getattr(args, 'eval_episodes', 100)
        )
        
        # 保存结果
        evaluator.save_evaluation_results(metrics)
        
        # 输出结果
        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Algorithm: {metrics.algorithm}")
        print(f"Task Type: {metrics.task_type}")
        print(f"Model Episode: {metrics.episode}")
        print(f"Total Episodes: {metrics.total_episodes}")
        print(f"Average Reward: {metrics.average_reward:.2f} ± {metrics.std_reward:.2f}")
        print(f"Max Reward: {metrics.max_reward:.2f}")
        print(f"Min Reward: {metrics.min_reward:.2f}")
        print(f"Success Rate: {metrics.success_rate:.2%}")
        print(f"Average Steps: {metrics.average_steps:.1f}")
        print(f"Evaluation Time: {metrics.evaluation_time:.2f}s")
        print("=" * 50)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def compare_algorithms_main():
    """
    算法比较主函数
    """
    args = get_args()
    
    # 获取可用算法
    available_algorithms = ['MADDPG', 'QMIX', 'MAPPO']
    
    evaluator = UnifiedEvaluator()
    
    try:
        # 比较算法
        comparison_results = evaluator.compare_algorithms(
            algorithms=available_algorithms,
            task_type=args.task_type,
            model_episode=args.load_model_episode,
            eval_episodes=getattr(args, 'eval_episodes', 50)  # 比较时使用较少回合
        )
        
        # 保存比较结果
        save_path = f"./models/algorithm_comparison_{args.task_type}.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # 输出比较结果
        print("=" * 60)
        print("ALGORITHM COMPARISON RESULTS")
        print("=" * 60)
        print(f"Task Type: {args.task_type}")
        print(f"Model Episode: {args.load_model_episode}")
        
        analysis = comparison_results.get('analysis', {})
        
        print(f"\nBest Average Reward: {analysis.get('best_average_reward', 'N/A')}")
        print(f"Best Success Rate: {analysis.get('best_success_rate', 'N/A')}")
        print(f"Most Efficient: {analysis.get('most_efficient', 'N/A')}")
        
        print("\nRankings by Average Reward:")
        for i, (alg, reward) in enumerate(analysis.get('rankings', {}).get('by_reward', []), 1):
            print(f"  {i}. {alg}: {reward:.2f}")
        
        print("\nRankings by Success Rate:")
        for i, (alg, rate) in enumerate(analysis.get('rankings', {}).get('by_success_rate', []), 1):
            print(f"  {i}. {alg}: {rate:.2%}")
        
        print("=" * 60)
        print(f"Detailed results saved to: {save_path}")
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"Algorithm comparison failed: {e}")
        raise

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_algorithms_main()
    else:
        evaluate() 