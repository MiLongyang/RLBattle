# -*- coding: utf-8 -*-
"""
独立测试 BattleEnv 环境
"""

import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.battle_env import BattleEnv
from arguments import get_args


def test_environment_basic():
    """基本环境功能测试"""
    print("=== 基本环境功能测试 ===")

    # 创建环境
    args = get_args()
    env = BattleEnv(args)

    print(f"环境创建成功")
    print(f"任务类型: {env.task_type}")
    print(f"红方智能体数量: {env.num_red}")
    print(f"蓝方智能体数量: {env.num_blue}")
    print(f"总智能体数量: {env.total_agents}")

    # 测试重置
    obs = env.reset()
    print(f"重置后观测形状: {obs.shape}")
    print(f"观测维度: {obs.shape[1] if len(obs.shape) > 1 else 'N/A'}")

    # 测试动作空间和观测空间
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")

    # 测试获取环境信息
    env_info = env.get_env_info()
    print(f"环境信息: {env_info}")

    return env


def test_environment_step(env):
    """环境step功能测试"""
    print("\n=== 环境Step功能测试 ===")

    # 重置环境
    obs = env.reset()
    print(f"初始观测: {obs[0][:5]}...")  # 只显示前5个元素

    # 打印初始导弹信息
    print_missile_info(env, 0)

    # 执行几步测试
    for step in range(5):
        # 生成随机动作
        actions = [np.random.uniform(-1, 1, 2) for _ in range(env.total_agents)]

        # 执行step
        next_obs, reward, done, info = env.step(actions)

        print(f"Step {step + 1}:")
        # 打印每个导弹的信息
        print_missile_info(env, step + 1)
        print(f"  奖励: {reward:.2f}")
        print(f"  是否结束: {done}")
        print(f"  当前时间步: {info.get('timestep', 'N/A')}")
        print(f"  红方存活: {info.get('red_alive', 'N/A')}")
        print(f"  蓝方存活: {info.get('blue_alive', 'N/A')}")
        print(f"  观测示例: {next_obs[0][:5]}...")

        if done:
            print("  回合结束")
            break


def print_missile_info(env, timestep):
    """打印每个导弹的信息"""
    print(f"  时间步 {timestep} 的导弹信息:")
    
    # 打印红方导弹信息
    for i, missile in enumerate(env.red_missiles):
        name = f"红方导弹_{i}"
        # 获取导弹种类
        missile_type = missile['type']
        position = missile['pos']
        velocity = missile['vel']
        print(f"    {name} ({missile_type}): 位置={position}, 速度={velocity}")
    
    # 打印蓝方导弹信息
    for i, missile in enumerate(env.blue_missiles):
        name = f"蓝方导弹_{i}"
        # 获取导弹种类
        missile_type = missile['type']
        position = missile['pos']
        velocity = missile['vel']
        print(f"    {name} ({missile_type}): 位置={position}, 速度={velocity}")


def test_all_task_types():
    """测试所有任务类型"""
    print("\n=== 所有任务类型测试 ===")

    task_types = ["recon", "feint", "strike"]

    for task_type in task_types:
        print(f"\n测试任务类型: {task_type}")

        # 创建对应任务类型的环境
        args = get_args()
        args.task_type = task_type

        try:
            env = BattleEnv(args)
            obs = env.reset()

            # 执行一步
            actions = [np.random.uniform(-1, 1, 2) for _ in range(env.total_agents)]
            next_obs, reward, done, info = env.step(actions)

            print(f"  成功执行，奖励: {reward:.2f}")

        except Exception as e:
            print(f"  错误: {e}")


def test_boundary_conditions(env):
    """测试边界条件"""
    print("\n=== 边界条件测试 ===")

    # 重置环境
    env.reset()

    # 尝试让导弹越界
    actions = []
    for i in range(env.total_agents):
        if i == 0:  # 让第一个导弹向负方向移动
            actions.append(np.array([-2.0, -2.0]))
        else:
            actions.append(np.array([0.0, 0.0]))

    # 执行多步让导弹越界
    for step in range(10):
        next_obs, reward, done, info = env.step(actions)

        # 检查位置是否被修正
        if hasattr(env, 'red_positions'):
            pos = env.red_positions[0] if len(env.red_positions) > 0 else None
            if pos is not None:
                print(f"Step {step + 1}: 位置 = {pos}, 奖励 = {reward:.2f}")
                # 检查是否所有坐标都 >= 0
                if all(coord >= 0 for coord in pos):
                    print(f"  位置有效")
                else:
                    print(f"  位置被修正")

        if done:
            break


def test_reward_components():
    """测试奖励组件"""
    print("\n=== 奖励组件测试 ===")

    # 测试侦察任务奖励
    args = get_args()
    args.task_type = "recon"
    env = BattleEnv(args)

    # 获取一个导弹状态进行测试
    env.reset()
    missile = env.red_missiles[0]

    # 测试侦察奖励计算
    reward, components = env._recon_rewards(missile)
    print(f"侦察任务奖励: {reward:.2f}")
    print(f"奖励组件: {components}")

    # 测试佯攻任务奖励
    args.task_type = "feint"
    env = BattleEnv(args)
    env.reset()
    missile = env.red_missiles[0]

    reward, components = env._feint_rewards(missile)
    print(f"佯攻任务奖励: {reward:.2f}")
    print(f"奖励组件: {components}")

    # 测试打击任务奖励
    args.task_type = "strike"
    env = BattleEnv(args)
    env.reset()
    missile = env.red_missiles[0]

    reward, components = env._strike_rewards(missile)
    print(f"打击任务奖励: {reward:.2f}")
    print(f"奖励组件: {components}")


def main():
    """主测试函数"""
    print("开始测试 BattleEnv 环境")
    print("=" * 50)

    try:
        # 基本功能测试
        env = test_environment_basic()

        # Step功能测试
        test_environment_step(env)

        # 所有任务类型测试
        test_all_task_types()

        # 边界条件测试
        test_boundary_conditions(env)

        # 奖励组件测试
        test_reward_components()

        print("\n" + "=" * 50)
        print("所有测试完成!")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()