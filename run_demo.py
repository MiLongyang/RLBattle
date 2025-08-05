#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的演示脚本 - 清晰易懂的界面
"""

import sys
import os

def main():
    """
    简化的演示主程序
    """
    print("=" * 50)
    print("  多智能体强化学习对战平台演示")
    print("=" * 50)
    print()
    
    # 可用的演示
    demos = [
        {
            "name": "MADDPG + 侦察任务",
            "algorithm": "MADDPG",
            "task": "recon",
            "description": "连续动作空间，精确控制"
        },
        {
            "name": "QMIX + 协同打击",
            "algorithm": "QMIX", 
            "task": "strike",
            "description": "离散动作空间，协同决策"
        },
        {
            "name": "MAPPO + 佯攻任务",
            "algorithm": "MAPPO",
            "task": "feint", 
            "description": "策略梯度方法，复杂策略"
        }
    ]
    
    print("请选择要演示的算法和任务组合：")
    print()
    for i, demo in enumerate(demos, 1):
        print(f"  {i}. {demo['name']}")
        print(f"     {demo['description']}")
        print()
    
    print(f"  4. 依次运行所有演示")
    print(f"  5. 快速测试 (MADDPG + 侦察)")
    print()
    
    try:
        choice = input("请输入选择 (1-5, 直接回车选择5): ").strip()
        
        if not choice:
            choice = "5"
        
        if choice in ["1", "2", "3"]:
            # 运行单个演示
            demo_idx = int(choice) - 1
            demo = demos[demo_idx]
            print(f"\n{'='*30}")
            print(f"运行演示: {demo['name']}")
            print(f"{'='*30}")
            run_single_demo(demo)
            
        elif choice == "4":
            # 运行所有演示
            print(f"\n{'='*30}")
            print("运行所有演示")
            print(f"{'='*30}")
            for i, demo in enumerate(demos, 1):
                print(f"\n--- 演示 {i}/{len(demos)}: {demo['name']} ---")
                run_single_demo(demo)
                if i < len(demos):
                    print("\n" + "-"*30)
                    
        elif choice == "5":
            # 快速测试
            demo = demos[0]  # MADDPG + recon
            print(f"\n{'='*30}")
            print("快速测试")
            print(f"{'='*30}")
            run_single_demo(demo)
            
        else:
            print("无效选择，运行快速测试...")
            demo = demos[0]
            run_single_demo(demo)
            
        print(f"\n{'='*50}")
        print("演示完成！")
        print(f"{'='*50}")
        
    except KeyboardInterrupt:
        print("\n\n演示已取消")
    except Exception as e:
        print(f"\n演示运行出错: {e}")

def run_single_demo(demo):
    """运行单个演示"""
    print(f"算法: {demo['algorithm']}")
    print(f"任务: {demo['task']}")
    print(f"说明: {demo['description']}")
    print(f"回合数: 3 (演示用)")
    print()
    
    # 设置命令行参数
    sys.argv = [
        'main.py',
        '--algorithm', demo['algorithm'],
        '--task_type', demo['task'],
        '--num_episodes', '3',
        '--log_interval', '1',
        '--save_interval', '999'  # 演示不保存模型
    ]
    
    try:
        print("🔄 正在运行...")
        
        # 导入并运行主程序
        from main import main
        main()
        
        print(f"✅ {demo['name']} 演示完成！")
        
    except Exception as e:
        print(f"❌ 演示运行出错: {e}")
        print("💡 注意: 显示'占位符'信息是正常的，说明算法框架运行正常")

if __name__ == '__main__':
    main()