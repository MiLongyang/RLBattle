# -*- coding: utf-8 -*-
"""
运行环境测试的脚本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_test():
    """运行测试"""
    try:
        from test_env import main
        main()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保在项目根目录下运行此脚本")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()