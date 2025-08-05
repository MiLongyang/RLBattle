# 多智能体强化学习算法框架

## 项目概述

本项目是一个专为军事仿真场景设计的多智能体强化学习算法框架，支持反舰导弹的智能协同作战训练。框架实现了MADDPG、QMIX和MAPPO三种主流强化学习算法，并针对侦察、佯攻和协同打击三种典型作战任务进行了优化配置。

## 主要特性

### 🎯 多算法支持
- **MADDPG**: 适用于连续动作空间的多智能体深度确定性策略梯度算法
- **QMIX**: 适用于离散动作空间的值函数分解算法  
- **MAPPO**: 通用的多智能体近端策略优化算法

### 🚀 任务特化
- **侦察任务(recon)**: 机动窥探，侦察态势
- **佯攻任务(feint)**: 佯攻消耗，干扰敌方
- **协同打击任务(strike)**: 伴飞压制，毁伤目标

### 🔧 核心功能
- 统一的算法接口和工厂模式
- 灵活的配置管理系统
- 高性能模型管理（≤4秒加载要求）
- 实时性能监控和资源管理
- 完整的异常处理和日志系统
- 可扩展的插件式架构

## 快速开始

### 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 或手动安装
pip install torch>=1.8.0 numpy>=1.19.0 psutil>=5.8.0
# 可选：GPU监控支持
pip install GPUtil>=1.4.0
# 可选：配置文件支持
pip install PyYAML>=5.4.0
```

### 🎯 推荐使用方式

#### 运行展示（一键演示）
```bash
# 自动运行所有算法演示，无需用户输入
python demo_for_client.py
```

#### 日常开发测试（交互式选择）
```bash
# 可选择运行特定算法演示
python run_demo.py
```

#### 命令行训练（完全自定义）
```bash
# 训练MADDPG算法进行侦察任务
python main.py --algorithm MADDPG --task_type recon --num_episodes 5000

# 训练QMIX算法进行协同打击任务
python main.py --algorithm QMIX --task_type strike --num_episodes 10000

# 训练MAPPO算法进行佯攻任务
python main.py --algorithm MAPPO --task_type feint --num_episodes 8000

# 评估训练好的模型
python evaluate.py --algorithm MADDPG --task_type recon --load_model_episode final

# 比较多个算法性能
python evaluate.py --compare --task_type recon
```

#### 配置文件运行
```bash
# 使用预设配置文件
python main.py --config_file configs/demo_config.json
```

### 编程接口

```python
from core import ConfigManager, TrainingManager
from envs.battle_env import BattleEnv
from arguments import get_args

# 初始化
args = get_args()
env = BattleEnv(args)
config_manager = ConfigManager()
training_manager = TrainingManager(config_manager)

# 开始训练
training_manager.initialize_training(
    algorithm_name=args.algorithm,
    task_type=args.task_type,
    env=env,
    custom_config=vars(args)
)

results = training_manager.run_training('./models')
```

## 项目结构

```
├── algorithms/                 # 算法实现
│   ├── base_algorithm.py      # 抽象基类
│   ├── algorithm_factory.py   # 算法工厂
│   ├── maddpg/                # MADDPG算法
│   ├── qmix/                  # QMIX算法
│   └── mappo/                 # MAPPO算法
├── core/                      # 核心模块
│   ├── config_manager.py      # 配置管理
│   ├── training_manager.py    # 训练管理
│   ├── model_manager.py       # 模型管理
│   ├── performance_monitor.py # 性能监控
│   ├── logger.py              # 日志系统
│   ├── error_handler.py       # 错误处理
│   └── exceptions.py          # 异常定义
├── common/                    # 公共组件
│   ├── replay_buffer.py       # MADDPG经验回放池
│   └── episode_replay_buffer.py # QMIX回合回放池
├── envs/                      # 环境模块
├── configs/                   # 配置文件
│   └── demo_config.json       # 演示配置
├── docs/                      # 文档
│   ├── api_reference.md       # API文档
│   └── user_guide.md          # 用户指南
├── tests/                     # 测试文件
│   └── test_algorithm_interface.py # 接口测试
├── logs/                      # 日志文件目录
├── models/                    # 模型保存目录
├── main.py                    # 主训练程序
├── evaluate.py                # 评估程序
├── arguments.py               # 参数配置
├── demo_for_client.py         # 展示脚本
├── run_demo.py                # 开发测试脚本
├── README.md                  # 项目说明
└── requirements.txt           # 依赖列表
```

## 算法特性对比

| 算法 | 动作空间 | 推荐任务 | 特点 |
|------|----------|----------|------|
| MADDPG | 连续 | 侦察、佯攻 | 精确控制，适合连续操作 |
| QMIX | 离散 | 佯攻、协同打击 | 协调性好，理论保证强 |
| MAPPO | 连续/离散 | 通用 | 训练稳定，通用性强 |

## 任务配置优化

### 侦察任务 (recon)
- **目标**: 隐蔽接近，获取信息
- **推荐算法**: MADDPG
- **关键参数**: 降低探索噪声(noise_std=0.05)

### 佯攻任务 (feint)  
- **目标**: 吸引注意，消耗资源
- **推荐算法**: MADDPG/QMIX
- **关键参数**: 增加探索性(noise_std=0.15)

### 协同打击任务 (strike)
- **目标**: 精确协同，毁伤目标  
- **推荐算法**: QMIX
- **关键参数**: 增强网络容量(mixer_hidden_dim=64)

## 🎯 演示脚本

### 展示专用
- `demo_for_client.py` - 自动演示所有算法，专业展示界面
- 无需用户输入，适合正式展示场合

### 开发测试专用  
- `run_demo.py` - 交互式算法选择，清晰的用户界面
- 支持单个算法测试和全部算法演示

## 扩展开发

### 添加新算法

```python
from algorithms.base_algorithm import BaseAlgorithm
from algorithms import AlgorithmRegistry

class MyAlgorithm(BaseAlgorithm):
    def select_actions(self, observations, **kwargs):
        # 实现动作选择逻辑
        pass
    
    def learn(self, **kwargs):
        # 实现学习逻辑
        pass
    
    # 实现其他抽象方法...

# 注册算法
AlgorithmRegistry.register('MyAlgorithm', MyAlgorithm, 'My custom algorithm')
```

### 自定义配置

```python
from core import ConfigManager

config_manager = ConfigManager()

# 获取推荐算法
recommended = config_manager.get_recommended_algorithm('recon')

# 获取任务特定配置
config = config_manager.get_config('MADDPG', 'recon', {
    'custom_param': 'custom_value'
})
```

## 监控和调试

### 性能监控
- CPU/GPU使用率监控
- 内存使用监控  
- 训练指标实时跟踪
- 自动性能优化建议

### 日志系统
- 分级日志记录 (DEBUG/INFO/WARNING/ERROR)
- 自动日志轮转
- 训练过程完整记录
- 错误诊断和恢复建议

### 异常处理
- 完整的异常类型定义
- 自动错误恢复策略
- 详细的错误诊断信息
- 用户友好的错误提示

## 文档

- [API参考文档](docs/api_reference.md) - 详细的接口说明
- [用户使用指南](docs/user_guide.md) - 完整的使用教程
- [算法接口测试](tests/test_algorithm_interface.py) - 接口一致性验证

## 技术规格

- **Python版本**: 3.8+
- **深度学习框架**: PyTorch 1.8+
- **支持设备**: CPU/GPU (CUDA)
- **并发支持**: 多进程训练
- **模型格式**: PyTorch (.pth), 支持扩展ONNX
- **配置格式**: JSON/YAML/命令行参数
- **日志格式**: 结构化日志，支持轮转
- **测试框架**: pytest
- **文档格式**: Markdown + Sphinx

## 贡献指南

1. 所有新算法必须继承`BaseAlgorithm`基类
2. 实现所有抽象方法并通过接口测试
3. 添加相应的配置参数和文档
4. 遵循代码规范和注释标准

