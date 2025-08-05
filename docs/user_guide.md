# 用户使用指南

## 快速开始

### 安装要求

- Python 3.8+
- PyTorch 1.8+
- NumPy
- 其他依赖见 requirements.txt

### 🎯 推荐使用方式

#### 1. 甲方展示（一键演示）
```bash
python demo_for_client.py
```
- 自动运行所有算法演示
- 专业的展示界面
- 无需用户输入

#### 2. 开发测试（交互式选择）
```bash
python run_demo.py
```
- 可选择运行特定算法
- 清晰的用户界面
- 支持单个或全部算法演示

#### 3. 完全自定义（命令行）
```bash
python main.py --algorithm MADDPG --task_type recon --num_episodes 10
```

### 基本使用流程

1. **选择运行方式** - 演示脚本或命令行
2. **选择算法和任务类型** - MADDPG/QMIX/MAPPO + recon/feint/strike
3. **开始训练或演示** - 查看运行结果
4. **评估模型**（可选）

## 详细使用说明

### 1. 训练模型

#### 使用命令行

```bash
# 使用MADDPG算法训练侦察任务
python main.py --algorithm MADDPG --task_type recon --num_episodes 5000

# 使用QMIX算法训练协同打击任务
python main.py --algorithm QMIX --task_type strike --num_episodes 10000

# 使用MAPPO算法训练佯攻任务
python main.py --algorithm MAPPO --task_type feint --num_episodes 8000
```

#### 使用配置文件

创建配置文件 `config.json`:

```json
{
    "algorithm": "MADDPG",
    "task_type": "recon",
    "num_episodes": 5000,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "batch_size": 256,
    "save_interval": 100
}
```

运行训练:

```bash
python main.py --config_file config.json
```

### 2. 评估模型

```bash
# 评估训练好的模型
python evaluate.py --algorithm MADDPG --task_type recon --load_model_episode final

# 比较多个算法
python evaluate.py --compare --task_type recon
```

### 3. 演示脚本使用

#### 甲方展示脚本
```python
# 运行自动演示
python demo_for_client.py

# 展示特点：
# - 自动运行MADDPG和QMIX演示
# - 专业的界面和说明
# - 突出技术特色
# - 无需用户交互
```

#### 开发测试脚本
```python
# 运行交互式演示
python run_demo.py

# 选择选项：
# 1. MADDPG + 侦察任务
# 2. QMIX + 协同打击  
# 3. MAPPO + 佯攻任务
# 4. 依次运行所有演示
# 5. 快速测试（默认）
```

### 4. 编程接口使用

```python
from arguments import get_args
from envs.battle_env import BattleEnv
from core.config_manager import ConfigManager
from core.training_manager import TrainingManager

# 获取参数
args = get_args()

# 初始化环境
env = BattleEnv(args)

# 初始化配置和训练管理器
config_manager = ConfigManager()
training_manager = TrainingManager(config_manager)

# 初始化训练
training_manager.initialize_training(
    algorithm_name=args.algorithm,
    task_type=args.task_type,
    env=env,
    custom_config=vars(args)
)

# 开始训练
results = training_manager.run_training('./models')
```

## 算法选择指南

### 任务类型与推荐算法

| 任务类型 | 推荐算法 | 备选算法 | 特点 |
|---------|---------|---------|------|
| 侦察(recon) | MADDPG | MAPPO | 需要精确控制和隐蔽性 |
| 佯攻(feint) | MADDPG | QMIX, MAPPO | 需要探索性和欺骗性 |
| 协同打击(strike) | QMIX | MAPPO | 需要精确协调和时机控制 |

### 算法特性对比

#### MADDPG
- **优点**: 连续动作控制精确，适合需要精细操作的任务
- **缺点**: 训练不稳定，对超参数敏感
- **适用场景**: 侦察任务、需要连续控制的场景

#### QMIX
- **优点**: 训练稳定，协调性好，理论保证强
- **缺点**: 仅支持离散动作，表达能力有限
- **适用场景**: 协同打击、需要精确时机控制的任务

#### MAPPO
- **优点**: 通用性强，训练稳定，支持连续和离散动作
- **缺点**: 样本效率相对较低
- **适用场景**: 通用算法，适合各种任务类型

## 参数配置详解

### 通用参数

- `algorithm`: 算法名称 (MADDPG/QMIX/MAPPO)
- `task_type`: 任务类型 (recon/feint/strike)
- `num_episodes`: 训练回合数
- `batch_size`: 批次大小
- `gamma`: 折扣因子
- `save_interval`: 模型保存间隔

### MADDPG专属参数

- `actor_lr`: Actor网络学习率 (推荐: 1e-4)
- `critic_lr`: Critic网络学习率 (推荐: 1e-3)
- `tau`: 目标网络软更新系数 (推荐: 1e-3)
- `noise_std`: 探索噪声标准差 (推荐: 0.1)
- `buffer_size_maddpg`: 经验回放池大小 (推荐: 1e6)

### QMIX专属参数

- `lr_qmix`: 学习率 (推荐: 5e-4)
- `target_update_interval`: 目标网络更新频率 (推荐: 200)
- `mixer_hidden_dim`: 混合网络隐藏层维度 (推荐: 32)
- `agent_hidden_dim`: 智能体RNN隐藏层维度 (推荐: 64)
- `buffer_size_qmix`: 经验回放池大小 (推荐: 5000)

### MAPPO专属参数

- `actor_lr_mappo`: Actor学习率 (推荐: 3e-4)
- `critic_lr_mappo`: Critic学习率 (推荐: 1e-3)
- `clip_ratio`: PPO裁剪比率 (推荐: 0.2)
- `ppo_epochs`: PPO更新轮数 (推荐: 4)
- `buffer_size_mappo`: 经验缓冲区大小 (推荐: 2048)

## 任务特定配置

### 侦察任务 (recon)

**目标**: 隐蔽接近目标，获取信息并安全撤离

**推荐配置**:
```json
{
    "algorithm": "MADDPG",
    "task_type": "recon",
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "noise_std": 0.05,
    "num_episodes": 5000
}
```

**关键点**:
- 降低探索噪声，提高隐蔽性
- 注重路径优化和安全性

### 佯攻任务 (feint)

**目标**: 吸引敌方注意力，消耗防御资源

**推荐配置**:
```json
{
    "algorithm": "MADDPG",
    "task_type": "feint",
    "actor_lr": 2e-4,
    "critic_lr": 2e-3,
    "noise_std": 0.15,
    "num_episodes": 6000
}
```

**关键点**:
- 增加探索性，提高欺骗效果
- 平衡风险和收益

### 协同打击任务 (strike)

**目标**: 多智能体协同，精确打击目标

**推荐配置**:
```json
{
    "algorithm": "QMIX",
    "task_type": "strike",
    "lr_qmix": 5e-4,
    "mixer_hidden_dim": 64,
    "agent_hidden_dim": 128,
    "num_episodes": 10000
}
```

**关键点**:
- 强调协调性和时机控制
- 增强网络容量以处理复杂协同

## 训练监控

### 训练指标

- **Episode Reward**: 回合奖励，反映整体性能
- **Actor Loss**: Actor网络损失 (MADDPG/MAPPO)
- **Critic Loss**: Critic网络损失 (MADDPG/MAPPO)
- **Q Loss**: Q网络损失 (QMIX)
- **Success Rate**: 任务成功率

### 性能监控

系统会自动监控:
- CPU和GPU使用率
- 内存使用情况
- 训练速度
- 模型保存/加载时间

### 日志文件

- `training.log`: 主要训练日志
- `AlgorithmFramework_detailed.log`: 详细调试信息
- `AlgorithmFramework_errors.log`: 错误日志

## 故障排除

### 常见问题

#### 1. 模型加载失败

**错误**: `ModelLoadError: Failed to load model`

**解决方案**:
- 检查模型文件路径是否正确
- 确认模型文件未损坏
- 验证算法类型匹配

#### 2. 内存不足

**错误**: `ResourceExhaustionError: Memory usage exceeds limit`

**解决方案**:
- 减小batch_size
- 减小buffer_size
- 使用CPU而非GPU训练

#### 3. 训练不收敛

**现象**: 奖励长时间不提升

**解决方案**:
- 调整学习率
- 检查奖励函数设计
- 增加训练回合数
- 尝试不同的算法

#### 4. GPU内存不足

**错误**: CUDA out of memory

**解决方案**:
- 减小batch_size
- 减小网络规模
- 使用梯度累积
- 切换到CPU训练

#### 5. 演示脚本问题

**问题**: 演示脚本运行异常

**解决方案**:
- 确保Python环境正确
- 检查依赖包是否安装完整
- 查看控制台错误信息
- 尝试使用命令行方式运行

**注意**: 显示"占位符"信息是正常的，表示算法框架运行正常，环境模块由其他成员负责。

### 性能优化建议

1. **硬件优化**:
   - 使用GPU加速训练
   - 确保足够的内存
   - 使用SSD存储模型

2. **参数调优**:
   - 从推荐参数开始
   - 逐步调整学习率
   - 平衡探索与利用

3. **训练策略**:
   - 使用课程学习
   - 定期保存检查点
   - 监控训练指标

## 扩展开发

### 添加新算法

1. 继承BaseAlgorithm类
2. 实现所有抽象方法
3. 注册算法到框架
4. 添加配置参数

示例:
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

### 自定义奖励函数

奖励函数的设计由环境模块负责，但可以通过配置参数影响:

```python
# 在环境配置中添加奖励权重
reward_config = {
    'success_reward': 100,
    'step_penalty': -0.1,
    'collision_penalty': -10
}
```

### 添加新任务类型

1. 在ConfigManager中添加任务映射
2. 配置任务特定参数
3. 更新环境以支持新任务

```python
# 在config_manager.py中添加
self.task_algorithm_mapping['new_task'] = 'RECOMMENDED_ALGORITHM'
```