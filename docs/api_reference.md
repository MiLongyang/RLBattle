# API Reference

## BaseAlgorithm 抽象基类

`BaseAlgorithm` 是所有强化学习算法的抽象基类，定义了统一的接口规范。

### 类定义

```python
class BaseAlgorithm(ABC):
    """
    所有强化学习算法的抽象基类。
    定义了所有算法必须实现的通用接口，以确保框架的可扩展性。
    任何新的算法都应该继承这个类，并实现其所有抽象方法。
    """
```

### 初始化方法

```python
def __init__(self, obs_dims: List[int], action_dims: List[int], 
             num_agents: int, state_dim: int, args: Any, device: torch.device):
```

**参数:**
- `obs_dims` (List[int]): 每个智能体的观测维度列表
- `action_dims` (List[int]): 每个智能体的动作维度列表  
- `num_agents` (int): 智能体数量
- `state_dim` (int): 全局状态维度
- `args` (Any): 算法配置参数
- `device` (torch.device): 计算设备

### 抽象方法

#### select_actions

```python
@abstractmethod
def select_actions(self, observations: List[np.ndarray], 
                  **kwargs) -> Tuple[List[np.ndarray], Optional[Any]]:
```

根据当前状态为所有智能体选择动作。

**参数:**
- `observations` (List[np.ndarray]): 所有智能体的观测列表
- `**kwargs`: 算法特定的额外参数（如hidden_states, epsilon, add_noise等）

**返回:**
- `Tuple[actions, additional_info]`: 
  - `actions` (List[np.ndarray]): 所有智能体的动作列表
  - `additional_info` (Optional[Any]): 算法特定的额外信息（如新的hidden_states）

#### learn

```python
@abstractmethod
def learn(self, **kwargs) -> Dict[str, float]:
```

执行一次学习/更新步骤。

**参数:**
- `**kwargs`: 算法特定的学习参数

**返回:**
- `Dict[str, float]`: 训练指标字典（如loss, actor_loss, critic_loss等）

#### save_models

```python
@abstractmethod
def save_models(self, save_dir: str, episode: Union[str, int]) -> None:
```

保存模型权重和相关信息。

**参数:**
- `save_dir` (str): 保存目录路径
- `episode` (Union[str, int]): 回合数或标识符

#### load_models

```python
@abstractmethod
def load_models(self, load_dir: str, episode: Union[str, int]) -> None:
```

加载模型权重和相关信息。

**参数:**
- `load_dir` (str): 加载目录路径
- `episode` (Union[str, int]): 回合数或标识符

#### get_training_metrics

```python
@abstractmethod
def get_training_metrics(self) -> Dict[str, float]:
```

获取当前训练指标。

**返回:**
- `Dict[str, float]`: 训练指标字典

### 公共方法

#### set_training_mode

```python
def set_training_mode(self, training: bool = True) -> None:
```

设置训练/评估模式。

**参数:**
- `training` (bool): True为训练模式，False为评估模式

#### update_training_step

```python
def update_training_step(self) -> None:
```

更新训练步数。

#### update_episode_count

```python
def update_episode_count(self) -> None:
```

更新回合数。

#### get_algorithm_info

```python
def get_algorithm_info(self) -> Dict[str, Any]:
```

获取算法基本信息。

**返回:**
- `Dict[str, Any]`: 算法信息字典，包含算法名称、智能体数量、维度信息等

#### validate_observations

```python
def validate_observations(self, observations: List[np.ndarray]) -> bool:
```

验证观测数据的有效性。

**参数:**
- `observations` (List[np.ndarray]): 观测数据列表

**返回:**
- `bool`: 验证是否通过

## AlgorithmFactory 算法工厂

### create_algorithm

```python
@staticmethod
def create_algorithm(algorithm_name: str, env_info: Dict[str, Any], 
                    config: Dict[str, Any]) -> BaseAlgorithm:
```

创建算法实例。

**参数:**
- `algorithm_name` (str): 算法名称
- `env_info` (Dict[str, Any]): 环境信息
- `config` (Dict[str, Any]): 配置参数

**返回:**
- `BaseAlgorithm`: 算法实例

**异常:**
- `AlgorithmNotFoundError`: 算法未找到
- `AlgorithmInitializationError`: 算法初始化失败

## AlgorithmRegistry 算法注册器

### register

```python
@classmethod
def register(cls, name: str, algorithm_class: Type[BaseAlgorithm], 
             description: str = "", **kwargs) -> None:
```

注册算法。

**参数:**
- `name` (str): 算法名称
- `algorithm_class` (Type[BaseAlgorithm]): 算法类
- `description` (str): 算法描述
- `**kwargs`: 额外信息

### get_algorithm

```python
@classmethod
def get_algorithm(cls, name: str) -> Type[BaseAlgorithm]:
```

获取算法类。

**参数:**
- `name` (str): 算法名称

**返回:**
- `Type[BaseAlgorithm]`: 算法类

### list_algorithms

```python
@classmethod
def list_algorithms(cls) -> List[str]:
```

列出所有已注册的算法。

**返回:**
- `List[str]`: 算法名称列表

## 具体算法实现

### MADDPG

Multi-Agent Deep Deterministic Policy Gradient算法实现。

**特点:**
- 支持连续动作空间
- 集中式训练，分布式执行
- 适用于侦察和佯攻任务

**特有参数:**
- `actor_lr`: Actor学习率
- `critic_lr`: Critic学习率
- `tau`: 目标网络软更新系数
- `noise_std`: 动作噪声标准差

### QMIX

Q-Mix算法实现，支持值函数分解。

**特点:**
- 支持离散动作空间
- 单调性约束的值函数分解
- 适用于佯攻和协同打击任务

**特有参数:**
- `lr_qmix`: 学习率
- `target_update_interval`: 目标网络更新频率
- `mixer_hidden_dim`: 混合网络隐藏层维度
- `agent_hidden_dim`: 智能体RNN隐藏层维度

### MAPPO

Multi-Agent Proximal Policy Optimization算法实现。

**特点:**
- 支持连续和离散动作空间
- PPO的多智能体扩展
- 通用算法，适用于所有任务类型

**特有参数:**
- `actor_lr_mappo`: Actor学习率
- `critic_lr_mappo`: Critic学习率
- `clip_ratio`: PPO裁剪比率
- `ppo_epochs`: PPO更新轮数
- `use_centralized_critic`: 是否使用中心化Critic

## 使用示例

### 基本使用

```python
from algorithms import AlgorithmFactory
from core.config_manager import ConfigManager

# 初始化配置管理器
config_manager = ConfigManager()

# 获取配置
config = config_manager.get_config('MADDPG', 'recon')
config['device'] = torch.device('cuda')

# 创建算法实例
algorithm = AlgorithmFactory.create_algorithm('MADDPG', env_info, config)

# 选择动作
actions, _ = algorithm.select_actions(observations, add_noise=True)

# 学习更新
metrics = algorithm.learn()

# 保存模型
algorithm.save_models('./models', 'final')
```

### 演示脚本使用

#### 甲方展示
```python
# 直接运行自动演示
python demo_for_client.py

# 或在代码中调用
from demo_for_client import main
main()
```

#### 开发测试
```python
# 交互式演示
python run_demo.py

# 或在代码中调用
from run_demo import main
main()
```

### 完整训练流程

```python
from arguments import get_args
from envs.battle_env import BattleEnv
from core.config_manager import ConfigManager
from core.training_manager import TrainingManager

# 获取参数
args = get_args()

# 初始化环境
env = BattleEnv(args)

# 初始化管理器
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

### 注册自定义算法

```python
from algorithms import AlgorithmRegistry
from algorithms.base_algorithm import BaseAlgorithm

class MyCustomAlgorithm(BaseAlgorithm):
    # 实现所有抽象方法
    pass

# 注册算法
AlgorithmRegistry.register(
    'MyAlgorithm', 
    MyCustomAlgorithm, 
    'My custom algorithm description'
)
```

## 异常处理

### 异常类型

- `AlgorithmNotFoundError`: 算法未找到或未注册
- `AlgorithmInitializationError`: 算法初始化失败
- `IncompatibleAlgorithmError`: 算法与任务类型不兼容
- `ModelLoadError`: 模型加载失败
- `ModelSaveError`: 模型保存失败
- `ConfigValidationError`: 配置参数验证失败

### 异常处理示例

```python
from algorithms import AlgorithmFactory, AlgorithmNotFoundError

try:
    algorithm = AlgorithmFactory.create_algorithm('UnknownAlg', env_info, config)
except AlgorithmNotFoundError as e:
    print(f"Algorithm not found: {e}")
    print(f"Available algorithms: {e.details['available_algorithms']}")
```