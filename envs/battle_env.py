# -*- coding: utf-8 -*-

import numpy as np
import json
from typing import List, Dict, Tuple, Any, Optional
import gym
from gym import spaces


class BattleEnv(gym.Env):
    """海军多智能体对抗环境

    :state_space: 每个智能体的观测空间是一个Box，包含:
        - 位置(2): 导弹当前位置 [x, y]
        - 速度(2): 导弹当前速度 [vx, vy]
        - 加速度(2): 导弹当前加速度 [ax, ay]
        - 与目标舰船距离(1): 最近敌方舰船距离
        总维度: 7

    :action_space: 每个智能体的动作空间是一个Box，表示二维加速度:
        - ax: [-1.0, 1.0] 水平加速度
        - ay: [-1.0, 1.0] 垂直加速度
    """

    def __init__(self, args):
        """初始化环境

        :param args: 包含环境配置的参数对象，必须包含以下字段:
            - num_red: 红方智能体数量
            - num_blue: 蓝方智能体数量
            - task_type: 任务类型
            - episode_limit: 回合步数限制
            - state_dim_env: 状态空间维度
            - ship_params: 舰船参数(JSON字符串)
            - missile_params: 导弹参数(JSON字符串)
            - w_recon_*等: 各任务特定奖励权重
        """
        super(BattleEnv, self).__init__()

        # 基础参数
        self.task_type = args.task_type
        self.episode_limit = args.episode_limit
        self.num_red = args.num_red  # 红方智能体数量
        self.num_blue = args.num_blue  # 蓝方智能体数量
        self.total_agents = self.num_red + self.num_blue  # 总智能体数量
        self.state_dim = args.state_dim_env
        self.action_dim = 2  # 连续动作空间: [ax, ay]
        
        # 动作空间类型支持
        self.action_type = getattr(args, 'action_type', 'continuous')

        # 加载舰船和导弹参数
        self.ship_params = json.loads(args.ship_params)
        self.missile_params = json.loads(args.missile_params)

        # 定义动作空间和观测空间
        self._setup_spaces()

        # 奖励权重配置
        self._init_reward_weights(args)

        # 初始化场景
        self._init_ships_and_missiles()
        self.timestep = 0
        self.last_red_velocities = None
        self.last_blue_velocities = None


    def _setup_spaces(self):
        """设置动作空间和观测空间"""
        # 动作空间: 2维连续动作 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        # 观测空间: 7维 [位置(2), 速度(2), 加速度(2), 与目标舰船距离(1)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )
        self.action_spaces = [self.action_space for _ in range(self.total_agents)]
        self.observation_spaces = [self.observation_space for _ in range(self.total_agents)]

    def reset(self) -> np.ndarray:
        """重置环境到初始状态，返回所有导弹的观测，形状为(n,7)"""
        self._init_ships_and_missiles()
        self.timestep = 0
        self.last_red_velocities = np.zeros((self.num_red, 2), dtype=np.float32)
        self.last_blue_velocities = np.zeros((self.num_blue, 2), dtype=np.float32)
        # 返回所有导弹的观测，shape=(n,7)
        obs = []
        for i in range(len(self.red_missiles)):
            obs.append(self._get_obs(i, 'red'))
        for i in range(len(self.blue_missiles)):
            obs.append(self._get_obs(i, 'blue'))
        return np.array(obs, dtype=np.float32)

    def step(self, actions: list) -> tuple:
        """执行一步环境交互

        :param actions: 所有智能体的动作列表
        :return: (obs, total_reward, done, info)
            - obs: 所有智能体的观测，形状为(n,7)
            - total_reward: 总奖励值
            - done: 是否结束
            - info: 环境信息字典
        """
        self.timestep += 1
        
        # 动作格式转换和验证
        actions = self._validate_and_convert_actions(actions)
        
        red_actions = actions[:self.num_red]
        blue_actions = actions[self.num_red:]
        self._update_velocities(red_actions, blue_actions)

        # 自动引爆检测
        self._check_explosions()
        # 失效修正：导弹引爆或超射程均立即无效化
        for missile in self.red_missiles + self.blue_missiles:
            if missile['alive'] and (missile.get('exploded', False) or missile['range_left'] <= 0):
                missile['alive'] = False

        rewards = self._compute_rewards()
        total_reward = sum(rewards)
        done = self._check_done()
        obs = []
        for i in range(len(self.red_missiles)):
            obs.append(self._get_obs(i, 'red'))
        for i in range(len(self.blue_missiles)):
            obs.append(self._get_obs(i, 'blue'))
        info = {
            'red_alive': np.sum(self.red_alive) if hasattr(self, 'red_alive') else None,
            'blue_alive': np.sum(self.blue_alive) if hasattr(self, 'blue_alive') else None,
            'timestep': self.timestep,
            'red_ships_status': [
                {'name': ship.get('name'), 'hp': ship.get('hp'),
                 'destroyed': ship.get('hp', 1) <= 0 or ship.get('destroyed', False)}
                for ship in self.red_ships
            ],
            'blue_ships_status': [
                {'name': ship.get('name'), 'hp': ship.get('hp'),
                 'destroyed': ship.get('hp', 1) <= 0 or ship.get('destroyed', False)}
                for ship in self.blue_ships
            ]
        }
        return np.array(obs, dtype=np.float32), total_reward, done, info

    def _get_obs(self, idx: int, team: str) -> np.ndarray:
        """获取指定导弹的7维观测: 位置(2), 速度(2), 加速度(2), 与目标舰船距离(1)"""
        if team == 'red':
            missile = self.red_missiles[idx]
            velocities = self.red_velocities
            last_velocities = self.last_red_velocities
            ships = self.blue_ships
        else:
            missile = self.blue_missiles[idx]
            velocities = self.blue_velocities
            last_velocities = self.last_blue_velocities
            ships = self.red_ships
        obs = []
        # 位置(2)
        obs.extend(missile['pos'])
        # 速度(2)
        obs.extend(missile['vel'])
        # 加速度(2)
        acc = missile['vel'] - last_velocities[idx] if last_velocities is not None else np.zeros(2)
        obs.extend(acc)
        # 与最近目标舰船距离(1)
        missile_pos = missile['pos']
        min_dist = min(
            np.linalg.norm(np.array(ship.get('pos', [0, 0])) - missile_pos)
            for ship in ships
        ) if ships else 1e6
        obs.append(min_dist)
        return np.array(obs, dtype=np.float32)

    def _init_reward_weights(self, args) -> None:
        """初始化各任务类型的奖励权重

        Args:
            args: 包含权重参数的对象
        """
        # 任务特定奖励权重
        self.recon_weights = {
            'stealth': getattr(args, 'w_recon_stealth', 1.0),
            'distance': getattr(args, 'w_recon_distance', 0.8),
            'survival': getattr(args, 'w_recon_survival', 3.0)
        }
        self.feint_weights = {
            'deception': getattr(args, 'w_feint_deception', 1.0),
            'survival': getattr(args, 'w_feint_survival', 2.0)
        }
        self.strike_weights = {
            'approach': getattr(args, 'w_strike_approach', 1.0),
            'hit': getattr(args, 'w_strike_hit', 2.0),
            'sync': getattr(args, 'w_strike_sync', 1.0)
        }

    def _init_ships_and_missiles(self) -> None:
        """初始化红蓝双方的舰船和导弹"""
        # 初始化舰船
        self.red_ships = self.ship_params.get('red', [])
        self.blue_ships = self.ship_params.get('blue', [])
        # 为舰船添加队伍信息
        for ship in self.red_ships:
            ship['team'] = 'red'
        for ship in self.blue_ships:
            ship['team'] = 'blue'
        # 预处理导弹参数字典，便于查找
        self.missile_type_params = {m['type']: m for m in self.missile_params.get('type_list', [])}

        # 初始化红方导弹(智能体)
        self.red_missiles = []
        for ship in self.red_ships:
            for missile in ship.get('missiles', []):
                if isinstance(missile, str):
                    # missile为型号字符串，查找参数
                    missile_param = self.missile_type_params.get(missile, {})
                    missile_info = {'type': missile}
                    missile_info.update(missile_param)
                    self.red_missiles.append(self._init_missile(missile_info, ship))
                else:
                    self.red_missiles.append(self._init_missile(missile, ship))

        # 初始化蓝方导弹
        self.blue_missiles = []
        for ship in self.blue_ships:
            for missile in ship.get('missiles', []):
                if isinstance(missile, str):
                    missile_param = self.missile_type_params.get(missile, {})
                    missile_info = {'type': missile}
                    missile_info.update(missile_param)
                    self.blue_missiles.append(self._init_missile(missile_info, ship))
                else:
                    self.blue_missiles.append(self._init_missile(missile, ship))

        # 初始化红蓝导弹速度数组
        self.red_velocities = np.array([m['vel'] for m in self.red_missiles],
                                       dtype=np.float32) if self.red_missiles else np.zeros((0, 2), dtype=np.float32)
        self.blue_velocities = np.array([m['vel'] for m in self.blue_missiles],
                                        dtype=np.float32) if self.blue_missiles else np.zeros((0, 2), dtype=np.float32)
        # 初始化红蓝导弹存活数组
        self.red_alive = np.array([m['alive'] for m in self.red_missiles],
                                  dtype=bool) if self.red_missiles else np.zeros(self.num_red, dtype=bool)
        self.blue_alive = np.array([m['alive'] for m in self.blue_missiles],
                                   dtype=bool) if self.blue_missiles else np.zeros(self.num_blue, dtype=bool)

    def _init_missile(self, missile_info: Dict, ship_info: Dict) -> Dict:
        """初始化单个导弹的状态

        :param missile_info: 导弹的配置参数（支持dict或仅type字符串）
            - type: 导弹型号
            - name: 导弹名称
            - max_speed: 最大速度(马赫数)
            - max_turn_angle: 最大转向角(度)
            - range: 最大射程(km)
            - damage: 基础伤害值
            - role: 导弹角色
        :param ship_info: 发射舰船的信息
            - pos: 舰船位置[x, y]
            - heading: 舰船朝向(度)
            - team: 舰船队伍('red'/'blue')
        :return: 包含导弹完整状态的字典
            - alive: 存活状态
            - pos: 当前位置[x, y]
            - vel: 当前速度[vx, vy]
            - range_left: 剩余航程
            - team: 所属队伍
            - task: 当前任务
            - launch_time: 发射时间
            - hit_time: 命中时间
            - target_id: 目标ID
        """
        # missile_info 必须为 dict，且包含 'type' 字段
        missile = dict(missile_info)
        # 基础属性
        missile['team'] = ship_info.get('team', 'unknown')
        missile['alive'] = True
        missile['pos'] = np.array(ship_info.get('pos', [0.0, 0.0]), dtype=np.float32)
        missile['vel'] = np.zeros(2, dtype=np.float32)
        # 查找导弹详细参数
        missile_type = missile.get('type')
        missile_param = self.missile_type_params.get(missile_type, {})
        # 注入最大转向角和最大速度
        missile['max_turn_angle'] = np.radians(missile_param.get('max_turn_angle', 30))  # 弧度
        missile['max_speed'] = missile_param.get('max_speed', 5.0)
        missile['range_left'] = missile_param.get('range', 0)
        missile['damage'] = missile_param.get('damage', 10)
        # 任务相关
        missile['role'] = missile.get('role', 'main_attack')
        missile['task'] = missile.get('task', self.task_type)
        missile['launch_time'] = 0
        missile['hit_time'] = None
        missile['target_id'] = None
        missile['prev_alive'] = True  # 初始时处于存活状态
        return missile

    def _update_velocities(self, red_actions: List[np.ndarray], blue_actions: List[np.ndarray]) -> None:
        """更新速度和位置，增加最大速度限制"""
        # 定义最大加速度（单位：km/s²）
        max_acceleration = 3.0  # 可调整的参数
        # 更新红方速度和位置
        for i, action in enumerate(red_actions):
            if not self.red_missiles[i]['alive']:
                continue
            missile = self.red_missiles[i]
            # 将动作映射到实际加速度
            ax, ay = float(action[0]) * max_acceleration, float(action[1]) * max_acceleration
            a = np.array([ax, ay], dtype=np.float32)
            # 计算新速度
            new_vel = missile['vel'] + a
            # 限制最大速度
            speed = np.linalg.norm(new_vel)
            if speed > missile['max_speed']:
                new_vel = new_vel / speed * missile['max_speed']
            missile['vel'] = new_vel
            missile['pos'] += missile['vel']
            # 更新剩余航程
            missile['range_left'] -= np.linalg.norm(missile['vel'])
            # 检查失效条件
            if missile['range_left'] <= 0:
                missile['alive'] = False
                self.red_alive[i] = False
        # 更新蓝方速度和位置
        for i, action in enumerate(blue_actions):
            if not self.blue_missiles[i]['alive']:
                continue
            missile = self.blue_missiles[i]
            ax, ay = float(action[0]), float(action[1])
            a = np.array([ax, ay], dtype=np.float32)
            new_vel = missile['vel'] + a
            speed = np.linalg.norm(new_vel)
            if speed > missile['max_speed']:
                new_vel = new_vel / speed * missile['max_speed']
            missile['vel'] = new_vel
            missile['pos'] += missile['vel']
            missile['range_left'] -= np.linalg.norm(missile['vel'])
            if missile['range_left'] <= 0:
                missile['alive'] = False
                self.blue_alive[i] = False

        # 更新导弹的前一时刻存活状态
        for missile in self.red_missiles:
            missile['prev_alive'] = missile.get('alive', True)

        for missile in self.blue_missiles:
            missile['prev_alive'] = missile.get('alive', True)

    def _compute_rewards(self) -> List[float]:
        """计算奖励"""
        rewards = []
        for i in range(self.total_agents):
            if i < self.num_red:
                team = 'red'
                missile = self.red_missiles[i]
            else:
                team = 'blue'
                missile = self.blue_missiles[i - self.num_red]

            if not missile['alive']:
                rewards.append(0.0)
                continue

            reward, _ = self._calc_reward(missile)
            rewards.append(reward)

        return rewards

    def _check_done(self) -> bool:
        """检查是否结束"""
        return all(not m['alive'] for m in self.red_missiles) or all(
            not m['alive'] for m in self.blue_missiles) or self.timestep >= self.episode_limit

    def _check_explosions(self):
        """检查导弹引爆"""
        for missile in self.red_missiles + self.blue_missiles:
            if not missile['alive']:
                continue
            missile_pos = missile['pos']
            explosion_distance = 3.0  # 自动引爆距离为3km
            if missile['team'] == 'red':
                enemy_ships = self.blue_ships
            else:
                enemy_ships = self.red_ships
            for ship in enemy_ships:
                ship_pos = np.array(ship.get('pos', [0, 0]))
                dist = np.linalg.norm(missile_pos - ship_pos)
                if dist <= explosion_distance:
                    missile['exploded'] = True
                    missile['hit_enemy'] = True
                    missile['hit_time'] = self.timestep
                    missile['target_id'] = ship.get('name', None)
                    # 伤害处理
                    damage = missile.get('damage', 10)
                    ship['hp'] = max(0, ship.get('hp', 10) - damage)
                    if ship['hp'] <= 0:
                        ship['destroyed'] = True
                    break
            else:
                missile['exploded'] = False

    def _calc_reward(self, missile: Dict) -> Tuple[float, Dict]:
        """计算导弹的奖励值

        根据任务类型选择不同的奖励计算方式:
        - recon: 考虑隐蔽性、探测距离和生存
        - feint: 考虑欺骗效果和生存
        - strike: 考虑接近程度、命中和协同

        Args:
            missile: 导弹状态字典

        Returns:
            tuple: (reward, info)
                - reward: 总奖励值
                - info: 包含各奖励分量的信息字典
        """
        task = self.task_type

        reward_components = {}
        if task == 'recon':
            reward, reward_components = self._recon_rewards(missile)
        elif task == 'feint':
            reward, reward_components = self._feint_rewards(missile)
        elif task == 'strike':
            reward, reward_components = self._strike_rewards(missile)
        else:
            reward = 0.0

        # 共同奖励：边界惩罚（边界惩罚权重可调整）
        penalty = 0.0
        for i in range(len(missile['pos'])):
            if missile['pos'][i] < 0:
                missile['pos'][i] = 0
                penalty = -1
            elif missile['pos'][i] > 100:  # 假设战场最大x坐标为100
                missile['pos'][i] = 100
                penalty = -1
        reward += penalty * 5

        # 整合信息
        info = {
            **reward_components,
            'target_id': missile.get('target_id', None),
            'decision': missile.get('task', self.task_type)
        }

        return reward, info

    def _recon_rewards(self, missile: Dict) -> Tuple[float, Dict]:
        """计算侦察任务的奖励

        考虑三个方面:
        1. 雷达暴露惩罚
        2. 安全返航奖励
        3. 目标接近奖励

        Args:
            missile: 导弹状态字典

        Returns:
            tuple: (total_reward, reward_components)
        """
        # 计算导弹与最近敌方舰船的最小距离
        missile_pos = missile.get('pos', np.zeros(2))
        min_dist = min(
            np.linalg.norm(np.array(ship.get('pos', [0, 0])) - missile_pos)
            for ship in self.blue_ships
        ) if self.blue_ships else 1e6

        # 1. 隐蔽性惩罚
        r_stealth = -1 if min_dist < 5.0 else 0
        # 2. 生存奖励
        prev_alive = missile.get('prev_alive', True)  # 默认为True（之前存活）
        current_alive = missile.get('alive', False)
        # 只有当从存活变为不存活时才惩罚
        if prev_alive and not current_alive:
            # 第一次变为不存活，给予较大惩罚
            r_survival = -5
        else:
            r_survival = 0
        # 3. 接近目标奖励(完成侦察任务)
        r_distance = 10*(1 - min_dist / 10.0) if min_dist < 10.0 else 0.0
        # 4. 实时态势回传奖励（假设每次step都回传）
        r_transmit = 1

        # 5. 计算总奖励
        reward = (
                self.recon_weights['stealth'] * r_stealth +
                self.recon_weights['distance'] * r_distance +
                self.recon_weights['survival'] * r_survival +
                1 * r_transmit  # 固定权重为1
        )
        components = {
            'stealth': r_stealth,
            'distance': r_distance,
            'survival': r_survival
        }

        return reward, components

    def _feint_rewards(self, missile: Dict) -> Tuple[float, Dict]:
        """计算佯攻任务的奖励

        考虑两个方面:
        1. 敌方防御资源消耗
        2. 生存状态

        Args:
            missile: 导弹状态字典

        Returns:
            tuple: (total_reward, reward_components)
        """
        # 1. 欺骗效果奖励
        # 初始化防御资源消耗计数
        if 'defense_resource_consumed' not in missile:
            missile['defense_resource_consumed'] = 0

        # 检查是否触发了蓝方防御系统
        missile_pos = missile['pos']
        for ship in self.blue_ships:
            ship_pos = np.array(ship.get('pos', [0, 0]))
            dist = np.linalg.norm(missile_pos - ship_pos)
            # 如果导弹进入防御范围，增加防御资源消耗
            if dist < ship.get('defense_range', 10.0):  # 假设默认防御范围10km
                missile['defense_resource_consumed'] += ship.get('defense_cost', 1.0)
        r_deception = missile['defense_resource_consumed']

        # 2. 生存奖励
        r_survival = 1 if missile.get('alive', True) else -5

        # 3. 计算总奖励
        reward = (
                self.feint_weights['deception'] * r_deception +
                self.feint_weights['survival'] * r_survival
        )

        components = {
            'deception': r_deception,
            'survival': r_survival
        }

        return reward, components

    def _strike_rewards(self, missile: Dict) -> Tuple[float, Dict]:
        """计算打击任务的奖励

        考虑三个方面:
        1. 接近目标程度
        2. 命中效果
        3. 协同打击效果

        Args:
            missile: 导弹状态字典

        Returns:
            tuple: (total_reward, reward_components)
        """
        # 1. 接近目标奖励
        missile_pos = missile.get('pos', np.zeros(2))
        min_dist = min(
            np.linalg.norm(np.array(ship.get('pos', [0, 0])) - missile_pos)
            for ship in self.blue_ships
        ) if self.blue_ships else 1e6

        r_approach = 10 * (1 - min_dist / 10.0) if min_dist <= 10 else 0.0

        # 2. 命中奖励
        r_hit = 20 if missile.get('hit_enemy', False) else 0

        # 3. 协同奖励
        team_missiles = self.red_missiles if missile['team'] == 'red' else self.blue_missiles
        velocities = np.array([m['vel'] for m in team_missiles])
        avg_vel = np.mean(velocities, axis=0)
        similarity = np.mean(np.linalg.norm(velocities - avg_vel, axis=1))
        r_cooperation = similarity

        # 4. 计算总奖励
        reward = (
                self.strike_weights['approach'] * r_approach +
                self.strike_weights['hit'] * r_hit +
                self.strike_weights['sync'] * r_cooperation
        )

        components = {
            'approach': r_approach,
            'hit': r_hit,
            'sync': r_cooperation
        }

        return reward, components

    def get_env_info(self) -> Dict[str, Any]:
        """获取环境信息

        Returns:
            包含环境关键信息的字典:
            - num_agents: 智能体数量
            - state_dim: 状态空间维度
            - action_dim: 动作空间维度
            - episode_limit: 回合步数限制
            - obs_dims: 每个智能体的观测维度
            - action_dims: 每个智能体的动作维度
            - action_space_low: 动作空间下界
            - action_space_high: 动作空间上界
        """
        return {
            "num_agents": self.total_agents,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "episode_limit": self.episode_limit,
            "obs_dims": [self.state_dim] * self.total_agents,
            "action_dims": [self.action_dim] * self.total_agents,
            "action_space_low": -1.0,
            "action_space_high": 1.0,
        }

    def _validate_and_convert_actions(self, actions: list) -> list:
        """验证和转换动作格式
        
        Args:
            actions: 原始动作列表
            
        Returns:
            转换后的动作列表，每个动作都是[ax, ay]格式
        """
        converted_actions = []
        
        for i, action in enumerate(actions):
            try:
                if self.action_type == 'discrete':
                    # 离散动作转换为连续动作
                    converted_action = self._discrete_to_continuous(action)
                else:
                    # 连续动作验证和格式化
                    converted_action = self._validate_continuous_action(action)
                
                converted_actions.append(converted_action)
                
            except Exception as e:
                # 如果转换失败，使用默认动作[0.0, 0.0]
                print(f"Warning: Action conversion failed for agent {i}: {e}")
                converted_actions.append([0.0, 0.0])
        
        return converted_actions
    
    def _discrete_to_continuous(self, discrete_action) -> list:
        """将离散动作转换为连续动作
        
        Args:
            discrete_action: 离散动作索引或数组
            
        Returns:
            连续动作[ax, ay]
        """
        # 处理不同的输入格式
        if isinstance(discrete_action, (list, np.ndarray)):
            if len(discrete_action) == 0:
                action_idx = 4  # 默认停止动作
            else:
                action_idx = int(discrete_action[0]) if len(discrete_action) > 0 else 4
        else:
            action_idx = int(discrete_action)
        
        # 9个离散动作对应9个方向
        action_map = {
            0: [-1.0, -1.0],  # 左上
            1: [-1.0, 0.0],   # 左
            2: [-1.0, 1.0],   # 左下
            3: [0.0, -1.0],   # 上
            4: [0.0, 0.0],    # 停止
            5: [0.0, 1.0],    # 下
            6: [1.0, -1.0],   # 右上
            7: [1.0, 0.0],    # 右
            8: [1.0, 1.0],    # 右下
        }
        
        # 确保动作索引在有效范围内
        action_idx = max(0, min(8, action_idx))
        return action_map[action_idx]
    
    def _validate_continuous_action(self, action) -> list:
        """验证和格式化连续动作
        
        Args:
            action: 连续动作
            
        Returns:
            格式化的连续动作[ax, ay]
        """
        # 处理不同的输入格式
        if isinstance(action, (list, tuple)):
            if len(action) >= 2:
                return [float(action[0]), float(action[1])]
            elif len(action) == 1:
                return [float(action[0]), 0.0]
            else:
                return [0.0, 0.0]
        elif isinstance(action, np.ndarray):
            if action.ndim == 0:
                # 0维数组（标量）
                return [float(action), 0.0]
            elif action.size >= 2:
                return [float(action.flat[0]), float(action.flat[1])]
            elif action.size == 1:
                return [float(action.flat[0]), 0.0]
            else:
                return [0.0, 0.0]
        else:
            # 标量值
            return [float(action), 0.0]

    def close(self):
        """
        关闭环境
        """
        print("环境已关闭 (占位符)")
        pass


