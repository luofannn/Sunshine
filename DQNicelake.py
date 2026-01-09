"""
Frozen Lake 游戏 - 强化学习实现框架
使用DQN（Deep Q-Network）进行学习（简化版：不考虑意外滑行）
"""

import numpy as np
import random
from typing import Tuple, List, Optional
from collections import deque
import copy

# 导入深度学习框架
# 使用 PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ==============================================================================
# 1. 游戏环境类
# ==============================================================================
class FrozenLakeEnvironment:
    """Frozen Lake 游戏环境"""

    def __init__(self, map_size: int = 4, holes: Optional[List[Tuple[int, int]]] = None):
        """
        初始化游戏环境

        参数:
        - map_size: 地图大小（默认4x4）
        - holes: 冰洞位置列表，如果为None则使用默认位置
        """
        # TODO: 初始化参数
        # - 地图大小
        self.map_size = map_size
        # - 起点位置 (0, 0)
        self.start_pos = (0, 0)
        # - 终点位置 (map_size-1, map_size-1)
        self.goal_pos = (map_size - 1, map_size - 1)
        # - 冰洞位置列表
        if holes is None:
            # 默认冰洞位置（4x4地图的标准配置）
            self.holes = [(1, 1), (1, 3), (2, 3), (3, 0)]
        else:
            self.holes = holes
        # - 当前智能体位置
        self.current_pos = (0, 0)

    def reset(self) -> int:
        """
        重置游戏到初始状态

        返回:
        - state: 初始状态索引（起点位置）
        """
        # TODO:
        # 1. 将智能体位置重置为起点 (0, 0)
        self.current_pos = (0, 0)
        # 2. 返回起点的状态索引
        return self.get_state()

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        执行一个动作

        参数:
        - action: 动作索引（0=上, 1=下, 2=左, 3=右）

        返回:
        - next_state: 新状态索引
        - reward: 即时奖励
        - done: 游戏是否结束
        - info: 额外信息（可选）
        """
        # TODO:
        # 1. 根据当前位置和动作计算新位置
        self.new_pos = self.get_next_position(self.current_pos, action)
        # 2. 检查边界（如果超出边界，停留在原地）
        if not self.is_valid_position(self.new_pos):
            self.new_pos = self.current_pos
        # 3. 检查新位置类型：
        #    - 如果是冰洞(H)：游戏结束，返回大惩罚
        if self.is_hole(self.new_pos):
            reward = -100
            done = True
        #    - 如果是终点(G)：游戏结束，返回大奖励
        elif self.is_goal(self.new_pos):
            reward = 100
            done = True
        #    - 如果是冰面(F)：继续游戏，返回小奖励或0
        else:
            reward = -0.1  # 普通移动的小惩罚，鼓励尽快到达终点
            done = False
        # 4. 更新智能体位置
        self.current_pos = self.new_pos
        # 5. 返回：新状态, 奖励, 是否结束, 额外信息
        return self.get_state(), reward, done, {}

    def get_state(self) -> int:
        """
        获取当前状态索引

        返回:
        - state_index: 状态索引（0到map_size^2-1）
        """
        # TODO:
        # 将当前位置坐标 (row, col) 转换为状态索引
        state_index = self.current_pos[0] * self.map_size + self.current_pos[1]
        return state_index

    def is_hole(self, pos: Tuple[int, int]) -> bool:
        """
        检查指定位置是否是冰洞

        参数:
        - pos: 位置坐标 (row, col)

        返回:
        - is_hole: True表示是冰洞
        """
        # 检查位置是否在冰洞列表中
        if self.holes is None:
            return False
        return pos in self.holes

    def is_goal(self, pos: Tuple[int, int]) -> bool:
        """
        检查指定位置是否是终点

        参数:
        - pos: 位置坐标 (row, col)

        返回:
        - is_goal: True表示是终点
        """
        # TODO: 检查位置是否是终点
        if pos == self.goal_pos:
            return True
        else:
            return False

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        检查位置是否在地图范围内

        参数:
        - pos: 位置坐标 (row, col)

        返回:
        - is_valid: True表示位置有效
        """
        # TODO: 检查 row 和 col 是否在 [0, map_size-1] 范围内
        if pos[0] >= 0 and pos[0] < self.map_size and pos[1] >= 0 and pos[1] < self.map_size:
            return True
        else:
            return False

    def get_next_position(self, current_pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        根据当前位置和动作计算下一个位置

        参数:
        - current_pos: 当前位置 (row, col)
        - action: 动作索引（0=上, 1=下, 2=左, 3=右）

        返回:
        - next_pos: 下一个位置 (row, col)
        """
        # 根据动作计算新位置：
        if action == 0:  # 上
            if current_pos[0] > 0:
                new_pos = (current_pos[0] - 1, current_pos[1])
            else:
                new_pos = current_pos
        elif action == 1:  # 下
            if current_pos[0] < self.map_size - 1:
                new_pos = (current_pos[0] + 1, current_pos[1])
            else:
                new_pos = current_pos
        elif action == 2:  # 左
            if current_pos[1] > 0:
                new_pos = (current_pos[0], current_pos[1] - 1)
            else:
                new_pos = current_pos
        elif action == 3:  # 右
            if current_pos[1] < self.map_size - 1:
                new_pos = (current_pos[0], current_pos[1] + 1)
            else:
                new_pos = current_pos
        else:
            # 无效动作，停留在原地
            new_pos = current_pos
        return new_pos
        # 0=上: (row-1, col)
        # 1=下: (row+1, col)
        # 2=左: (row, col-1)
        # 3=右: (row, col+1)

    def render(self):
        """
        可视化当前游戏状态（可选）
        打印地图，显示起点(S)、终点(G)、冰洞(H)、智能体位置(A)、冰面(F)

        原理说明：
        grid 是一个二维列表（列表的列表），用来存储地图每个位置应该显示的字符
        例如：grid = [['S', 'F', 'F', 'F'], ['F', 'H', 'F', 'H'], ...]
        - 外层列表的每个元素代表一行
        - 内层列表的每个元素代表这一行的每个格子
        - grid[row][col] 表示第 row 行、第 col 列的字符
        """
        # ======================================================================
        # 步骤1：创建二维列表 grid，存储每个位置应该显示的字符
        # ======================================================================
        grid = []  # 初始化空的二维列表（外层列表）

        # 外层循环：遍历每一行（从上到下）
        for row in range(self.map_size):
            # 为当前行创建一个空列表（内层列表），用来存储这一行的所有字符
            grid_row = []

            # 内层循环：遍历当前行的每一列（从左到右）
            for col in range(self.map_size):
                # 确定当前位置的坐标
                pos = (row, col)

                # 判断当前位置应该显示什么字符（优先级从高到低）
                if pos == self.current_pos:
                    # 优先级1：智能体当前位置（用 'A' 表示）
                    grid_row.append('A')
                elif pos == self.start_pos:
                    # 优先级2：起点（用 'S' 表示）
                    grid_row.append('S')
                elif pos == self.goal_pos:
                    # 优先级3：终点（用 'G' 表示）
                    grid_row.append('G')
                elif self.is_hole(pos):
                    # 优先级4：冰洞（用 'H' 表示）
                    grid_row.append('H')
                else:
                    # 优先级5：普通冰面（用 'F' 表示）
                    grid_row.append('F')

            # 当前行的所有格子都处理完了，把这一行添加到 grid 中
            # 例如：处理完第0行后，grid = [['S', 'F', 'F', 'F']]
            grid.append(grid_row)

        
        print("\n" + "=" * (self.map_size * 3 + 1))

        # 遍历 grid 的每一行，打印出来
        for row in grid:
            # row 是一个列表，例如：['S', 'F', 'F', 'F']
            # " | ".join(row) 将列表中的元素用 " | " 连接起来
            # 例如：['S', 'F', 'F', 'F'] → "S | F | F | F"
            # 然后在前后加上 "| " 和 " |"，形成 "| S | F | F | F |"
            print("| " + " | ".join(row) + " |")
            # 打印行之间的分隔线
            print("=" * (self.map_size * 3 + 1))

        # 打印图例
        print("\n图例:")
        print("  S = 起点 (Start)")
        print("  G = 终点 (Goal)")
        print("  H = 冰洞 (Hole)")
        print("  F = 冰面 (Frozen)")
        print("  A = 智能体当前位置 (Agent)")
        print()


# ==============================================================================
# 2. 经验回放缓冲区
# ==============================================================================
class ReplayBuffer:
    """
    经验回放缓冲区（Experience Replay Buffer）
    用于存储和采样训练样本，打破数据相关性
    """
    
    def __init__(self, capacity: int = 10000):
        """
        初始化经验回放缓冲区
        
        参数:
        - capacity: 缓冲区最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)  # 使用deque自动处理容量限制

    
    def store(self, state: int, action: int, reward: float, 
              next_state: int, done: bool):
        """
        存储一个经验样本到缓冲区
        
        参数:
        - state: 当前状态
        - action: 执行的动作
        - reward: 获得的奖励
        - next_state: 下一个状态
        - done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
     
    
    def sample(self, batch_size: int) -> Tuple:
        """
        从缓冲区随机采样一批经验
        
        参数:
        - batch_size: 批次大小
        
        返回:
        - states: 状态批次
        - actions: 动作批次
        - rewards: 奖励批次
        - next_states: 下一个状态批次
        - dones: 结束标志批次
        """
        # 随机采样batch_size个样本
        batch = random.sample(self.buffer, batch_size)
        # 分离成不同的数组
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        return states, actions, rewards, next_states, dones
   
    
    def __len__(self) -> int:
        """
        返回缓冲区中样本的数量
        """
        return len(self.buffer)
    


# ==============================================================================
# 3. DQN算法（Deep Q-Network）
# ==============================================================================
class DQN:
    """
    Deep Q-Network (DQN) 算法实现
    使用神经网络近似Q函数，替代Q-table
    """

    def __init__(self, num_states: int, num_actions: int,
                 learning_rate: float = 0.001, gamma: float = 0.9, 
                 epsilon: float = 0.1, hidden_size: int = 128):
        """
        初始化DQN

        参数:
        - num_states: 状态数量（地图大小^2）
        - num_actions: 动作数量（4个方向）
        - learning_rate: 学习率（用于优化器）
        - gamma: 折扣因子
        - epsilon: 探索率
        - hidden_size: 隐藏层神经元数量
        """
        # 初始化DQN组件
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 创建主网络（Main Network）
        # 网络结构: 全连接网络，适合Frozen Lake的离散状态空间
        # 输入: one-hot编码的状态向量 (num_states维)
        # 输出: 每个动作的Q值 (num_actions维)
        self.main_network = nn.Sequential(
            nn.Linear(num_states, hidden_size),  # 输入层 -> 隐藏层1
            nn.ReLU(),                           # ReLU激活函数
            nn.Linear(hidden_size, hidden_size // 2),  # 隐藏层1 -> 隐藏层2
            nn.ReLU(),                           # ReLU激活函数
            nn.Linear(hidden_size // 2, num_actions)    # 隐藏层2 -> 输出层（Q值）
        )
        
        # 创建目标网络（Target Network）
        # 目标网络结构与主网络相同，用于计算目标Q值（稳定训练）
        self.target_network = nn.Sequential(
            nn.Linear(num_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # 创建优化器（用于更新主网络参数）
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        
        # 初始化目标网络参数（与主网络相同）
        # 在创建目标网络后，立即同步参数
        self.target_network.load_state_dict(self.main_network.state_dict())
        # 设置目标网络为评估模式（不需要梯度）
        self.target_network.eval()

    def _state_to_tensor(self, state: int):
        """
        将状态索引转换为神经网络输入格式（One-hot编码）
        
        参数:
        - state: 状态索引（整数）
        
        返回:
        - state_tensor: 转换后的状态张量
        """
        # One-hot编码：创建一个长度为num_states的向量，state位置为1，其他为0
        state_one_hot = np.zeros(self.num_states)
        state_one_hot[state] = 1
        # 转换为tensor
        return torch.tensor(state_one_hot, dtype=torch.float32)

    def choose_action(self, state: int, epsilon: Optional[float] = None) -> int:
        """
        Epsilon-greedy策略选择动作

        参数:
        - state: 当前状态索引
        - epsilon: 探索率（如果为None，使用self.epsilon）

        返回:
        - action: 选择的动作索引（0-3）
        """
        # 如果未指定 epsilon，使用默认值
        if epsilon is None:
            epsilon = self.epsilon

        # 实现 Epsilon-greedy 策略
        if random.random() < epsilon:
            # 探索：随机选择动作
            return random.randint(0, self.num_actions - 1)
        else:
            # 利用：使用主网络选择Q值最大的动作
            # 1. 将state转换为tensor格式（使用_state_to_tensor）
            # 2. 添加batch维度（unsqueeze(0)）
            # 3. 通过主网络前向传播获取Q值
            # 4. 选择Q值最大的动作索引
            state_tensor = self._state_to_tensor(state).unsqueeze(0)  # 添加batch维度
            with torch.no_grad():  # 不需要梯度
                q_values = self.main_network(state_tensor)
            return q_values.argmax().item()
           

    def store_transition(self, replay_buffer: ReplayBuffer, state: int, 
                        action: int, reward: float, next_state: int, done: bool):
        """
        将经验存储到经验回放缓冲区
        
        参数:
        - replay_buffer: 经验回放缓冲区对象
        - state: 当前状态
        - action: 执行的动作
        - reward: 获得的奖励
        - next_state: 下一个状态
        - done: 是否结束
        """
        # TODO: 调用replay_buffer的store方法
        # 提示: replay_buffer.store(state, action, reward, next_state, done)
        replay_buffer.store(state,action,reward,next_state,done)


    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 32):
        """
        从经验回放缓冲区采样并更新网络参数
        
        参数:
        - replay_buffer: 经验回放缓冲区对象
        - batch_size: 批次大小
        """
        # 检查缓冲区是否有足够的样本
        if len(replay_buffer) < batch_size:
            return None
        
        # 从缓冲区采样批次数据
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # 将状态转换为one-hot编码的tensor
        states_one_hot = np.zeros((batch_size, self.num_states))
        states_one_hot[np.arange(batch_size), states] = 1
        states_tensor = torch.tensor(states_one_hot, dtype=torch.float32)
        
        next_states_one_hot = np.zeros((batch_size, self.num_states))
        next_states_one_hot[np.arange(batch_size), next_states] = 1
        next_states_tensor = torch.tensor(next_states_one_hot, dtype=torch.float32)
        
        # 将其他数据转换为tensor
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        
        # 计算当前Q值
        # 1. 将states输入主网络，得到所有动作的Q值
        # 2. 使用gather选择对应actions的Q值
        q_values = self.main_network(states_tensor)
        current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        # 1. 将next_states输入目标网络，得到所有动作的Q值
        # 2. 取最大值
        # 3. 根据done标志计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            max_next_q = next_q_values.max(1)[0]
            # 如果done=True: target_q = rewards
            # 如果done=False: target_q = rewards + gamma * max_next_q
            target_q = rewards_tensor + self.gamma * max_next_q * (1 - dones_tensor)
        
        # 计算损失（MSE损失）
        loss = F.mse_loss(current_q, target_q)
        
        # 反向传播和参数更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        

    def update_target_network(self):
        """
        将主网络的参数同步到目标网络
        通常在训练一定步数后调用（如每100步）
        """
        # TODO: 同步目标网络参数
        # 提示 (PyTorch):
        #   self.target_network.load_state_dict(self.main_network.state_dict())
        # 提示 (TensorFlow):
        #   self.target_network.set_weights(self.main_network.get_weights())
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
        

    def save_model(self, filepath: str):
        """
        保存模型参数到文件
        
        参数:
        - filepath: 保存路径
        """
        # TODO: 保存主网络和目标网络的参数
        # 提示 (PyTorch):
        #   torch.save({
        #       'main_network': self.main_network.state_dict(),
        #       'target_network': self.target_network.state_dict()
        #   }, filepath)
        # 提示 (TensorFlow):
        #   self.main_network.save_weights(filepath + '_main.h5')
        #   self.target_network.save_weights(filepath + '_target.h5')
        torch.save(
            {
                'main_network': self.main_network.state_dict(),
                'target_network':self.target_network.state_dict()
            },
            filepath
        )

    def load_model(self, filepath: str):
        """
        从文件加载模型参数
        
        参数:
        - filepath: 加载路径
        """
        # TODO: 加载主网络和目标网络的参数
        # 提示 (PyTorch):
        #   checkpoint = torch.load(filepath)
        #   self.main_network.load_state_dict(checkpoint['main_network'])
        #   self.target_network.load_state_dict(checkpoint['target_network'])
        # 提示 (TensorFlow):
        #   self.main_network.load_weights(filepath + '_main.h5')
        #   self.target_network.load_weights(filepath + '_target.h5')
        checkpoint = torch.load(filepath)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])



# ==============================================================================
# 4. 训练函数
# ==============================================================================
def train(num_episodes: int = 10000, epsilon_start: float = 1.0,
          epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
          batch_size: int = 32, replay_buffer_size: int = 10000,
          min_buffer_size: int = 1000, target_update_frequency: int = 100):
    """
    训练DQN智能体

    参数:
    - num_episodes: 训练轮数
    - epsilon_start: 初始探索率
    - epsilon_end: 最终探索率
    - epsilon_decay: 探索率衰减系数
    - batch_size: 批次大小（用于经验回放采样）
    - replay_buffer_size: 经验回放缓冲区大小
    - min_buffer_size: 开始训练前的最小缓冲区大小
    - target_update_frequency: 目标网络更新频率（每N步更新一次）

    返回:
    - dqn: 训练好的DQN对象
    - rewards_history: 每轮的奖励历史
    - success_rate_history: 成功率历史
    """
    # TODO: 1. 创建环境和DQN对象
    env = FrozenLakeEnvironment()
    dqn = DQN(num_states=env.map_size ** 2, num_actions=4)
    
    # TODO: 2. 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
    
    # TODO: 3. 初始化训练相关变量
    epsilon = epsilon_start
    max_steps = 100
    rewards_history = []
    success_rate_history = []
    total_steps = 0  # 用于跟踪总步数，用于目标网络更新
    
    # 用于跟踪最佳模型
    best_success_rate = 0.0  # 最佳成功率
    best_avg_reward = float('-inf')  # 最佳平均奖励
    best_model_state = None  # 最佳模型的状态字典
    best_model_filepath = 'best_dqn_model.pth'  # 最佳模型保存路径
    evaluation_interval = 100  # 每N个回合评估一次
    
    # TODO: 4. 训练循环
    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        done = False
        
        for step in range(max_steps):
            # TODO: 4.1 选择动作
            action = dqn.choose_action(state, epsilon)
            
            # TODO: 4.2 执行动作
            next_state, reward, done, info = env.step(action)
            
            # TODO: 4.3 存储经验到回放缓冲区
            dqn.store_transition(replay_buffer, state, action, reward, next_state, done)
            
            # TODO: 4.4 如果缓冲区有足够样本，进行训练
            # 提示: if len(replay_buffer) >= min_buffer_size:
            #           dqn.update(replay_buffer, batch_size)
            if len(replay_buffer) >= min_buffer_size:
                dqn.update(replay_buffer, batch_size)
            
            # TODO: 4.5 定期更新目标网络
            # 提示: 每 target_update_frequency 步更新一次
            # 示例: if total_steps % target_update_frequency == 0:
            #           dqn.update_target_network()
            total_steps += 1
            if total_steps % target_update_frequency == 0:
                dqn.update_target_network()
            
            # 记录奖励
            episode_rewards.append(reward)
            state = next_state
            
            if done:
                break
        
        # TODO: 4.6 记录和统计
        avg_reward = np.mean(episode_rewards)
        rewards_history.append(avg_reward)
        success = (done and reward > 0)
        success_rate_history.append(1 if success else 0)
        
        # TODO: 4.7 衰减epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # TODO: 4.8 定期评估并保存最佳模型
        if (episode + 1) % evaluation_interval == 0:
            # 计算最近N个回合的成功率和平均奖励
            recent_successes = success_rate_history[-evaluation_interval:] if len(success_rate_history) >= evaluation_interval else success_rate_history
            recent_success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0
            recent_avg_reward = np.mean(rewards_history[-evaluation_interval:]) if len(rewards_history) >= evaluation_interval else np.mean(rewards_history)
            
            # 如果当前模型性能更好，保存为最佳模型
            # 优先考虑成功率，如果成功率相同则考虑平均奖励
            if recent_success_rate > best_success_rate or \
               (recent_success_rate == best_success_rate and recent_avg_reward > best_avg_reward):
                best_success_rate = recent_success_rate
                best_avg_reward = recent_avg_reward
                # 保存当前模型的状态（深拷贝）
                best_model_state = {
                    'main_network': copy.deepcopy(dqn.main_network.state_dict()),
                    'target_network': copy.deepcopy(dqn.target_network.state_dict())
                }
                # 立即保存最佳模型到文件（防止训练中断丢失）
                torch.save(best_model_state, best_model_filepath)
                print(f"✓ 发现更好的模型！成功率: {recent_success_rate:.2%}, 平均奖励: {recent_avg_reward:.2f} | 已保存到 {best_model_filepath}")
            
            print(f"回合{episode + 1}/{num_episodes}|平均奖励{avg_reward:.2f}|成功率{recent_success_rate:.2%}|Epsilon{epsilon:.3f}|缓冲区大小{len(replay_buffer)}|最佳成功率{best_success_rate:.2%}")

    # 加载最佳模型（优先从文件加载，如果文件不存在则使用内存中的）
    import os
    if os.path.exists(best_model_filepath):
        print(f"\n从文件加载最佳模型: {best_model_filepath}")
        print(f"最佳模型性能: 成功率 {best_success_rate:.2%}, 平均奖励 {best_avg_reward:.2f}")
        checkpoint = torch.load(best_model_filepath)
        dqn.main_network.load_state_dict(checkpoint['main_network'])
        dqn.target_network.load_state_dict(checkpoint['target_network'])
    elif best_model_state is not None:
        print(f"\n加载内存中的最佳模型（成功率: {best_success_rate:.2%}, 平均奖励: {best_avg_reward:.2f}）")
        dqn.main_network.load_state_dict(best_model_state['main_network'])
        dqn.target_network.load_state_dict(best_model_state['target_network'])
    else:
        print("\n未找到最佳模型，使用训练结束时的模型")

    return dqn, rewards_history, success_rate_history


# ==============================================================================
# 5. 测试/评估函数
# ==============================================================================
def evaluate(dqn: DQN, num_games: int = 100):
    """
    评估训练好的智能体

    参数:
    - dqn: 训练好的DQN对象
    - num_games: 测试游戏数量

    返回:
    - success_rate: 成功率
    - avg_steps: 平均步数
    - avg_reward: 平均奖励
    """
    # TODO: 1. 创建环境
    env = FrozenLakeEnvironment()
    
    # TODO: 2. 设置epsilon=0（只利用，不探索）
    epsilon = 0
    
    # TODO: 3. 初始化统计变量
    success_count = 0
    total_steps = 0
    total_reward = 0
    
    # TODO: 4. 运行多局游戏进行评估
    for game in range(num_games):
        state = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        max_steps = 100  # 防止无限循环
        
        while not done and step_count < max_steps:
            # TODO: 使用DQN选择动作（epsilon=0，纯利用）
            action = dqn.choose_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            step_count += 1
            episode_reward += reward
            state = next_state
        
        # TODO: 统计结果
        if done and reward > 0:
            success_count += 1
        total_steps += step_count
        total_reward += episode_reward
    
    # TODO: 5. 计算平均指标
    success_rate = success_count / num_games
    avg_steps = total_steps / num_games
    avg_reward = total_reward / num_games
    
    return success_rate, avg_steps, avg_reward


# ==============================================================================
# 6. 可视化函数（可选）
# ==============================================================================
def visualize_path(dqn: DQN, env: FrozenLakeEnvironment):
    """
    可视化智能体学习到的最优路径

    参数:
    - dqn: 训练好的DQN对象
    - env: 游戏环境
    """
    # TODO: 1. 初始化路径和重置环境
    path = []
    env.current_pos = env.start_pos
    max_steps = 100
    step_count = 0

    print("开始可视化最优路径：")
    print("=" * 50)

    # TODO: 2. 逐步移动到终点
    while not env.is_goal(env.current_pos) and step_count < max_steps:
        # 获取当前状态
        state = env.get_state()

        # TODO: 使用DQN选择最优动作（epsilon=0，只利用，不探索）
        action = dqn.choose_action(state, epsilon=0)

        # 执行动作（会自动更新 env.current_pos）
        next_state, reward, done, info = env.step(action)

        # 记录路径（使用环境的实际位置）
        path.append(env.current_pos)

        # 打印当前步骤
        print(f"步骤 {step_count + 1}: 位置 {env.current_pos} | 动作 {action} | 奖励 {reward:.2f}")

        step_count += 1

        # 如果游戏结束，退出循环
        if done:
            break

    # TODO: 3. 打印路径结果
    print("=" * 50)
    print(f"路径总长度: {len(path)} 步")
    print(f"路径: {path}")

    # 检查是否成功到达终点
    if env.is_goal(env.current_pos):
        print("✓ 成功到达终点！")
    else:
        print("✗ 未能到达终点（可能达到最大步数限制）")


# ==============================================================================
# 7. 主程序
# ==============================================================================
if __name__ == "__main__":
    # TODO: 主程序入口
    # 1. 训练模型
    dqn, rewards_history, success_rate_history = train()
    
    # TODO: 2. 保存模型
    # 注意：此时dqn已经是最佳模型（train函数已自动加载最佳模型）
    # 最佳模型也会自动保存到 'best_dqn_model.pth'（训练过程中实时保存）
    dqn.save_model('dqn_model.pth')  # 保存最终模型（也是最佳模型）
    
    # TODO: 3. 评估模型
    success_rate, avg_steps, avg_reward = evaluate(dqn)
    print(f"\n评估结果:")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均步数: {avg_steps:.2f}")
    print(f"平均奖励: {avg_reward:.2f}")
    
    # TODO: 4. 可视化最优路径（可选）
    env = FrozenLakeEnvironment()
    visualize_path(dqn, env)


 