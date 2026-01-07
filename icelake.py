"""
Frozen Lake 游戏 - 强化学习实现框架
使用Q-table进行学习（简化版：不考虑意外滑行）
"""

import numpy as np
import random
from typing import Tuple, List, Optional


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
        self.map_size=map_size
        # - 起点位置 (0, 0)
        self.start_pos=(0,0)
        # - 终点位置 (map_size-1, map_size-1)
        self.goal_pos=(map_size-1,map_size-1)
        # - 冰洞位置列表
        if holes is None:
            # 默认冰洞位置（4x4地图的标准配置）
            self.holes = [(1, 1), (1, 3), (2, 3), (3, 0)]
        else:
            self.holes = holes
        # - 当前智能体位置
        self.current_pos=(0,0)
        
        
    def reset(self) -> int:
        """
        重置游戏到初始状态
        
        返回:
        - state: 初始状态索引（起点位置）
        """
        # TODO:
        # 1. 将智能体位置重置为起点 (0, 0)
        self.current_pos=(0,0)
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
        self.new_pos=self.get_next_position(self.current_pos,action)
        # 2. 检查边界（如果超出边界，停留在原地）
        if not self.is_valid_position(self.new_pos):
            self.new_pos=self.current_pos
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
        self.current_pos=self.new_pos
        # 5. 返回：新状态, 奖励, 是否结束, 额外信息
        return self.get_state(),reward,done,{}
        
    def get_state(self) -> int:
        """
        获取当前状态索引
        
        返回:
        - state_index: 状态索引（0到map_size^2-1）
        """
        # TODO:
        # 将当前位置坐标 (row, col) 转换为状态索引
        state_index=self.current_pos[0]*self.map_size+self.current_pos[1]
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
        if pos==self.goal_pos:
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
        if pos[0]>=0 and pos[0]<self.map_size and pos[1]>=0 and pos[1]<self.map_size:
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
        # TODO:
        # 根据动作计算新位置：
        if action==0:
            if current_pos[0]>0:
                new_pos=(current_pos[0]-1,current_pos[1])
            else:
                new_pos=current_pos
        if action==1:
            if current_pos[0]<self.map_size-1:
                new_pos=(current_pos[0]+1,current_pos[1])
            else:
                new_pos=current_pos
        if action==2:
            if current_pos[1]>0:
                new_pos=(current_pos[0],current_pos[1]-1)
            else:
                new_pos=current_pos
        if action==3:
            if current_pos[1]<self.map_size-1:
                new_pos=(current_pos[0],current_pos[1]+1)
            else:
                new_pos=current_pos
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
        
        # 循环结束后，grid 变成了完整的二维列表
        # 例如（4x4地图）：
        # grid = [
        #     ['S', 'F', 'F', 'F'],  # 第0行
        #     ['F', 'H', 'F', 'H'],  # 第1行
        #     ['F', 'F', 'F', 'H'],  # 第2行
        #     ['H', 'F', 'F', 'G']   # 第3行
        # ]
        
        # ======================================================================
        # 步骤2：打印地图（将二维列表转换为可视化的字符串）
        # ======================================================================
        # 打印上边框
        # "=" * (self.map_size * 3 + 1) 表示重复 "=" 符号
        # 例如：map_size=4 时，生成 "============="（13个等号）
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
# 2. Q-Learning算法
# ==============================================================================
class QLearning:
    """Q-learning算法实现"""
    
    def __init__(self, num_states: int, num_actions: int, 
                 alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        """
        初始化Q-learning
        
        参数:
        - num_states: 状态数量（地图大小^2）
        - num_actions: 动作数量（4个方向）
        - alpha: 学习率
        - gamma: 折扣因子
        - epsilon: 探索率
        """
        # TODO: 初始化Q-table
        # self.q_table = np.zeros((num_states, num_actions))
        self.q_table=np.zeros((num_states,num_actions))
        # self.alpha = alpha
        self.alpha=alpha
        # self.gamma = gamma
        self.gamma=gamma
        # self.epsilon = epsilonep
        self.epsilon=epsilon
        
        
        
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
        
        # Epsilon-greedy 策略
        if random.random() < epsilon:
            # 探索：随机选择动作
            return random.randint(0, 3)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.q_table[state])
           
        #     return random_action()  # 探索：随机选择动作
        # else:
        #     return argmax(self.q_table[state])  # 利用：选择同一状态下Q值最大的动作
        
    def update_q_table(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool):
        """
        Q-learning更新公式
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        
        参数:
        - state: 当前状态
        - action: 执行的动作
        - reward: 获得的奖励
        - next_state: 下一个状态
        - done: 是否结束
        
        更新逻辑：
        1. 获取当前Q值：Q(s,a)
        2. 计算目标Q值：
           - 如果游戏结束（done=True）：目标Q值 = reward（只有即时奖励）
           - 如果游戏继续（done=False）：目标Q值 = reward + gamma * max(Q(s',a'))
        3. 更新Q值：Q(s,a) = Q(s,a) + alpha * (目标Q值 - 当前Q值)
        """
        # 步骤1：获取当前Q值
        current_q = self.q_table[state, action] 
        
        # 步骤2：计算目标Q值
        if done:
            # 游戏结束：没有下一个状态，目标Q值 = 即时奖励
            target_q = reward
        else:
            # 游戏继续：目标Q值 = 即时奖励 + 折扣后的未来最大Q值
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # 步骤3：更新Q值
        # Q(s,a) = Q(s,a) + alpha * (目标Q值 - 当前Q值)
        self.q_table[state, action] += self.alpha * (target_q - current_q)
       
        
    def save_q_table(self, filepath: str):
        """保存Q-table到文件"""
        np.save(filepath, self.q_table)
        
    def load_q_table(self, filepath: str):
        """从文件加载Q-table"""
        self.q_table = np.load(filepath)
        


# ==============================================================================
# 3. 训练函数
# ==============================================================================
def train(num_episodes: int = 10000, epsilon_start: float = 1.0, 
         epsilon_end: float = 0.01, epsilon_decay: float = 0.995):
    """
    训练Q-learning智能体
    
    参数:
    - num_episodes: 训练轮数
    - epsilon_start: 初始探索率
    - epsilon_end: 最终探索率
    - epsilon_decay: 探索率衰减系数
    
    返回:
    - q_learner: 训练好的Q-learning对象
    - rewards_history: 每轮的奖励历史
    - success_rate_history: 成功率历史
    """
    # TODO:
    # 1. 创建环境和Q-learning对象
    env=FrozenLakeEnvironment()
    q_learner=QLearning(num_states=env.map_size**2,num_actions=4)
    # 2. 初始化epsilon
    epsilon=epsilon_start
    # 3. 循环训练：
    max_steps=100
    rewards_history=[]
    success_rate_history=[]
    for episode in range(num_episodes):
        state=env.reset()
        episode_rewards=[]
        done=False
        for step in range(max_steps):
            action=q_learner.choose_action(state,epsilon)
            next_state,reward,done,info=env.step(action)
            q_learner.update_q_table(state,action,reward,next_state,done)
            episode_rewards.append(reward)
            state=next_state
            if done:
                break
        avg_reward=np.mean(episode_rewards)
        rewards_history.append(avg_reward)
        success=(done and reward>0)
        success_rate_history.append(1 if success else 0)
        epsilon=max(epsilon_end,epsilon*epsilon_decay)
        if (episode+1)%100==0:
            # 计算最近100轮的成功率
            recent_successes = success_rate_history[-100:] if len(success_rate_history) >= 100 else success_rate_history
            recent_success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0
            print(f"回合{episode+1}/{num_episodes}|平均奖励{avg_reward:.2f}|成功率{recent_success_rate:.2%}|Epsilon{epsilon:.3f}")
    
    return q_learner,rewards_history,success_rate_history
    #    - 重置环境
    #    - 选择动作
    #    - 执行动作
    #    - 更新Q-table
    #    - 记录奖励和是否成功
    #    - 衰减epsilon
    # 4. 返回训练结果


# ==============================================================================
# 4. 测试/评估函数
# ==============================================================================
def evaluate(q_learner: QLearning, num_games: int = 100):
    """
    评估训练好的智能体
    
    参数:
    - q_learner: 训练好的Q-learning对象
    - num_games: 测试游戏数量
    
    返回:
    - success_rate: 成功率
    - avg_steps: 平均步数
    - avg_reward: 平均奖励
    """
    # TODO:
    # 1. 创建环境
    env=FrozenLakeEnvironment()
    epsilon=0
    success_count=0
    total_steps=0
    total_reward=0
    for game in range(num_games):
        state=env.reset()
        done=False
        step_count=0
        episode_reward=0
        max_steps = 100  # 防止无限循环
        while not done and step_count < max_steps:
            action=q_learner.choose_action(state,epsilon)
            next_state,reward,done,info=env.step(action)
            step_count+=1
            episode_reward+=reward
            state=next_state
        if done and reward>0:
            success_count+=1
        total_steps+=step_count
        total_reward+=episode_reward
    success_rate=success_count/num_games
    avg_steps=total_steps/num_games
    avg_reward=total_reward/num_games
    return success_rate,avg_steps,avg_reward

        


    # 2. 设置epsilon=0（只利用，不探索）
    # 3. 运行多局游戏
    # 4. 统计成功率、平均步数、平均奖励
# ==============================================================================
# 5. 可视化函数（可选）
# ==============================================================================
def visualize_path(q_learner: QLearning, env: FrozenLakeEnvironment):
    """
    可视化智能体学习到的最优路径
    
    参数:
    - q_learner: 训练好的Q-learning对象
    - env: 游戏环境
    """
    # 1. 从起点开始，根据Q-table选择最优动作
    path = []
    # 重置环境到起点
    env.current_pos = env.start_pos
    max_steps = 100
    step_count = 0
    
    print("开始可视化最优路径：")
    print("=" * 50)
    
    # 2. 逐步移动到终点
    while not env.is_goal(env.current_pos) and step_count < max_steps:
        # 获取当前状态
        state = env.get_state()
        
        # 选择最优动作（epsilon=0，只利用，不探索）
        action = q_learner.choose_action(state, epsilon=0)
        
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
    
    # 3. 打印或绘制路径
    print("=" * 50)
    print(f"路径总长度: {len(path)} 步")
    print(f"路径: {path}")
    
    # 检查是否成功到达终点
    if env.is_goal(env.current_pos):
        print("✓ 成功到达终点！")
    else:
        print("✗ 未能到达终点（可能达到最大步数限制）")


# ==============================================================================
# 6. 主程序
# ==============================================================================
if __name__ == "__main__":
    # TODO: 主程序入口
    # 1. 训练模型
    q_learner,rewards_history,success_rate_history=train()
    q_learner.save_q_table('q_table.npy')
    success_rate,avg_steps,avg_reward=evaluate(q_learner)
    print(f"成功率: {success_rate:.2%}")
    print(f"平均步数: {avg_steps:.2f}")
    print(f"平均奖励: {avg_reward:.2f}")
    env = FrozenLakeEnvironment()
    visualize_path(q_learner, env)

    # 2. 保存Q-table
    # 3. 评估模型
    # 4. 可视化最优路径（可选）
   

