# 数据预处理
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 深度学习依赖 (需要安装: pip install torch)
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


# ==============================================================================
# 1. 数据处理
# ==============================================================================
def load_excel_and_preprocess(excel_path):
    df = pd.read_excel(excel_path, header=None, usecols=[0, 1], skiprows=1)  # 读表
    V_original = df.iloc[:, 0].astype(float).values  # 第1列数据为电压，第2列为电流
    I_original = df.iloc[:, 1].astype(float).values
    valid_mask = np.isfinite(V_original) & np.isfinite(I_original)  # 数据清洗，最后得到一个bool数组，确认数据有效的位置
    V_processed = V_original[valid_mask]  # 得到处理之后的有效的电压，电流
    I_meas_processed = I_original[valid_mask]
    I_min = np.min(I_meas_processed[I_meas_processed > 0]) if np.any(
        I_meas_processed > 0) else 1e-16  # 找到大于0的电流的最小值和最大值
    I_max = np.max(I_meas_processed) if np.any(I_meas_processed > 0) else 1e-4
    return V_processed, I_meas_processed, V_original, I_original, I_min, I_max  # 返回处理后的电压电流，原始电压电流，最小电流，最大电流

# ==# ==============================================================================
 # 2. 物理模型 (紧急修复：必须使用 Code A 的宽范围逻辑)
 # ==============================================================================
def solar_cell_model(V, params, I_min, I_max):
    I_ph, I0, n, Rs, Rsh = params
    Vt = 0.026
    I = np.zeros_like(V, dtype=np.float64)

    # 【关键修改】针对3V高压，给到150
    CLIP_MIN, CLIP_MAX = -50, 150

    for i, v in enumerate(V):
        # --- 步骤1：定义函数 ---
        def f(I_val):
            exp_arg = (v + I_val * Rs) / (n * Vt)
            # 使用宽范围 150
            exp_arg = np.clip(exp_arg, CLIP_MIN, CLIP_MAX)
            exp_term = np.exp(exp_arg) - 1
            shunt_term = (v + I_val * Rs) / Rsh
            return I_val - (I_ph - I0 * exp_term - shunt_term)

        def f_prime(I_val):
            exp_arg = (v + I_val * Rs) / (n * Vt)
            # 使用宽范围 150
            exp_arg = np.clip(exp_arg, CLIP_MIN, CLIP_MAX)
            exp_term = np.exp(exp_arg)
            return 1 + (I0 * Rs / (n * Vt)) * exp_term + (Rs / Rsh)

        # --- 步骤2：牛顿迭代求初值 ---
        init_I = I_ph
        for _ in range(5):
            if not np.isfinite(init_I) or init_I <= 0: break
            f_val = f(init_I)
            f_p_val = f_prime(init_I)
            if abs(f_p_val) < 1e-12: break
            new_init_I = init_I - f_val / f_p_val
            new_init_I = np.clip(new_init_I, I_min * 0.1, I_max * 2)
            if abs(new_init_I - init_I) < 1e-8:
                init_I = new_init_I
                break
            init_I = new_init_I

        # --- 步骤3：不动点迭代精修 ---
        I_i = init_I
        for _ in range(100):
            exp_argument = (v + I_i * Rs) / (n * Vt)
            # 这里也要改用宽范围 150
            exp_argument = np.clip(exp_argument, CLIP_MIN, CLIP_MAX)

            exp_term = np.exp(exp_argument) - 1
            shunt_term = (v + I_i * Rs) / Rsh
            new_I_i = I_ph - I0 * exp_term - shunt_term

            new_I_i = np.clip(new_I_i, I_min * 0.1, I_max * 2)
            if abs(new_I_i - I_i) < 1e-8:
                I_i = new_I_i
                break
            I_i = new_I_i

        I[i] = I_i if np.isfinite(I_i) else init_I

    return I

# ==============================================================================
# 3. 目标函数 (完全保持代码B原样)
# ==============================================================================
def objective_function(params, V, I_meas, I_min, I_max):
    if len(V) == 0 or len(I_meas) == 0:
        return 1e10
    I_sim = solar_cell_model(V, params, I_min, I_max)  # 计算模拟电流
    # 过滤异常值
    if np.any(np.isnan(I_sim)) or np.any(np.isinf(I_sim)):  # 过滤不合理的模拟电流
        return 1e10
    valid_meas_mask = (I_meas > 1e-10) & np.isfinite(I_meas)  # 过滤不合理的实际电流，返回一个bool数组
    if not np.any(valid_meas_mask):
        return 1e10
    # V_valid = V[valid_meas_mask]                              #过滤后的有效实际电流对应的电压
    I_meas_valid = I_meas[valid_meas_mask]  # 有效的实际电流
    I_sim_valid = I_sim[valid_meas_mask]  # 有效的计算电流
    numerator = I_meas_valid - I_sim_valid
    relative_error = numerator / np.max(I_meas_valid)  # 除以测量电流最大值
    loss = np.sqrt((1 / len(I_meas_valid)) * np.sum(relative_error ** 2))

    # 4. 最后校验损失是否有效
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    return loss


class DQN_Network(nn.Module):
    """
    Deep Q-Network 神经网络（只需要定义网络结构）
    注意：主网络和目标网络是同一个类的两个不同实例，在Q_TLBO的__init__中创建
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN_Network, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, action_dim)
        # 可选：添加Dropout防止过拟合
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state):
        """
        前向传播
        输入: state (batch_size, state_dim)
        输出: Q值 (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = self.dropout(x)  # 可选：训练时使用Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # 可选：训练时使用Dropout
        return self.fc3(x)  # 输出层不需要激活函数（直接输出Q值）
    

class ReplayBuffer:
    def __init__(self,capacity:int=10000):
        self.buffer=deque(maxlen=capacity)
    def store(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    def sample(self,batch_size:int):
        states,actions,rewards,next_states,dones=zip(*random.sample(self.buffer,batch_size))
        return np.array(states),np.array(actions),np.array(rewards),np.array(next_states),np.array(dones)
    def __len__(self):
        return len(self.buffer)
    def clear(self):
        return self.buffer.clear()
    def is_empty(self):
        return len(self.buffer)==0
    def capacity(self):
        return self.buffer.maxlen


# ==============================================================================
# 4. Q-Learning 优化的 TLBO 算法 (核心修改部分 - 扩展为DQN)
# ==============================================================================
class Q_TLBO:
    def __init__(self, V: np.ndarray, I_meas: np.ndarray, I_min: float, I_max: float, pop_size: int = 30,
                 max_iter: int = 100, param_bounds: Optional[np.ndarray] = None):
        assert len(V) == len(I_meas), "电压和电流数据长度必须一致"
        assert pop_size > 0, "种群大小必须为正数"
        assert max_iter > 0, "最大迭代次数必须为正数"
        # 绑定实验数据
        self.V = V
        self.I_meas = I_meas
        self.I_min = I_min
        self.I_max = I_max
        # 算法超参数
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.param_dim = 5  # 对应 [I_ph, I0, n, Rs, Rsh]

        # 设置参数边界
        if param_bounds is not None:
            self.param_bounds = param_bounds.astype(np.float64)
        else:
            self.param_bounds = np.array([
                [0.1, 10.0], [1e-60, 1e-50], [1.0, 1.3], [0.001, 0.5], [50, 150]
            ], dtype=np.float64)

        # 运行状态变量
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.iterations = 0

        # ========== DQN 新增属性 ==========
        self.last_best_fitness = float('inf')
        
        # ========== 扩展的状态空间 ==========
        # TODO: 状态空间从2维离散扩展到10-12维连续向量
        # 提示：状态向量应包含：
        #   1. 适应度改进比例 (improvement_ratio)
        #   2. 当前最优适应度（归一化）
        #   3. 种群平均适应度（归一化）
        #   4. 适应度标准差（归一化）
        #   5. 迭代进度 (iteration_ratio)
        #   6. 停滞计数器（归一化）
        #   7. 收敛速率
        #   8. 种群多样性
        #   9. 参数分布范围
        #   10. 历史趋势（可选）
        # ========== 扩展的动作空间 ==========
        # 注意：必须先定义 num_actions，再创建网络
        self.num_actions = 8  # 动作数量（可根据需要调整 4-12）
        self.state_dim = 10  # 状态向量维度（可根据需要调整 7-15）
        self.main_network=DQN_Network(self.state_dim,self.num_actions)
        self.target_network=DQN_Network(self.state_dim,self.num_actions)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
        self.optimizer=optim.Adam(self.main_network_network.parameters(),lr=1e-4)
        self.replay_buffer=ReplayBuffer(capacity=10000)
        self.min_buffer_size=1000
        self.gamma=0.9
        self.epsilon=0.3
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.batch_size=32
        self.target_update_freq=20
        self.state_min=np.zeros(self.state_dim)
        self.state_max=np.ones(self.state_dim)
        self.stagnation_counter=0
        self.fitness_history_window=[]
        self.total_steps = 0
        self.best_fitness_history =[]
        self.mean_fitness_history=[]
        self.std_fitness_history=[]

        # ========== 扩展的动作空间 ==========
        # TODO: 动作空间从2个扩展到6-8个
        # 提示：动作定义建议：
        #   0 = 标准TLBO（保持原逻辑）
        #   1 = 温和变异TLBO（小幅噪声）
        #   2 = 强变异TLBO（大幅噪声）
        #   3 = DE变异（已有_learner_phase_mutation）
        #   4 = 自适应TF TLBO（动态调整TF因子）
        #   5 = 精英引导TLBO（偏向最优个体）
        #   6 = 多样性保持TLBO（防止过早收敛）
        #   7 = 混合策略（组合多种策略）
        self.num_actions = 8  # 动作数量（可根据需要调整 4-12）
        
        # ========== DQN 组件初始化 ==========
        # TODO: 初始化DQN主网络和目标网络
        # 提示：取消下面的注释并实现
        # self.main_network = DQN_Network(self.state_dim, self.num_actions)
        # self.target_network = DQN_Network(self.state_dim, self.num_actions)
        # self.target_network.load_state_dict(self.main_network.state_dict())
        # self.target_network.eval()  # 目标网络设为评估模式
        
        # TODO: 初始化优化器
        # 提示：使用Adam优化器，学习率约 1e-4
        # self.optimizer = optim.Adam(self.main_network.parameters(), lr=1e-4)
        
        # TODO: 初始化经验回放缓冲池
        # 提示：容量建议 10000，最小开始训练大小 1000
        # self.replay_buffer = ReplayBuffer(capacity=10000)
        # self.min_buffer_size = 1000
        
        # ========== RL超参数（调整后）==========
        self.gamma = 0.9  # 折扣因子（提高，考虑长期影响）
        self.epsilon = 0.3  # 初始探索率（提高，动作空间更大）
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减率
        self.batch_size = 32  # 经验回放批次大小
        self.target_update_freq = 20  # 目标网络更新频率（每N步）
        
        # ========== 状态统计和归一化参数 ==========
        # TODO: 用于状态归一化（可选，但推荐）
        self.state_min = np.zeros(self.state_dim) - 1.0  # 状态最小值（动态更新）
        self.state_max = np.ones(self.state_dim) * 2.0   # 状态最大值（动态更新）
        self.stagnation_counter = 0  # 停滞计数器（新增）
        self.fitness_history_window = []  # 用于计算趋势的窗口（可选）
        

    def _initialize_population(self) -> None:
        """保持代码B的线性分布初始化 (虽然这对I0不好，但为了遵守'只改TLBO逻辑'的约束)"""
        self.population = np.zeros((self.pop_size, self.param_dim))
        for d in range(self.param_dim):
            low, high = self.param_bounds[d]
            self.population[:, d] = np.random.uniform(low, high, self.pop_size)

        self.fitness = np.array([self._evaluate_fitness(ind) for ind in self.population])

        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.fitness_history.append(self.best_fitness)
        self.last_best_fitness = self.best_fitness

    def _evaluate_fitness(self, params: np.ndarray) -> float:
        return objective_function(params, self.V, self.I_meas, self.I_min, self.I_max)

    def _apply_bounds(self, individual: np.ndarray) -> np.ndarray:
        for d in range(self.param_dim):
            low, high = self.param_bounds[d]
            individual[d] = np.clip(individual[d], low, high)
        return individual

    # --- DQN 辅助方法 ---
    def _get_state(self) -> np.ndarray:
        """
        TODO: 扩展状态空间 - 返回向量化状态（替代原来的0/1离散状态）
        
        提示：实现状态特征提取，返回归一化的状态向量
        参考维度：
        1. 适应度改进比例 = (last_best - current_best) / last_best
        2. 当前最优适应度（归一化，使用对数或倒数）
        3. 种群平均适应度（归一化）
        4. 适应度标准差（归一化，变异系数）
        5. 迭代进度 = iterations / max_iter
        6. 停滞计数器（归一化）
        7. 收敛速率（最近N代的变化率）
        8. 种群多样性（欧氏距离均值，归一化）
        9. 参数分布范围（各维度标准差均值，归一化）
        10. 历史趋势（最近N代线性拟合斜率，可选）
        
        注意：所有特征都需要归一化到[0,1]或[-1,1]范围
        """
        # TODO: 实现向量化状态提取
        features = []
        
        # 示例：适应度改进比例
        if self.last_best_fitness > 1e-10:
            improvement = (self.last_best_fitness - self.best_fitness) / self.last_best_fitness
            improvement = np.clip(improvement, -1.0, 1.0)
        else:
            improvement = 0.0
        features.append(improvement)
        
        # TODO: 添加其他特征（参考上面的提示）
        # features.append(fitness_norm)
        # features.append(mean_fit_norm)
        # ... 等等
        
        # 确保特征数量匹配state_dim
        while len(features) < self.state_dim:
            features.append(0.0)
        features = features[:self.state_dim]
        
        # 归一化到[0, 1]
        state_vector = np.array(features, dtype=np.float32)
        # TODO: 实现归一化（Min-Max或Z-score）
        # state_vector = (state_vector - self.state_min) / (self.state_max - self.state_min + 1e-10)
        # state_vector = np.clip(state_vector, 0, 1)
        
        return state_vector

    def _choose_action(self, state: np.ndarray) -> int:
        """
        使用DQN选择动作（epsilon-greedy策略）
        """
        if np.random.rand() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.num_actions)
        else:
            # 利用：使用主网络预测Q值，选择Q值最大的动作
            # TODO: 使用主网络预测Q值
            # 提示：
            # state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 添加batch维度
            # with torch.no_grad():
            #     q_values = self.main_network(state_tensor)
            # return q_values.argmax().item()
            return 0  # 临时返回，需要实现

    def _teacher_phase(self) -> None:
        """教师阶段 (完全保持代码B的逻辑)"""
        teacher_idx = np.argmin(self.fitness)
        teacher = self.population[teacher_idx]
        mean_population = np.mean(self.population, axis=0)
        TF = np.random.randint(1, 3)

        for i in range(self.pop_size):
            r = np.random.rand(self.param_dim)
            new_individual = (self.population[i] + r * (teacher - TF * mean_population))
            new_individual = self._apply_bounds(new_individual)
            new_fitness = self._evaluate_fitness(new_individual)

            if new_fitness < self.fitness[i]:
                self.population[i] = new_individual
                self.fitness[i] = new_fitness

    # --- 拆分学习者阶段 ---

    def _learner_phase_standard(self) -> None:
        """动作0: 标准学习者阶段 (即代码B原本的 _learner_phase)"""
        indices = np.random.permutation(self.pop_size)
        for k in range(0, self.pop_size - 1, 2):
            i, j = indices[k], indices[k + 1]

            if self.fitness[i] < self.fitness[j]:
                teacher, learner = self.population[i], self.population[j]
                teacher_fit, learner_fit = self.fitness[i], self.fitness[j]
                update_idx = j
            else:
                teacher, learner = self.population[j], self.population[i]
                teacher_fit, learner_fit = self.fitness[j], self.fitness[i]
                update_idx = i

            r = np.random.rand(self.param_dim)
            new_individual = learner + r * (teacher - learner)
            new_individual = self._apply_bounds(new_individual)
            new_fitness = self._evaluate_fitness(new_individual)

            if new_fitness < learner_fit:
                self.population[update_idx] = new_individual
                self.fitness[update_idx] = new_fitness

    def _learner_phase_mutation(self) -> None:
        """动作3: 变异学习者阶段 (DE变异策略)"""
        for i in range(self.pop_size):
            # 随机选择三个不同的个体进行差分变异 (DE策略)
            idxs = [x for x in range(self.pop_size) if x != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            a, b, c = self.population[a_idx], self.population[b_idx], self.population[c_idx]

            F = 0.5  # 缩放因子
            # 变异公式: new = a + F * (b - c)
            new_individual = a + F * (b - c)

            # 偶尔也结合当前最优解进行引导
            if np.random.rand() < 0.5:
                new_individual = self.best_solution + F * (a - b)

            new_individual = self._apply_bounds(new_individual)
            new_fitness = self._evaluate_fitness(new_individual)

            if new_fitness < self.fitness[i]:
                self.population[i] = new_individual
                self.fitness[i] = new_fitness
    
    # ==============================================================================
    # TODO: 扩展的动作执行方法（需要实现）
    # ==============================================================================
    
    def _learner_phase_mild_mutation(self) -> None:
        """
        TODO: 动作1 - 温和变异TLBO
        提示：在标准学习者阶段后，添加小幅高斯噪声
        """
        # TODO: 实现
        # 1. 执行标准学习者阶段（或调用_learner_phase_standard）
        # 2. 添加噪声：noise = np.random.normal(0, 0.01, self.population.shape)
        # 3. 更新种群：self.population += noise
        # 4. 边界约束：对每个个体应用_apply_bounds
        # 5. 重新评估适应度
        pass
    
    def _learner_phase_strong_mutation(self) -> None:
        """
        TODO: 动作2 - 强变异TLBO
        提示：与温和变异类似，但噪声更大（0.1左右）
        """
        # TODO: 实现
        pass
    
    def _teacher_phase_adaptive_tf(self) -> None:
        """
        TODO: 动作4 - 自适应TF的教师阶段
        提示：根据停滞情况动态调整TF因子
        - 如果stagnation_counter > 5，使用TF=1.5（降低探索）
        - 否则，使用随机TF (1.0-2.0)
        """
        # TODO: 实现
        # 参考_teacher_phase，但TF根据状态自适应调整
        pass
    
    def _learner_phase_elite_guided(self) -> None:
        """
        TODO: 动作5 - 精英引导TLBO
        提示：优先选择top 30%精英个体作为教师，引导其他个体学习
        """
        # TODO: 实现
        # 1. 选择精英：elite_indices = np.argsort(self.fitness)[:int(self.pop_size*0.3)]
        # 2. 在学习者阶段，如果教师是精英，增加学习强度
        pass
    
    def _learner_phase_diversity_preserving(self) -> None:
        """
        TODO: 动作6 - 多样性保持TLBO
        提示：在标准学习者阶段基础上，如果多样性过低，添加扰动
        """
        # TODO: 实现
        # 1. 计算种群多样性：diversity = _calculate_diversity()
        # 2. 如果diversity < 0.3，添加扰动项
        # 3. 其他逻辑类似标准学习者阶段
        pass
    
    def _execute_action(self, action: int) -> None:
        """
        TODO: 根据动作编号执行对应的学习策略
        提示：用if-elif结构，调用对应的phase方法
        """
        # 总是先执行教师阶段（TLBO基础）
        if action == 4:
            # 动作4需要特殊的教师阶段
            self._teacher_phase_adaptive_tf()
        else:
            self._teacher_phase()
        
        # 根据动作执行不同的学习者阶段
        if action == 0:
            self._learner_phase_standard()
        elif action == 1:
            self._learner_phase_mild_mutation()
        elif action == 2:
            self._learner_phase_strong_mutation()
        elif action == 3:
            self._learner_phase_mutation()
        elif action == 4:
            self._learner_phase_standard()
        elif action == 5:
            self._learner_phase_elite_guided()
        elif action == 6:
            self._learner_phase_diversity_preserving()
        elif action == 7:
            # 动作7：混合策略（随机组合两种策略）
            sub_actions = np.random.choice([1, 2, 3, 5], size=2, replace=False)
            self._execute_action(sub_actions[0])
            # 可以执行两次或只执行一次，根据需要调整
        else:
            # 默认使用标准策略
            self._learner_phase_standard()
        
        # 重新评估适应度（某些动作可能已经更新，但确保全部更新）
        # self.fitness = np.array([self._evaluate_fitness(ind) for ind in self.population])
    
    def _calculate_diversity(self) -> float:
        """
        TODO: 计算种群多样性（用于状态特征和动作6）
        提示：计算种群中个体之间的欧氏距离均值
        """
        # TODO: 实现
        # 1. 采样计算（提高效率）：sample_size = min(10, self.pop_size)
        # 2. 计算所有个体对的距离：dist = np.linalg.norm(ind1 - ind2)
        # 3. 返回归一化的多样性值
        return 0.5  # 临时返回值
    
    def _calculate_reward(self, fit_before: float, fit_after: float) -> float:
        """
        TODO: 扩展的奖励函数（替代原来简单的奖励）
        提示：考虑多个因素：
        1. 适应度改进（最重要）：如果best_fitness改进，给大奖励
        2. 种群平均改进：如果fit_after < fit_before，给奖励
        3. 收敛奖励：如果best_fitness < 1e-6，给额外奖励
        4. 多样性奖励/惩罚：多样性过低给惩罚，良好给奖励
        5. 停滞惩罚：stagnation_counter过大时给惩罚
        """
        reward = 0.0
        
        # TODO: 实现奖励计算逻辑
        # 参考：
        # if self.best_fitness < self.last_best_fitness:
        #     improvement = (self.last_best_fitness - self.best_fitness) / self.last_best_fitness
        #     if improvement > 0.1:
        #         reward += 10.0
        #     elif improvement > 0.05:
        #         reward += 5.0
        #     ...
        # else:
        #     if fit_after < fit_before:
        #         reward += 1.0
        #     ...
        
        # 临时简单实现（需要完善）
        if self.best_fitness < self.last_best_fitness:
            reward = 10.0
        elif fit_after < fit_before:
            reward = 1.0
        else:
            reward = -1.0
        
        return reward
    
    def _train_dqn(self) -> Optional[float]:
        """
        TODO: DQN训练方法（从经验回放缓冲池采样并训练网络）
        提示：
        1. 检查缓冲池大小是否足够（>= min_buffer_size）
        2. 从缓冲池采样batch
        3. 计算当前Q值（main_network）
        4. 计算目标Q值（target_network，使用r + gamma * max(Q_next)）
        5. 计算损失（MSE）
        6. 反向传播和参数更新
        返回loss值（用于监控）
        """
        # TODO: 检查缓冲池大小
        # if len(self.replay_buffer) < self.min_buffer_size:
        #     return None
        
        # TODO: 采样批次
        # states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # TODO: 转换为tensor
        # states_tensor = torch.FloatTensor(states)
        # actions_tensor = torch.LongTensor(actions)
        # rewards_tensor = torch.FloatTensor(rewards)
        # next_states_tensor = torch.FloatTensor(next_states)
        # dones_tensor = torch.FloatTensor(dones)
        
        # TODO: 计算当前Q值
        # q_values = self.main_network(states_tensor)
        # current_q = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # TODO: 计算目标Q值（使用target_network）
        # with torch.no_grad():
        #     next_q_values = self.target_network(next_states_tensor)
        #     max_next_q = next_q_values.max(1)[0]
        #     target_q = rewards_tensor + self.gamma * max_next_q * (1 - dones_tensor)
        
        # TODO: 计算损失和反向传播
        # loss = F.mse_loss(current_q, target_q)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        
        # return loss.item()
        return None
    
    def _update_target_network(self) -> None:
        """
        TODO: 更新目标网络（将主网络参数复制到目标网络）
        提示：每target_update_freq步调用一次
        """
        # TODO: 实现
        # self.target_network.load_state_dict(self.main_network.state_dict())
        # self.target_network.eval()
        pass

    def _update_global_best(self) -> None:
        """更新全局历史最优解"""
        current_best_idx = np.argmin(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]
        if current_best_fitness < self.best_fitness:
            self.best_solution = self.population[current_best_idx].copy()
            self.best_fitness = current_best_fitness
        self.fitness_history.append(self.best_fitness)

    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO: DQN控制的主优化循环（替代原来的Q-Learning）
        
        提示：主要改动点：
        1. 状态从离散变为连续向量
        2. 动作空间扩展到8个
        3. 使用DQN（神经网络）替代Q-table
        4. 添加经验回放和目标网络
        5. 扩展的动作执行方法
        6. 更精细的奖励函数
        """
        print("=" * 50)
        print("DQN-TLBO 优化器启动 (扩展状态和动作空间)")
        print(f"状态维度: {self.state_dim}, 动作数量: {self.num_actions}")
        
        self._initialize_population()
        self.last_best_fitness = self.best_fitness
        self.stagnation_counter = 0  # 初始化停滞计数器
        
        # TODO: 动作名称字典（扩展到8个动作）
        action_names = {
            0: "标准TLBO",
            1: "温和变异",
            2: "强变异",
            3: "DE变异",
            4: "自适应TF",
            5: "精英引导",
            6: "多样性保持",
            7: "混合策略"
        }

        # TODO: 用于DQN训练的计数器
        total_steps = 0  # 总步数（用于目标网络更新）

        for iter_num in range(1, self.max_iter + 1):
            # 1. 获取当前状态（现在是向量，不是离散值）
            state = self._get_state()  # 返回np.ndarray，维度为state_dim

            # 2. 选择动作（使用DQN或Q-table）
            action = self._choose_action(state)

            # 3. 记录执行前的种群状态（用于计算奖励）
            fit_before = np.mean(self.fitness)

            # 4. 执行动作（使用扩展的动作执行方法）
            # TODO: 调用_execute_action方法
            self._execute_action(action)
            
            # 确保适应度已更新
            self.fitness = np.array([self._evaluate_fitness(ind) for ind in self.population])
            fit_after = np.mean(self.fitness)

            # 5. 更新最优解
            self._update_global_best()
            
            # 更新停滞计数器
            if abs(self.last_best_fitness - self.best_fitness) < 1e-9:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            # 6. 计算奖励（使用扩展的奖励函数）
            reward = self._calculate_reward(fit_before, fit_after)

            # 7. 获取下一个状态
            next_state = self._get_state()
            
            # 8. TODO: 存储经验到回放缓冲池
            # TODO: 存储经验
            # done = False  # 通常为False（除非达到终止条件）
            # self.replay_buffer.store(state, action, reward, next_state, done)

            # 9. TODO: DQN训练（缓冲池足够大时）
            loss = None
            loss = self._train_dqn()
            
            # 10. TODO: 定期更新目标网络
            total_steps += 1
            if total_steps % self.target_update_freq == 0:
                self._update_target_network()

            # 11. 更新历史记录
            self.last_best_fitness = self.best_fitness
            self.iterations = iter_num
            
            # 12. 衰减探索率（epsilon-greedy）
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # 13. 打印信息
            if iter_num % 20 == 0 or iter_num == self.max_iter:
                action_name = action_names.get(action, f"动作{action}")
                loss_info = f" | DQN Loss: {loss:.6f}" if loss is not None else ""
                print(f"迭代 {iter_num:4d} | 动作: {action_name:12s} | "
                      f"最优Loss: {self.best_fitness:.6e} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"停滞: {self.stagnation_counter}{loss_info}")

        return self.best_solution, np.array(self.fitness_history)


# ========== 主程序示例 ==========
if __name__ == "__main__":
    # 1. 加载你的数据
    excel_path = r"C:\Users\18372\PycharmProjects\pythonProject1\5.xls    "

    try:
        # 尝试读取数据
        V_processed, I_meas_processed, _, _, I_min, I_max = load_excel_and_preprocess(excel_path)
        print("数据加载成功！")
    except Exception as e:
        print(f"数据读取失败: {e}")
        print("正在生成模拟数据用于演示...")
        # 如果读不到文件，生成假数据防止报错
        V_processed = np.linspace(0, 0.6, 30)
        I_meas_processed = np.linspace(0, 3, 30) * (1 - V_processed / 0.7)
        I_min, I_max = 0, 3

    # 2. 自定义参数边界
    custom_bounds = np.array([
        [0.1, 10.0],  # I_ph
        [1e-60, 1e-50],  # I0
        [1.0, 1.3],  # n
        [0.001, 0.5],  # Rs
        [50, 150]  # Rsh
    ])

    # 3. 创建 DQN-TLBO 优化器
    # TODO: 注意 - 在使用DQN之前，需要：
    # 1. 取消import torch等依赖的注释（文件开头）
    # 2. 实现DQN_Network和ReplayBuffer类（第3.5和3.6节）
    # 3. 在__init__中取消DQN初始化的注释
    # 4. 实现所有TODO标记的方法
    
    optimizer = Q_TLBO(
        V=V_processed,
        I_meas=I_meas_processed,
        I_min=I_min,
        I_max=I_max,
        pop_size=40,
        max_iter=150,
        param_bounds=custom_bounds
    )

    # 4. 执行优化
    best_params, history = optimizer.optimize()

    # ==========================================
    # 5. 计算拟合曲线并画图
    # ==========================================

    # 使用找到的最优参数 (best_params) 重新计算一遍电流
    I_fitted = solar_cell_model(V_processed, best_params, I_min, I_max)

    # 设置画图
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 左图：迭代收敛曲线 ---
    axes[0].plot(history, 'b-', linewidth=2)
    axes[0].set_xlabel('迭代次数 (Iteration)')
    axes[0].set_ylabel('适应度 (Loss)')
    axes[0].set_title('Q-TLBO 收敛过程')
    axes[0].grid(True, alpha=0.3)
    axes[0].semilogy()  # 对数坐标看细节

    # --- Right Plot: The Fitted Curve You Want! (右图：拟合曲线) ---
    # 蓝点：真实数据
    axes[1].scatter(V_processed, I_meas_processed, s=15, alpha=0.7,
                    label='实测数据 (Measured)', color='blue')
    # 红线：算法拟合出来的线
    axes[1].plot(V_processed, I_fitted, 'r-', linewidth=2,
                 label='Q-TLBO 拟合 (Fitted)', zorder=10)

    axes[1].set_xlabel('电压 Voltage (V)')
    axes[1].set_ylabel('电流 Current (A)')
    axes[1].legend()
    axes[1].set_title('I-V 特性曲线拟合对比')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 6. 打印误差分析
    mse = np.mean((I_meas_processed - I_fitted) ** 2)
    rmse = np.sqrt(mse)
    print("\n" + "=" * 30)
    print("拟合结果分析:")
    print(f"最优参数: {best_params}")
    print(f"均方误差 (MSE): {mse:.4e}")
    print(f"均方根误差 (RMSE): {rmse:.4e}")
    print("=" * 30)


# ==============================================================================
# TODO: 实现清单和注意事项
# ==============================================================================
"""
=== DQN集成实现清单 ===

1. 【必需】安装依赖：
   pip install torch pandas openpyxl matplotlib numpy

2. 【必需】实现DQN网络类（DQN_Network）：
   - 位置：第3.5节
   - 提示：参考DQNicelake.py中的实现
   - 结构：全连接网络，输入state_dim维，输出num_actions维

3. 【必需】实现经验回放缓冲池（ReplayBuffer）：
   - 位置：第3.6节
   - 功能：存储和采样经验元组 (state, action, reward, next_state, done)

4. 【必需】实现扩展状态向量（_get_state方法）：
   - 位置：Q_TLBO类中
   - 要求：从2维离散扩展到10维连续向量
   - 提示：参考DQN_State_Action_Extension_Example.py中的示例

5. 【必需】实现扩展动作执行（_execute_action及相关方法）：
   - 位置：Q_TLBO类中
   - 要求：从2个动作扩展到8个动作
   - 需要实现：_learner_phase_mild_mutation, _learner_phase_strong_mutation,
              _teacher_phase_adaptive_tf, _learner_phase_elite_guided,
              _learner_phase_diversity_preserving

6. 【必需】实现DQN训练方法（_train_dqn）：
   - 位置：Q_TLBO类中
   - 功能：从缓冲池采样，计算损失，反向传播

7. 【必需】实现奖励函数（_calculate_reward）：
   - 位置：Q_TLBO类中
   - 要求：考虑更多因素（改进、多样性、停滞等）

8. 【必需】在__init__中取消注释DQN初始化：
   - 取消main_network, target_network, optimizer, replay_buffer的注释
   - 取消import torch相关语句的注释

9. 【可选】实现辅助方法：
   - _calculate_diversity: 计算种群多样性
   - _update_target_network: 更新目标网络（已有框架）

=== 测试建议 ===

1. 先测试状态空间扩展：
   - 将use_dqn设为False，保留Q-table模式
   - 测试_get_state返回的向量是否正确

2. 再测试动作空间扩展：
   - 实现部分动作（如动作0,1,3），测试是否正常工作

3. 最后集成DQN：
   - 实现DQN网络和缓冲池
   - 逐步启用DQN训练

=== 关键参数调整 ===

- state_dim: 建议7-12维（太少信息不足，太多计算复杂）
- num_actions: 建议4-8个（太少策略单一，太多难以学习）
- hidden_dim: 建议64-128（根据状态维度调整）
- batch_size: 建议32-64
- target_update_freq: 建议10-50步
- epsilon_decay: 建议0.99-0.999

=== 注意事项 ===

1. 状态归一化非常重要！不同特征范围差异大，必须归一化到[0,1]或[-1,1]

2. 经验回放缓冲池大小建议10000，但需要至少1000条经验才开始训练

3. 目标网络更新不要太频繁，否则不稳定；但也不能太慢，否则学习效率低

4. 探索率epsilon需要合理衰减，早期多探索，后期多利用

5. 奖励函数设计要合理，避免奖励过大导致数值不稳定

6. 如果训练不稳定，可以：
   - 降低学习率（lr = 1e-5）
   - 增加batch_size
   - 调整网络结构（增加或减少层数）
   - 检查状态归一化是否正确

7. 可以先在小规模测试（pop_size=20, max_iter=50）上验证，再扩展到完整规模
"""