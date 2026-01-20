# ==============================================================================
# 修复OpenMP库冲突问题（必须在导入torch之前设置）
# ==============================================================================
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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


# ==============================================================================
# 1. 数据处理
# ==============================================================================
def load_excel_and_preprocess(excel_path):
    df = pd.read_excel(excel_path, header=None, usecols=[0, 1], skiprows=1)
    V_original = df.iloc[:, 0].astype(float).values
    I_original = df.iloc[:, 1].astype(float).values
    valid_mask = np.isfinite(V_original) & np.isfinite(I_original)
    V_processed = V_original[valid_mask]
    I_meas_processed = I_original[valid_mask]
    I_min = np.min(I_meas_processed[I_meas_processed > 0]) if np.any(
        I_meas_processed > 0) else 1e-16
    I_max = np.max(I_meas_processed) if np.any(I_meas_processed > 0) else 1e-4
    return V_processed, I_meas_processed, V_original, I_original, I_min, I_max


# ==============================================================================
# 2. 物理模型
# ==============================================================================
def solar_cell_model(V, params, I_min, I_max):
    I_ph, I0, n, Rs, Rsh = params
    Vt = 0.026
    I = np.zeros_like(V, dtype=np.float64)
    CLIP_MIN, CLIP_MAX = -50, 150

    for i, v in enumerate(V):
        def f(I_val):
            exp_arg = (v + I_val * Rs) / (n * Vt)
            exp_arg = np.clip(exp_arg, CLIP_MIN, CLIP_MAX)
            exp_term = np.exp(exp_arg) - 1
            shunt_term = (v + I_val * Rs) / Rsh
            return I_val - (I_ph - I0 * exp_term - shunt_term)

        def f_prime(I_val):
            exp_arg = (v + I_val * Rs) / (n * Vt)
            exp_arg = np.clip(exp_arg, CLIP_MIN, CLIP_MAX)
            exp_term = np.exp(exp_arg)
            return 1 + (I0 * Rs / (n * Vt)) * exp_term + (Rs / Rsh)

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

        I_i = init_I
        for _ in range(100):
            exp_argument = (v + I_i * Rs) / (n * Vt)
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
# 3. 目标函数
# ==============================================================================
def objective_function(params, V, I_meas, I_min, I_max):
    if len(V) == 0 or len(I_meas) == 0:
        return 1e10
    I_sim = solar_cell_model(V, params, I_min, I_max)
    if np.any(np.isnan(I_sim)) or np.any(np.isinf(I_sim)):
        return 1e10
    valid_meas_mask = (I_meas > 1e-10) & np.isfinite(I_meas)
    if not np.any(valid_meas_mask):
        return 1e10
    I_meas_valid = I_meas[valid_meas_mask]
    I_sim_valid = I_sim[valid_meas_mask]
    numerator = I_meas_valid - I_sim_valid
    relative_error = numerator / np.max(I_meas_valid)
    loss = np.sqrt((1 / len(I_meas_valid)) * np.sum(relative_error ** 2))
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    return loss


# ==============================================================================
# 4. 简化版DQN网络（只替换Q-table，保持简单）
# ==============================================================================
class SimpleDQN(nn.Module):
    """
    简化版DQN网络
    输入: 2维one-hot状态向量 [停滞=0, 进步=1]
    输出: 2个动作的Q值 [动作0=标准TLBO, 动作1=变异]
    """

    def __init__(self, num_states=2, num_actions=2):
        super(SimpleDQN, self).__init__()
        # 简单的全连接网络：2 -> 16 -> 2
        self.fc1 = nn.Linear(num_states, 16)
        self.fc2 = nn.Linear(16, num_actions)

    def forward(self, state):
        """
        前向传播
        输入: state (batch_size, 2) - one-hot编码的状态
        输出: Q值 (batch_size, 2) - 两个动作的Q值
        """
        x = F.relu(self.fc1(state))
        return self.fc2(x)  # 输出Q值，不需要激活函数


# ==============================================================================
# 5. 简化版DQN-TLBO（最小改动版本）
# ==============================================================================
class Q_TLBO:
    def __init__(self, V: np.ndarray, I_meas: np.ndarray, I_min: float, I_max: float,
                 pop_size: int = 30, max_iter: int = 100, param_bounds: Optional[np.ndarray] = None):
        assert len(V) == len(I_meas), "电压和电流数据长度必须一致"
        assert pop_size > 0, "种群大小必须为正数"
        assert max_iter > 0, "最大迭代次数必须为正数"

        # 绑定实验数据
        self.V = V
        self.I_meas = I_meas
        self.I_min = I_min
        self.I_max = I_max
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.param_dim = 5

        # 设置参数边界
        if param_bounds is not None:
            self.param_bounds = param_bounds.astype(np.float64)
        else:
            self.param_bounds = np.array([
                [0.01, 200.0],  # R_sh（并联电阻）：0.01~200Ω
                [1e-70, 1e-10],  # I₀（反向饱和电流）：1e-70~1e-10 A（包含1e-53这类极小数）
                [1.0, 2.2],  # n（理想因子）：1.0~2.2（覆盖特殊薄膜电池）
                [0.00001, 2.0],  # R_s（串联电阻）：0.00001~2Ω（更宽的mΩ级场景）
                [1e-4, 300.0]  # I_ph（光生电流）：1e-4~300 A（覆盖更小弱光电流）
            ], dtype=np.float64)

        # 运行状态变量
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.iterations = 0

        # ========== 简化版DQN属性（保持原有的2状态2动作）==========
        self.last_best_fitness = float('inf')
        self.num_actions = 2  # 动作0=标准TLBO, 动作1=变异
        self.num_states = 2  # 状态0=停滞, 状态1=进步

        # 创建DQN网络（替代Q-table）
        self.dqn_network = SimpleDQN(num_states=self.num_states, num_actions=self.num_actions)
        self.optimizer = optim.Adam(self.dqn_network.parameters(), lr=0.001)  # 学习率

        # RL超参数（保持原有）
        self.alpha = 0.5  # Q-learning学习率（用于计算目标Q值）
        self.gamma = 0.5  # 折扣因子
        self.epsilon = 0.2  # 探索率

        # I₀参数索引（用于对数空间优化，解决浮点数精度问题）
        self.I0_index = 1  # I₀是第2个参数（索引1）

    # --- I₀对数空间转换辅助方法 ---
    def _I0_to_log_space(self, I0_value: float) -> float:
        """将I₀值转换为对数空间"""
        return np.log10(max(I0_value, 1e-100))  # 防止log(0)

    def _I0_from_log_space(self, log_I0: float) -> float:
        """将对数空间的I₀值转换回线性空间"""
        return 10 ** log_I0

    def _I0_apply_bounds_log(self, I0_value: float) -> float:
        """在对数空间中对I₀进行边界裁剪"""
        low, high = self.param_bounds[self.I0_index]
        log_low = np.log10(max(low, 1e-100))
        log_high = np.log10(high)
        log_I0 = np.clip(self._I0_to_log_space(I0_value), log_low, log_high)
        return self._I0_from_log_space(log_I0)

    def _initialize_population(self) -> None:
        """初始化种群（I₀使用对数空间采样，解决浮点数精度问题）"""
        self.population = np.zeros((self.pop_size, self.param_dim))
        for d in range(self.param_dim):
            low, high = self.param_bounds[d]
            
            # I₀参数使用对数空间采样，避免浮点数精度问题
            if d == self.I0_index:
                log_low = np.log10(max(low, 1e-100))  # 限制最小值为1e-100，避免log(0)
                log_high = np.log10(high)
                log_samples = np.random.uniform(log_low, log_high, self.pop_size)
                self.population[:, d] = 10 ** log_samples  # 转换回线性空间
            else:
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
        """应用参数边界（I₀使用对数空间裁剪）"""
        for d in range(self.param_dim):
            # I₀参数使用对数空间裁剪
            if d == self.I0_index:
                individual[d] = self._I0_apply_bounds_log(individual[d])
            else:
                low, high = self.param_bounds[d]
                individual[d] = np.clip(individual[d], low, high)
        return individual

    # --- 简化版DQN方法（保持原有逻辑，只替换Q-table）---

    def _get_state(self) -> int:
        """
        获取当前状态（保持原有逻辑：0=停滞，1=进步）
        返回: 状态索引 (0 或 1)
        """
        if (self.last_best_fitness - self.best_fitness) > 1e-9:
            return 1  # 进步
        else:
            return 0  # 停滞

    def _state_to_onehot(self, state: int) -> np.ndarray:
        """
        将状态索引转换为one-hot编码（用于神经网络输入）
        状态0 -> [1, 0]
        状态1 -> [0, 1]
        """
        onehot = np.zeros(self.num_states)
        onehot[state] = 1.0
        return onehot

    def _choose_action(self, state: int) -> int:
        """
        使用DQN选择动作（替代原来的Q-table查询）
        """
        if np.random.rand() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.num_actions)
        else:
            # 利用：使用DQN网络预测Q值，选择Q值最大的动作
            state_onehot = self._state_to_onehot(state)
            state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0)  # 添加batch维度

            # 确保网络处于评估模式
            self.dqn_network.eval()
            with torch.no_grad():  # 不需要梯度
                q_values = self.dqn_network(state_tensor)  # 输出: (1, 2)
            self.dqn_network.train()  # 恢复训练模式

            return q_values.argmax().item()  # 返回Q值最大的动作索引

    def _update_dqn(self, state: int, action: int, reward: float, next_state: int):
        """
        更新DQN网络（替代原来的Q-table更新）
        使用在线学习，每次立即更新（简化版，不用经验回放）
        """
        # 确保网络处于训练模式
        self.dqn_network.train()

        # 1. 将状态转换为one-hot编码
        state_onehot = self._state_to_onehot(state)
        next_state_onehot = self._state_to_onehot(next_state)

        state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0)  # (1, 2)
        next_state_tensor = torch.FloatTensor(next_state_onehot).unsqueeze(0)  # (1, 2)

        # 2. 计算当前Q值
        q_values = self.dqn_network(state_tensor)  # (1, 2)
        current_q = q_values[0, action]  # 当前状态-动作对的Q值

        # 3. 计算目标Q值（使用Q-learning公式）
        with torch.no_grad():
            next_q_values = self.dqn_network(next_state_tensor)  # (1, 2)
            max_next_q = next_q_values.max().item()  # 下一状态的最大Q值

        target_q = reward + self.gamma * max_next_q

        # 4. 计算损失并更新
        # 注意：current_q是标量tensor，需要与target_q（Python float）计算损失
        target_q_tensor = torch.tensor(target_q, dtype=torch.float32)
        loss = (current_q - target_q_tensor) ** 2  # MSE损失

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _teacher_phase(self) -> None:
        """教师阶段（I₀使用对数空间相对变化）"""
        teacher_idx = np.argmin(self.fitness)
        teacher = self.population[teacher_idx]
        mean_population = np.mean(self.population, axis=0)
        TF = np.random.randint(1, 3)

        for i in range(self.pop_size):
            r = np.random.rand(self.param_dim)
            new_individual = self.population[i].copy()
            
            # 对每个参数维度进行更新
            for d in range(self.param_dim):
                if d == self.I0_index:
                    # I₀使用对数空间的相对变化
                    log_teacher = self._I0_to_log_space(teacher[d])
                    log_mean = self._I0_to_log_space(mean_population[d])
                    log_current = self._I0_to_log_space(new_individual[d])
                    log_change = r[d] * (log_teacher - TF * log_mean)
                    new_individual[d] = self._I0_from_log_space(log_current + log_change)
                else:
                    # 其他参数使用线性空间
                    new_individual[d] = new_individual[d] + r[d] * (teacher[d] - TF * mean_population[d])
            
            new_individual = self._apply_bounds(new_individual)
            new_fitness = self._evaluate_fitness(new_individual)

            if new_fitness < self.fitness[i]:
                self.population[i] = new_individual
                self.fitness[i] = new_fitness

    def _learner_phase_standard(self) -> None:
        """动作0: 标准学习者阶段（I₀使用对数空间相对变化）"""
        indices = np.random.permutation(self.pop_size)
        for k in range(0, self.pop_size - 1, 2):
            i, j = indices[k], indices[k + 1]

            if self.fitness[i] < self.fitness[j]:
                teacher, learner = self.population[i], self.population[j]
                update_idx = j
            else:
                teacher, learner = self.population[j], self.population[i]
                update_idx = i

            r = np.random.rand(self.param_dim)
            new_individual = learner.copy()
            
            # 对每个参数维度进行更新
            for d in range(self.param_dim):
                if d == self.I0_index:
                    # I₀使用对数空间的相对变化
                    log_teacher = self._I0_to_log_space(teacher[d])
                    log_learner = self._I0_to_log_space(learner[d])
                    log_change = r[d] * (log_teacher - log_learner)
                    new_individual[d] = self._I0_from_log_space(log_learner + log_change)
                else:
                    # 其他参数使用线性空间
                    new_individual[d] = learner[d] + r[d] * (teacher[d] - learner[d])
            
            new_individual = self._apply_bounds(new_individual)
            new_fitness = self._evaluate_fitness(new_individual)

            if new_fitness < self.fitness[update_idx]:
                self.population[update_idx] = new_individual
                self.fitness[update_idx] = new_fitness

    def _learner_phase_mutation(self) -> None:
        """动作1: 变异学习者阶段（DE策略，I₀使用对数空间）"""
        for i in range(self.pop_size):
            idxs = [x for x in range(self.pop_size) if x != i]
            a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
            a, b, c = self.population[a_idx], self.population[b_idx], self.population[c_idx]

            F = 0.5
            new_individual = a.copy()
            
            # 对每个参数维度进行变异
            for d in range(self.param_dim):
                if d == self.I0_index:
                    # I₀使用对数空间的相对变化
                    log_a = self._I0_to_log_space(a[d])
                    log_b = self._I0_to_log_space(b[d])
                    log_c = self._I0_to_log_space(c[d])
                    log_change = F * (log_b - log_c)
                    new_individual[d] = self._I0_from_log_space(log_a + log_change)
                else:
                    # 其他参数使用线性空间
                    new_individual[d] = a[d] + F * (b[d] - c[d])

            if np.random.rand() < 0.5:
                # 基于最优解的变异
                for d in range(self.param_dim):
                    if d == self.I0_index:
                        log_best = self._I0_to_log_space(self.best_solution[d])
                        log_a = self._I0_to_log_space(a[d])
                        log_b = self._I0_to_log_space(b[d])
                        log_change = F * (log_a - log_b)
                        new_individual[d] = self._I0_from_log_space(log_best + log_change)
                    else:
                        new_individual[d] = self.best_solution[d] + F * (a[d] - b[d])

            new_individual = self._apply_bounds(new_individual)
            new_fitness = self._evaluate_fitness(new_individual)

            if new_fitness < self.fitness[i]:
                self.population[i] = new_individual
                self.fitness[i] = new_fitness

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
        简化版DQN控制的主循环
        只做最小改动：将Q-table替换为DQN网络
        """
        print("=" * 50)
        print("简化版DQN-TLBO 优化器启动")
        print("状态空间: 2维（停滞/进步）")
        print("动作空间: 2个（标准TLBO/变异）")
        print("=" * 50)

        self._initialize_population()
        self.last_best_fitness = self.best_fitness

        action_names = {0: "标准TLBO", 1: "变异扰动"}

        for iter_num in range(1, self.max_iter + 1):
            # 1. 获取当前状态（保持原有逻辑）
            state = self._get_state()  # 返回0或1

            # 2. 选择动作（使用DQN，替代Q-table）
            action = self._choose_action(state)

            # 3. 执行教师阶段
            self._teacher_phase()

            # 4. 根据动作执行不同的学习者阶段
            fit_before = np.mean(self.fitness)

            if action == 0:
                self._learner_phase_standard()
            else:
                self._learner_phase_mutation()

            fit_after = np.mean(self.fitness)

            # 5. 更新最优解
            self._update_global_best()

            # 6. 计算奖励（保持原有逻辑）
            if self.best_fitness < self.last_best_fitness:
                reward = 10.0
            elif fit_after < fit_before:
                reward = 1.0
            else:
                reward = -1.0

            # 7. 获取下一个状态
            next_state = self._get_state()

            # 8. 更新DQN网络（替代Q-table更新）
            loss = self._update_dqn(state, action, reward, next_state)

            # 9. 更新历史记录
            self.last_best_fitness = self.best_fitness
            self.iterations = iter_num

            # 10. 打印信息
            if iter_num % 20 == 0 or iter_num == self.max_iter:
                state_name = "进步" if state == 1 else "停滞"
                print(f"迭代 {iter_num:4d} | 状态: {state_name} | 动作: {action_names[action]} | "
                      f"最优Loss: {self.best_fitness:.6e} | DQN Loss: {loss:.6f}")

        return self.best_solution, np.array(self.fitness_history)


# ========== 主程序示例 ==========
if __name__ == "__main__":
    excel_path = r"C:\Users\18372\PycharmProjects\pythonProject1\11.xls"

    try:
        V_processed, I_meas_processed, _, _, I_min, I_max = load_excel_and_preprocess(excel_path)
        print("数据加载成功！")
    except Exception as e:
        print(f"数据读取失败: {e}")
        print("正在生成模拟数据用于演示...")
        V_processed = np.linspace(0, 0.6, 30)
        I_meas_processed = np.linspace(0, 3, 30) * (1 - V_processed / 0.7)
        I_min, I_max = 0, 3

    custom_bounds = np.array([
        [1e-4, 300.0], # I_ph（光生电流）：1e-4~300 A（覆盖更小弱光电流）
        [1e-70, 1e-10],  # I₀（反向饱和电流）：1e-70~1e-10 A（包含1e-53这类极小数）
        [1.0, 2.2],  # n（理想因子）：1.0~2.2（覆盖特殊薄膜电池）
        [0.00001, 2.0],  # R_s（串联电阻）：0.00001~2Ω（更宽的mΩ级场景
        [0.01, 200.0],  # R_sh（并联电阻）：0.01~200Ω
    ])

    optimizer = Q_TLBO(
        V=V_processed,
        I_meas=I_meas_processed,
        I_min=I_min,
        I_max=I_max,
        pop_size=40,
        max_iter=150,
        param_bounds=custom_bounds
    )

    best_params, history = optimizer.optimize()

    I_fitted = solar_cell_model(V_processed, best_params, I_min, I_max)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history, 'b-', linewidth=2)
    axes[0].set_xlabel('迭代次数 (Iteration)')
    axes[0].set_ylabel('适应度 (Loss)')
    axes[0].set_title('简化版DQN-TLBO 收敛过程')
    axes[0].grid(True, alpha=0.3)
    axes[0].semilogy()

    axes[1].scatter(V_processed, I_meas_processed, s=15, alpha=0.7,
                    label='实测数据 (Measured)', color='blue')
    axes[1].plot(V_processed, I_fitted, 'r-', linewidth=2,
                 label='DQN-TLBO 拟合 (Fitted)', zorder=10)
    axes[1].set_xlabel('电压 Voltage (V)')
    axes[1].set_ylabel('电流 Current (A)')
    axes[1].legend()
    axes[1].set_title('I-V 特性曲线拟合对比')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    mse = np.mean((I_meas_processed - I_fitted) ** 2)
    rmse = np.sqrt(mse)
    print("\n" + "=" * 30)
    print("拟合结果分析:")
    print(f"最优参数: {best_params}")
    print(f"均方误差 (MSE): {mse:.4e}")
    print(f"均方根误差 (RMSE): {rmse:.4e}")
    print("=" * 30)