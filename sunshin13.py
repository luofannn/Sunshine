# 数据预处理
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# ==============================================================================
# 4. Q-Learning 优化的 TLBO 算法 (核心修改部分)
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

        # ========== Q-Learning 新增属性 ==========
        self.last_best_fitness = float('inf')
        # 动作空间: 0 = 标准TLBO学习(代码B原逻辑), 1 = 随机扰动/变异(用于跳出局部最优)
        self.num_actions = 2
        # 状态空间: 0 = 停滞(Stagnation), 1 = 进步(Improvement)
        self.num_states = 2
        # Q表初始化 (2行 x 2列)
        self.q_table = np.zeros((self.num_states, self.num_actions))

        # RL超参数
        self.alpha = 0.5  # 学习率
        self.gamma = 0.5  # 折扣因子
        self.epsilon = 0.2  # 探索率

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

    # --- Q-Learning 辅助方法 ---
    def _get_state(self):
        """判断当前处于 '进步' 还是 '停滞' 状态"""
        # 如果本轮迭代相比上一轮，最优适应度下降幅度大于阈值，则认为在进步
        if (self.last_best_fitness - self.best_fitness) > 1e-9:
            return 1  # 进步
        else:
            return 0  # 停滞

    def _choose_action(self, state):
        """Epsilon-Greedy 策略"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])

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
        """动作1: 变异学习者阶段 (引入随机性，帮助跳出代码B容易陷入的局部最优)"""
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

    def _update_global_best(self) -> None:
        """更新全局历史最优解"""
        current_best_idx = np.argmin(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]
        if current_best_fitness < self.best_fitness:
            self.best_solution = self.population[current_best_idx].copy()
            self.best_fitness = current_best_fitness
        self.fitness_history.append(self.best_fitness)

    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Q-Learning 控制的主循环"""
        print("=" * 50)
        print("Q-TLBO 优化器启动 (基于代码B修改)")
        self._initialize_population()
        self.last_best_fitness = self.best_fitness

        action_names = {0: "标准TLBO", 1: "变异扰动"}

        for iter_num in range(1, self.max_iter + 1):
            # 1. 获取当前状态
            state = self._get_state()

            # 2. 选择动作 (标准学习 or 变异)
            action = self._choose_action(state)

            # 3. 总是执行教师阶段 (TLBO的基础)
            self._teacher_phase()

            # 4. 根据动作执行不同的学习者阶段
            # 记录学习前的平均适应度，用于辅助计算奖励
            fit_before = np.mean(self.fitness)

            if action == 0:
                self._learner_phase_standard()  # 代码B的原逻辑
            else:
                self._learner_phase_mutation()  # 新增逻辑

            fit_after = np.mean(self.fitness)

            # 5. 更新最优解
            self._update_global_best()

            # 6. 计算奖励并更新Q表
            # 如果产生了新的全局最优，给大奖励
            if self.best_fitness < self.last_best_fitness:
                reward = 10.0
            # 如果种群整体变好了，给小奖励
            elif fit_after < fit_before:
                reward = 1.0
            else:
                reward = -1.0  # 没效果，惩罚

            # Q-Learning 更新
            next_state = self._get_state()
            max_next_q = np.max(self.q_table[next_state])
            self.q_table[state, action] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[state, action])

            # 更新历史记录
            self.last_best_fitness = self.best_fitness
            self.iterations = iter_num

            if iter_num % 20 == 0 or iter_num == self.max_iter:
                print(f"迭代 {iter_num:4d} | 状态: {state} | 动作: {action_names[action]} | "
                      f"最优Loss: {self.best_fitness:.6e}")

        return self.best_solution, np.array(self.fitness_history)


# ========== 主程序示例 ==========
if __name__ == "__main__":
    # 1. 加载你的数据
    excel_path = r"D:\HuaweiMoveData\Users\35128\Desktop\graduate design\11.xls"

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

    # 3. 创建 Q-TLBO 优化器 (注意这里用的是 Q_TLBO)
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