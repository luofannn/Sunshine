#数据预处理
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_excel_and_preprocess(excel_path):
    df = pd.read_excel(excel_path, header=None, usecols=[0, 1], skiprows=1)  #读表
    V_original = df.iloc[:, 0].astype(float).values     #第1列数据为电压，第2列为电流
    I_original = df.iloc[:, 1].astype(float).values
    valid_mask = np.isfinite(V_original) & np.isfinite(I_original)    #数据清洗，最后得到一个bool数组，确认数据有效的位置
    V_processed = V_original[valid_mask]     #得到处理之后的有效的电压，电流
    I_meas_processed = I_original[valid_mask]
    I_min = np.min(I_meas_processed[I_meas_processed > 0]) if np.any(I_meas_processed > 0) else 1e-16    #找到大于0的电流的最小值和最大值
    I_max = np.max(I_meas_processed) if np.any(I_meas_processed > 0) else 1e-4
    return V_processed, I_meas_processed, V_original, I_original, I_min, I_max     #返回处理后的电压电流，原始电压电流，最小电流，最大电流

#定义太阳能电池模型，导师给的用来计算电流的模型，算法找到的参数（I_ph, I0, n, Rs, Rsh ）输入这个模型得到对应的输出电流，一个电压对应一个推出来的电流，所以电压也要作为输入参数
def solar_cell_model(V, params, I_min, I_max):
    I_ph, I0, n, Rs, Rsh = params
    Vt = 0.026
    I = np.zeros_like(V, dtype=np.float64)

    # 统一截断范围，针对3V高压必须给到150
    CLIP_MIN, CLIP_MAX = -50, 150

    for i, v in enumerate(V):
        # --- 步骤1：定义函数 ---
        def f(I):
            exp_arg = (v + I * Rs) / (n * Vt)
            # [修正1] 这里的 20 必须改成 150，否则二极管效应出不来
            exp_arg = np.clip(exp_arg, CLIP_MIN, CLIP_MAX)
            exp_term = np.exp(exp_arg) - 1
            shunt_term = (v + I * Rs) / Rsh
            return I - (I_ph - I0 * exp_term - shunt_term)

        def f_prime(I):
            exp_arg = (v + I * Rs) / (n * Vt)
            # [修正2] 这里你已经改对了，保持 150
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
            # [修正3] 这里的 20 也必须改成 150！这是造成直线的罪魁祸首
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

#定义目标函数，输入参数（I_ph, I0, n, Rs, Rsh ）和电压还有原始电流计算误差，这个电压是从现成的数据集中导入的，I_meas是实际电流
def objective_function(params, V, I_meas, I_min, I_max):
    if len(V) == 0 or len(I_meas) == 0:
        return 1e10
    I_sim = solar_cell_model(V, params, I_min, I_max)    #计算模拟电流
    # 过滤异常值
    if np.any(np.isnan(I_sim)) or np.any(np.isinf(I_sim)):   #过滤不合理的模拟电流
        return 1e10
    valid_meas_mask = (I_meas > 1e-10) & np.isfinite(I_meas)   #过滤不合理的实际电流，返回一个bool数组
    if not np.any(valid_meas_mask):
        return 1e10
    #V_valid = V[valid_meas_mask]                              #过滤后的有效实际电流对应的电压
    I_meas_valid = I_meas[valid_meas_mask]                    #有效的实际电流
    I_sim_valid = I_sim[valid_meas_mask]                      #有效的计算电流
    numerator = I_meas_valid - I_sim_valid
    relative_error = numerator /np.max(I_meas_valid)  # 除以测量电流最大值
    loss = np.sqrt((1 / len(I_meas_valid)) * np.sum(relative_error ** 2))

    # 4. 最后校验损失是否有效
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    return loss

#TLBO算法实现
# ========== TLBO 算法超参数 ==========
# POP_SIZE = 30          # 种群大小（学生数量）
# MAX_ITER = 100         # 最大迭代次数
# PARAM_DIM = 5          # 要优化的参数维度：[I_ph, I0, n, Rs, Rsh]
#
# # 参数搜索范围（请务必根据你的物理模型调整！）
# PARAM_BOUNDS = np.array([
#     [0.1, 8],  # I_ph: 参考值4.2在此范围内，但上下限宽松
#     [1e-60, 1e-20],  # I0: **关键！** 鉴于参考值极小且不确定，必须用极宽的对数范围覆盖可能区间
#     [1.0, 2.0],  # n: 保持标准物理范围
#     [0.01, 0.5],  # Rs: 适当放宽，包含参考值0.067
#     [20, 5000]  # Rsh: 放宽范围，既包含参考值50，也包含您原设的较高值
# ])
class TLBO:
    def __init__(self, V: np.ndarray, I_meas: np.ndarray, I_min: float, I_max: float, pop_size: int = 30,
                 max_iter: int = 100, param_bounds: Optional[np.ndarray] = None):
        # ... (保持原本的初始化代码不变) ...
        assert len(V) == len(I_meas), "电压和电流数据长度必须一致"
        self.V = V
        self.I_meas = I_meas
        self.I_min = I_min
        self.I_max = I_max
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.param_dim = 5

        if param_bounds is not None:
            self.param_bounds = param_bounds.astype(np.float64)
        else:
            self.param_bounds = np.array([
                [0.1, 10.0], [1e-60, 1e-50], [1.0, 1.3], [0.001, 0.5], [50, 150]
            ], dtype=np.float64)

        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.iterations = 0

    def _initialize_population(self) -> None:
        """在参数边界内随机初始化种群 (引入A的对数分布策略)。"""
        self.population = np.zeros((self.pop_size, self.param_dim))

        for d in range(self.param_dim):
            low, high = self.param_bounds[d]
            # [关键修改]：借鉴A代码，对 I0 (索引1) 使用对数分布初始化
            # 因为 I0 在 1e-60 到 1e-50 之间，线性分布无法覆盖极小值
            if d == 1:
                log_low, log_high = np.log10(low), np.log10(high)
                self.population[:, d] = 10 ** np.random.uniform(log_low, log_high, self.pop_size)
            else:
                self.population[:, d] = np.random.uniform(low, high, self.pop_size)

        # 计算初始适应度
        self.fitness = np.array([self._evaluate_fitness(ind) for ind in self.population])

        # 初始化全局最优
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.fitness_history.append(self.best_fitness)

    def _evaluate_fitness(self, params: np.ndarray) -> float:
        return objective_function(params, self.V, self.I_meas, self.I_min, self.I_max)

    def _apply_bounds(self, individual: np.ndarray) -> np.ndarray:
        for d in range(self.param_dim):
            individual[d] = np.clip(individual[d], self.param_bounds[d, 0], self.param_bounds[d, 1])
        return individual

    def _teacher_phase(self) -> None:
        """教师阶段：向最优个体学习 (保持B的逻辑，逻辑本身是通用的)。"""
        teacher_idx = np.argmin(self.fitness)
        teacher = self.population[teacher_idx]
        mean_population = np.mean(self.population, axis=0)

        for i in range(self.pop_size):
            TF = np.random.randint(1, 3)
            r = np.random.rand(self.param_dim)

            # 标准TLBO公式
            new_individual = self.population[i] + r * (teacher - TF * mean_population)
            new_individual = self._apply_bounds(new_individual)
            new_fitness = self._evaluate_fitness(new_individual)

            if new_fitness < self.fitness[i]:
                self.population[i] = new_individual
                self.fitness[i] = new_fitness

    def _learner_phase(self) -> None:
        """学习者阶段：(关键修改) 引入A的混合策略 (TLBO + 差分进化)。"""
        for i in range(self.pop_size):
            # [关键修改]：不再固定两两配对，而是随机选择伙伴 (增加随机性)
            idxs = [x for x in range(self.pop_size) if x != i]
            partner_idx = np.random.choice(idxs)
            partner = self.population[partner_idx]

            # 策略选择：80%概率用标准TLBO，20%概率用差分变异(Mutation)
            if np.random.rand() < 0.8:
                # === 标准 TLBO 逻辑 ===
                r = np.random.rand(self.param_dim)
                if self.fitness[i] < self.fitness[partner_idx]:
                    step = self.population[i] - partner
                else:
                    step = partner - self.population[i]
                new_individual = self.population[i] + r * step
            else:
                # === 借鉴A代码：差分进化变异逻辑 (Mutation) ===
                # 利用全局最优 + 两个随机个体的差值，强行跳出局部最优
                a_idx, b_idx = np.random.choice(idxs, 2, replace=False)
                a, b = self.population[a_idx], self.population[b_idx]
                F = 0.5  # 缩放因子
                # new = Best + F * (Random_A - Random_B)
                new_individual = self.best_solution + F * (a - b)

            new_individual = self._apply_bounds(new_individual)
            new_fitness = self._evaluate_fitness(new_individual)

            if new_fitness < self.fitness[i]:
                self.population[i] = new_individual
                self.fitness[i] = new_fitness

    def _update_global_best(self) -> None:
        """更新全局历史最优解。"""
        current_best_idx = np.argmin(self.fitness)
        if self.fitness[current_best_idx] < self.best_fitness:
            self.best_solution = self.population[current_best_idx].copy()
            self.best_fitness = self.fitness[current_best_idx]
        self.fitness_history.append(self.best_fitness)

    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        # ... (保持原本的打印和循环逻辑不变) ...
        print("TLBO优化器启动 (增强版模式)")
        self._initialize_population()

        for iter_num in range(1, self.max_iter + 1):
            self._teacher_phase()
            self._learner_phase()  # 这里现在调用的是增强版逻辑
            self._update_global_best()

            if iter_num % 20 == 0 or iter_num == self.max_iter:
                print(f"迭代 {iter_num:4d}/{self.max_iter} | 最优Loss: {self.best_fitness:.6e}")

        return self.best_solution, np.array(self.fitness_history)


    def get_optimization_summary(self) -> dict:
        """获取优化过程的摘要信息。"""
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'fitness_history': np.array(self.fitness_history),
            'iterations': self.iterations,
            'param_bounds': self.param_bounds,
            'pop_size': self.pop_size,
            'max_iter': self.max_iter
        }





# ========== 主程序示例 ==========
if __name__ == "__main__":
    # 1. 加载你的数据（使用你现有的函数）
    excel_path = r"D:\HuaweiMoveData\Users\35128\Desktop\graduate design\11.xls"  # 替换为你的文件路径
    V_processed, I_meas_processed, _, _, I_min, I_max = load_excel_and_preprocess(excel_path)

    # 2. (可选) 自定义参数边界 - 如果不指定，将使用类中的默认值
    custom_bounds = np.array([
        [0.1, 10.0],  # I_ph
        [1e-60, 1e-50],  # I0 - 放宽范围
        [1.0, 1.3],  # n - 放宽范围
        [0.001, 0.5],  # Rs
        [50, 150]  # Rsh
    ])

    # 3. 创建TLBO优化器实例
    optimizer = TLBO(
        V=V_processed,
        I_meas=I_meas_processed,
        I_min=I_min,
        I_max=I_max,
        pop_size=40,  # 可以尝试30-50
        max_iter=150,  # 可以尝试100-200
        param_bounds=custom_bounds  # 使用自定义边界，或设为None用默认值
    )

    # 4. 执行优化
    best_params, history = optimizer.optimize()

    # 5. 使用最优参数进行拟合验证
    I_fitted = solar_cell_model(V_processed, best_params, I_min, I_max)

    # 6. 可视化结果（以下代码需要matplotlib）
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 子图1: 收敛曲线
    axes[0].plot(history, 'b-', linewidth=2)
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('最优适应度')
    axes[0].set_title('TLBO收敛曲线')
    axes[0].grid(True, alpha=0.3)
    axes[0].semilogy()  # 使用对数坐标更易观察收敛

    # 子图2: I-V曲线拟合效果
    axes[1].scatter(V_processed, I_meas_processed, s=10, alpha=0.6,
                    label='实测数据', color='blue')
    axes[1].plot(V_processed, I_fitted, 'r-', linewidth=2,
                 label='TLBO拟合')
    axes[1].set_xlabel('电压 (V)')
    axes[1].set_ylabel('电流 (I)')
    axes[1].legend()
    axes[1].set_title('太阳能电池I-V曲线拟合效果')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 7. 打印拟合误差
    mse = np.mean((I_meas_processed - I_fitted) ** 2)
    rmse = np.sqrt(mse)
    print(f"\n拟合误差分析:")
    print(f"均方误差 (MSE): {mse:.4e}")
    print(f"均方根误差 (RMSE): {rmse:.4e}")
    print(f"相对误差: {rmse / np.mean(np.abs(I_meas_processed)) * 100:.2f}%")