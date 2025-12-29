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
    Vt = 0.026  # 导师给的参数
    I = np.zeros_like(V, dtype=np.float64)   #初始化全0的np数组，用于储存后面算出来的电流值，一个电压对应一个电流，所以这里的数组是电压数组的形状
    for i, v in enumerate(V):     #遍历电压的索引和元素值   这两个都可以取出来
        # ========== 核心修改：用牛顿迭代法计算精准初始值 ==========
        # 步骤1：定义牛顿迭代所需的函数f(I)和导数f'(I)
        def f(I):
            """光伏模型的残差函数 f(I)=0"""
            exp_arg = (v + I * Rs) / (n * Vt)
            exp_arg = np.clip(exp_arg, -20, 20)  # 防止指数爆炸
            exp_term = np.exp(exp_arg) - 1
            shunt_term = (v + I * Rs) / Rsh
            # 单二极管模型：I = I_ph - I0*exp_term - shunt_term → f(I)=I - (I_ph - I0*exp_term - shunt_term)
            return I - (I_ph - I0 * exp_term - shunt_term)

        def f_prime(I):
            """f(I)对I的导数（牛顿迭代核心）"""
            exp_arg = (v + I * Rs) / (n * Vt)
            exp_arg = np.clip(exp_arg, -20, 20)
            exp_term = np.exp(exp_arg)
            # 导数推导：f'(I) = 1 + I0*Rs/(n*Vt)*exp_term + Rs/Rsh
            return 1 + (I0 * Rs / (n * Vt)) * exp_term + (Rs / Rsh)

        # 步骤2：牛顿迭代法求解初始值（仅迭代3-5次，足够收敛）
        init_I = I_ph  # 牛顿迭代的初始猜测值（仅起点，非近似）
        # 牛顿迭代（少量迭代即可得到精准初始值）
        for _ in range(5):
            if not np.isfinite(init_I) or init_I <= 0:
                break  # 异常值直接退出，避免迭代发散
            f_val = f(init_I)
            f_p_val = f_prime(init_I)

            # 防止导数为0导致数值爆炸
            if abs(f_p_val) < 1e-12:
                break

            # 牛顿迭代核心公式：I_new = I - f(I)/f'(I)
            new_init_I = init_I - f_val / f_p_val
            # 物理约束：初始值需在合理范围（避免无效值）
            new_init_I = np.clip(new_init_I, I_min * 0.1, I_max * 2)

            # 收敛判断：提前终止迭代
            if abs(new_init_I - init_I) < 1e-8:
                init_I = new_init_I
                break
            init_I = new_init_I
        # ========== 牛顿迭代法初始值计算结束 ==========

        # 原有迭代求解逻辑（保留，进一步优化精度）
        I_i = init_I  # 初始值替换为牛顿迭代的结果
        for _ in range(100):
            exp_argument = (v + I_i * Rs) / (n * Vt)
            exp_argument = np.clip(exp_argument, -20, 20)
            exp_term = np.exp(exp_argument) - 1
            shunt_term = (v + I_i * Rs) / Rsh
            new_I_i = I_ph - I0 * exp_term - shunt_term  # 正确的单二极管模型公式

            # 裁剪避免数值异常
            new_I_i = np.clip(new_I_i, I_min * 0.1, I_max * 2)

            # 收敛判断
            if abs(new_I_i - I_i) < 1e-8:
                I_i = new_I_i
                break
            I_i = new_I_i

        # 核心修复：优先使用迭代收敛后的I_i，无效时才用初始值
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
    def __init__(self,V: np.ndarray,I_meas: np.ndarray,I_min: float,I_max: float,pop_size: int = 30, max_iter: int = 100,param_bounds: Optional[np.ndarray] = None):
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
        # 设置参数边界（使用默认值或传入值）
        if param_bounds is not None:
            assert param_bounds.shape == (5, 2), "param_bounds形状必须为(5, 2)"
            self.param_bounds = param_bounds.astype(np.float64)
        else:
            # 合理的默认物理边界（基于典型太阳能电池参数）
            self.param_bounds = np.array([
                [0.1, 8.0],  # I_ph: 光生电流 (A)，覆盖常见范围
                [1e-60, 1e-6],  # I0: 反向饱和电流 (A)，关键参数！
                [1.0, 2.0],  # n: 理想因子，物理约束1-2
                [0.01, 0.5],  # Rs: 串联电阻 (Ω)，通常较小
                [1000, 1e6]  # Rsh: 并联电阻 (Ω)，通常较大
            ], dtype=np.float64)
        # 运行状态变量（会在优化过程中初始化）
        self.population = None  # 当前种群，形状(pop_size, param_dim)
        self.fitness = None  # 当前适应度，形状(pop_size,)
        self.best_solution = None  # 历史最优解
        self.best_fitness = float('inf')  # 历史最优适应度
        self.fitness_history = []  # 每代最优适应度记录
        self.iterations = 0  # 已完成的迭代次数

    def _initialize_population(self) -> None:
        """在参数边界内随机初始化种群。"""
        self.population = np.zeros((self.pop_size, self.param_dim))     #pop_size就相当于有多少个学生，每个学生身上背着param_dim这么多个参数信息

        for d in range(self.param_dim):    #param_dim是参数维度，这里每组参数有五个
            low, high = self.param_bounds[d]    #跟着for依次取出每个参数的最小最大值信息
            # 为每个参数维度生成均匀分布的随机值
            self.population[:, d] = np.random.uniform(low, high, self.pop_size)    #按列生成数据，种群初始化

        # 计算初始适应度
        self.fitness = np.array([self._evaluate_fitness(ind)      #每个学生都有吧自己的适应度，也就是好坏，适应度越小越好(loss)
                                 for ind in self.population])

        # 初始化全局最优记录
        best_idx = np.argmin(self.fitness)        #找到最好的学生，通过找到最小适应度的索引，不然没办法找到最好的学生
        self.best_solution = self.population[best_idx].copy()    #通过最好的适应度索引找到最好的学生，深拷贝，把最好的那组参数存在best_solution中
        self.best_fitness = self.fitness[best_idx]          #存下最好的适应度
        self.fitness_history.append(self.best_fitness)    #把适应度存在数组中

    def _evaluate_fitness(self, params: np.ndarray) -> float:
        """
        评估单个解（一组参数）的适应度。

        参数:
        ----------
        params : np.ndarray
            形状为(5,)的参数数组：[I_ph, I0, n, Rs, Rsh]

        返回:
        ----------
        float
            适应度值（目标函数值），越小越好。
        """
        # 调用你之前定义好的目标函数
        # 确保你的objective_function能接受这些参数
        return objective_function(params, self.V, self.I_meas,
                                  self.I_min, self.I_max)

    def _apply_bounds(self, individual: np.ndarray) -> np.ndarray:
        """确保个体参数不超出预设边界。"""
        # 使用np.clip进行向量化裁剪，效率更高
        for d in range(self.param_dim):
            low, high = self.param_bounds[d]
            individual[d] = np.clip(individual[d], low, high)
        return individual

    def _teacher_phase(self) -> None:
        """教师阶段：所有学生向当前最优个体（教师）学习。"""
        # 1. 确定当前教师（适应度最好的个体）
        teacher_idx = np.argmin(self.fitness)
        teacher = self.population[teacher_idx]

        # 2. 计算种群在当前参数空间中的均值
        mean_population = np.mean(self.population, axis=0)  #axis=0 沿垂直方向计算   计算班级平均分  找到班级基准线

        # 3. 教学因子：随机选择1或2
        TF = np.random.randint(1, 3)  # 在[1, 2]中随机  相当于学习步长

        # 4. 每个学生进行学习更新
        for i in range(self.pop_size):
            # 生成随机向量r（与参数维度相同）
            r = np.random.rand(self.param_dim)  #随机学习权重，就是向老师学多少

            # TLBO核心更新公式
            new_individual = (self.population[i] +r * (teacher - TF * mean_population))

            # 边界处理
            new_individual = self._apply_bounds(new_individual)

            # 评估新解
            new_fitness = self._evaluate_fitness(new_individual)

            # 贪婪选择：只接受改进的解
            if new_fitness < self.fitness[i]:
                self.population[i] = new_individual
                self.fitness[i] = new_fitness

    def _learner_phase(self) -> None:
        """学习者阶段：学生之间随机配对互相学习。"""
        # 创建随机排列，确保每个学生都能参与
        indices = np.random.permutation(self.pop_size)

        # 两两配对（如果种群大小为奇数，最后一个保持不变）
        for k in range(0, self.pop_size - 1, 2):
            i, j = indices[k], indices[k + 1]

            # 确定学习方向：适应度好的作为老师  越小越好
            if self.fitness[i] < self.fitness[j]:
                teacher, learner = self.population[i], self.population[j]
                teacher_fit, learner_fit = self.fitness[i], self.fitness[j]
                update_idx = j
            else:
                teacher, learner = self.population[j], self.population[i]
                teacher_fit, learner_fit = self.fitness[j], self.fitness[i]
                update_idx = i

            # 生成随机向量
            r = np.random.rand(self.param_dim)

            # 学习者向老师学习，r是步长因子
            new_individual = learner + r * (teacher - learner)

            # 边界处理
            new_individual = self._apply_bounds(new_individual)

            # 评估新解
            new_fitness = self._evaluate_fitness(new_individual)

            # 贪婪选择，向老师学习后的个体和原来的个体进行比较，使用更好的作为新种群
            if new_fitness < learner_fit:
                self.population[update_idx] = new_individual
                self.fitness[update_idx] = new_fitness

    def _update_global_best(self) -> None:
        """更新全局历史最优解。"""
        current_best_idx = np.argmin(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]

        # 如果找到更好的解，则更新
        if current_best_fitness < self.best_fitness:
            self.best_solution = self.population[current_best_idx].copy()
            self.best_fitness = current_best_fitness

        # 记录本次迭代后的最优适应度（用于绘制收敛曲线）
        self.fitness_history.append(self.best_fitness)

    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行完整的TLBO优化流程。

        返回:
        ----------
        best_solution : np.ndarray
            找到的最优参数向量 [I_ph, I0, n, Rs, Rsh]
        fitness_history : np.ndarray
            每次迭代后的最优适应度记录，长度为max_iter+1
        """
        print("=" * 50)
        print("TLBO优化器启动")
        print(f"问题维度: {self.param_dim} | 种群大小: {self.pop_size} | 最大迭代: {self.max_iter}")
        print(f"参数边界:")
        param_names = ['I_ph', 'I0', 'n', 'Rs', 'Rsh']
        for name, (low, high) in zip(param_names, self.param_bounds):
            print(f"  {name:4s}: [{low:8.2e}, {high:8.2e}]")
        print("=" * 50)

        # 步骤1: 初始化种群
        self._initialize_population()
        print(f"迭代 0/{self.max_iter} | 初始最优适应度: {self.best_fitness:.6e}")

        # 步骤2: 主优化循环
        for iter_num in range(1, self.max_iter + 1):
            # 教师阶段
            self._teacher_phase()

            # 学习者阶段
            self._learner_phase()

            # 更新全局最优记录
            self._update_global_best()

            # 记录迭代次数
            self.iterations = iter_num

            # 打印进度信息
            if iter_num % 20 == 0 or iter_num == self.max_iter:
                # 计算当前种群的适应度统计信息
                avg_fitness = np.mean(self.fitness)
                std_fitness = np.std(self.fitness)

                print(f"迭代 {iter_num:4d}/{self.max_iter} | "
                      f"最优适应度: {self.best_fitness:.6e} | "
                      f"平均适应度: {avg_fitness:.6e} ± {std_fitness:.6e}")

        print("=" * 50)
        print("优化完成！")
        print(f"总迭代次数: {self.iterations}")
        print(f"最终最优适应度: {self.best_fitness:.6e}")
        print("最优参数:")
        for name, value in zip(param_names, self.best_solution):
            print(f"  {name:4s}: {value:.6e}")
        print("=" * 50)

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
    excel_path = r"C:\Users\18372\PycharmProjects\pythonProject1\1.xls"  # 替换为你的文件路径
    V_processed, I_meas_processed, _, _, I_min, I_max = load_excel_and_preprocess(excel_path)

    # 2. (可选) 自定义参数边界 - 如果不指定，将使用类中的默认值
    custom_bounds = np.array([
        [0.1, 8.0],  # I_ph: 光生电流 (A)，覆盖常见范围
        [1e-60, 1e-6],  # I0: 反向饱和电流 (A)，关键参数！
        [1.0, 2.0],  # n: 理想因子，物理约束1-2
        [0.01, 0.5],  # Rs: 串联电阻 (Ω)，通常较小
        [1000, 1e6]  # Rsh: 并联电阻 (Ω)，通常较大
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
