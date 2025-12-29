import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- 1. 数据预处理（复用你的逻辑） ----------------------
def load_excel_and_preprocess(excel_path):
    df = pd.read_excel(excel_path, header=None, usecols=[0, 1], skiprows=1)
    V_original = df.iloc[:, 0].astype(float).values
    I_original = df.iloc[:, 1].astype(float).values
    valid_mask = np.isfinite(V_original) & np.isfinite(I_original)
    V_processed = V_original[valid_mask]
    I_meas_processed = I_original[valid_mask]
    I_min = np.min(I_meas_processed[I_meas_processed > 0]) if np.any(I_meas_processed > 0) else 1e-16
    I_max = np.max(I_meas_processed) if np.any(I_meas_processed > 0) else 1e-4
    return V_processed, I_meas_processed, I_min, I_max


# ---------------------- 2. 太阳能电池模型（复用你的逻辑） ----------------------
def solar_cell_model(V, params, I_min, I_max):
    I_ph, I0, n, Rs, Rsh = params
    Vt = 0.026  # 热电压，25℃时约0.026V
    I = np.zeros_like(V, dtype=np.float64)

    for i, v in enumerate(V):
        # 初始值：用I_ph近似I，加入Rs项
        exp_arg_init = (v + I_ph * Rs) / (n * Vt)
        exp_arg_init = np.clip(exp_arg_init, -20, 20)
        exp_term_init = np.exp(exp_arg_init) - 1
        shunt_term_init = (v + I_ph * Rs) / Rsh
        init_I = I_ph - I0 * exp_term_init - shunt_term_init
        init_I = np.clip(init_I, I_min * 0.1, I_max * 2)
        I_i = init_I

        # 迭代求解隐式方程
        for _ in range(100):
            exp_argument = (v + I_i * Rs) / (n * Vt)
            exp_argument = np.clip(exp_argument, -20, 20)
            exp_term = np.exp(exp_argument) - 1
            shunt_term = (v + I_i * Rs) / Rsh
            new_I_i = I_ph - I0 * exp_term - shunt_term  # 单二极管模型公式

            new_I_i = np.clip(new_I_i, I_min * 0.1, I_max * 2)
            if abs(new_I_i - I_i) < 1e-8:
                I_i = new_I_i
                break
            I_i = new_I_i

        I[i] = I_i if np.isfinite(I_i) else init_I
    return I


# ---------------------- 3. 目标函数（复用你的逻辑） ----------------------
def objective_function(params, V, I_meas, I_min, I_max):
    if len(V) == 0 or len(I_meas) == 0:
        return 1e10

    I_sim = solar_cell_model(V, params, I_min, I_max)
    if np.any(np.isnan(I_sim)) or np.any(np.isinf(I_sim)):
        return 1e10

    valid_meas_mask = (I_meas > 1e-10) & np.isfinite(I_meas)
    if not np.any(valid_meas_mask):
        return 1e10

    V_valid = V[valid_meas_mask]
    I_meas_valid = I_meas[valid_meas_mask]
    I_sim_valid = I_sim[valid_meas_mask]
    m = len(I_meas_valid)

    loss = 100 * np.sqrt((1 / m) * np.sum(((I_meas_valid - I_sim_valid) ** 2) / (I_meas_valid + 1e-8)))
    return loss if np.isfinite(loss) else 1e10


# ---------------------- 4. 传统TLBO算法（核心） ----------------------
class TraditionalTLBO:
    def __init__(self, V, I_meas, I_min, I_max, pop_size=100, max_iter=2000):
        self.V = V
        self.I_meas = I_meas
        self.I_min = I_min
        self.I_max = I_max
        self.pop_size = pop_size  # 种群规模
        self.max_iter = max_iter  # 最大迭代次数

        # 太阳能电池参数的物理范围（和你之前的一致）
        self.param_bounds = [
            (10, 20),  # I_ph（短路电流）
            (1e-12, 1e-7),  # I0（反向饱和电流）
            (1.0, 2.0),  # n（品质因子）
            (0.01, 1.0),  # Rs（串联电阻）
            (100, 1000)  # Rsh（并联电阻）
        ]

        # 初始化种群（随机生成符合范围的参数）
        self.population = self._init_population()

    def _init_population(self):
        """初始化种群：生成符合物理范围的随机参数"""
        population = []
        for _ in range(self.pop_size):
            param = [
                np.random.uniform(*self.param_bounds[0]),  # I_ph
                np.random.uniform(*self.param_bounds[1]),  # I0
                np.random.uniform(*self.param_bounds[2]),  # n
                np.random.uniform(*self.param_bounds[3]),  # Rs
                np.random.uniform(*self.param_bounds[4])  # Rsh
            ]
            population.append(param)
        return population

    def _evaluate_population(self):
        """评估种群：计算每个个体的损失（目标函数值）"""
        loss_list = []
        for params in self.population:
            loss = objective_function(params, self.V, self.I_meas, self.I_min, self.I_max)
            loss_list.append(loss)

        # 找到最优个体（损失最小）
        best_idx = np.argmin(loss_list)
        best_params = self.population[best_idx]
        best_loss = loss_list[best_idx]
        return best_params, best_loss, loss_list

    def _teaching_phase(self, best_params):
        """教师阶段：最优个体（教师）向种群传授知识"""
        new_population = []
        teacher = np.array(best_params)
        pop_mean = np.mean(self.population, axis=0)  # 种群均值

        for params in self.population:
            params_np = np.array(params)
            T_F = np.random.choice([1, 2])  # 教学因子（传统TLBO固定选1或2）
            r = np.random.random()  # 0~1的随机数

            # 教师阶段更新公式：X_new = X + r*(教师 - T_F*种群均值)
            new_params = params_np + r * (teacher - T_F * pop_mean)
            # 裁剪参数到物理范围
            new_params = self._clip_params(new_params)
            new_population.append(new_params.tolist())
        return new_population

    def _learning_phase(self):
        """学习者阶段：个体之间相互学习"""
        new_population = []
        for i in range(self.pop_size):
            # 随机选另一个不同的个体j
            j = np.random.choice([k for k in range(self.pop_size) if k != i])
            params_i = np.array(self.population[i])
            params_j = np.array(self.population[j])
            r = np.random.random()

            # 学习者阶段更新公式：若i的损失 > j的损失，则向j学习；否则j向i学习
            loss_i = objective_function(params_i.tolist(), self.V, self.I_meas, self.I_min, self.I_max)
            loss_j = objective_function(params_j.tolist(), self.V, self.I_meas, self.I_min, self.I_max)

            if loss_i > loss_j:
                new_params = params_i + r * (params_j - params_i)
            else:
                new_params = params_j + r * (params_i - params_j)

            # 裁剪参数到物理范围
            new_params = self._clip_params(new_params)
            new_population.append(new_params.tolist())
        return new_population

    def _clip_params(self, params):
        """裁剪参数到物理范围内"""
        clipped = []
        for param, (low, high) in zip(params, self.param_bounds):
            clipped.append(max(low, min(param, high)))
        return clipped

    def train(self):
        """TLBO训练主流程"""
        global_best_loss = 1e10
        global_best_params = None
        loss_history = []  # 记录损失变化（用于观察收敛）

        for iter in range(self.max_iter):
            # 1. 评估种群，找到当前最优
            best_params, best_loss, loss_list = self._evaluate_population()

            # 2. 更新全局最优
            if best_loss < global_best_loss:
                global_best_loss = best_loss
                global_best_params = best_params.copy()
            loss_history.append(global_best_loss)

            # 3. 教师阶段
            self.population = self._teaching_phase(best_params)
            # 4. 学习者阶段
            self.population = self._learning_phase()

            # 打印迭代信息
            if (iter + 1) % 100 == 0:
                print(f"迭代{iter + 1:4d} | 全局最优损失: {global_best_loss:.2e}")

            # 收敛条件：损失足够小
            if global_best_loss < 1e-2:
                print(f"提前收敛！迭代{iter + 1}次，全局最优损失={global_best_loss:.2e}")
                break

        # 打印最优参数合理性校验
        print("\n===== 最优参数合理性校验 =====")
        params_check = {
            "I_ph": (global_best_params[0], 10, 20),
            "I0": (global_best_params[1], 1e-12, 1e-7),
            "n": (global_best_params[2], 1.0, 2.0),
            "Rs": (global_best_params[3], 0.01, 1.0),
            "Rsh": (global_best_params[4], 100, 1000)
        }
        for name, (val, low, high) in params_check.items():
            status = "✅" if (low <= val <= high) else "❌"
            print(f"{name}: {val:.2e} {status} (范围：{low:.2e}~{high:.2e})")

        # 打印最终拟合指标
        print("\n===== 最终拟合效果 =====")
        final_I_sim = solar_cell_model(self.V, global_best_params, self.I_min, self.I_max)
        valid_mask = (self.I_meas > 1e-10) & np.isfinite(self.I_meas) & np.isfinite(final_I_sim)
        if np.any(valid_mask):
            I_meas_v = self.I_meas[valid_mask]
            I_sim_v = final_I_sim[valid_mask]
            mae = np.mean(np.abs(I_sim_v - I_meas_v))
            rmse = np.sqrt(np.mean((I_sim_v - I_meas_v) ** 2))
            ss_res = np.sum((I_sim_v - I_meas_v) ** 2)
            ss_tot = np.sum((I_meas_v - np.mean(I_meas_v)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            print(f"MAE: {mae:.4f} A | RMSE: {rmse:.4f} A | R²: {r2:.4f}")

        return global_best_params, global_best_loss, final_I_sim, loss_history


# ---------------------- 5. 主程序 ----------------------
if __name__ == "__main__":
    # 替换为你的Excel文件路径
    EXCEL_PATH = "C:\\Users\\18372\\PycharmProjects\\pythonProject1\\5.xls"
    V_processed, I_meas_processed, I_min, I_max = load_excel_and_preprocess(EXCEL_PATH)

    # 初始化并训练传统TLBO
    tlbo = TraditionalTLBO(
        V=V_processed,
        I_meas=I_meas_processed,
        I_min=I_min,
        I_max=I_max,
        pop_size=100,
        max_iter=2000
    )
    final_params, final_loss, final_I_pred, loss_history = tlbo.train()

    # 打印结果
    print("\n" + "=" * 70)
    print("拟合完成！")
    print(f"全局最优损失：{final_loss:.2e}")
    print(f"最优参数：")
    print(f"  - I_ph: {final_params[0]:.2f} A")
    print(f"  - I0: {final_params[1]:.2e} A")
    print(f"  - n: {final_params[2]:.2f}")
    print(f"  - Rs: {final_params[3]:.3f} Ω")
    print(f"  - Rsh: {final_params[4]:.0f} Ω")
    print("=" * 70)

    # 可视化（损失变化 + 拟合曲线）
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图1：损失变化曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, 'b-', linewidth=1.5)
    plt.xlabel('迭代次数')
    plt.ylabel('全局最优损失')
    plt.title('TLBO迭代损失变化')
    plt.grid(alpha=0.3)

    # 图2：拟合曲线
    plt.subplot(1, 2, 2)
    plt.plot(V_processed, I_meas_processed, 'o', label='原始实测电流', markersize=4)
    plt.plot(V_processed, final_I_pred, '-r', label='模型预测电流', linewidth=2)
    plt.xlabel('电压 (V)')
    plt.ylabel('电流 (A)')
    plt.title('太阳能电池模型拟合结果')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()