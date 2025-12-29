import numpy as np
import random
import pandas as pd


# ---------------------- 1. 读取Excel+数据预处理（保留电压拐点数据） ----------------------
def load_excel_and_preprocess(excel_path):
    df = pd.read_excel(excel_path, header=1)
    x_original = df['VG'].values  # 原始电压
    I_original = df[['S1', 'S2', 'S3']].mean(axis=1).values  # S1/S2/S3均值

    # 导师要求：以x=10为对称轴镜像 + 平移原点到原x=10
    x_mirror = 20 - x_original  # 镜像
    VG_processed = x_mirror - 10  # 平移原点
    I_meas_processed = I_original  # 电流保持不变

    # 【优化1】仅过滤极端异常点（不再过滤0.1V附近，保留拐点数据）
    valid_mask = np.abs(VG_processed) < 20  # 只过滤电压绝对值>20的极端点
    VG_processed = VG_processed[valid_mask]
    I_meas_processed = I_meas_processed[valid_mask]

    # 返回实测电流的极值（用于模型裁剪）
    I_min = np.min(I_meas_processed[I_meas_processed > 0]) if np.any(I_meas_processed > 0) else 1e-16
    I_max = np.max(I_meas_processed) if np.any(I_meas_processed > 0) else 1e-4
    return VG_processed, I_meas_processed, x_original, I_original, I_min, I_max


# ---------------------- 2. 肖特基模型（优化电流裁剪+增加迭代次数） ----------------------
def schottky_model_V1(V, params, I_min, I_max):
    """导师要求的文献公式1：J = J0*(exp((V-Rs*J)/(n*Vt)) - 1) + (V-Rs*J)/Rp"""
    J0, n, Rs, Rp = params
    Vt = 0.026  # 常温热电压（26mV）
    J = np.zeros_like(V, dtype=np.float64)
    for i, v in enumerate(V):
        # 【优化2】增加迭代次数到10次，让模型计算更收敛
        J_i = J0  # 初始猜测
        for _ in range(50):
            # 限制指数部分的范围（避免exp溢出）
            exp_argument = (v - Rs * J_i) / (n * Vt)
            exp_argument = np.clip(exp_argument, -30, 30)  # 【优化3】放宽指数范围
            # 计算模型值
            J_i = J0 * (np.exp(exp_argument) - 1) + (v - Rs * J_i) / Rp
            # 【优化4】基于实测电流范围裁剪（不再用固定值）
            J_i = np.clip(J_i, I_min * 0.1, I_max * 10)
        J[i] = J_i
    return J


# ---------------------- 3. 导师指定的目标函数（适配新参数） ----------------------
def objective_function(params, V, I_meas, I_min, I_max):
    """导师指定的目标函数：归一化RMSE"""
    I_sim = schottky_model_V1(V, params, I_min, I_max)
    I_meas_M = np.max(np.abs(I_meas))  # 实测最大值
    m = len(I_meas)
    if I_meas_M < 1e-20:
        I_meas_M = 1e-20
    loss = np.sqrt((1 / m) * np.sum(((I_meas - I_sim) / I_meas_M) ** 2))
    return loss


# ---------------------- 4. RL控制器（保持不变） ----------------------
class RLController:
    def __init__(self, action_num=3, lr=0.1, gamma=0.9, epsilon=0.3):
        self.action_num = action_num
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q_table = {}

    def get_q(self, state):
        state = round(state, 10)
        if state not in self.Q_table:
            self.Q_table[state] = [0.0] * self.action_num
        return self.Q_table[state]

    def choose_action(self, state):
        q_values = self.get_q(state)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_num)
        else:
            return np.argmax(q_values)

    def update_q(self, state, action, reward, next_state):
        current_q = self.get_q(state)[action]
        next_max_q = max(self.get_q(next_state)) if round(next_state, 10) in self.Q_table else 0.0
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.Q_table[round(state, 10)][action] = new_q


# ---------------------- 5. RLRTLBO算法（优化参数范围） ----------------------
class RLRTLBO:
    def __init__(self, V, I_meas, I_min, I_max, rl_controller, pop_size=50, max_iter=1000):
        self.V = V
        self.I_meas = I_meas
        self.I_min = I_min
        self.I_max = I_max
        self.rl = rl_controller
        self.pop_size = pop_size  # 【优化5】增加种群规模
        self.max_iter = max_iter  # 【优化6】增加迭代次数
        # 【优化7】放宽参数范围，适配实测数据特性
        self.param_bounds = [
            (1e-16, 1e-7),  # J0（饱和电流，适配更小值）
            (1.0, 5.0),      # n（理想因子，放宽范围）
            (10, 500),       # Rs（串联电阻，大幅放宽）
            (1e4, 1e8)      # Rp（并联电阻，放宽范围）
        ]
        self.population = self._init_population()

    def _init_population(self):
        population = []
        for _ in range(self.pop_size):
            J0 = np.random.uniform(*self.param_bounds[0])
            n = np.random.uniform(*self.param_bounds[1])
            Rs = np.random.uniform(*self.param_bounds[2])
            Rp = np.random.uniform(*self.param_bounds[3])
            population.append([J0, n, Rs, Rp])
        return population

    def _evaluate_population(self):
        loss_list = []
        for params in self.population:
            # 传入实测电流极值
            loss = objective_function(params, self.V, self.I_meas, self.I_min, self.I_max)
            loss_list.append(loss)
        best_idx = np.argmin(loss_list)
        best_params = self.population[best_idx]
        best_loss = loss_list[best_idx]
        avg_loss = np.mean(loss_list)
        return best_params, best_loss, avg_loss

    def _teaching_phase(self, best_params, explore_ratio):
        new_population = []
        teacher = np.array(best_params)
        mean_params = np.mean(self.population, axis=0)
        for params in self.population:
            params_np = np.array(params)
            TF = round(1 + np.random.random())
            if np.random.uniform(0, 1) < explore_ratio:
                # 随机探索（缩小步长）
                new_params = [
                    params[0] * np.random.uniform(0.8, 1.2),  # 【优化8】放宽探索步长
                    params[1] * np.random.uniform(0.95, 1.05),
                    params[2] * np.random.uniform(0.8, 1.2),
                    params[3] * np.random.uniform(0.8, 1.2)
                ]
            else:
                # 向老师学习（调整学习步长）
                new_params = (params_np + np.random.random(4) * 0.3 * (teacher - TF * mean_params)).tolist()
            new_params = self._clip_params(new_params)
            new_population.append(new_params)
        return new_population

    def _learning_phase(self):
        new_population = []
        for i in range(self.pop_size):
            j = random.sample([k for k in range(self.pop_size) if k != i], 1)[0]
            params_i = np.array(self.population[i])
            params_j = np.array(self.population[j])
            # 传入实测电流极值
            loss_i = objective_function(params_i.tolist(), self.V, self.I_meas, self.I_min, self.I_max)
            loss_j = objective_function(params_j.tolist(), self.V, self.I_meas, self.I_min, self.I_max)
            # 向更优的个体学习（调整学习步长）
            if loss_i > loss_j:
                new_params = (params_i + np.random.random(4) * 0.3 * (params_j - params_i)).tolist()
            else:
                new_params = (params_j + np.random.random(4) * 0.3 * (params_i - params_j)).tolist()
            new_params = self._clip_params(new_params)
            new_population.append(new_params)
        return new_population

    def _clip_params(self, params):
        clipped = []
        for param, (low, high) in zip(params, self.param_bounds):
            clipped.append(max(low, min(param, high)))
        return clipped

    def train(self):
        for iter in range(self.max_iter):
            best_params, best_loss, avg_loss = self._evaluate_population()
            current_state = best_loss
            action = self.rl.choose_action(current_state)
            explore_ratio = [0.3, 0.5, 0.7][action]  # 【优化9】提高探索比例
            # 教学+学习阶段
            self.population = self._teaching_phase(best_params, explore_ratio)
            self.population = self._learning_phase()
            # 评估新种群
            new_best_params, new_best_loss, new_avg_loss = self._evaluate_population()
            # 奖励（与损失负相关）
            reward = 1 / (new_best_loss + 1e-10)
            next_state = new_best_loss
            self.rl.update_q(current_state, action, reward, next_state)
            # 打印进度
            if (iter + 1) % 100 == 0:
                print(
                    f"迭代{iter + 1:3d} | 最优损失: {best_loss:.2e} | 最优参数: J0={best_params[0]:.2e}, n={best_params[1]:.2f}, Rs={best_params[2]:.0f}, Rp={best_params[3]:.1e}")
            # 提前收敛（放宽收敛条件）
            if new_best_loss < 1e-3:
                print(f"提前收敛！迭代{iter + 1}次，损失={new_best_loss:.2e}")
                break
        # 最终结果
        final_best_params, final_best_loss, _ = self._evaluate_population()
        final_I_pred = schottky_model_V1(self.V, final_best_params, self.I_min, self.I_max)
        return final_best_params, final_best_loss, final_I_pred


# ---------------------- 6. 运行+可视化 ----------------------
if __name__ == "__main__":
    EXCEL_PATH = "C:\\Users\\18372\\PycharmProjects\\pythonProject1\\tfti.xls"  # 替换为你的路径

    # 1. 数据预处理（获取电流极值）
    VG_processed, I_meas_processed, _, _, I_min, I_max = load_excel_and_preprocess(EXCEL_PATH)
    # 2. 初始化RL+算法（传入电流极值）
    rl_controller = RLController()
    rlrtlbo = RLRTLBO(VG_processed, I_meas_processed, I_min, I_max, rl_controller)
    # 3. 训练拟合
    final_params, final_loss, final_I_pred = rlrtlbo.train()
    # 4. 输出结果
    print("\n" + "=" * 70)
    print("拟合完成！")
    print(f"目标函数最小值：{final_loss:.2e}")
    print(f"最优参数（导师要求的4个参数）：")
    print(f"  - 饱和电流J0: {final_params[0]:.2e} A")
    print(f"  - 理想因子n: {final_params[1]:.2f}（导师要求≈1）")
    print(f"  - 串联电阻Rs: {final_params[2]:.0f} Ω")
    print(f"  - 并联电阻Rp: {final_params[3]:.1e} Ω")
    print("=" * 70)

    # 5. 可视化
    try:
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        plt.figure(figsize=(12, 7))
        # 绘制实测数据和预测曲线（对数坐标）
        plt.semilogy(VG_processed, I_meas_processed, 'o', label='处理后实测电流（S1/S2/S3均值）', markersize=4)
        plt.semilogy(VG_processed, final_I_pred, '-r', label='肖特基模型预测电流（文献公式1）', linewidth=2)
        plt.xlabel('处理后电压 (V)', fontsize=12)
        plt.ylabel('电流 (A)', fontsize=12)
        plt.title('TFT数据拟合结果（RLRTLBO+经典肖特基模型）', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("未安装matplotlib，跳过可视化")