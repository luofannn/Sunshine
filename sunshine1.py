import numpy as np
import random
import pandas as pd


# ---------------------- 1. 数据预处理（完全不变） ----------------------
def load_excel_and_preprocess(excel_path):
    df = pd.read_excel(excel_path, header=None, usecols=[0, 1], skiprows=1)
    V_original = df.iloc[:, 0].astype(float).values
    I_original = df.iloc[:, 1].astype(float).values
    valid_mask = np.isfinite(V_original) & np.isfinite(I_original)
    V_processed = V_original[valid_mask]
    I_meas_processed = I_original[valid_mask]
    I_min = np.min(I_meas_processed[I_meas_processed > 0]) if np.any(I_meas_processed > 0) else 1e-16
    I_max = np.max(I_meas_processed) if np.any(I_meas_processed > 0) else 1e-4
    return V_processed, I_meas_processed, V_original, I_original, I_min, I_max


# ---------------------- 2. 太阳能电池模型（完全不变，仅保留你改的初始值） ----------------------
def solar_cell_model(V, params, I_min, I_max):
    I_ph, I0, n, Rs, Rsh = params
    Vt = 0.026
    I = np.zeros_like(V, dtype=np.float64)
    for i, v in enumerate(V):
        # 显式近似解初始值（你之前改的，保留）
        exp_arg_init = v / (n * Vt)
        exp_arg_init = np.clip(exp_arg_init, -20, 20)
        exp_term_init = np.exp(exp_arg_init) - 1
        shunt_term_init = v / Rsh
        init_I = I_ph - I0 * exp_term_init - shunt_term_init
        init_I = np.clip(init_I, I_min * 0.1, I_max * 2)
        I_i = init_I

        for _ in range(100):
            exp_argument = (v + I_i * Rs) / (n * Vt)
            exp_argument = np.clip(exp_argument, -20, 20)
            exp_term = np.exp(exp_argument) - 1
            shunt_term = (v + I_i * Rs) / Rsh
            new_I_i = I_ph - I0 * exp_term - shunt_term  # 模型公式完全不动
            new_I_i = np.clip(new_I_i, I_min * 0.1, I_max * 2)
            if abs(new_I_i - I_i) < 1e-8:
                I_i = new_I_i
                break
            I_i = new_I_i
        I[i] = init_I if np.isnan(I_i) else I_i
    return I


# ---------------------- 3. 目标函数（微调奖励感知） ----------------------
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
    # 调整损失计算：增加全局缩放+绝对误差，让RL更敏感
    loss = 1000 * np.sqrt((1 / m) * np.sum(((I_meas_valid - I_sim_valid) ** 2) / (I_meas_valid + 1e-8)))
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    return loss


# ---------------------- 4. 升级后的RL控制器（动态ε+自适应学习率） ----------------------
class RLController:
    def __init__(self, action_num=5, lr=0.1, gamma=0.9, epsilon_start=0.8, epsilon_end=0.1, epsilon_decay=0.995):
        self.action_num = action_num  # 动作数增加到5，更多探索比例选择
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start  # 初始探索率更高
        self.epsilon_end = epsilon_end  # 最终探索率
        self.epsilon_decay = epsilon_decay  # 衰减系数
        self.epsilon = epsilon_start  # 当前探索率
        self.Q_table = {}
        self.iter_count = 0  # 迭代计数，用于ε衰减

    def get_q(self, state):
        if np.isnan(state):
            return [0.0] * self.action_num
        state = round(state, 8)  # 降低离散化精度，减少Q表维度
        if state not in self.Q_table:
            self.Q_table[state] = [0.0] * self.action_num
        return self.Q_table[state]

    def choose_action(self, state):
        self.iter_count += 1
        # 动态ε衰减：前期多探索，后期多利用
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if np.isnan(state):
            return np.random.choice(self.action_num)
        q_values = self.get_q(state)
        # 带噪声的贪心策略：避免局部最优
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_num)
        else:
            # 加入小噪声，打破平局
            noisy_q = q_values + np.random.normal(0, 0.01, len(q_values))
            return np.argmax(noisy_q)

    def update_q(self, state, action, reward, next_state):
        if np.isnan(state) or np.isnan(next_state):
            return
        current_q = self.get_q(state)[action]
        next_q = self.get_q(next_state)
        next_max_q = max(next_q) if next_q else 0.0
        # 自适应学习率：损失越小，学习率越低
        adaptive_lr = self.lr * (1 - min(0.9, state / 1e5))
        new_q = current_q + adaptive_lr * (reward + self.gamma * next_max_q - current_q)
        self.Q_table[round(state, 8)][action] = new_q


# ---------------------- 5. 升级后的RLRTLBO算法（自适应步长+种群多样性） ----------------------
class RLRTLBO:
    def __init__(self, V, I_meas, I_min, I_max, rl_controller, pop_size=100, max_iter=2000):
        self.V = V
        self.I_meas = I_meas
        self.I_min = I_min
        self.I_max = I_max
        self.rl = rl_controller
        self.pop_size = pop_size  # 种群规模翻倍
        self.max_iter = max_iter  # 迭代次数翻倍
        # 终极参数范围（彻底放开，让RL+TLBO充分搜索）
        self.param_bounds = [
            (0.1, 50),  # I_ph
            (0.01, 50),  # I0
            (0.1, 10.0),  # n
            (0.001, 200),  # Rs
            (0.1, 1000)  # Rsh
        ]
        self.population = self._init_population()

    def _init_population(self):
        """增强种群多样性：加入随机扰动"""
        population = []
        for _ in range(self.pop_size):
            # 基础随机参数
            I_ph = np.random.uniform(*self.param_bounds[0])
            I0 = np.random.uniform(*self.param_bounds[1])
            n = np.random.uniform(*self.param_bounds[2])
            Rs = np.random.uniform(*self.param_bounds[3])
            Rsh = np.random.uniform(*self.param_bounds[4])
            # 加入随机扰动，避免种群同质化
            perturb = np.random.normal(1, 0.1, 5)  # 10%的随机扰动
            I_ph *= perturb[0]
            I0 *= perturb[1]
            n *= perturb[2]
            Rs *= perturb[3]
            Rsh *= perturb[4]
            # 裁剪回范围
            I_ph = max(self.param_bounds[0][0], min(I_ph, self.param_bounds[0][1]))
            I0 = max(self.param_bounds[1][0], min(I0, self.param_bounds[1][1]))
            n = max(self.param_bounds[2][0], min(n, self.param_bounds[2][1]))
            Rs = max(self.param_bounds[3][0], min(Rs, self.param_bounds[3][1]))
            Rsh = max(self.param_bounds[4][0], min(Rsh, self.param_bounds[4][1]))
            population.append([I_ph, I0, n, Rs, Rsh])
        return population

    def _evaluate_population(self):
        loss_list = []
        for params in self.population:
            loss = objective_function(params, self.V, self.I_meas, self.I_min, self.I_max)
            loss_list.append(loss)
        valid_loss_mask = np.isfinite(loss_list) & (np.array(loss_list) < 1e9)
        if not np.any(valid_loss_mask):
            return self.population[0], 1e10, 1e10
        valid_losses = np.array(loss_list)[valid_loss_mask]
        valid_pop = np.array(self.population)[valid_loss_mask]
        best_idx = np.argmin(valid_losses)
        return valid_pop[best_idx].tolist(), valid_losses[best_idx], np.mean(valid_losses)

    def _teaching_phase(self, best_params, explore_ratio):
        new_population = []
        teacher = np.array(best_params)
        mean_params = np.mean(self.population, axis=0)
        for params in self.population:
            params_np = np.array(params)
            TF = round(1 + np.random.random())
            # 自适应步长：损失越大，步长越大
            current_loss = objective_function(params, self.V, self.I_meas, self.I_min, self.I_max)
            adaptive_step = 0.3 + min(0.7, current_loss / 1e4)  # 步长0.3~1.0

            if np.random.uniform(0, 1) < explore_ratio:
                # 扩大探索范围：0.1~5.0倍
                new_params = [
                    params[0] * np.random.uniform(0.1, 5.0),
                    params[1] * np.random.uniform(0.1, 5.0),
                    params[2] * np.random.uniform(0.5, 2.0),
                    params[3] * np.random.uniform(0.1, 5.0),
                    params[4] * np.random.uniform(0.1, 5.0)
                ]
            else:
                # 自适应向老师学习
                new_params = (params_np + np.random.random(5) * adaptive_step * (teacher - TF * mean_params)).tolist()
            new_params = self._clip_params(new_params)
            new_population.append(new_params)
        return new_population

    def _learning_phase(self):
        new_population = []
        for i in range(self.pop_size):
            j = random.sample([k for k in range(self.pop_size) if k != i], 1)[0]
            params_i = np.array(self.population[i])
            params_j = np.array(self.population[j])
            loss_i = objective_function(params_i.tolist(), self.V, self.I_meas, self.I_min, self.I_max)
            loss_j = objective_function(params_j.tolist(), self.V, self.I_meas, self.I_min, self.I_max)

            # 自适应学习步长
            adaptive_step = 0.3 + min(0.7, max(loss_i, loss_j) / 1e4)
            if loss_i > loss_j:
                new_params = (params_i + np.random.random(5) * adaptive_step * (params_j - params_i)).tolist()
            else:
                new_params = (params_j + np.random.random(5) * adaptive_step * (params_i - params_j)).tolist()
            new_params = self._clip_params(new_params)
            new_population.append(new_params)
        return new_population

    def _clip_params(self, params):
        clipped = []
        for param, (low, high) in zip(params, self.param_bounds):
            clipped.append(max(low, min(param, high)))
        return clipped

    def train(self):
        # 记录最优参数，避免迭代中丢失
        global_best_loss = 1e10
        global_best_params = None

        for iter in range(self.max_iter):
            best_params, best_loss, avg_loss = self._evaluate_population()
            # 更新全局最优
            if best_loss < global_best_loss:
                global_best_loss = best_loss
                global_best_params = best_params
            current_state = best_loss

            # RL选择探索比例（动作数增加到5，更多选择）
            action = self.rl.choose_action(current_state)
            explore_ratio = [0.1, 0.3, 0.5, 0.7, 0.9][action]

            self.population = self._teaching_phase(best_params, explore_ratio)
            self.population = self._learning_phase()

            new_best_params, new_best_loss, new_avg_loss = self._evaluate_population()
            # 重构奖励函数：奖励与损失负相关，且放大差异
            reward = 10000 / (new_best_loss + 1e-8)
            next_state = new_best_loss
            self.rl.update_q(current_state, action, reward, next_state)

            if (iter + 1) % 100 == 0:
                print(f"迭代{iter + 1:4d} | 全局最优损失: {global_best_loss:.2e} | 当前最优损失: {new_best_loss:.2e}")

            # 更宽松的收敛条件
            if global_best_loss < 1e-2:
                print(f"提前收敛！迭代{iter + 1}次，全局最优损失={global_best_loss:.2e}")
                break

        # 最终用全局最优参数计算
        final_I_pred = solar_cell_model(self.V, global_best_params, self.I_min, self.I_max)
        return global_best_params, global_best_loss, final_I_pred


# ---------------------- 6. 主程序（调整RL初始化） ----------------------
if __name__ == "__main__":
    EXCEL_PATH = "C:\\Users\\18372\\PycharmProjects\\pythonProject1\\5.xls"
    V_processed, I_meas_processed, _, _, I_min, I_max = load_excel_and_preprocess(EXCEL_PATH)

    # 初始化升级后的RL控制器
    rl_controller = RLController(
        action_num=5,
        lr=0.15,  # 提高初始学习率
        gamma=0.95,  # 提高折扣因子
        epsilon_start=0.8,
        epsilon_end=0.1,
        epsilon_decay=0.995
    )
    # 初始化升级后的RLRTLBO
    rlrtlbo = RLRTLBO(
        V_processed, I_meas_processed, I_min, I_max,
        rl_controller,
        pop_size=100,  # 种群规模翻倍
        max_iter=2000  # 迭代次数翻倍
    )

    final_params, final_loss, final_I_pred = rlrtlbo.train()

    print("\n" + "=" * 70)
    print("拟合完成！")
    print(f"全局最优损失：{final_loss:.2e}")
    print(f"最优参数：")
    print(f"  - I_ph: {final_params[0]:.2e} A")
    print(f"  - I0: {final_params[1]:.2e} A")
    print(f"  - n: {final_params[2]:.2f}")
    print(f"  - Rs: {final_params[3]:.3f} Ω")
    print(f"  - Rsh: {final_params[4]:.0f} Ω")
    print("=" * 70)

    # 可视化
    try:
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(12, 7))
        plt.plot(V_processed, I_meas_processed, 'o', label='原始实测电流', markersize=4)
        plt.plot(V_processed, final_I_pred, '-r', label='模型预测电流', linewidth=2)
        plt.xlabel('电压 (V)', fontsize=12)
        plt.ylabel('电流 (A)', fontsize=12)
        plt.title('太阳能电池模型拟合结果（升级RLRTLBO算法）', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("未安装matplotlib，跳过可视化")