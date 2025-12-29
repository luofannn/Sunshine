import numpy as np
import random
import pandas as pd


# ---------------------- 1. 读取Excel+数据预处理（去掉镜像和平移，直接用原始数据） ----------------------
def load_excel_and_preprocess(excel_path):
    # 跳过第一行"Line #1"文本，读取前两列（A列=原始电压，B列=原始电流）
    df = pd.read_excel(excel_path, header=None, usecols=[0, 1], skiprows=1)

    # 强制转为浮点型，避免类型错误
    V_original = df.iloc[:, 0].astype(float).values  # 原始电压（直接作为模型输入）
    I_original = df.iloc[:, 1].astype(float).values  # 原始电流（直接作为实测值）

    # 过滤极端异常点（保留所有点，同时过滤NaN/inf的原始数据）
    valid_mask = np.isfinite(V_original) & np.isfinite(I_original)  # 过滤原始数据中的NaN/inf
    V_processed = V_original[valid_mask]
    I_meas_processed = I_original[valid_mask]

    # 计算实测电流极值（用于模型裁剪，避免0值报错）
    I_min = np.min(I_meas_processed[I_meas_processed > 0]) if np.any(I_meas_processed > 0) else 1e-16
    I_max = np.max(I_meas_processed) if np.any(I_meas_processed > 0) else 1e-4
    return V_processed, I_meas_processed, V_original, I_original, I_min, I_max


# ---------------------- 2. 太阳能电池5参数模型（强化数值稳定性，防止NaN） ----------------------
def solar_cell_model(V, params, I_min, I_max):
    """
    太阳能电池5参数模型：完全保留导师给定公式，仅优化初始值
    """
    I_ph, I0, n, Rs, Rsh = params
    Vt = 0.026  # 常温热电压（固定值）
    I = np.zeros_like(V, dtype=np.float64)  # 初始化电流数组

    # 迭代求解（避免隐式方程无法直接计算）
    for i, v in enumerate(V):
        # ========== 优化：用模型显式近似解做初始值（不碰模型公式） ==========
        # 第一步：忽略I·Rs项，把隐式方程转为显式，计算初始值
        exp_arg_init = v / (n * Vt)
        exp_arg_init = np.clip(exp_arg_init, -20, 20)
        exp_term_init = np.exp(exp_arg_init) - 1
        shunt_term_init = v / Rsh
        # 用显式近似解作为初始值（贴合模型公式的逻辑）
        init_I = I_ph - I0 * exp_term_init - shunt_term_init
        # 兜底：确保初始值在合理范围
        init_I = np.clip(init_I, I_min * 0.1, I_max * 2)
        I_i = init_I

        for _ in range(100):  # 迭代次数从50→100，确保收敛
            # 限制指数项范围（避免exp溢出/下溢）
            exp_argument = (v + I_i * Rs) / (n * Vt)
            exp_argument = np.clip(exp_argument, -20, 20)

            # 核心公式（完全保留导师给的，一个符号都不改）
            exp_term = np.exp(exp_argument) - 1
            shunt_term = (v + I_i * Rs) / Rsh
            new_I_i = I_ph - I0 * exp_term - shunt_term

            # 裁剪电流值（仅限制极端值）
            new_I_i = np.clip(new_I_i, I_min * 0.1, I_max * 2)

            # 收敛判断，提前终止迭代
            if abs(new_I_i - I_i) < 1e-8:
                I_i = new_I_i
                break
            I_i = new_I_i

        # 最终兜底：如果I_i是NaN，用显式近似解填充（而非I_min）
        I[i] = init_I if np.isnan(I_i) else I_i
    return I


# ---------------------- 3. 目标函数（NaN防护，确保损失有效） ----------------------
def objective_function(params, V, I_meas, I_min, I_max):
    """归一化RMSE损失函数：微调缩放，保留原逻辑"""
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
    # ========== 优化：增加全局缩放因子，让损失值更敏感 ==========
    loss = 100 * np.sqrt((1 / m) * np.sum(((I_meas_valid - I_sim_valid) / (I_meas_valid + 1e-8)) ** 2))

    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    return loss


# ---------------------- 4. RL控制器（NaN防护，避免无效状态更新） ----------------------
class RLController:
    def __init__(self, action_num=3, lr=0.1, gamma=0.9, epsilon=0.3):
        self.action_num = action_num  # 动作数（对应不同探索比例）
        self.lr = lr  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.Q_table = {}  # Q表存储状态-动作值

    def get_q(self, state):
        """获取状态对应的Q值（无则初始化，NaN状态返回默认值）"""
        # 防护：如果state是NaN，返回默认Q值
        if np.isnan(state):
            return [0.0] * self.action_num
        state = round(state, 10)  # 状态离散化，避免精度问题
        if state not in self.Q_table:
            self.Q_table[state] = [0.0] * self.action_num
        return self.Q_table[state]

    def choose_action(self, state):
        """ε-贪心策略选择动作（NaN状态随机选动作）"""
        # 防护：NaN状态随机探索
        if np.isnan(state):
            return np.random.choice(self.action_num)
        q_values = self.get_q(state)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_num)  # 随机探索
        else:
            return np.argmax(q_values)  # 贪心利用

    def update_q(self, state, action, reward, next_state):
        """更新Q表（时序差分学习，跳过NaN状态）"""
        # 核心防护：跳过NaN状态的更新，避免KeyError
        if np.isnan(state) or np.isnan(next_state):
            return
        current_q = self.get_q(state)[action]
        next_max_q = max(self.get_q(next_state)) if round(next_state, 10) in self.Q_table else 0.0
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.Q_table[round(state, 10)][action] = new_q


# ---------------------- 5. RLRTLBO算法（强化种群评估，过滤NaN损失） ----------------------
class RLRTLBO:
    def __init__(self, V, I_meas, I_min, I_max, rl_controller, pop_size=50, max_iter=1000):
        self.V = V  # 电压数据（原始电压）
        self.I_meas = I_meas  # 实测电流
        self.I_min = I_min  # 电流最小值
        self.I_max = I_max  # 电流最大值
        self.rl = rl_controller  # RL控制器
        self.pop_size = pop_size  # 种群规模
        self.max_iter = max_iter  # 最大迭代次数

        # 参数范围（参考导师给的最优参数调整）
        self.param_bounds = [
            (1, 50),  # I_ph（光生电流）：直接覆盖你的实测电流范围（0.3~15A）
            (1, 20),  # I0（反向饱和电流）：从1e-5→1，大幅放大
            (0.1, 10.0),  # n（理想因子）：放宽到10
            (0.001, 100),  # Rs（串联电阻）：放大到100
            (0.1, 500)  # Rsh（并联电阻）：放大到500
        ]
        self.population = self._init_population()  # 初始化种群

    def _init_population(self):
        """初始化种群（随机生成参数）"""
        population = []
        for _ in range(self.pop_size):
            I_ph = np.random.uniform(*self.param_bounds[0])
            I0 = np.random.uniform(*self.param_bounds[1])
            n = np.random.uniform(*self.param_bounds[2])
            Rs = np.random.uniform(*self.param_bounds[3])
            Rsh = np.random.uniform(*self.param_bounds[4])
            population.append([I_ph, I0, n, Rs, Rsh])
        return population

    def _evaluate_population(self):
        """评估种群（过滤NaN/inf损失，防止最优损失为NaN）"""
        loss_list = []
        for params in self.population:
            loss = objective_function(params, self.V, self.I_meas, self.I_min, self.I_max)
            loss_list.append(loss)

        # 核心防护：过滤NaN/inf损失，只保留有效损失
        valid_loss_mask = np.isfinite(loss_list) & (np.array(loss_list) < 1e9)
        if not np.any(valid_loss_mask):
            # 全是无效损失，返回第一个参数和极大损失
            return self.population[0], 1e10, 1e10

        valid_losses = np.array(loss_list)[valid_loss_mask]
        valid_pop = np.array(self.population)[valid_loss_mask]
        best_idx = np.argmin(valid_losses)
        return valid_pop[best_idx].tolist(), valid_losses[best_idx], np.mean(valid_losses)

    def _teaching_phase(self, best_params, explore_ratio):
        """教学阶段（增大探索步长，保留原逻辑）"""
        new_population = []
        teacher = np.array(best_params)
        mean_params = np.mean(self.population, axis=0)
        for params in self.population:
            params_np = np.array(params)
            TF = round(1 + np.random.random())  # 教学因子（1或2）
            if np.random.uniform(0, 1) < explore_ratio:
                # ========== 优化：增大探索步长 ==========
                new_params = [
                    params[0] * np.random.uniform(0.5, 2.0),  # 从0.8~1.2→0.5~2.0
                    params[1] * np.random.uniform(0.5, 2.0),
                    params[2] * np.random.uniform(0.8, 1.2),  # 理想因子n小幅调整
                    params[3] * np.random.uniform(0.5, 2.0),
                    params[4] * np.random.uniform(0.5, 2.0)
                ]
            else:
                # 向老师学习（步长从0.3→0.5，加快收敛）
                new_params = (params_np + np.random.random(5) * 0.5 * (teacher - TF * mean_params)).tolist()
            new_params = self._clip_params(new_params)  # 裁剪到参数范围
            new_population.append(new_params)
        return new_population

    def _learning_phase(self):
        """学习阶段（个体间互相学习）"""
        new_population = []
        for i in range(self.pop_size):
            # 随机选一个不同的个体
            j = random.sample([k for k in range(self.pop_size) if k != i], 1)[0]
            params_i = np.array(self.population[i])
            params_j = np.array(self.population[j])

            # 计算两个个体的损失
            loss_i = objective_function(params_i.tolist(), self.V, self.I_meas, self.I_min, self.I_max)
            loss_j = objective_function(params_j.tolist(), self.V, self.I_meas, self.I_min, self.I_max)

            # 向更优的个体学习
            if loss_i > loss_j:
                new_params = (params_i + np.random.random(5) * 0.3 * (params_j - params_i)).tolist()
            else:
                new_params = (params_j + np.random.random(5) * 0.3 * (params_i - params_j)).tolist()

            new_params = self._clip_params(new_params)
            new_population.append(new_params)
        return new_population

    def _clip_params(self, params):
        """裁剪参数到指定范围"""
        clipped = []
        for param, (low, high) in zip(params, self.param_bounds):
            clipped.append(max(low, min(param, high)))
        return clipped

    def train(self):
        """训练主流程"""
        for iter in range(self.max_iter):
            # 评估当前种群
            best_params, best_loss, avg_loss = self._evaluate_population()
            current_state = best_loss

            # RL选择探索比例
            action = self.rl.choose_action(current_state)
            explore_ratio = [0.3, 0.5, 0.7][action]

            # 教学+学习阶段更新种群
            self.population = self._teaching_phase(best_params, explore_ratio)
            self.population = self._learning_phase()

            # 评估新种群
            new_best_params, new_best_loss, new_avg_loss = self._evaluate_population()

            # RL更新Q表（奖励与损失负相关）
            reward = 1 / (new_best_loss + 1e-10)
            next_state = new_best_loss
            self.rl.update_q(current_state, action, reward, next_state)

            # 打印迭代进度
            if (iter + 1) % 100 == 0:
                print(
                    f"迭代{iter + 1:3d} | 最优损失: {best_loss:.2e} | 最优参数: I_ph={best_params[0]:.2e}, I0={best_params[1]:.2e}, n={best_params[2]:.2f}, Rs={best_params[3]:.3f}, Rsh={best_params[4]:.0f}")

            # 提前收敛（损失小于阈值则停止）
            if new_best_loss < 1e-3:
                print(f"提前收敛！迭代{iter + 1}次，损失={new_best_loss:.2e}")
                break

        # 最终结果
        final_best_params, final_loss, _ = self._evaluate_population()
        final_I_pred = solar_cell_model(self.V, final_best_params, self.I_min, self.I_max)
        return final_best_params, final_loss, final_I_pred


# ---------------------- 6. 主程序（运行+可视化） ----------------------
if __name__ == "__main__":
    # 替换为你的Excel文件路径
    EXCEL_PATH = "C:\\Users\\18372\\PycharmProjects\\pythonProject1\\5.xls"

    # 1. 数据预处理（去掉镜像平移，直接用原始电压）
    V_processed, I_meas_processed, _, _, I_min, I_max = load_excel_and_preprocess(EXCEL_PATH)

    # 2. 初始化RL控制器和RLRTLBO算法
    rl_controller = RLController()
    rlrtlbo = RLRTLBO(V_processed, I_meas_processed, I_min, I_max, rl_controller)

    # 3. 训练拟合
    final_params, final_loss, final_I_pred = rlrtlbo.train()

    # 4. 输出结果
    print("\n" + "=" * 70)
    print("拟合完成！")
    print(f"目标函数最小值：{final_loss:.2e}")
    print(f"最优参数（导师要求的5个参数）：")
    print(f"  - 光生电流I_ph: {final_params[0]:.2e} A")
    print(f"  - 反向饱和电流I0: {final_params[1]:.2e} A")
    print(f"  - 理想因子n: {final_params[2]:.2f}")
    print(f"  - 串联电阻Rs: {final_params[3]:.3f} Ω")
    print(f"  - 并联电阻Rsh: {final_params[4]:.0f} Ω")
    print("=" * 70)

    # 5. 可视化结果（X轴为原始电压）
    try:
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]  # 支持中文
        plt.rcParams["axes.unicode_minus"] = False  # 支持负号

        plt.figure(figsize=(12, 7))
        plt.plot(V_processed, I_meas_processed, 'o', label='原始实测电流', markersize=4)
        plt.plot(V_processed, final_I_pred, '-r', label='太阳能电池模型预测电流', linewidth=2)
        plt.xlabel('原始电压 (V)', fontsize=12)
        plt.ylabel('电流 (A)', fontsize=12)
        plt.title('TFT数据拟合结果（RLRTLBO+太阳能电池5参数模型）', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("未安装matplotlib，跳过可视化")