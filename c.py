import numpy as np
import random
import pandas as pd


# ---------------------- 1. 数据预处理（新增对数缩放+异常值过滤） ----------------------
def load_excel_and_preprocess(excel_path):
    df = pd.read_excel(excel_path, header=1)
    x_original = df['VG'].values  # 原始电压
    I_original = df[['S1', 'S2', 'S3']].mean(axis=1).values  # S1/S2/S3均值

    # 导师要求：镜像+平移
    x_mirror = 20 - x_original
    VG_processed = x_mirror - 10
    I_meas_processed = I_original

    # 优化1：过滤极端异常点（保留99%数据）
    valid_mask = (np.abs(VG_processed) < 20) & (I_meas_processed > 1e-14) & (I_meas_processed < 1e-6)
    VG_processed = VG_processed[valid_mask]
    I_meas_processed = I_meas_processed[valid_mask]

    # 计算实测电流统计值（用于后续归一化）
    I_min = np.min(I_meas_processed) if len(I_meas_processed) > 0 else 1e-15
    I_max = np.max(I_meas_processed) if len(I_meas_processed) > 0 else 1e-7
    I_mean = np.mean(I_meas_processed) if len(I_meas_processed) > 0 else 1e-10
    I_std = np.std(I_meas_processed) if len(I_meas_processed) > 0 else 1e-9

    return VG_processed, I_meas_processed, I_min, I_max, I_mean, I_std


# ---------------------- 2. 优化版TFT模型（适配小电流+非对称特性） ----------------------
def tft_optimized_model(V, params, I_min, I_max):
    """
    优化版TFT场效应模型：
    - 新增漏电流项（适配负电压区小电流）
    - 分段函数更贴合实测斜率
    - 电流缩放适配1e-14~1e-7 A量级
    """
    mu, Vth, beta, Rs, I_leak = params  # 新增漏电流参数I_leak
    I = np.zeros_like(V, dtype=np.float64)

    for i, v in enumerate(V):
        # 迭代求解实际沟道电压（增加迭代次数）
        V_g = v
        I_i = I_leak  # 初始值设为漏电流
        for _ in range(8):  # 迭代次数从5→8，提高收敛性
            V_drop = Rs * I_i  # 串联电阻电压降
            V_eff = V_g - V_drop - Vth  # 有效栅压

            # 优化2：分段模型精准匹配实测斜率
            if V_eff <= 0:
                # 截止区：漏电流主导
                I_i = I_leak * np.exp(v / 50)  # 微弱的电压依赖性
            elif np.abs(V_eff) < 2:
                # 弱反型区：线性增长（适配实测低电压段）
                I_i = beta * mu * np.abs(V_eff) + I_leak
            elif V_eff <= 5:
                # 强反型区：平方根增长（适配实测中电压段）
                I_i = beta * mu * np.sqrt(np.abs(V_eff)) + I_leak
            else:
                # 饱和区：平方增长（适配实测高电压段）
                I_i = beta * mu * (V_eff) ** 2 + I_leak

        # 优化3：精准裁剪到实测范围
        I_i = np.clip(I_i, I_min * 0.5, I_max * 1.5)
        I[i] = I_i

    return I


# ---------------------- 3. 目标函数（对数+标准差归一化，聚焦相对误差） ----------------------
def objective_function(params, V, I_meas, I_min, I_max, I_mean, I_std):
    # 生成预测电流
    I_sim = tft_optimized_model(V, params, I_min, I_max)

    # 优化4：对数变换（适配小电流量级）
    I_meas_log = np.log10(I_meas + 1e-16)  # 加偏移避免log(0)
    I_sim_log = np.log10(I_sim + 1e-16)

    # 优化5：标准差归一化（降低极端值影响）
    loss = np.sqrt(np.mean(((I_meas_log - I_sim_log) / (I_std / I_mean)) ** 2))

    # 惩罚项：避免参数溢出合理范围
    mu, Vth, beta, Rs, I_leak = params
    penalty = 0
    if mu < 1e-6 or mu > 1e-2:
        penalty += 10
    if Vth < -3 or Vth > 3:
        penalty += 10
    if I_leak < 1e-15 or I_leak > 1e-12:
        penalty += 10

    return loss + penalty * 0.1


# ---------------------- 4. RL控制器（优化探索策略） ----------------------
class RLController:
    def __init__(self, action_num=3, lr=0.05, gamma=0.95, epsilon=0.2):
        self.action_num = action_num
        self.lr = lr  # 学习率降低，更稳定
        self.gamma = gamma  # 折扣因子提高
        self.epsilon = epsilon  # 探索率降低，聚焦最优方向
        self.Q_table = {}

    def get_q(self, state):
        state = round(state, 8)  # 精度降低，减少Q表维度
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
        next_max_q = max(self.get_q(next_state)) if round(next_state, 8) in self.Q_table else 0.0
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.Q_table[round(state, 8)][action] = new_q


# ---------------------- 5. RLRTLBO算法（全维度优化） ----------------------
class RLRTLBO:
    def __init__(self, V, I_meas, I_min, I_max, I_mean, I_std, rl_controller):
        self.V = V
        self.I_meas = I_meas
        self.I_min = I_min
        self.I_max = I_max
        self.I_mean = I_mean
        self.I_std = I_std
        self.rl = rl_controller

        # 优化6：超参数适配小电流拟合
        self.pop_size = 80  # 种群规模从50→80
        self.max_iter = 1500  # 迭代次数从1000→1500
        self.early_stop_patience = 50  # 早停机制

        # 优化7：参数范围精准匹配实测（基于1e-14~1e-7 A量级）
        self.param_bounds = [
            (1e-6, 1e-3),  # mu: 迁移率（适配小电流）
            (-3, 3),  # Vth: 阈值电压（聚焦实测拐点）
            (1e-10, 1e-7),  # beta: 器件参数（核心适配电流量级）
            (500, 2000),  # Rs: 串联电阻（增大以降低电流）
            (1e-15, 1e-12)  # I_leak: 漏电流（适配负电压区）
        ]
        self.population = self._init_population()
        self.best_loss_history = []  # 早停监控

    def _init_population(self):
        population = []
        for _ in range(self.pop_size):
            params = [np.random.uniform(*b) for b in self.param_bounds]
            population.append(params)
        return population

    def _evaluate_population(self):
        loss_list = []
        for params in self.population:
            loss = objective_function(params, self.V, self.I_meas,
                                      self.I_min, self.I_max, self.I_mean, self.I_std)
            loss_list.append(loss)
        best_idx = np.argmin(loss_list)
        return (self.population[best_idx], loss_list[best_idx],
                np.mean(loss_list), loss_list)

    def _teaching_phase(self, best_params, explore_ratio):
        new_population = []
        teacher = np.array(best_params)
        mean_params = np.mean(self.population, axis=0)

        for params in self.population:
            params_np = np.array(params)
            TF = 1 if np.random.random() < 0.7 else 2  # 降低TF=2的概率，更稳定

            if np.random.uniform(0, 1) < explore_ratio:
                # 优化8：自适应探索步长（小参数小步长）
                new_params = []
                for i, p in enumerate(params):
                    if i in [0, 2, 4]:  # 小量级参数
                        step = np.random.uniform(0.9, 1.1)
                    else:  # 电压/电阻参数
                        step = np.random.uniform(0.85, 1.15)
                    new_params.append(p * step)
            else:
                # 优化9：降低学习步长，避免震荡
                new_params = (params_np + np.random.random(5) * 0.08 *
                              (teacher - TF * mean_params)).tolist()

            # 严格裁剪参数
            new_params = [max(b[0], min(b[1], p)) for p, b in zip(new_params, self.param_bounds)]
            new_population.append(new_params)

        return new_population

    def _learning_phase(self):
        new_population = []
        for i in range(self.pop_size):
            # 优化10：选择Top30%的优秀个体学习
            all_loss = [objective_function(p, self.V, self.I_meas,
                                           self.I_min, self.I_max, self.I_mean, self.I_std)
                        for p in self.population]
            top_idx = np.argsort(all_loss)[:int(self.pop_size * 0.3)]
            j = random.choice([k for k in top_idx if k != i]) if len(top_idx) > 0 else i

            params_i = np.array(self.population[i])
            params_j = np.array(self.population[j])

            loss_i = objective_function(params_i.tolist(), self.V, self.I_meas,
                                        self.I_min, self.I_max, self.I_mean, self.I_std)
            loss_j = objective_function(params_j.tolist(), self.V, self.I_meas,
                                        self.I_min, self.I_max, self.I_mean, self.I_std)

            # 优化11：动态学习步长
            step = 0.12 if loss_i > loss_j else 0.08
            if loss_i > loss_j:
                new_params = (params_i + np.random.random(5) * step * (params_j - params_i)).tolist()
            else:
                new_params = (params_j + np.random.random(5) * step * (params_i - params_j)).tolist()

            # 裁剪参数
            new_params = [max(b[0], min(b[1], p)) for p, b in zip(new_params, self.param_bounds)]
            new_population.append(new_params)

        return new_population

    def train(self):
        best_loss = float('inf')
        best_params = None
        patience_counter = 0

        for iter in range(self.max_iter):
            # 评估种群
            current_best_params, current_best_loss, avg_loss, loss_list = self._evaluate_population()

            # 早停机制
            self.best_loss_history.append(current_best_loss)
            if current_best_loss < best_loss:
                best_loss = current_best_loss
                best_params = current_best_params
                patience_counter = 0
            else:
                patience_counter += 1

            # RL决策
            current_state = current_best_loss
            action = self.rl.choose_action(current_state)
            explore_ratio = [0.2, 0.4, 0.6][action]  # 降低探索比例

            # 教学+学习
            self.population = self._teaching_phase(current_best_params, explore_ratio)
            self.population = self._learning_phase()

            # 奖励更新
            new_best_params, new_best_loss, _, _ = self._evaluate_population()
            reward = 1 / (new_best_loss + 1e-8)  # 奖励放大
            self.rl.update_q(current_state, action, reward, new_best_loss)

            # 打印进度（每50次迭代）
            if (iter + 1) % 50 == 0:
                print(
                    f"迭代{iter + 1:4d} | 最优损失: {best_loss:.4f} | 平均损失: {avg_loss:.4f} | 探索率: {explore_ratio:.1f}")

            # 早停
            if patience_counter >= self.early_stop_patience:
                print(f"早停触发！迭代{iter + 1}次，最优损失{best_loss:.4f}")
                break

        # 最终预测
        final_I_pred = tft_optimized_model(self.V, best_params, self.I_min, self.I_max)
        return best_params, best_loss, final_I_pred


# ---------------------- 6. 主程序（新增结果分析） ----------------------
if __name__ == "__main__":
    EXCEL_PATH = "C:\\Users\\18372\\PycharmProjects\\pythonProject1\\tfti.xls"

    # 1. 数据预处理
    VG_processed, I_meas_processed, I_min, I_max, I_mean, I_std = load_excel_and_preprocess(EXCEL_PATH)
    print(f"数据预处理完成 | 电压范围: [{VG_processed.min():.2f}, {VG_processed.max():.2f}] V")
    print(f"电流范围: [{I_min:.2e}, {I_max:.2e}] A | 数据量: {len(VG_processed)}")

    # 2. 初始化
    rl_controller = RLController()
    rlrtlbo = RLRTLBO(VG_processed, I_meas_processed, I_min, I_max, I_mean, I_std, rl_controller)

    # 3. 训练拟合
    final_params, final_loss, final_I_pred = rlrtlbo.train()

    # 4. 结果输出
    print("\n" + "=" * 80)
    print("TFT模型拟合完成！")
    print(f"最终损失值: {final_loss:.4f}（越小越好）")
    print(f"最优参数：")
    print(f"  - 载流子迁移率μ: {final_params[0]:.2e} cm²/(V·s)")
    print(f"  - 阈值电压Vth: {final_params[1]:.2f} V")
    print(f"  - 器件参数β: {final_params[2]:.2e} A/V²")
    print(f"  - 串联电阻Rs: {final_params[3]:.0f} Ω")
    print(f"  - 漏电流I_leak: {final_params[4]:.2e} A")
    print("=" * 80)

    # 5. 可视化（优化显示效果）
    try:
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.dpi"] = 100  # 提高分辨率

        fig, ax = plt.subplots(figsize=(14, 8))
        # 绘制实测点
        ax.semilogy(VG_processed, I_meas_processed, 'o', color='#1f77b4',
                    label='实测电流', markersize=5, alpha=0.8)
        # 绘制预测曲线（平滑处理）
        sort_idx = np.argsort(VG_processed)
        ax.semilogy(VG_processed[sort_idx], final_I_pred[sort_idx], '-r',
                    label='预测电流', linewidth=2.5, alpha=0.9)

        # 优化显示
        ax.set_xlabel('处理后电压 (V)', fontsize=14)
        ax.set_ylabel('电流 (A)', fontsize=14)
        ax.set_title('TFT电流-电压特性拟合结果（优化版模型）', fontsize=16, pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim([I_min * 0.1, I_max * 10])  # 适配电流范围
        plt.tight_layout()
        plt.show()

        # 输出拟合优度
        r2 = 1 - np.sum((I_meas_processed - final_I_pred) ** 2) / np.sum((I_meas_processed - I_mean) ** 2)
        print(f"\n拟合优度R²: {r2:.4f}（越接近1越好）")

    except ImportError:
        print("未安装matplotlib，跳过可视化")
    except Exception as e:
        print(f"可视化出错: {e}")