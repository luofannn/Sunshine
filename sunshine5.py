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


# ---------------------- 2. 太阳能电池模型（修复核心逻辑错误） ----------------------
def solar_cell_model(V, params, I_min, I_max):
    I_ph, I0, n, Rs, Rsh = params
    Vt = 0.026  # 热电压，25℃时约0.026V
    I = np.zeros_like(V, dtype=np.float64)

    for i, v in enumerate(V):
        # 改进初始值：加入Rs的近似（更接近真实公式）
        exp_arg_init = (v + I_ph * Rs) / (n * Vt)  # 用I_ph近似I，加入Rs项
        exp_arg_init = np.clip(exp_arg_init, -20, 20)  # 防止指数爆炸
        exp_term_init = np.exp(exp_arg_init) - 1
        shunt_term_init = (v + I_ph * Rs) / Rsh
        init_I = I_ph - I0 * exp_term_init - shunt_term_init
        init_I = np.clip(init_I, I_min * 0.1, I_max * 2)  # 合理裁剪范围
        I_i = init_I

        # 迭代求解隐式方程
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


# ---------------------- 3. 目标函数（微调奖励感知） ----------------------
def objective_function(params, V, I_meas, I_min, I_max):
    if len(V) == 0 or len(I_meas) == 0:
        return 1e10

    I_sim = solar_cell_model(V, params, I_min, I_max)

    # 过滤异常值
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
    loss = 100 * np.sqrt((1 / m) * np.sum(((I_meas_valid - I_sim_valid) ** 2) / (I_meas_valid + 1e-8)))

    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    return loss


# ---------------------- 4. 新增：单参数贡献度计算函数 ----------------------
def calculate_single_param_contrib(params, V, I_meas, I_min, I_max, param_bounds, DELTA):
    """
    计算每个参数的贡献度，反映该参数当前是否需要调整、往哪个方向调整
    :param params: 当前参数列表 [I_ph, I0, n, Rs, Rsh]
    :param V: 电压数据
    :param I_meas: 实测电流数据
    :param I_min/I_max: 电流约束
    :param param_bounds: 参数范围
    :param DELTA: 单参数微调步长
    :return: 单参数贡献度字典
    """
    PARAMS_NAME = ["I_ph", "I0", "n", "Rs", "Rsh"]
    param_dict = dict(zip(PARAMS_NAME, params))
    global_error = objective_function(params, V, I_meas, I_min, I_max)

    param_contribution = {}
    for idx, param_name in enumerate(PARAMS_NAME):
        original_val = param_dict[param_name]   #相当于根据索引名去取值
        delta_val = DELTA[param_name]
        low, high = param_bounds[idx]

        # 尝试上调参数
        param_dict[param_name] = original_val + delta_val
        param_dict[param_name] = np.clip(param_dict[param_name], low, high)
        loss_up = objective_function(list(param_dict.values()), V, I_meas, I_min, I_max)

        # 尝试下调参数
        param_dict[param_name] = original_val - delta_val
        param_dict[param_name] = np.clip(param_dict[param_name], low, high)
        loss_down = objective_function(list(param_dict.values()), V, I_meas, I_min, I_max)

        # 恢复原值
        param_dict[param_name] = original_val

        # 计算贡献度：负数=当前参数差，需要调整；正数=当前参数优，无需调整
        best_loss = min(loss_up, loss_down)
        contribution = global_error - best_loss
        param_contribution[param_name] = {
            "contribution": contribution,
            "in_bound": (low <= original_val <= high),
            "loss_up": loss_up,
            "loss_down": loss_down,
            "delta": delta_val
        }
    return param_contribution


# ---------------------- 5. 升级后的RL控制器（动态ε+自适应学习率） ----------------------
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


# ---------------------- 6. 升级后的RLRTLBO算法（加入单参数贡献度指导） ----------------------
class RLRTLBO:
    def __init__(self, V, I_meas, I_min, I_max, rl_controller, pop_size=100, max_iter=2000):
        self.V = V
        self.I_meas = I_meas
        self.I_min = I_min
        self.I_max = I_max
        self.rl = rl_controller
        self.pop_size = pop_size
        self.max_iter = max_iter
        # 优化参数范围（符合太阳能电池物理特性）
        self.param_bounds = [
            (10, 20),  # I_ph（短路电流量级，避免过大）
            (1e-12, 1e-7),  # I0（反向饱和电流，物理上极小）
            (1.0, 2.0),  # n（二极管品质因子，通常1~2）
            (0.01, 1.0),  # Rs（串联电阻，通常很小）
            (100, 1000)  # Rsh（并联电阻，通常很大）
        ]
        # 新增：单参数微调步长（与param_bounds顺序对应）
        self.DELTA = {
            "I_ph": 0.01,
            "I0": 1e-10,
            "n": 0.01,
            "Rs": 0.01,
            "Rsh": 1.0
        }
        self.PARAMS_NAME = ["I_ph", "I0", "n", "Rs", "Rsh"]

        self.population = self._init_population()

    def _init_population(self):
        """增强种群多样性：加入随机扰动"""
        population = []
        for _ in range(self.pop_size):
            # 基础随机参数（符合物理特性）
            I_ph = np.random.uniform(*self.param_bounds[0])
            I0 = np.random.uniform(*self.param_bounds[1])
            n = np.random.uniform(*self.param_bounds[2])
            Rs = np.random.uniform(*self.param_bounds[3])
            Rsh = np.random.uniform(*self.param_bounds[4])

            # 加入小幅度随机扰动（5%），避免种群同质化
            perturb = np.random.normal(1, 0.05, 5)
            I_ph *= perturb[0]
            I0 *= perturb[1]
            n *= perturb[2]
            Rs *= perturb[3]
            Rsh *= perturb[4]

            # 裁剪回合理范围
            I_ph = max(self.param_bounds[0][0], min(I_ph, self.param_bounds[0][1]))
            I0 = max(self.param_bounds[1][0], min(I0, self.param_bounds[1][1]))
            n = max(self.param_bounds[2][0], min(n, self.param_bounds[2][1]))
            Rs = max(self.param_bounds[3][0], min(Rs, self.param_bounds[3][1]))
            Rsh = max(self.param_bounds[4][0], min(Rsh, self.param_bounds[4][1]))

            population.append([I_ph, I0, n, Rs, Rsh])
        return population

    def _evaluate_population(self):
        loss_list = []
        # 新增：存储每个个体的单参数贡献度
        contrib_list = []
        for params in self.population:
            loss = objective_function(params, self.V, self.I_meas, self.I_min, self.I_max)
            loss_list.append(loss)
            # 计算并存储单参数贡献度
            contrib = calculate_single_param_contrib(
                params, self.V, self.I_meas, self.I_min, self.I_max,
                self.param_bounds, self.DELTA
            )
            contrib_list.append(contrib)

        # 过滤无效损失
        valid_loss_mask = np.isfinite(loss_list) & (np.array(loss_list) < 1e9)
        if not np.any(valid_loss_mask):
            return self.population[0], 1e10, 1e10, {}  # 返回空贡献度

        valid_losses = np.array(loss_list)[valid_loss_mask]
        valid_pop = np.array(self.population)[valid_loss_mask]
        valid_contribs = np.array(contrib_list)[valid_loss_mask]
        best_idx = np.argmin(valid_losses)

        return valid_pop[best_idx].tolist(), valid_losses[best_idx], np.mean(valid_losses), valid_contribs[best_idx]

    def _smart_adjust_params(self, params, contrib):
        """
        基于单参数贡献度智能调整参数，替代随机调整
        :param params: 待调整参数列表
        :param contrib: 该参数的贡献度字典
        :return: 调整后的参数列表
        """
        adjusted_params = params.copy()
        param_dict = dict(zip(self.PARAMS_NAME, adjusted_params))

        for idx, param_name in enumerate(self.PARAMS_NAME):
            c = contrib[param_name]
            # 1. 超出范围先拉回
            if not c["in_bound"]:
                low, high = self.param_bounds[idx]
                param_dict[param_name] = np.clip(param_dict[param_name], low, high)
                adjusted_params[idx] = param_dict[param_name]
                continue

            # 2. 贡献度为负（参数差）：往误差更小的方向调整
            if c["contribution"] < 0:
                if c["loss_up"] < c["loss_down"]:
                    param_dict[param_name] += c["delta"]
                else:
                    param_dict[param_name] -= c["delta"]
                adjusted_params[idx] = param_dict[param_name]

            # 3. 贡献度为正（参数优）：小幅扰动避免局部最优
            else:
                adjusted_params[idx] += np.random.uniform(-c["delta"] / 2, c["delta"] / 2)

        # 最终裁剪到范围
        adjusted_params = self._clip_params(adjusted_params)
        return adjusted_params

    def _teaching_phase(self, best_params, best_contrib, explore_ratio):
        """修改教学阶段：加入单参数贡献度指导"""
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
                # 探索阶段：用单参数贡献度指导探索，而非随机
                params_contrib = calculate_single_param_contrib(
                    params, self.V, self.I_meas, self.I_min, self.I_max,
                    self.param_bounds, self.DELTA
                )
                new_params = self._smart_adjust_params(params, params_contrib)
            else:
                # 利用阶段：自适应向老师学习 + 贡献度微调
                new_params = (params_np + np.random.random(5) * adaptive_step * (teacher - TF * mean_params)).tolist()
                # 用贡献度微调老师方向的参数
                new_params = self._smart_adjust_params(new_params, best_contrib)

            new_params = self._clip_params(new_params)
            new_population.append(new_params)

        return new_population

    def _learning_phase(self):
        """修改学习阶段：加入单参数贡献度指导"""
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
                # 用j的贡献度指导调整
                contrib_j = calculate_single_param_contrib(
                    params_j.tolist(), self.V, self.I_meas, self.I_min, self.I_max,
                    self.param_bounds, self.DELTA
                )
                new_params = self._smart_adjust_params(new_params, contrib_j)
            else:
                new_params = (params_j + np.random.random(5) * adaptive_step * (params_i - params_j)).tolist()
                # 用i的贡献度指导调整
                contrib_i = calculate_single_param_contrib(
                    params_i.tolist(), self.V, self.I_meas, self.I_min, self.I_max,
                    self.param_bounds, self.DELTA
                )
                new_params = self._smart_adjust_params(new_params, contrib_i)

            new_params = self._clip_params(new_params)
            new_population.append(new_params)

        return new_population

    def _clip_params(self, params):
        clipped = []
        for param, (low, high) in zip(params, self.param_bounds):
            clipped.append(max(low, min(param, high)))
        return clipped

    # def train(self):
    #     # 记录最优参数，避免迭代中丢失
    #     global_best_loss = 1e10
    #     global_best_params = None
    #     global_best_contrib = None
    #
    #     for iter in range(self.max_iter):
    #         # 新增：获取最优参数的贡献度
    #         best_params, best_loss, avg_loss, best_contrib = self._evaluate_population()
    #
    #         # 更新全局最优（显式拷贝，避免引用问题）
    #         if best_loss < global_best_loss:
    #             global_best_loss = best_loss
    #             global_best_params = best_params.copy()
    #             global_best_contrib = best_contrib  # 保存最优参数的贡献度
    #
    #         current_state = best_loss
    #
    #         # RL选择探索比例（动作数增加到5，更多选择）
    #         action = self.rl.choose_action(current_state)
    #         explore_ratio = [0.1, 0.3, 0.5, 0.7, 0.9][action]
    #
    #         # 修改：传入最优贡献度指导教学阶段
    #         self.population = self._teaching_phase(best_params, best_contrib, explore_ratio)
    #         self.population = self._learning_phase()
    #
    #         new_best_params, new_best_loss, new_avg_loss, _ = self._evaluate_population()
    #
    #         # 重构奖励函数：奖励与损失负相关，且放大差异
    #         reward = 10000 / (new_best_loss + 1e-8)
    #         next_state = new_best_loss
    #         self.rl.update_q(current_state, action, reward, next_state)
    #
    #         if (iter + 1) % 100 == 0:
    #             print(f"迭代{iter + 1:4d} | 全局最优损失: {global_best_loss:.2e} | 当前最优损失: {new_best_loss:.2e}")
    #             # 新增：打印最优参数的单参数贡献度（调试用）
    #             if global_best_contrib:
    #                 print("  单参数贡献度：", end="")
    #                 for name in self.PARAMS_NAME:
    #                     print(f"{name}: {global_best_contrib[name]['contribution']:.2e} ", end="")
    #                 print()
    #
    #         # 更宽松的收敛条件
    #         if global_best_loss < 1e-2:
    #             print(f"提前收敛！迭代{iter + 1}次，全局最优损失={global_best_loss:.2e}")
    #             break
    #
    #     # 最终用全局最优参数计算
    #     final_I_pred = solar_cell_model(self.V, global_best_params, self.I_min, self.I_max)
    #     return global_best_params, global_best_loss, final_I_pred


    def train(self):
        # 记录最优参数，避免迭代中丢失
        global_best_loss = 1e10
        global_best_params = None
        global_best_contrib = None
        # 新增：初始化损失/指标历史，用于趋势监控
        loss_history = []
        mae_history = []
        r2_history = []

        for iter in range(self.max_iter):
            # 新增：获取最优参数的贡献度
            best_params, best_loss, avg_loss, best_contrib = self._evaluate_population()

            # 更新全局最优（显式拷贝，避免引用问题）
            if best_loss < global_best_loss:
                global_best_loss = best_loss
                global_best_params = best_params.copy()
                global_best_contrib = best_contrib  # 保存最优参数的贡献度

                # 新增：计算并记录核心拟合指标（MAE/RMSE/R²）
                if global_best_params is not None:
                    # 用最优参数计算模拟电流
                    I_sim = solar_cell_model(self.V, global_best_params, self.I_min, self.I_max)
                    # 过滤有效数据
                    valid_mask = (self.I_meas > 1e-10) & np.isfinite(self.I_meas) & np.isfinite(I_sim)
                    if np.any(valid_mask):
                        I_meas_v = self.I_meas[valid_mask]
                        I_sim_v = I_sim[valid_mask]
                        # 计算MAE（平均绝对误差）
                        mae = np.mean(np.abs(I_sim_v - I_meas_v))
                        # 计算RMSE（均方根误差）
                        rmse = np.sqrt(np.mean((I_sim_v - I_meas_v) ** 2))
                        # 计算R²（决定系数）
                        ss_res = np.sum((I_sim_v - I_meas_v) ** 2)
                        ss_tot = np.sum((I_meas_v - np.mean(I_meas_v)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

                        # 记录指标
                        mae_history.append(mae)
                        r2_history.append(r2)

            # 新增：记录损失趋势
            loss_history.append(global_best_loss)
            current_state = best_loss

            # RL选择探索比例（动作数增加到5，更多选择）
            action = self.rl.choose_action(current_state)
            explore_ratio = [0.1, 0.3, 0.5, 0.7, 0.9][action]

            # 修改：传入最优贡献度指导教学阶段
            self.population = self._teaching_phase(best_params, best_contrib, explore_ratio)
            self.population = self._learning_phase()

            new_best_params, new_best_loss, new_avg_loss, _ = self._evaluate_population()

            # 重构奖励函数：奖励与损失负相关，且放大差异
            reward = 10000 / (new_best_loss + 1e-8)
            next_state = new_best_loss
            self.rl.update_q(current_state, action, reward, next_state)

            if (iter + 1) % 100 == 0:
                print(f"迭代{iter + 1:4d} | 全局最优损失: {global_best_loss:.2e} | 当前最优损失: {new_best_loss:.2e}")

                # 新增：打印核心拟合指标（直观判断效果）
                if len(mae_history) > 0 and len(r2_history) > 0:
                    print(f"        拟合指标 | MAE: {mae_history[-1]:.4f} A | RMSE: {rmse:.4f} A | R²: {r2:.4f}")

                # 新增：打印最优参数的单参数贡献度（调试用）
                if global_best_contrib:
                    print("  单参数贡献度：", end="")
                    for name in self.PARAMS_NAME:
                        print(f"{name}: {global_best_contrib[name]['contribution']:.2e} ", end="")
                    print()

                # 新增：监控损失下降趋势（判断是否收敛）
                if len(loss_history) > 100:
                    recent_loss = loss_history[-100:]
                    loss_drop_rate = (recent_loss[0] - recent_loss[-1]) / recent_loss[0]
                    if loss_drop_rate < 0.01:  # 近100次迭代损失下降<1%，说明收敛
                        print(f"⚠️  警告：损失下降停滞（下降率{loss_drop_rate:.2%}），拟合效果已稳定")

            # 更宽松的收敛条件
            if global_best_loss < 1e-2:
                print(f"提前收敛！迭代{iter + 1}次，全局最优损失={global_best_loss:.2e}")
                break

        # 新增：迭代结束后，打印参数物理合理性校验（快速排错）
        print("\n===== 最优参数合理性校验 =====")
        params_check = {
            "I_ph": (global_best_params[0], 10, 20),
            "I0": (global_best_params[1], 1e-12, 1e-7),
            "n": (global_best_params[2], 1.0, 2.0),
            "Rs": (global_best_params[3], 0.01, 1.0),
            "Rsh": (global_best_params[4], 100, 1000)
        }
        for name, (val, low, high) in params_check.items():
            if low <= val <= high:
                print(f"{name}: {val:.2e} ✅ (合理范围：{low:.2e}~{high:.2e})")
            else:
                print(f"{name}: {val:.2e} ❌ (超出范围：{low:.2e}~{high:.2e})")

        # 新增：迭代结束后，打印最终拟合指标总结
        print("\n===== 最终拟合效果总结 =====")
        final_I_sim = solar_cell_model(self.V, global_best_params, self.I_min, self.I_max)
        valid_mask = (self.I_meas > 1e-10) & np.isfinite(self.I_meas) & np.isfinite(final_I_sim)
        if np.any(valid_mask):
            I_meas_v = self.I_meas[valid_mask]
            I_sim_v = final_I_sim[valid_mask]
            final_mae = np.mean(np.abs(I_sim_v - I_meas_v))
            final_rmse = np.sqrt(np.mean((I_sim_v - I_meas_v) ** 2))
            final_ss_res = np.sum((I_sim_v - I_meas_v) ** 2)
            final_ss_tot = np.sum((I_meas_v - np.mean(I_meas_v)) ** 2)
            final_r2 = 1 - (final_ss_res / final_ss_tot) if final_ss_tot != 0 else 0.0
            print(f"MAE（平均电流偏差）：{final_mae:.4f} A (越小越好，<0.1为优)")
            print(f"RMSE（均方根误差）：{final_rmse:.4f} A (越小越好，<0.2为优)")
            print(f"R²（拟合程度）：{final_r2:.4f} (越接近1越好，>0.95为优)")
        else:
            print("无有效拟合数据！")

        # 最终用全局最优参数计算
        final_I_pred = solar_cell_model(self.V, global_best_params, self.I_min, self.I_max)
        return global_best_params, global_best_loss, final_I_pred

# ---------------------- 7. 主程序（调整RL初始化） ----------------------
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
        pop_size=100,
        max_iter=2000
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
        plt.title('太阳能电池模型拟合结果（单参数贡献度指导RLRTLBO）', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("未安装matplotlib，跳过可视化")