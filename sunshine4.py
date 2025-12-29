import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 改进单二极管模型：增加迭代次数，避免提前收敛
def your_pv_model(params, V_total):
    I_ph, I_0, n, R_s, R_sh, N = params
    Vt_single = 0.02585  # 室温下热电压
    Vt_total = N * Vt_single
    I = []
    for v in V_total:
        I_i = I_ph  # 初始电流设为光生电流
        # 增加迭代次数到300，提高收敛精度
        for _ in range(300):
            exp_arg = (v + I_i * R_s) / (n * Vt_total)
            # 防止指数溢出（限制exp_arg最大值）
            exp_arg = min(exp_arg, 40)
            exp_term = np.exp(exp_arg)
            I_new = I_ph - I_0 * (exp_term - 1) - (v + I_i * R_s) / R_sh
            I_new = max(I_new, 0)  # 电流不能为负
            if abs(I_new - I_i) < 1e-8:  # 提高收敛阈值
                break
            I_i = I_new
        I.append(I_i)
    return np.array(I)

# 读取数据
data_path = r"C:\Users\18372\PycharmProjects\pythonProject1\5.xls"
df = pd.read_excel(data_path, header=2)
V_data = df.iloc[:, 0].values
I_data = df.iloc[:, 1].values

# 改进损失函数：按电压分段加权（前半段+拐点+后半段）
def loss(params):
    I_pred = your_pv_model(params, V_data)
    # 分段权重：
    # 1. 前半段（V<80）：权重5（拟合平台区）
    # 2. 拐点区（80≤V≤90）：权重10（拟合曲线转折）
    # 3. 后半段（V>90）：权重8（拟合快速下降区）
    weight = np.where(V_data < 80, 5,
                np.where(V_data <= 90, 10, 8))
    return np.mean(weight * (I_pred - I_data)**2)

# 优化初始参数和边界（更贴近实际光伏组件参数）
initial_params = [12.8, 5e-13, 1.65, 4.2, 65, 60]
res = minimize(
    loss,
    initial_params,
    bounds=[
        (12.5, 13.2),   # I_ph（光生电流，更窄范围）
        (1e-13, 1e-12), # I_0（反向饱和电流，更窄范围）
        (1.6, 1.7),     # n（理想因子）
        (4.0, 4.5),     # R_s（串联电阻）
        (60, 70),       # R_sh（并联电阻）
        (58, 62)        # N（串联电池片数）
    ],
    tol=1e-9,  # 提高优化精度
    options={"maxiter": 1000}  # 增加最大迭代次数
)
I_ph_opt, I_0_opt, n_opt, R_s_opt, R_sh_opt, N_opt = res.x

# 计算拟合曲线
I_fit = your_pv_model([I_ph_opt, I_0_opt, n_opt, R_s_opt, R_sh_opt, N_opt], V_data)

# 绘图（增加残差子图，观察拟合误差）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})

# I-V曲线
ax1.scatter(V_data, I_data, label="你的数据", color="blue", s=30)
ax1.plot(V_data, I_fit, label="优化拟合曲线", color="red", linewidth=2.5)
ax1.set_xlabel("总电压 V")
ax1.set_ylabel("电流 I (A)")
ax1.set_title("光伏组件I-V曲线（优化拟合）")
ax1.legend()
ax1.grid(alpha=0.3)

# 残差图（I_pred - I_data）
residual = I_fit - I_data
ax2.bar(V_data, residual, color="green", alpha=0.6)
ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
ax2.set_xlabel("总电压 V")
ax2.set_ylabel("拟合残差 (A)")
ax2.set_title("拟合残差分布")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("最终优化参数：")
print(f"I_ph={I_ph_opt:.3f}A, I_0={I_0_opt:.2e}A, n={n_opt:.3f}, R_s={R_s_opt:.3f}Ω, R_sh={R_sh_opt:.3f}Ω, N={int(N_opt)}")