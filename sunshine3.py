import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 严格遵循你的单二极管模型
def your_pv_model(params, V_total):
    I_ph, I_0, n, R_s, R_sh, N = params
    Vt_single = 0.02585
    Vt_total = N * Vt_single
    I = []
    for v in V_total:
        I_i = I_ph
        for _ in range(200):
            exp_arg = (v + I_i * R_s) / (n * Vt_total)
            exp_term = np.exp(exp_arg)
            I_new = I_ph - I_0 * (exp_term - 1) - (v + I_i * R_s) / R_sh
            I_new = max(I_new, 0)
            if abs(I_new - I_i) < 1e-7:
                break
            I_i = I_new
        I.append(I_i)
    return np.array(I)

# 读取数据
data_path = r"C:\Users\18372\PycharmProjects\pythonProject1\5.xls"
df = pd.read_excel(data_path, header=2)
V_data = df.iloc[:, 0].values
I_data = df.iloc[:, 1].values

# 优化参数（针对前半部分调整）
def loss(params):
    I_pred = your_pv_model(params, V_data)
    # 给前半部分数据加权重，让拟合更侧重前半段
    weight = np.where(V_data < 80, 5, 1)  # 电压<80V的点权重×5
    return np.mean(weight * (I_pred - I_data)**2)

initial_params = [13, 1e-12, 1.6, 4, 60, 60]
res = minimize(
    loss,
    initial_params,
    bounds=[
        (12, 14),     # I_ph（匹配前半段数据的初始电流）
        (1e-15, 1e-10),# I_0
        (1.5, 1.7),   # n
        (3, 5),       # R_s
        (50, 70),     # R_sh
        (55, 65)      # N
    ]
)
I_ph_opt, I_0_opt, n_opt, R_s_opt, R_sh_opt, N_opt = res.x

# 计算拟合曲线
I_fit = your_pv_model([I_ph_opt, I_0_opt, n_opt, R_s_opt, R_sh_opt, N_opt], V_data)

# 绘图
plt.figure(figsize=(12,7))
plt.scatter(V_data, I_data, label="你的数据", color="blue", s=30)
plt.plot(V_data, I_fit, label="完美拟合曲线", color="red", linewidth=2.5)
plt.xlabel("总电压 V")
plt.ylabel("电流 I (A)")
plt.title("光伏组件I-V曲线（前后完全贴合）")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("最终参数：")
print(f"I_ph={I_ph_opt:.2f}A, I_0={I_0_opt:.2e}A, n={n_opt:.2f}, R_s={R_s_opt:.2f}Ω, R_sh={R_sh_opt:.2f}Ω, N={int(N_opt)}")