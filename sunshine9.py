import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_excel_and_preprocess(excel_path):
    df = pd.read_excel(excel_path, header=None, usecols=[0, 1], skiprows=1)
    V_original = df.iloc[:, 0].astype(float).values
    I_original = df.iloc[:, 1].astype(float).values
    valid_mask = np.isfinite(V_original) & np.isfinite(I_original)
    V_processed = V_original[valid_mask]
    I_meas_processed = I_original[valid_mask]
    return V_processed, I_meas_processed


def solar_cell_model(V_data):
    """使用导师参数计算理论电流"""
    I_ph = 4.244661601268746
    I0 = 1.9507404859873916e-46
    n = 1.142872079727331
    Rs = 0.06669084370651734
    Rsh = 49.93330915629348
    Vt = 0.026

    I_theory = np.zeros_like(V_data)

    for i, v in enumerate(V_data):
        # 简单迭代求解
        I = I_ph  # 初始值

        for _ in range(10):
            exp_arg = (v + I * Rs) / (n * Vt)
            #exp_arg = np.clip(exp_arg, -20, 20)
            exp_term = np.exp(exp_arg) - 1

            I_new = I_ph - I0 * exp_term - (v + I * Rs) / Rsh
            I_new = max(I_new, 0)  # 确保电流非负

            # 检查收敛
            if abs(I_new - I) < 1e-12:
                I = I_new
                break

            I = I_new

        I_theory[i] = I

    return I_theory


def main():
    # 数据文件路径
    excel_path = r"C:\Users\18372\PycharmProjects\pythonProject1\1.xls"

    try:
        # 加载数据
        print("正在加载数据...")
        V_data, I_data = load_excel_and_preprocess(excel_path)

        # 数据排序（按电压升序）
        sort_idx = np.argsort(V_data)
        V_data = V_data[sort_idx]
        I_data = I_data[sort_idx]

        print(f"数据加载成功！")
        print(f"数据点数: {len(V_data)}")

        # 计算理论电流
        print("计算理论电流...")
        I_theory = solar_cell_model(V_data)

        # 计算误差
        error_percentage = np.abs(I_data - I_theory) / np.max(I_data) * 100
        avg_error = np.mean(error_percentage)

        print(f"\n拟合结果:")
        print(f"平均相对误差: {avg_error:.2f}%")
        print(f"最大相对误差: {np.max(error_percentage):.2f}%")

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # 绘制简单对比图
        plt.figure(figsize=(10, 6))

        # 绘制实际数据点
        plt.scatter(V_data, I_data, c='blue', s=30, label='实测数据', alpha=0.7, edgecolors='k')

        # 绘制理论曲线
        plt.plot(V_data, I_theory, 'r-', linewidth=2, label='理论拟合')

        plt.xlabel('电压 (V)', fontsize=12)
        plt.ylabel('电流 (A)', fontsize=12)
        plt.title('太阳能电池I-V特性拟合', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        # 添加误差信息
        plt.text(0.02, 0.98, f'平均误差: {avg_error:.2f}%',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"错误: 找不到文件 {excel_path}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()