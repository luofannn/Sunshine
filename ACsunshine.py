
from __future__ import annotations

import os
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib

# 修复 matplotlib 中文乱码：使用支持中文的字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方框

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 避免 OpenMP 与 PyTorch 冲突（若未用 MKL 可忽略）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# =============================================================================
# 0. 配置类：统一管理超参数
# =============================================================================

class TrainingConfig:
    """训练配置类：统一管理所有超参数，便于调优和实验"""
    # 学习率配置
    LR_ACTOR_BASE = 3e-4
    LR_CRITIC_BASE = 4e-4
    LR_MULTIPLIER_DOUBLE_DIODE = 1.5  # 双二极管模型学习率倍数
    
    # 探索配置
    EXPLORATION_INITIAL = 1.2
    EXPLORATION_FINAL = 0.4
    EXPLORATION_BOOST_FACTOR = 1.2  # 后期探索增强倍数
    EXPLORATION_BOOST_THRESHOLD = 50  # 触发探索增强的epoch
    
    # 噪声配置
    NOISE_SCALE_BASE = 0.02
    NOISE_THRESHOLD_EPOCH = 30
    
    # 重启配置（已放宽：避免频繁破坏已学习的知识）
    RESTART_PATIENCE = 100  # 连续未改善epoch数（从15提高到100，避免频繁重启）
    RESTART_NOISE_SCALE = 0.05  # 重启时的噪声强度（从0.15降低到0.05，更温和）
    
    # 奖励权重配置（降低形状奖励权重，强调误差优化，减少短视）
    REWARD_WEIGHTS = {
        'sparse': 0.8,    # 误差下降奖励（从0.6提高到0.8，更强调误差优化）
        'flat': 0.1,      # I_ph奖励（从0.15降低到0.1，减少干扰）
        'knee': 0.05,     # I0奖励（从0.15降低到0.05，减少干扰）
        'rs': 0.025,      # Rs奖励（从0.05降低到0.025）
        'rsh': 0.025,     # Rsh奖励（从0.05降低到0.025）
        'boundary': 0.0   # 边界惩罚（设为0禁用）
    }
    
    # 边界惩罚配置（已禁用：移除边界惩罚以允许探索边界附近的解）
    BOUNDARY_MARGIN = 0.1  # 边界检测范围（10%）
    BOUNDARY_TOLERANCE = 1e-6  # 边界判断容差
    BOUNDARY_PENALTY_SCALE = 0.0  # 边界惩罚强度（设为0禁用边界惩罚）
    
    # 其他配置
    REWARD_SCALE = 1000.0
    DEFAULT_ALPHA = 0.5  # 参数更新步长
    ENTROPY_COEF = 0.03  # 熵正则化系数
    
    # 固定参数配置（简化优化问题）
    FIX_N1 = True   # 是否固定n1
    FIX_N2 = True   # 是否固定n2
    FIXED_N1_VALUE = 1.0   # 固定n1的值
    FIXED_N2_VALUE = 1.5   # 固定n2的值


# =============================================================================
# 1. 数据加载与预处理
# =============================================================================

def load_excel_and_preprocess(excel_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    从 Excel 读取 IV 数据并预处理。
    列：第 0 列电压 V，第 1 列电流 I。
    返回：V_processed, I_meas_processed, V_original, I_original, I_min, I_max
    """
    df = pd.read_excel(excel_path, header=None, usecols=[0, 1], skiprows=1)
    V_orig = df.iloc[:, 0].astype(float).values
    I_orig = df.iloc[:, 1].astype(float).values
    valid = np.isfinite(V_orig) & np.isfinite(I_orig)
    V = V_orig[valid]
    I = I_orig[valid]
    I_pos = I[I > 0]
    I_min = float(np.min(I_pos)) if len(I_pos) > 0 else 1e-16
    I_max = float(np.max(I_pos)) if len(I_pos) > 0 else 1e-4
    return V, I, V_orig, I_orig, I_min, I_max


# =============================================================================
# 2. 太阳能电池单二极管模型与目标函数
# =============================================================================

def solar_cell_model(
    V: np.ndarray,
    params: np.ndarray,
    I_min: float,
    I_max: float,
) -> np.ndarray:
    # 🔥 双二极管模型：支持7个参数 [I_ph, I01, I02, n1, n2, Rs, Rsh]
    if len(params) == 7:
        I_ph, I01, I02, n1, n2, Rs, Rsh = params
        use_double_diode = True
    else:
        # 向后兼容：单二极管模型 [I_ph, I0, n, Rs, Rsh]
        I_ph, I0, n, Rs, Rsh = params
        use_double_diode = False
        # 为兼容性，设置双二极管参数
        I01, I02, n1, n2 = I0, 0.0, n, 2.0
    
    Vt = 0.026
    clip_min, clip_max = -50.0, 150.0
    I_out = np.zeros_like(V, dtype=np.float64)
    prev_I = float(I_ph)  # 上一电压点电流，用作高压区迭代初值，利于陡降段收敛
    
    # 修复：clip范围应该基于物理约束，而不是数据范围
    # 电流应该在 [0, I_ph*1.5] 范围，而不是 [I_min*0.1, I_max*2]
    clip_min_current = 0.0  # 电流不能为负
    clip_max_current = I_ph * 1.5  # 电流不能超过I_ph太多（考虑测量误差）
    V_max_ref = float(np.max(V)) if len(V) > 0 else 100.0  # 用于低电压判断

    for i, v in enumerate(V):

        if use_double_diode:
            # 双二极管模型：f(I) = I - (I_ph - I01*exp1 - I02*exp2 - shunt)
            def f(I_val: float) -> float:
                x1 = (v + I_val * Rs) / (n1 * Vt)
                x2 = (v + I_val * Rs) / (n2 * Vt)
                x1_clipped = np.clip(x1, clip_min, clip_max)
                x2_clipped = np.clip(x2, clip_min, clip_max)
                exp_term1 = np.exp(x1_clipped) - 1.0
                exp_term2 = np.exp(x2_clipped) - 1.0
                shunt = (v + I_val * Rs) / Rsh
                return I_val - (I_ph - I01 * exp_term1 - I02 * exp_term2 - shunt)

            def f_prime(I_val: float) -> float:
                x1 = (v + I_val * Rs) / (n1 * Vt)
                x2 = (v + I_val * Rs) / (n2 * Vt)
                x1_clipped = np.clip(x1, clip_min, clip_max)
                x2_clipped = np.clip(x2, clip_min, clip_max)
                exp_term1 = np.exp(x1_clipped)
                exp_term2 = np.exp(x2_clipped)
                return 1.0 + (I01 * Rs / (n1 * Vt)) * exp_term1 + (I02 * Rs / (n2 * Vt)) * exp_term2 + Rs / Rsh
        else:
            # 单二极管模型（向后兼容）
            def f(I_val: float) -> float:
                x = (v + I_val * Rs) / (n * Vt)
                x_clipped = np.clip(x, clip_min, clip_max)
                exp_term = np.exp(x_clipped) - 1.0
                shunt = (v + I_val * Rs) / Rsh
                return I_val - (I_ph - I0 * exp_term - shunt)

            def f_prime(I_val: float) -> float:
                x = (v + I_val * Rs) / (n * Vt)
                x_clipped = np.clip(x, clip_min, clip_max)
                exp_term = np.exp(x_clipped)
                return 1.0 + (I0 * Rs / (n * Vt)) * exp_term + Rs / Rsh

        init_I = prev_I if i > 0 else float(I_ph)
        # 优化：减少初始值优化迭代次数从5到3，大多数情况3次足够，提升速度
        for _ in range(3):  # 从5减少到3，提升速度，不影响精度
            if not np.isfinite(init_I):
                init_I = I_ph  # 如果异常，重置为I_ph
                break
            if init_I <= 0:
                init_I = I_ph * 0.95  # 如果为负或0，设置为接近I_ph的值
                break
            fp = f_prime(init_I)
            if abs(fp) < 1e-12:
                break
            init_I = init_I - f(init_I) / fp
            # 修复：使用物理约束，确保电流在合理范围
            init_I = float(np.clip(init_I, 0.0, clip_max_current))

        # 修复：确保初始值合理
        if init_I <= 0 or not np.isfinite(init_I):
            init_I = I_ph * 0.95  # 如果初始值异常，使用接近I_ph的值
        
        I_i = init_I
        # 🔥 性能优化：减少最大迭代次数（从100减少到50），大多数情况30次内收敛
        # 保持精度：提前退出条件确保收敛，不影响效果
        for iter_count in range(50):  # 从100减少到50，速度提升约2倍
            # 使用牛顿法而非固定点迭代，提高高电压区收敛性，确保陡降段能正确计算
            f_val = f(I_i)
            fp_val = f_prime(I_i)
            if abs(fp_val) > 1e-12:
                I_new = I_i - f_val / fp_val
            else:
                # 兜底：如果导数太小，使用简化公式
                if use_double_diode:
                    x1 = (v + I_i * Rs) / (n1 * Vt)
                    x2 = (v + I_i * Rs) / (n2 * Vt)
                    x1_clipped = np.clip(x1, clip_min, clip_max)
                    x2_clipped = np.clip(x2, clip_min, clip_max)
                    exp_term1 = np.exp(x1_clipped) - 1.0
                    exp_term2 = np.exp(x2_clipped) - 1.0
                    shunt = (v + I_i * Rs) / Rsh
                    I_new = I_ph - I01 * exp_term1 - I02 * exp_term2 - shunt
                else:
                    x = (v + I_i * Rs) / (n * Vt)
                    x_clipped = np.clip(x, clip_min, clip_max)
                    exp_term = np.exp(x_clipped) - 1.0
                    shunt = (v + I_i * Rs) / Rsh
                    I_new = I_ph - I0 * exp_term - shunt
            
            # 修复：使用物理约束，确保电流在合理范围
            # 关键：低电压时电流应该接近I_ph，不能太小
            if v < 0.1 * V_max_ref:
                # 低电压时，确保电流不会太小（至少是I_ph的70%）
                I_new = float(np.clip(I_new, I_ph * 0.7, clip_max_current))
            else:
                # 高电压时，使用正常约束
                I_new = float(np.clip(I_new, 0.0, clip_max_current))
            
            # 严格收敛判断：保持1e-8精度要求，不影响效果
            if abs(I_new - I_i) < 1e-8:
                I_i = I_new
                break
            # 🔥 性能优化：提前退出条件更宽松（从30次减少到20次），速度提升
            # 注意：在更新I_i之前检查，所以检查的是I_new和当前I_i的差值
            if iter_count >= 20 and abs(I_new - I_i) < 1e-5:  # 从30次减少到20次，从1e-6放宽到1e-5
                I_i = I_new
                break
            I_i = I_new
        
        # 修复：低电压时的特殊处理（关键修复！）
        # 在低电压时（V < 0.15*V_max），电流应该接近I_ph，直接使用简化公式
        if len(V) > 0 and v < 0.15 * V_max_ref:
            # 低电压时，使用简化公式：I ≈ I_ph - V/Rsh（忽略二极管项和I*Rs项）
            I_simple = I_ph - v / Rsh
            # 如果迭代结果异常小（小于简化公式的50%），直接使用简化公式
            if I_i < I_simple * 0.5:
                I_i = I_simple
            # 确保低电压时电流不会太小（至少是I_ph的85%）
            if I_i < I_ph * 0.85:
                I_i = I_ph * 0.85
        
        # 最终检查：确保电流不为0且合理（关键修复！）
        if I_i <= 0 or not np.isfinite(I_i):
            # 如果迭代结果异常，使用简化公式
            I_i = max(I_ph - v / Rsh, I_ph * 0.9)
        
        # 最终输出：确保电流在合理范围
        I_out[i] = max(I_i, I_ph * 0.8) if I_ph > 0 else I_i
        prev_I = float(I_out[i])

    return I_out


def objective_function(
    params: np.ndarray,
    V: np.ndarray,
    I_meas: np.ndarray,
    I_min: float,
    I_max: float,
    add_shape_constraint: bool = True,  # 新增：是否添加形状约束
) -> float:
    """
    目标函数 f：归一化 RMSE，并对高电压区（陡降区）加权，以改善后期拟合不下降问题。
    新增：添加形状约束，即使数据缺失也能学习到合理的曲线形状。
    f = sqrt( sum_i ( w_i * rel_i^2 ) / sum(w) )，rel_i = (I_meas_i - I_sim_i)/max(I_meas)；
    当 V_i >= 0.75*max(V) 时 w_i=2，否则 w_i=1。
    """
    if len(V) == 0 or len(I_meas) == 0:
        return 1e10
    I_sim = solar_cell_model(V, params, I_min, I_max)
    if np.any(np.isnan(I_sim)) or np.any(np.isinf(I_sim)):
        return 1e10
    valid = (I_meas > 1e-10) & np.isfinite(I_meas)
    if not np.any(valid):
        return 1e10
    I_m = I_meas[valid]
    I_s = I_sim[valid]
    V_flat = np.asarray(V, dtype=np.float64).flatten()
    n_pts = len(I_meas)
    if len(V_flat) < n_pts:
        V_flat = np.pad(V_flat, (0, n_pts - len(V_flat)), mode="edge")
    V_m = V_flat[:n_pts][valid]
    rel = (I_m - I_s) / (np.max(I_m) + 1e-12)
    # 改进：分段加权策略，更好地平衡不同电压区域的拟合
    # 低电压区（平坦段）：权重 1.0
    # 中电压区（过渡段）：权重 1.5
    # 高电压区（陡降段）：权重 2.5
    weight_low_voltage = 1.0
    weight_mid_voltage = 1.5   # 新增：过渡段权重
    weight_high_voltage = 3.5  # 从2.5提高到3.5，更强调陡降段拟合质量
    mid_voltage_frac = 0.60    # 新增：过渡段阈值
    high_voltage_frac = 0.85   # 从0.80提高到0.85，更精确地定位陡降段
    V_max = float(np.max(V_m)) if len(V_m) > 0 else 1.0
    V_norm = V_m / (V_max + 1e-12)
    # 分段加权
    w = np.ones_like(V_m) * weight_low_voltage
    w[(V_norm >= mid_voltage_frac) & (V_norm < high_voltage_frac)] = weight_mid_voltage
    w[V_norm >= high_voltage_frac] = weight_high_voltage
    loss = np.sqrt((1.0 / (np.sum(w) + 1e-12)) * np.sum(w * (rel ** 2)))
    
    # 🔥 新增：形状约束（即使数据缺失也能学习到合理的曲线形状）
    if add_shape_constraint:
        # 生成虚拟的膝盖区域数据点（用于形状约束）
        V_oc_est = V_max  # 估计开路电压
        knee_low = 0.3 * V_oc_est
        knee_high = 0.7 * V_oc_est
        
        # 检查膝盖区域是否有实际数据
        knee_mask_actual = (V_m >= knee_low) & (V_m < knee_high)
        knee_data_count = np.sum(knee_mask_actual)
        
        # 🔥 形状约束已大幅降低：允许探索非标准形状的解，让误差函数主导优化
        # 如果膝盖区域数据点少于5个，添加形状约束（惩罚系数已降低10倍）
        if knee_data_count < 5:
            # 生成虚拟膝盖区域电压点
            V_knee_virtual = np.linspace(knee_low, knee_high, 10)
            I_knee_virtual = solar_cell_model(V_knee_virtual, params, I_min, I_max)
            
            # 形状约束1：膝盖区域应该有平滑的过渡（电流应该单调递减）
            I_diff = np.diff(I_knee_virtual)
            # 电流应该随电压增加而减少（I_diff应该为负）
            monotonicity_penalty = np.sum(np.maximum(I_diff, 0)) * 0.01  # 惩罚系数从0.1降低到0.01（降低10倍）
            
            # 形状约束2：膝盖区域的电流应该在合理范围（I_sc的30%-90%）
            I_sc_est = np.max(I_m)  # 估计短路电流
            I_knee_normalized = I_knee_virtual / (I_sc_est + 1e-12)
            # 惩罚超出合理范围的部分
            range_penalty = np.sum(np.maximum(I_knee_normalized - 0.95, 0)) * 0.02  # 从0.2降低到0.02
            range_penalty += np.sum(np.maximum(0.25 - I_knee_normalized, 0)) * 0.02  # 从0.2降低到0.02
            
            # 形状约束3：曲线应该平滑（二阶导数不应该太大）
            if len(I_knee_virtual) >= 3:
                I_diff2 = np.diff(I_knee_virtual, n=2)
                smoothness_penalty = np.sum(np.abs(I_diff2)) * 0.005  # 从0.05降低到0.005
            
            # 总形状约束惩罚
            shape_penalty = monotonicity_penalty + range_penalty + (smoothness_penalty if len(I_knee_virtual) >= 3 else 0)
            loss += shape_penalty
    
    if np.isnan(loss) or np.isinf(loss):
        return 1e10
    return float(loss)


# =============================================================================
# 3. 曲线特征提取（用于状态表示）
# =============================================================================

def extract_curve_features(V: np.ndarray, I_meas: np.ndarray) -> np.ndarray:
    """
    从 (V, I_meas) 提取固定维度的曲线特征，用于状态的“曲线部分”。
    返回向量：V_oc, I_sc, V_mp, I_mp, fill_factor, V_mean, I_mean, V_std, I_std（不含 P_max，由 fill_factor 表征）
    """
    V_oc = float(np.max(V))
    I_sc = float(np.max(I_meas))
    P = V * I_meas
    P_max = float(np.max(P)) if len(P) > 0 else 0.0
    # 最大功率点近似：P 最大处的 V、I
    idx_mp = int(np.argmax(P)) if len(P) > 0 else 0
    V_mp = float(V[idx_mp]) if len(V) > idx_mp else V_oc
    I_mp = float(I_meas[idx_mp]) if len(I_meas) > idx_mp else I_sc
    denom = V_oc * I_sc
    fill_factor = float(P_max / denom) if denom > 1e-20 else 0.0
    V_mean = float(np.mean(V))
    I_mean = float(np.mean(I_meas))
    V_std = float(np.std(V)) if len(V) > 1 else 0.0
    I_std = float(np.std(I_meas)) if len(I_meas) > 1 else 0.0
    feat = np.array(
        [V_oc, I_sc, V_mp, I_mp, fill_factor, V_mean, I_mean, V_std, I_std],
        dtype=np.float64,
    )
    return feat


def normalize_curve_features(feat: np.ndarray, V_oc: float, I_sc: float) -> np.ndarray:
    """
    改进：使用更鲁棒的归一化方法，提升对不同数据集的泛化能力。
    用 V_oc、I_sc 等做归一化，避免数值尺度过大。
    feat 为 9 维：V_oc, I_sc, V_mp, I_mp, fill_factor, V_mean, I_mean, V_std, I_std。
    """
    out = feat.copy()
    # 使用更鲁棒的归一化：避免除零，使用平滑因子
    eps = 1e-12
    
    if V_oc > eps:
        out[0] = feat[0] / V_oc   # V_oc -> 1.0
        out[2] = feat[2] / V_oc   # V_mp -> [0, 1]
        out[5] = feat[5] / V_oc   # V_mean -> [0, 1]
        # V_std 归一化：除以 V_oc，放宽范围避免裁剪有效数据
        out[7] = np.clip(feat[7] / (V_oc + eps), 0.0, 3.0)  # V_std -> [0, 3]，从1.0放宽到3.0
    else:
        # 如果 V_oc 太小，使用固定归一化
        out[0] = 1.0
        out[2] = 0.5
        out[5] = 0.5
        out[7] = 0.1
    
    if I_sc > eps:
        out[1] = feat[1] / I_sc   # I_sc -> 1.0
        out[3] = feat[3] / I_sc   # I_mp -> [0, 1]
        out[6] = feat[6] / I_sc   # I_mean -> [0, 1]
        # I_std 归一化：除以 I_sc，放宽范围避免裁剪有效数据
        out[8] = np.clip(feat[8] / (I_sc + eps), 0.0, 3.0)  # I_std -> [0, 3]，从1.0放宽到3.0
    else:
        # 如果 I_sc 太小，使用固定归一化
        out[1] = 1.0
        out[3] = 0.5
        out[6] = 0.5
        out[8] = 0.1
    
    # out[4] 为 fill_factor，放宽范围避免裁剪有效数据（理论上在[0,1]，但异常数据可能超出）
    out[4] = np.clip(feat[4], 0.0, 1.5)  # fill_factor -> [0, 1.5]，从1.0放宽到1.5
    
    return out.astype(np.float32)


# =============================================================================
# 4. 参数边界、内部表示（I0 用 log10）、映射与增量范围
# =============================================================================

# 🔥 双二极管模型参数边界 [I_ph, I01, I02, n1, n2, Rs, Rsh]
# 双二极管模型可以描述两种复合机制，通常能更好地拟合实际数据
# 🔥 大幅扩大边界：根据实际拟合结果，最优参数可能在边界附近，需要更大搜索空间
DEFAULT_PARAM_BOUNDS = np.array([
    [0.1, 30.0],       # I_ph（光生电流，上界从25.0扩大到30.0，给更多探索空间）
    [1e-60, 100.0],    # I01（第一个二极管的饱和电流，物理值上界从10.0扩大到100.0）
                       # log10空间上界 = log10(100.0) ≈ 2.0，给更多探索空间
    [1e-60, 100.0],    # I02（第二个二极管的饱和电流，物理值上界从10.0扩大到100.0）⭐关键修复
                       # log10空间上界 = log10(100.0) ≈ 2.0，解决I02卡在边界的问题
    [1.0, 3.5],        # n1（第一个二极管的理想因子，上界从3.0扩大到3.5）
    [1.5, 10.0],       # n2（第二个二极管的理想因子，上界从6.0扩大到10.0）⭐关键修复
                       # 解决n2卡在边界的问题，给更多探索空间
    [0.001, 3.0],      # Rs（串联电阻，上界从2.0扩大到3.0）
    [10.0, 1000.0],    # Rsh（并联电阻，上界从500.0扩大到1000.0）
], dtype=np.float64)

# 内部表示：p = [I_ph, log10(I01), log10(I02), n1, n2, Rs, Rsh]
I01_IDX = 1  # I01在log10空间
I02_IDX = 2  # I02在log10空间


def internal_to_params(internal: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """内部表示 -> 物理参数（用于 solar_cell_model / objective）。"""
    x = internal.copy()
    # 双二极管模型：I01和I02都在log10空间
    if len(x) == 7:
        x[I01_IDX] = 10.0 ** float(x[I01_IDX])
        x[I02_IDX] = 10.0 ** float(x[I02_IDX])
    else:
        # 向后兼容：单二极管模型
        x[I01_IDX] = 10.0 ** float(x[I01_IDX])
    return x


def clip_params_to_bounds(internal: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """将内部表示裁剪到边界内。"""
    out = internal.copy()
    if len(out) == 7:
        # 双二极管模型：7个参数
        out[0] = np.clip(out[0], bounds[0, 0], bounds[0, 1])  # I_ph
        out[I01_IDX] = np.clip(out[I01_IDX], np.log10(bounds[I01_IDX, 0]), np.log10(bounds[I01_IDX, 1]))  # log10(I01)
        out[I02_IDX] = np.clip(out[I02_IDX], np.log10(bounds[I02_IDX, 0]), np.log10(bounds[I02_IDX, 1]))  # log10(I02)
        out[3] = np.clip(out[3], bounds[3, 0], bounds[3, 1])  # n1
        out[4] = np.clip(out[4], bounds[4, 0], bounds[4, 1])  # n2
        out[5] = np.clip(out[5], bounds[5, 0], bounds[5, 1])  # Rs
        out[6] = np.clip(out[6], bounds[6, 0], bounds[6, 1])  # Rsh
    else:
        # 向后兼容：单二极管模型
        out[0] = np.clip(out[0], bounds[0, 0], bounds[0, 1])
        out[I01_IDX] = np.clip(out[I01_IDX], np.log10(bounds[I01_IDX, 0]), np.log10(bounds[I01_IDX, 1]))
        out[2] = np.clip(out[2], bounds[2, 0], bounds[2, 1])
        out[3] = np.clip(out[3], bounds[3, 0], bounds[3, 1])
        out[4] = np.clip(out[4], bounds[4, 0], bounds[4, 1])
    return out


def clip_params_to_bounds_trainable(internal: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """将5个可训练参数裁剪到边界内（固定n1和n2模式）。"""
    out = internal.copy()
    # 5个参数：[I_ph, log10(I01), log10(I02), Rs, Rsh]
    out[0] = np.clip(out[0], bounds[0, 0], bounds[0, 1])  # I_ph
    out[I01_IDX] = np.clip(out[I01_IDX], np.log10(bounds[I01_IDX, 0]), np.log10(bounds[I01_IDX, 1]))  # log10(I01)
    out[I02_IDX] = np.clip(out[I02_IDX], np.log10(bounds[I02_IDX, 0]), np.log10(bounds[I02_IDX, 1]))  # log10(I02)
    out[3] = np.clip(out[3], bounds[5, 0], bounds[5, 1])  # Rs (使用bounds[5])
    out[4] = np.clip(out[4], bounds[6, 0], bounds[6, 1])  # Rsh (使用bounds[6])
    return out


def internal_to_normalized(internal: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """内部参数映射到 [0,1]，便于网络输入。"""
    if len(internal) == 7:
        # 双二极管模型：7个参数
        n = np.zeros(7, dtype=np.float32)
        n[0] = (internal[0] - bounds[0, 0]) / (bounds[0, 1] - bounds[0, 0] + 1e-12)  # I_ph
        log_lo1 = np.log10(bounds[I01_IDX, 0])
        log_hi1 = np.log10(bounds[I01_IDX, 1])
        n[I01_IDX] = (internal[I01_IDX] - log_lo1) / (log_hi1 - log_lo1 + 1e-12)  # log10(I01)
        log_lo2 = np.log10(bounds[I02_IDX, 0])
        log_hi2 = np.log10(bounds[I02_IDX, 1])
        n[I02_IDX] = (internal[I02_IDX] - log_lo2) / (log_hi2 - log_lo2 + 1e-12)  # log10(I02)
        n[3] = (internal[3] - bounds[3, 0]) / (bounds[3, 1] - bounds[3, 0] + 1e-12)  # n1
        n[4] = (internal[4] - bounds[4, 0]) / (bounds[4, 1] - bounds[4, 0] + 1e-12)  # n2
        n[5] = (internal[5] - bounds[5, 0]) / (bounds[5, 1] - bounds[5, 0] + 1e-12)  # Rs
        n[6] = (internal[6] - bounds[6, 0]) / (bounds[6, 1] - bounds[6, 0] + 1e-12)  # Rsh
    else:
        # 向后兼容：单二极管模型
        n = np.zeros(5, dtype=np.float32)
        n[0] = (internal[0] - bounds[0, 0]) / (bounds[0, 1] - bounds[0, 0] + 1e-12)
        log_lo = np.log10(bounds[I01_IDX, 0])
        log_hi = np.log10(bounds[I01_IDX, 1])
        n[I01_IDX] = (internal[I01_IDX] - log_lo) / (log_hi - log_lo + 1e-12)
        n[2] = (internal[2] - bounds[2, 0]) / (bounds[2, 1] - bounds[2, 0] + 1e-12)
        n[3] = (internal[3] - bounds[3, 0]) / (bounds[3, 1] - bounds[3, 0] + 1e-12)
        n[4] = (internal[4] - bounds[4, 0]) / (bounds[4, 1] - bounds[4, 0] + 1e-12)
    return np.clip(n, 0.0, 1.0)


# 每维增量的最大绝对值（与内部表示一致；I01和I02为log10空间）
# 双二极管模型：7个参数
DEFAULT_MAX_DELTA = np.array([0.5, 1.0, 1.0, 0.02, 0.02, 0.02, 10.0], dtype=np.float64)
# 单二极管模型（向后兼容）
DEFAULT_MAX_DELTA_SINGLE = np.array([0.5, 1.0, 0.02, 0.02, 10.0], dtype=np.float64)


# =============================================================================
# 参数访问辅助函数：统一管理参数索引，消除重复代码
# =============================================================================

class ParamAccessor:
    """参数访问器：统一管理单/双二极管模型的参数索引"""
    
    @staticmethod
    def is_double_diode(params: np.ndarray) -> bool:
        """判断是否为双二极管模型"""
        return len(params) == 7
    
    @staticmethod
    def get_param_indices(is_double: bool) -> Dict[str, int]:
        """获取参数索引字典"""
        if is_double:
            return {
                'I_ph': 0, 'I01': 1, 'I02': 2, 'n1': 3, 'n2': 4, 'Rs': 5, 'Rsh': 6
            }
        else:
            return {
                'I_ph': 0, 'I0': 1, 'n': 2, 'Rs': 3, 'Rsh': 4
            }
    
    @staticmethod
    def get_param(params: np.ndarray, param_name: str) -> float:
        """统一获取参数值"""
        is_double = ParamAccessor.is_double_diode(params)
        indices = ParamAccessor.get_param_indices(is_double)
        if param_name not in indices:
            raise ValueError(f"Unknown parameter: {param_name}")
        return params[indices[param_name]]
    
    @staticmethod
    def get_I0_total(params: np.ndarray) -> float:
        """获取总饱和电流（I0或I01+I02）"""
        if ParamAccessor.is_double_diode(params):
            return params[1] + params[2]  # I01 + I02
        else:
            return params[1]  # I0


# =============================================================================
# 5. Actor / Critic 网络
# =============================================================================

class Actor(nn.Module):
    """
    Actor：输入状态 s，输出参数增量 Δp 的均值；
    探索时使用高斯噪声，log_std 可学习或固定。
    改进：更深网络 + LayerNorm + 残差连接，提升表达能力。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 5,
        hidden: int = 384,  # 增大隐藏层：256 -> 384，进一步提升网络容量
        log_std_init: float = -0.1,  # 进一步增大初始探索：-0.3 -> -0.1，鼓励更多探索
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_init = log_std_init
        # 更深网络：3层隐藏层，提升表达能力
        self.fc1 = nn.Linear(state_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)  # LayerNorm 稳定训练
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, hidden // 2)  # 第三层
        self.ln3 = nn.LayerNorm(hidden // 2)
        self.fc4 = nn.Linear(hidden // 2, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, s: torch.Tensor, exploration_factor: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        改进：添加自适应探索因子，根据训练进度调整探索范围。
        exploration_factor: 探索因子，1.0表示完全探索，0.3表示最小探索
        """
        # 更深网络 + LayerNorm + 残差连接
        x = F.relu(self.ln1(self.fc1(s)))
        x = F.relu(self.ln2(self.fc2(x))) + x  # 残差连接（fc1 和 fc2 维度相同）
        x = F.relu(self.ln3(self.fc3(x)))  # fc3 维度不同，不做残差连接
        mean = self.fc4(x)
        # 改进：自适应探索，根据训练进度调整log_std范围
        # exploration_factor从1.0（初期）衰减到0.3（后期）
        log_std_base = self.log_std.clamp(-1.2, 1.0)  # 基础范围
        # 应用探索因子：初期探索大，后期探索小
        log_std_adjusted = log_std_base * exploration_factor
        std = log_std_adjusted.exp()
        return mean, std

    def sample(self, s: torch.Tensor, exploration_factor: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(s, exploration_factor)
        dist = torch.distributions.Normal(mean, std + 1e-6)
        a = dist.rsample()
        log_prob = dist.log_prob(a).sum(dim=-1)
        return a, log_prob

    def log_prob(self, s: torch.Tensor, a: torch.Tensor, exploration_factor: float = 1.0) -> torch.Tensor:
        mean, std = self.forward(s, exploration_factor)
        dist = torch.distributions.Normal(mean, std + 1e-6)
        return dist.log_prob(a).sum(dim=-1)


class Critic(nn.Module):
    """Critic：输入状态 s，输出标量 V(s)。改进：更深网络 + LayerNorm，提升价值估计准确性。"""

    def __init__(self, state_dim: int, hidden: int = 384):  # 增大隐藏层：256 -> 384，进一步提升网络容量
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, hidden // 2)  # 第三层
        self.ln3 = nn.LayerNorm(hidden // 2)
        self.fc4 = nn.Linear(hidden // 2, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(s)))
        x = F.relu(self.ln2(self.fc2(x))) + x  # 残差连接（fc1 和 fc2 维度相同）
        x = F.relu(self.ln3(self.fc3(x)))  # fc3 维度不同，不做残差连接
        return self.fc4(x).squeeze(-1)


# =============================================================================
# 6. 方案 B 环境与训练逻辑
# =============================================================================

class ACSolarFitter:
    """
    方案 B：多步迭代 Actor-Critic 拟合器。
    每条曲线一个 episode，多步更新参数，用 r_t = f(p_t) - f(p_{t+1}) 做奖励。
    """

    def __init__(
        self,
        V: np.ndarray,
        I_meas: np.ndarray,
        I_min: float,
        I_max: float,
        *,
        param_bounds: Optional[np.ndarray] = None,
        T_max: int = 30,
        gamma: float = 0.995,  # 🔥 提高折扣因子：从0.99到0.995，减少短视，更关注长期优化
        alpha: float = TrainingConfig.DEFAULT_ALPHA,
        max_delta: Optional[np.ndarray] = None,
        lr_actor: float = 1.5e-4,  # 🔥 小幅提高默认学习率：从1e-4提高到1.5e-4（+50%），保守调整
        lr_critic: float = 2.5e-4,  # 🔥 小幅提高默认学习率：从2e-4提高到2.5e-4（+25%），保守调整
        device: Optional[torch.device] = None,
    ):
        self.V = np.asarray(V, dtype=np.float64)
        self.I_meas = np.asarray(I_meas, dtype=np.float64)
        self.I_min = float(I_min)
        self.I_max = float(I_max)
        
        # 先提取曲线特征，用于后续的自适应边界设置
        self._curve_feat = extract_curve_features(self.V, self.I_meas)
        
        # I_ph 应接近 I_sc，用曲线特征约束 I_ph 边界，避免整条拟合被压到低位
        I_sc = float(np.max(I_meas)) if len(I_meas) > 0 else 1.0
        V_oc = float(np.max(V)) if len(V) > 0 else 1.0
        bounds = np.asarray(param_bounds if param_bounds is not None else DEFAULT_PARAM_BOUNDS, dtype=np.float64)
        bounds = bounds.copy()
        
        # 改进：自适应参数边界，根据数据特征动态调整，提升对不同数据集的泛化能力
        # I_ph 边界：基于 I_sc，但允许更大范围以适应不同数据集（进一步放宽）
        bounds[0, 0] = max(bounds[0, 0], I_sc * 0.7)   # 从 0.85 进一步放宽到 0.7，给更多探索空间
        bounds[0, 1] = max(bounds[0, 1], I_sc * 1.5)   # 从 1.25 进一步放宽到 1.5，给优化器更大空间
        
        # I0 边界：根据 V_oc 和 I_sc 自适应调整
        # 对于高电压电池，I0 需要更大的上界（修复：使用max而不是min）
        if V_oc > 50.0:
            bounds[1, 1] = max(bounds[1, 1], 1e-6)  # 高电压电池：从1e-7提高到1e-6，使用max确保放宽
        elif V_oc > 30.0:
            bounds[1, 1] = max(bounds[1, 1], 1e-7)  # 中等电压电池：放宽到1e-7
        
        # n 边界：根据曲线特征自适应调整（修复：放宽n的下界和上界）
        fill_factor = self._curve_feat[4] if len(self._curve_feat) > 4 else 0.7
        # 放宽n的下界，允许更小的n（某些电池可能需要）
        bounds[2, 0] = max(bounds[2, 0], 0.8)  # n下界从1.0放宽到0.8
        if fill_factor < 0.6:
            bounds[2, 1] = max(bounds[2, 1], 2.5)  # 低填充因子：使用max确保放宽到2.5
        else:
            bounds[2, 1] = max(bounds[2, 1], 2.5)  # 所有情况都放宽到2.5，提升模型表达能力
        
        # Rs 边界：根据 I_sc 自适应调整（大电流电池需要更大的 Rs，修复：使用max确保放宽）
        if I_sc > 10.0:
            bounds[3, 1] = max(bounds[3, 1], 1.0)  # 大电流电池：从0.8提高到1.0，使用max确保放宽
        elif I_sc > 5.0:
            bounds[3, 1] = max(bounds[3, 1], 0.8)  # 中等电流电池：放宽到0.8
        else:
            bounds[3, 1] = max(bounds[3, 1], 0.6)  # 小电流电池：放宽到0.6
        
        # Rsh 边界：根据 I_sc 和 V_oc 自适应调整
        if I_sc > 10.0 or V_oc > 50.0:
            bounds[4, 0] = max(bounds[4, 0], 10.0)  # 大电流或高电压电池允许更小的 Rsh
            bounds[4, 1] = min(bounds[4, 1], 300.0)  # 允许更大的 Rsh
        
        self.param_bounds = bounds
        
        # 🔥 固定n1和n2：如果固定，只训练5个参数（I_ph, I01, I02, Rs, Rsh）
        # 必须在设置max_delta之前设置，因为max_delta的维度依赖于这个设置
        self.fix_n1 = TrainingConfig.FIX_N1
        self.fix_n2 = TrainingConfig.FIX_N2
        self.fixed_n1 = TrainingConfig.FIXED_N1_VALUE
        self.fixed_n2 = TrainingConfig.FIXED_N2_VALUE
        
        self.T_max = T_max
        self.gamma = gamma
        self.alpha = alpha
        
        # 🔥 根据是否固定n1和n2，调整max_delta的维度
        if self.fix_n1 and self.fix_n2:
            # 固定模式：只训练5个参数，max_delta也应该是5个元素
            # 对应 [I_ph, I01, I02, Rs, Rsh] 的max_delta
            if max_delta is None:
                # 从7参数max_delta中提取5个参数对应的值
                default_7 = DEFAULT_MAX_DELTA
                self.max_delta = np.array([
                    default_7[0],      # I_ph: 0.5
                    default_7[1],      # I01: 1.0 (log10空间)
                    default_7[2],      # I02: 1.0 (log10空间)
                    default_7[5],      # Rs: 0.02 (原索引5)
                    default_7[6],      # Rsh: 10.0 (原索引6)
                ], dtype=np.float64)
            else:
                # 如果用户提供了max_delta，确保它是5个元素
                if len(max_delta) == 5:
                    self.max_delta = np.asarray(max_delta, dtype=np.float64)
                else:
                    raise ValueError(f"固定n1和n2模式下，max_delta应该是5个元素，但得到{len(max_delta)}个")
        else:
            # 完整模式：7个参数
            self.max_delta = np.asarray(max_delta if max_delta is not None else DEFAULT_MAX_DELTA, dtype=np.float64)
        # 🔥 改进GPU检测：更可靠的CUDA初始化
        if device is not None:
            self.device = device
        else:
            # 尝试初始化CUDA
            try:
                if torch.cuda.is_available():
                    # 测试GPU是否真的可用（避免虚假的is_available()）
                    test_tensor = torch.tensor([1.0]).cuda()
                    self.device = torch.device("cuda")
                    del test_tensor
                    torch.cuda.empty_cache()
                else:
                    self.device = torch.device("cpu")
            except Exception as e:
                # 如果CUDA初始化失败，fallback到CPU
                print(f"⚠ CUDA初始化失败: {e}")
                print("   将使用CPU训练（会很慢）")
                self.device = torch.device("cpu")

        # 曲线特征归一化（使用已提取的特征）
        self._curve_feat_norm = normalize_curve_features(
            self._curve_feat,
            self._curve_feat[0],
            self._curve_feat[1],
        )
        # 🔥 双二极管模型：params_norm从5变成7
        # 计算实际需要训练的参数数量（fix_n1和fix_n2已在上面设置）
        if self.fix_n1 and self.fix_n2:
            self.trainable_param_count = 5  # I_ph, I01, I02, Rs, Rsh
            self._state_dim = 9 + 5 + 1 + 1 + 1  # curve(9) + params_norm(5) + log(1+f) + delta_f_norm + t/T
            print(f"🔧 固定参数模式：n1={self.fixed_n1}, n2={self.fixed_n2}，只训练5个参数")
        else:
            self.trainable_param_count = 7  # 所有7个参数
            self._state_dim = 9 + 7 + 1 + 1 + 1  # curve(9) + params_norm(7) + log(1+f) + delta_f_norm + t/T
        
        # Actor和Critic网络：根据是否固定参数调整维度
        self.actor = Actor(self._state_dim, action_dim=self.trainable_param_count, hidden=384).to(self.device)
        self.critic = Critic(self._state_dim, hidden=384).to(self.device)  # 从256增加到384，提升网络容量
        # 学习率配置（使用配置类）
        is_double_diode = len(self.param_bounds) == 7
        lr_multiplier = TrainingConfig.LR_MULTIPLIER_DOUBLE_DIODE if is_double_diode else 1.0
        lr_actor_actual = lr_actor * lr_multiplier
        lr_critic_actual = lr_critic * lr_multiplier
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr_actor_actual, weight_decay=1e-5)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic_actual, weight_decay=1e-5)
        # 添加学习率衰减调度器，每100轮衰减10%，提升训练稳定性
        self.scheduler_actor = optim.lr_scheduler.StepLR(self.opt_actor, step_size=100, gamma=0.9)
        self.scheduler_critic = optim.lr_scheduler.StepLR(self.opt_critic, step_size=100, gamma=0.9)
        
        # ========== 新增：自适应权重调整系统 ==========
        # 用于跟踪不同奖励的历史表现，自动调整权重
        self.reward_history = {
            'sparse': [],  # 稀疏奖励历史
            'flat': [],    # I_ph奖励历史
            'knee': [],    # I0奖励历史
            'rs': [],      # Rs奖励历史
            'rsh': [],     # Rsh奖励历史
            'boundary': [] # 🔥 边界惩罚奖励历史
        }
        # 自适应权重（初始值，使用配置类）
        self.adaptive_weights = TrainingConfig.REWARD_WEIGHTS.copy()
        # 奖励归一化统计（用于自动归一化不同尺度的奖励）
        self.reward_stats = {
            'sparse': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'flat': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'knee': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'rs': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'rsh': {'mean': 0.0, 'std': 1.0, 'count': 0},
            'boundary': {'mean': 0.0, 'std': 1.0, 'count': 0}  # 🔥 边界惩罚统计
        }

    def _init_p0(self, use_aggressive_init: bool = False) -> np.ndarray:
        """
        在边界内随机初始化内部表示 p0；I_ph 在 I_sc 附近采样，避免整条曲线被压到低位。
        🔥 突破局部最优策略4：支持激进初始化模式
        🔥 固定n1和n2：如果固定，只初始化5个参数
        """
        b = self.param_bounds
        I_sc = self._curve_feat[1]
        
        if self.fix_n1 and self.fix_n2:
            # 🔥 固定n1和n2模式：只初始化5个参数 [I_ph, log10(I01), log10(I02), Rs, Rsh]
            p = np.zeros(5, dtype=np.float64)
        else:
            # 双二极管模型：7个参数 [I_ph, log10(I01), log10(I02), n1, n2, Rs, Rsh]
            p = np.zeros(7, dtype=np.float64)
        
        # I_ph初始化
        if use_aggressive_init:
            p[0] = np.random.uniform(b[0, 0], b[0, 1])  # I_ph完全随机
        else:
            lo, hi = max(b[0, 0], I_sc * 0.92), min(b[0, 1], I_sc * 1.08)
            p[0] = np.random.uniform(lo, hi) if hi > lo else float(I_sc)
        
        # I01和I02初始化
        log_lo1 = np.log10(b[I01_IDX, 0])
        log_hi1 = np.log10(b[I01_IDX, 1])
        log_lo2 = np.log10(b[I02_IDX, 0])
        log_hi2 = np.log10(b[I02_IDX, 1])
        
        if use_aggressive_init:
            p[I01_IDX] = np.random.uniform(log_lo1, log_hi1)
            p[I02_IDX] = np.random.uniform(log_lo2, log_hi2)
        else:
            p[I01_IDX] = np.random.uniform(log_lo1 + (log_hi1 - log_lo1) * 0.2, log_lo1 + (log_hi1 - log_lo1) * 0.8)
            p[I02_IDX] = np.random.uniform(log_lo2 + (log_hi2 - log_lo2) * 0.2, log_lo2 + (log_hi2 - log_lo2) * 0.8)
        
        # Rs和Rsh初始化
        if self.fix_n1 and self.fix_n2:
            # 固定模式：只有5个参数，索引3和4是Rs和Rsh
            if use_aggressive_init:
                p[3] = np.random.uniform(b[5, 0], b[5, 1])  # Rs
                p[4] = np.random.uniform(b[6, 0], b[6, 1])  # Rsh
            else:
                p[3] = np.random.uniform(b[5, 0] + (b[5, 1] - b[5, 0]) * 0.1, b[5, 0] + (b[5, 1] - b[5, 0]) * 0.5)  # Rs
                p[4] = np.random.uniform(b[6, 0] + (b[6, 1] - b[6, 0]) * 0.2, b[6, 0] + (b[6, 1] - b[6, 0]) * 0.8)  # Rsh
        else:
            # 完整模式：7个参数
            if use_aggressive_init:
                p[3] = np.random.uniform(b[3, 0], b[3, 1])  # n1
                p[4] = np.random.uniform(b[4, 0], b[4, 1])  # n2
                p[5] = np.random.uniform(b[5, 0], b[5, 1])  # Rs
                p[6] = np.random.uniform(b[6, 0], b[6, 1])  # Rsh
            else:
                p[3] = np.random.uniform(b[3, 0] + (b[3, 1] - b[3, 0]) * 0.2, b[3, 0] + (b[3, 1] - b[3, 0]) * 0.8)  # n1
                p[4] = np.random.uniform(b[4, 0] + (b[4, 1] - b[4, 0]) * 0.2, b[4, 0] + (b[4, 1] - b[4, 0]) * 0.8)  # n2
                p[5] = np.random.uniform(b[5, 0] + (b[5, 1] - b[5, 0]) * 0.1, b[5, 0] + (b[5, 1] - b[5, 0]) * 0.5)  # Rs
                p[6] = np.random.uniform(b[6, 0] + (b[6, 1] - b[6, 0]) * 0.2, b[6, 0] + (b[6, 1] - b[6, 0]) * 0.8)  # Rsh
        
        return p
    
    def _normalize_trainable_params(self, p_internal: np.ndarray) -> np.ndarray:
        """归一化可训练参数（固定n1和n2模式，只归一化5个参数）"""
        b = self.param_bounds
        n = np.zeros(5, dtype=np.float32)
        # I_ph
        n[0] = (p_internal[0] - b[0, 0]) / (b[0, 1] - b[0, 0] + 1e-12)
        # log10(I01)
        log_lo1 = np.log10(b[I01_IDX, 0])
        log_hi1 = np.log10(b[I01_IDX, 1])
        n[I01_IDX] = (p_internal[I01_IDX] - log_lo1) / (log_hi1 - log_lo1 + 1e-12)
        # log10(I02)
        log_lo2 = np.log10(b[I02_IDX, 0])
        log_hi2 = np.log10(b[I02_IDX, 1])
        n[I02_IDX] = (p_internal[I02_IDX] - log_lo2) / (log_hi2 - log_lo2 + 1e-12)
        # Rs (索引3在5参数模式中对应原索引5)
        n[3] = (p_internal[3] - b[5, 0]) / (b[5, 1] - b[5, 0] + 1e-12)
        # Rsh (索引4在5参数模式中对应原索引6)
        n[4] = (p_internal[4] - b[6, 0]) / (b[6, 1] - b[6, 0] + 1e-12)
        return np.clip(n, 0.0, 1.0)
    
    def _expand_to_full_params(self, p_trainable: np.ndarray) -> np.ndarray:
        """将5个可训练参数扩展为7个完整参数（添加固定的n1和n2）"""
        if self.fix_n1 and self.fix_n2:
            p_full = np.zeros(7, dtype=np.float64)
            p_full[0] = p_trainable[0]  # I_ph
            p_full[I01_IDX] = p_trainable[I01_IDX]  # log10(I01)
            p_full[I02_IDX] = p_trainable[I02_IDX]  # log10(I02)
            p_full[3] = self.fixed_n1  # n1固定
            p_full[4] = self.fixed_n2  # n2固定
            p_full[5] = p_trainable[3]  # Rs
            p_full[6] = p_trainable[4]  # Rsh
            return p_full
        else:
            return p_trainable
    
    def _build_state(
        self,
        t: int,
        p_internal: np.ndarray,
        f_t: float,
        delta_f_t: float,
        f0: float,
    ) -> np.ndarray:
        """
        改进：构建状态向量，使用更鲁棒的归一化方法，提升泛化能力。
        状态向量：曲线特征(9) + 参数归一(5或7) + log(1+f) + delta_f 归一 + t/T。
        🔥 固定n1和n2模式：参数归一从7变成5。
        """
        # 🔥 如果固定n1和n2，只归一化5个参数
        if self.fix_n1 and self.fix_n2:
            params_norm = self._normalize_trainable_params(p_internal)
        else:
            params_norm = internal_to_normalized(p_internal, self.param_bounds)
        # 改进：f值的归一化，使用更鲁棒的方法
        # 使用平滑的对数变换，避免f值尺度差异过大
        log_f = np.log10(1.0 + f_t)
        # 放宽log_f的范围，避免裁剪有效数据
        log_f = np.clip(log_f, -5.0, 3.0)  # 从[-3.0, 2.0]放宽到[-5.0, 3.0]
        
        # 改进：delta_f的归一化，使用更鲁棒的方法
        denom = max(f0, 1e-10)
        # 放宽delta_f_norm的范围，避免裁剪有效数据
        delta_f_norm = np.clip(delta_f_t / denom, -5.0, 5.0)  # 从[-2.0, 2.0]放宽到[-5.0, 5.0]
        
        progress = float(t) / max(self.T_max, 1)
        s = np.concatenate([
            self._curve_feat_norm,
            params_norm,
            [log_f, delta_f_norm, progress],
        ]).astype(np.float32)
        return s

    def _action_to_delta(self, a: np.ndarray) -> np.ndarray:
        """     
        将 Actor 输出的原始动作 a 映射为参数增量 Δp。
        使用 tanh 压缩到 [-1,1] 再按 max_delta 缩放，避免幅度过大。
        策略梯度仍基于原始 a 的 log π(a|s)。
        """
        x = np.tanh(np.asarray(a, dtype=np.float64))
        return x * self.max_delta
    
    def _compute_boundary_penalty(self, params: np.ndarray, bounds: np.ndarray) -> float:
        """计算边界惩罚：参数接近边界时给予适度惩罚（已禁用）"""
        # 🔥 已禁用边界惩罚：允许探索边界附近的解，让clip_params_to_bounds处理边界即可
        return 0.0
        
        # 以下代码已禁用，保留供参考
        if not ParamAccessor.is_double_diode(params):
            return 0.0  # 单二极管模型暂不处理边界惩罚
        
        penalty = 0.0
        margin = TrainingConfig.BOUNDARY_MARGIN
        tolerance = TrainingConfig.BOUNDARY_TOLERANCE
        scale = TrainingConfig.REWARD_SCALE * TrainingConfig.BOUNDARY_PENALTY_SCALE
        
        # I02和n2是重点关注的参数（经常卡在边界）
        param_configs = [
            ('I02', 2, True, 0.3),   # (参数名, 索引, 是否log10空间, 惩罚系数)
            ('n2', 4, False, 0.4),
            ('I01', 1, True, 0.3),
            ('I_ph', 0, False, 0.2),
            ('n1', 3, False, 0.2),
            ('Rs', 5, False, 0.2),
            ('Rsh', 6, False, 0.2),
        ]
        
        for param_name, idx, is_log, penalty_coef in param_configs:
            if is_log:
                # log10空间参数
                log_val = np.log10(max(params[idx], 1e-60))
                log_range = np.log10(bounds[idx, 1]) - np.log10(bounds[idx, 0])
                dist_low = (log_val - np.log10(bounds[idx, 0])) / (log_range + 1e-12)
                dist_high = (np.log10(bounds[idx, 1]) - log_val) / (log_range + 1e-12)
            else:
                # 线性空间参数
                param_range = bounds[idx, 1] - bounds[idx, 0]
                dist_low = (params[idx] - bounds[idx, 0]) / (param_range + 1e-12)
                dist_high = (bounds[idx, 1] - params[idx]) / (param_range + 1e-12)
            
            dist_min = min(dist_low, dist_high)
            if dist_min < tolerance:
                penalty -= scale * penalty_coef
            elif dist_min < margin:
                penalty -= (margin - dist_min) / margin * scale * penalty_coef
        
        return penalty
    
    def _compute_rewards(
        self, 
        params_next: np.ndarray, 
        f_prev: float, 
        f_next: float,
        bounds: np.ndarray
    ) -> Dict[str, float]:
        """计算所有奖励分量（简化版本）"""
        I_sc = self._curve_feat[1]
        is_double = ParamAccessor.is_double_diode(params_next)
        
        # 1. 稀疏奖励（主要信号）
        r_sparse = float(f_prev - f_next) * TrainingConfig.REWARD_SCALE
        
        # 2. 形状奖励
        I_ph = ParamAccessor.get_param(params_next, 'I_ph')
        I_ph_error = abs(I_ph - I_sc) / (I_sc + 1e-6)
        r_flat = -I_ph_error * TrainingConfig.REWARD_SCALE
        
        # 3. 陡降段奖励
        I0_total = ParamAccessor.get_I0_total(params_next)
        log_I0 = np.log10(max(I0_total, 1e-60))
        r_knee = (log_I0 + 40.0) / 10.0 * TrainingConfig.REWARD_SCALE
        
        # 4. Rs奖励
        Rs = ParamAccessor.get_param(params_next, 'Rs')
        Rs_idx = 5 if is_double else 3
        Rs_range = bounds[Rs_idx, 1] - bounds[Rs_idx, 0]
        Rs_center = (bounds[Rs_idx, 0] + bounds[Rs_idx, 1]) / 2.0
        Rs_error = abs(Rs - Rs_center) / (Rs_range + 1e-6)
        r_rs = -Rs_error * TrainingConfig.REWARD_SCALE
        
        # 5. Rsh奖励
        Rsh = ParamAccessor.get_param(params_next, 'Rsh')
        Rsh_idx = 6 if is_double else 4
        Rsh_min = bounds[Rsh_idx, 0] * 1.5
        r_rsh = -((Rsh_min - Rsh) / (Rsh_min + 1e-6)) * TrainingConfig.REWARD_SCALE if Rsh < Rsh_min else 0.0
        
        # 6. 边界惩罚
        r_boundary = self._compute_boundary_penalty(params_next, bounds)
        
        return {
            'sparse': r_sparse,
            'flat': r_flat,
            'knee': r_knee,
            'rs': r_rs,
            'rsh': r_rsh,
            'boundary': r_boundary
        }

    def _get_reward_weights(self, epoch: int, total_epochs: int) -> Tuple[float, float, float]:
        """
        改进：根据训练进度动态调整奖励权重，多阶段训练策略。
        训练初期：关注整体拟合（sparse奖励权重高）
        训练中期：平衡各项（各项权重均衡）
        训练后期：关注细节（shape奖励权重高）
        """
        progress = epoch / max(total_epochs, 1)
        if progress < 0.3:
            # 训练初期（0-30%）：关注整体拟合，快速降低误差
            return (0.5, 0.3, 0.2)  # [sparse, flat, knee]
        elif progress < 0.7:
            # 训练中期（30-70%）：平衡各项，稳定优化，开始关注I0
            return (0.35, 0.35, 0.30)
        else:
            # 训练后期（70-100%）：更关注I0细节，精细调整（I0权重提升到45%）
            return (0.25, 0.30, 0.45)  # I0权重从40%提升到45%，更强制学习
    
    def _normalize_reward(self, reward: float, reward_type: str) -> float:
        """
        自动归一化奖励，让不同奖励的尺度一致，减少对权重的依赖。
        使用在线更新的均值和标准差进行Z-score归一化。
        """
        stats = self.reward_stats[reward_type]
        count = stats['count']
        
        if count == 0:
            # 第一次：直接使用原始值
            stats['mean'] = reward
            stats['std'] = abs(reward) + 1e-6
            stats['count'] = 1
            return reward / (abs(reward) + 1e-6)  # 归一化到[-1, 1]附近
        else:
            # 在线更新均值和标准差（使用指数移动平均）
            alpha = 0.01  # 更新率
            old_mean = stats['mean']
            old_std = stats['std']
            
            # 更新均值
            new_mean = (1 - alpha) * old_mean + alpha * reward
            # 更新标准差（使用移动平均）
            new_std = np.sqrt((1 - alpha) * (old_std ** 2) + alpha * (reward - old_mean) ** 2)
            new_std = max(new_std, 1e-6)  # 避免除零
            
            stats['mean'] = new_mean
            stats['std'] = new_std
            stats['count'] += 1
            
            # Z-score归一化
            normalized = (reward - new_mean) / new_std
            # 裁剪到合理范围，避免极端值
            return np.clip(normalized, -10.0, 10.0)
    
    def _update_adaptive_weights(self, epoch: int, total_epochs: int):
        """
        根据奖励历史表现自动调整权重。
        如果某个奖励一直很小（说明网络没有学习到），就增加它的权重。
        如果某个奖励一直很大（说明网络已经学得很好），就减少它的权重。
        """
        if len(self.reward_history['sparse']) < 100:
            return  # 历史数据不足，不调整
        
        # 计算最近100个episode的平均奖励
        window = 100
        recent_sparse = np.mean(self.reward_history['sparse'][-window:])
        recent_flat = np.mean(self.reward_history['flat'][-window:])
        recent_knee = np.mean(self.reward_history['knee'][-window:])
        recent_rs = np.mean(self.reward_history['rs'][-window:])
        recent_rsh = np.mean(self.reward_history['rsh'][-window:])
        recent_boundary = np.mean(self.reward_history['boundary'][-window:]) if len(self.reward_history['boundary']) > 0 else 0.0
        
        # 归一化到[0, 1]范围（相对大小）
        rewards = np.array([recent_sparse, recent_flat, recent_knee, recent_rs, recent_rsh, recent_boundary])
        rewards_min = np.min(rewards)
        rewards_max = np.max(rewards)
        if rewards_max - rewards_min > 1e-6:
            rewards_norm = (rewards - rewards_min) / (rewards_max - rewards_min)
        else:
            rewards_norm = np.array([0.15, 0.15, 0.15, 0.05, 0.05, 0.15])  # 如果都差不多，给boundary较高权重
        
        # 奖励小的权重增加，奖励大的权重减少（但保持总和为1）
        # 使用softmax的逆操作：奖励小的给更大权重
        inverse_rewards = 1.0 - rewards_norm + 0.1  # +0.1避免权重为0
        weights = inverse_rewards / np.sum(inverse_rewards)
        
        # 平滑更新（避免权重变化太快）
        alpha = 0.05  # 更新率
        self.adaptive_weights['sparse'] = (1 - alpha) * self.adaptive_weights['sparse'] + alpha * weights[0]
        self.adaptive_weights['flat'] = (1 - alpha) * self.adaptive_weights['flat'] + alpha * weights[1]
        self.adaptive_weights['knee'] = (1 - alpha) * self.adaptive_weights['knee'] + alpha * weights[2]
        self.adaptive_weights['rs'] = (1 - alpha) * self.adaptive_weights['rs'] + alpha * weights[3]
        self.adaptive_weights['rsh'] = (1 - alpha) * self.adaptive_weights['rsh'] + alpha * weights[4]
        # 边界惩罚权重更新（已禁用，因为边界惩罚已设为0）
        # if 'boundary' in self.adaptive_weights:
        #     self.adaptive_weights['boundary'] = (1 - alpha) * self.adaptive_weights['boundary'] + alpha * weights[5]
        
        # 归一化确保总和为1
        total = sum(self.adaptive_weights.values())
        for key in self.adaptive_weights:
            self.adaptive_weights[key] /= total
        
        # 理性调整：边界惩罚权重保持适度（不要过大）
        # 边界惩罚权重限制（已禁用，因为边界惩罚已设为0）
        # if 'boundary' in self.adaptive_weights:
        #     self.adaptive_weights['boundary'] = min(self.adaptive_weights['boundary'], 0.15)  # 最多15%权重
            # 重新归一化
            total = sum(self.adaptive_weights.values())
            for key in self.adaptive_weights:
                self.adaptive_weights[key] /= total
    
    def _run_episode(
        self,
        explore: bool = True,
        current_epoch: int = 0,  # 新增：当前训练轮数
        total_epochs: int = 500,  # 新增：总训练轮数
    ) -> Tuple[List[Dict[str, Any]], float, np.ndarray]:
        """
        跑一个 episode：多步迭代，收集 (s_t, a_t, r_t, log_prob_t, V_t)。
        奖励 r_t = f(p_t) - f(p_{t+1})；第一步用 f(p_0)-f(p_1)。
        返回：(trajectory, 最终 f, 最终参数物理值)
        """
        b = self.param_bounds
        trajectory: List[Dict[str, Any]] = []
        # 激进初始化（适度使用）
        use_aggressive = (current_epoch > TrainingConfig.EXPLORATION_BOOST_THRESHOLD and 
                         current_epoch % 20 == 0)
        p = self._init_p0(use_aggressive_init=use_aggressive)
        # 🔥 固定n1和n2模式：扩展为完整参数用于计算目标函数
        if self.fix_n1 and self.fix_n2:
            p_full = self._expand_to_full_params(p)
            params_phys = internal_to_params(p_full, b)
        else:
            params_phys = internal_to_params(p, b)
        f_prev = objective_function(params_phys, self.V, self.I_meas, self.I_min, self.I_max)
        f0 = max(f_prev, 1e-10)
        delta_f_prev = 0.0

        # 计算探索因子（使用配置类）
        base_exploration = TrainingConfig.EXPLORATION_INITIAL - \
                          (TrainingConfig.EXPLORATION_INITIAL - TrainingConfig.EXPLORATION_FINAL) * \
                          (current_epoch / max(total_epochs, 1))
        base_exploration = max(base_exploration, TrainingConfig.EXPLORATION_FINAL)
        
        # 🔥 检测参数是否在边界，如果在边界则大幅增强探索
        is_at_boundary = False
        # 根据是否固定n1和n2，调整边界检测逻辑
        if self.fix_n1 and self.fix_n2:
            # 5参数模式：只检测5个可训练参数
            param_indices_to_check = [0, I01_IDX, I02_IDX, 3, 4]  # I_ph, I01, I02, Rs, Rsh
            bound_indices = [0, I01_IDX, I02_IDX, 5, 6]  # 对应的边界索引
        else:
            # 7参数模式：检测所有参数
            param_indices_to_check = list(range(len(p)))
            bound_indices = list(range(len(p)))
        
        for idx, param_idx in enumerate(param_indices_to_check):
            bound_idx = bound_indices[idx]
            if param_idx in [I01_IDX, I02_IDX]:
                # log10空间参数
                log_val = p[param_idx]
                log_lo = np.log10(b[bound_idx, 0])
                log_hi = np.log10(b[bound_idx, 1])
                if abs(log_val - log_lo) < 1e-5 or abs(log_val - log_hi) < 1e-5:
                    is_at_boundary = True
                    break
            else:
                # 线性空间参数
                if abs(p[param_idx] - b[bound_idx, 0]) < 1e-5 or abs(p[param_idx] - b[bound_idx, 1]) < 1e-5:
                    is_at_boundary = True
                    break
        
        # 训练后期适度增加探索，如果在边界则大幅增强
        if is_at_boundary and explore:
            exploration_factor = min(base_exploration * 3.0, 2.0)  # 边界时探索因子×3，最多2.0
        elif current_epoch > TrainingConfig.EXPLORATION_BOOST_THRESHOLD:
            exploration_factor = min(base_exploration * TrainingConfig.EXPLORATION_BOOST_FACTOR, 1.5)
        else:
            exploration_factor = base_exploration
        
        for t in range(self.T_max):
            # 🔥 固定n1和n2模式：构建状态时使用正确的参数维度
            # 构建当前状态（曲线特征 + 当前参数 + 当前误差 + 步数）
            s = self._build_state(t, p, f_prev, delta_f_prev, f0)
            # 用 tensor(list) 避免 torch.from_numpy，防止 NumPy DLL 与 PyTorch 冲突时报错
            s_t = torch.tensor(s.tolist(), dtype=torch.float32, device=self.device).unsqueeze(0)
      
            with torch.no_grad():
                V_t = self.critic(s_t).squeeze(0).item()

            # 改进：探索时使用自适应探索因子，评估时使用最小探索（确定性策略）
            # 用 .tolist() 代替 .numpy()，避免 PyTorch 与 NumPy DLL 冲突时报 "Numpy is not available"
            if explore:
                a_tensor, log_prob_t = self.actor.sample(s_t, exploration_factor)
                a_raw = a_tensor.cpu().squeeze(0).tolist()
            else:
                mean, _ = self.actor(s_t, exploration_factor=0.3)  # 评估时使用最小探索
                a_raw = mean.cpu().squeeze(0).tolist()
                log_prob_t = self.actor.log_prob(s_t, mean, exploration_factor=0.3)

            delta = self._action_to_delta(a_raw)
            # 适度添加噪声（使用配置类）
            if current_epoch > TrainingConfig.NOISE_THRESHOLD_EPOCH and explore:
                noise_scale = TrainingConfig.NOISE_SCALE_BASE * (current_epoch / max(total_epochs, 1))
                extra_noise = np.random.randn(len(delta)) * noise_scale
                delta = delta + extra_noise
            
            # 🔥 固定n1和n2模式：只更新5个参数
            if self.fix_n1 and self.fix_n2:
                # 只更新5个可训练参数
                p_next_trainable = clip_params_to_bounds_trainable(p + self.alpha * delta, b)
                # 扩展为完整7参数用于计算目标函数
                p_next_full = self._expand_to_full_params(p_next_trainable)
                params_next = internal_to_params(p_next_full, b)
                p_next = p_next_trainable  # 保存5参数版本用于下一轮
                
                # 🔥 边界反弹机制（5参数模式）
                if explore:
                    bounce_factor = 0.05
                    param_mapping = [0, I01_IDX, I02_IDX, 5, 6]  # 5参数到7参数边界的映射
                    for i, bound_idx in enumerate(param_mapping):
                        if i in [I01_IDX, I02_IDX]:
                            log_val = p_next[i]
                            log_lo = np.log10(b[bound_idx, 0])
                            log_hi = np.log10(b[bound_idx, 1])
                            if abs(log_val - log_lo) < 1e-6:
                                p_next[i] += bounce_factor * (log_hi - log_lo)
                            elif abs(log_val - log_hi) < 1e-6:
                                p_next[i] -= bounce_factor * (log_hi - log_lo)
                        else:
                            param_range = b[bound_idx, 1] - b[bound_idx, 0]
                            if abs(p_next[i] - b[bound_idx, 0]) < 1e-6:
                                p_next[i] += bounce_factor * param_range
                            elif abs(p_next[i] - b[bound_idx, 1]) < 1e-6:
                                p_next[i] -= bounce_factor * param_range
                    p_next = clip_params_to_bounds_trainable(p_next, b)
                    p_next_full = self._expand_to_full_params(p_next)
                    params_next = internal_to_params(p_next_full, b)
            else:
                p_next = clip_params_to_bounds(p + self.alpha * delta, b)
                params_next = internal_to_params(p_next, b)
                
                # 🔥 边界反弹机制（7参数模式）
                if explore:
                    bounce_factor = 0.05
                    for i in range(len(p_next)):
                        if i in [I01_IDX, I02_IDX]:
                            log_val = p_next[i]
                            log_lo = np.log10(b[i, 0])
                            log_hi = np.log10(b[i, 1])
                            if abs(log_val - log_lo) < 1e-6:
                                p_next[i] += bounce_factor * (log_hi - log_lo)
                            elif abs(log_val - log_hi) < 1e-6:
                                p_next[i] -= bounce_factor * (log_hi - log_lo)
                        else:
                            param_range = b[i, 1] - b[i, 0]
                            if abs(p_next[i] - b[i, 0]) < 1e-6:
                                p_next[i] += bounce_factor * param_range
                            elif abs(p_next[i] - b[i, 1]) < 1e-6:
                                p_next[i] -= bounce_factor * param_range
                    p_next = clip_params_to_bounds(p_next, b)
                    params_next = internal_to_params(p_next, b)
            
            # 参数合理性检查
            I_sc = self._curve_feat[1]
            I_ph_next = ParamAccessor.get_param(params_next, 'I_ph')
            Rs_next = ParamAccessor.get_param(params_next, 'Rs')
            
            # 参数合理性检查（惩罚会在奖励计算中处理）
            
            f_next = objective_function(params_next, self.V, self.I_meas, self.I_min, self.I_max)
            
            # 使用统一的奖励计算函数（简化代码，消除重复）
            rewards_raw = self._compute_rewards(params_next, f_prev, f_next, b)
            r_sparse_raw = rewards_raw['sparse']
            shape_reward_flat_raw = rewards_raw['flat']
            shape_reward_knee_raw = rewards_raw['knee']
            shape_reward_rs_raw = rewards_raw['rs']
            shape_reward_rsh_raw = rewards_raw['rsh']
            boundary_penalty_raw = rewards_raw['boundary']
            
            # ========== 自动归一化奖励（让不同奖励的尺度一致）==========
            r_sparse_norm = self._normalize_reward(r_sparse_raw, 'sparse')
            shape_reward_flat_norm = self._normalize_reward(shape_reward_flat_raw, 'flat')
            shape_reward_knee_norm = self._normalize_reward(shape_reward_knee_raw, 'knee')
            shape_reward_rs_norm = self._normalize_reward(shape_reward_rs_raw, 'rs')
            shape_reward_rsh_norm = self._normalize_reward(shape_reward_rsh_raw, 'rsh')
            # 边界惩罚归一化
            if 'boundary' in self.reward_stats:
                boundary_penalty_norm = self._normalize_reward(boundary_penalty_raw, 'boundary')
            else:
                boundary_penalty_norm = boundary_penalty_raw / TrainingConfig.REWARD_SCALE
            
            # ========== 记录奖励历史（用于自适应权重调整）==========
            self.reward_history['sparse'].append(r_sparse_raw)
            self.reward_history['flat'].append(shape_reward_flat_raw)
            self.reward_history['knee'].append(shape_reward_knee_raw)
            self.reward_history['rs'].append(shape_reward_rs_raw)
            self.reward_history['rsh'].append(shape_reward_rsh_raw)
            self.reward_history['boundary'].append(boundary_penalty_raw)  # 🔥 记录边界惩罚
            
            # ========== 组合奖励：使用自适应权重 ==========
            # 选项1：使用自适应权重（自动调整）- 推荐！
            # 设置为False可以禁用自适应权重，使用固定权重
            use_adaptive = True  # 设置为False可以禁用自适应权重
            
            if use_adaptive:
                # 使用自适应权重（会根据历史表现自动调整）
                r_t = (self.adaptive_weights['sparse'] * r_sparse_norm +
                       self.adaptive_weights['flat'] * shape_reward_flat_norm +
                       self.adaptive_weights['knee'] * shape_reward_knee_norm +
                       self.adaptive_weights['rs'] * shape_reward_rs_norm +
                       self.adaptive_weights['rsh'] * shape_reward_rsh_norm +
                       self.adaptive_weights.get('boundary', 0.2) * boundary_penalty_norm)  # 🔥 添加边界惩罚（使用get避免KeyError）
            else:
                # 使用固定权重（原来的方法，需要手动调参）
                w_sparse, w_flat, w_knee = self._get_reward_weights(current_epoch, total_epochs)
                r_t = (w_sparse * r_sparse_norm +
                       w_flat * shape_reward_flat_norm +
                       w_knee * shape_reward_knee_norm +
                       0.04 * shape_reward_rs_norm +
                       0.04 * shape_reward_rsh_norm)
            
            # 注意：参数合理性惩罚已包含在形状奖励中，无需额外添加

            # 异常时给大负奖励并提前结束
            if np.isnan(f_next) or np.isinf(f_next) or f_next >= 1e9:
                r_t = -10.0  # 异常奖励（归一化后的值）
                trajectory.append({
                    "s": s, "a": a_raw, "r": r_t,
                    "log_prob": log_prob_t.detach(),
                    "V": V_t,
                })
                break

            trajectory.append({
                "s": s, "a": a_raw, "r": r_t,
                "log_prob": log_prob_t.detach(),
                "V": V_t,
            })
            # 🔥 改进：使用归一化后的奖励作为delta_f_prev，而不是原始r_t
            # 这样可以更好地反映长期趋势，减少短视
            delta_f_prev = r_t  # r_t已经是归一化后的奖励
            f_prev = f_next
            p = p_next

        # 🔥 固定n1和n2模式：返回完整参数
        if self.fix_n1 and self.fix_n2:
            p_full = self._expand_to_full_params(p)
            final_params = internal_to_params(p_full, b)
        else:
            final_params = internal_to_params(p, b)
        return trajectory, f_prev, final_params

    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """
        计算折扣回报 R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...，从轨迹末尾倒推。
        🔥 改进：添加最终奖励加成，鼓励长期优化路径
        """
        R = [0.0] * len(rewards)
        run = 0.0
        
        # 🔥 最终奖励加成：如果轨迹最终误差下降，给予额外奖励
        # 这鼓励"先变差再变好"的路径
        if len(rewards) > 0:
            # 计算轨迹的最终趋势（最后5步的平均奖励）
            final_trend = np.mean(rewards[-min(5, len(rewards)):]) if len(rewards) >= 5 else rewards[-1]
            # 如果最终趋势为正（误差下降），给予加成
            final_bonus = max(0.0, final_trend * 0.1)  # 10%的最终趋势加成
        else:
            final_bonus = 0.0
        
        for t in reversed(range(len(rewards))):
            run = rewards[t] + self.gamma * run
            # 🔥 在最后几步添加最终奖励加成，鼓励长期优化
            if t >= len(rewards) - 3:  # 最后3步
                R[t] = run + final_bonus
            else:
                R[t] = run
        return R

    def update(self, trajectories: List[List[Dict[str, Any]]]) -> Tuple[float, float]:
        """
        用多条轨迹的 (s,a,r,R,V) 批量更新 Actor 和 Critic。
        Actor: 梯度上升 E[log pi(a|s) * A]，A = R - V(s)。
        Critic: MSE(V(s), R)。
        """
        all_s, all_a, all_R = [], [], []
        for traj in trajectories:
            rewards = [x["r"] for x in traj]
            R_list = self._compute_returns(rewards)
            for i, x in enumerate(traj):
                all_s.append(x["s"])
                all_a.append(x["a"])
                all_R.append(R_list[i])

        # 用 tensor(list) 避免 torch.from_numpy，防止 NumPy DLL 与 PyTorch 冲突时报错
        # all_a 中可能是 list（来自 .tolist()）或 array，统一转为 list
        S = torch.tensor([(x.tolist() if hasattr(x, "tolist") else x) for x in all_s], dtype=torch.float32, device=self.device)
        A = torch.tensor([(x.tolist() if hasattr(x, "tolist") else x) for x in all_a], dtype=torch.float32, device=self.device)
        R = torch.tensor(all_R, dtype=torch.float32, device=self.device)

        # Critic：最小化 MSE(V(s), R)，让 V 逼近折扣回报
        V_pred = self.critic(S)
        loss_c = F.mse_loss(V_pred, R)
        self.opt_critic.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) 
        self.opt_critic.step()

        # Advantage A = R - V(s)，用当前 Critic 估计（更新后）减少方差
        with torch.no_grad():
            V_new = self.critic(S)
        adv = R - V_new.detach()

        # Actor：最大化 E[log π(a|s) · A]，即 loss = -mean(log π · A)
        # 改进：添加熵正则化，鼓励探索，避免过早收敛到次优解
        # 注意：update函数中无法直接获取epoch信息，使用默认exploration_factor=1.0
        # 实际的探索调整在_run_episode中完成
        log_prob = self.actor.log_prob(S, A, exploration_factor=1.0)
        mean, std = self.actor.forward(S, exploration_factor=1.0)
        dist = torch.distributions.Normal(mean, std + 1e-6)
        entropy = dist.entropy().sum(dim=-1).mean()  # 策略熵
        loss_a = -(log_prob * adv).mean() - TrainingConfig.ENTROPY_COEF * entropy
        self.opt_actor.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        return float(loss_c.item()), float(loss_a.item())

    def fit(
        self,
        n_epochs: int = 500,  # 增加训练轮数：300 -> 500，给网络更多学习时间
        episodes_per_epoch: int = 128,  # 保持128，确保足够的批量大小，不影响训练稳定性
        eval_interval: int = 10,
        early_stop_patience: int = 10000,  # 早停：大幅提高（几乎禁用），给优化器充分探索时间
        early_stop_min_delta: float = 1e-7,  # 早停：从1e-6降低到1e-7，允许更小的改善
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        训练主循环。每轮跑多条 episode，更新 AC；
        定期用无探索跑一条轨迹，记录最终 f。
        返回：(best_params, best_f, f_history)
        """
        best_f = float("inf")
        best_params: Optional[np.ndarray] = None
        f_history: List[float] = []
        no_improve_count = 0  # 记录连续未改善的轮数
        restart_count = 0  # 🔥 重启计数器
        
        # 🔥 突破局部最优策略2：记录最佳参数，用于重启
        best_params_internal: Optional[np.ndarray] = None

        for epoch in range(n_epochs):
            trajs = []
            for _ in range(episodes_per_epoch):
                tr, f_end, _ = self._run_episode(explore=True, current_epoch=epoch, total_epochs=n_epochs)
                trajs.append(tr)
            if trajs:
                _ = self.update(trajs)
                # 更新学习率（每100轮衰减10%）
                self.scheduler_actor.step()
                self.scheduler_critic.step()

            # 🔥 修改打印逻辑：前50次每次都打印，后面每10次打印一次
            should_eval = False
            if epoch < 50:
                # 前50次每次都评估和打印
                should_eval = True
            else:
                # 后面每10次评估和打印一次
                should_eval = (epoch + 1) % eval_interval == 0
            
            if should_eval:
                _, f_eval, p_eval = self._run_episode(explore=False, current_epoch=epoch, total_epochs=n_epochs)  # 无探索，取确定性策略
                f_history.append(f_eval)
                improved = False
                if f_eval < best_f - early_stop_min_delta:  # 有显著改善
                    best_f = f_eval
                    best_params = p_eval.copy()
                    # 🔥 保存内部表示用于重启
                    _, _, best_params_internal_temp = self._run_episode(explore=False, current_epoch=epoch, total_epochs=n_epochs)
                    best_params_internal = best_params_internal_temp.copy() if best_params_internal_temp is not None else None
                    no_improve_count = 0  # 重置计数器
                    improved = True
                else:
                    # 计算未改善计数：前50次每次+1，后面每次+eval_interval
                    increment = 1 if epoch < 50 else eval_interval  
                    no_improve_count += increment  # 增加未改善计数
                
                # ========== 自适应权重更新（每10轮更新一次）==========
                if (epoch + 1) % eval_interval == 0:
                    self._update_adaptive_weights(epoch, n_epochs)
                    # 打印当前自适应权重（每50轮打印一次，避免输出太多）
                    if (epoch + 1) % 50 == 0:
                        print(f"\n  自适应权重: sparse={self.adaptive_weights['sparse']:.3f}, "
                              f"flat={self.adaptive_weights['flat']:.3f}, "
                              f"knee={self.adaptive_weights['knee']:.3f}, "
                              f"rs={self.adaptive_weights['rs']:.3f}, "
                              f"rsh={self.adaptive_weights['rsh']:.3f}, "
                              f"boundary={self.adaptive_weights.get('boundary', 0.0):.3f}")
                
                # 🔥 添加详细诊断信息（前20个epoch）
                if epoch < 20:
                    # 打印当前参数值（帮助诊断）
                    params_str = ""
                    if len(p_eval) == 7:
                        params_str = f" | I_ph={p_eval[0]:.4f}, I01={p_eval[1]:.2e}, I02={p_eval[2]:.2e}, n1={p_eval[3]:.3f}, n2={p_eval[4]:.3f}, Rs={p_eval[5]:.4f}, Rsh={p_eval[6]:.2f}"
                    elif len(p_eval) == 5:
                        # 🔥 固定n1和n2模式：显示5个参数 + 固定的n1和n2
                        params_str = f" | I_ph={p_eval[0]:.4f}, I01={p_eval[1]:.2e}, I02={p_eval[2]:.2e}, n1={TrainingConfig.FIXED_N1_VALUE:.1f}(固定), n2={TrainingConfig.FIXED_N2_VALUE:.1f}(固定), Rs={p_eval[3]:.4f}, Rsh={p_eval[4]:.2f}"
                    else:
                        params_str = f" | I_ph={p_eval[0]:.4f}, I0={p_eval[1]:.2e}, n={p_eval[2]:.3f}, Rs={p_eval[3]:.4f}, Rsh={p_eval[4]:.2f}"
                    print(f"Epoch {epoch+1:4d} | eval f = {f_eval:.6e} | best f = {best_f:.6e}{params_str}", end="")
                else:
                    print(f"Epoch {epoch+1:4d} | eval f = {f_eval:.6e} | best f = {best_f:.6e}", end="")
                
                if improved:
                    print(" *")  # 标记有改善
                else:
                    print(f" (no improve: {no_improve_count}/{early_stop_patience})")
                
            # 🔥 前10个epoch添加诊断信息
            if epoch < 10 and not improved:
                # 检查参数是否卡在边界
                boundary_hits = []
                b = self.param_bounds
                if len(p_eval) == 7:
                    if abs(p_eval[0] - b[0, 0]) < 1e-6 or abs(p_eval[0] - b[0, 1]) < 1e-6:
                        boundary_hits.append("I_ph")
                    if abs(p_eval[1] - 10**np.log10(b[1, 0])) < 1e-10 or abs(p_eval[1] - 10**np.log10(b[1, 1])) < 1e-10:
                        boundary_hits.append("I01")
                    if abs(p_eval[2] - 10**np.log10(b[2, 0])) < 1e-10 or abs(p_eval[2] - 10**np.log10(b[2, 1])) < 1e-10:
                        boundary_hits.append("I02")
                    if not self.fix_n1 and (abs(p_eval[3] - b[3, 0]) < 1e-6 or abs(p_eval[3] - b[3, 1]) < 1e-6):
                        boundary_hits.append("n1")
                    if not self.fix_n2 and (abs(p_eval[4] - b[4, 0]) < 1e-6 or abs(p_eval[4] - b[4, 1]) < 1e-6):
                        boundary_hits.append("n2")
                    if abs(p_eval[5] - b[5, 0]) < 1e-6 or abs(p_eval[5] - b[5, 1]) < 1e-6:
                        boundary_hits.append("Rs")
                    if abs(p_eval[6] - b[6, 0]) < 1e-6 or abs(p_eval[6] - b[6, 1]) < 1e-6:
                        boundary_hits.append("Rsh")
                elif len(p_eval) == 5:
                    # 🔥 固定n1和n2模式：只检查5个可训练参数
                    if abs(p_eval[0] - b[0, 0]) < 1e-6 or abs(p_eval[0] - b[0, 1]) < 1e-6:
                        boundary_hits.append("I_ph")
                    if abs(p_eval[1] - 10**np.log10(b[1, 0])) < 1e-10 or abs(p_eval[1] - 10**np.log10(b[1, 1])) < 1e-10:
                        boundary_hits.append("I01")
                    if abs(p_eval[2] - 10**np.log10(b[2, 0])) < 1e-10 or abs(p_eval[2] - 10**np.log10(b[2, 1])) < 1e-10:
                        boundary_hits.append("I02")
                    if abs(p_eval[3] - b[5, 0]) < 1e-6 or abs(p_eval[3] - b[5, 1]) < 1e-6:
                        boundary_hits.append("Rs")
                    if abs(p_eval[4] - b[6, 0]) < 1e-6 or abs(p_eval[4] - b[6, 1]) < 1e-6:
                        boundary_hits.append("Rsh")
                
                if boundary_hits:
                    print(f"   ⚠ 警告：参数 {', '.join(boundary_hits)} 卡在边界，可能限制搜索空间")
                else:
                    print(f"   提示：前几个epoch误差波动是正常的，双二极管模型需要更多探索时间")
                
                # 重启机制（使用配置类）
                if no_improve_count >= TrainingConfig.RESTART_PATIENCE and \
                   no_improve_count % TrainingConfig.RESTART_PATIENCE == 0 and \
                   epoch < n_epochs - 10:
                    restart_count += 1
                    print(f"\n🔄 重启机制触发（第{restart_count}次）：连续{no_improve_count}轮未改善")
                    # 适度重置网络参数
                    with torch.no_grad():
                        for name, param in self.actor.named_parameters():
                            if 'fc' in name and 'weight' in name:
                                noise = torch.randn_like(param) * TrainingConfig.RESTART_NOISE_SCALE
                                param.data += noise
                                param.data = torch.clamp(param.data, -2.0, 2.0)
                    print(f"   已添加随机噪声（{TrainingConfig.RESTART_NOISE_SCALE*100:.0f}%），继续训练...")
                    no_improve_count = 0
                
                # 早停检查（已禁用：允许充分探索）
                # 🔥 已禁用早停机制：允许优化器充分探索搜索空间
                # if no_improve_count >= early_stop_patience:
                #     print(f"\n早停触发：best_f 连续 {no_improve_count} 轮未改善，提前停止训练")
                #     print(f"  当前 best_f = {best_f:.6e}")
                #     break

        if best_params is None:
            _, _, best_params = self._run_episode(explore=False, current_epoch=n_epochs-1, total_epochs=n_epochs)
            best_f = objective_function(best_params, self.V, self.I_meas, self.I_min, self.I_max)
        return best_params, best_f, f_history


# =============================================================================
# 7. 拟合结果可视化
# =============================================================================

def plot_fit_result(
    V: np.ndarray,
    I_meas: np.ndarray,
    best_params: np.ndarray,
    I_min: float,
    I_max: float,
    best_f: float,
    f_history: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    绘制拟合结果：左图 IV 曲线（实测 vs 拟合），右图目标函数 f 随评估轮次的变化。
    """
    I_sim = solar_cell_model(V, best_params, I_min, I_max)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 左图：IV 曲线 — 实测点 vs 拟合曲线
    ax1 = axes[0]
    ax1.scatter(V, I_meas, c="tab:blue", s=20, label="实测", zorder=2)
    ax1.plot(V, I_sim, "r-", lw=1.5, label="拟合", zorder=1)
    ax1.set_xlabel("电压 V (V)")
    ax1.set_ylabel("电流 I (A)")
    ax1.set_title(f"IV 曲线：实测 vs 拟合 (f = {best_f:.4e})")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))

    # 右图：目标函数 f 随评估轮次
    ax2 = axes[1]
    n_evals = len(f_history)
    ax2.plot(range(1, n_evals + 1), f_history, "b-o", markersize=4, lw=0.8)
    ax2.set_xlabel("评估轮次")
    ax2.set_ylabel("目标函数 f")
    ax2.set_title("目标函数 f 随评估轮次变化")
    ax2.grid(True, alpha=0.3)
    if n_evals > 0 and min(f_history) < max(f_history):
        ax2.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  图像已保存: {save_path}")
    plt.show()


# =============================================================================
# 8. 主入口
# =============================================================================

def main() -> None:
    # 🔥 完全自动化：自动检测数据缺失，自动增强，自动训练
    # 确保对任何数据都能成功拟合
    base_dir = os.path.dirname(__file__)
    
    # 优先级1：使用增强数据（如果存在）
    excel_path_augmented = os.path.join(base_dir, "5_augmented_spline.xls")
    excel_path_augmented_physics = os.path.join(base_dir, "5_augmented_physics.xls")
    excel_path_original = os.path.join(base_dir, "5.xls")
    
    # 自动选择数据文件
    if os.path.exists(excel_path_augmented):
        excel_path = excel_path_augmented
        print("✓ 使用增强数据（样条插值）:", excel_path)
    elif os.path.exists(excel_path_augmented_physics):
        excel_path = excel_path_augmented_physics
        print("✓ 使用增强数据（物理模型）:", excel_path)
    else:
        excel_path = excel_path_original
        print("📂 加载原始数据:", excel_path)
    
    # 加载数据
    V, I_meas, _, _, I_min, I_max = load_excel_and_preprocess(excel_path)
    print(f"  V.shape={V.shape}, I.shape={I_meas.shape}, I_min={I_min:.2e}, I_max={I_max:.2e}")
    
    # 🔥 自动检测并增强数据（如果膝盖区域数据缺失）
    curve_feat = extract_curve_features(V, I_meas)
    V_oc = curve_feat[0]
    knee_low = 0.3 * V_oc
    knee_high = 0.7 * V_oc
    knee_mask = (V >= knee_low) & (V < knee_high)
    knee_count = np.sum(knee_mask)
    
    if knee_count < 5:
        print(f"\n⚠ 检测到膝盖区域数据缺失（只有 {knee_count} 个点）")
        print(f"   🔧 自动进行数据增强...")
        
        # 导入数据增强函数
        try:
            from data_augmentation import augment_data_with_interpolation
            V, I_meas = augment_data_with_interpolation(V, I_meas, knee_region_points=15, method='spline')
            print(f"   ✓ 数据增强完成：现在有 {len(V)} 个数据点")
            
            # 更新统计
            knee_mask_new = (V >= knee_low) & (V < knee_high)
            knee_count_new = np.sum(knee_mask_new)
            print(f"   ✓ 膝盖区域现在有 {knee_count_new} 个数据点")
        except Exception as e:
            print(f"   ⚠ 数据增强失败：{e}")
            print(f"   → 继续使用原始数据（形状约束会自动处理数据缺失）")
    else:
        print(f"\n✓ 数据质量良好（膝盖区域有 {knee_count} 个数据点）")

    # 改进：使用更稳定的超参数，提升训练稳定性和泛化能力
    fitter = ACSolarFitter(
        V, I_meas, I_min, I_max,
        T_max=30,  # 保持30，确保足够的优化步数，不影响训练效果
        gamma=0.99,
        alpha=0.5,
        lr_actor=3.5e-4,  # 🔥 小幅提高学习率：从3e-4提高到3.5e-4（+17%），帮助跳出局部最优（实际=5.25e-4）
        lr_critic=4.5e-4,  # 🔥 小幅提高学习率：从4e-4提高到4.5e-4（+12.5%），帮助跳出局部最优（实际=6.75e-4）
    )

    # 🔍 检查训练设备（改进：更详细的GPU信息）
    device_info = fitter.device
    print("\n" + "=" * 70)
    print("🖥️ 训练设备信息")
    print("=" * 70)
    
    if str(device_info) == 'cpu':
        print("⚠ 警告：使用CPU训练（会很慢！）")
        print("\n可能的原因：")
        print("   1. 没有NVIDIA GPU")
        print("   2. CUDA驱动未安装或版本不匹配")
        print("   3. PyTorch未编译CUDA支持")
        print("\n建议：")
        print("   1. 运行诊断脚本: python check_gpu.py")
        print("   2. 运行修复脚本: python fix_gpu.py")
        print("   3. 检查NVIDIA驱动和CUDA安装")
        print("\n快速测试：将使用减少的训练参数（减少计算量）")
        use_fast_config = True
    else:
        print(f"✓ 使用GPU训练: {device_info}")
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU名称: {gpu_name}")
            print(f"   显存: {gpu_memory:.2f} GB")
            print(f"   CUDA版本: {torch.version.cuda}")
            print("\n✅ GPU环境正常，训练速度会快很多！")
        except Exception as e:
            print(f"   ⚠ 无法获取GPU详细信息: {e}")
        use_fast_config = False
    
    print("方案 B 多步 Actor-Critic 训练开始 (改进版：提升泛化能力和训练稳定性)")
    print(f"  数据特征: I_sc={np.max(I_meas):.4f}A, V_oc={np.max(V):.4f}V")
    
    # 根据设备选择训练参数
    if use_fast_config:
        # CPU训练：使用快速配置（减少计算量）
        print("   使用快速配置（CPU模式）：episodes_per_epoch=32, T_max=15")
        # 临时修改T_max（仅用于快速测试）
        fitter.T_max = 15
        best_params, best_f, f_history = fitter.fit(
            n_epochs=50,  # 减少训练轮数
            episodes_per_epoch=32,  # 减少episode数（128→32，减少4倍）
            eval_interval=5,
            early_stop_patience=20,
            early_stop_min_delta=1e-5,  # 放宽早停条件
        )
    else:
        # GPU训练：使用优化配置（平衡速度和效果）
        print("   使用优化配置：episodes_per_epoch=64, T_max=30")
        print("   提示：虽然GPU已启用，但物理模型计算在CPU上，这是性能瓶颈")
        print("   如果训练仍然慢，可以进一步减少episodes_per_epoch或T_max")
        best_params, best_f, f_history = fitter.fit(
            n_epochs=200,  # 从300减少到200，平衡速度和效果
            episodes_per_epoch=64,  # 从128减少到64，速度提升约2倍
            eval_interval=10,
            early_stop_patience=40,  # 从50减少到40
            early_stop_min_delta=1e-6,  # 早停：改善小于1e-6认为没有改善
        )

    print("\n优化完成。")
    print(f"  best f = {best_f:.6e}")
    # 🔥 双二极管模型：输出7个参数（如果固定n1和n2，会显示固定值）
    if len(best_params) == 7:
        print("  最优参数 [I_ph, I01, I02, n1, n2, Rs, Rsh] (双二极管模型):")
        param_names = ["I_ph", "I01", "I02", "n1", "n2", "Rs", "Rsh"]
        for i, name in enumerate(param_names):
            if name == "n1" and TrainingConfig.FIX_N1:
                print(f"    {name} = {TrainingConfig.FIXED_N1_VALUE:.6f} (固定)")
            elif name == "n2" and TrainingConfig.FIX_N2:
                print(f"    {name} = {TrainingConfig.FIXED_N2_VALUE:.6f} (固定)")
            else:
                fmt = ".6e" if "I0" in name or "I_ph" in name else ".6f"
                print(f"    {name} = {best_params[i]:{fmt}}")
    elif len(best_params) == 5:
        print("  最优参数 [I_ph, I01, I02, Rs, Rsh] (双二极管模型，n1和n2固定):")
        print(f"    I_ph = {best_params[0]:.6e} A")
        print(f"    I01  = {best_params[1]:.2e} A")
        print(f"    I02  = {best_params[2]:.2e} A")
        print(f"    n1   = {TrainingConfig.FIXED_N1_VALUE:.6f} (固定)")
        print(f"    n2   = {TrainingConfig.FIXED_N2_VALUE:.6f} (固定)")
        print(f"    Rs   = {best_params[3]:.6f} Ω")
        print(f"    Rsh  = {best_params[4]:.6f} Ω")
    else:
        print("  最优参数 [I_ph, I0, n, Rs, Rsh] (单二极管模型):")
        for name, val in zip(["I_ph", "I0", "n", "Rs", "Rsh"], best_params):
            fmt = ".6e" if "I0" in name or "I_ph" in name else ".6f"
            print(f"    {name} = {val:{fmt}}")

    # 拟合结果可视化
    save_path = os.path.join(os.path.dirname(__file__), "AC_fit_result.png")
    plot_fit_result(V, I_meas, best_params, I_min, I_max, best_f, f_history, save_path=save_path)


if __name__ == "__main__":
    main()
