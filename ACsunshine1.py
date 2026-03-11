from __future__ import annotations

from typing import Tuple, Dict, Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from torch import optim
import os

# 修复 matplotlib 中文乱码：使用支持中文的字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi", "FangSong"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方框
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class ObjectiveFunctionError(Exception):
    """当目标函数计算失败时抛出的异常"""
    pass

class Actor(nn.Module):
    def __init__(self,state_dim=9,action_dim=5,hidden_dim=256):
        super(Actor,self).__init__()

        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)


        self.action_scale_base = 0.1  # 默认步长

    def get_action_scale(self, best_error: float | None = None) -> float:
        """根据当前最佳误差自适应调整步长，误差越低步长越小便于精细搜索"""
        if best_error is None:
            return self.action_scale_base
        # 只有当误差非常低时才使用极小步长
        if best_error <= 0.05:
            return 0.001
        if best_error <= 0.10:
            return 0.002
        if best_error <= 0.15:
            return 0.005
        if best_error <= 0.20:
            return 0.01
        if best_error <= 0.30:
            return 0.02      # 误差在 0.2~0.3 之间时步长 0.02
        if best_error <= 0.40:
            return 0.05      # 误差在 0.3~0.4 时步长 0.05
        return 0.1            # 误差 >0.4 时使用基础步长 0.1

    def forward(self,state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        raw_action =torch.tanh(self.fc3(x))
        scaled_action =raw_action * self.action_scale_base
        return scaled_action

class Env:
    def __init__(self,excel_path,params_bounds=None):
        super(Env,self).__init__()
        self.excel_path=excel_path
        self.V, self.I, self.I_min, self.I_max = self._load_data()

        # 使用 float64：I0 下界可能设为 1e-60 等极小值，float32 会下溢为 0 导致 log(0)=-inf 报错
        self.param_bounds = np.array([
            [0.1, 10.0],   # I_ph
            [1e-60, 1e-6], # I0（float64 可表示 1e-60，float32 最小约 1.2e-38）
            [0.8, 1.5],    # n
            [0.001, 0.3],  # Rs
            [10.0, 300.0], # Rsh
        ], dtype=np.float64)
        # I0 对数空间边界（与传统优化一致，便于 RL 精细调整）
        self._log_I0_low = np.log(max(self.param_bounds[1, 0], 1e-80))
        self._log_I0_high = np.log(self.param_bounds[1, 1])

        self.errors={}

        self.default_params=np.array([
            5.05,  # I_ph: (0.1+10.0)/2 ≈ 5.05
            5.05e-10,  # I0: 几何中值 √(1e-15*1e-9)=1e-12，取5.05e-10
            1.1,  # n: (0.9+1.3)/2 = 1.1
            0.0775,  # Rs: (0.005+0.15)/2 = 0.0775
            77.5,  # Rsh: (5.0+150.0)/2 = 77.5
        ])
        self.state_dim=9
        self.action_dim=5
        # 环境状态
        self.current_params = None
        self.current_state = None
        self.step_count = 0
        self.best_error = float('inf')
        self.best_params = None
        self.prev_error = None
        self.prev_params = None
        self.rl_start_error = 0.5  # 传统优化初始误差，供_build_state精细归一化用，main中会覆盖

        # 计算热电压 (假设温度为25°C)
        self.Vt = 0.026  # 热电压 (V)

        # 奖励函数权重：主误差为主，物理量误差为辅，引导 RL 兼顾整体与关键点
        self.reward_weights = {
            'main_error': 1.0,  # 主误差（整体拟合）
            'boundary': 0.025,  # 边界惩罚，避免参数贴边
            'mpp': 0.0,  # 最大功率点误差
            'short_circuit': 0.0,  # 短路电流误差
            'open_voltage': 0.0,  # 开路电压误差
            'action_penalty': 0,  # 动作惩罚（保持 0，由 action_scale 控制）
            'fill_factor': 0.0,  # 填充因子误差
            'step_penalty': 0,  # 步数惩罚（保持 0，允许充分探索）
        }

        # 稀疏奖励阈值和值（强化 0.1 以下目标）
        self.sparse_thresholds = {
            'excellent': 0.01,  # 全局最优（极低误差）
            'good': 0.05,  # 中期进展良好
            'medium': 0.1,  # 误差 < 0.1 里程碑
            'near_target': 0.15,  # 接近目标，引导向 0.1 收敛
        }

        self.sparse_rewards = {
            'global_optimum': 10.0,  # 全局最优收敛奖励
            'key_progress': 2.0,  # 中期关键进展奖励
            'target_01': 5.0,  # 误差 < 0.1 额外奖励，强化目标
            'slow_converge': -5.0,  # 慢收敛惩罚
            'severe_penalty': -2.0,  # 严重物理无效惩罚
        }

        # 记录已获得的稀疏奖励
        self.achieved_milestones = set()
        self.no_improvement_steps = 0  # 无改善步数计数

        # 重置环境
        self.reset()

    def _load_data(self):
        df = pd.read_excel(self.excel_path,header=None,usecols=[0,1],skiprows=1)
        V_orig =df.iloc[:,0].astype(float).values
        I_orig =df.iloc[:,1].astype(float).values

        vaild = np.isfinite(V_orig)&np.isfinite(I_orig)
        V=V_orig[vaild]
        I=I_orig[vaild]

        I_pos=I[I>0]
        if len(I_pos)==0:
            raise ValueError(f"数据文件 '{self.excel_path}' 无效：未找到任何正电流数据点（I > 0）。\n")
        I_min =float(np.min(I_pos))
        I_max =float(np.max(I_pos))
        return V, I, I_min, I_max

    def _solar_cell_model(
            self,
            V: np.ndarray,
            params: np.ndarray,
    ) -> np.ndarray:
        # 🔥 单二极管模型：5个参数 [I_ph, I0, n, Rs, Rsh]
        if len(params) != 5:
            raise ValueError(f"单二极管模型需要5个参数，得到{len(params)}个")
        I_ph, I0, n, Rs, Rsh = params

        Vt = 0.026
        clip_min, clip_max = -50.0, 150.0
        I_out = np.zeros_like(V, dtype=np.float64)
        prev_I = float(I_ph)  # 上一电压点电流，用作高压区迭代初值，利于陡降段收敛

        # 修复：clip范围应该基于物理约束，而不是数据范围
        # 电流应该在 [0, I_ph*1.5] 范围，而不是 [I_min*0.1, I_max*2]
        clip_min_current = 0.0  # 电流不能为负
        clip_max_current = I_ph * 1.5  # 电流不能超过I_ph太多（考虑测量误差）

        for i, v in enumerate(V):
            # 单二极管模型：f(I) = I - (I_ph - I0*exp - shunt)
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
            for _ in range(1):  # 预热迭代，保证初值足够好
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
            for iter_count in range(20):  # 正式迭代
                # 使用牛顿法而非固定点迭代，提高高电压区收敛性，确保陡降段能正确计算
                f_val = f(I_i)
                fp_val = f_prime(I_i)
                if abs(fp_val) > 1e-12:
                    I_new = I_i - f_val / fp_val
                else:
                    # 兜底：如果导数太小，使用简化公式
                    x = (v + I_i * Rs) / (n * Vt)
                    x_clipped = np.clip(x, clip_min, clip_max)
                    exp_term = np.exp(x_clipped) - 1.0
                    shunt = (v + I_i * Rs) / Rsh
                    I_new = I_ph - I0 * exp_term - shunt

                # 物理约束：电流在 [0, I_ph*1.5]，不强制低电压下限以免扭曲拟合
                I_new = float(np.clip(I_new, 0.0, clip_max_current))

                if abs(I_new - I_i) < 1e-8:
                    I_i = I_new
                    break
                if iter_count >= 10 and abs(I_new - I_i) < 1e-5:
                    I_i = I_new
                    break
                I_i = I_new

            # 仅当迭代结果异常时使用兜底，避免过度约束扭曲拟合
            if I_i <= 0 or not np.isfinite(I_i):
                I_i = max(I_ph - v / Rsh, I_ph)

            I_out[i] = float(I_i)
            prev_I = float(I_out[i])

        return I_out

    def _objective_function(self,params: np.ndarray, V: np.ndarray, I_meas: np.ndarray) -> float:
        """
        使用 I_max 归一化的残差，避免低电流点（开路附近）相对误差爆炸。
        与 traditional_fit_test 一致，利于传统优化收敛到 ~0.3 量级。
        """
        # 计算模拟电流
        I_sim = self._solar_cell_model(V, params)

        # 选择有效点（测量电流>0）
        valid = (I_meas > 1e-10) & np.isfinite(I_meas) & np.isfinite(I_sim) & (I_sim > 0)
        if not np.any(valid):
            return 1e10

        I_m = I_meas[valid]
        I_s = I_sim[valid]
        I_max_ref = float(np.max(I_m))  # 用测量电流最大值归一化，避免 I_m 很小时爆炸

        # 归一化残差：(I_m - I_s) / I_max_ref，尺度稳定
        residuals = (I_m - I_s) / (I_max_ref + 1e-12)
        loss = np.sqrt(np.mean(residuals ** 2))

        if not np.isfinite(loss):
            raise ObjectiveFunctionError("目标函数计算结果为无穷大或NaN")

        return float(loss)

    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """归一化参数到[0, 1]范围。I0 在对数空间归一化，与传统优化一致"""
        norm_params = np.zeros_like(params)

        for i in range(len(params)):
            if i == 1:  # I0: 对数空间
                I0_safe = max(float(params[1]), 1e-80)
                log_I0 = np.log(I0_safe)
                norm_params[i] = (log_I0 - self._log_I0_low) / (self._log_I0_high - self._log_I0_low)
            else:
                min_val, max_val = self.param_bounds[i]
                norm_params[i] = (params[i] - min_val) / (max_val - min_val)
            norm_params[i] = float(np.clip(norm_params[i], 0, 1))

        return norm_params

    def _denormalize_params(self, norm_params: np.ndarray) -> np.ndarray:
        """反归一化参数。I0 从对数空间还原。使用 float64 以保证 I0 等极小值不会下溢。"""
        params = np.zeros(len(norm_params), dtype=np.float64)
        for i in range(len(norm_params)):
            if i == 1:  # I0: 对数空间
                n = float(np.clip(norm_params[i], 0, 1))
                log_I0 = self._log_I0_low + n * (self._log_I0_high - self._log_I0_low)
                params[i] = np.exp(log_I0)
            else:
                min_val, max_val = self.param_bounds[i]
                params[i] = min_val + float(norm_params[i]) * (max_val - min_val)
        return params

    def _calculate_errors(self,params:np.ndarray)->Dict[str,float]:
        main_error =self._objective_function(params,self.V,self.I)
        I_calc = self._solar_cell_model(self.V, params)
        # MPP误差,最大功率误差，通常在拐点
        if len(self.V) > 0 and len(I_calc) > 0:
            P_meas = self.V * self.I
            P_calc = self.V * I_calc
            mpp_meas = np.max(P_meas) if len(P_meas) > 0 else 0
            mpp_calc = np.max(P_calc) if len(P_calc) > 0 else 0
            mpp_error = abs(mpp_meas - mpp_calc) / mpp_meas if mpp_meas > 0 else abs(mpp_meas - mpp_calc)
        else:
            mpp_error = 0
        # 短路电流误差（简化的计算）
        if len(self.V) > 0:
            zero_voltage_idx = np.argmin(np.abs(self.V))
            short_circuit_abs_error = abs(I_calc[zero_voltage_idx] - self.I[zero_voltage_idx])
            short_circuit_error=short_circuit_abs_error/self.I_max
            short_circuit_error=np.clip(short_circuit_error,0,1)
        else:
            short_circuit_error = 0

        #开路电压误差（归一化处理）
        min_I_idx=np.argmin(np.abs(self.I))
        V_at_min_I=self.V[min_I_idx]
        I_meas_min=self.I[min_I_idx]
        I_calc_at_Vmin=self._solar_cell_model(np.array([V_at_min_I]),params)[0]
        open_voltage_error=abs(I_calc_at_Vmin-I_meas_min)
        ov_error_norm =open_voltage_error/self.I_max
        ov_error_norm=np.clip(ov_error_norm,0,1)

        # 新增：填充因子误差
        # 测量数据的填充因子
        I_sc_meas = self.I[np.argmin(np.abs(self.V))]
        V_oc_meas = self.V[np.argmin(np.abs(self.I))]
        P_max_meas = np.max(self.V * self.I) if len(self.V) > 0 else 0
        FF_meas = P_max_meas / (V_oc_meas * I_sc_meas) if V_oc_meas > 0 and I_sc_meas > 0 else 0

        # 模拟数据的填充因子
        I_sc_sim = I_calc[np.argmin(np.abs(self.V))]
        V_oc_sim = self.V[np.argmin(np.abs(I_calc))]
        P_max_sim = np.max(self.V * I_calc) if len(self.V) > 0 else 0
        FF_sim = P_max_sim / (V_oc_sim * I_sc_sim) if V_oc_sim > 0 and I_sc_sim > 0 else 0

        fill_factor_error = abs(FF_meas - FF_sim) / FF_meas if FF_meas > 0 else abs(FF_meas - FF_sim)
        fill_factor_error = np.clip(fill_factor_error, 0, 1)  # 归一化

        return {
            'main_error':main_error,
            'mpp_error':mpp_error,
            'short_circuit_error':short_circuit_error,
            'ov_error_norm': ov_error_norm,
            'fill_factor_error': fill_factor_error,
        }

    def _check_physical_validity(self, params: np.ndarray) -> bool:
        """检查物理有效性：参数在边界内、模拟电流正常、误差未过大"""
        # 参数边界检查（允许超出5%）
        for i, (low, high) in enumerate(self.param_bounds):
            margin = (high - low) * 0.05
            if params[i] < low - margin or params[i] > high + margin:
                return False
        # 模拟电流检查
        I_sim = self._solar_cell_model(self.V, params)
        if not np.all(np.isfinite(I_sim)):
            return False
        # 主误差检查（超过1.0视为严重无效）
        main_error = self._objective_function(params, self.V, self.I)
        if main_error > 1.0:
            return False
        return True

    def _calculate_boundary_penalty(self, params: np.ndarray) -> float:
        """边界惩罚：参数越靠近边界惩罚越大，指数形式"""
        penalty = 0.0
        for i, (low, high) in enumerate(self.param_bounds):
            # 计算到边界的距离（取最近边界的距离）
            dist_to_low = params[i] - low
            dist_to_high = high - params[i]
            if dist_to_low < 0 or dist_to_high < 0:
                # 超出边界，直接给大惩罚（但严重惩罚会在物理有效性中处理）
                penalty += 10.0
            else:
                # 归一化距离
                range_len = high - low
                norm_dist = min(dist_to_low, dist_to_high) / (range_len * 0.1)  # 0.1倍范围作为尺度
                penalty += np.exp(-norm_dist)
        return penalty

    def _calculate_reward(self, action: np.ndarray, current_error: float, done: bool) -> float:
        """
        势能型奖励函数：
        - 势能定义为 -error，奖励为势能差（current - prev）；
        - 等价于奖励 = -(current_error - prev_error)，误差下降时为正；
        - 不需要额外缩放因子，仅做小范围裁剪，保持数值稳定。
        """
        if self.prev_error is None or not np.isfinite(self.prev_error):
            return 0.0

        potential = -float(current_error)
        prev_potential = -float(self.prev_error)
        reward = potential - prev_potential  # = -(current_error - prev_error)

        # 裁剪到合理范围，防止异常大步导致的不稳定
        reward = float(np.clip(reward, -0.1, 0.1))
        return reward


    def reset(self, perturb: bool = False, perturb_scale: float = 0.02, initial_params: np.ndarray | None = None) -> np.ndarray:
        """
        重置环境到初始状态。
        initial_params: 若给出，则从该参数起步（用于从历史最佳热启动）；否则从 default_params 起步。
        perturb=True 时在起步点上加小扰动。
        """
        if initial_params is not None:
            self.current_params = np.asarray(initial_params, dtype=np.float64).copy()
        else:
            self.current_params = self.default_params.copy()
        if perturb:
            norm = self._normalize_params(self.current_params)
            norm = norm + np.random.uniform(-perturb_scale, perturb_scale, 5)
            norm = np.clip(norm, 0, 1)
            self.current_params = self._denormalize_params(norm)
        self.step_count = 0
        self.prev_error = None
        self.prev_params = None
        self.achieved_milestones = set()
        self.no_improvement_steps = 0

        # 计算初始误差并构建状态；用初始解作为当前最佳，避免第一步探索就覆盖掉低误差起点
        self.errors = self._calculate_errors(self.current_params)
        self.best_error = float(self.errors['main_error'])
        self.best_params = self.current_params.copy()
        self.current_state = self._build_state(self.current_params, self.errors)
        return self.current_state

    def _build_state(self, params: np.ndarray, errors: Dict[str, float]) -> np.ndarray:
        """
        构建9维状态向量：
        - 5个归一化参数
        - 4个误差指标（主误差精细归一化、MPP误差归一化、短路误差、开路误差）
        """
        norm_params = self._normalize_params(params)

        err = errors['main_error']
        # 用传统优化初始误差作为归一化基准，保留精细梯度
        ref = max(self.rl_start_error, 1e-6)
        main_error_norm = err / ref
        main_error_norm = np.clip(main_error_norm, 0, 2)
        mpp_error_norm = errors['mpp_error']  # 已经是0~1之间（相对误差）
        sc_error_norm = errors['short_circuit_error']  # 已归一化
        ov_error_norm = errors['ov_error_norm']  # 已归一化

        state = np.array([
            norm_params[0], norm_params[1], norm_params[2], norm_params[3], norm_params[4],
            main_error_norm,
            mpp_error_norm,
            sc_error_norm,
            ov_error_norm
        ], dtype=np.float32)
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步动作。
        action: 5维数组，建议范围 [-0.2, 0.2]（归一化空间内的调整量）
        返回: (next_state, reward, done, info)
        """
        self.step_count += 1


        # 2. 参数更新（在归一化空间内）
        current_norm = self._normalize_params(self.current_params)
        new_norm = current_norm + action
        new_norm = np.clip(new_norm, 0, 1)
        new_params = self._denormalize_params(new_norm)

        # 3. 计算新参数下的误差
        current_error = self._objective_function(new_params, self.V, self.I)  # 注意传入V,I
        errors = self._calculate_errors(new_params)

        # 4. 更新最佳记录和无改善步数
        if current_error < self.best_error:
            self.best_error = current_error
            self.best_params = new_params.copy()
            self.no_improvement_steps = 0
        else:
            self.no_improvement_steps += 1

        # 5. 更新状态
        self.current_params = new_params
        self.current_state = self._build_state(new_params, errors)

        # 6. 计算奖励
        reward = self._calculate_reward(action, current_error, done=False)  # done暂时False，后面再判断

        # 7. 判断是否终止（max_steps 由外部传入，支持动态步数）
        done = self._check_done(max_steps=getattr(self, '_current_max_steps', 500))

        # 8. 保存历史（供平滑惩罚等使用）
        self.prev_error = current_error
        self.prev_params = new_params.copy()

        # 9. 构建info字典
        info = {
            'step': self.step_count,
            'objective_error': current_error,
            'best_objective_error': self.best_error,
            'params': new_params.copy(),
            'errors': errors.copy(),
            'no_improvement_steps': self.no_improvement_steps,
            'achieved_milestones': list(self.achieved_milestones),
        }

        return self.current_state, reward, done, info

    def _check_done(self, max_steps: int = 500) -> bool:
        """检查终止条件。低误差时放宽无改善步数阈值，给更多精细探索机会"""
        if self.best_error < 1e-4:
            return True
        if self.step_count >= max_steps:
            return True
        no_improve_limit = 200
        if self.no_improvement_steps >= no_improve_limit:
            return True
        return False

class Critic(nn.Module):
    def __init__(self, state_dim=9, action_dim=5, hidden_dim=512):
        super(Critic, self).__init__()
        # 将状态和动作拼接后输入
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)  # 输出 Q(s,a)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # 拼接
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)
        q = self.fc4(x)
        return q



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class TD3Agent:
    """
    TD3 (Twin Delayed DDPG) 算法，解决 Q 值过估计问题：
    1. Clipped Double Q-learning: 双 Critic 取最小值作为目标
    2. Target Policy Smoothing: 对目标动作加噪声，平滑策略
    3. Delayed Policy Updates: 延迟 Actor 更新
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005,
                 buffer_capacity=100000, batch_size=64,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.update_count = 0
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.policy_noise = policy_noise  # 目标策略平滑噪声标准差
        self.noise_clip = noise_clip      # 噪声裁剪范围
        self.policy_delay = policy_delay  # Actor 延迟更新间隔

        # 主网络
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor.id = id(self.actor)

        # TD3: 双 Critic 网络，缓解 Q 值过估计
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)

        # 目标网络
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr_critic
        )

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.noise = GaussianNoise(action_dim)

    def select_action(self, state, add_noise=True, current_best_error: float | None = None):
        """
        选择动作。
        - current_best_error 用于动态调整步长和噪声，误差低时更精细；
        - 在「误差平台区」（例如约 0.28~0.31）增加额外随机探索，帮助跳出局部最优。
        """
        self.actor.eval() 
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = self.actor(state_tensor)
            action = action_tensor.cpu().numpy()[0]
        self.actor.train()

        in_burst = self.noise.burst_remaining > 0  # sample() 会递减，需先判断

        # 基础步长：再探索期间始终使用较大步长，其余时间用误差自适应步长
        scale = self.actor.action_scale_base if in_burst else self.actor.get_action_scale(current_best_error)

        # 平台区间监测：在 0.28~0.31 之间认为可能陷入局部最优，增加随机探索
        plateau = current_best_error is not None and 0.28 <= current_best_error <= 0.31

        if plateau and np.random.rand() < 0.3:
            # 以一定概率忽略当前策略，直接在 [-scale, scale] 上均匀采样，强制跳出平台
            action = np.random.uniform(-scale, scale, size=self.action_dim)
        else:
            if add_noise:
                noise = self.noise.sample(current_best_error=current_best_error)
                action += noise

        return np.clip(action, -scale, scale)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        self.update_count += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # -------------------- TD3: 更新双 Critic --------------------
        with torch.no_grad():
            # 目标策略平滑：对目标动作加噪声，缓解 Q 过估计
            next_actions = self.actor_target(next_states)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions_smooth = (next_actions + noise).clamp(
                -self.actor.action_scale_base, self.actor.action_scale_base
            )

            # Clipped Double Q: 取两个 Q 的最小值作为目标，减少过估计
            target_q1 = self.critic1_target(next_states, next_actions_smooth)
            target_q2 = self.critic2_target(next_states, next_actions_smooth)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = nn.SmoothL1Loss()(current_q1, target_q)
        critic2_loss = nn.SmoothL1Loss()(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss

        self.optim_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.optim_critic.step()

        # -------------------- TD3: 延迟策略更新 --------------------
        if self.update_count % self.policy_delay == 0:
            # Actor 仅使用 Q1 的梯度（减少方差）
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.optim_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.optim_actor.step()

            # 软更新目标网络
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        else:
            with torch.no_grad():
                actor_loss = -self.critic1(states, self.actor(states)).mean()

        # 周期性再探索：每 3000 次 update 强制放大噪声 200 步，跳出局部最优
        if self.update_count > 0 and self.update_count % 3000 == 0:
            self.noise.start_burst(200)
            print(f"[再探索] Update {self.update_count}: 启动 200 步大探索，噪声重置为 {self.noise.std_original:.3f}")

        if self.update_count % 100 == 0:
            q_mean = (current_q1.mean().item() + current_q2.mean().item()) / 2
            print(f"Update {self.update_count}: Critic loss = {critic_loss.item():.4f}, "
                  f"Q mean = {q_mean:.4f}, Actor loss = {actor_loss.item():.4f}")

class GaussianNoise:
    """
    用于 DDPG 探索的高斯噪声生成器。
    支持根据 current_best_error 动态调整噪声：误差低时噪声更小，便于精细探索。
    支持周期性「再探索」：强制放大噪声以跳出局部最优。
    """
    def __init__(self, action_dim, std=0.1, std_decay=0.999, std_min=0.015):
        self.action_dim = action_dim
        self.std = std
        self.std_decay = std_decay
        self.std_min = std_min
        self.std_original = std  # 保存初始值，便于重置
        self.burst_remaining = 0  # 再探索剩余步数，>0 时使用 std_original 强制大探索
        self.burst_boost = 1.0  # 再探索时噪声放大倍数，start_burst 会设为 1.5

    def start_burst(self, steps: int = 200):
        """启动再探索：在接下来 steps 步内使用 1.5 倍初始噪声，强制跳出舒适区"""
        self.burst_remaining = steps
        self.burst_boost = 1.5  # 再探索时噪声放大倍数

    def _get_effective_std(self, current_best_error: float | None) -> float:
        """根据当前误差返回有效噪声标准差：误差低时噪声更小"""
        base = max(self.std * self.std_decay, self.std_min)
        self.std = base  # 更新衰减
        if current_best_error is None:
            return base
        if current_best_error <= 0.05:
            return max(base * 0.02, 0.0005)
        if current_best_error <= 0.10:
            return max(base * 0.05, 0.001)
        if current_best_error <= 0.15:
            return max(base * 0.1, 0.002)
        if current_best_error <= 0.20:
            return max(base * 0.2, 0.005)
        if current_best_error <= 0.30:
            return max(base * 0.4, 0.01)   # 误差 0.2~0.3 时，噪声约为 base*0.4
        if current_best_error <= 0.40:
            return max(base * 0.6, 0.02)   # 误差 0.3~0.4 时，噪声更大
        return base

    def sample(self, current_best_error: float | None = None):
        """
        生成一个噪声向量，形状为 (action_dim,)。
        current_best_error: 当前最佳误差，用于动态缩小噪声（低误差时更精细探索）。
        当 burst_remaining > 0 时，强制使用 std_original 进行大探索，跳出局部最优。
        """
        if self.burst_remaining > 0:
            self.burst_remaining -= 1
            burst_std = self.std_original * getattr(self, 'burst_boost', 1.0)
            return np.random.normal(0, burst_std, size=self.action_dim)
        eff_std = self._get_effective_std(current_best_error)
        return np.random.normal(0, eff_std, size=self.action_dim)

    def reset(self):
        """将标准差重置为初始值。每个 episode 开始时调用。"""
        self.std = self.std_original


def diagnose_critic(agent: TD3Agent, env: Env, num_episodes: int = 5, rollout_len: int = 200):
    """
    使用当前策略对环境进行若干次采样，比较：
    - MC 折扣回报 G_t
    - Critic 预测的 Q(s_t, a_t)
    计算两者的皮尔逊相关系数 corr，判断价值网络打分是否合理。
    """
    agent.actor.eval()
    agent.critic1.eval()

    returns = []
    q_values = []

    for _ in range(num_episodes):
        state = env.reset(perturb=False)
        episode_states = []
        episode_actions = []
        episode_rewards = []

        # 用当前策略（不加噪声）采样一条轨迹
        for _ in range(rollout_len):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_tensor = agent.actor(state_tensor)
            action = action_tensor.cpu().numpy()[0]

            next_state, reward, done, info = env.step(action)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state
            if done:
                break

        # 从后往前计算每个时间步的折扣回报 G_t
        G = 0.0
        disc_returns = []
        for r in reversed(episode_rewards):
            G = r + agent.gamma * G
            disc_returns.insert(0, G)

        # 收集对应的 Q 预测和 MC 回报
        for s, a, g in zip(episode_states, episode_actions, disc_returns):
            s_tensor = torch.FloatTensor(s).unsqueeze(0)
            a_tensor = torch.FloatTensor(a).unsqueeze(0)
            with torch.no_grad():
                q_pred = agent.critic1(s_tensor, a_tensor).item()
            returns.append(g)
            q_values.append(q_pred)

    agent.actor.train()
    agent.critic1.train()

    returns = np.array(returns, dtype=np.float32)
    q_values = np.array(q_values, dtype=np.float32)

    if len(returns) > 1:
        corr = np.corrcoef(returns, q_values)[0, 1]
    else:
        corr = float("nan")

    print(f"[诊断] Critic Q 与 MC 回报相关系数 corr = {corr:.3f}")
    print(f"[诊断] 回报范围: {returns.min():.3f} ~ {returns.max():.3f}")
    print(f"[诊断] Q值范围: {q_values.min():.3f} ~ {q_values.max():.3f}")


def main():
    # ========== 超参数设置 ==========
    EXCEL_PATH = r"C:\Users\18372\PycharmProjects\pythonProject1\2 (1).xls"  # 请替换为实际数据文件路径
    STATE_DIM = 9
    ACTION_DIM = 5
    HIDDEN_DIM = 256  # 提高容量，避免表达能力不足导致误差卡在 0.34 附近
    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-4  # 进一步降低以稳定 Critic，减少 loss 尖峰
    GAMMA = 0.95
    TAU = 0.001  # 目标网络更新更慢，Q 估计更稳定
    BUFFER_CAPACITY = 100000
    BATCH_SIZE = 128  # 更大 batch 使梯度更平滑
    NUM_EPISODES = 800  # 给 RL 更多时间探索
    MAX_STEPS_BASE = 500  # 每 episode 最多步数，避免低效重复
    MAX_STEPS_LOW_ERR = 500  # 与 BASE 一致
    NOISE_STD = 0.08   # 原0.06→0.08，初始噪声更大
    NOISE_DECAY = 0.9995  # 原0.998→0.9995，衰减更慢，探索期更长
    NOISE_MIN = 0.015  # 原0.008→0.015，最小噪声更大，后期仍有探索
    TARGET_ERROR = 1e-4  # 达到此误差提前停止
    PERTURB_SCALE = 0.04  # 常规扰动幅度
    PERTURB_SCALE_STRONG = 0.08  # 卡住时强制大扰动
    STUCK_THRESHOLD = 25  # 连续多少 episode 无改善则视为卡住
    SAVE_DIR = "./models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ========== 初始化环境和智能体 ==========
    env = Env(excel_path=EXCEL_PATH)

    # ---------- 传统优化：用 scipy 拟合得到更好的默认参数 ----------
    # I0 跨多个数量级，在 log 空间优化更易收敛；边界与 Env 一致，从 param_bounds 读取
    I0_LOW = float(env.param_bounds[1, 0])
    I0_HIGH = float(env.param_bounds[1, 1])
    log_I0_low = np.log(I0_LOW)
    log_I0_high = np.log(I0_HIGH)

    def _x_to_params(x):
        """优化变量 x = [I_ph, log_I0, n, Rs, Rsh] -> 物理参数 [I_ph, I0, n, Rs, Rsh]"""
        return np.array([x[0], np.exp(x[1]), x[2], x[3], x[4]], dtype=np.float64)

    def _scipy_objective(x):
        """供 scipy 调用的目标函数。x 中 I0 为 log(I0)。"""
        try:
            params = _x_to_params(x)
            return env._objective_function(params, env.V, env.I)
        except Exception:
            return 1e10

    bounds_linear = [
        (float(env.param_bounds[i, 0]), float(env.param_bounds[i, 1]))
        for i in range(5)
    ]
    bounds_log_I0 = [
        (bounds_linear[0][0], bounds_linear[0][1]),   # I_ph
        (log_I0_low, log_I0_high),                       # log(I0)
        (bounds_linear[2][0], bounds_linear[2][1]),   # n
        (bounds_linear[3][0], bounds_linear[3][1]),   # Rs
        (bounds_linear[4][0], bounds_linear[4][1]),   # Rsh
    ]
    # 使用数据驱动的初值：I_ph ≈ I_sc（短路电流）
    I_sc_data = float(np.max(env.I))
    x0_linear = env.default_params.copy()
    x0_linear[0] = np.clip(I_sc_data * 1.02, env.param_bounds[0, 0], env.param_bounds[0, 1])
    x0 = np.array([
        x0_linear[0],
        np.log(max(x0_linear[1], 1e-20)),
        x0_linear[2], x0_linear[3], x0_linear[4]
    ])
    initial_error = _scipy_objective(x0)
    print(f"[传统优化] 初始误差 (default_params): {initial_error:.6f}")

    # 多组随机种子跑 DE，取最优；同时保留若干较优解作为 RL 的候选起点
    best_de_x = x0.copy()
    best_de_fun = float("inf")
    de_seeds = [42, 123, 456, 789, 2024]  # 5 个种子，提高找到更优解概率
    de_candidates: list[tuple[float, np.ndarray]] = []
    for run, seed in enumerate(de_seeds):
        print(f"[传统优化] 全局优化 第 {run+1}/{len(de_seeds)} 次 (seed={seed})，请稍候...")
        result_de = differential_evolution(
            _scipy_objective,
            bounds_log_I0,
            strategy="best1bin",
            maxiter=1000,
            popsize=50,
            tol=1e-6,
            seed=seed,
            polish=True,
            disp=False,
            atol=1e-8,
        )
        de_candidates.append((float(result_de.fun), result_de.x.copy()))
        if result_de.fun < best_de_fun:
            best_de_fun = result_de.fun
            best_de_x = result_de.x
    result_de.x = best_de_x
    result_de.fun = best_de_fun
    best_params_de = _x_to_params(best_de_x)
    error_de = best_de_fun
    if error_de < initial_error:
        env.default_params = best_params_de
        print(f"[传统优化] 全局优化最佳误差: {error_de:.6f}，将作为 RL 的默认参数")
    else:
        print(f"[传统优化] 全局优化未优于初值 (error={error_de:.6f})，保留原 default_params")

    # 选取若干较优解作为候选起点（按误差排序取前 5 个）
    de_candidates.sort(key=lambda t: t[0])
    candidate_params: list[np.ndarray] = []
    for fun, x_sol in de_candidates[:5]:
        try:
            candidate_params.append(_x_to_params(x_sol))
        except Exception:
            continue

    # 在全局结果基础上用 L-BFGS-B 精修（两轮精修，第二轮从第一轮结果出发）
    try:
        x_start = result_de.x.copy()
        best_so_far = float(result_de.fun)
        for pass_idx in range(1):  # 1轮精修，留优化空间给 RL
            result_lbfgs = minimize(
                _scipy_objective, x_start,
                method="L-BFGS-B", bounds=bounds_log_I0,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            if result_lbfgs.success and result_lbfgs.fun < best_so_far:
                env.default_params = _x_to_params(result_lbfgs.x)
                x_start = result_lbfgs.x.copy()
                best_so_far = result_lbfgs.fun
                print(f"[传统优化] L-BFGS-B 第{pass_idx+1}轮精修后误差: {result_lbfgs.fun:.6f}")
            else:
                break
    except Exception:
        pass

    # 明确打印 RL 起点误差，便于排查“误差卡在 0.34”是否因传统优化只做到该水平
    rl_start_error = env._objective_function(env.default_params, env.V, env.I)
    env.rl_start_error = rl_start_error
    print(f"[传统优化] RL 将从此误差起点开始: {rl_start_error:.6f}")

    # 诊断：误差主要来自哪些点（便于判断是否被少数点“绑架”）
    try:
        I_sim = env._solar_cell_model(env.V, env.default_params)
        valid = env.I > 1e-10
        rel_err = np.zeros_like(env.I)
        rel_err[valid] = np.abs((env.I[valid] - I_sim[valid]) / env.I[valid])
        idx = np.argsort(rel_err)[::-1][:5]
        print("[传统优化] 相对误差最大的 5 个点: V, I_meas, rel_err =")
        for i in idx:
            if valid[i]:
                print(f"  {env.V[i]:.4f}, {env.I[i]:.6e}, {rel_err[i]:.4f}")
    except Exception:
        pass

    agent = TD3Agent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
        tau=TAU,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        policy_noise=0.1,   # 适当增大目标策略平滑噪声，利于探索
        noise_clip=0.15,   # 噪声裁剪范围略放宽
        policy_delay=2      # 每 2 次 Critic 更新 1 次 Actor
    )
    # 替换默认噪声（也可在Agent初始化时直接传入参数）
    agent.noise = GaussianNoise(ACTION_DIM, std=NOISE_STD, std_decay=NOISE_DECAY, std_min=NOISE_MIN)

    # ========== 训练记录 ==========
    best_error_overall = float('inf')
    best_params_overall = None
    episodes_without_improvement = 0  # 连续无改善 episode 数
    episode_rewards = []
    episode_errors = []

    # ========== 训练循环 ==========
    for episode in range(1, NUM_EPISODES + 1):
        # 起点选择策略：
        # - 若检测到长期无改进：从传统优化默认解出发，施加强扰动，尝试跳出局部最优；
        # - 每 5 个 episode：从 env.default_params 加较大扰动，打破「总是从历史最佳起步」的桎梏；
        # - 其它情况：在若干候选较优解之间轮换起点，并施加中等扰动，增加参数空间覆盖度。
        is_stuck = episodes_without_improvement >= STUCK_THRESHOLD
        if is_stuck:
            use_perturb = True
            perturb_scale = PERTURB_SCALE_STRONG
            start_params = env.default_params.copy()
            episodes_without_improvement = 0
            print(f"[卡住检测] Episode {episode}: 强制从 default_params 大扰动重启 (scale={perturb_scale})")
        elif episode % 5 == 0:
            # 周期性从传统优化默认解出发并施加强扰动
            use_perturb = True
            perturb_scale = PERTURB_SCALE_STRONG
            start_params = env.default_params.copy()
            print(f"[多样起点] Episode {episode}: 从 default_params 强扰动起步 (scale={perturb_scale})")
        elif len(candidate_params) > 0:
            # 在若干 DE 得到的较优解之间轮换作为起点
            use_perturb = True
            perturb_scale = PERTURB_SCALE
            idx = (episode - 1) % len(candidate_params)
            start_params = candidate_params[idx].copy()
            print(f"[多样起点] Episode {episode}: 使用候选解 #{idx+1} 并加扰动起步")
        else:
            # 回退策略：从全局历史最佳起步
            use_perturb = True
            perturb_scale = PERTURB_SCALE
            start_params = best_params_overall if best_params_overall is not None else None

        state = env.reset(perturb=use_perturb, perturb_scale=perturb_scale, initial_params=start_params)
        episode_reward = 0
        agent.noise.reset()  # 每个episode重置噪声，让探索强度重新开始

        # 低误差时增加步数，给 RL 更多精细探索机会
        max_steps = MAX_STEPS_LOW_ERR if best_error_overall < 0.32 else MAX_STEPS_BASE
        env._current_max_steps = max_steps

        for step in range(max_steps):
            # 选择动作（添加噪声）；传入当前最佳误差，低误差时步长和噪声更小
            action = agent.select_action(state, add_noise=True, current_best_error=env.best_error)
            # 环境交互
            next_state, reward, done, info = env.step(action)
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            # 更新网络
            agent.update()
            # 转移状态
            state = next_state
            episode_reward += reward
            if done:
                break

        # 记录本episode的统计
        episode_rewards.append(episode_reward)
        episode_errors.append(env.best_error)
        if env.best_error < best_error_overall:
            best_error_overall = env.best_error
            best_params_overall = env.best_params.copy()
            episodes_without_improvement = 0
            # 保存最佳模型
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic1': agent.critic1.state_dict(),
                'critic2': agent.critic2.state_dict(),
                'best_error': best_error_overall,
            }, os.path.join(SAVE_DIR, 'best_model.pth'))
        elif best_params_overall is None:
            best_params_overall = env.best_params.copy()
            best_error_overall = env.best_error
        else:
            episodes_without_improvement += 1

        # 打印进度（带热启动提示）
        warm = " [从历史最佳热启动]" if (episode > 1 and start_params is not None) else ""
        print(f"Episode {episode:3d} | Reward: {episode_reward:8.2f} | Best Error: {env.best_error:.6f} | 全局最佳: {best_error_overall:.6f}{warm}")

        # 周期性诊断 Critic 质量（每 20 个 episode 一次）
        if episode % 20 == 0:
            try:
                diagnose_critic(agent, env, num_episodes=3, rollout_len=200)
            except Exception as e:
                print(f"[诊断] Episode {episode} 期间 Critic 诊断出错: {e}")

        # 提前停止
        if best_error_overall < TARGET_ERROR:
            print(f"🎯 Target error reached at episode {episode}. Stopping training.")
            break

    print("训练完成！")
    print(f"最佳误差: {best_error_overall}")

    # 训练结束后，对 Critic 进行一次诊断
    try:
        diagnose_critic(agent, env, num_episodes=5, rollout_len=200)
    except Exception as e:
        print(f"[诊断] 运行 Critic 诊断时出错: {e}")

    # 可选：绘制奖励和误差曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode Rewards")
        plt.subplot(1, 2, 2)
        plt.plot(episode_errors)
        plt.xlabel("Episode")
        plt.ylabel("Best Error")
        plt.title("Best Error per Episode")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"))
        plt.show()
    except ImportError:
        print("matplotlib未安装，跳过绘图。")


if __name__ == "__main__":
    main()