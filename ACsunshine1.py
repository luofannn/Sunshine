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
    def __init__(self,state_dim=10,action_dim=5,hidden_dim=256):
        super(Actor,self).__init__()

        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)


        self.action_scale = 0.1  # 更小步长，便于在 0.28 附近做精细调整

    def forward(self,state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        raw_action =torch.tanh(self.fc3(x))
        scaled_action =raw_action * self.action_scale
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

        self.errors={}

        self.default_params=np.array([
            5.05,  # I_ph: (0.1+10.0)/2 ≈ 5.05
            5.05e-10,  # I0: 几何中值 √(1e-15*1e-9)=1e-12，取5.05e-10
            1.1,  # n: (0.9+1.3)/2 = 1.1
            0.0775,  # Rs: (0.005+0.15)/2 = 0.0775
            77.5,  # Rsh: (5.0+150.0)/2 = 77.5
        ])
        self.state_dim=10
        self.action_dim=5
        # 环境状态
        self.current_params = None
        self.current_state = None
        self.step_count = 0
        self.best_error = float('inf')
        self.best_params = None
        self.prev_error = None
        self.prev_params = None

        # 计算热电压 (假设温度为25°C)
        self.Vt = 0.026  # 热电压 (V)

        # 奖励函数权重：主误差为主，物理量误差为辅，引导 RL 兼顾整体与关键点
        self.reward_weights = {
            'main_error': 1.0,  # 主误差（整体拟合）
            'boundary': 0.025,  # 边界惩罚，避免参数贴边
            'mpp': 0.25,  # 最大功率点误差，拐点区域关键
            'short_circuit': 0.15,  # 短路电流误差
            'open_voltage': 0.15,  # 开路电压误差
            'action_penalty': 0,  # 动作惩罚（保持 0，由 action_scale 控制）
            'fill_factor': 0.1,  # 填充因子误差，反映整体曲线形状
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
            for _ in range(3):  # 预热迭代，保证初值足够好
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
            for iter_count in range(50):  # 恢复足够迭代，保证拟合阶段模型精度
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
                if iter_count >= 25 and abs(I_new - I_i) < 1e-5:
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
        """归一化参数到[0, 1]范围"""
        norm_params = np.zeros_like(params)

        for i in range(len(params)):
            min_val, max_val = self.param_bounds[i]
            norm_params[i] = (params[i] - min_val) / (max_val - min_val)
            norm_params[i] = np.clip(norm_params[i], 0, 1)

        return norm_params

    def _denormalize_params(self, norm_params: np.ndarray) -> np.ndarray:
        """反归一化参数。使用 float64 以保证 I0 等极小值（如 1e-60）不会在 float32 中下溢为 0。"""
        params = np.zeros(len(norm_params), dtype=np.float64)
        for i in range(len(norm_params)):
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
        综合奖励函数（重构：只有降误差才有奖，不降就惩罚，避免不动就有奖导致局部最优）
        - 核心：奖励与“本步误差相对上一步的下降量”挂钩，不降则给惩罚。
        """
        errors = self._calculate_errors(self.current_params)
        reward = 0.0

        # 1. 核心：仅凭“误差是否下降”给奖/罚（不再有 base_reward，不动无奖）
        if self.prev_error is not None:
            delta = self.prev_error - current_error  # 正值表示误差下降
            if delta > 0:
                # 降误差才有奖：奖励与下降量成正比，低误差区间放大系数
                scale = 80.0 if current_error <= 0.35 else 20.0
                reward += scale * delta
            else:
                # 不降就惩罚；惩罚不宜过大，否则 Q 全负、策略梯度难以学习（易卡在 0.34）
                reward -= 0.35
        # 第一步 prev_error 为 None，不奖不罚改进项，仅保留边界/物理惩罚

        # 2. 边界惩罚（次要，避免参数贴边）
        boundary_pen = self._calculate_boundary_penalty(self.current_params)
        reward -= self.reward_weights['boundary'] * boundary_pen

        # 3. 各物理量误差惩罚（次要，引导曲线形状）
        reward -= self.reward_weights['mpp'] * (errors['mpp_error']**2)
        reward -= self.reward_weights['short_circuit'] * (errors['short_circuit_error']**2)
        reward -= self.reward_weights['open_voltage'] * (errors['ov_error_norm']**2)
        reward -= self.reward_weights['fill_factor'] * (errors['fill_factor_error']**2)

        # 4. 稀疏奖励：仅当“首次突破”某误差阈值时给一次性奖励（属于降误差的里程碑）
        milestone_map = {
            'init_break': 0.30,
            'break_028': 0.28,
            'break_025': 0.25,
            'break_020': 0.20
        }
        for milestone, thr in milestone_map.items():
            if current_error < thr and milestone not in self.achieved_milestones:
                reward += 10.0 if milestone == 'break_028' else 5.0
                self.achieved_milestones.add(milestone)

        if current_error < self.sparse_thresholds['excellent'] and 'excellent' not in self.achieved_milestones:
            reward += self.sparse_rewards['global_optimum'] * 1.0
            self.achieved_milestones.add('excellent')

        # 5. 连续多步无改善时的额外惩罚
        if self.no_improvement_steps > 100:
            reward -= 2.0
            self.no_improvement_steps = 0

        reward = np.clip(reward, -5.0, 15.0)  # 限制奖励范围，稳定Q值估计

        return float(reward)


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
        构建10维状态向量：
        - 5个归一化参数
        - 5个误差指标（主误差归一化、步数归一化、MPP误差归一化、短路误差、开路误差）
        """
        norm_params = self._normalize_params(params)

        err = errors['main_error']
        main_error_norm = err/0.5
        main_error_norm=np.clip(main_error_norm,0,1)
        step_norm = min(self.step_count / 1200.0, 1.0)  # 与 MAX_STEPS 一致
        mpp_error_norm = errors['mpp_error']  # 已经是0~1之间（相对误差）
        sc_error_norm = errors['short_circuit_error']  # 已归一化
        ov_error_norm = errors['ov_error_norm']  # 已归一化

        state = np.array([
            norm_params[0], norm_params[1], norm_params[2], norm_params[3], norm_params[4],
            main_error_norm,
            step_norm,
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

        # 7. 判断是否终止
        done = self._check_done()

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

    def _check_done(self) -> bool:
        """检查终止条件"""
        if self.best_error < 1e-4:
            return True
        if self.step_count >= 1200:
            return True
        if self.no_improvement_steps >= 200:
            return True
        return False

class Critic(nn.Module):
    def __init__(self, state_dim=10, action_dim=5, hidden_dim=512):
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

    def select_action(self, state, add_noise=True):
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = self.actor(state_tensor)
            action = action_tensor.cpu().numpy()[0]
        self.actor.train()
        if add_noise:
            noise = self.noise.sample()
            action += noise
        return np.clip(action, -self.actor.action_scale, self.actor.action_scale)

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
                -self.actor.action_scale, self.actor.action_scale
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

        if self.update_count % 100 == 0:
            q_mean = (current_q1.mean().item() + current_q2.mean().item()) / 2
            print(f"Update {self.update_count}: Critic loss = {critic_loss.item():.4f}, "
                  f"Q mean = {q_mean:.4f}, Actor loss = {actor_loss.item():.4f}")

class GaussianNoise:
    """
    用于 DDPG 探索的高斯噪声生成器。
    参数：
        action_dim (int): 动作空间的维度。
        std (float): 初始标准差，控制噪声的幅度，默认为 0.1。
        std_decay (float): 每次调用 sample() 后标准差的衰减系数，默认为 0.999。
        std_min (float): 标准差的最小值，防止衰减到零，默认为 0.01。
    """
    def __init__(self, action_dim, std=0.1, std_decay=0.999, std_min=0.01):
        self.action_dim = action_dim
        self.std = std
        self.std_decay = std_decay
        self.std_min = std_min
        self.std_original = std  # 保存初始值，便于重置

    def sample(self):
        """
        生成一个噪声向量，形状为 (action_dim,)。
        每次调用后，标准差按衰减系数减小（但不会低于最小值）。
        """
        noise = np.random.normal(0, self.std, size=self.action_dim)
        # 更新标准差
        self.std = max(self.std * self.std_decay, self.std_min)
        return noise

    def reset(self):
        """
        将标准差重置为初始值。可在每个 episode 开始时调用，
        使噪声在每个 episode 重新开始衰减。
        """
        self.std = self.std_original


def main():
    # ========== 超参数设置 ==========
    EXCEL_PATH = r"C:\Users\18372\PycharmProjects\pythonProject1\2 (1).xls"  # 请替换为实际数据文件路径
    STATE_DIM = 10
    ACTION_DIM = 5
    HIDDEN_DIM = 256  # 提高容量，避免表达能力不足导致误差卡在 0.34 附近
    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-4  # 进一步降低以稳定 Critic，减少 loss 尖峰
    GAMMA = 0.95
    TAU = 0.001  # 目标网络更新更慢，Q 估计更稳定
    BUFFER_CAPACITY = 100000
    BATCH_SIZE = 128  # 更大 batch 使梯度更平滑
    NUM_EPISODES = 500
    MAX_STEPS = 1200  # 更多步数，给 RL 足够时间从 0.28 精细调整
    NOISE_STD = 0.06   # 适当增大探索噪声，便于跳出局部最优
    NOISE_DECAY = 0.998 # 衰减稍慢，前期探索更充分
    NOISE_MIN = 0.008  # 提高最小噪声，后期仍有一定探索
    TARGET_ERROR = 1e-4  # 达到此误差提前停止
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

    # 多组随机种子跑 DE，取最优；加强搜索以争取误差 ~0.3
    best_de_x = x0.copy()
    best_de_fun = float("inf")
    de_seeds = [42, 123, 456, 789, 2024]  # 5 个种子，提高找到更优解概率
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

    # 在全局结果基础上用 L-BFGS-B 精修（两轮精修，第二轮从第一轮结果出发）
    try:
        x_start = result_de.x.copy()
        best_so_far = float(result_de.fun)
        for pass_idx in range(2):
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
    best_params_overall = None  # 历史最佳参数，用于热启动，避免每轮都从 0.34 重来
    episode_rewards = []
    episode_errors = []

    # ========== 训练循环 ==========
    for episode in range(1, NUM_EPISODES + 1):
        # 优先从历史最佳参数热启动（>1 轮且已有最佳），约 30% 从传统优化起点+扰动探索
        use_perturb = (episode > 3) and (episode % 3 == 0)
        if episode > 1 and best_params_overall is not None and not use_perturb:
            start_params = best_params_overall
        elif episode > 1 and best_params_overall is not None and use_perturb:
            start_params = best_params_overall  # 在最佳点附近扰动
        else:
            start_params = None  # 第 1 轮或显式用 default_params
        state = env.reset(perturb=use_perturb, initial_params=start_params)
        episode_reward = 0
        agent.noise.reset()  # 每个episode重置噪声，让探索强度重新开始

        for step in range(MAX_STEPS):
            # 选择动作（添加噪声）
            action = agent.select_action(state, add_noise=True)
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
            best_params_overall = env.best_params.copy()  # 下一轮从此热启动
            # 保存最佳模型
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic1': agent.critic1.state_dict(),
                'critic2': agent.critic2.state_dict(),
                'best_error': best_error_overall,
            }, os.path.join(SAVE_DIR, 'best_model.pth'))
        elif best_params_overall is None:
            # 首轮未改进时也记录当前最佳，便于第 2 轮热启动
            best_params_overall = env.best_params.copy()
            best_error_overall = env.best_error

        # 打印进度（带热启动提示）
        if episode % 5 == 0:
            warm = " [从历史最佳热启动]" if (episode > 1 and start_params is not None) else ""
            print(f"Episode {episode:3d} | Reward: {episode_reward:8.2f} | Best Error: {env.best_error:.6f} | 全局最佳: {best_error_overall:.6f}{warm}")

        # 提前停止
        if best_error_overall < TARGET_ERROR:
            print(f"🎯 Target error reached at episode {episode}. Stopping training.")
            break

    print("训练完成！")
    print(f"最佳误差: {best_error_overall}")

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