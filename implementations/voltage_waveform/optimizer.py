import numpy as np
from scipy.optimize import differential_evolution
from typing import Tuple, Dict, Any, Optional

from implementations.voltage_waveform.generators import TrapezoidalWaveGenerator, SawtoothWaveGenerator

class GAWaveformOptimizer:
    """
    基于遗传算法的波形参数优化器 (GA Waveform Optimizer)

    目标：优化步距一致性（即速度波动最小化）与能耗。
    在给定的参数边界内搜索最优的:
    对于梯形波: (amplitude, frequency, overlap_ratio)
    对于锯齿波: (amplitude, frequency, slope_ratio)

    硬性约束：
    - 步距稳定性误差 < 5% (通过罚函数在适应度函数中体现)
    - 无回退运动 (通过重叠率 5-15% 约束及目标函数体现)
    """

    def __init__(self, mode: str = "trapezoidal"):
        """
        初始化优化器。

        参数:
            mode (str): 优化的波形类型, "trapezoidal" 或 "sawtooth"
        """
        self.mode = mode
        if mode not in ["trapezoidal", "sawtooth"]:
            raise ValueError("不支持的波形模式。请使用 'trapezoidal' 或 'sawtooth'")

    def _simulate_step_distance(self, params: tuple) -> float:
        """
        模拟单步步距。
        这里使用简化的经验公式替代有限元仿真。

        对于梯形波(Clamp-Drive)：
            假设 step_dist ∝ amplitude，并且频率过高会使步距减小（低通滤波效应）
            step_dist = k * amplitude * np.exp(-frequency / 5000.0)

        对于锯齿波(Stick-Slip)：
            step_dist = k * amplitude * (1.0 - 1.0/slope_ratio) * np.exp(-frequency / 8000.0)

        参数:
            params: 梯形波为(amp, freq, overlap), 锯齿波为(amp, freq, slope)

        返回:
            float: 模拟步距 (μm)
        """
        k = 0.01  # 比例系数
        if self.mode == "trapezoidal":
            amp, freq, overlap = params
            return k * amp * np.exp(-freq / 5000.0)
        else:
            amp, freq, slope = params
            # 粘滑模型近似
            return k * amp * (1.0 - 1.0 / slope) * np.exp(-freq / 8000.0)

    def _simulate_stability_error(self, params: tuple) -> float:
        """
        模拟步距稳定性误差（由于机械系统谐振和参数不匹配导致的波动）。
        返回误差百分比（0.0 ~ 1.0）。
        """
        if self.mode == "trapezoidal":
            amp, freq, overlap = params
            # 假设重叠率越偏离 10% 误差越大，频率过高或过低误差大
            overlap_penalty = abs(overlap - 0.10) * 100.0  # 5%偏离带来 5% 误差
            freq_penalty = abs(freq - 2000.0) / 20000.0
            return overlap_penalty + freq_penalty
        else:
            amp, freq, slope = params
            # 锯齿波斜率过低会导致动摩擦段过长，引起步距不稳定
            # 由于斜率在 4.0 到 10.0，1.0/4.0 = 0.25 已经大于 5% 的约束。
            # 这里应该是一个可被最小化至低于 0.05 的函数。
            # 修改斜率的理想值为 10.0
            slope_penalty = abs(10.0 - slope) * 0.005
            freq_penalty = abs(freq - 3000.0) / 30000.0
            return slope_penalty + freq_penalty

    def _energy_consumption(self, params: tuple) -> float:
        """
        估算能耗 (单位时间内电容充放电消耗的能量)。
        E ∝ C * V^2 * f
        """
        amp, freq, _ = params
        C = 1e-6  # 假设等效电容 1uF
        return C * (amp ** 2) * freq

    def fitness_function(self, params: tuple) -> float:
        """
        适应度函数 (越小越好)。

        目标：
        1. 最小化能耗
        2. 最大化步距 (通过取倒数或负值)
        3. 约束：稳定性误差 < 5% (0.05)。如果不满足则施加极大惩罚。
        """
        stability_err = self._simulate_stability_error(params)
        energy = self._energy_consumption(params)
        step_dist = self._simulate_step_distance(params)

        # 为了避免除零，给步距加一个极小值
        step_dist = max(step_dist, 1e-6)

        # 目标值：单位位移能耗
        cost = energy / step_dist

        # 稳定性约束惩罚
        if stability_err > 0.05:
            penalty = 1e6 * (stability_err - 0.05)
        else:
            penalty = 0.0

        return cost + penalty

    def optimize(self, target_frequency_bounds: Tuple[float, float] = (100.0, 5000.0)) -> Dict[str, Any]:
        """
        执行差分进化算法以寻找最佳波形参数。

        参数:
            target_frequency_bounds: 频率搜索区间

        返回:
            Dict: 包含优化后的参数和性能指标的字典。
        """
        # 定义边界
        if self.mode == "trapezoidal":
            # (amplitude, frequency, overlap_ratio)
            # amplitude: 10~100V
            # overlap_ratio: 0.05~0.15
            bounds = [(10.0, TrapezoidalWaveGenerator.MAX_VOLTAGE),
                      target_frequency_bounds,
                      (TrapezoidalWaveGenerator.MIN_OVERLAP, TrapezoidalWaveGenerator.MAX_OVERLAP)]
        else:
            # (amplitude, frequency, slope_ratio)
            bounds = [(10.0, SawtoothWaveGenerator.MAX_VOLTAGE),
                      target_frequency_bounds,
                      (SawtoothWaveGenerator.MIN_SLOPE_RATIO, SawtoothWaveGenerator.MAX_SLOPE_RATIO)]

        result = differential_evolution(self.fitness_function, bounds, strategy='best1bin',
                                        popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7,
                                        seed=42)

        best_params = result.x
        stability_err = self._simulate_stability_error(best_params)

        ret = {
            "success": result.success,
            "message": result.message,
            "amplitude": best_params[0],
            "frequency": best_params[1],
            "stability_error": stability_err
        }

        if self.mode == "trapezoidal":
            ret["overlap_ratio"] = best_params[2]
        else:
            ret["slope_ratio"] = best_params[2]

        return ret
