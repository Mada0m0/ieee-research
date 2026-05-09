import numpy as np
from typing import Tuple, Dict, Any, List

class WaveGeneratorError(Exception):
    """波形发生器基础异常类"""
    pass

class TrapezoidalWaveGenerator:
    """
    四相步态梯形波发生器 (Trapezoidal Wave Generator)

    用于夹持-驱动(Clamp-Drive)交替型压电电机腿。
    支持相序、占空比、重叠率(Overlap ratio)的配置。
    保证电压幅值 ≤ 100V，相位重叠率控制在 5-15% 以防止回退运动。
    """

    MAX_VOLTAGE = 100.0
    MIN_OVERLAP = 0.05
    MAX_OVERLAP = 0.15

    def __init__(self, amplitude: float = 80.0, frequency: float = 1000.0, overlap_ratio: float = 0.10):
        """
        初始化梯形波发生器。

        参数:
            amplitude (float): 电压幅值 (V)，必须在 (0, 100] 之间。
            frequency (float): 驱动频率 (Hz)，必须 > 0。
            overlap_ratio (float): 相位重叠率，必须在 [0.05, 0.15] 之间。
        """
        self.set_parameters(amplitude, frequency, overlap_ratio)

    def set_parameters(self, amplitude: float, frequency: float, overlap_ratio: float) -> None:
        """
        设置发生器参数。

        参数:
            amplitude (float): 电压幅值 (V)。
            frequency (float): 驱动频率 (Hz)。
            overlap_ratio (float): 相位重叠率。

        异常:
            ValueError: 当参数不符合物理或系统硬性约束时抛出。
        """
        if not (0 < amplitude <= self.MAX_VOLTAGE):
            raise ValueError(f"电压幅值必须在 (0, {self.MAX_VOLTAGE}] 范围内，当前值为: {amplitude}V")
        if frequency <= 0:
            raise ValueError("频率必须为正数")
        if not (self.MIN_OVERLAP <= overlap_ratio <= self.MAX_OVERLAP):
            raise ValueError(f"重叠率必须在 [{self.MIN_OVERLAP}, {self.MAX_OVERLAP}] 之间，当前值为: {overlap_ratio}")

        self.amplitude = amplitude
        self.frequency = frequency
        self.overlap_ratio = overlap_ratio
        self.period = 1.0 / frequency

    def generate(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成四相梯形波信号。

        时序规划:
        四相平分一个周期(Period)，每相基础时间为 T/4。
        重叠时间(t_overlap) = T * overlap_ratio。

        相位1 (Clamp-L): [0, T/4 + t_overlap]
        相位2 (Drive L->R): [T/4, T/2 + t_overlap]
        相位3 (Clamp-R): [T/2, 3T/4 + t_overlap]
        相位4 (Drive R->L): [3T/4, T + t_overlap]
        (由于周期性，相位时间会对周期取模)

        为了简化平滑过渡，上升沿和下降沿采用线性插值(梯形)。
        我们假设上升沿和下降沿时间均为 t_overlap / 2。

        参数:
            t (np.ndarray): 时间向量。

        返回:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 包含四相电压波形的元组。
        """
        t_mod = t % self.period
        T = self.period

        # 定义上升下降沿时间，取重叠时间的一半或一个极小值避免除零
        t_edge = max(T * self.overlap_ratio / 2.0, 1e-6)

        # 每相的开启时间和关闭时间
        # Phase 1
        p1_start = 0
        p1_end = T / 4 + T * self.overlap_ratio

        # Phase 2
        p2_start = T / 4
        p2_end = T / 2 + T * self.overlap_ratio

        # Phase 3
        p3_start = T / 2
        p3_end = 3 * T / 4 + T * self.overlap_ratio

        # Phase 4
        p4_start = 3 * T / 4
        p4_end = T + T * self.overlap_ratio

        def trapezoid(t_array: np.ndarray, start: float, end: float, edge: float) -> np.ndarray:
            """生成单相的梯形波"""
            y = np.zeros_like(t_array)
            # 处理周期跨界的情况，例如 end > T
            if end > T:
                # 拆分为两段：[start, T] 和 [0, end - T]
                end1 = T
                end2 = end - T
                # 第一段
                y = np.where((t_array >= start) & (t_array < start + edge), (t_array - start) / edge, y)
                y = np.where((t_array >= start + edge) & (t_array < end1), 1.0, y)
                # 第二段 (包裹到开头)
                y = np.where((t_array >= 0) & (t_array < end2 - edge), 1.0, y)
                y = np.where((t_array >= end2 - edge) & (t_array <= end2), 1.0 - (t_array - (end2 - edge)) / edge, y)
            else:
                y = np.where((t_array >= start) & (t_array < start + edge), (t_array - start) / edge, y)
                y = np.where((t_array >= start + edge) & (t_array < end - edge), 1.0, y)
                y = np.where((t_array >= end - edge) & (t_array <= end), 1.0 - (t_array - (end - edge)) / edge, y)
            return y

        v1 = self.amplitude * trapezoid(t_mod, p1_start, p1_end, t_edge)
        v2 = self.amplitude * trapezoid(t_mod, p2_start, p2_end, t_edge)
        v3 = self.amplitude * trapezoid(t_mod, p3_start, p3_end, t_edge)
        v4 = self.amplitude * trapezoid(t_mod, p4_start, p4_end, t_edge)

        return v1, v2, v3, v4

class SawtoothWaveGenerator:
    """
    锯齿波发生器 (Sawtooth Wave Generator)

    用于粘滑(Stick-Slip)驱动的压电电机腿。
    可调慢升/快缩斜率比(Slow expansion / Fast contraction ratio)，支持范围 4:1 ~ 10:1。
    """

    MAX_VOLTAGE = 100.0
    MIN_SLOPE_RATIO = 4.0
    MAX_SLOPE_RATIO = 10.0

    def __init__(self, amplitude: float = 60.0, frequency: float = 2000.0, slope_ratio: float = 8.0):
        """
        初始化锯齿波发生器。

        参数:
            amplitude (float): 电压幅值 (V)，必须在 (0, 100] 之间。
            frequency (float): 驱动频率 (Hz)，必须 > 0。
            slope_ratio (float): 升降斜率比，即 t_rise / t_fall，必须在 [4.0, 10.0] 之间。
        """
        self.set_parameters(amplitude, frequency, slope_ratio)

    def set_parameters(self, amplitude: float, frequency: float, slope_ratio: float) -> None:
        """
        设置发生器参数。

        参数:
            amplitude (float): 电压幅值 (V)。
            frequency (float): 驱动频率 (Hz)。
            slope_ratio (float): 升降时间比 (t_rise / t_fall)。

        异常:
            ValueError: 当参数不满足约束时抛出。
        """
        if not (0 < amplitude <= self.MAX_VOLTAGE):
            raise ValueError(f"电压幅值必须在 (0, {self.MAX_VOLTAGE}] 范围内，当前值为: {amplitude}V")
        if frequency <= 0:
            raise ValueError("频率必须为正数")
        if not (self.MIN_SLOPE_RATIO <= slope_ratio <= self.MAX_SLOPE_RATIO):
            raise ValueError(f"斜率比必须在 [{self.MIN_SLOPE_RATIO}, {self.MAX_SLOPE_RATIO}] 之间，当前值为: {slope_ratio}")

        self.amplitude = amplitude
        self.frequency = frequency
        self.slope_ratio = slope_ratio
        self.period = 1.0 / frequency

    def generate(self, t: np.ndarray) -> np.ndarray:
        """
        生成单相锯齿波信号。

        数学模型:
        设周期为 T, t_rise + t_fall = T
        t_rise / t_fall = slope_ratio => t_rise = T * slope_ratio / (1 + slope_ratio)

        当 t % T < t_rise 时, V = A * (t % T) / t_rise
        当 t % T >= t_rise 时, V = A * (1 - (t % T - t_rise) / t_fall)

        参数:
            t (np.ndarray): 时间向量。

        返回:
            np.ndarray: 电压波形。
        """
        t_mod = t % self.period
        T = self.period

        t_rise = T * self.slope_ratio / (1.0 + self.slope_ratio)
        t_fall = T - t_rise

        # 慢升段
        rise_part = self.amplitude * (t_mod / t_rise)
        # 快缩段
        fall_part = self.amplitude * (1.0 - (t_mod - t_rise) / t_fall)

        return np.where(t_mod < t_rise, rise_part, fall_part)
