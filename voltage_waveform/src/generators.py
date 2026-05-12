import numpy as np
from typing import Tuple

class TrapezoidalWaveGenerator:
    """
    四相步态梯形波生成器 (Four-Phase Trapezoidal Wave Generator)。

    用于压电电机驱动，通过调节上升时间、平顶时间来生成梯形波形，
    并输出相位相差90度（即0, pi/2, pi, 3pi/2）的四相波形。
    """

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0, rise_time_ratio: float = 0.2, offset: float = 0.0) -> None:
        """
        初始化梯形波生成器。

        Args:
            amplitude (float): 波形的幅值 (峰-峰值的一半，或者峰值)。
            frequency (float): 波形的频率 (Hz)。
            rise_time_ratio (float): 上升时间占整个周期的比例，范围 (0, 0.5)。
                                     如果等于0.25，则是三角波；如果趋近于0，则是方波。
            offset (float): 波形的直流偏置。
        """
        if not 0 < rise_time_ratio < 0.5:
            raise ValueError("The rise time scale must be between (0, 0.5).")
        self.amplitude = amplitude
        self.frequency = frequency
        self.rise_time_ratio = rise_time_ratio
        self.offset = offset

    def _generate_single_phase(self, t: np.ndarray, phase_shift: float) -> np.ndarray:
        """
        生成单相梯形波。

        Args:
            t (np.ndarray): 时间数组。
            phase_shift (float): 相位偏移（单位为周期比例，0到1）。

        Returns:
            np.ndarray: 对应时间的单相梯形波数组。
        """
        # Calculate the current phase and map it to the interval [0, 1)
        phase = (t * self.frequency + phase_shift) % 1.0

        # Trapezoidal wave calculation
        # Define four intervals:
        # 1. [0, rise_time_ratio): rising edge (-1 to 1)
        # 2. [rise_time_ratio, 0.5): flat top (1)
        # 3. [0.5, 0.5 + rise_time_ratio): falling edge (1 to -1)
        # 4. [0.5 + rise_time_ratio, 1): flat bottom (-1)

        y = np.zeros_like(phase)

        # rising edge
        mask1 = phase < self.rise_time_ratio
        y[mask1] = -1.0 + 2.0 * (phase[mask1] / self.rise_time_ratio)

        # flat top
        mask2 = (phase >= self.rise_time_ratio) & (phase < 0.5)
        y[mask2] = 1.0

        # falling edge
        mask3 = (phase >= 0.5) & (phase < 0.5 + self.rise_time_ratio)
        y[mask3] = 1.0 - 2.0 * ((phase[mask3] - 0.5) / self.rise_time_ratio)

        # flat
        mask4 = phase >= 0.5 + self.rise_time_ratio
        y[mask4] = -1.0

        return self.amplitude * y + self.offset

    def generate(self, t: np.ndarray) -> np.ndarray:
        """
        生成四相梯形波。

        Args:
            t (np.ndarray): 时间数组。

        Returns:
            np.ndarray: 形状为 (4, len(t)) 的数组，包含四相信号。
        """
        phase_shifts = [0.0, 0.25, 0.5, 0.75]
        waves = [self._generate_single_phase(t, shift) for shift in phase_shifts]
        return np.vstack(waves)


class SawtoothWaveGenerator:
    """
    可调斜率锯齿波生成器 (Adjustable Slope Sawtooth Wave Generator)。

    用于压电电机驱动，可通过调节不对称度（width）生成正向锯齿波、反向锯齿波或三角波。
    """

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0, width: float = 1.0, offset: float = 0.0) -> None:
        """
        初始化锯齿波生成器。

        Args:
            amplitude (float): 波形的幅值。
            frequency (float): 波形的频率 (Hz)。
            width (float): 不对称度，范围 [0, 1]。
                           1.0 为标准正向锯齿波，0.0 为标准反向锯齿波，0.5 为三角波。
            offset (float): 波形的直流偏置。
        """
        if not 0.0 <= width <= 1.0:
            raise ValueError("The asymmetry (width) must be between [0, 1].")
        self.amplitude = amplitude
        self.frequency = frequency
        self.width = width
        self.offset = offset

    def generate(self, t: np.ndarray) -> np.ndarray:
        """
        生成锯齿波。

        Args:
            t (np.ndarray): 时间数组。

        Returns:
            np.ndarray: 形状为 (len(t),) 的锯齿波数组。
        """
        phase = (t * self.frequency) % 1.0

        y = np.zeros_like(phase)

        if self.width == 1.0:
            y = -1.0 + 2.0 * phase
        elif self.width == 0.0:
            y = 1.0 - 2.0 * phase
        else:
            mask1 = phase < self.width
            y[mask1] = -1.0 + 2.0 * (phase[mask1] / self.width)

            mask2 = phase >= self.width
            y[mask2] = 1.0 - 2.0 * ((phase[mask2] - self.width) / (1.0 - self.width))

        return self.amplitude * y + self.offset
