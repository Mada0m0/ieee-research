import numpy as np
from typing import Tuple

class TrapezoidalWaveGenerator:
    """
    Four-Phase Trapezoidal Wave Generator.

    It is used to drive piezoelectric motors and generate trapezoidal waveforms by adjusting rise time and flat top time.
    And output a four-phase waveform with a phase difference of 90 degrees (i.e. 0, pi/2, pi, 3pi/2).
    """

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0, rise_time_ratio: float = 0.2, offset: float = 0.0) -> None:
        """
        Initialize the trapezoidal wave generator.

        Args:
            amplitude (float): The amplitude of the waveform (peak-to-half peak value, or peak value).
            frequency (float): The frequency (Hz) of the waveform.
            rise_time_ratio (float): The ratio of rise time to the entire cycle, range (0, 0.5).
                                     If it is equal to 0.25, it is a triangle wave; if it is close to 0, it is a square wave.
            offset (float): DC offset of the waveform.
        """
        if not 0 < rise_time_ratio < 0.5:
            raise ValueError("Rise time scale must be between (0, 0.5).")
        self.amplitude = amplitude
        self.frequency = frequency
        self.rise_time_ratio = rise_time_ratio
        self.offset = offset

    def _generate_single_phase(self, t: np.ndarray, phase_shift: float) -> np.ndarray:
        """
        Generates a single-phase trapezoidal wave.

        Args:
            t (np.ndarray): time array.
            phase_shift (float): Phase offset (unit is period ratio, 0 to 1).

        Returns:
            np.ndarray: Single-phase trapezoidal wave array corresponding to time.
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

        # Falling edge
        mask3 = (phase >= 0.5) & (phase < 0.5 + self.rise_time_ratio)
        y[mask3] = 1.0 - 2.0 * ((phase[mask3] - 0.5) / self.rise_time_ratio)

        # flat bottom
        mask4 = phase >= 0.5 + self.rise_time_ratio
        y[mask4] = -1.0

        return self.amplitude * y + self.offset

    def generate(self, t: np.ndarray) -> np.ndarray:
        """
        Generate four-phase trapezoidal waves.

        Args:
            t (np.ndarray): time array.

        Returns:
            np.ndarray: Array of shape (4, len(t)), containing four-phase signals.
        """
        phase_shifts = [0.0, 0.25, 0.5, 0.75]
        waves = [self._generate_single_phase(t, shift) for shift in phase_shifts]
        return np.vstack(waves)


class SawtoothWaveGenerator:
    """
    Adjustable Slope Sawtooth Wave Generator.

    Used for piezoelectric motor driving, it can generate forward sawtooth wave, reverse sawtooth wave or triangle wave by adjusting the asymmetry (width).
    """

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0, width: float = 1.0, offset: float = 0.0) -> None:
        """
        Initialize the sawtooth wave generator.

        Args:
            amplitude (float): The amplitude of the waveform.
            frequency (float): The frequency (Hz) of the waveform.
            width (float): Asymmetry, range [0, 1].
                           1.0 is a standard forward sawtooth wave, 0.0 is a standard reverse sawtooth wave, and 0.5 is a triangle wave.
            offset (float): DC offset of the waveform.
        """
        if not 0.0 <= width <= 1.0:
            raise ValueError("Asymmetry (width) must be between [0, 1].")
        self.amplitude = amplitude
        self.frequency = frequency
        self.width = width
        self.offset = offset

    def generate(self, t: np.ndarray) -> np.ndarray:
        """
        Generate a sawtooth wave.

        Args:
            t (np.ndarray): time array.

        Returns:
            np.ndarray: array of sawtooth waves of shape (len(t),).
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
