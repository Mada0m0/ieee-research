import numpy as np
import pytest
from voltage_waveform.src.generators import TrapezoidalWaveGenerator, SawtoothWaveGenerator

def test_trapezoidal_wave_generator_initialization():
    """Test trapezoidal wave generator initialization and parameter exception handling"""
    gen = TrapezoidalWaveGenerator(amplitude=2.0, frequency=50.0, rise_time_ratio=0.1, offset=1.0)
    assert gen.amplitude == 2.0
    assert gen.frequency == 50.0
    assert gen.rise_time_ratio == 0.1
    assert gen.offset == 1.0

    with pytest.raises(ValueError):
        TrapezoidalWaveGenerator(rise_time_ratio=0.6)
    with pytest.raises(ValueError):
        TrapezoidalWaveGenerator(rise_time_ratio=0.0)

def test_trapezoidal_wave_generator_output_shape():
    """Test the shape and basic range of four-phase trapezoidal wave generation"""
    gen = TrapezoidalWaveGenerator(amplitude=1.0, frequency=1.0, offset=0.0)
    t = np.linspace(0, 2, 1000)
    waves = gen.generate(t)

    assert waves.shape == (4, 1000)
    assert np.all((waves >= -1.0) & (waves <= 1.0))

def test_trapezoidal_wave_generator_phases():
    """Test the four-phase trapezoidal wave phase relationship (the first phase and the second phase after translation should match)"""
    gen = TrapezoidalWaveGenerator(amplitude=1.0, frequency=1.0, rise_time_ratio=0.2)
    # Take a whole period
    t = np.linspace(0, 1, 1000, endpoint=False)
    waves = gen.generate(t)

    # Phase 0 (0 shift) and Phase 1 (0.25 shift)
    # For a frequency of 1.0, a phase difference of 0.25 is equivalent to a time lapse of 0.25 seconds
    shift_idx = int(0.25 * 1000)

    # Circularly shift the phase 0 signal to the left by 0.25 cycles, which should match phase 1
    phase_0_shifted = np.roll(waves[0], -shift_idx)

    # Allow certain numerical errors
    np.testing.assert_allclose(phase_0_shifted, waves[1], atol=1e-2)


def test_sawtooth_wave_generator_initialization():
    """Test sawtooth wave generator initialization and parameter exception handling"""
    gen = SawtoothWaveGenerator(amplitude=3.0, frequency=10.0, width=0.5, offset=-1.0)
    assert gen.amplitude == 3.0
    assert gen.frequency == 10.0
    assert gen.width == 0.5
    assert gen.offset == -1.0

    with pytest.raises(ValueError):
        SawtoothWaveGenerator(width=-0.1)
    with pytest.raises(ValueError):
        SawtoothWaveGenerator(width=1.1)

def test_sawtooth_wave_generator_output():
    """Test the output characteristics of the sawtooth wave generator under different widths"""
    t = np.linspace(0, 1, 1000, endpoint=False)

    # Standard forward sawtooth waveform (width=1.0)
    gen_forward = SawtoothWaveGenerator(amplitude=1.0, frequency=1.0, width=1.0)
    wave_forward = gen_forward.generate(t)
    assert np.min(wave_forward) >= -1.0
    assert np.max(wave_forward) <= 1.0
    # It should start close to -1 and end close to 1
    assert np.isclose(wave_forward[0], -1.0, atol=1e-2)
    assert np.isclose(wave_forward[-1], 1.0, atol=1e-2)

    # Standard inverse sawtooth waveform (width=0.0)
    gen_backward = SawtoothWaveGenerator(amplitude=1.0, frequency=1.0, width=0.0)
    wave_backward = gen_backward.generate(t)
    assert np.isclose(wave_backward[0], 1.0, atol=1e-2)
    assert np.isclose(wave_backward[-1], -1.0, atol=1e-2)

    # Triangular wave (width=0.5)
    gen_triangle = SawtoothWaveGenerator(amplitude=1.0, frequency=1.0, width=0.5)
    wave_triangle = gen_triangle.generate(t)
    # The midpoint (0.5 seconds) should be the maximum value of 1.0
    mid_idx = 500
    assert np.isclose(wave_triangle[mid_idx], 1.0, atol=1e-2)
    assert np.isclose(wave_triangle[0], -1.0, atol=1e-2)
