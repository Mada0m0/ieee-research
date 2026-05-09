import numpy as np
import pytest
from voltage_waveform.src.generators import TrapezoidalWaveGenerator, SawtoothWaveGenerator

def test_trapezoidal_wave_generator_initialization():
    """测试梯形波生成器初始化和参数异常处理"""
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
    """测试四相梯形波生成的形状和基本范围"""
    gen = TrapezoidalWaveGenerator(amplitude=1.0, frequency=1.0, offset=0.0)
    t = np.linspace(0, 2, 1000)
    waves = gen.generate(t)

    assert waves.shape == (4, 1000)
    assert np.all((waves >= -1.0) & (waves <= 1.0))

def test_trapezoidal_wave_generator_phases():
    """测试四相梯形波相位关系（第一相和平移后的第二相应该匹配）"""
    gen = TrapezoidalWaveGenerator(amplitude=1.0, frequency=1.0, rise_time_ratio=0.2)
    # 取一个整周期
    t = np.linspace(0, 1, 1000, endpoint=False)
    waves = gen.generate(t)

    # 第0相 (0 shift) 和 第1相 (0.25 shift)
    # 对于频率 1.0, 相位差 0.25 等同于时间推移 0.25 秒
    shift_idx = int(0.25 * 1000)

    # 把 phase 0 的信号向左循环移位 0.25 个周期，应该与 phase 1 匹配
    phase_0_shifted = np.roll(waves[0], -shift_idx)

    # 允许一定的数值误差
    np.testing.assert_allclose(phase_0_shifted, waves[1], atol=1e-2)


def test_sawtooth_wave_generator_initialization():
    """测试锯齿波生成器初始化和参数异常处理"""
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
    """测试锯齿波生成器在不同 width 下的输出特征"""
    t = np.linspace(0, 1, 1000, endpoint=False)

    # 标准正向锯齿波 (width=1.0)
    gen_forward = SawtoothWaveGenerator(amplitude=1.0, frequency=1.0, width=1.0)
    wave_forward = gen_forward.generate(t)
    assert np.min(wave_forward) >= -1.0
    assert np.max(wave_forward) <= 1.0
    # 开始时应该接近 -1，结束时应该接近 1
    assert np.isclose(wave_forward[0], -1.0, atol=1e-2)
    assert np.isclose(wave_forward[-1], 1.0, atol=1e-2)

    # 标准反向锯齿波 (width=0.0)
    gen_backward = SawtoothWaveGenerator(amplitude=1.0, frequency=1.0, width=0.0)
    wave_backward = gen_backward.generate(t)
    assert np.isclose(wave_backward[0], 1.0, atol=1e-2)
    assert np.isclose(wave_backward[-1], -1.0, atol=1e-2)

    # 三角波 (width=0.5)
    gen_triangle = SawtoothWaveGenerator(amplitude=1.0, frequency=1.0, width=0.5)
    wave_triangle = gen_triangle.generate(t)
    # 中点（0.5秒）应该是最大值 1.0
    mid_idx = 500
    assert np.isclose(wave_triangle[mid_idx], 1.0, atol=1e-2)
    assert np.isclose(wave_triangle[0], -1.0, atol=1e-2)
