import pytest
import numpy as np

from implementations.voltage_waveform.generators import TrapezoidalWaveGenerator, SawtoothWaveGenerator
from implementations.voltage_waveform.optimizer import GAWaveformOptimizer

def test_trapezoidal_normal():
    """测试梯形波发生器常规用例"""
    gen = TrapezoidalWaveGenerator(amplitude=80.0, frequency=1000.0, overlap_ratio=0.10)
    t = np.linspace(0, 0.002, 1000)
    v1, v2, v3, v4 = gen.generate(t)

    # 验证输出维度
    assert v1.shape == t.shape
    assert v2.shape == t.shape
    assert v3.shape == t.shape
    assert v4.shape == t.shape

    # 验证最大电压不超过振幅
    assert np.max(v1) <= 80.0
    assert np.max(v2) <= 80.0
    assert np.max(v3) <= 80.0
    assert np.max(v4) <= 80.0

def test_sawtooth_normal():
    """测试锯齿波发生器常规用例"""
    gen = SawtoothWaveGenerator(amplitude=60.0, frequency=2000.0, slope_ratio=8.0)
    t = np.linspace(0, 0.001, 1000)
    v = gen.generate(t)

    assert v.shape == t.shape
    assert np.max(v) <= 60.0

def test_trapezoidal_boundaries():
    """测试梯形波边界用例：最大电压，最小/最大重叠率"""
    gen = TrapezoidalWaveGenerator(amplitude=100.0, frequency=5000.0, overlap_ratio=0.05)
    assert gen.amplitude == 100.0

    gen.set_parameters(amplitude=10.0, frequency=100.0, overlap_ratio=0.15)
    assert gen.overlap_ratio == 0.15

def test_sawtooth_boundaries():
    """测试锯齿波边界用例：极端斜率比"""
    gen = SawtoothWaveGenerator(amplitude=100.0, frequency=100.0, slope_ratio=4.0)
    assert gen.slope_ratio == 4.0

    gen.set_parameters(amplitude=10.0, frequency=10000.0, slope_ratio=10.0)
    assert gen.slope_ratio == 10.0

def test_trapezoidal_exceptions():
    """测试梯形波异常输入：负电压，超界重叠率"""
    with pytest.raises(ValueError, match="电压幅值必须"):
        TrapezoidalWaveGenerator(amplitude=-10.0)

    with pytest.raises(ValueError, match="电压幅值必须"):
        TrapezoidalWaveGenerator(amplitude=100.1)

    with pytest.raises(ValueError, match="频率必须"):
        TrapezoidalWaveGenerator(frequency=0)

    with pytest.raises(ValueError, match="重叠率必须"):
        TrapezoidalWaveGenerator(overlap_ratio=0.04)

    with pytest.raises(ValueError, match="重叠率必须"):
        TrapezoidalWaveGenerator(overlap_ratio=0.16)

def test_sawtooth_exceptions():
    """测试锯齿波异常输入"""
    with pytest.raises(ValueError, match="电压幅值必须"):
        SawtoothWaveGenerator(amplitude=150.0)

    with pytest.raises(ValueError, match="斜率比必须"):
        SawtoothWaveGenerator(slope_ratio=3.9)

    with pytest.raises(ValueError, match="斜率比必须"):
        SawtoothWaveGenerator(slope_ratio=10.1)

def test_optimizer_trapezoidal():
    """测试 GA 优化器能否成功运行并找到满足稳定性的参数（梯形波）"""
    opt = GAWaveformOptimizer(mode="trapezoidal")
    res = opt.optimize(target_frequency_bounds=(1000.0, 3000.0))

    assert res["success"] is True
    # 验证输出参数符合约束
    assert 10.0 <= res["amplitude"] <= 100.0
    assert 1000.0 <= res["frequency"] <= 3000.0
    assert 0.05 <= res["overlap_ratio"] <= 0.15
    # 验证稳定性满足硬性约束 (< 5%)
    assert res["stability_error"] < 0.05

def test_optimizer_sawtooth():
    """测试 GA 优化器能否成功运行并找到满足稳定性的参数（锯齿波）"""
    opt = GAWaveformOptimizer(mode="sawtooth")
    res = opt.optimize(target_frequency_bounds=(1000.0, 5000.0))

    assert res["success"] is True
    assert 10.0 <= res["amplitude"] <= 100.0
    assert 1000.0 <= res["frequency"] <= 5000.0
    assert 4.0 <= res["slope_ratio"] <= 10.0
    assert res["stability_error"] < 0.05
