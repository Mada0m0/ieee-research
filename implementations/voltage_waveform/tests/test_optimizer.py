import numpy as np
from implementations.voltage_waveform.optimizer import WaveOptimizer

def dummy_fitness(params: np.ndarray) -> float:
    """
    一个简单的适应度函数：目标是让三个参数分别接近 5.0, 10.0, 15.0
    返回负的平方误差和。
    """
    target = np.array([5.0, 10.0, 15.0])
    return -np.sum((params - target)**2)

def test_wave_optimizer_initialization():
    """Test optimizer initialization"""
    bounds = [(0.0, 10.0), (0.0, 20.0), (0.0, 30.0)]
    opt = WaveOptimizer(fitness_function=dummy_fitness, param_bounds=bounds, population_size=20, generations=10)

    assert opt.num_params == 3
    assert opt.population_size == 20
    assert opt.generations == 10
    assert opt.num_elites >= 1

def test_wave_optimizer_run():
    """Test a complete run of the genetic algorithm"""
    bounds = [(0.0, 10.0), (0.0, 20.0), (0.0, 30.0)]
    opt = WaveOptimizer(
        fitness_function=dummy_fitness,
        param_bounds=bounds,
        population_size=100,
        generations=50,
        mutation_rate=0.1,
        elite_ratio=0.1
    )

    # To test reproducibility (optional)
    opt.rng = np.random.default_rng(42)

    result = opt.optimize()

    assert "best_param" in result
    assert "best_fitness" in result
    assert "history" in result

    best_param = result["best_param"]
    assert len(best_param) == 3

    # Verify that convergence is possible (does not need to be exactly equal, but should be close to the target)
    target = np.array([5.0, 10.0, 15.0])
    # The expected error is within the acceptable range
    assert np.allclose(best_param, target, atol=1.0)

    # Verify that historical fitness is non-decreasing (because of the elite retention strategy)
    history = result["history"]
    assert len(history) == 50
    for i in range(1, len(history)):
        assert history[i] >= history[i-1]
