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
    """测试优化器初始化"""
    bounds = [(0.0, 10.0), (0.0, 20.0), (0.0, 30.0)]
    opt = WaveOptimizer(fitness_function=dummy_fitness, param_bounds=bounds, population_size=20, generations=10)

    assert opt.num_params == 3
    assert opt.population_size == 20
    assert opt.generations == 10
    assert opt.num_elites >= 1

def test_wave_optimizer_run():
    """测试遗传算法的完整运行"""
    bounds = [(0.0, 10.0), (0.0, 20.0), (0.0, 30.0)]
    opt = WaveOptimizer(
        fitness_function=dummy_fitness,
        param_bounds=bounds,
        population_size=100,
        generations=50,
        mutation_rate=0.1,
        elite_ratio=0.1
    )

    # 为了测试可复现性（可选）
    opt.rng = np.random.default_rng(42)

    result = opt.optimize()

    assert "best_param" in result
    assert "best_fitness" in result
    assert "history" in result

    best_param = result["best_param"]
    assert len(best_param) == 3

    # 验证是否能够收敛（不需要完全等于，但应该靠近目标）
    target = np.array([5.0, 10.0, 15.0])
    # 期望误差在可接受范围内
    assert np.allclose(best_param, target, atol=1.0)

    # 验证历史适应度是非递减的（因为有精英保留策略）
    history = result["history"]
    assert len(history) == 50
    for i in range(1, len(history)):
        assert history[i] >= history[i-1]
