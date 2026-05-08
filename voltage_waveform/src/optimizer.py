import numpy as np
from typing import Callable, Tuple, List, Dict, Any

class WaveOptimizer:
    """
    遗传算法波形参数优化器 (Genetic Algorithm Waveform Parameter Optimizer)。

    使用遗传算法在给定的参数空间中搜索，以最大化适应度函数的返回值。
    完全基于纯 NumPy 实现。
    """

    def __init__(
        self,
        fitness_function: Callable[[np.ndarray], float],
        param_bounds: List[Tuple[float, float]],
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1
    ) -> None:
        """
        初始化波形优化器。

        Args:
            fitness_function (Callable): 适应度函数，接受一个表示参数组合的 1D numpy 数组，返回一个浮点数。
            param_bounds (List[Tuple[float, float]]): 参数边界，每个元素为一个元组 (min, max)，表示该参数的取值范围。
            population_size (int): 种群大小。
            generations (int): 迭代代数。
            mutation_rate (float): 变异率 [0, 1]。
            crossover_rate (float): 交叉率 [0, 1]。
            elite_ratio (float): 精英保留比例 [0, 1]。
        """
        self.fitness_function = fitness_function
        self.param_bounds = np.array(param_bounds)
        self.num_params = len(param_bounds)

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.num_elites = max(1, int(population_size * elite_ratio))

        # 为了保证结果可复现，可以外部设置随机种子
        self.rng = np.random.default_rng()

    def _initialize_population(self) -> np.ndarray:
        """
        初始化种群。

        Returns:
            np.ndarray: 形状为 (population_size, num_params) 的初始种群数组。
        """
        lower_bounds = self.param_bounds[:, 0]
        upper_bounds = self.param_bounds[:, 1]

        # 在边界内均匀分布初始化
        population = self.rng.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(self.population_size, self.num_params)
        )
        return population

    def _evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """
        评估种群的适应度。

        Args:
            population (np.ndarray): 当前种群。

        Returns:
            np.ndarray: 形状为 (population_size,) 的适应度数组。
        """
        fitness_scores = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness_scores[i] = self.fitness_function(population[i])
        return fitness_scores

    def _selection(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """
        轮盘赌选择 (Roulette Wheel Selection) 选取交配池。

        Args:
            population (np.ndarray): 当前种群。
            fitness_scores (np.ndarray): 对应的适应度分数。

        Returns:
            np.ndarray: 被选中的父代种群。
        """
        # 将适应度转换为正值并计算概率
        min_fitness = np.min(fitness_scores)
        if min_fitness < 0:
            adjusted_fitness = fitness_scores - min_fitness + 1e-6
        else:
            adjusted_fitness = fitness_scores + 1e-6

        probabilities = adjusted_fitness / np.sum(adjusted_fitness)

        # 选择父代
        selected_indices = self.rng.choice(
            np.arange(self.population_size),
            size=self.population_size,
            p=probabilities,
            replace=True
        )
        return population[selected_indices]

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """
        多点交叉或模拟二进制交叉 (Simulated Binary Crossover, 简化版)。
        此处使用简单的混合交叉。

        Args:
            parents (np.ndarray): 选中的父代种群。

        Returns:
            np.ndarray: 交叉后生成的子代种群。
        """
        offspring = np.empty_like(parents)

        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            # 如果是奇数大小，最后剩一个直接复制
            if i + 1 >= self.population_size:
                offspring[i] = parent1
                break

            parent2 = parents[i+1]

            if self.rng.random() < self.crossover_rate:
                # 随机权重交叉
                alpha = self.rng.random(size=self.num_params)
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = (1 - alpha) * parent1 + alpha * parent2
                offspring[i] = child1
                offspring[i+1] = child2
            else:
                # 不交叉
                offspring[i] = parent1
                offspring[i+1] = parent2

        return offspring

    def _mutation(self, offspring: np.ndarray) -> np.ndarray:
        """
        高斯变异 (Gaussian Mutation)。

        Args:
            offspring (np.ndarray): 交叉后的子代种群。

        Returns:
            np.ndarray: 变异后的子代种群。
        """
        mutation_mask = self.rng.random(size=offspring.shape) < self.mutation_rate

        # 变异步长（可以根据范围调整）
        ranges = self.param_bounds[:, 1] - self.param_bounds[:, 0]
        mutation_steps = self.rng.normal(loc=0.0, scale=0.1, size=offspring.shape) * ranges

        offspring[mutation_mask] += mutation_steps[mutation_mask]

        # 确保变异后不超出边界
        lower_bounds = self.param_bounds[:, 0]
        upper_bounds = self.param_bounds[:, 1]
        offspring = np.clip(offspring, lower_bounds, upper_bounds)

        return offspring

    def optimize(self) -> Dict[str, Any]:
        """
        执行遗传算法优化过程。

        Returns:
            Dict[str, Any]: 包含最佳参数、最佳适应度和历史最佳适应度记录的字典。
        """
        population = self._initialize_population()

        best_param = None
        best_fitness = -np.inf
        history = []

        for generation in range(self.generations):
            fitness_scores = self._evaluate_fitness(population)

            # 记录当前代最佳
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_param = population[current_best_idx].copy()

            history.append(best_fitness)

            # 找出精英保留
            elite_indices = np.argsort(fitness_scores)[-self.num_elites:]
            elites = population[elite_indices].copy()

            # 选择
            parents = self._selection(population, fitness_scores)

            # 交叉
            offspring = self._crossover(parents)

            # 变异
            offspring = self._mutation(offspring)

            # 将子代放回种群，并用精英替换部分子代
            population = offspring
            population[:self.num_elites] = elites

        return {
            "best_param": best_param,
            "best_fitness": best_fitness,
            "history": history
        }
