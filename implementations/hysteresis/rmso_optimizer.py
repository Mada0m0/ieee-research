import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional

class RMSOOptimizer:
    """
    基于区域的混合种群粒子群优化算法 (Region-based Mixed-Species Swarm Optimization)。

    用于在给定的参数空间中寻找最优解，特别适用于复杂的非线性模型参数辨识。
    该算法将种群划分为多个子区域（种群），以平衡全局探索（exploration）和局部开发（exploitation）。
    """

    def __init__(self, n_particles: int = 50, n_regions: int = 5, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """
        初始化 RMSO 优化器。

        Args:
            n_particles (int): 粒子总数。
            n_regions (int): 划分的区域（子种群）数量。
            w (float): 惯性权重，控制历史速度对当前速度的影响。
            c1 (float): 个体学习因子，控制粒子向自身历史最优位置的趋近程度。
            c2 (float): 社会（区域/全局）学习因子，控制粒子向区域最优位置的趋近程度。

        Raises:
            ValueError: 当粒子总数无法被区域数量整除时抛出。
        """
        if n_particles % n_regions != 0:
            raise ValueError("粒子总数 (n_particles) 必须能被区域数量 (n_regions) 整除。")

        self.n_particles = n_particles
        self.n_regions = n_regions
        self.particles_per_region = n_particles // n_regions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.convergence_history = []
        self.best_position = None
        self.best_fitness = float('inf')

    def optimize(self, fitness_fn: Callable[[np.ndarray], float], bounds: dict, max_iter: int = 200) -> Tuple[np.ndarray, float]:
        """
        运行 RMSO 优化算法。

        Args:
            fitness_fn (Callable[[np.ndarray], float]): 适应度函数，输入参数数组，返回标量适应度（越小越好）。
            bounds (dict): 参数的边界字典。格式应为 `{"param_name": (min_val, max_val), ...}`。
                           由于粒子状态只通过数组索引访问，此处的 bounds 仅用于确定维度和对应维度的上下界。
            max_iter (int): 最大迭代次数。

        Returns:
            Tuple[np.ndarray, float]: 最优参数数组及其对应的适应度值。
        """
        n_dims = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds.values()])
        upper_bounds = np.array([b[1] for b in bounds.values()])

        # 初始化位置和速度
        positions = np.random.uniform(lower_bounds, upper_bounds, (self.n_particles, n_dims))
        velocities = np.zeros((self.n_particles, n_dims))

        # 初始化个体最优和区域最优
        personal_best_positions = positions.copy()
        personal_best_fitness = np.array([fitness_fn(p) for p in positions])

        region_best_positions = np.zeros((self.n_regions, n_dims))
        region_best_fitness = np.full(self.n_regions, float('inf'))

        # 初始化全局最优
        self.best_fitness = np.min(personal_best_fitness)
        self.best_position = personal_best_positions[np.argmin(personal_best_fitness)].copy()
        self.convergence_history = [self.best_fitness]

        for region in range(self.n_regions):
            start_idx = region * self.particles_per_region
            end_idx = start_idx + self.particles_per_region
            region_fitness = personal_best_fitness[start_idx:end_idx]
            best_in_region_idx = np.argmin(region_fitness)
            region_best_fitness[region] = region_fitness[best_in_region_idx]
            region_best_positions[region] = personal_best_positions[start_idx + best_in_region_idx].copy()

        # 开始迭代优化
        for i in range(1, max_iter):
            # 动态调整惯性权重 (可选项，这里使用线性递减策略来提高后期局部搜索能力)
            current_w = self.w - (self.w - 0.4) * (i / max_iter)

            for region in range(self.n_regions):
                start_idx = region * self.particles_per_region
                end_idx = start_idx + self.particles_per_region

                # 更新速度和位置
                r1 = np.random.rand(self.particles_per_region, n_dims)
                r2 = np.random.rand(self.particles_per_region, n_dims)

                velocities[start_idx:end_idx] = (
                    current_w * velocities[start_idx:end_idx] +
                    self.c1 * r1 * (personal_best_positions[start_idx:end_idx] - positions[start_idx:end_idx]) +
                    self.c2 * r2 * (region_best_positions[region] - positions[start_idx:end_idx])
                )

                positions[start_idx:end_idx] += velocities[start_idx:end_idx]

                # 边界处理 (限制在边界内)
                positions[start_idx:end_idx] = np.clip(positions[start_idx:end_idx], lower_bounds, upper_bounds)

                # 更新适应度和个体最优
                for j in range(start_idx, end_idx):
                    fit = fitness_fn(positions[j])
                    if fit < personal_best_fitness[j]:
                        personal_best_fitness[j] = fit
                        personal_best_positions[j] = positions[j].copy()

                        # 更新区域最优
                        if fit < region_best_fitness[region]:
                            region_best_fitness[region] = fit
                            region_best_positions[region] = positions[j].copy()

                            # 更新全局最优
                            if fit < self.best_fitness:
                                self.best_fitness = fit
                                self.best_position = positions[j].copy()

            self.convergence_history.append(self.best_fitness)

        return self.best_position, self.best_fitness

    def get_convergence_history(self) -> np.ndarray:
        """
        获取算法优化的收敛历史。

        Returns:
            np.ndarray: 包含每次迭代最佳适应度的数组。
        """
        return np.array(self.convergence_history)

    def plot_convergence(self, save_path: Optional[str] = None):
        """
        绘制优化算法的收敛曲线。

        Args:
            save_path (Optional[str]): 图像保存路径。如果为 None，则直接显示图像。
        """
        if not self.convergence_history:
            print("尚未进行优化，没有收敛历史可绘制。")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history, 'b-', linewidth=2)
        plt.title('RMSO Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (Log Scale)')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"收敛图已保存至: {save_path}")
        else:
            plt.show()

        plt.close()
