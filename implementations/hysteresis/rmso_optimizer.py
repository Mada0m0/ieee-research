import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional

class RMSOOptimizer:
    """
    Region-based Mixed-Species Swarm Optimization.

    It is used to find the optimal solution in a given parameter space, especially suitable for complex nonlinear model parameter identification.
    This algorithm divides the population into multiple sub-regions (populations) to balance global exploration and local exploitation.
    """

    def __init__(self, n_particles: int = 50, n_regions: int = 5, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """
        Initialize the RMSO optimizer.

        Args:
            n_particles (int): Total number of particles.
            n_regions (int): Number of divided regions (sub-populations).
            w (float): Inertia weight, controlling the influence of historical speed on current speed.
            c1 (float): Individual learning factor, controlling the degree of particle approach to its historical optimal position.
            c2 (float): Social (regional/global) learning factor, which controls the approach of particles to the optimal position in the region.

        Raises:
            ValueError: Thrown when the total number of particles is not divisible by the number of regions.
        """
        if n_particles % n_regions != 0:
            raise ValueError("The total number of particles (n_particles) must be divisible by the number of regions (n_regions).")

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
        Run the RMSO optimization algorithm.

        Args:
            fitness_fn (Callable[[np.ndarray], float]): Fitness function, input parameter array, return scalar fitness (the smaller the better).
            bounds (dict): Dictionary of bounds for parameters. The format should be `{"param_name": (min_val, max_val), ...}`.
                           Since the particle state is only accessed through the array index, the bounds here are only used to determine the dimensions and the upper and lower bounds of the corresponding dimensions.
            max_iter (int): Maximum number of iterations.

        Returns:
            Tuple[np.ndarray, float]: optimal parameter array and its corresponding fitness value.
        """
        n_dims = len(bounds)
        lower_bounds = np.array([b[0] for b in bounds.values()])
        upper_bounds = np.array([b[1] for b in bounds.values()])

        #Initialize position and velocity
        positions = np.random.uniform(lower_bounds, upper_bounds, (self.n_particles, n_dims))
        velocities = np.zeros((self.n_particles, n_dims))

        # Euler method to solve differential equations
        personal_best_positions = positions.copy()
        personal_best_fitness = np.array([fitness_fn(p) for p in positions])

        region_best_positions = np.zeros((self.n_regions, n_dims))
        region_best_fitness = np.full(self.n_regions, float('inf'))

        # Initialize global optimal
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

        # Start iterative optimization
        for i in range(1, max_iter):
            # Dynamically adjust inertia weight (optional, a linear decrease strategy is used here to improve later local search capabilities)
            current_w = self.w - (self.w - 0.4) * (i / max_iter)

            for region in range(self.n_regions):
                start_idx = region * self.particles_per_region
                end_idx = start_idx + self.particles_per_region

                # Update speed and position
                r1 = np.random.rand(self.particles_per_region, n_dims)
                r2 = np.random.rand(self.particles_per_region, n_dims)

                velocities[start_idx:end_idx] = (
                    current_w * velocities[start_idx:end_idx] +
                    self.c1 * r1 * (personal_best_positions[start_idx:end_idx] - positions[start_idx:end_idx]) +
                    self.c2 * r2 * (region_best_positions[region] - positions[start_idx:end_idx])
                )

                positions[start_idx:end_idx] += velocities[start_idx:end_idx]

                # Boundary handling (limited to boundaries)
                positions[start_idx:end_idx] = np.clip(positions[start_idx:end_idx], lower_bounds, upper_bounds)

                # Update fitness and individual optimality
                for j in range(start_idx, end_idx):
                    fit = fitness_fn(positions[j])
                    if fit < personal_best_fitness[j]:
                        personal_best_fitness[j] = fit
                        personal_best_positions[j] = positions[j].copy()

                        # Update regional optimal
                        if fit < region_best_fitness[region]:
                            region_best_fitness[region] = fit
                            region_best_positions[region] = positions[j].copy()

                            # Update global optimal
                            if fit < self.best_fitness:
                                self.best_fitness = fit
                                self.best_position = positions[j].copy()

            self.convergence_history.append(self.best_fitness)

        return self.best_position, self.best_fitness

    def get_convergence_history(self) -> np.ndarray:
        """
        Get the convergence history of algorithm optimization.

        Returns:
            np.ndarray: Array containing the best fitness for each iteration.
        """
        return np.array(self.convergence_history)

    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Draw the convergence curve of the optimization algorithm.

        Args:
            save_path (Optional[str]): Image saving path. If None, the image is displayed directly.
        """
        if not self.convergence_history:
            print("Optimization has not yet been performed and there is no convergence history to draw.")
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
            print(f"The convergence graph has been saved to: {save_path}")
        else:
            plt.show()

        plt.close()
