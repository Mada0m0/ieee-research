import numpy as np
from typing import Callable, Tuple, List, Dict, Any

class WaveOptimizer:
    """
    Genetic Algorithm Waveform Parameter Optimizer.

    Use a genetic algorithm to search in a given parameter space to maximize the return value of the fitness function.
    Completely based on pure NumPy implementation.
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
        Initialize the waveform optimizer.

        Args:
            fitness_function (Callable): Fitness function, accepts a 1D numpy array representing a combination of parameters, and returns a floating point number.
            param_bounds (List[Tuple[float, float]]): Parameter boundaries, each element is a tuple (min, max), indicating the value range of the parameter.
            population_size (int): population size.
            generations (int): iteration generation.
            mutation_rate (float): mutation rate [0, 1].
            crossover_rate (float): crossover rate [0, 1].
            elite_ratio (float): Elite retention ratio [0, 1].
        """
        self.fitness_function = fitness_function
        self.param_bounds = np.array(param_bounds)
        self.num_params = len(param_bounds)

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.num_elites = max(1, int(population_size * elite_ratio))

        # In order to ensure that the results are reproducible, random seeds can be set externally
        self.rng = np.random.default_rng()

    def _initialize_population(self) -> np.ndarray:
        """
        Initialize the population.

        Returns:
            np.ndarray: Initial population array of shape (population_size, num_params).
        """
        lower_bounds = self.param_bounds[:, 0]
        upper_bounds = self.param_bounds[:, 1]

        # Initialize evenly distributed within the boundaries
        population = self.rng.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(self.population_size, self.num_params)
        )
        return population

    def _evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """
        Assess the fitness of a population.

        Args:
            population (np.ndarray): current population.

        Returns:
            np.ndarray: fitness array of shape (population_size,).
        """
        fitness_scores = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness_scores[i] = self.fitness_function(population[i])
        return fitness_scores

    def _selection(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """
        Roulette Wheel Selection Selects the mating pool.

        Args:
            population (np.ndarray): current population.
            fitness_scores (np.ndarray): Corresponding fitness scores.

        Returns:
            np.ndarray: The selected parent population.
        """
        # Convert fitness to positive value and calculate probability
        min_fitness = np.min(fitness_scores)
        if min_fitness < 0:
            adjusted_fitness = fitness_scores - min_fitness + 1e-6
        else:
            adjusted_fitness = fitness_scores + 1e-6

        probabilities = adjusted_fitness / np.sum(adjusted_fitness)

        # Select parent
        selected_indices = self.rng.choice(
            np.arange(self.population_size),
            size=self.population_size,
            p=probabilities,
            replace=True
        )
        return population[selected_indices]

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """
        Multipoint crossover or Simulated Binary Crossover (simplified version).
        A simple hybrid crossover is used here.

        Args:
            parents (np.ndarray): The selected parent population.

        Returns:
            np.ndarray: The offspring population generated after crossover.
        """
        offspring = np.empty_like(parents)

        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            # If it is an odd size, the last one will be copied directly.
            if i + 1 >= self.population_size:
                offspring[i] = parent1
                break

            parent2 = parents[i+1]

            if self.rng.random() < self.crossover_rate:
                # Random weight crossover
                alpha = self.rng.random(size=self.num_params)
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = (1 - alpha) * parent1 + alpha * parent2
                offspring[i] = child1
                offspring[i+1] = child2
            else:
                # No intersection
                offspring[i] = parent1
                offspring[i+1] = parent2

        return offspring

    def _mutation(self, offspring: np.ndarray) -> np.ndarray:
        """
        Gaussian Mutation.

        Args:
            offspring (np.ndarray): offspring population after crossover.

        Returns:
            np.ndarray: The mutated offspring population.
        """
        mutation_mask = self.rng.random(size=offspring.shape) < self.mutation_rate

        #Mutation step size (can be adjusted according to the range)
        ranges = self.param_bounds[:, 1] - self.param_bounds[:, 0]
        mutation_steps = self.rng.normal(loc=0.0, scale=0.1, size=offspring.shape) * ranges

        offspring[mutation_mask] += mutation_steps[mutation_mask]

        # Ensure that the mutation does not exceed the boundary
        lower_bounds = self.param_bounds[:, 0]
        upper_bounds = self.param_bounds[:, 1]
        offspring = np.clip(offspring, lower_bounds, upper_bounds)

        return offspring

    def optimize(self) -> Dict[str, Any]:
        """
        Perform genetic algorithm optimization process.

        Returns:
            Dict[str, Any]: Dictionary containing the best parameters, best fitness and historical best fitness records.
        """
        population = self._initialize_population()

        best_param = None
        best_fitness = -np.inf
        history = []

        for generation in range(self.generations):
            fitness_scores = self._evaluate_fitness(population)

            # Record the best of the current generation
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_param = population[current_best_idx].copy()

            history.append(best_fitness)

            # Find elite reservations
            elite_indices = np.argsort(fitness_scores)[-self.num_elites:]
            elites = population[elite_indices].copy()

            # choose
            parents = self._selection(population, fitness_scores)

            #cross
            offspring = self._crossover(parents)

            # Mutations
            offspring = self._mutation(offspring)

            # Put the offspring back into the population and replace some of them with elites
            population = offspring
            population[:self.num_elites] = elites

        return {
            "best_param": best_param,
            "best_fitness": best_fitness,
            "history": history
        }
