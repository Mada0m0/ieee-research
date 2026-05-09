import numpy as np
from typing import Optional, Tuple, Dict, List

from implementations.hysteresis.rmso_optimizer import RMSOOptimizer


class RMSO_BW_Model:
    """
    Piezoelectric actuator hysteresis modeling based on the Bouc-Wen model supports parameter identification using the RMSO algorithm.

    The classic Bouc-Wen model describes the hysteresis characteristics of the system through nonlinear differential equations and is often used to characterize the rate-dependent hysteresis phenomenon of piezoelectric actuators.
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize Bouc-Wen model parameters.

        Args:
            params (Optional[Dict[str, float]]): Dictionary of model parameters, including:
                - A: Control hysteresis amplitude
                - alpha: controls hysteresis shape
                - beta: controls hysteresis shape
                - gamma: controls hysteresis shape
                - n: controls hysteresis smoothness (usually 1 or 2)
        """
        if params is None:
            #Default parameters
            self.params = {
                'A': 1.0,
                'alpha': 0.1,
                'beta': 0.1,
                'gamma': 0.1,
                'n': 1.0
            }
        else:
            self.params = params

    def simulate(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Simulates a Bouc-Wen hysteresis output given a given input displacement/voltage sequence and time step.

        Model equation (simplified hysteresis term calculation for displacement input):
        \\dot{h} = A \\cdot \\dot{u} - \\beta \\cdot |\\dot{u}| \\cdot h \\cdot |h|^{n-1} - \\gamma \\cdot \\dot{u} \\cdot |h|^n
        The final output is often the linear term plus the hysteresis term y = \\alpha \\cdot u + h, this implementation only computes the output hysteresis term and the full output.

        Args:
            u (np.ndarray): input sequence (1D array).
            dt (float): time step.

        Returns:
            np.ndarray: model output sequence (1D array), the same length as the input.
        """
        N = len(u)
        h = np.zeros(N)
        y = np.zeros(N)

        A = self.params['A']
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']
        n = self.params['n']

        # Calculate the derivative of the input (velocity)
        # Use simple difference, assuming the velocity before u[0] is 0
        u_dot = np.zeros(N)
        if N > 1:
            u_dot[1:] = (u[1:] - u[:-1]) / dt
            u_dot[0] = u_dot[1] # Simple boundary processing

        # Euler method to solve differential equations
        for i in range(1, N):
            h_prev = h[i-1]
            u_dot_curr = u_dot[i]

            # Bouc-Wen differential equations
            term1 = A * u_dot_curr
            term2 = beta * abs(u_dot_curr) * h_prev * (abs(h_prev) ** (n - 1))
            term3 = gamma * u_dot_curr * (abs(h_prev) ** n)

            h_dot = term1 - term2 - term3
            h[i] = h_prev + h_dot * dt

            # Calculate the total output y = alpha * u + h
            y[i] = alpha * u[i] + h[i]

        return y

    def identify_with_rmso(self, u_meas: np.ndarray, y_meas: np.ndarray, dt: float, bounds: Dict[str, Tuple[float, float]], pop_size: int = 50, max_iter: int = 200) -> Dict[str, float]:
        """
        Identification of Bouc-Wen model parameters using RMSO algorithm.

        Args:
            u_meas (np.ndarray): Input sequence of measurements.
            y_meas (np.ndarray): Output sequence of measurements.
            dt (float): time step.
            bounds (Dict[str, Tuple[float, float]]): parameter bounds, including A, alpha, beta, gamma, n.
            pop_size (int): Total number of particles.
            max_iter (int): Maximum number of iterations.

        Returns:
            Dict[str, float]: Dictionary of identified parameters.
        """
        param_names = list(bounds.keys())

        def fitness_fn(p: np.ndarray) -> float:
            # Map the array back to the parameter dictionary
            current_params = {name: val for name, val in zip(param_names, p)}
            self.params.update(current_params)

            #Analog output
            y_sim = self.simulate(u_meas, dt)

            # Calculate mean square error (MSE)
            mse = np.mean((y_sim - y_meas) ** 2)
            return mse

        # Ensure that the total number of particles is divisible by the number of regions. The default number of regions is 5.
        n_regions = 5
        if pop_size % n_regions != 0:
            pop_size = (pop_size // n_regions + 1) * n_regions

        optimizer = RMSOOptimizer(n_particles=pop_size, n_regions=n_regions)
        best_p, best_fitness = optimizer.optimize(fitness_fn, bounds, max_iter)

        # Save the optimal parameters to the model and return
        self.params = {name: val for name, val in zip(param_names, best_p)}
        print(f"RMSO identification completed. Optimal MSE: {best_fitness:.6f}")
        return self.params

    def compute_hysteresis_loop(self, u: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the hysteresis loop output for plotting.

        Args:
            u (np.ndarray): input sequence.
            dt (float): time step.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (input sequence, output sequence).
        """
        y = self.simulate(u, dt)
        return u, y

    def rate_dependent_simulate(self, frequencies: List[float], amplitude: float) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate rate-dependent hysteresis loops at multiple frequencies.

        Args:
            frequencies (List[float]): List of frequencies (Hz).
            amplitude (float): the amplitude of the sinusoidal input signal.

        Returns:
            Dict[float, Tuple[np.ndarray, np.ndarray]]: Contains (input, output) hysteresis loop data at each frequency.
        """
        loops = {}
        for freq in frequencies:
            # Set the sampling rate to ensure there are enough points in each cycle
            fs = max(1000, int(100 * freq))
            dt = 1.0 / fs
            t = np.arange(0, 2.0 / freq, dt) # Simulate 2 cycles to reach steady state

            u = amplitude * np.sin(2 * np.pi * freq * t)
            y = self.simulate(u, dt)

            # Extract only the second period data after stabilization for display
            samples_per_cycle = int(fs / freq)
            u_stable = u[samples_per_cycle:]
            y_stable = y[samples_per_cycle:]

            loops[freq] = (u_stable, y_stable)

        return loops
