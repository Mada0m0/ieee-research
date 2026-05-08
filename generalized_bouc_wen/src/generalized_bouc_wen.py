import numpy as np
from scipy import signal
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt

class GeneralizedBoucWen:
    """
    Generalized Bouc-Wen hysteresis model.
    Includes classical parameters (A, alpha, beta, gamma, n),
    an asymmetry parameter (delta), and an IIR filter applied
    to the resulting restoring force.
    """

    def __init__(self, A: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
                 gamma: float = 1.0, n: float = 1.0, delta: float = 0.0,
                 b: List[float] = None, a: List[float] = None):
        """
        Initialize the Generalized Bouc-Wen model.

        Args:
            A: Classical BW parameter A
            alpha: Classical BW parameter alpha (ratio of post-yield to pre-yield stiffness)
            beta: Classical BW parameter beta
            gamma: Classical BW parameter gamma
            n: Classical BW parameter n (smoothness)
            delta: Asymmetry parameter
            b: Numerator coefficients of the IIR filter (default: [1.0])
            a: Denominator coefficients of the IIR filter (default: [1.0])
        """
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.delta = delta

        # Default IIR filter is identity
        self.b = b if b is not None else [1.0]
        self.a = a if a is not None else [1.0]

    def get_params(self) -> np.ndarray:
        return np.array([self.A, self.alpha, self.beta, self.gamma, self.n, self.delta])

    def set_params(self, params: np.ndarray):
        self.A = params[0]
        self.alpha = params[1]
        self.beta = params[2]
        self.gamma = params[3]
        self.n = params[4]
        self.delta = params[5]

    def simulate(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Simulate the generalized Bouc-Wen model.

        Args:
            t: Time array
            x: Displacement input array

        Returns:
            np.ndarray: Filtered restoring force array
        """
        if len(t) != len(x):
            raise ValueError("Time array and input array must have the same length.")

        N = len(t)
        if N < 2:
            return np.zeros_like(x)

        z = np.zeros(N)
        dx = np.zeros(N)

        # Calculate derivative of input x (velocity) using finite differences
        dx[0] = (x[1] - x[0]) / (t[1] - t[0])
        dx[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
        if N > 2:
            dx[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])

        # Integrate Bouc-Wen differential equation for z
        # dz/dt = A * dx/dt - beta * |dx/dt| * |z|^(n-1) * z - gamma * dx/dt * |z|^n
        # Asymmetry modification: A is modified by (1 + delta * sign(x))
        for i in range(1, N):
            dt = t[i] - t[i-1]

            # Classical Bouc-Wen derivative
            # Modified with asymmetry on A or based on direction
            # We apply asymmetry as a factor on A based on sign of x
            asym_factor = 1.0 + self.delta * np.sign(x[i-1])

            # Handle potential division by zero or invalid value when z is 0 and n < 1
            z_term_1 = 0.0
            if np.abs(z[i-1]) > 0 or self.n >= 1:
                z_term_1 = (np.abs(z[i-1]) ** (self.n - 1)) * z[i-1]

            z_term_2 = 0.0
            if np.abs(z[i-1]) > 0 or self.n >= 0:
                z_term_2 = np.abs(z[i-1]) ** self.n

            dz_dt = (asym_factor * self.A * dx[i-1] -
                     self.beta * np.abs(dx[i-1]) * z_term_1 -
                     self.gamma * dx[i-1] * z_term_2)

            # Euler integration
            z[i] = z[i-1] + dz_dt * dt

        # Restoring force: F = alpha * k * x + (1 - alpha) * k * z
        # Assuming k=1 for simplicity, so F = alpha * x + (1 - alpha) * z
        # Or more classically just using z as the hysteresis component:
        # F = alpha * x + z (when A controls the initial slope)
        F = self.alpha * x + z

        # Apply IIR filter
        F_filtered = signal.lfilter(self.b, self.a, F)

        return F_filtered

    def inverse_simulate(self, t: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Simulate the inverse generalized Bouc-Wen model.
        Calculates displacement x given restoring force F.
        Assumes IIR filter is identity (a=[1], b=[1]) for analytical inversion.
        F = alpha * x + z
        dz/dt = A * dx/dt - beta * |dx/dt| * |z|^(n-1) * z - gamma * dx/dt * |z|^n

        Args:
            t: Time array
            F: Target restoring force array

        Returns:
            np.ndarray: Displacement x
        """
        N = len(t)
        if N < 2:
            return np.zeros_like(F)

        x = np.zeros(N)
        z = np.zeros(N)

        # dF/dt
        dF = np.zeros(N)
        dF[0] = (F[1] - F[0]) / (t[1] - t[0])
        dF[-1] = (F[-1] - F[-2]) / (t[-1] - t[-2])
        if N > 2:
            dF[1:-1] = (F[2:] - F[:-2]) / (t[2:] - t[:-2])

        for i in range(1, N):
            dt = t[i] - t[i-1]

            # z_term calculation
            z_term_1 = 0.0
            if np.abs(z[i-1]) > 0 or self.n >= 1:
                z_term_1 = (np.abs(z[i-1]) ** (self.n - 1)) * z[i-1]

            z_term_2 = 0.0
            if np.abs(z[i-1]) > 0 or self.n >= 0:
                z_term_2 = np.abs(z[i-1]) ** self.n

            # Inverse model relationship
            # F = alpha * x + z => dF = alpha * dx + dz
            # dz = dx * (A - beta * sign(dx) * |z|^(n-1) * z - gamma * |z|^n)
            # dx = dF / (alpha + A - beta * sign(dx) * |z|^(n-1) * z - gamma * |z|^n)

            # Approximate sign(dx) with sign(dF) for hysteresis loop driving
            sign_dx = np.sign(dF[i-1])
            if sign_dx == 0:
                sign_dx = 1.0 # Default to positive

            asym_factor = 1.0 + self.delta * np.sign(x[i-1])

            denominator = (self.alpha + asym_factor * self.A -
                          self.beta * sign_dx * z_term_1 -
                          self.gamma * z_term_2)

            # Avoid division by zero
            if np.abs(denominator) < 1e-8:
                dx_dt = 0
            else:
                dx_dt = dF[i-1] / denominator

            dz_dt = dF[i-1] - self.alpha * dx_dt

            x[i] = x[i-1] + dx_dt * dt
            z[i] = z[i-1] + dz_dt * dt

        return x

    def identify_parameters_pso(self, t: np.ndarray, x: np.ndarray, F_target: np.ndarray,
                                num_particles: int = 30, max_iter: int = 50,
                                bounds: Tuple[np.ndarray, np.ndarray] = None) -> np.ndarray:
        """
        Identify classical BW parameters + asymmetry using Particle Swarm Optimization.

        Args:
            t: Time array
            x: Displacement input array
            F_target: Target restoring force array
            num_particles: Number of particles in the swarm
            max_iter: Maximum number of iterations
            bounds: Tuple of (lower_bounds, upper_bounds) for the 6 parameters
                    (A, alpha, beta, gamma, n, delta)

        Returns:
            np.ndarray: Best identified parameters
        """
        # Default bounds if not provided
        if bounds is None:
            # (A, alpha, beta, gamma, n, delta)
            lb = np.array([0.1, 0.0, -10.0, -10.0, 0.5, -0.9])
            ub = np.array([10.0, 5.0, 10.0, 10.0, 5.0, 0.9])
        else:
            lb, ub = bounds

        num_params = 6

        # Initialize swarm
        positions = np.random.uniform(lb, ub, (num_particles, num_params))
        velocities = np.zeros((num_particles, num_params))

        # Personal bests
        pbest_positions = positions.copy()
        pbest_scores = np.full(num_particles, np.inf)

        # Global best
        gbest_position = positions[0].copy()
        gbest_score = np.inf

        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5 # Cognitive parameter
        c2 = 1.5 # Social parameter

        # Backup original parameters
        orig_params = self.get_params()

        for iter in range(max_iter):
            for i in range(num_particles):
                # Evaluate fitness
                self.set_params(positions[i])
                F_sim = self.simulate(t, x)
                mse = np.mean((F_sim - F_target) ** 2)

                # Update personal best
                if mse < pbest_scores[i]:
                    pbest_scores[i] = mse
                    pbest_positions[i] = positions[i].copy()

                # Update global best
                if mse < gbest_score:
                    gbest_score = mse
                    gbest_position = positions[i].copy()

            # Update velocities and positions
            r1 = np.random.rand(num_particles, num_params)
            r2 = np.random.rand(num_particles, num_params)

            velocities = (w * velocities +
                          c1 * r1 * (pbest_positions - positions) +
                          c2 * r2 * (gbest_position - positions))

            positions = positions + velocities

            # Apply bounds
            positions = np.clip(positions, lb, ub)

        # Restore original parameters or set to best found
        self.set_params(gbest_position)

        return gbest_position

    def plot_hysteresis(self, t: np.ndarray, x: np.ndarray, F: np.ndarray,
                        F_pred: np.ndarray = None, title: str = 'Generalized Bouc-Wen Hysteresis',
                        filename: str = 'hysteresis.png'):
        """
        Plot the hysteresis loop (x vs F) and time histories.

        Args:
            t: Time array
            x: Displacement input array
            F: Restoring force array (target or true)
            F_pred: Predicted restoring force array (optional)
            title: Title of the plot
            filename: Filename to save the plot
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Time vs Displacement
        axs[0].plot(t, x, 'b-', label='Displacement (x)')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Displacement')
        axs[0].set_title('Time vs Displacement')
        axs[0].grid(True)
        axs[0].legend()

        # Plot 2: Time vs Force
        axs[1].plot(t, F, 'k-', label='Force (True)')
        if F_pred is not None:
            axs[1].plot(t, F_pred, 'r--', label='Force (Pred)')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Force')
        axs[1].set_title('Time vs Force')
        axs[1].grid(True)
        axs[1].legend()

        # Plot 3: Hysteresis Loop
        axs[2].plot(x, F, 'k-', label='True')
        if F_pred is not None:
            axs[2].plot(x, F_pred, 'r--', label='Predicted')
        axs[2].set_xlabel('Displacement (x)')
        axs[2].set_ylabel('Force (F)')
        axs[2].set_title('Hysteresis Loop (x vs F)')
        axs[2].grid(True)
        axs[2].legend()

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
