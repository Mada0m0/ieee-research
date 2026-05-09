import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional

from implementations.hysteresis.rmso_bw_model import RMSO_BW_Model
from implementations.controls.fuzzy_nn_controller import FuzzyNNController


class RMSO_BW_Compensator:
    """
    RMSO-BW hysteresis compensator.

    Incorporated Bouc-Wen inverse model based on RMSO identification (as feedforward)
    and adaptive Fuzzy-NN controller (as feedback error compensation),
    Realize high-precision trajectory tracking control of piezoelectric actuators.
    """

    def __init__(self, bw_model: RMSO_BW_Model, fuzzy_controller: FuzzyNNController):
        """
        Initialize the compensator.

        Args:
            bw_model (RMSO_BW_Model): RMSO_BW_Model instance with identified parameters.
            fuzzy_controller (FuzzyNNController): Fuzzy-NN controller instance for online compensation.
        """
        self.bw_model = bw_model
        self.fuzzy_controller = fuzzy_controller

        #Save the first-order hysteresis term state for forward calculation
        self.h_state = 0.0

    def compute_control(self, desired_position: float, current_position: float, desired_velocity: float, dt: float) -> float:
        """
        Calculate the control voltage.
        Contains feedforward control (based on approximate inversion of BW model) and feedback control (Fuzzy-NN).

        Since it is difficult to directly obtain the analytical solution of the Bouc-Wen inverse model, approximate feedforward or incremental calculation is often used.
        A simplified approximate feedforward is used here:
        y = alpha * u + h  =>  u_ff = (desired_position - h) / alpha
        Plus the amount of feedback from the fuzzy controller.

        Args:
            desired_position (float): desired displacement.
            current_position (float): current actual displacement.
            desired_velocity (float): Desired velocity (used to estimate hysteresis).
            dt (float): time step.

        Returns:
            float: control voltage u.
        """
        # 1. Estimate hysteresis internal state h_dot
        A = self.bw_model.params['A']
        beta = self.bw_model.params['beta']
        gamma = self.bw_model.params['gamma']
        n = self.bw_model.params['n']
        alpha = self.bw_model.params['alpha']

        # Use desired velocity to approximate advancing internal hysteresis state (feedforward observer)
        term1 = A * desired_velocity
        term2 = beta * abs(desired_velocity) * self.h_state * (abs(self.h_state) ** (n - 1))
        term3 = gamma * desired_velocity * (abs(self.h_state) ** n)

        h_dot = term1 - term2 - term3
        self.h_state += h_dot * dt

        # 2. Feedforward control amount (Feedforward)
        u_ff = (desired_position - self.h_state) / (alpha + 1e-8)

        # 3. Feedback control amount (Feedback via Fuzzy-NN)
        error = desired_position - current_position
        # The input of the fuzzy controller is usually the error and the rate of change of the error. For simplicity, [error, current_position] or just error are used here.
        # Assume n_inputs = 2: [error, error_dot] (Here we simply use the error difference from the previous moment. If not provided, pass a fixed value or a simple estimate)
        # For simplicity, assume that the state passed to the controller is the error
        fuzzy_input = np.array([error, error]) if self.fuzzy_controller.n_inputs == 2 else np.array([error])

        u_fb = self.fuzzy_controller.forward(fuzzy_input)[0]

        # 4. Total control volume
        u_total = u_ff + u_fb

        return u_total

    def track(self, desired_trajectory: np.ndarray, plant_fn: Callable[[float, float], float], dt: float, online_learning_rate: float = 0.001) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Closed loop tracking simulation.

        Args:
            desired_trajectory (np.ndarray): Desired displacement sequence.
            plant_fn (Callable[[float, float], float]): The real controlled object function, the input is (voltage u, time step dt), and returns the actual displacement.
            dt (float): time step.
            online_learning_rate (float): The online learning rate of the fuzzy neural network.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (time series, expected displacement series, actual displacement series)
        """
        N = len(desired_trajectory)
        time_seq = np.arange(N) * dt
        actual_trajectory = np.zeros(N)
        control_signals = np.zeros(N)

        #Initialization state
        self.h_state = 0.0
        current_pos = 0.0

        # Calculate the desired speed sequence
        desired_velocity = np.zeros(N)
        if N > 1:
            desired_velocity[1:] = (desired_trajectory[1:] - desired_trajectory[:-1]) / dt
            desired_velocity[0] = desired_velocity[1]

        for i in range(N):
            pos_d = desired_trajectory[i]
            vel_d = desired_velocity[i]

            # Calculate control signal
            u = self.compute_control(pos_d, current_pos, vel_d, dt)
            control_signals[i] = u

            # Apply control to real objects
            current_pos = plant_fn(u, dt)
            actual_trajectory[i] = current_pos

            # Online adaptive update Fuzzy-NN
            error = pos_d - current_pos
            fuzzy_input = np.array([error, error]) if self.fuzzy_controller.n_inputs == 2 else np.array([error])

            # The controller's goal is to make the error tend to 0, so the ideal output is a compensation value that can offset the current error.
            # The simple gradient update target can be directly set to the current u_fb + k*error (guided output correction error)
            # Simplified version: We set the ideal reference to the value required to produce 0 error.
            # Here we call adapt_online and directly use the tracking error to adjust.
            # In order to use the previous adapt_online interface (fitting y_true), we set a virtual goal: we hope that the blur output will be larger or smaller to reduce the error.
            virtual_target = np.array([self.fuzzy_controller.forward(fuzzy_input)[0] + error])
            self.fuzzy_controller.adapt_online(fuzzy_input, virtual_target, lr=online_learning_rate)

        return time_seq, desired_trajectory, actual_trajectory

    def plot_tracking(self, t: np.ndarray, desired: np.ndarray, actual: np.ndarray, save_path: Optional[str] = None):
        """
        Plot tracking results and errors.

        Args:
            t (np.ndarray): time series.
            desired (np.ndarray): desired trajectory.
            actual (np.ndarray): actual trajectory.
            save_path (Optional[str]): Image saving path.
        """
        error = desired - actual
        rmse = np.sqrt(np.mean(error ** 2))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Trajectory map
        ax1.plot(t, desired, 'b--', label='Desired Trajectory')
        ax1.plot(t, actual, 'r-', label='Actual Trajectory')
        ax1.set_title(f'Tracking Performance (RMSE: {rmse:.6f})')
        ax1.set_ylabel('Position')
        ax1.legend()
        ax1.grid(True)

        # Error graph
        ax2.plot(t, error, 'k-')
        ax2.set_title('Tracking Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error')
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"The tracking result graph has been saved to: {save_path}")
        else:
            plt.show()
        plt.close()
