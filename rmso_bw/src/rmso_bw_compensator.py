import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional

from rmso_bw.src.rmso_bw_model import RMSO_BW_Model
from rmso_bw.src.fuzzy_nn_controller import FuzzyNNController


class RMSO_BW_Compensator:
    """
    RMSO-BW 迟滞补偿器.

    结合了基于 RMSO 辨识的 Bouc-Wen 逆模型（作为前馈）
    和自适应 Fuzzy-NN 控制器（作为反馈误差补偿），
    实现对压电执行器的高精度轨迹跟踪控制。
    """

    def __init__(self, bw_model: RMSO_BW_Model, fuzzy_controller: FuzzyNNController):
        """
        初始化补偿器。

        Args:
            bw_model (RMSO_BW_Model): 已辨识好参数的 RMSO_BW_Model 实例。
            fuzzy_controller (FuzzyNNController): 用于在线补偿的 Fuzzy-NN 控制器实例。
        """
        self.bw_model = bw_model
        self.fuzzy_controller = fuzzy_controller

        # Save the state of the first-order hysteresis term used in the forward calculation
        self.h_state = 0.0

    def compute_control(self, desired_position: float, current_position: float, desired_velocity: float, dt: float) -> float:
        """
        计算控制电压。
        包含前馈控制 (基于 BW 模型近似求逆) 和反馈控制 (Fuzzy-NN).

        由于直接求 Bouc-Wen 逆模型解析解困难，常使用近似前馈或增量计算。
        这里使用简化的近似前馈：
        y = alpha * u + h  =>  u_ff = (desired_position - h) / alpha
        加上模糊控制器的反馈量。

        Args:
            desired_position (float): 期望位移。
            current_position (float): 当前实际位移。
            desired_velocity (float): 期望速度 (用于估计迟滞状态)。
            dt (float): 时间步长。

        Returns:
            float: 控制电压 u。
        """
        # 1. Estimate the hysteretic internal state h_dot
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
        # Assume n_inputs = 2: [error, error_dot] (here simply use the error difference from the previous moment, if not provided, pass a fixed value or a simple estimate)
        # For simplicity, assume that the state passed to the controller is the error
        fuzzy_input = np.array([error, error]) if self.fuzzy_controller.n_inputs == 2 else np.array([error])

        u_fb = self.fuzzy_controller.forward(fuzzy_input)[0]

        # 4. Total control amount
        u_total = u_ff + u_fb

        return u_total

    def track(self, desired_trajectory: np.ndarray, plant_fn: Callable[[float, float], float], dt: float, online_learning_rate: float = 0.001) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        闭环跟踪仿真。

        Args:
            desired_trajectory (np.ndarray): 期望位移序列。
            plant_fn (Callable[[float, float], float]): 真实的被控对象函数，输入为 (电压u, 时间步长dt)，返回实际位移。
            dt (float): 时间步长。
            online_learning_rate (float): 模糊神经网络的在线学习率。

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (时间序列, 期望位移序列, 实际位移序列)
        """
        N = len(desired_trajectory)
        time_seq = np.arange(N) * dt
        actual_trajectory = np.zeros(N)
        control_signals = np.zeros(N)

        # initialization state
        self.h_state = 0.0
        current_pos = 0.0

        # Calculate desired velocity sequence
        desired_velocity = np.zeros(N)
        if N > 1:
            desired_velocity[1:] = (desired_trajectory[1:] - desired_trajectory[:-1]) / dt
            desired_velocity[0] = desired_velocity[1]

        for i in range(N):
            pos_d = desired_trajectory[i]
            vel_d = desired_velocity[i]

            # Calculate control signals
            u = self.compute_control(pos_d, current_pos, vel_d, dt)
            control_signals[i] = u

            # Apply control to real objects
            current_pos = plant_fn(u, dt)
            actual_trajectory[i] = current_pos

            # Online adaptive update Fuzzy-NN
            error = pos_d - current_pos
            fuzzy_input = np.array([error, error]) if self.fuzzy_controller.n_inputs == 2 else np.array([error])

            # The controller goal is to make the error tend to 0, so the ideal output is a compensation value that can offset the current error.
            # A simple gradient update target can be directly set to the current u_fb + k*error (guided output correction error)
            # Simplified version: We set the ideal reference to the value required to produce 0 error.
            # Here adapt_online is called to directly use the tracking error to adjust.
            # To use the previous adapt_online interface (fitting y_true), we set a dummy goal: we want the blur output to be larger or smaller to reduce the error.
            virtual_target = np.array([self.fuzzy_controller.forward(fuzzy_input)[0] + error])
            self.fuzzy_controller.adapt_online(fuzzy_input, virtual_target, lr=online_learning_rate)

        return time_seq, desired_trajectory, actual_trajectory

    def plot_tracking(self, t: np.ndarray, desired: np.ndarray, actual: np.ndarray, save_path: Optional[str] = None):
        """
        绘制跟踪结果和误差。

        Args:
            t (np.ndarray): 时间序列。
            desired (np.ndarray): 期望轨迹。
            actual (np.ndarray): 实际轨迹。
            save_path (Optional[str]): 图像保存路径。
        """
        error = desired - actual
        rmse = np.sqrt(np.mean(error ** 2))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Trajectory diagram
        ax1.plot(t, desired, 'b--', label='Desired Trajectory')
        ax1.plot(t, actual, 'r-', label='Actual Trajectory')
        ax1.set_title(f'Tracking Performance (RMSE: {rmse:.6f})')
        ax1.set_ylabel('Position')
        ax1.legend()
        ax1.grid(True)

        # error plot
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
