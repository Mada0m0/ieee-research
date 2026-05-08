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

        # 保存用于前向计算的一阶迟滞项状态
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
        # 1. 估计迟滞内部状态 h_dot
        A = self.bw_model.params['A']
        beta = self.bw_model.params['beta']
        gamma = self.bw_model.params['gamma']
        n = self.bw_model.params['n']
        alpha = self.bw_model.params['alpha']

        # 使用期望速度来近似推进内部迟滞状态 (前馈观测器)
        term1 = A * desired_velocity
        term2 = beta * abs(desired_velocity) * self.h_state * (abs(self.h_state) ** (n - 1))
        term3 = gamma * desired_velocity * (abs(self.h_state) ** n)

        h_dot = term1 - term2 - term3
        self.h_state += h_dot * dt

        # 2. 前馈控制量 (Feedforward)
        u_ff = (desired_position - self.h_state) / (alpha + 1e-8)

        # 3. 反馈控制量 (Feedback via Fuzzy-NN)
        error = desired_position - current_position
        # 模糊控制器的输入通常是误差和误差的变化率，为了简化这里用 [error, current_position] 或只用 error
        # 假设 n_inputs = 2: [error, error_dot] (这里简单用上一时刻误差差分，如果没提供就传固定值或简单估计)
        # 为简单起见，假设传给控制器的状态就是误差
        fuzzy_input = np.array([error, error]) if self.fuzzy_controller.n_inputs == 2 else np.array([error])

        u_fb = self.fuzzy_controller.forward(fuzzy_input)[0]

        # 4. 总控制量
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

        # 初始化状态
        self.h_state = 0.0
        current_pos = 0.0

        # 计算期望速度序列
        desired_velocity = np.zeros(N)
        if N > 1:
            desired_velocity[1:] = (desired_trajectory[1:] - desired_trajectory[:-1]) / dt
            desired_velocity[0] = desired_velocity[1]

        for i in range(N):
            pos_d = desired_trajectory[i]
            vel_d = desired_velocity[i]

            # 计算控制信号
            u = self.compute_control(pos_d, current_pos, vel_d, dt)
            control_signals[i] = u

            # 施加控制到真实对象
            current_pos = plant_fn(u, dt)
            actual_trajectory[i] = current_pos

            # 在线自适应更新 Fuzzy-NN
            error = pos_d - current_pos
            fuzzy_input = np.array([error, error]) if self.fuzzy_controller.n_inputs == 2 else np.array([error])

            # 控制器目标是让 error 趋于 0，所以理想输出是能抵消当前 error 的补偿值。
            # 简单的梯度更新目标可以直接设为当前的 u_fb + k*error (引导输出修正误差)
            # 简化版：我们将理想的参考设定为产生0误差所需的值。
            # 这里调用 adapt_online，直接利用跟踪误差来调整。
            # 为了使用之前的 adapt_online 接口 (拟合 y_true)，我们设定虚拟目标：希望模糊输出更大或更小来减小误差。
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

        # 轨迹图
        ax1.plot(t, desired, 'b--', label='Desired Trajectory')
        ax1.plot(t, actual, 'r-', label='Actual Trajectory')
        ax1.set_title(f'Tracking Performance (RMSE: {rmse:.6f})')
        ax1.set_ylabel('Position')
        ax1.legend()
        ax1.grid(True)

        # 误差图
        ax2.plot(t, error, 'k-')
        ax2.set_title('Tracking Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error')
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"跟踪结果图已保存至: {save_path}")
        else:
            plt.show()
        plt.close()
