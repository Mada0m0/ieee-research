import numpy as np
from typing import Optional, Tuple, Dict, List

from implementations.hysteresis.rmso_optimizer import RMSOOptimizer


class RMSO_BW_Model:
    """
    基于 Bouc-Wen 模型的压电执行器迟滞建模，支持使用 RMSO 算法进行参数辨识。

    经典的 Bouc-Wen 模型通过非线性微分方程描述系统的迟滞特性，常用于表征压电执行器的率相关迟滞现象。
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        初始化 Bouc-Wen 模型参数。

        Args:
            params (Optional[Dict[str, float]]): 模型参数字典，包含:
                - A: 控制迟滞幅值
                - alpha: 控制迟滞形状
                - beta: 控制迟滞形状
                - gamma: 控制迟滞形状
                - n: 控制迟滞平滑度 (通常为 1 或 2)
        """
        if params is None:
            # 默认参数
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
        根据给定的输入位移/电压序列和时间步长，模拟 Bouc-Wen 迟滞输出。

        模型方程 (简化的针对位移输入的迟滞项计算):
        \\dot{h} = A \\cdot \\dot{u} - \\beta \\cdot |\\dot{u}| \\cdot h \\cdot |h|^{n-1} - \\gamma \\cdot \\dot{u} \\cdot |h|^n
        最终输出往往是线性项加上迟滞项 y = \\alpha \\cdot u + h，本实现仅计算输出迟滞项和完整输出。

        Args:
            u (np.ndarray): 输入序列 (1D array)。
            dt (float): 时间步长。

        Returns:
            np.ndarray: 模型输出序列 (1D array)，长度与输入相同。
        """
        N = len(u)
        h = np.zeros(N)
        y = np.zeros(N)

        A = self.params['A']
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']
        n = self.params['n']

        # 计算输入的导数 (速度)
        # 使用简单的差分，假设 u[0] 前的速度为 0
        u_dot = np.zeros(N)
        if N > 1:
            u_dot[1:] = (u[1:] - u[:-1]) / dt
            u_dot[0] = u_dot[1] # 简单的边界处理

        # 欧拉法求解微分方程
        for i in range(1, N):
            h_prev = h[i-1]
            u_dot_curr = u_dot[i]

            # Bouc-Wen 微分方程
            term1 = A * u_dot_curr
            term2 = beta * abs(u_dot_curr) * h_prev * (abs(h_prev) ** (n - 1))
            term3 = gamma * u_dot_curr * (abs(h_prev) ** n)

            h_dot = term1 - term2 - term3
            h[i] = h_prev + h_dot * dt

            # 计算总输出 y = alpha * u + h
            y[i] = alpha * u[i] + h[i]

        return y

    def identify_with_rmso(self, u_meas: np.ndarray, y_meas: np.ndarray, dt: float, bounds: Dict[str, Tuple[float, float]], pop_size: int = 50, max_iter: int = 200) -> Dict[str, float]:
        """
        使用 RMSO 算法辨识 Bouc-Wen 模型参数。

        Args:
            u_meas (np.ndarray): 测量的输入序列。
            y_meas (np.ndarray): 测量的输出序列。
            dt (float): 时间步长。
            bounds (Dict[str, Tuple[float, float]]): 参数边界，包含 A, alpha, beta, gamma, n。
            pop_size (int): 粒子总数。
            max_iter (int): 最大迭代次数。

        Returns:
            Dict[str, float]: 辨识得到的参数字典。
        """
        param_names = list(bounds.keys())

        def fitness_fn(p: np.ndarray) -> float:
            # 将数组映射回参数字典
            current_params = {name: val for name, val in zip(param_names, p)}
            self.params.update(current_params)

            # 模拟输出
            y_sim = self.simulate(u_meas, dt)

            # 计算均方误差 (MSE)
            mse = np.mean((y_sim - y_meas) ** 2)
            return mse

        # 确保粒子总数能被区域数整除，这里默认区域数为 5
        n_regions = 5
        if pop_size % n_regions != 0:
            pop_size = (pop_size // n_regions + 1) * n_regions

        optimizer = RMSOOptimizer(n_particles=pop_size, n_regions=n_regions)
        best_p, best_fitness = optimizer.optimize(fitness_fn, bounds, max_iter)

        # 将最优参数保存到模型中并返回
        self.params = {name: val for name, val in zip(param_names, best_p)}
        print(f"RMSO 辨识完成。最优 MSE: {best_fitness:.6f}")
        return self.params

    def compute_hysteresis_loop(self, u: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算迟滞环输出，用于绘图。

        Args:
            u (np.ndarray): 输入序列。
            dt (float): 时间步长。

        Returns:
            Tuple[np.ndarray, np.ndarray]: (输入序列, 输出序列)。
        """
        y = self.simulate(u, dt)
        return u, y

    def rate_dependent_simulate(self, frequencies: List[float], amplitude: float) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """
        生成在多个频率下的率相关迟滞环。

        Args:
            frequencies (List[float]): 频率列表 (Hz)。
            amplitude (float): 正弦输入信号的幅值。

        Returns:
            Dict[float, Tuple[np.ndarray, np.ndarray]]: 包含各个频率下的 (输入, 输出) 迟滞环数据。
        """
        loops = {}
        for freq in frequencies:
            # 设定采样率，确保每个周期有足够的点
            fs = max(1000, int(100 * freq))
            dt = 1.0 / fs
            t = np.arange(0, 2.0 / freq, dt) # 模拟 2 个周期以达到稳定状态

            u = amplitude * np.sin(2 * np.pi * freq * t)
            y = self.simulate(u, dt)

            # 只提取稳定后的第二个周期数据用于展示
            samples_per_cycle = int(fs / freq)
            u_stable = u[samples_per_cycle:]
            y_stable = y[samples_per_cycle:]

            loops[freq] = (u_stable, y_stable)

        return loops
