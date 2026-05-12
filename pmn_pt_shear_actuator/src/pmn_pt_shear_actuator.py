"""
PMN-PT 剪切模式（d36）压电执行器模型及控制算法

该模块提供了基于 PMN-PT 压电单晶 d36 剪切模式的执行器模型、
迟滞补偿器、扩展状态观测器（ESO）和自抗扰控制器（ADRC）。
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union

class ShearActuatorPlant:
    """
    PMN-PT d36 剪切模式执行器模型。

    建模 PMN-PT d36 剪切模式的电-机耦合行为，包括：
    - 压电效应（基于 d36 剪切模式参数）
    - 速率依赖迟滞（Bouc-Wen 模型改进）
    - 对数型蠕变特性
    """

    def __init__(self,
                 d36: float = 2500e-12,
                 s55E: float = 60e-12,
                 epsilon33T: float = 5000 * 8.854e-12,
                 length: float = 10e-3,
                 width: float = 10e-3,
                 thickness: float = 1e-3):
        """
        初始化执行器模型。

        参数:
            d36: 压电剪切系数 (C/N 或 m/V), 典型值 2000-3000 pC/N
            s55E: 弹性柔顺系数 (m^2/N)
            epsilon33T: 自由状态介电常数 (F/m)
            length: 长度 (m)
            width: 宽度 (m)
            thickness: 厚度 (m)
        """
        # Geometric parameters
        self.length = length
        self.width = width
        self.thickness = thickness
        self.area = length * width

        # Physical parameters
        self.d36 = d36
        self.s55E = s55E
        self.epsilon33T = epsilon33T

        # Stiffness (shear)
        self.k_shear = self.area / (self.s55E * self.thickness)

        # state variables
        self.gamma = 0.0          # shear strain
        self.gamma_dot = 0.0      # strain rate
        self.displacement = 0.0   # actual displacement
        self.Q = 0.0              # charge
        self.voltage = 0.0        # input voltage

        # Hysteresis model parameters (Bouc-Wen modified)
        self.A = 1.0
        self.beta = 0.1
        self.gamma_bw = 0.1
        self.n = 1
        self.h_var = 0.0          # hysteresis internal state

        # Creep model parameters
        self.gamma_0 = 0.05       # Creep coefficient
        self.tau_creep = 0.1      # Creep time constant
        self.time = 0.0
        self.last_voltage_change_time = 0.0
        self.voltage_step = 0.0

        # Dynamic mass, damping
        self.mass = 0.01  # Effective mass kg
        # Increase damping to avoid numerical instability caused by extremely high stiffness
        self.damping = 1000.0 # Damping N/(m/s)

    def get_parameters(self) -> Dict[str, float]:
        """Get the current model parameters."""
        return {
            'd36': self.d36,
            's55E': self.s55E,
            'epsilon33T': self.epsilon33T,
            'length': self.length,
            'width': self.width,
            'thickness': self.thickness,
            'k_shear': self.k_shear,
            'mass': self.mass,
            'damping': self.damping
        }

    def _creep_model(self, t: float, dt: float) -> float:
        """
        对数型蠕变模型计算。
        gamma_creep(t) = gamma_0 * log(1 + t/tau)
        """
        if self.voltage_step == 0:
            return 0.0

        elapsed_time = t - self.last_voltage_change_time
        if elapsed_time < 0:
            return 0.0

        creep_strain = self.gamma_0 * self.voltage_step * np.log10(1 + elapsed_time / self.tau_creep)
        return creep_strain * self.d36 / self.thickness

    def _hysteresis_model(self, v_dot: float, dt: float) -> float:
        """
        基于 Bouc-Wen 的速率依赖迟滞模型。
        """
        # rate dependence factor
        rate_factor = 1.0 + 0.1 * np.abs(v_dot)

        # Bouc-Wen status update
        h_dot = self.A * v_dot - rate_factor * self.beta * np.abs(v_dot) * self.h_var * (np.abs(self.h_var)**(self.n-1)) - rate_factor * self.gamma_bw * v_dot * (np.abs(self.h_var)**self.n)

        self.h_var += h_dot * dt
        return self.h_var

    def simulate(self, voltage_seq: np.ndarray, dt: float, load_seq: Optional[np.ndarray] = None, temp_seq: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        模拟执行器在给定电压序列下的响应。

        参数:
            voltage_seq: 电压时间序列 (V)
            dt: 时间步长 (s)
            load_seq: 外部负载力序列 (N), 可选
            temp_seq: 温度序列 (摄氏度), 可选

        返回:
            (时间序列, 位移序列, 应变序列)
        """
        n_steps = len(voltage_seq)
        time_seq = np.arange(n_steps) * dt
        displacement_seq = np.zeros(n_steps)
        strain_seq = np.zeros(n_steps)

        # initialization state
        displacement = 0.0
        velocity = 0.0
        self.h_var = 0.0
        self.last_voltage_change_time = 0.0
        self.time = 0.0

        prev_v = 0.0

        for i in range(n_steps):
            v = voltage_seq[i]
            t = time_seq[i]
            self.time = t

            load = load_seq[i] if load_seq is not None else 0.0
            temp = temp_seq[i] if temp_seq is not None else 25.0

            # Effect of temperature on d36 (simple linear model)
            temp_factor = 1.0 + 0.005 * (temp - 25.0)
            current_d36 = self.d36 * temp_factor

            v_dot = (v - prev_v) / dt if i > 0 else 0.0

            # Record voltage steps for creep calculations (simplified: detect large changes)
            if np.abs(v - prev_v) > 0.1:
                self.last_voltage_change_time = t
                self.voltage_step = v - prev_v

            # Hysteresis calculation
            hysteresis_force = self._hysteresis_model(v_dot, dt)

            # Creep calculation
            creep_strain = self._creep_model(t, dt)

            # Piezoelectric actuation force (F = d36/s55 * V/thickness * area) Simplified
            # Here we use the displacement model: x = d36 * V + hysteresis + creep

            # ideal piezoelectric displacement
            ideal_displacement = current_d36 * v

            # Total equivalent displacement considering hysteresis and creep
            # Assuming that the hysteresis term is of the same magnitude as the voltage, scaling is introduced
            hysteresis_displacement = hysteresis_force * current_d36 * 0.5
            creep_displacement = creep_strain * self.thickness

            target_displacement = ideal_displacement - hysteresis_displacement + creep_displacement

            # Add external load effects (static compliance)
            load_displacement = load / self.k_shear
            target_displacement -= load_displacement

            # Simplify to a first-order system or direct static mapping to avoid the numerical instability of the explicit Euler method at extremely high stiffnesses
            # x = target_x (assuming the dynamic response is much faster than the control period dt=1ms)
            # Introduce a simple first-order low-pass characteristic to simulate the dynamic response
            tau_dynamic = 0.001 # 1ms
            displacement = displacement + (target_displacement - displacement) * (dt / (dt + tau_dynamic))
            velocity = (displacement - displacement_seq[i-1]) / dt if i > 0 else 0.0

            displacement_seq[i] = displacement
            strain_seq[i] = displacement / self.thickness

            prev_v = v

        self.displacement = displacement
        self.gamma = displacement / self.thickness
        self.gamma_dot = velocity / self.thickness

        return time_seq, displacement_seq, strain_seq

class ShearHysteresisCompensator:
    """
    基于 PI（Prandtl-Ishlinskii）逆模型的前馈补偿器。

    使用叠加算子和 play 算子对速率依赖阈值进行分布，
    并通过 RLS（递归最小二乘）算法在线更新参数。
    """
    def __init__(self, num_operators: int = 10, max_threshold: float = 10.0):
        self.num_operators = num_operators
        self.thresholds = np.linspace(0, max_threshold, num_operators)
        self.weights = np.ones(num_operators) * 0.1 # Initial weight
        self.states = np.zeros(num_operators)       # play operator status

        # RLS parameters
        self.P = np.eye(num_operators) * 1000.0     # covariance matrix
        self.lambda_rls = 0.995                     # forgetting factor

    def _play_operator(self, v_in: float, threshold: float, state: float) -> float:
        """Basic Play operator"""
        return max(v_in - threshold, min(v_in + threshold, state))

    def compensate(self, reference_displacement: float, current_v: float = 0.0) -> float:
        """
        基于当前参考位移计算前馈补偿电压。
        (简化实现：将参考位移映射为补偿电压)
        """
        # Calculate the output of each operator
        for i in range(self.num_operators):
            self.states[i] = self._play_operator(reference_displacement, self.thresholds[i], self.states[i])

        # Compensation voltage = weighted sum of operator outputs
        v_comp = np.dot(self.weights, self.states)
        return v_comp

    def update_parameters_rls(self, actual_displacement: float, reference_displacement: float):
        """
        在线辨识：递归最小二乘参数更新。
        (此方法在闭环中运行，以减小模型误差)
        """
        # Feature vector (current states)
        phi = self.states.reshape(-1, 1)

        # Error (expected - actual)
        error = reference_displacement - actual_displacement

        # RLS updates
        # K = P * phi / (lambda + phi^T * P * phi)
        # P = (P - K * phi^T * P) / lambda
        # w = w + K * error

        numerator = self.P @ phi
        denominator = self.lambda_rls + phi.T @ self.P @ phi

        # Avoid dividing by 0
        if denominator[0, 0] > 1e-6:
            K = numerator / denominator[0, 0]
            self.P = (self.P - K @ phi.T @ self.P) / self.lambda_rls
            self.weights += (K.flatten() * error)


class ShearMotorESO:
    """
    3阶扩展状态观测器 (ESO)。

    用于观测位移、速度以及总扰动（包含蠕变、迟滞、负载、温度漂移等）。
    特针对剪切模式优化了 fal 函数参数。
    """
    def __init__(self, w0: float = 300.0, dt: float = 0.001):
        """
        参数:
            w0: 观测器带宽 (rad/s), 典型值 100-500
            dt: 采样时间
        """
        self.w0 = w0
        self.dt = dt

        # Gain configuration (based on bandwidth w0)
        self.beta1 = 3 * w0
        self.beta2 = 3 * (w0 ** 2)
        self.beta3 = w0 ** 3

        # fal parameter for shear mode (simplify to linear ESO or adopt parameters more friendly to displacement magnitude)
        self.alpha = [1.0, 1.0, 1.0] # First use linear ESO to test convergence to avoid problems caused by nonlinearity
        self.delta = 1e-6

        # Status: z1 (displacement estimate), z2 (velocity estimate), z3 (total disturbance estimate)
        self.z = np.zeros(3)

        # Nominal control gain b0 (usually 1/mass, or based on system calibration)
        self.b0 = 100.0

    def _fal(self, e: float, alpha: float, delta: float) -> float:
        """nonlinear function fal"""
        if abs(e) <= delta:
            return e / (delta ** (1 - alpha))
        else:
            return (abs(e) ** alpha) * np.sign(e)

    def update(self, y_meas: float, u: float):
        """
        更新状态估计。

        参数:
            y_meas: 测量位移
            u: 控制输入 (电压)
        """
        e = self.z[0] - y_meas

        fe = self._fal(e, self.alpha[0], self.delta)
        fe1 = self._fal(e, self.alpha[1], self.delta)
        fe2 = self._fal(e, self.alpha[2], self.delta)

        # Euler discretization of continuous-time state equation (sign corrected, e = z1 - y, observer equation should be minus error correction term)
        z1_dot = self.z[1] - self.beta1 * e
        z2_dot = self.z[2] - self.beta2 * fe + self.b0 * u
        z3_dot = -self.beta3 * fe1

        self.z[0] += z1_dot * self.dt
        self.z[1] += z2_dot * self.dt
        self.z[2] += z3_dot * self.dt

    def get_disturbance(self) -> float:
        """Get the current total disturbance estimate"""
        return self.z[2]

    def get_estimated_displacement(self) -> float:
        """Get the filtered displacement estimate"""
        return self.z[0]

    def get_estimated_velocity(self) -> float:
        """Get estimated speed"""
        return self.z[1]

class ShearADRCController:
    """
    自抗扰控制器 (ADRC) 用于 PMN-PT 剪切模式执行器。

    集成了：
    - 跟踪微分器 (TD): 安排过渡过程并提取微分
    - 扩展状态观测器 (ESO): 估计状态和总扰动
    - 非线性状态误差反馈 (NLSEF): 计算控制律
    支持防积分饱和、蠕变前馈补偿和温度自适应。
    """
    def __init__(self, wc: float = 100.0, w0: float = 300.0, b0: float = 100.0, dt: float = 0.001):
        self.wc = wc
        self.dt = dt
        self.b0 = b0

        # submodule
        self.eso = ShearMotorESO(w0=w0, dt=dt)
        self.eso.b0 = b0

        # TD status
        self.v1 = 0.0 # arranged displacement
        self.v2 = 0.0 # Arrangement speed
        # Greatly reduce the speed factor of TD to avoid overshoot and oscillation caused by sudden changes in the reference signal
        self.r = 1e-4 # speed factor
        self.h = dt   # filter factor

        # NLSEF parameter (greatly reduces control gain to prevent oscillation from diverging)
        self.kp = wc ** 2 * 0.01
        self.kd = 2 * wc * 0.01
        self.lambda_nl = 1.0 # Use linear PD to simplify debugging and avoid divergence

        # other
        self.u_prev = 0.0
        self.u_max = 1000.0  # Maximum voltage (relaxed limit to allow faster dynamic response)
        self.u_min = -1000.0 # minimum voltage

        # creep feedforward
        self.gamma_0 = 0.05
        self.tau_creep = 0.1
        self.last_step_time = 0.0
        self.last_setpoint = 0.0
        self.time = 0.0

    def _fhan(self, x1: float, x2: float, r: float, h: float) -> float:
        """最速控制综合函数 (离散系统)"""
        d = r * h
        d0 = h * d
        y = x1 + h * x2
        a0 = np.sqrt(d ** 2 + 8 * r * abs(y))

        if abs(y) > d0:
            a = x2 + (a0 - d) / 2 * np.sign(y)
        else:
            a = x2 + y / h

        if abs(a) > d:
            return -r * np.sign(a)
        else:
            return -r * a / d

    def track(self, setpoint: float, y_meas: float, temp: float = 25.0) -> float:
        """
        计算控制电压。

        参数:
            setpoint: 目标位移
            y_meas: 测量位移
            temp: 环境温度 (用于自适应)
        """
        self.time += self.dt

        # Temperature adaptation: adjust b0 of ESO (since d36 changes with temperature)
        # Assuming that b0 is nominal at 25 degrees, each degree change results in a 0.5% change in d36, which in turn affects b0
        temp_factor = 1.0 + 0.005 * (temp - 25.0)
        self.eso.b0 = self.b0 * temp_factor

        # 1. Tracking Differentiator (TD)
        fh = self._fhan(self.v1 - setpoint, self.v2, self.r, self.h)
        self.v1 += self.v2 * self.dt
        self.v2 += fh * self.dt

        # 2. Extended State Observer (ESO)
        self.eso.update(y_meas, self.u_prev)
        z1 = self.eso.get_estimated_displacement()
        z2 = self.eso.get_estimated_velocity()
        z3 = self.eso.get_disturbance()

        # 3. Nonlinear State Error Feedback (NLSEF)
        e1 = self.v1 - z1
        e2 = self.v2 - z2

        # PD control with nonlinearity (modify the delta threshold of the fal function to the appropriate displacement magnitude, or directly use linear control e1, e2)
        if self.lambda_nl == 1.0:
            u0 = self.kp * e1 + self.kd * e2
        else:
            u0 = self.kp * self.eso._fal(e1, self.lambda_nl, 1e-6) + self.kd * self.eso._fal(e2, self.lambda_nl, 1e-6)

        # 4. Disturbance compensation
        u = (u0 - z3) / self.eso.b0

        # 5. Creep feedforward compensation
        if abs(setpoint - self.last_setpoint) > 1e-6:
            self.last_step_time = self.time
            self.last_setpoint = setpoint

        elapsed = self.time - self.last_step_time
        creep_comp = 0.0
        if elapsed > 0:
            # Simplified version of creep feedforward: predict creep and reverse compensation
            predicted_creep = self.gamma_0 * setpoint * np.log10(1 + elapsed / self.tau_creep)
            # Convert displacement creep to compensation voltage (simplification factor)
            creep_comp = -predicted_creep * 0.1

        u += creep_comp

        # 6. Anti-saturation
        u = np.clip(u, self.u_min, self.u_max)
        self.u_prev = u

        return u


class ShearCharacterizer:
    """
    自动表征压电迟滞主环和次环、参数辨识。
    """
    def __init__(self):
        self.d36_identified = 0.0
        self.hysteresis_width = 0.0
        self.tau_creep = 0.0

    def generate_characterization_waveforms(self, max_voltage: float = 100.0, dt: float = 0.001) -> np.ndarray:
        """
        生成用于表征的电压序列 (包含主环和次环的三角波)。
        """
        t1 = np.arange(0, 1.0, dt)
        # main ring
        w1 = max_voltage * np.sin(2 * np.pi * 1.0 * t1)

        # secondary ring
        t2 = np.arange(0, 0.5, dt)
        w2 = (max_voltage * 0.5) * np.sin(2 * np.pi * 2.0 * t2) + (max_voltage * 0.5)

        # Splice and ensure smooth transitions
        w = np.concatenate([w1, w2])
        return w

    def identify_parameters(self, voltage_seq: np.ndarray, displacement_seq: np.ndarray, time_seq: np.ndarray) -> Dict[str, float]:
        """
        根据测试数据辨识核心参数 d36, 迟滞宽度, 和近似蠕变常数。
        """
        # Simple linear regression identification d36
        # Select a linear segment with a higher voltage, or roughly estimate by maximum displacement/maximum voltage
        max_idx = np.argmax(voltage_seq)
        min_idx = np.argmin(voltage_seq)

        dv = voltage_seq[max_idx] - voltage_seq[min_idx]
        dd = displacement_seq[max_idx] - displacement_seq[min_idx]

        self.d36_identified = dd / dv if dv != 0 else 2500e-12

        # Hysteresis width estimation (find the displacement difference when the voltage is 0)
        zero_crossings = np.where(np.diff(np.sign(voltage_seq)))[0]
        widths = []
        for z in zero_crossings:
            widths.append(abs(displacement_seq[z]))
        self.hysteresis_width = np.mean(widths) if widths else 0.0

        # The creep constant tau_creep is assumed to be given or identified through step response. Here is a basic identification structure.
        self.tau_creep = 0.1 # Simplified identification results

        return {
            'd36': self.d36_identified,
            'hysteresis_width': self.hysteresis_width,
            'tau_creep': self.tau_creep
        }
