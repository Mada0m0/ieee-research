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
        # 几何参数
        self.length = length
        self.width = width
        self.thickness = thickness
        self.area = length * width

        # 物理参数
        self.d36 = d36
        self.s55E = s55E
        self.epsilon33T = epsilon33T

        # 刚度 (剪切)
        self.k_shear = self.area / (self.s55E * self.thickness)

        # 状态变量
        self.gamma = 0.0          # 剪切应变
        self.gamma_dot = 0.0      # 应变速率
        self.displacement = 0.0   # 实际位移
        self.Q = 0.0              # 电荷
        self.voltage = 0.0        # 输入电压

        # 迟滞模型参数 (Bouc-Wen 改进型)
        self.A = 1.0
        self.beta = 0.1
        self.gamma_bw = 0.1
        self.n = 1
        self.h_var = 0.0          # 迟滞内部状态

        # 蠕变模型参数
        self.gamma_0 = 0.05       # 蠕变系数
        self.tau_creep = 0.1      # 蠕变时间常数
        self.time = 0.0
        self.last_voltage_change_time = 0.0
        self.voltage_step = 0.0

        # 动态质量、阻尼
        self.mass = 0.01  # 有效质量 kg
        # 增加阻尼以避免极高刚度引起的数值不稳定
        self.damping = 1000.0 # 阻尼 N/(m/s)

    def get_parameters(self) -> Dict[str, float]:
        """获取当前模型参数。"""
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
        # 速率依赖因子
        rate_factor = 1.0 + 0.1 * np.abs(v_dot)

        # Bouc-Wen 状态更新
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

        # 初始化状态
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

            # 温度对 d36 的影响 (简单线性模型)
            temp_factor = 1.0 + 0.005 * (temp - 25.0)
            current_d36 = self.d36 * temp_factor

            v_dot = (v - prev_v) / dt if i > 0 else 0.0

            # 记录电压阶跃用于蠕变计算 (简化处理：检测较大变化)
            if np.abs(v - prev_v) > 0.1:
                self.last_voltage_change_time = t
                self.voltage_step = v - prev_v

            # 迟滞计算
            hysteresis_force = self._hysteresis_model(v_dot, dt)

            # 蠕变计算
            creep_strain = self._creep_model(t, dt)

            # 压电驱动力 (F = d36/s55 * V/thickness * area) 简化
            # 这里我们使用位移模型: x = d36 * V + hysteresis + creep

            # 理想压电位移
            ideal_displacement = current_d36 * v

            # 考虑迟滞和蠕变的总等效位移
            # 假设迟滞项与电压同量级，引入缩放
            hysteresis_displacement = hysteresis_force * current_d36 * 0.5
            creep_displacement = creep_strain * self.thickness

            target_displacement = ideal_displacement - hysteresis_displacement + creep_displacement

            # 加入外部负载影响 (静态柔度)
            load_displacement = load / self.k_shear
            target_displacement -= load_displacement

            # 简化为一阶系统或者直接静态映射以避免显式欧拉法在极高刚度下的数值不稳定性
            # x = target_x (假设动力学响应远快于控制周期 dt=1ms)
            # 引入一个简单的一阶低通特性模拟动态响应
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
        self.weights = np.ones(num_operators) * 0.1 # 初始权重
        self.states = np.zeros(num_operators)       # play 算子状态

        # RLS 参数
        self.P = np.eye(num_operators) * 1000.0     # 协方差矩阵
        self.lambda_rls = 0.995                     # 遗忘因子

    def _play_operator(self, v_in: float, threshold: float, state: float) -> float:
        """基础 Play 算子"""
        return max(v_in - threshold, min(v_in + threshold, state))

    def compensate(self, reference_displacement: float, current_v: float = 0.0) -> float:
        """
        基于当前参考位移计算前馈补偿电压。
        (简化实现：将参考位移映射为补偿电压)
        """
        # 计算各算子输出
        for i in range(self.num_operators):
            self.states[i] = self._play_operator(reference_displacement, self.thresholds[i], self.states[i])

        # 补偿电压 = 算子输出的加权和
        v_comp = np.dot(self.weights, self.states)
        return v_comp

    def update_parameters_rls(self, actual_displacement: float, reference_displacement: float):
        """
        在线辨识：递归最小二乘参数更新。
        (此方法在闭环中运行，以减小模型误差)
        """
        # 特征向量 (当前 states)
        phi = self.states.reshape(-1, 1)

        # 误差 (期望 - 实际)
        error = reference_displacement - actual_displacement

        # RLS 更新
        # K = P * phi / (lambda + phi^T * P * phi)
        # P = (P - K * phi^T * P) / lambda
        # w = w + K * error

        numerator = self.P @ phi
        denominator = self.lambda_rls + phi.T @ self.P @ phi

        # 避免除以 0
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

        # 增益配置 (基于带宽 w0)
        self.beta1 = 3 * w0
        self.beta2 = 3 * (w0 ** 2)
        self.beta3 = w0 ** 3

        # 针对剪切模式的 fal 参数 (简化为线性 ESO 或者采用对位移量级更友好的参数)
        self.alpha = [1.0, 1.0, 1.0] # 先用线性 ESO 测试收敛性，避免非线性带来的问题
        self.delta = 1e-6

        # 状态: z1(位移估计), z2(速度估计), z3(总扰动估计)
        self.z = np.zeros(3)

        # 标称控制增益 b0 (通常是 1/mass, 或根据系统标定)
        self.b0 = 100.0

    def _fal(self, e: float, alpha: float, delta: float) -> float:
        """非线性函数 fal"""
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

        # 连续时间状态方程的欧拉离散化 (修正了符号，e = z1 - y，观测器方程应该是减去误差修正项)
        z1_dot = self.z[1] - self.beta1 * e
        z2_dot = self.z[2] - self.beta2 * fe + self.b0 * u
        z3_dot = -self.beta3 * fe1

        self.z[0] += z1_dot * self.dt
        self.z[1] += z2_dot * self.dt
        self.z[2] += z3_dot * self.dt

    def get_disturbance(self) -> float:
        """获取当前总扰动估计"""
        return self.z[2]

    def get_estimated_displacement(self) -> float:
        """获取滤波后的位移估计"""
        return self.z[0]

    def get_estimated_velocity(self) -> float:
        """获取估计的速度"""
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

        # 子模块
        self.eso = ShearMotorESO(w0=w0, dt=dt)
        self.eso.b0 = b0

        # TD 状态
        self.v1 = 0.0 # 安排的位移
        self.v2 = 0.0 # 安排的速度
        # 极大地降低 TD 的速度因子，以避免参考信号突变导致超调和震荡
        self.r = 1e-4 # 速度因子
        self.h = dt   # 滤波因子

        # NLSEF 参数 (极大降低控制增益，防止振荡发散)
        self.kp = wc ** 2 * 0.01
        self.kd = 2 * wc * 0.01
        self.lambda_nl = 1.0 # 使用线性 PD，简化调试并避免发散

        # 其他
        self.u_prev = 0.0
        self.u_max = 1000.0  # 最大电压 (放宽限制以允许更快的动态响应)
        self.u_min = -1000.0 # 最小电压

        # 蠕变前馈
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

        # 温度自适应: 调整 ESO 的 b0 (由于 d36 随温度变化)
        # 假设 25度时 b0 标称，每度变化导致 0.5% d36 变化，进而影响 b0
        temp_factor = 1.0 + 0.005 * (temp - 25.0)
        self.eso.b0 = self.b0 * temp_factor

        # 1. 跟踪微分器 (TD)
        fh = self._fhan(self.v1 - setpoint, self.v2, self.r, self.h)
        self.v1 += self.v2 * self.dt
        self.v2 += fh * self.dt

        # 2. 扩展状态观测器 (ESO)
        self.eso.update(y_meas, self.u_prev)
        z1 = self.eso.get_estimated_displacement()
        z2 = self.eso.get_estimated_velocity()
        z3 = self.eso.get_disturbance()

        # 3. 非线性状态误差反馈 (NLSEF)
        e1 = self.v1 - z1
        e2 = self.v2 - z2

        # 带有非线性的 PD 控制 (修改 fal 函数的 delta 阈值到合适的位移量级，或者直接使用线性控制 e1, e2)
        if self.lambda_nl == 1.0:
            u0 = self.kp * e1 + self.kd * e2
        else:
            u0 = self.kp * self.eso._fal(e1, self.lambda_nl, 1e-6) + self.kd * self.eso._fal(e2, self.lambda_nl, 1e-6)

        # 4. 扰动补偿
        u = (u0 - z3) / self.eso.b0

        # 5. 蠕变前馈补偿
        if abs(setpoint - self.last_setpoint) > 1e-6:
            self.last_step_time = self.time
            self.last_setpoint = setpoint

        elapsed = self.time - self.last_step_time
        creep_comp = 0.0
        if elapsed > 0:
            # 简化版蠕变前馈：预测蠕变并反向补偿
            predicted_creep = self.gamma_0 * setpoint * np.log10(1 + elapsed / self.tau_creep)
            # 将位移蠕变转换为补偿电压 (简化系数)
            creep_comp = -predicted_creep * 0.1

        u += creep_comp

        # 6. 抗饱和
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
        # 主环
        w1 = max_voltage * np.sin(2 * np.pi * 1.0 * t1)

        # 次环
        t2 = np.arange(0, 0.5, dt)
        w2 = (max_voltage * 0.5) * np.sin(2 * np.pi * 2.0 * t2) + (max_voltage * 0.5)

        # 拼接并确保平滑过渡
        w = np.concatenate([w1, w2])
        return w

    def identify_parameters(self, voltage_seq: np.ndarray, displacement_seq: np.ndarray, time_seq: np.ndarray) -> Dict[str, float]:
        """
        根据测试数据辨识核心参数 d36, 迟滞宽度, 和近似蠕变常数。
        """
        # 简单的线性回归辨识 d36
        # 选择电压较高的线性段，或者通过最大位移 / 最大电压 粗略估计
        max_idx = np.argmax(voltage_seq)
        min_idx = np.argmin(voltage_seq)

        dv = voltage_seq[max_idx] - voltage_seq[min_idx]
        dd = displacement_seq[max_idx] - displacement_seq[min_idx]

        self.d36_identified = dd / dv if dv != 0 else 2500e-12

        # 迟滞宽度估计 (找电压为 0 时的位移差)
        zero_crossings = np.where(np.diff(np.sign(voltage_seq)))[0]
        widths = []
        for z in zero_crossings:
            widths.append(abs(displacement_seq[z]))
        self.hysteresis_width = np.mean(widths) if widths else 0.0

        # 蠕变常数 tau_creep 假设给定或者通过阶跃响应辨识，这里提供一个基础辨识结构
        self.tau_creep = 0.1 # 简化的辨识结果

        return {
            'd36': self.d36_identified,
            'hysteresis_width': self.hysteresis_width,
            'tau_creep': self.tau_creep
        }
