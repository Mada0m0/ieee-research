import numpy as np
from typing import Dict, Tuple, List, Optional


class WalkingPiezoMotorPlant:
    """
    行走式压电电机模型 (Walking Piezoelectric Motor Plant)
    利用四组压电叠堆产生宏观位移：左夹持 (left_clamp)、右夹持 (right_clamp)、左驱动 (left_drive)、右驱动 (right_drive)。
    """

    def __init__(
        self,
        stiffness: Dict[str, float] = None,
        damping: Dict[str, float] = None,
        piezo_coeff: Dict[str, float] = None,
        clamp_preload: float = 100.0,
        mass: float = 0.5,
    ):
        """
        初始化电机模型。

        参数:
        stiffness: 各叠堆刚度 k_i (N/m)
        damping: 各叠堆阻尼 c_i (N*s/m)
        piezo_coeff: 各叠堆压电系数 d_i (m/V)
        clamp_preload: 夹持预压力 (N)
        mass: 动子质量 (kg)
        """
        default_k = {'left_clamp': 1e7, 'right_clamp': 1e7, 'left_drive': 5e6, 'right_drive': 5e6}
        default_c = {'left_clamp': 1e3, 'right_clamp': 1e3, 'left_drive': 5e2, 'right_drive': 5e2}
        default_d = {'left_clamp': 1e-8, 'right_clamp': 1e-8, 'left_drive': 2e-8, 'right_drive': 2e-8}

        self.k = stiffness if stiffness else default_k
        self.c = damping if damping else default_c
        self.d = piezo_coeff if piezo_coeff else default_d

        self.clamp_preload = clamp_preload
        self.mass = mass

        # 状态变量
        self.position = 0.0  # 动子位移 x (m)
        self.velocity = 0.0  # 动子速度 v (m/s)

        # 各叠堆历史最大应变，用于简化的Bouc-Wen迟滞模型
        self._hys_state = {key: 0.0 for key in self.d.keys()}

    def _calculate_strain(self, key: str, voltage: float) -> float:
        """
        计算单个叠堆的应变，包含简化的迟滞效应。
        """
        ideal_strain = self.d[key] * voltage

        # 简化迟滞模型
        alpha = 0.8 # 线性比例
        hys_param = 0.2

        # 迟滞状态更新
        self._hys_state[key] += 0.1 * (ideal_strain - self._hys_state[key])

        actual_strain = alpha * ideal_strain + hys_param * self._hys_state[key]
        return actual_strain

    def gait_cycle(self, voltages: Dict[str, float], dt: float) -> float:
        """
        执行一个步态周期的微小时间步的模拟，返回这段时间的位移。

        参数:
        voltages: 包含 'left_clamp', 'right_clamp', 'left_drive', 'right_drive' 电压的字典 (V)
        dt: 时间步长 (s)

        返回:
        float: 该时间步内的位移变化 (m)
        """
        strains = {k: self._calculate_strain(k, voltages.get(k, 0.0)) for k in self.d.keys()}

        # 判断夹持状态
        left_clamped = strains['left_clamp'] * self.k['left_clamp'] > self.clamp_preload / 2
        right_clamped = strains['right_clamp'] * self.k['right_clamp'] > self.clamp_preload / 2

        force = 0.0
        # 如果左侧夹持，驱动力由左驱动器提供
        if left_clamped and not right_clamped:
            force = self.k['left_drive'] * strains['left_drive']
        # 如果右侧夹持，驱动力由右驱动器提供
        elif right_clamped and not left_clamped:
            # 假设右侧是向另一方向推动，或者相同方向，这里假设为相反相推的典型设计
            force = self.k['right_drive'] * strains['right_drive']
        # 全夹持或者全松开的情况
        elif left_clamped and right_clamped:
            force = 0.0 # 锁死
            self.velocity = 0.0
        else:
            force = 0.0 # 自由打滑，忽略重力
            self.velocity *= 0.9 # 摩擦衰减

        # 运动学方程: m*a + c*v = F
        # 为了简化，采用简单的等效阻尼
        eq_damping = 1e4
        acceleration = (force - eq_damping * self.velocity) / self.mass

        self.velocity += acceleration * dt
        dx = self.velocity * dt
        self.position += dx

        return dx

    def simulate(self, voltage_sequence: List[Dict[str, float]], dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟给定的电压序列。

        参数:
        voltage_sequence: 电压序列列表
        dt: 时间步长

        返回:
        times: 时间数组
        positions: 位移数组
        """
        n_steps = len(voltage_sequence)
        times = np.arange(n_steps) * dt
        positions = np.zeros(n_steps)

        # 重置状态
        self.position = 0.0
        self.velocity = 0.0
        for k in self._hys_state:
            self._hys_state[k] = 0.0

        for i in range(n_steps):
            self.gait_cycle(voltage_sequence[i], dt)
            positions[i] = self.position

        return times, positions


class WalkingGaitController:
    """
    步态生成控制器 (Walking Gait Controller)
    负责生成四相步态: Clamp_L -> Drive_L -> Clamp_R -> Drive_R。
    """

    def __init__(self, dt: float = 1e-4):
        """
        初始化控制器

        参数:
        dt: 控制周期 (s)
        """
        self.dt = dt

    def generate_gait(
        self,
        freq: float,
        amplitude: float,
        duration: float,
        phase_delay: float = 0.25,
        asymmetric: bool = False
    ) -> List[Dict[str, float]]:
        """
        生成步态电压序列。

        参数:
        freq: 步态频率 (Hz)
        amplitude: 驱动电压幅值 (V)
        duration: 总持续时间 (s)
        phase_delay: 相位延迟比例 (通常为 0.25, 即 90度)
        asymmetric: 是否为非对称步态模式

        返回:
        包含每步电压字典的列表
        """
        n_steps = int(duration / self.dt)
        times = np.arange(n_steps) * self.dt

        sequence = []
        period = 1.0 / freq if freq > 0 else 1.0

        for t in times:
            # 归一化相位 [0, 1)
            phase = (t % period) / period

            # Clamp_L (左夹持) 阶段：0.0 ~ 0.5
            clamp_l_vol = amplitude if phase < 0.5 or phase > 0.9 else 0.0

            # Drive_L (左驱动) 阶段：在Clamp_L阶段伸长
            drive_l_vol = 0.0
            if 0.1 < phase < 0.4:
                drive_l_vol = amplitude * np.sin((phase - 0.1) / 0.3 * np.pi / 2)
            elif phase >= 0.4:
                drive_l_vol = amplitude # 保持伸长直到夹持解除
            if phase > 0.9 or phase < 0.1:
                drive_l_vol = 0.0 # 归零

            # Clamp_R (右夹持) 阶段：0.4 ~ 0.9 (有重叠区)
            clamp_r_vol = amplitude if 0.4 < phase < 0.9 else 0.0

            # Drive_R (右驱动) 阶段
            drive_r_vol = 0.0
            if asymmetric:
                # 示例非对称步态
                if 0.5 < phase < 0.8:
                    drive_r_vol = amplitude * 0.8 * np.sin((phase - 0.5) / 0.3 * np.pi / 2)
            else:
                if 0.5 < phase < 0.8:
                    drive_r_vol = amplitude * np.sin((phase - 0.5) / 0.3 * np.pi / 2)

            # 简单的微步进模式：可以用更连续的波形替代方波
            voltages = {
                'left_clamp': clamp_l_vol,
                'right_clamp': clamp_r_vol,
                'left_drive': drive_l_vol,
                'right_drive': drive_r_vol
            }
            sequence.append(voltages)

        return sequence

    def generate_microstep_gait(
        self,
        freq: float,
        amplitude: float,
        duration: float,
        resolution: int = 10
    ) -> List[Dict[str, float]]:
        """
        微步进模式，产生平滑的正弦过渡波形，以实现亚步距分辨率。
        """
        n_steps = int(duration / self.dt)
        times = np.arange(n_steps) * self.dt
        sequence = []

        for t in times:
            # 使用正弦和余弦波形实现平滑过渡
            omega = 2 * np.pi * freq

            # 夹持器：偏置正弦波确保交替夹持
            clamp_l = amplitude * (0.5 * np.sin(omega * t) + 0.5)
            clamp_r = amplitude * (0.5 * np.sin(omega * t + np.pi) + 0.5)

            # 驱动器：相差90度的正弦波
            drive_l = amplitude * (0.5 * np.sin(omega * t - np.pi/2) + 0.5)
            drive_r = amplitude * (0.5 * np.sin(omega * t + np.pi/2) + 0.5)

            # 增强夹持力，模拟真实的夹持信号
            clamp_l = amplitude if clamp_l > amplitude/2 else 0.0
            clamp_r = amplitude if clamp_r > amplitude/2 else 0.0

            sequence.append({
                'left_clamp': clamp_l,
                'right_clamp': clamp_r,
                'left_drive': drive_l,
                'right_drive': drive_r
            })

        return sequence

def fal(e: float, alpha: float, delta: float) -> float:
    """
    非线性函数 fal (用于ESO和NLSEF)
    """
    if abs(e) <= delta:
        return e / (delta ** (1 - alpha))
    else:
        return (abs(e) ** alpha) * np.sign(e)


class WalkingMotorESO:
    """
    4阶扩展状态观测器 (Extended State Observer)
    用于估计位移(z1), 速度(z2), 加速度/模型不确定性(z3), 外部负载扰动(z4)
    """

    def __init__(self, w0: float = 100.0, dt: float = 1e-4, b0: float = 1.0):
        """
        初始化ESO。

        参数:
        w0: 观测器带宽
        dt: 采样时间
        b0: 系统控制增益
        """
        self.w0 = w0
        self.dt = dt
        self.b0 = b0

        # 极点配置增益 (采用带宽参数化)
        self.beta1 = 4 * w0
        self.beta2 = 6 * (w0 ** 2)
        self.beta3 = 4 * (w0 ** 3)
        self.beta4 = w0 ** 4

        # 状态估计值: z = [z1, z2, z3, z4] (位移，速度，系统扰动，外部扰动)
        self.z = np.zeros(4)

    def update(self, y_meas: float, u: float, phase: str = 'drive') -> np.ndarray:
        """
        更新观测器状态。

        参数:
        y_meas: 测量的位移
        u: 当前控制量 (驱动电压)
        phase: 当前步态相 ('clamp', 'drive')，用于自适应调整增益

        返回:
        np.ndarray: 更新后的状态估计 [z1, z2, z3, z4]
        """
        e = self.z[0] - y_meas

        # 如果是夹持阶段，系统的动态会发生变化，降低对扰动的敏感度
        k = 0.1 if phase == 'clamp' else 1.0

        # 4阶连续系统离散化 (欧拉法)
        # z1_dot = z2 - beta1 * e
        # z2_dot = z3 - beta2 * fal(e, 0.5, delta) + b0 * u
        # z3_dot = z4 - beta3 * fal(e, 0.25, delta)
        # z4_dot = - beta4 * fal(e, 0.125, delta)

        delta = 0.01
        fe = fal(e, 0.5, delta)
        fe1 = fal(e, 0.25, delta)
        fe2 = fal(e, 0.125, delta)

        dz1 = self.z[1] - self.beta1 * e
        dz2 = self.z[2] - k * self.beta2 * fe + self.b0 * u
        dz3 = self.z[3] - k * self.beta3 * fe1
        dz4 = - k * self.beta4 * fe2

        self.z[0] += dz1 * self.dt
        self.z[1] += dz2 * self.dt
        self.z[2] += dz3 * self.dt
        self.z[3] += dz4 * self.dt

        return self.z

    def get_estimated_states(self) -> np.ndarray:
        """获取当前估计的全状态"""
        return self.z.copy()


class WalkingMotorADRC:
    """
    自抗扰控制器 (Active Disturbance Rejection Controller)
    集成步态规划器、ESO和非线性状态误差反馈 (NLSEF)。
    """

    def __init__(self, wc: float = 10.0, w0: float = 100.0, b0: float = 1.0, dt: float = 1e-4):
        """
        初始化ADRC。

        参数:
        wc: 控制带宽
        w0: 观测器带宽
        b0: 控制增益
        dt: 控制周期
        """
        self.dt = dt
        self.b0 = b0
        self.eso = WalkingMotorESO(w0=w0, dt=dt, b0=b0)
        self.gait_controller = WalkingGaitController(dt=dt)

        # PD 控制参数配置
        self.kp = wc ** 2
        self.kd = 2 * wc

        self.target_position = 0.0

    def disturbance_compensation(self, u0: float, z3: float, z4: float) -> float:
        """
        扰动补偿。

        参数:
        u0: PD控制器输出
        z3: 估计的内部扰动
        z4: 估计的外部扰动

        返回:
        float: 补偿后的控制量 u
        """
        u = (u0 - z3 - z4) / self.b0

        # 限制输出电压范围
        u_max = 150.0
        u_min = -150.0
        return np.clip(u, u_min, u_max)

    def track(self, target_position: float, y_meas: float, current_u: float, phase: str = 'drive') -> float:
        """
        执行单步跟踪控制。

        参数:
        target_position: 目标位移
        y_meas: 测量的实际位移
        current_u: 当前实际施加的控制量
        phase: 步态相位

        返回:
        float: 下一步的控制量 (等效驱动电压幅度)
        """
        self.target_position = target_position

        # 1. 更新ESO获取状态估计
        z = self.eso.update(y_meas, current_u, phase)
        z1, z2, z3, z4 = z[0], z[1], z[2], z[3]

        # 2. 状态误差反馈控制 (PD律)
        e1 = target_position - z1
        e2 = 0.0 - z2  # 目标速度为0

        u0 = self.kp * e1 + self.kd * e2

        # 3. 扰动补偿
        u = self.disturbance_compensation(u0, z3, z4)
        return u


class WalkingMotorOptimizer:
    """
    步态参数优化器 (Walking Motor Optimizer)
    可选模块：用于优化步态的频率和幅值以实现步距一致性和速度平滑性。
    """

    def __init__(self, plant: WalkingPiezoMotorPlant, controller: WalkingGaitController):
        self.plant = plant
        self.controller = controller

    def optimize_gait(self, target_speed: float, n_iter: int = 20) -> Tuple[float, float]:
        """
        使用简单的网格搜索或随机搜索寻找最优的(freq, amplitude)组合。
        目标: 最小化速度误差。

        参数:
        target_speed: 目标速度 (m/s)
        n_iter: 迭代次数

        返回:
        Tuple[float, float]: 最优的 (freq, amplitude)
        """
        best_freq = 100.0
        best_amp = 100.0
        min_error = float('inf')

        # 简单的随机搜索作为演示
        np.random.seed(42)
        freqs = np.random.uniform(50.0, 500.0, n_iter)
        amps = np.random.uniform(50.0, 150.0, n_iter)

        dt = self.controller.dt
        duration = 0.05 # 模拟 50ms

        for f, a in zip(freqs, amps):
            # 生成测试步态
            seq = self.controller.generate_gait(freq=f, amplitude=a, duration=duration)

            # 模拟
            times, positions = self.plant.simulate(seq, dt)

            # 计算平均速度
            avg_speed = positions[-1] / duration

            error = abs(avg_speed - target_speed)
            if error < min_error:
                min_error = error
                best_freq = f
                best_amp = a

        return best_freq, best_amp
