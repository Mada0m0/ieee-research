import numpy as np
from typing import List, Tuple

class PiezoStepperPlant:
    """
    压电步进电机的动态模型类。

    描述压电步进电机的动态过程，包括了迟滞非线性。
    这里使用简化的 Bouc-Wen 模型来描述迟滞效应。

    动态方程：
    m * x''(t) + c * x'(t) + k * x(t) = d * u(t) - F_h(t) + F_ext(t)
    其中 F_h(t) 为迟滞力，简化为与位移相关的变量。
    这里为了控制算法设计，可以简化为：
    x1' = x2
    x2' = -k/m * x1 - c/m * x2 + d/m * u - 1/m * F_h
    令 m=1，则：
    x2' = -k * x1 - c * x2 + d * u - F_h
    """

    def __init__(self, k: float = 100.0, c: float = 5.0, d: float = 10.0,
                 alpha: float = 0.5, beta: float = 0.1, gamma: float = 0.1):
        """
        初始化系统参数。

        参数:
        k: 系统刚度
        c: 阻尼系数
        d: 压电系数
        alpha, beta, gamma: Bouc-Wen迟滞模型的参数
        """
        self.k = k
        self.c = c
        self.d = d

        # 迟滞模型参数
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 状态变量
        self.x1 = 0.0  # 位移
        self.x2 = 0.0  # 速度
        self.h = 0.0   # 迟滞状态变量

    def reset(self) -> None:
        """重置状态"""
        self.x1 = 0.0
        self.x2 = 0.0
        self.h = 0.0

    def step(self, u: float, dt: float, disturbance: float = 0.0) -> float:
        """
        模拟系统单步演化（采用四阶龙格库塔法或者欧拉法，这里使用简化的欧拉法）

        参数:
        u: 当前电压输入
        dt: 仿真步长
        disturbance: 外部总扰动 (如负载)

        返回:
        当前的位移 x1 (即测量输出 y)
        """
        # Bouc-Wen 模型更新微分
        h_dot = self.alpha * self.d * self.x2 - self.beta * abs(self.x2) * self.h - self.gamma * self.x2 * abs(self.h)
        self.h += h_dot * dt

        # 系统的状态方程
        x1_dot = self.x2
        x2_dot = -self.k * self.x1 - self.c * self.x2 + self.d * u - self.h + disturbance

        # 更新状态
        self.x1 += x1_dot * dt
        self.x2 += x2_dot * dt

        return self.x1

    def simulate(self, u_seq: np.ndarray, dt: float, disturbance_seq: np.ndarray = None) -> np.ndarray:
        """
        对输入序列进行仿真。

        参数:
        u_seq: 电压输入序列 (N,)
        dt: 仿真时间步长
        disturbance_seq: 扰动序列，若为 None，则假设全零

        返回:
        位移序列 y_seq (N,)
        """
        n_steps = len(u_seq)
        y_seq = np.zeros(n_steps)
        if disturbance_seq is None:
            disturbance_seq = np.zeros(n_steps)

        for i in range(n_steps):
            y_seq[i] = self.step(u_seq[i], dt, disturbance_seq[i])

        return y_seq

def fal(e: float, alpha: float, delta: float) -> float:
    """
    自抗扰控制中的非线性函数 fal。

    参数:
    e: 误差输入
    alpha: 非线性指数，通常在 0 到 1 之间
    delta: 线性区阈值，控制原点附近的线性范围

    返回:
    fal函数的输出值
    """
    if abs(e) <= delta:
        return e / (delta ** (1 - alpha))
    else:
        return (abs(e) ** alpha) * np.sign(e)


class ExtendedStateObserver:
    """
    三阶非线性/线性扩展状态观测器 (ESO)。

    状态：
    z1: 位移的估计
    z2: 速度的估计
    z3: 总扰动的估计 (未建模动态 + 外部扰动)
    """

    def __init__(self, w0: float, b0: float, nonlinear: bool = False,
                 alpha1: float = 0.5, alpha2: float = 0.25, delta: float = 0.01):
        """
        初始化ESO参数。

        参数:
        w0: 观测器带宽 (Observer bandwidth)
        b0: 控制增益 (近似值)
        nonlinear: 是否使用非线性ESO (NLESO)。为 False 则为线性ESO (LESO)
        alpha1, alpha2: 非线性ESO的指数参数
        delta: 非线性函数的线性区区间
        """
        self.w0 = w0
        self.b0 = b0
        self.nonlinear = nonlinear

        # 观测器增益配置 (通常基于极点配置到 -w0)
        self.beta1 = 3 * w0
        self.beta2 = 3 * w0**2
        self.beta3 = w0**3

        # 非线性参数
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.delta = delta

        # 状态估计值
        self.z1 = 0.0
        self.z2 = 0.0
        self.z3 = 0.0

    def reset(self) -> None:
        """重置观测器状态"""
        self.z1 = 0.0
        self.z2 = 0.0
        self.z3 = 0.0

    def update(self, y_meas: float, u: float, dt: float) -> None:
        """
        根据当前测量输出和控制输入，更新ESO的状态估计。

        参数:
        y_meas: 实际测量到的位移输出
        u: 当前时刻施加的控制电压输入
        dt: 控制步长
        """
        e = self.z1 - y_meas

        if self.nonlinear:
            # 非线性ESO
            fe1 = fal(e, self.alpha1, self.delta)
            fe2 = fal(e, self.alpha2, self.delta)

            z1_dot = self.z2 - self.beta1 * e
            z2_dot = self.z3 - self.beta2 * fe1 + self.b0 * u
            z3_dot = -self.beta3 * fe2
        else:
            # 线性ESO
            z1_dot = self.z2 - self.beta1 * e
            z2_dot = self.z3 - self.beta2 * e + self.b0 * u
            z3_dot = -self.beta3 * e

        # 欧拉法更新状态
        self.z1 += z1_dot * dt
        self.z2 += z2_dot * dt
        self.z3 += z3_dot * dt

    def get_disturbance(self) -> float:
        """
        返回当前的总扰动估计值。

        返回:
        z3: 扰动估计
        """
        return self.z3

class TrackingDifferentiator:
    """
    二阶跟踪微分器 (TD)。
    用于平滑过渡过程并提取微分信号。
    """
    def __init__(self, r: float = 100.0, h: float = 0.001):
        """
        初始化TD参数。

        参数:
        r: 跟踪速度因子
        h: 仿真/离散化步长
        """
        self.r = r
        self.h = h
        self.v1 = 0.0  # 跟踪信号
        self.v2 = 0.0  # 微分信号

    def reset(self):
        self.v1 = 0.0
        self.v2 = 0.0

    def update(self, v: float, dt: float) -> Tuple[float, float]:
        """
        更新并返回(跟踪值, 微分值)。
        使用最速下降函数的简化版本或线性逼近。这里使用线性逼近以简化。
        """
        # 简化的线性TD
        v1_dot = self.v2
        v2_dot = -self.r**2 * (self.v1 - v) - 2 * self.r * self.v2

        self.v1 += v1_dot * dt
        self.v2 += v2_dot * dt

        return self.v1, self.v2


class ESOController:
    """
    非线性状态误差反馈 (NLSEF) 及 基于ESO的扰动补偿逻辑。
    """
    def __init__(self, wc: float, b0: float, r: float = 100.0):
        """
        初始化控制器参数。

        参数:
        wc: 控制器带宽 (Controller bandwidth)
        b0: 控制增益 (近似值)
        r: TD跟踪速度因子
        """
        self.wc = wc
        self.b0 = b0

        # PD 控制器增益配置 (基于极点配置)
        self.kp = wc**2
        self.kd = 2 * wc

        self.td = TrackingDifferentiator(r=r)

    def reset(self):
        self.td.reset()

    def control(self, setpoint: float, z1: float, z2: float, z3: float, dt: float) -> float:
        """
        计算控制律。

        参数:
        setpoint: 目标设定点
        z1, z2, z3: ESO的观测状态
        dt: 仿真步长
        """
        # TD 提取过渡过程
        v1, v2 = self.td.update(setpoint, dt)

        # 误差计算
        e1 = v1 - z1
        e2 = v2 - z2

        # 线性 PD 控制律 (即简化的 LSEF)
        u0 = self.kp * e1 + self.kd * e2

        # 扰动补偿
        u = (u0 - z3) / self.b0

        return u


class ADRController:
    """
    完整的自抗扰控制器封装。
    """
    def __init__(self):
        self.eso = None
        self.controller = None

    def tune(self, wc: float, w0: float, b0: float, nonlinear: bool = False):
        """
        一键整定自抗扰控制器参数。

        参数:
        wc: 控制器带宽
        w0: 观测器带宽
        b0: 控制增益
        nonlinear: 是否使用非线性ESO
        """
        self.eso = ExtendedStateObserver(w0=w0, b0=b0, nonlinear=nonlinear)
        self.controller = ESOController(wc=wc, b0=b0)

    def reset(self):
        if self.eso is not None:
            self.eso.reset()
        if self.controller is not None:
            self.controller.reset()

    def control_step(self, setpoint: float, y_meas: float, dt: float, u_prev: float) -> float:
        """
        执行单步控制并返回控制输入 u。
        """
        # 1. ESO 更新
        self.eso.update(y_meas, u_prev, dt)

        # 2. 控制器计算
        u = self.controller.control(setpoint, self.eso.z1, self.eso.z2, self.eso.z3, dt)

        return u

    def track(self, reference_traj: np.ndarray, plant: PiezoStepperPlant, dt: float,
              disturbance_seq: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        跟踪给定轨迹进行仿真。

        参数:
        reference_traj: 参考轨迹序列
        plant: 压电步进电机模型
        dt: 仿真步长
        disturbance_seq: 外部扰动序列

        返回:
        y_seq: 实际测量轨迹
        u_seq: 控制输入序列
        z3_seq: 观测器估计的扰动序列
        """
        n_steps = len(reference_traj)
        y_seq = np.zeros(n_steps)
        u_seq = np.zeros(n_steps)
        z3_seq = np.zeros(n_steps)

        if disturbance_seq is None:
            disturbance_seq = np.zeros(n_steps)

        self.reset()
        plant.reset()

        u_prev = 0.0
        for i in range(n_steps):
            setpoint = reference_traj[i]
            disturbance = disturbance_seq[i]

            # Plant 演化
            y_meas = plant.step(u_prev, dt, disturbance)
            y_seq[i] = y_meas

            # 控制器计算新的输入
            u = self.control_step(setpoint, y_meas, dt, u_prev)
            u_seq[i] = u
            z3_seq[i] = self.eso.z3

            u_prev = u

        return y_seq, u_seq, z3_seq
