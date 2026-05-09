import numpy as np
from typing import List, Tuple

class PiezoStepperPlant:
    """
    Dynamic model class for piezoelectric stepper motors.

    Describe the dynamics of piezoelectric stepper motors, including hysteresis nonlinearity.
    A simplified Bouc-Wen model is used here to describe the hysteresis effect.

    Dynamic equation:
    m * x''(t) + c * x'(t) + k * x(t) = d * u(t) - F_h(t) + F_ext(t)
    where F_h(t) is the hysteresis force, simplified to a variable related to displacement.
    In order to control the algorithm design here, it can be simplified to:
    x1' = x2
    x2' = -k/m * x1 - c/m * x2 + d/m * u - 1/m * F_h
    Let m=1, then:
    x2' = -k * x1 - c * x2 + d * u - F_h
    """

    def __init__(self, k: float = 100.0, c: float = 5.0, d: float = 10.0,
                 alpha: float = 0.5, beta: float = 0.1, gamma: float = 0.1):
        """
        Initialize system parameters.

        parameter:
        k: system stiffness
        c: damping coefficient
        d: Piezoelectric coefficient
        alpha, beta, gamma: parameters of the Bouc-Wen hysteresis model
        """
        self.k = k
        self.c = c
        self.d = d

        # Hysteresis model parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        #State variables
        self.x1 = 0.0 # Displacement
        self.x2 = 0.0 # speed
        self.h = 0.0 # Hysteresis state variable

    def reset(self) -> None:
        """Reset status"""
        self.x1 = 0.0
        self.x2 = 0.0
        self.h = 0.0

    def step(self, u: float, dt: float, disturbance: float = 0.0) -> float:
        """
        Simulate the single-step evolution of the system (using the fourth-order Runge-Kutta method or Euler method, here the simplified Euler method is used)

        parameter:
        u: current voltage input
        dt: simulation step size
        disturbance: total external disturbance (such as load)

        return:
        Current displacement x1 (i.e. measurement output y)
        """
        # Bouc-Wen model update differential
        h_dot = self.alpha * self.d * self.x2 - self.beta * abs(self.x2) * self.h - self.gamma * self.x2 * abs(self.h)
        self.h += h_dot * dt

        # State equation of the system
        x1_dot = self.x2
        x2_dot = -self.k * self.x1 - self.c * self.x2 + self.d * u - self.h + disturbance

        # Update status
        self.x1 += x1_dot * dt
        self.x2 += x2_dot * dt

        return self.x1

    def simulate(self, u_seq: np.ndarray, dt: float, disturbance_seq: np.ndarray = None) -> np.ndarray:
        """
        Simulate the input sequence.

        parameter:
        u_seq: voltage input sequence (N,)
        dt: simulation time step
        disturbance_seq: disturbance sequence, if None, all zeros are assumed

        return:
        Displacement sequence y_seq (N,)
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
    Nonlinear function fal in active disturbance rejection control.

    parameter:
    e: error input
    alpha: non-linear exponent, usually between 0 and 1
    delta: linear area threshold, controlling the linear range near the origin

    return:
    The output value of the fal function
    """
    if abs(e) <= delta:
        return e / (delta ** (1 - alpha))
    else:
        return (abs(e) ** alpha) * np.sign(e)


class ExtendedStateObserver:
    """
    Third-order nonlinear/linear extended state observer (ESO).

    state:
    z1: estimate of displacement
    z2: estimate of speed
    z3: Estimate of total disturbance (unmodeled dynamics + external disturbances)
    """

    def __init__(self, w0: float, b0: float, nonlinear: bool = False,
                 alpha1: float = 0.5, alpha2: float = 0.25, delta: float = 0.01):
        """
        Initialize ESO parameters.

        parameter:
        w0: Observer bandwidth
        b0: Control gain (approximate value)
        nonlinear: Whether to use nonlinear ESO (NLESO). False is Linear ESO (LESO)
        alpha1, alpha2: exponential parameters of nonlinear ESO
        delta: linear range of a nonlinear function
        """
        self.w0 = w0
        self.b0 = b0
        self.nonlinear = nonlinear

        # Observer gain configuration (usually pole-based configuration to -w0)
        self.beta1 = 3 * w0
        self.beta2 = 3 * w0**2
        self.beta3 = w0**3

        # Nonlinear parameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.delta = delta

        #State estimate
        self.z1 = 0.0
        self.z2 = 0.0
        self.z3 = 0.0

    def reset(self) -> None:
        """Reset observer status"""
        self.z1 = 0.0
        self.z2 = 0.0
        self.z3 = 0.0

    def update(self, y_meas: float, u: float, dt: float) -> None:
        """
        Update the ESO's state estimate based on the current measurement output and control input.

        parameter:
        y_meas: actual measured displacement output
        u: Control voltage input applied at the current moment
        dt: control step size
        """
        e = self.z1 - y_meas

        if self.nonlinear:
            # Nonlinear ESO
            fe1 = fal(e, self.alpha1, self.delta)
            fe2 = fal(e, self.alpha2, self.delta)

            z1_dot = self.z2 - self.beta1 * e
            z2_dot = self.z3 - self.beta2 * fe1 + self.b0 * u
            z3_dot = -self.beta3 * fe2
        else:
            # Linear ESO
            z1_dot = self.z2 - self.beta1 * e
            z2_dot = self.z3 - self.beta2 * e + self.b0 * u
            z3_dot = -self.beta3 * e

        # Euler method update status
        self.z1 += z1_dot * dt
        self.z2 += z2_dot * dt
        self.z3 += z3_dot * dt

    def get_disturbance(self) -> float:
        """
        Returns the current total disturbance estimate.

        return:
        z3: Disturbance estimation
        """
        return self.z3

class TrackingDifferentiator:
    """
    Second-order tracking differentiator (TD).
    Used to smooth the transition process and extract differential signals.
    """
    def __init__(self, r: float = 100.0, h: float = 0.001):
        """
        Initialize TD parameters.

        parameter:
        r: tracking speed factor
        h: simulation/discretization step size
        """
        self.r = r
        self.h = h
        self.v1 = 0.0 # Track signal
        self.v2 = 0.0 # Differential signal

    def reset(self):
        self.v1 = 0.0
        self.v2 = 0.0

    def update(self, v: float, dt: float) -> Tuple[float, float]:
        """
        Update and return (trace value, differential value).
        Use a simplified version or linear approximation of the steepest descent function. A linear approximation is used here for simplicity.
        """
        # Simplified linear TD
        v1_dot = self.v2
        v2_dot = -self.r**2 * (self.v1 - v) - 2 * self.r * self.v2

        self.v1 += v1_dot * dt
        self.v2 += v2_dot * dt

        return self.v1, self.v2


class ESOController:
    """
    Nonlinear state error feedback (NLSEF) and ESO-based disturbance compensation logic.
    """
    def __init__(self, wc: float, b0: float, r: float = 100.0):
        """
        Initialize controller parameters.

        parameter:
        wc: Controller bandwidth
        b0: Control gain (approximate value)
        r: TD tracking speed factor
        """
        self.wc = wc
        self.b0 = b0

        # PD controller gain configuration (based on pole configuration)
        self.kp = wc**2
        self.kd = 2 * wc

        self.td = TrackingDifferentiator(r=r)

    def reset(self):
        self.td.reset()

    def control(self, setpoint: float, z1: float, z2: float, z3: float, dt: float) -> float:
        """
        Compute control laws.

        parameter:
        setpoint: target set point
        z1, z2, z3: ESO observation status
        dt: simulation step size
        """
        # TD extraction transition process
        v1, v2 = self.td.update(setpoint, dt)

        # Error calculation
        e1 = v1 - z1
        e2 = v2 - z2

        # Linear PD control law (i.e. simplified LSEF)
        u0 = self.kp * e1 + self.kd * e2

        # Disturbance compensation
        u = (u0 - z3) / self.b0

        return u


class ADRController:
    """
    Complete active interference rejection controller package.
    """
    def __init__(self):
        self.eso = None
        self.controller = None

    def tune(self, wc: float, w0: float, b0: float, nonlinear: bool = False):
        """
        Adjust the parameters of the automatic disturbance rejection controller with one click.

        parameter:
        wc: controller bandwidth
        w0: observer bandwidth
        b0: control gain
        nonlinear: whether to use nonlinear ESO
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
        Execute single step control and return control input u.
        """
        # 1. ESO update
        self.eso.update(y_meas, u_prev, dt)

        # 2. Controller calculation
        u = self.controller.control(setpoint, self.eso.z1, self.eso.z2, self.eso.z3, dt)

        return u

    def track(self, reference_traj: np.ndarray, plant: PiezoStepperPlant, dt: float,
              disturbance_seq: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Follow a given trajectory for simulation.

        parameter:
        reference_traj: reference trajectory sequence
        plant: piezoelectric stepper motor model
        dt: simulation step size
        disturbance_seq: external disturbance sequence

        return:
        y_seq: actual measurement trajectory
        u_seq: control input sequence
        z3_seq: perturbation sequence estimated by the observer
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

            #Plantevolution
            y_meas = plant.step(u_prev, dt, disturbance)
            y_seq[i] = y_meas

            #Controller calculates new input
            u = self.control_step(setpoint, y_meas, dt, u_prev)
            u_seq[i] = u
            z3_seq[i] = self.eso.z3

            u_prev = u

        return y_seq, u_seq, z3_seq
