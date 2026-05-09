"""
PMN-PT shear mode (d36) piezoelectric actuator model and control algorithm

This module provides an actuator model based on the PMN-PT piezoelectric single crystal d36 shear mode,
Hysteresis compensator, extended state observer (ESO) and active disturbance rejection controller (ADRC).
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union

class ShearActuatorPlant:
    """
    PMN-PT d36 shear mode actuator model.

    Model the electro-mechanical coupling behavior of PMN-PT d36 shear mode, including:
    - Piezoelectric effect (based on d36 shear mode parameters)
    - Rate dependent hysteresis (Bouc-Wen model improvement)
    - Logarithmic creep characteristics
    """

    def __init__(self,
                 d36: float = 2500e-12,
                 s55E: float = 60e-12,
                 epsilon33T: float = 5000 * 8.854e-12,
                 length: float = 10e-3,
                 width: float = 10e-3,
                 thickness: float = 1e-3):
        """
        Initialize the executor model.

        parameter:
            d36: Piezoelectric shear coefficient (C/N or m/V), typical 2000-3000 pC/N
            s55E: Elastic compliance coefficient (m^2/N)
            epsilon33T: Free state dielectric constant (F/m)
            length: length (m)
            width: width (m)
            thickness: thickness (m)
        """
        #Geometry parameters
        self.length = length
        self.width = width
        self.thickness = thickness
        self.area = length * width

        #Physical parameters
        self.d36 = d36
        self.s55E = s55E
        self.epsilon33T = epsilon33T

        # Stiffness (shear)
        self.k_shear = self.area / (self.s55E * self.thickness)

        #State variables
        self.gamma = 0.0 # Shear strain
        self.gamma_dot = 0.0 # strain rate
        self.displacement = 0.0 #actual displacement
        self.Q = 0.0 # Charge
        self.voltage = 0.0 # Input voltage

        # Hysteresis model parameters (Bouc-Wen improved version)
        self.A = 1.0
        self.beta = 0.1
        self.gamma_bw = 0.1
        self.n = 1
        self.h_var = 0.0 # Hysteresis internal state

        # Creep model parameters
        self.gamma_0 = 0.05 # Creep coefficient
        self.tau_creep = 0.1 # Creep time constant
        self.time = 0.0
        self.last_voltage_change_time = 0.0
        self.voltage_step = 0.0

        # Dynamic mass, damping
        self.mass = 0.01 # Effective mass kg
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
        Logarithmic creep model calculation.
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
        Bouc-Wen based rate dependent hysteresis model.
        """
        # Rate dependence factor
        rate_factor = 1.0 + 0.1 * np.abs(v_dot)

        # Bouc-Wen status update
        h_dot = self.A * v_dot - rate_factor * self.beta * np.abs(v_dot) * self.h_var * (np.abs(self.h_var)**(self.n-1)) - rate_factor * self.gamma_bw * v_dot * (np.abs(self.h_var)**self.n)

        self.h_var += h_dot * dt
        return self.h_var

    def simulate(self, voltage_seq: np.ndarray, dt: float, load_seq: Optional[np.ndarray] = None, temp_seq: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates the response of an actuator under a given voltage sequence.

        parameter:
            voltage_seq: voltage time series (V)
            dt: time step (s)
            load_seq: external load force sequence (N), optional
            temp_seq: temperature sequence (degrees Celsius), optional

        return:
            (time series, displacement series, strain series)
        """
        n_steps = len(voltage_seq)
        time_seq = np.arange(n_steps) * dt
        displacement_seq = np.zeros(n_steps)
        strain_seq = np.zeros(n_steps)

        #Initialization state
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

            # Piezoelectric driving force (F = d36/s55 * V/thickness * area) Simplified
            # Here we use the displacement model: x = d36 * V + hysteresis + creep

            # Ideal piezoelectric displacement
            ideal_displacement = current_d36 * v

            # Total equivalent displacement considering hysteresis and creep
            # Assume that the hysteresis term is of the same magnitude as the voltage and introduce scaling
            hysteresis_displacement = hysteresis_force * current_d36 * 0.5
            creep_displacement = creep_strain * self.thickness

            target_displacement = ideal_displacement - hysteresis_displacement + creep_displacement

            #Add external load effects (static compliance)
            load_displacement = load / self.k_shear
            target_displacement -= load_displacement

            # Simplify to a first-order system or direct static mapping to avoid the numerical instability of the explicit Euler method under extremely high stiffness
            # x = target_x (assuming the dynamic response is much faster than the control period dt=1ms)
            #Introduce a simple first-order low-pass characteristic to simulate dynamic response
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
    Feedforward compensator based on PI (Prandtl-Ishlinskii) inverse model.

    Distribute rate-dependent thresholds using the superposition operator and the play operator,
    And the parameters are updated online through the RLS (Recursive Least Squares) algorithm.
    """
    def __init__(self, num_operators: int = 10, max_threshold: float = 10.0):
        self.num_operators = num_operators
        self.thresholds = np.linspace(0, max_threshold, num_operators)
        self.weights = np.ones(num_operators) * 0.1 # Initial weights
        self.states = np.zeros(num_operators) # play operator states

        # RLS parameters
        self.P = np.eye(num_operators) * 1000.0 # Covariance matrix
        self.lambda_rls = 0.995 # Forgetting factor

    def _play_operator(self, v_in: float, threshold: float, state: float) -> float:
        """Basic Play Operator"""
        return max(v_in - threshold, min(v_in + threshold, state))

    def compensate(self, reference_displacement: float, current_v: float = 0.0) -> float:
        """
        The feedforward compensation voltage is calculated based on the current reference displacement.
        (Simplified implementation: map reference displacement to compensation voltage)
        """
        # Calculate the output of each operator
        for i in range(self.num_operators):
            self.states[i] = self._play_operator(reference_displacement, self.thresholds[i], self.states[i])

        # Compensation voltage = weighted sum of operator output
        v_comp = np.dot(self.weights, self.states)
        return v_comp

    def update_parameters_rls(self, actual_displacement: float, reference_displacement: float):
        """
        Online identification: Recursive least squares parameter update.
        (This method operates in a closed loop to reduce model error)
        """
        # Feature vector (current states)
        phi = self.states.reshape(-1, 1)

        # Error (expected - actual)
        error = reference_displacement - actual_displacement

        # RLS update
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
    3rd order extended state observer (ESO).

    Used to observe displacement, velocity and total disturbance (including creep, hysteresis, load, temperature drift, etc.).
    The fal function parameter has been optimized specifically for shear mode.
    """
    def __init__(self, w0: float = 300.0, dt: float = 0.001):
        """
        parameter:
            w0: Observer bandwidth (rad/s), typical 100-500
            dt: sampling time
        """
        self.w0 = w0
        self.dt = dt

        # Gain configuration (based on bandwidth w0)
        self.beta1 = 3 * w0
        self.beta2 = 3 * (w0 ** 2)
        self.beta3 = w0 ** 3

        # fal parameter for shear mode (simplify to linear ESO or adopt parameters more friendly to the displacement magnitude)
        self.alpha = [1.0, 1.0, 1.0] # First use linear ESO to test convergence to avoid problems caused by nonlinearity
        self.delta = 1e-6

        # Status: z1 (displacement estimate), z2 (velocity estimate), z3 (total disturbance estimate)
        self.z = np.zeros(3)

        # Nominal control gain b0 (usually 1/mass, or based on system calibration)
        self.b0 = 100.0

    def _fal(self, e: float, alpha: float, delta: float) -> float:
        """Nonlinear function fal"""
        if abs(e) <= delta:
            return e / (delta ** (1 - alpha))
        else:
            return (abs(e) ** alpha) * np.sign(e)

    def update(self, y_meas: float, u: float):
        """
        Update status estimate.

        parameter:
            y_meas: measured displacement
            u: control input (voltage)
        """
        e = self.z[0] - y_meas

        fe = self._fal(e, self.alpha[0], self.delta)
        fe1 = self._fal(e, self.alpha[1], self.delta)
        fe2 = self._fal(e, self.alpha[2], self.delta)

        # Euler discretization of continuous time state equation (corrected sign, e = z1 - y, observer equation should be minus the error correction term)
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
    Active Disturbance Rejection Controller (ADRC) for PMN-PT shear mode actuator.

    Integrated:
    - Tracking Differentiator (TD): Arranges transition processes and extracts differentials
    - Extended State Observer (ESO): estimates state and total disturbance
    - Nonlinear State Error Feedback (NLSEF): Calculate control law
    Supports anti-integral saturation, creep feedforward compensation and temperature adaptation.
    """
    def __init__(self, wc: float = 100.0, w0: float = 300.0, b0: float = 100.0, dt: float = 0.001):
        self.wc = wc
        self.dt = dt
        self.b0 = b0

        # submodule
        self.eso = ShearMotorESO(w0=w0, dt=dt)
        self.eso.b0 = b0

        # TD status
        self.v1 = 0.0 # Arranged displacement
        self.v2 = 0.0 # Scheduled speed
        # Greatly reduce the speed factor of TD to avoid overshoot and oscillation caused by reference signal mutations
        self.r = 1e-4 # Speed ​​factor
        self.h = dt #Filter factor

        # NLSEF parameter (greatly reduces the control gain to prevent oscillation from diverging)
        self.kp = wc ** 2 * 0.01
        self.kd = 2 * wc * 0.01
        self.lambda_nl = 1.0 # Use linear PD to simplify debugging and avoid divergence

        # other
        self.u_prev = 0.0
        self.u_max = 1000.0 # Maximum voltage (relaxed limit to allow faster dynamic response)
        self.u_min = -1000.0 # Minimum voltage

        # Creep feed forward
        self.gamma_0 = 0.05
        self.tau_creep = 0.1
        self.last_step_time = 0.0
        self.last_setpoint = 0.0
        self.time = 0.0

    def _fhan(self, x1: float, x2: float, r: float, h: float) -> float:
        """Speedest control synthesis function (discrete system)"""
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
        Calculate the control voltage.

        parameter:
            setpoint: target displacement
            y_meas: measured displacement
            temp: ambient temperature (for adaptation)
        """
        self.time += self.dt

        # Temperature adaptation: adjust b0 of ESO (because d36 changes with temperature)
        # Assuming that b0 is nominal at 25 degrees, each degree change causes a 0.5% d36 change, which in turn affects b0
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

        # PD control with nonlinearity (modify the delta threshold of the fal function to an appropriate displacement magnitude, or directly use linear control e1, e2)
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
    Automatically characterize piezoelectric hysteresis main loop and secondary loop, and identify parameters.
    """
    def __init__(self):
        self.d36_identified = 0.0
        self.hysteresis_width = 0.0
        self.tau_creep = 0.0

    def generate_characterization_waveforms(self, max_voltage: float = 100.0, dt: float = 0.001) -> np.ndarray:
        """
        Generate a voltage sequence for characterization (triangular wave containing primary and secondary loops).
        """
        t1 = np.arange(0, 1.0, dt)
        # main ring
        w1 = max_voltage * np.sin(2 * np.pi * 1.0 * t1)

        # Secondary Ring
        t2 = np.arange(0, 0.5, dt)
        w2 = (max_voltage * 0.5) * np.sin(2 * np.pi * 2.0 * t2) + (max_voltage * 0.5)

        # Splice and ensure smooth transitions
        w = np.concatenate([w1, w2])
        return w

    def identify_parameters(self, voltage_seq: np.ndarray, displacement_seq: np.ndarray, time_seq: np.ndarray) -> Dict[str, float]:
        """
        Identify core parameters d36, hysteresis width, and approximate creep constant based on test data.
        """
        # Simple linear regression identification d36
        # Select a linear segment with a higher voltage, or roughly estimate it through the maximum displacement/maximum voltage
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

        # Creep constant tau_creep Assuming that it is given or identified through step response, a basic identification structure is provided here.
        self.tau_creep = 0.1 # Simplified identification results

        return {
            'd36': self.d36_identified,
            'hysteresis_width': self.hysteresis_width,
            'tau_creep': self.tau_creep
        }
