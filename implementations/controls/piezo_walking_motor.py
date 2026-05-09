import numpy as np
from typing import Dict, Tuple, List, Optional


class WalkingPiezoMotorPlant:
    """
    Walking Piezoelectric Motor Plant
    Four groups of piezoelectric stacks are used to generate macroscopic displacements: left clamp (left_clamp), right clamp (right_clamp), left drive (left_drive), and right drive (right_drive).
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
        Initialize the motor model.

        parameter:
        stiffness: stiffness of each stack k_i (N/m)
        damping: damping of each stack c_i (N*s/m)
        piezo_coeff: piezoelectric coefficient d_i (m/V) of each stack
        clamp_preload: clamping preload (N)
        mass: mover mass (kg)
        """
        default_k = {'left_clamp': 1e7, 'right_clamp': 1e7, 'left_drive': 5e6, 'right_drive': 5e6}
        default_c = {'left_clamp': 1e3, 'right_clamp': 1e3, 'left_drive': 5e2, 'right_drive': 5e2}
        default_d = {'left_clamp': 1e-8, 'right_clamp': 1e-8, 'left_drive': 2e-8, 'right_drive': 2e-8}

        self.k = stiffness if stiffness else default_k
        self.c = damping if damping else default_c
        self.d = piezo_coeff if piezo_coeff else default_d

        self.clamp_preload = clamp_preload
        self.mass = mass

        #State variables
        self.position = 0.0 #Motor displacement x (m)
        self.velocity = 0.0 # Mover speed v (m/s)

        #Historical maximum strain for each stack, used in simplified Bouc-Wen hysteresis model
        self._hys_state = {key: 0.0 for key in self.d.keys()}

    def _calculate_strain(self, key: str, voltage: float) -> float:
        """
        Calculate the strain of a single stack, including simplified hysteresis effects.
        """
        ideal_strain = self.d[key] * voltage

        # Simplify hysteresis model
        alpha = 0.8 # linear scale
        hys_param = 0.2

        # Lazy status update
        self._hys_state[key] += 0.1 * (ideal_strain - self._hys_state[key])

        actual_strain = alpha * ideal_strain + hys_param * self._hys_state[key]
        return actual_strain

    def gait_cycle(self, voltages: Dict[str, float], dt: float) -> float:
        """
        Performs a simulation of a tiny time step of the gait cycle and returns the displacement during this time.

        parameter:
        voltages: dictionary containing 'left_clamp', 'right_clamp', 'left_drive', 'right_drive' voltages (V)
        dt: time step (s)

        return:
        float: Displacement change within this time step (m)
        """
        strains = {k: self._calculate_strain(k, voltages.get(k, 0.0)) for k in self.d.keys()}

        # Determine the clamping status
        left_clamped = strains['left_clamp'] * self.k['left_clamp'] > self.clamp_preload / 2
        right_clamped = strains['right_clamp'] * self.k['right_clamp'] > self.clamp_preload / 2

        force = 0.0
        # If clamped on the left side, the driving force is provided by the left driver
        if left_clamped and not right_clamped:
            force = self.k['left_drive'] * strains['left_drive']
        # If clamped on the right side, the driving force is provided by the right driver
        elif right_clamped and not left_clamped:
            # Assume that the right side is pushed in the other direction, or in the same direction. It is assumed here to be a typical design of opposite pushing.
            force = self.k['right_drive'] * strains['right_drive']
        # Full clamping or full release
        elif left_clamped and right_clamped:
            force = 0.0 # lock
            self.velocity = 0.0
        else:
            force = 0.0 # Free sliding, ignoring gravity
            self.velocity *= 0.9 # Friction decay

        # Kinematic equation: m*a + c*v = F
        # For simplicity, simple equivalent damping is used
        eq_damping = 1e4
        acceleration = (force - eq_damping * self.velocity) / self.mass

        self.velocity += acceleration * dt
        dx = self.velocity * dt
        self.position += dx

        return dx

    def simulate(self, voltage_sequence: List[Dict[str, float]], dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates a given voltage sequence.

        parameter:
        voltage_sequence: voltage sequence list
        dt: time step

        return:
        times: time array
        positions: displacement array
        """
        n_steps = len(voltage_sequence)
        times = np.arange(n_steps) * dt
        positions = np.zeros(n_steps)

        # Reset state
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
    Walking Gait Controller
    Responsible for generating four-phase gait: Clamp_L -> Drive_L -> Clamp_R -> Drive_R.
    """

    def __init__(self, dt: float = 1e-4):
        """
        Initialize controller

        parameter:
        dt: control period (s)
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
        Generate gait voltage sequence.

        parameter:
        freq: gait frequency (Hz)
        amplitude: driving voltage amplitude (V)
        duration: total duration (s)
        phase_delay: phase delay ratio (usually 0.25, which is 90 degrees)
        asymmetric: Whether it is an asymmetric gait mode

        return:
        List containing voltage dictionaries for each step
        """
        n_steps = int(duration / self.dt)
        times = np.arange(n_steps) * self.dt

        sequence = []
        period = 1.0 / freq if freq > 0 else 1.0

        for t in times:
            # Normalized phase [0, 1)
            phase = (t % period) / period

            # Clamp_L (left clamp) stage: 0.0 ~ 0.5
            clamp_l_vol = amplitude if phase < 0.5 or phase > 0.9 else 0.0

            # Drive_L (left drive) stage: elongation in Clamp_L stage
            drive_l_vol = 0.0
            if 0.1 < phase < 0.4:
                drive_l_vol = amplitude * np.sin((phase - 0.1) / 0.3 * np.pi / 2)
            elif phase >= 0.4:
                drive_l_vol = amplitude # Keep stretching until the clamp is released
            if phase > 0.9 or phase < 0.1:
                drive_l_vol = 0.0 #Reset to zero

            # Clamp_R (right clamp) stage: 0.4 ~ 0.9 (with overlapping area)
            clamp_r_vol = amplitude if 0.4 < phase < 0.9 else 0.0

            # Drive_R (right drive) stage
            drive_r_vol = 0.0
            if asymmetric:
                # Example asymmetric gait
                if 0.5 < phase < 0.8:
                    drive_r_vol = amplitude * 0.8 * np.sin((phase - 0.5) / 0.3 * np.pi / 2)
            else:
                if 0.5 < phase < 0.8:
                    drive_r_vol = amplitude * np.sin((phase - 0.5) / 0.3 * np.pi / 2)

            # Simple micro-stepping mode: can replace the square wave with a more continuous waveform
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
        Micro-stepping mode produces smooth sinusoidal transition waveforms to achieve sub-step resolution.
        """
        n_steps = int(duration / self.dt)
        times = np.arange(n_steps) * self.dt
        sequence = []

        for t in times:
            # Use sine and cosine waveforms for smooth transitions
            omega = 2 * np.pi * freq

            # Gripper: Biased sine wave ensures alternating gripping
            clamp_l = amplitude * (0.5 * np.sin(omega * t) + 0.5)
            clamp_r = amplitude * (0.5 * np.sin(omega * t + np.pi) + 0.5)

            # Driver: Sine waves 90 degrees apart
            drive_l = amplitude * (0.5 * np.sin(omega * t - np.pi/2) + 0.5)
            drive_r = amplitude * (0.5 * np.sin(omega * t + np.pi/2) + 0.5)

            # Enhance clamping force and simulate real clamping signal
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
    Nonlinear function fal (for ESO and NLSEF)
    """
    if abs(e) <= delta:
        return e / (delta ** (1 - alpha))
    else:
        return (abs(e) ** alpha) * np.sign(e)


class WalkingMotorESO:
    """
    dt: control step size
    Used to estimate displacement (z1), velocity (z2), acceleration/model uncertainty (z3), external load disturbance (z4)
    """

    def __init__(self, w0: float = 100.0, dt: float = 1e-4, b0: float = 1.0):
        """
        Initialize ESO.

        parameter:
        w0: observer bandwidth
        dt: sampling time
        b0: system control gain
        """
        self.w0 = w0
        self.dt = dt
        self.b0 = b0

        # Pole configuration gain (parameterized using bandwidth)
        self.beta1 = 4 * w0
        self.beta2 = 6 * (w0 ** 2)
        self.beta3 = 4 * (w0 ** 3)
        self.beta4 = w0 ** 4

        #State estimate: z = [z1, z2, z3, z4] (displacement, velocity, system disturbance, external disturbance)
        self.z = np.zeros(4)

    def update(self, y_meas: float, u: float, phase: str = 'drive') -> np.ndarray:
        """
        Update observer status.

        parameter:
        y_meas: measured displacement
        u: Current control quantity (driving voltage)
        phase: current gait phase ('clamp', 'drive'), used for adaptive gain adjustment

        return:
        np.ndarray: updated state estimate [z1, z2, z3, z4]
        """
        e = self.z[0] - y_meas

        # If it is the clamping stage, the dynamics of the system will change, reducing the sensitivity to disturbances
        k = 0.1 if phase == 'clamp' else 1.0

        # 4th order continuous system discretization (Eulerian method)
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
        """Get the current estimated full status"""
        return self.z.copy()


class WalkingMotorADRC:
    """
    Active Disturbance Rejection Controller
    Integrated gait planner, ESO and nonlinear state error feedback (NLSEF).
    """

    def __init__(self, wc: float = 10.0, w0: float = 100.0, b0: float = 1.0, dt: float = 1e-4):
        """
        Initialize ADRC.

        parameter:
        wc: control bandwidth
        w0: observer bandwidth
        b0: control gain
        dt: control period
        """
        self.dt = dt
        self.b0 = b0
        self.eso = WalkingMotorESO(w0=w0, dt=dt, b0=b0)
        self.gait_controller = WalkingGaitController(dt=dt)

        # PD control parameter configuration
        self.kp = wc ** 2
        self.kd = 2 * wc

        self.target_position = 0.0

    def disturbance_compensation(self, u0: float, z3: float, z4: float) -> float:
        """
        Disturbance compensation.

        parameter:
        u0: PD controller output
        z3: Estimated internal disturbance
        z4: Estimated external disturbance

        return:
        float: Control variable u after compensation
        """
        u = (u0 - z3 - z4) / self.b0

        # Limit the output voltage range
        u_max = 150.0
        u_min = -150.0
        return np.clip(u, u_min, u_max)

    def track(self, target_position: float, y_meas: float, current_u: float, phase: str = 'drive') -> float:
        """
        Perform single-step tracking control.

        parameter:
        target_position: target displacement
        y_meas: measured actual displacement
        current_u: The actual control amount currently applied
        phase: gait phase

        return:
        float: next step control quantity (equivalent driving voltage amplitude)
        """
        self.target_position = target_position

        # 1. Update ESO to obtain state estimate
        z = self.eso.update(y_meas, current_u, phase)
        z1, z2, z3, z4 = z[0], z[1], z[2], z[3]

        # 2. State error feedback control (PD law)
        e1 = target_position - z1
        e2 = 0.0 - z2 # The target speed is 0

        u0 = self.kp * e1 + self.kd * e2

        # 3. Disturbance compensation
        u = self.disturbance_compensation(u0, z3, z4)
        return u


class WalkingMotorOptimizer:
    """
    Walking Motor Optimizer
    Optional module: used to optimize the frequency and amplitude of gait for stride consistency and speed smoothness.
    """

    def __init__(self, plant: WalkingPiezoMotorPlant, controller: WalkingGaitController):
        self.plant = plant
        self.controller = controller

    def optimize_gait(self, target_speed: float, n_iter: int = 20) -> Tuple[float, float]:
        """
        Use a simple grid search or random search to find the optimal (freq, amplitude) combination.
        Goal: Minimize speed error.

        parameter:
        target_speed: target speed (m/s)
        n_iter: number of iterations

        return:
        Tuple[float, float]: optimal (freq, amplitude)
        """
        best_freq = 100.0
        best_amp = 100.0
        min_error = float('inf')

        # Simple random search as a demonstration
        np.random.seed(42)
        freqs = np.random.uniform(50.0, 500.0, n_iter)
        amps = np.random.uniform(50.0, 150.0, n_iter)

        dt = self.controller.dt
        duration = 0.05 # simulate 50ms

        for f, a in zip(freqs, amps):
            # Generate test gait
            seq = self.controller.generate_gait(freq=f, amplitude=a, duration=duration)

            # Simulation
            times, positions = self.plant.simulate(seq, dt)

            # Calculate average speed
            avg_speed = positions[-1] / duration

            error = abs(avg_speed - target_speed)
            if error < min_error:
                min_error = error
                best_freq = f
                best_amp = a

        return best_freq, best_amp
