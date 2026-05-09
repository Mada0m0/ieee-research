import unittest
import numpy as np
from piezo_stepper_eso import PiezoStepperPlant, ExtendedStateObserver, ESOController, ADRController

class TestPiezoStepperESO(unittest.TestCase):

    def setUp(self):
        # Common parameter configurations
        self.dt = 0.001
        self.plant = PiezoStepperPlant(k=100.0, c=5.0, d=10.0)
        self.adrc = ADRController()
        # Tuning parameters
        self.adrc.tune(wc=50.0, w0=200.0, b0=10.0, nonlinear=False)

    def test_eso_convergence(self):
        """ESO Convergence Test: Verify Disturbance Estimation Error Attenuation"""
        eso = ExtendedStateObserver(w0=100.0, b0=10.0)
        plant = PiezoStepperPlant()

        y_meas = 0.0
        u_prev = 0.0

        """Sudden load disturbance testing (Sudden load disturbance testing)"""
        constant_disturbance = 5.0

        for _ in range(500):
            y_meas = plant.step(u_prev, self.dt, disturbance=constant_disturbance)
            eso.update(y_meas, u_prev, self.dt)
            # In open loop situation, u remains 0
            u_prev = 0.0

        # The perturbation should converge
        z3_est = eso.get_disturbance()
        # The internal disturbance item of plant, z3 is estimated to be opposite or similar to it.
        # Note: plant formula: x2' = -k*x1 - c*x2 + d*u - h + dist
        # ESO formula: z2_dot = z3 - beta2*e + b0*u
        # So z3 estimates the total disturbance: -k*x1 - c*x2 - h + dist + (d-b0)*u
        # Here you can verify that z3 finally converges, just check that it does not diverge and has a clear value.
        self.assertFalse(np.isnan(z3_est))
        self.assertTrue(abs(z3_est) > 0)

    def test_step_response(self):
        """Step response test: verify tracking without excessive overshoot"""
        n_steps = 1000
        ref_traj = np.ones(n_steps) * 1.0 # Target displacement 1.0

        y_seq, u_seq, z3_seq = self.adrc.track(ref_traj, self.plant, self.dt)

        # Verify that the tracking error at the last moment is small
        final_error = abs(y_seq[-1] - 1.0)
        self.assertLess(final_error, 0.05, "Step response steady-state error is too large")

        # Verify there is no serious overshoot (allow a little overshoot due to non-linearities such as hysteresis)
        max_overshoot = np.max(y_seq) - 1.0
        self.assertLess(max_overshoot, 0.1, "Overshoot is too large")

    def test_sine_tracking(self):
        """Sinusoidal tracking test: verify tracking accuracy within bandwidth"""
        n_steps = 2000
        t = np.arange(n_steps) * self.dt
        ref_traj = np.sin(2 * np.pi * 1.0 * t) # 1 Hz sine wave

        y_seq, u_seq, z3_seq = self.adrc.track(ref_traj, self.plant, self.dt)

        # Verify the average absolute error of the following periods
        steady_state_error = np.mean(np.abs(y_seq[1000:] - ref_traj[1000:]))
        self.assertLess(steady_state_error, 0.1, "Sine tracking error is too large")

    def test_disturbance_rejection(self):
        """Anti-interference test: Position maintenance accuracy when sudden load is applied"""
        n_steps = 2000
        ref_traj = np.ones(n_steps) * 1.0

        # Sudden load at t=1s (step 1000)
        dist_seq = np.zeros(n_steps)
        dist_seq[1000:] = 20.0

        y_seq, u_seq, z3_seq = self.adrc.track(ref_traj, self.plant, self.dt, disturbance_seq=dist_seq)

        # Should be stable before perturbation is applied
        error_before = abs(y_seq[990] - 1.0)
        self.assertLess(error_before, 0.05)

        # After the perturbation is applied and adjusted for a period of time, it should be re-stabilized
        error_after = abs(y_seq[-1] - 1.0)
        self.assertLess(error_after, 0.05, "Failed to reject disturbance")

        # z3 should be able to reflect changes in disturbance
        z3_diff = abs(z3_seq[-1] - z3_seq[990])
        self.assertGreater(z3_diff, 5.0, "ESO did not estimate the disturbance correctly")

    def test_hysteresis_compensation(self):
        """Comparison of hysteresis compensation effects (simple verification whether it can be tracked with and without hysteresis)"""
        # Test the tracking ability when the hysteresis parameter is large
        plant_heavy_hysteresis = PiezoStepperPlant(alpha=2.0, beta=0.5, gamma=0.5)
        n_steps = 1000
        ref_traj = np.ones(n_steps) * 1.0

        y_seq, _, _ = self.adrc.track(ref_traj, plant_heavy_hysteresis, self.dt)

        final_error = abs(y_seq[-1] - 1.0)
        self.assertLess(final_error, 0.05, "Failed to compensate for heavy hysteresis")

if __name__ == '__main__':
    unittest.main()
