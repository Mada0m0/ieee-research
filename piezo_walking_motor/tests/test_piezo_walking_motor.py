import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from piezo_walking_motor.src.piezo_walking_motor import (
    WalkingPiezoMotorPlant,
    WalkingGaitController,
    WalkingMotorESO,
    WalkingMotorADRC,
    WalkingMotorOptimizer,
)

class TestWalkingPiezoMotorPlant(unittest.TestCase):
    def setUp(self):
        self.plant = WalkingPiezoMotorPlant()
        self.dt = 1e-4

    def test_initialization(self):
        self.assertEqual(self.plant.position, 0.0)
        self.assertEqual(self.plant.velocity, 0.0)
        self.assertEqual(self.plant.clamp_preload, 100.0)

    def test_gait_cycle_all_clamped(self):
        voltages = {
            'left_clamp': 100.0,
            'right_clamp': 100.0,
            'left_drive': 0.0,
            'right_drive': 0.0
        }
        dx = self.plant.gait_cycle(voltages, self.dt)
        self.assertEqual(dx, 0.0)
        self.assertEqual(self.plant.velocity, 0.0)

    def test_simulate(self):
        # Create a simple sequence where we clamp left, drive left
        # to ensure position changes.
        # Need high voltage for left_clamp to overcome clamp_preload / 2
        # k['left_clamp'] * d['left_clamp'] = 1e7 * 1e-8 = 0.1
        # strain * k > 50 => voltage * 0.1 > 50 => voltage > 500
        sequence = [
            {'left_clamp': 600.0, 'right_clamp': 0.0, 'left_drive': 100.0, 'right_drive': 0.0},
            {'left_clamp': 600.0, 'right_clamp': 0.0, 'left_drive': 200.0, 'right_drive': 0.0},
        ]
        times, positions = self.plant.simulate(sequence, self.dt)
        self.assertEqual(len(times), 2)
        self.assertEqual(len(positions), 2)
        # Position should increase as we drive
        self.assertGreater(positions[-1], 0.0)


class TestWalkingGaitController(unittest.TestCase):
    def setUp(self):
        self.dt = 1e-4
        self.controller = WalkingGaitController(dt=self.dt)

    def test_generate_gait(self):
        freq = 100.0 # 100 Hz -> 0.01s period -> 100 steps per period
        amplitude = 120.0
        duration = 0.02 # 2 periods -> 200 steps

        seq = self.controller.generate_gait(freq, amplitude, duration)

        self.assertEqual(len(seq), 200)

        # Test first step (t=0)
        step_0 = seq[0]
        self.assertAlmostEqual(step_0['left_clamp'], amplitude)
        self.assertAlmostEqual(step_0['right_clamp'], 0.0)
        self.assertAlmostEqual(step_0['left_drive'], 0.0)

    def test_microstep_gait(self):
        freq = 50.0
        amplitude = 100.0
        duration = 0.02

        seq = self.controller.generate_microstep_gait(freq, amplitude, duration)
        self.assertEqual(len(seq), 200)

        step_0 = seq[0]
        self.assertTrue('left_clamp' in step_0)
        self.assertTrue('right_clamp' in step_0)


class TestWalkingMotorESO(unittest.TestCase):
    def setUp(self):
        self.dt = 1e-4
        self.eso = WalkingMotorESO(w0=100.0, dt=self.dt)

    def test_eso_convergence(self):
        # The initial state is all 0, give a measured steady-state input, and see if the state converges to it.
        y_meas = 1.0
        u = 0.0
        for _ in range(100):
            z = self.eso.update(y_meas, u, 'drive')

        # z1 should converge towards y_meas
        self.assertGreater(z[0], 0.0)

class TestWalkingMotorADRC(unittest.TestCase):
    def setUp(self):
        self.dt = 1e-4
        self.adrc = WalkingMotorADRC(wc=10.0, w0=100.0, b0=1.0, dt=self.dt)

    def test_track(self):
        # Test a single tracking step
        target_pos = 10.0
        current_y = 0.0
        current_u = 0.0

        next_u = self.adrc.track(target_pos, current_y, current_u)

        # Since target > current, controller should output a positive control signal
        self.assertGreater(next_u, 0.0)

    def test_sudden_load_disturbance(self):
        """突加负载抗扰测试 (Sudden load disturbance testing)"""
        # Test the ability of ADRC to compensate when subjected to internal and external disturbances under a given target

        # Use target and current_y that won't clip the PD controller output (+/- 150)
        target_pos = 1.0
        current_y = 1.0

        self.adrc.eso.z = np.array([1.0, 0.0, 0.0, 0.0])
        next_u_initial = self.adrc.track(target_pos, current_y, 0.0)

        # Inject artificial disturbances into ESO's estimate to simulate sudden load
        self.adrc.eso.z = np.array([1.0, 0.0, 0.0, 50.0])

        next_u_disturbed = self.adrc.track(target_pos, current_y, 0.0)

        # Since disturbance is positive, compensated control should be lower to counteract it
        self.assertLess(next_u_disturbed, next_u_initial)

    def test_step_consistency(self):
        """步距一致性对比 (Open-loop vs ADRC step consistency comparison)"""
        # A mock test showing how step consistency might be evaluated
        # This just verifies the logic paths for ADRC vs Open loop
        open_loop_u = 100.0

        # Under near perfect conditions, test ADRC output behavior
        target = 5.0
        # ESO might change z slightly during update, so we update it explicitly
        _ = self.adrc.eso.update(5.0, open_loop_u, 'drive')

        # Force states for testing disturbance compensation
        self.adrc.eso.z = np.array([5.0, 0.0, 0.0, 0.0])
        u_base = self.adrc.disturbance_compensation(0.0, 0.0, 0.0)

        self.assertAlmostEqual(u_base, 0.0)

class TestWalkingMotorOptimizer(unittest.TestCase):
    def test_optimizer_runs(self):
        # Simple test whether the optimizer can run properly and return reasonable results
        plant = WalkingPiezoMotorPlant()
        controller = WalkingGaitController()
        optimizer = WalkingMotorOptimizer(plant, controller)

        target_speed = 0.01 # m/s

        # Only run 2 iterations to keep test fast
        best_freq, best_amp = optimizer.optimize_gait(target_speed, n_iter=2)

        self.assertGreater(best_freq, 0.0)
        self.assertGreater(best_amp, 0.0)

if __name__ == '__main__':
    unittest.main()
