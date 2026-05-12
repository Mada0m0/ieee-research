import unittest
import numpy as np
import sys
import os

# Add the project's root directory to the python path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from generalized_bouc_wen.src.generalized_bouc_wen import GeneralizedBoucWen

class TestGeneralizedBoucWen(unittest.TestCase):

    def setUp(self):
        self.t = np.linspace(0, 10, 100)
        self.x = np.sin(self.t)

    def test_initialization(self):
        model = GeneralizedBoucWen(A=2.0, alpha=0.5, beta=1.5, gamma=0.8, n=2.0, delta=0.2)
        params = model.get_params()
        self.assertTrue(np.allclose(params, [2.0, 0.5, 1.5, 0.8, 2.0, 0.2]))

        # Test IIR filter defaults
        self.assertEqual(model.b, [1.0])
        self.assertEqual(model.a, [1.0])

    def test_set_params(self):
        model = GeneralizedBoucWen()
        new_params = np.array([3.0, 1.2, 0.5, 0.5, 1.5, -0.1])
        model.set_params(new_params)
        self.assertTrue(np.allclose(model.get_params(), new_params))

    def test_simulate_shapes(self):
        model = GeneralizedBoucWen()
        F_sim = model.simulate(self.t, self.x)
        self.assertEqual(F_sim.shape, self.x.shape)

    def test_simulate_value_error(self):
        model = GeneralizedBoucWen()
        t_wrong = np.linspace(0, 10, 50)
        with self.assertRaises(ValueError):
            model.simulate(t_wrong, self.x)

    def test_identify_parameters_pso(self):
        # Create a target response
        target_model = GeneralizedBoucWen(A=1.5, alpha=0.3, beta=0.8, gamma=0.8, n=1.0, delta=0.1)
        F_target = target_model.simulate(self.t, self.x)

        # Initialize a model with different parameters
        model_to_identify = GeneralizedBoucWen(A=1.0, alpha=0.1, beta=1.0, gamma=1.0, n=1.0, delta=0.0)

        # Run PSO
        # Use narrow bounds around the target to make the test run quickly and reliably
        bounds = (
            np.array([1.0, 0.0, 0.5, 0.5, 0.8, -0.2]),
            np.array([2.0, 0.5, 1.5, 1.5, 1.2, 0.2])
        )

        best_params = model_to_identify.identify_parameters_pso(
            self.t, self.x, F_target,
            num_particles=10, max_iter=10, bounds=bounds
        )

        # Check that the model's parameters were updated to the best found
        self.assertTrue(np.allclose(model_to_identify.get_params(), best_params))

        # Check that the error is smaller than with initial parameters
        # Re-initialize to get initial error
        model_initial = GeneralizedBoucWen(A=1.0, alpha=0.1, beta=1.0, gamma=1.0, n=1.0, delta=0.0)
        F_initial = model_initial.simulate(self.t, self.x)
        initial_mse = np.mean((F_initial - F_target) ** 2)

        F_identified = model_to_identify.simulate(self.t, self.x)
        identified_mse = np.mean((F_identified - F_target) ** 2)

        self.assertLess(identified_mse, initial_mse)

    def test_inverse_simulate(self):
        # Create model with typical parameters
        model = GeneralizedBoucWen(A=1.0, alpha=0.5, beta=0.1, gamma=0.1, n=1.0, delta=0.0)

        # Simulate forward
        F_sim = model.simulate(self.t, self.x)

        # Simulate inverse (predict x from F_sim)
        x_inv = model.inverse_simulate(self.t, F_sim)

        # Inverse simulation often has numerical drift due to Euler integration
        # and approximations. We check if shapes match and values are reasonably correlated.
        self.assertEqual(x_inv.shape, self.x.shape)

        # Pearson correlation should be high
        corr = np.corrcoef(self.x, x_inv)[0, 1]
        self.assertGreater(corr, 0.90)

if __name__ == '__main__':
    unittest.main()
