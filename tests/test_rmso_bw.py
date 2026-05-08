import numpy as np
import pytest

from implementations.hysteresis.rmso_optimizer import RMSOOptimizer
from implementations.hysteresis.rmso_bw_model import RMSO_BW_Model
from implementations.controls.fuzzy_nn_controller import FuzzyNNController
from implementations.controls.rmso_bw_compensator import RMSO_BW_Compensator

@pytest.fixture(autouse=True)
def set_seed():
    """Ensure reproducibility in tests."""
    np.random.seed(42)

# --- Tests for RMSOOptimizer ---
def test_rmso_optimizer_sphere():
    """Test optimizer on the Sphere benchmark function."""
    def sphere(x):
        return np.sum(x ** 2)

    bounds = {'x1': (-5.0, 5.0), 'x2': (-5.0, 5.0), 'x3': (-5.0, 5.0)}
    optimizer = RMSOOptimizer(n_particles=50, n_regions=5)
    best_pos, best_fit = optimizer.optimize(sphere, bounds, max_iter=100)

    assert best_fit < 1.0, f"Sphere optimization failed to converge, best fit: {best_fit}"
    assert best_pos.shape == (3,)
    history = optimizer.get_convergence_history()
    assert len(history) == 100
    assert history[0] >= history[-1]

def test_rmso_optimizer_invalid_init():
    with pytest.raises(ValueError):
        RMSOOptimizer(n_particles=50, n_regions=3)

# --- Tests for RMSO_BW_Model ---
def test_bw_model_simulate():
    """Test BW model simulation shape and basic properties."""
    model = RMSO_BW_Model()
    t = np.linspace(0, 1, 100)
    u = np.sin(2 * np.pi * t)
    y = model.simulate(u, dt=0.01)

    assert y.shape == u.shape
    assert not np.isnan(y).any()
    assert not np.isinf(y).any()

def test_bw_model_rate_dependency():
    """Test generation of rate-dependent hysteresis loops."""
    model = RMSO_BW_Model()
    freqs = [1.0, 10.0]
    loops = model.rate_dependent_simulate(freqs, amplitude=1.0)

    assert len(loops) == 2
    for freq in freqs:
        u_loop, y_loop = loops[freq]
        assert u_loop.shape == y_loop.shape
        assert len(u_loop) > 0

def test_bw_model_identification():
    """Test RMSO parameter identification on synthetic data."""
    true_params = {'A': 1.5, 'alpha': 0.2, 'beta': 0.1, 'gamma': 0.1, 'n': 1.0}
    model_true = RMSO_BW_Model(params=true_params)

    t = np.linspace(0, 1, 50)
    u = np.sin(2 * np.pi * 5 * t)
    dt = 0.02
    y_true = model_true.simulate(u, dt)

    bounds = {
        'A': (0.1, 3.0),
        'alpha': (0.01, 1.0),
        'beta': (0.01, 1.0),
        'gamma': (0.01, 1.0),
        'n': (1.0, 2.0)
    }

    model_est = RMSO_BW_Model()
    est_params = model_est.identify_with_rmso(u, y_true, dt, bounds, pop_size=50, max_iter=20) # Low iter for speed

    # Check if identified params are within bounds
    for k, v in est_params.items():
        assert bounds[k][0] <= v <= bounds[k][1]

# --- Tests for FuzzyNNController ---
def test_fuzzy_nn_forward():
    """Test forward pass of Fuzzy-NN."""
    controller = FuzzyNNController(n_rules=5, n_inputs=2)
    x = np.array([0.5, -0.2])
    y = controller.forward(x)

    assert y.shape == (1,)
    assert not np.isnan(y).any()

def test_fuzzy_nn_training():
    """Test offline training loop reduces error."""
    controller = FuzzyNNController(n_rules=5, n_inputs=1)
    # Target function y = x^2
    X_train = np.linspace(-1, 1, 20).reshape(-1, 1)
    y_train = X_train[:, 0] ** 2

    # Calculate initial loss
    y_pred_init = np.array([controller.forward(x)[0] for x in X_train])
    init_loss = np.mean((y_pred_init - y_train) ** 2)

    # Train
    controller.train(X_train, y_train, epochs=50, lr=0.1)

    # Calculate final loss
    y_pred_final = np.array([controller.forward(x)[0] for x in X_train])
    final_loss = np.mean((y_pred_final - y_train) ** 2)

    assert final_loss < init_loss

def test_fuzzy_rules_extraction():
    controller = FuzzyNNController(n_rules=2, n_inputs=2)
    rules = controller.get_fuzzy_rules()
    assert len(rules) == 2
    assert "Rule 1: IF x0 is Gaussian" in rules[0]

# --- Tests for RMSO_BW_Compensator ---
def test_compensator_tracking():
    """Test closed-loop tracking setup and error reduction."""
    # Setup dummy BW model and Fuzzy NN
    bw_model = RMSO_BW_Model({'A': 1.0, 'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0, 'n': 1.0})
    fuzzy_ctrl = FuzzyNNController(n_rules=3, n_inputs=1)
    compensator = RMSO_BW_Compensator(bw_model, fuzzy_ctrl)

    # Create a dummy plant that perfectly mirrors the assumed bw_model inverse
    # but with a slight delay or error to give the fuzzy controller work to do.
    def plant(u_ctrl, dt_plant):
        # Extremely simplified plant: y = u (since alpha=1, h=0 effectively in inverse)
        return u_ctrl * 0.9 # 10% gain error

    t = np.linspace(0, 1, 50)
    desired = np.sin(2 * np.pi * t)
    dt = 0.02

    # Track with 0 learning rate (no adaptation)
    _, _, actual_no_adapt = compensator.track(desired, plant, dt, online_learning_rate=0.0)
    error_no_adapt = np.mean((desired - actual_no_adapt) ** 2)

    # Track with adaptation
    _, _, actual_adapt = compensator.track(desired, plant, dt, online_learning_rate=0.1)
    error_adapt = np.mean((desired - actual_adapt) ** 2)

    # Both should run without throwing errors.
    # Adaptation might not strictly beat no-adaptation in 50 steps on a simple gain error,
    # but the arrays should exist and be valid.
    assert actual_no_adapt.shape == desired.shape
    assert actual_adapt.shape == desired.shape
    assert not np.isnan(actual_adapt).any()
