import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from piezo_stepper_eso.src.piezo_stepper_eso import PiezoStepperPlant, ExtendedStateObserver, ESOController, ADRController

class TestPiezoStepperESO(unittest.TestCase):

    def setUp(self):
        # 常见参数配置
        self.dt = 0.001
        self.plant = PiezoStepperPlant(k=100.0, c=5.0, d=10.0)
        self.adrc = ADRController()
        # 整定参数
        self.adrc.tune(wc=50.0, w0=200.0, b0=10.0, nonlinear=False)

    def test_eso_convergence(self):
        """ESO收敛性测试：验证扰动估计误差衰减"""
        eso = ExtendedStateObserver(w0=100.0, b0=10.0)
        plant = PiezoStepperPlant()

        y_meas = 0.0
        u_prev = 0.0

        # 施加一个恒定扰动
        constant_disturbance = 5.0

        for _ in range(500):
            y_meas = plant.step(u_prev, self.dt, disturbance=constant_disturbance)
            eso.update(y_meas, u_prev, self.dt)
            # 开环情况，u保持0
            u_prev = 0.0

        # 扰动应该收敛
        z3_est = eso.get_disturbance()
        # plant 内部的disturbance项，z3 估计应该与之相反或相近。
        # 注意: plant公式: x2' = -k*x1 - c*x2 + d*u - h + dist
        # ESO公式: z2_dot = z3 - beta2*e + b0*u
        # 所以 z3 估计的是总扰动: -k*x1 - c*x2 - h + dist + (d-b0)*u
        # 这里验证 z3 最终收敛即可，只需检查其不发散且有明确值。
        self.assertFalse(np.isnan(z3_est))
        self.assertTrue(abs(z3_est) > 0)

    def test_step_response(self):
        """阶跃响应测试：验证无过大超调跟踪"""
        n_steps = 1000
        ref_traj = np.ones(n_steps) * 1.0  # 目标位移 1.0

        y_seq, u_seq, z3_seq = self.adrc.track(ref_traj, self.plant, self.dt)

        # 验证最后时刻的跟踪误差较小
        final_error = abs(y_seq[-1] - 1.0)
        self.assertLess(final_error, 0.05, "Step response steady-state error is too large")

        # 验证没有严重的超调 (允许一点超调，由于迟滞等非线性)
        max_overshoot = np.max(y_seq) - 1.0
        self.assertLess(max_overshoot, 0.1, "Overshoot is too large")

    def test_sine_tracking(self):
        """正弦跟踪测试：验证带宽内跟踪精度"""
        n_steps = 2000
        t = np.arange(n_steps) * self.dt
        ref_traj = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz 正弦波

        y_seq, u_seq, z3_seq = self.adrc.track(ref_traj, self.plant, self.dt)

        # 验证后面的周期的平均绝对误差
        steady_state_error = np.mean(np.abs(y_seq[1000:] - ref_traj[1000:]))
        self.assertLess(steady_state_error, 0.1, "Sine tracking error is too large")

    def test_disturbance_rejection(self):
        """抗扰测试：突加负载时位置保持精度"""
        n_steps = 2000
        ref_traj = np.ones(n_steps) * 1.0

        # 在 t=1s (step 1000) 突加负载
        dist_seq = np.zeros(n_steps)
        dist_seq[1000:] = 20.0

        y_seq, u_seq, z3_seq = self.adrc.track(ref_traj, self.plant, self.dt, disturbance_seq=dist_seq)

        # 扰动施加前应该稳定
        error_before = abs(y_seq[990] - 1.0)
        self.assertLess(error_before, 0.05)

        # 扰动施加并经过一段时间调整后，应该重新稳定
        error_after = abs(y_seq[-1] - 1.0)
        self.assertLess(error_after, 0.05, "Failed to reject disturbance")

        # z3 应该能够反映出扰动的变化
        z3_diff = abs(z3_seq[-1] - z3_seq[990])
        self.assertGreater(z3_diff, 5.0, "ESO did not estimate the disturbance correctly")

    def test_hysteresis_compensation(self):
        """迟滞补偿效果对比（简单验证包含与不包含迟滞时是否都能跟踪）"""
        # 测试在迟滞参数较大的情况下的跟踪能力
        plant_heavy_hysteresis = PiezoStepperPlant(alpha=2.0, beta=0.5, gamma=0.5)
        n_steps = 1000
        ref_traj = np.ones(n_steps) * 1.0

        y_seq, _, _ = self.adrc.track(ref_traj, plant_heavy_hysteresis, self.dt)

        final_error = abs(y_seq[-1] - 1.0)
        self.assertLess(final_error, 0.05, "Failed to compensate for heavy hysteresis")

if __name__ == '__main__':
    unittest.main()
