import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from piezo_walking_motor.src.piezo_walking_motor import (
    WalkingPiezoMotorPlant,
    WalkingGaitController,
    WalkingMotorADRC
)

def plot_gait_sequence():
    """绘制四相步态电压序列"""
    controller = WalkingGaitController(dt=1e-4)
    # 生成2个周期的步态
    freq = 100.0
    amplitude = 150.0
    duration = 0.02

    seq = controller.generate_gait(freq, amplitude, duration)
    times = np.arange(len(seq)) * 1e-4

    clamp_l = [step['left_clamp'] for step in seq]
    drive_l = [step['left_drive'] for step in seq]
    clamp_r = [step['right_clamp'] for step in seq]
    drive_r = [step['right_drive'] for step in seq]

    plt.figure(figsize=(10, 6))
    plt.plot(times, clamp_l, label='Clamp L (左夹持)', color='b', alpha=0.8)
    plt.plot(times, drive_l, label='Drive L (左驱动)', color='c', linestyle='--')
    plt.plot(times, clamp_r, label='Clamp R (右夹持)', color='r', alpha=0.8)
    plt.plot(times, drive_r, label='Drive R (右驱动)', color='m', linestyle='--')

    plt.title('Walking Piezo Motor - 4 Phase Gait Sequence')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('implementations/controls/motor_gait.png')
    plt.close()

def plot_displacement():
    """绘制电机开环运动位移曲线"""
    plant = WalkingPiezoMotorPlant()
    controller = WalkingGaitController(dt=1e-4)

    freq = 100.0
    amplitude = 600.0 # 需要较高的电压来克服夹持预紧力
    duration = 0.05   # 5个周期

    seq = controller.generate_gait(freq, amplitude, duration)
    times, positions = plant.simulate(seq, dt=1e-4)

    plt.figure(figsize=(10, 6))
    plt.plot(times, positions * 1e6, label='Displacement (位移)', color='g')
    plt.title('Walking Piezo Motor - Open Loop Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (um)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('implementations/controls/motor_displacement.png')
    plt.close()

if __name__ == '__main__':
    plot_gait_sequence()
    plot_displacement()
    print("Plots saved to implementations/controls/motor_gait.png and motor_displacement.png")
