import numpy as np
import matplotlib.pyplot as plt
from implementations.controls.piezo_walking_motor import (
    WalkingPiezoMotorPlant,
    WalkingGaitController,
    WalkingMotorADRC
)

def plot_gait_sequence():
    """Draw the four-phase gait voltage sequence"""
    controller = WalkingGaitController(dt=1e-4)
    # Generate 2 cycles of gait
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
    plt.plot(times, clamp_l, label='Clamp L (left clamp)', color='b', alpha=0.8)
    plt.plot(times, drive_l, label='Drive L (left drive)', color='c', linestyle='--')
    plt.plot(times, clamp_r, label="Clamp R (right clamp)", color="r", alpha=0.8)
    plt.plot(times, drive_r, label='Drive R (right drive)', color='m', linestyle='--')

    plt.title('Walking Piezo Motor - 4 Phase Gait Sequence')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('implementations/controls/motor_gait.png')
    plt.close()

def plot_displacement():
    """Draw the open-loop motion displacement curve of the motor"""
    plant = WalkingPiezoMotorPlant()
    controller = WalkingGaitController(dt=1e-4)

    freq = 100.0
    amplitude = 600.0 # A higher voltage is required to overcome the clamping preload force
    duration = 0.05 # 5 periods

    seq = controller.generate_gait(freq, amplitude, duration)
    times, positions = plant.simulate(seq, dt=1e-4)

    plt.figure(figsize=(10, 6))
    plt.plot(times, positions * 1e6, label='Displacement', color='g')
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
