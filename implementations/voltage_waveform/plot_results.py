import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from implementations.voltage_waveform.generators import TrapezoidalWaveGenerator, SawtoothWaveGenerator
from implementations.voltage_waveform.optimizer import GAWaveformOptimizer

def plot_all():
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(15, 12))

    # --- Plot 1: 时域电压波形图（标注四相时序） ---
    ax1 = plt.subplot(3, 1, 1)

    gen = TrapezoidalWaveGenerator(amplitude=80.0, frequency=1000.0, overlap_ratio=0.10)
    T = gen.period
    t = np.linspace(0, 2.5 * T, 2000)
    v1, v2, v3, v4 = gen.generate(t)

    t_ms = t * 1000.0  # 转换为 ms

    ax1.plot(t_ms, v1, label="Phase 1 (Clamp-L)", linewidth=2, alpha=0.8)
    ax1.plot(t_ms, v2, label="Phase 2 (Drive L->R)", linewidth=2, alpha=0.8)
    ax1.plot(t_ms, v3, label="Phase 3 (Clamp-R)", linewidth=2, alpha=0.8)
    ax1.plot(t_ms, v4, label="Phase 4 (Drive R->L)", linewidth=2, alpha=0.8)

    ax1.set_title("Time-domain Voltage Waveform (4-Phase Trapezoidal, f=1kHz, overlap=10%)", fontsize=14)
    ax1.set_xlabel("Time (ms)", fontsize=12)
    ax1.set_ylabel("Voltage (V)", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.set_ylim(-5, 90)

    # --- Plot 2: 步距-频率特性曲线 ---
    ax2 = plt.subplot(3, 1, 2)

    opt = GAWaveformOptimizer(mode="trapezoidal")
    freqs = np.linspace(100, 10000, 100)

    # 模拟在恒定振幅下的步距随频率变化关系
    amp = 80.0
    overlap = 0.10
    step_distances = [opt._simulate_step_distance((amp, f, overlap)) * 1000 for f in freqs] # 转成纳米方便显示

    ax2.plot(freqs, step_distances, 'b-', linewidth=2)
    ax2.set_title("Step Distance vs Frequency (Trapezoidal Wave, V=80V)", fontsize=14)
    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.set_ylabel("Step Distance (nm)", fontsize=12)

    # --- Plot 3: 步进位移-时间曲线（体现无回退特性） ---
    ax3 = plt.subplot(3, 1, 3)

    # 简化模型：Clamp-L和Drive L->R产生正向位移，Clamp-R和Drive R->L保持位移或准备下一循环。
    # 我们根据梯形波计算理论位移
    # 推进速度正比于 Phase 2 (Drive L->R) 的电压导数，但为了简化，直接将有效驱动时间转化为位移

    disp = np.zeros_like(t)
    current_disp = 0.0
    step_size = opt._simulate_step_distance((80.0, 1000.0, 0.10)) * 1000  # nm

    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        # 当 Phase 1 和 Phase 3重叠时，或者处于推进相位时
        # 这里用一个简单的逻辑，只有当 Phase 2 > 0 且 Phase 1 > 0 时开始推进
        # 在我们的四相时序中，Drive L->R (Phase 2) 开启时产生位移
        if v2[i] > v2[i-1] and v1[i] > 10.0:  # v1处于夹持状态，v2正在上升
            current_disp += (step_size / (T/4)) * dt
        disp[i] = current_disp

    ax3.plot(t_ms, disp, 'g-', linewidth=2)
    ax3.set_title("Step Displacement Over Time (Illustrating No Backward Motion)", fontsize=14)
    ax3.set_xlabel("Time (ms)", fontsize=12)
    ax3.set_ylabel("Cumulative Displacement (nm)", fontsize=12)

    plt.tight_layout()

    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/latest_research_figure.svg', format='svg', dpi=300)
    print("Plot saved to assets/latest_research_figure.svg")

if __name__ == "__main__":
    plot_all()
