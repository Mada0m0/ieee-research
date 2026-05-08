# 步进式 PZT 电机腿电压波形设计

## 一、电机腿的类型与拓扑

步进式 PZT 电机按腿的驱动方式分为三大类：

### 1. 夹持-驱动交替型（Clamp-Drive Alternating）
**代表结构：** 双足交替驱动，左右腿各含一个压电叠堆
- 左腿夹持 → 右腿驱动推进 → 右腿夹持 → 左腿驱动推进（四相步态）
- 论文：*Design and Experiment of a Clamping-Drive Alternating Operation Piezoelectric Actuator* (Micromachines, 2023, DOI: 10.3390/mi14030525)
- 特点：双柔性铰链防翻转，双层定子结构，前向运动无回退

### 2. 尺蠖型（Inchworm）
**代表结构：** 两端夹持 + 中间驱动
- 左夹持-右释放 → 驱动伸长 → 右夹持-左释放 → 驱动收缩
- 论文：*A Novel Piezoelectric Inchworm Actuator Driven by One Channel DC Signal* (IEEE TIE, 2020, DOI: 10.1109/tie.2020.2975493)
- 特点：单通道 DC 信号简化驱动，步距稳定（75V/3.2Hz 下 0.241μm/步）

### 3. 行走-推进型（Walker-Pusher）
**代表结构：** 两个压电叠堆分别负责抬腿和推进
- 论文：*A walker-pusher inchworm actuator driven by two piezoelectric stacks* (MSSP, 2021, DOI: 10.1016/j.ymssp.2021.108636)
- 特点：抬腿+推进解耦，步距可调范围大

---

## 二、电压波形类型与设计要点

### 波形类型对比

| 波形类型 | 适用模式 | 步距特性 | 速度平滑性 | 回退抑制 | 电路复杂度 |
|---------|---------|---------|-----------|---------|-----------|
| **锯齿波** Sawtooth | 粘滑 Stick-Slip | 微步距 0.1-10μrad | 中等 | 需补偿 | 低 |
| **梯形波** Trapezoidal | 夹持-驱动 | 大且稳定 | 好 | 好 | 中 |
| **正弦波** Sinusoidal | 超声/谐振 | 极小(纳米级) | 平滑 | 无回退 | 低 |
| **矩形波** Square | 冲击驱动 | 大但不稳定 | 差 | 差 | 最低 |
| **DC + 调制** | 改进尺蠖 | 稳定 | 好 | 好 | 最低 |

### 各类波形的详细设计

#### A. 锯齿波 (Sawtooth) — 粘滑驱动核心

最经典的波形，用于 stick-slip 压电电机腿。

**参数设计三要素：**
1. **上升沿斜率** (slow expansion)：决定慢伸/快缩比。典型 4:1 ~ 10:1
   - 上升沿(t_rise)占周期 80-90%，静摩擦驱动腿前进
   - 下降沿(t_fall)占 10-20%，动摩擦腿滑回
2. **幅值 V_pp**：决定步距大小。典型 13V（微步）~ 93V（高速）
   - 论文实测：3.5μrad/步 @ 13V/3000Hz | 0.44rad/s @ 93V/3000Hz
3. **频率 f**：影响速度与步距一致性
   - 低频(<100Hz)：步距大但抖动明显
   - 中频(1-5kHz)：最佳平衡区
   - 高频(>10kHz)：步距衰减，趋近连续运动

**锯齿波变体：**
```
标准锯齿波:   ▁▁▁▁██████▁▁▁▁██████▁▁▁▁
对称锯齿波:   ▁▁▁▁██████▁▁██▁▁▁▁██████
优化锯齿波:   ▁▁▁▁██████▁▁▁▁██████▁▁▁▁    ← 指数型缓升
                     ↘ 指数曲线 tanh(t/τ)
```

#### B. 梯形波 (Trapezoidal) — 夹持-驱动首选

用于 clamp-drive 交替型电机，每个相位有明确的保持时间。

**四相梯形时序：**
```
Phase 1 (Clamp-L):     ████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
Phase 2 (Drive-L→R):   ▁▁▁▁▁▁██████████▁▁▁▁▁▁
Phase 3 (Clamp-R):     ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁████████
Phase 4 (Drive-R→L):   ▁▁▁▁██████████▁▁▁▁▁▁▁▁
                        ↑       ↑       ↑       ↑
                       t0      t1      t2      t3
```

**关键设计参数：**
- **相位重叠** (overlap)：相邻相位重叠 5-15%，防止同时松开导致滑回
- **上升/下降斜率**：典型 50-100V/ms，过陡产生冲击，过缓降低步距
- **保持电压 V_hold**：夹持腿维持电压（典型 30-50% V_max），减少蠕变
- **占空比可调**：对称模式（各占25%）| 非对称（推进相长于夹持相）

#### C. 梯形波变体 — 微步进模式

将每个大步细分 N 个子步（N=8/16/32/128），电压阶梯递增：
```
Full step:  ██▁▁   步距 = 0.241μm
Microstep:  ▄▆██▄▆  步距 = 0.241/N μm

电压序列：0 → V/8 → 2V/8 → ... → V
               ↘ 正弦加权的微步距更均匀
```

#### D. DC 单通道驱动 — 最新趋势

*IEEE TIE 2020* 提出的创新方案：只用一路 DC 信号驱动尺蠖电机。
- 夹持力由 DC 电机 + 永磁铁产生（非压电）
- 压电叠堆由光电传感器检测磁铁位置触发
- 优势：驱动电路极简，无回退，步距稳定 0.241μm
- 局限：速度受限（最高 3.2Hz），需机械传感器

#### E. 正弦波 + 偏置 — 超声电机模式

超声电机使用两相或多相正弦波驱动，通过驻波/行波产生连续运动：
```
Phase A: V_a = V_0 * sin(ωt)
Phase B: V_b = V_0 * sin(ωt + φ)

φ = 90°  → 行波（旋转）
φ = 0°   → 驻波（振动）
偏置 V_DC 可调节零位
```

---

## 三、关键设计公式

### 步距估算

**夹持-驱动型：**
```
Δx_step = n * d33 * V_drive
其中：
  n = 压电叠堆层数
  d33 = 压电系数 (pm/V)
  V_drive = 驱动电压 (V)
```

**粘滑型：**
```
Δx_step = (d33 * V_pp) * (1 - t_fall / t_rise)
         ≈ d33 * V_pp * k
其中：
  k = 1 - (t_fall/t_rise)，典型值 0.6-0.9
  t_rise = (0.8-0.9) * T
  t_fall = (0.1-0.2) * T
```

### 速度估算

```
v = Δx_step * f * η
其中：
  f = 驱动频率 (Hz)
  η = 效率系数 (0.5-0.8，取决于负载和摩擦)
```

### 夹持力设计

```
F_clamp = k_preload * Δx_preload + F_electrostatic
其中：
  k_preload = 预紧弹簧刚度
  Δx_preload = 预压缩量
  F_electrostatic ≈ ε * A * (V_clamp / d)^2 / 2
```

---

## 四、前沿研究方向 (2023-2026)

| 方向 | 代表工作 | 亮点 |
|------|---------|------|
| 单通道DC驱动 | IEEE TIE 2020 | 驱动电路极简化 |
| 无回退双足交替 | Micromachines 2023 | 双柔性铰链防翻转 |
| 双足行走+推进 | MSSP 2021 | 步行+推进耦合设计 |
| 紧凑高速尺蠖 | MSSP 2022 | 桥式位移放大 |
| 纳米压印驱动 | Sens. Actuators A 2020 | 桥型柔性铰链 |
| 粘滑+尺蠖双模 | IEEE Access 2023 | 模式切换 |

---

## 五、推荐波形选择矩阵

| 电机类型 | 推荐波形 | 原因 |
|---------|---------|------|
| 双足夹持-驱动 | 梯形波（四相） | 相位可控，无回退 |
| 尺蠖型 | 梯形波（三相） | 夹持→驱动→释放时序 |
| 粘滑型 | 锯齿波 | 慢伸快缩原理 |
| 行走-推进型 | DC+调制 | 抬腿/推进解耦 |
| 超声型 | 正弦波（多相） | 共振驱动 |

### 实际波形调优建议

1. **优先优化上升/下降斜率** → 决定步距一致性
2. **相位重叠 5-15%** → 消除回退运动
3. **电压幅值由小到大扫频** → 找到谐振点（若有）
4. **微步进模式测试 N=8/16** → 平滑性 vs 速度的 trade-off
5. **频率扫描 1Hz→10kHz** → 确定最佳工作区间
6. **温度补偿 V_T** → 每 10°C 调整 V_pp 3-5%

---

## 六、关键参考文献

1. Li et al., "A Novel Piezoelectric Inchworm Actuator Driven by One Channel Direct Current Signal", *IEEE Trans. Ind. Electron.*, 2020. [DOI: 10.1109/tie.2020.2975493]
2. Wang et al., "A compact inchworm piezoelectric actuator with high speed: Design, modeling, and experimental evaluation", *Mech. Syst. Signal Process.*, 2022. [DOI: 10.1016/j.ymssp.2022.109704]
3. Yang et al., "A walker-pusher inchworm actuator driven by two piezoelectric stacks", *Mech. Syst. Signal Process.*, 2021. [DOI: 10.1016/j.ymssp.2021.108636]
4. Chen et al., "Design and experiment of a piezoelectric actuator based on inchworm working principle", *Sens. Actuators A*, 2020. [DOI: 10.1016/j.sna.2020.111950]
5. Zhang et al., "Design and Experiment of a Clamping-Drive Alternating Operation Piezoelectric Actuator", *Micromachines*, 2023. [DOI: 10.3390/mi14030525]
6. Liu et al., "Long Stroke Design of Piezoelectric Walking Actuator for Wafer Probe Station", *Micromachines*, 2022. [DOI: 10.3390/mi13030412]
7. Wang et al., "A Novel Stick-Slip Type Rotary Piezoelectric Actuator", *Adv. Mater. Sci. Eng.*, 2020. [DOI: 10.1155/2020/2659475]
8. Li et al., "Developments and Challenges of Miniature Piezoelectric Robots: A Review", *Adv. Sci.*, 2023. [DOI: 10.1002/advs.202305128]

---

*报告生成于 2026-05-07 | 数据源: OpenAlex.org*
