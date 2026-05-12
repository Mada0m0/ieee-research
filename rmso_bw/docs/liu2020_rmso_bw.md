# Liu 2020 — RMSO-BW + Fuzzy-NN

**Title:** Intelligent Rate-Dependent Hysteresis Control Compensator Design With Bouc-Wen Model Based on RMSO for Piezoelectric Actuator
**Authors:** Dongbo Liu, Yu Fang, Haibin Wang
**Journal:** IEEE Access, 2020
**DOI:** 10.1109/ACCESS.2020.2984645
**Open Access:** GOLD OA (https://ieeexplore.ieee.org/ielx7/6287639/8948470/09051673.pdf)
**Citations:** 11

## Abstract
Piezoelectric actuators (PAs) require high precision positioning for the applications of micro electrical mechanical systems, but it exhibits hysteresis nonlinearity which deteriorates positioning accuracy if no proper compensation is given. Hysteresis nonlinear modeling of PAs is a prime choice for hysteresis compensation. This paper proposes a novel intelligent positioning control algorithm based on Bouc-Wen (BW) model for the compensation of a bi-morph type piezoelectric actuator (PA) suffering rate-dependent hysteresis. A region based mixed-species swarm optimization (RMSO) algorithm is proposed for BW modeling to capture the dynamic nonlinearity of a piezoelectric actuator which exhibits rate-dependent hysteresis. Results of numerical simulations have been disclosed to illustrate the performance enhancement of RMSO over classical algorithm while they are applied to the parameter fitting problem of BW model for experimentally acquired datasets. An model based adaptive Fuzzy neural network (Fuzzy-NN) controller of PA is utilized to compensate the hysteresis for the positioning tracking control. Experimental results also illustrate the good performance of the proposed RMSO-BW based control scheme for the hysteresis compensation control of the PA.

## Methodology

### 1. Bouc-Wen Hysteresis Model
Classical BW differential equation:
```
dz/dt = A·dx/dt - β·|dx/dt|·|z|^(n-1)·z - γ·dx/dt·|z|^n
y = α·x + (1-α)·D·z
```
where x is input displacement, z is hysteresis state, y is output force.

### 2. RMSO (Region-based Mixed-Species Swarm Optimization)
- Divides swarm into multiple species regions
- Each region explores different areas of parameter space
- Species migrate between regions based on fitness
- Balances exploration (diverse species regions) and exploitation (local best within regions)
- Superior to classical PSO for BW parameter identification

### 3. Adaptive Fuzzy Neural Network Controller
- Model-based controller using Takagi-Sugeno fuzzy system
- Neural network adaptation for online parameter tuning
- Compensates rate-dependent hysteresis in real-time
- Combines feedforward BW inverse with adaptive feedback

## Innovation
1. RMSO algorithm for BW parameter identification (outperforms PSO, GA)
2. Fuzzy-NN adaptive controller for rate-dependent hysteresis compensation
3. Complete experimental validation on bi-morph PA setup

## Relevance
- ★★★★★ Piezoelectric Actuator Control – Directly Related
- ★★★★★ Bouc-Wen hysteresis model — core algorithm
- ★★★★★ Control algorithm — RMSO + Fuzzy-NN
- ★★☆☆☆ PMN material — not covered
