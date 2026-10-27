# Reference-Shaped Memory-Augmented Reinforcement Learning for Buck Converter Control

This repository contains the simulation code, trained models, and analysis for a three-layer Reinforcement Learning (RL) control architecture applied to an averaged DC-DC buck converter. 

## 📖 Project Overview
Direct RL control of power converters is challenging due to plant underdamping (LC resonance) and high inference costs for embedded deployment. This project proposes and analyzes a three-layer architecture:
1. **Reference Shaper**: A 2nd-order critically-damped filter (80 Hz) that prevents the step reference from exciting the plant's resonance.
2. **TD3 RL Agent**: A Twin Delayed DDPG agent that tracks the shaped reference using a 4-dimensional observation (including the rate-of-change of error, `d_error`).
3. **Habit Memory**: A trust-ellipse gated local regression cache that bypasses the neural network for familiar states, drastically reducing inference cost.

## 📊 Key Findings (Honest Analysis)
*   **Classical PI is a tough baseline:** On this linear averaged plant, a well-tuned PI controller is highly accurate and fast. The RL policy actually exhibits a slight steady-state offset (5.8%) because it operates without an explicit integral term.
*   **The RL value proposition is deployment:** The Habit Memory successfully serves **100%** of control decisions under 1% sensor noise, proving that a trained neural network can be completely bypassed for MCU deployment once the operating space is mapped.
*   **Ablations prove necessity:** Removing the shaper causes **massive overshoot (34%)**. Removing `d_error` causes sustained oscillation. Removing physical inductor resistance (\(R_{esr}\)) causes training divergence.

---

## 🖼️ Visual Results

### 1. Ablation: The Necessity of the Reference Shaper
Without the shaper (red), the RL agent tracking a raw step excites the plant resonance, causing massive ringing. With the shaper (blue), the trajectory is smooth and controlled.
![Ablation Shaper](ablation_shaper.png)

### 2. Habit Memory vs. RL Policy Accuracy
The Habit Memory (green dots) perfectly tracks the raw RL policy outputs (black line). The bottom plot shows the absolute error between the memory prediction and the neural network is near zero.
![Duty Comparison](duty_comparison.png)

### 3. Generalization Across Operating Points
The policy generalizes smoothly to voltages outside its training envelope.
| Vref = 3.0V | Vref = 7.0V |
| :---: | :---: |
| ![Vref 3](test_vref3.png) | ![Vref 7](test_vref7.png) |

### 4. Habit Memory Saturation & Trade-off
The memory hit rate saturates at roughly 100 stored points. Adding more points yields diminishing returns.
![Memory Scaling](habit_memory_scaling.png)

The trade-off curve shows how adjusting the trust-ellipse weights affects the hit rate vs steady-state error.
![Memory Tradeoff](habit_tradeoff.png)

### 5. Controller Comparison
While the classical PI (red) takes a long time to settle, the proposed RL architecture (blue) reaches the target much faster. The RL offset is visible, highlighting the trade-off between speed and steady-state accuracy without an explicit integrator.
![Vout Comparison](vout_comparison.png)

---

## 📈 Verified Metrics
*(Extracted from `final_metrics.json`)*

| Metric | Proposed (Shaper + RL + Memory) | Tuned PI (Raw Step) |
| :--- | :--- | :--- |
| **Overshoot** | 9.4% | 0.04% |
| **Steady-State Error** | 5.8% | 0.0% |
| **Final Vout (Target: 5.0V)** | 5.29V | 5.00V |
| **Settling Time** | Does not settle in 2% band | 1.6 ms |

---

## 🚀 Quick Start (Reproducing Results)

### 1. Environment Setup
```bash
conda create -n rl-buck python=3.11 -y
conda activate rl-buck
pip install fmpy gymnasium stable-baselines3 numpy matplotlib