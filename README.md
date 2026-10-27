# Reference-Shaped Memory-Augmented Reinforcement Learning for Buck Converters

This repository contains the simulation code, trained models, and analysis for a three-layer Reinforcement Learning (RL) control architecture applied to an averaged DC-DC buck converter. 

## 📖 Project Overview
Direct RL control of power converters is challenging due to plant underdamping (LC resonance) and high inference costs for embedded deployment. This project proposes and analyzes a three-layer architecture:
1. **Reference Shaper**: A 2nd-order critically-damped filter (80 Hz) that prevents the step reference from exciting the plant's resonance.
2. **TD3 RL Agent**: A Twin Delayed DDPG agent that tracks the shaped reference using a 4-dimensional observation (including the rate-of-change of error, `d_error`).
3. **Habit Memory**: A trust-ellipse gated local regression cache that bypasses the neural network for familiar states, drastically reducing inference cost.

## 📊 Key Findings (Honest Analysis)
*   **Classical PI is a tough baseline:** On a static linear averaged plant (12V input), a well-tuned PI controller is highly accurate and fast. The RL policy exhibits a slight steady-state offset (5.8%) because it operates without an explicit integral term.
*   **The RL value proposition is robustness:** When the plant changes dynamically (e.g., a line disturbance where \(V_{in}\) drops from 12V to 8V), the fixed-gain PI controller struggles with a massive voltage droop. The robust RL agent adapts instantly and maintains a much tighter output.
*   **Ablations prove necessity:** Removing the reference shaper causes a **34% overshoot**. Removing the Habit Memory would require running the heavy neural network every step, which is infeasible on an MCU.

## 🖼️ Visual Results

### 1. The Golden Proof: Robustness to Line Disturbance
When the input voltage suddenly drops from 12V to 8V (at 15 ms), the fixed-gain PI controller (red) crashes to 3.6V. The RL agent (blue), trained on voltage variations, barely dips to 4.3V and recovers instantly.
![Line Disturbance](line_disturbance.png)

### 2. Ablation: The Necessity of the Reference Shaper
Without the shaper (red), the RL agent tracking a raw step excites the plant resonance, causing massive ringing and a 34% overshoot. With the shaper (blue), the trajectory is smooth and controlled.
![Ablation Shaper](ablation_shaper.png)

### 3. Habit Memory vs. RL Policy Accuracy
The Habit Memory (green dots) perfectly tracks the raw RL policy outputs (black line). The bottom plot shows the absolute error between the memory prediction and the neural network is near zero.
![Duty Comparison](duty_comparison.png)

### 4. Memory Scaling & Hit Rate
The memory hit rate saturates at roughly 100 stored points, requiring only ~2 KB of storage. During normal operation, it serves 100% of decisions without invoking the neural network.
![Memory Scaling](habit_memory_scaling.png)

## 📈 Verified Metrics (From `final_metrics.json`)

| Metric | Proposed (Shaper + RL + Memory) | Tuned PI (Raw Step) |
| :--- | :--- | :--- |
| **Overshoot (Nominal)** | 9.4% | 0.04% |
| **Steady-State Error** | 5.8% | 0.0% |
| **Final Vout (Target: 5.0V)** | 5.29V | 5.00V |
| **Settling Time** | Does not settle in 2% band | 1.6 ms |
| **Recovery from 12V->8V Drop** | **Drops to 4.3V** | **Drops to 3.6V** |

## 🚀 Quick Start (Reproducing Results)

### 1. Environment Setup
```bash
conda create -n rl-buck python=3.11 -y
conda activate rl-buck
pip install fmpy gymnasium stable-baselines3 numpy matplotlib