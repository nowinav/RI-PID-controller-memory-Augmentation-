# Reference-Shaped Memory-Augmented Reinforcement Learning for Buck Converters

This repository contains the simulation code, trained models, and analysis for a three-layer Reinforcement Learning (RL) control architecture applied to an averaged DC-DC buck converter. 

## 📖 Project Overview
Direct RL control of power converters is challenging due to plant underdamping (LC resonance) and high inference costs for embedded deployment. This project proposes a three-layer architecture:
1. **Reference Shaper**: A 2nd-order critically-damped filter (80 Hz) that prevents the step reference from exciting the plant's resonance.
2. **TD3 RL Agent**: A Twin Delayed DDPG agent that tracks the shaped reference using a 4-dimensional observation (including the rate-of-change of error, `d_error`).
3. **Habit Memory**: A trust-ellipse gated local regression cache that bypasses the neural network for familiar states, drastically reducing inference cost.

## 📊 Key Findings (Honest Analysis)
*   **Classical PI is tough to beat on linear plants:** A well-tuned PI controller achieves excellent transient response and zero steady-state error on the averaged linear model. 
*   **The RL value proposition is deployment:** The Habit Memory reduces neural network inference by **[INSERT HIT RATE]%**, making a 10 kHz control loop feasible on a microcontroller.
*   **Ablations prove necessity:** Removing the shaper causes **[INSERT ABLATION]%** overshoot. Removing `d_error` causes sustained oscillation. Removing physical inductor resistance (\(R_{esr}\)) causes training divergence.

## 🚀 Quick Start (Reproducing Results)

### 1. Environment Setup
```bash
conda create -n rl-buck python=3.11 -y
conda activate rl-buck
pip install fmpy gymnasium stable-baselines3 numpy matplotlib