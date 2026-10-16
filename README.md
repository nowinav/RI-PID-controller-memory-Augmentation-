# Reference-Shaped Memory-Augmented RL for Buck Converters

A three-layer control architecture for a switching-level buck converter:
reference shaper + TD3 reinforcement learning policy + habit memory cache.

## Project structure

    ├── buck_env.py                # Gymnasium environment wrapping the FMU
    ├── train.py                    # TD3 training script
    ├── evaluate.py                 # Evaluation and plot generation
    ├── habit_memory.py             # Trust-ellipse + local regression cache
    ├── populate_memory.py          # Populate the habit memory bank
    ├── evaluate_habit.py           # Compare RL-only vs RL+memory
    ├── sweep_weights.py            # Memory weight sweep
    ├── save_deployment_memory.py   # Save a deployable 500-point memory
    ├── test_*.py                   # Sanity and validation scripts
    ├── BuckConverter_Averaged.mo   # Modelica source of the plant
    ├── BuckConverter_Averaged.fmu  # Compiled FMU used by Python
    ├── models/                     # Trained policies and habit memory
    └── results/                    # Paper figures and metrics

## Reproducing results

    conda create -n rl-buck python=3.11 -y
    conda activate rl-buck
    pip install fmpy gymnasium stable-baselines3 numpy matplotlib

    python train.py       # ~15 minutes
    python evaluate.py    # generates policy_response.png

## Architecture

    Vref ──► [Shaper] ──► Vref_shaped
                              │
                              ▼
                    [Habit Memory] ──► duty
                              │ (fallback)
                              ▼
                        [TD3 Policy] ──► duty
                              │
                              ▼
                       [Plant (FMU)] ──► Vout

## Key results

- Overshoot: 0.19%
- Steady-state error: 0.19%
- Generalization: Vref in [3, 7] with <1% error
- Habit memory saturation: ~100 points
- RL inference reduction: 85% at 16 KB of storage
