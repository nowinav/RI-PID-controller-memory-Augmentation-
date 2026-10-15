# Memory Architecture Block

A 2-D lookup table based memory block for RL-controlled buck converters.
Replaces the traditional PID controller path.

## Interface

| Port | Direction | Type | Description |
|---|---|---|---|
| `Duty_Cycle` | Input | Real | Previous duty cycle (feedback) |
| `Error` | Input | Real | Control error (V_ref - V_out) |
| `Best_Duty_Cycle` | Output | Real | Optimal duty cycle from lookup table |

## Contents

- `Memory_Architecture.mo` — the reusable block
- `Test_Memory.mo` — standalone verification harness

## Usage

Instantiate in any model:

```modelica
Memory_Architecture mem;
connect(errorSignal, mem.Error);
connect(prevDuty,    mem.Duty_Cycle);
connect(mem.Best_Duty_Cycle, pwm.duty_in);
