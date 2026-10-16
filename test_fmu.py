import numpy as np
from fmpy import read_model_description, simulate_fmu

fmu_path = "BuckConverter_Averaged.fmu"

md = read_model_description(fmu_path)
print("FMI type:", "Model Exchange" if md.modelExchange else "Co-Simulation")
print("\n=== Interface ===")
for v in md.modelVariables:
    if v.causality in ('input', 'output'):
        print(f"  {v.name:15s}  causality={v.causality}")

input_data = np.array(
    [(0.0, 0.42, 5.0), (0.03, 0.42, 5.0)],
    dtype=[
        ('time',       np.float64),
        ('duty_cycle', np.float64),
        ('Vref',       np.float64),
    ]
)

result = simulate_fmu(
    fmu_path,
    start_time=0.0,
    stop_time=0.03,
    solver='CVode',
    input=input_data,
    output=['Vout'],
    relative_tolerance=1e-6,
    output_interval=1e-4,
)

vout = result['Vout']
t    = result['time']
print(f"\nFinal time:  {t[-1]:.4f} s")
print(f"Final Vout:  {vout[-1]:.4f} V")
print(f"Vout range: [{vout.min():.4f}, {vout.max():.4f}] V")