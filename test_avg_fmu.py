# test_avg_fmu.py
import numpy as np
from fmpy import read_model_description, simulate_fmu

fmu_path = "BuckConverter_Averaged.fmu"

md = read_model_description(fmu_path)
print("FMI type:", "Model Exchange" if md.modelExchange else "Co-Simulation")
print("Model ID:", md.coSimulation.modelIdentifier if md.coSimulation else "N/A")
print("\nInterface:")
for v in md.modelVariables:
    if v.causality in ('input', 'output'):
        print(f"  {v.name:15s}  causality={v.causality}")

input_data = np.array(
    [(0.0, 0.42, 5.0), (0.03, 0.42, 5.0)],
    dtype=[('time', np.float64), ('duty_cycle', np.float64), ('Vref', np.float64)]
)

result = simulate_fmu(
    fmu_path,
    start_time=0.0,
    stop_time=0.03,
    solver='CVode',
    input=input_data,
    output=['Vout'],
    relative_tolerance=1e-8,
    output_interval=1e-5,
)

vout = result['Vout']
t = result['time']
print(f"\nFinal Vout: {vout[-1]:.4f} V")
print(f"Vout range: [{vout.min():.4f}, {vout.max():.4f}] V")

# Show the trajectory at a few points
for i in range(0, len(t), len(t)//10):
    print(f"  t={t[i]*1000:.2f} ms  Vout={vout[i]:.4f} V")