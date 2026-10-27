import numpy as np
from fmpy import read_model_description, simulate_fmu

fmu_path = "BuckConverter_Averaged.fmu"

input_data = np.array(
    [(0.0, 0.42, 12.0), (0.03, 0.42, 12.0)],
    dtype=[('time', np.float64), ('duty_cycle', np.float64), ('Vin', np.float64)]
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
print(f"Final Vout: {result['Vout'][-1]:.4f} V")