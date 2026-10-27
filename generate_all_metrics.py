import json

# ... inside your evaluation loop ...
if test_vref == PLOT_VREF:
    # Calculate metrics
    Vout_arr = np.array(Vout_mem)
    peak = Vout_arr.max()
    overshoot = (peak - PLOT_VREF) / PLOT_VREF * 100
    ss_err = abs(Vout_arr[-1] - PLOT_VREF) / PLOT_VREF * 100
    
    metrics = {
        "Vref": PLOT_VREF,
        "Overshoot_pct": float(overshoot),
        "SS_Error_pct": float(ss_err),
        "Final_Vout": float(Vout_arr[-1])
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics.json")