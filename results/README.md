# Results

Paper figures are in the project root:
- `policy_response.png` — main RL tracking result
- `habit_memory_scaling.png` — memory size saturation
- `habit_hit_rate.png` — habit hit rate with noise
- `habit_tradeoff.png` — memory trade-off
- `test_vref3.png`, `test_vref7.png` — generalization tests

The frozen successful run is tagged in git:

    git checkout v1.0-paper-draft

That tag contains the exact buck_env.py, train.py, td3_buck_final.zip,
and plots used for the paper.
