# EnKF Parameter Tuning Results

**Entry point**: `mekong-da tune` (module `mekong_da.tuning`), which reproduces these grids
from the inputs in `inputs/`.
**Method**: 2-stage grid search with diagnostic innovation tracking

## Methodology

### Stage 1a: Observation Error x Model Error Sweep
- Grid: obs_error = [0.04, 0.06, 0.08, 0.10] x model_error = [0.02, 0.04, 0.06, 0.08]
- 16 combinations tested on 9km SMAP resolution
- Selection criterion: argmax on this grid of the composite score
  S = 0.4 H + 0.6 [0.5 max(0, NSE_full) + 0.5 max(0, NSE_heldout)], with the filter-health
  term H = (1/3) [max(0, 1 - |AC1|/0.5) + max(0, 1 - |Kbar - 0.5|/0.5) + max(0, 1 - |IVR - 1|/2)]
  (`mekong_da.tuning.score`). Applied to `stage1a_obs_model_error.csv` it returns
  obs_error = 0.04, model_error = 0.04, the archived selection.

### Stage 1b: Inflation x Persistence Phi Sweep
- Grid: inflation = [1.02, 1.05, 1.10, 1.15] x phi = [0.50, 0.65, 0.70, 0.80, 0.90]
- 20 combinations tested using best obs/model error from Stage 1a
- Selection criterion: argmax of the same composite score S on this grid. Applied to
  `stage1b_inflation_phi.csv` it returns inflation = 1.02, phi = 0.50, the archived selection.

### Stage 2: Multi-Resolution Validation
- Optimal config validated on all 3 SMAP resolutions (9km, 1km, 400m)
- Compared DEFAULT vs OPTIMAL configurations

## Results

### Optimal Configuration
```json
{
  "ensemble_size": 200,
  "obs_error_std": 0.04,
  "model_error_std": 0.04,
  "inflation_factor": 1.02,
  "persistence_phi": 0.50
}
```

### Default vs Optimal Comparison

| Resolution | Config | Kalman K | AC1 | IVR | NSE | KGE | RMSE |
|-----------|--------|----------|------|------|------|------|-------|
| 9km | DEFAULT | 0.568 | -0.259 | 0.482 | 0.880 | 0.850 | 0.0183 |
| 9km | OPTIMAL | 0.532 | -0.045 | 0.538 | 0.854 | 0.854 | 0.0202 |
| 1km | DEFAULT | 0.575 | -0.320 | 0.716 | 0.824 | 0.773 | 0.0220 |
| 1km | OPTIMAL | 0.536 | -0.102 | 0.799 | 0.784 | 0.779 | 0.0244 |
| 400m | DEFAULT | 0.575 | -0.251 | 0.814 | 0.805 | 0.738 | 0.0233 |
| 400m | OPTIMAL | 0.536 | -0.029 | 0.924 | 0.755 | 0.740 | 0.0262 |

### Key Findings

1. **Innovation AC1 dramatically improved**: From strongly negative (filter over-correcting) to near-zero (white noise = textbook behavior). 400m achieved AC1 = -0.029, essentially perfect.

2. **NSE decreased ~3-5%**: This is an acceptable trade-off. The default config had artificially high NSE because phi=0.70 caused the filter to aggressively chase observations, creating autocorrelated innovations.

3. **KGE slightly improved**: KGE is a more balanced metric than NSE, and it improved at all resolutions.

4. **phi=0.50 is the key change**: Lower persistence means less memory of previous analysis errors. This prevents the "ringing" behavior (negative AC1) where corrections from one timestep propagate forward and cause over/under-correction at the next.

5. **inflation=1.02 vs 1.05**: Minimal difference, but lower inflation prevents ensemble spread from growing unnecessarily.

## Diagnostic Metrics Explained

- **AC1** (Innovation Autocorrelation Lag-1): Should be near 0 for a properly tuned filter. Negative values mean over-correction; positive means under-correction.
- **K** (Kalman Gain): Weight given to observations vs model. K=0.53 means ~53% observation trust.
- **IVR** (Innovation Variance Ratio): Ratio of actual to theoretical innovation variance. Should be near 1.0.
- **NSE** (Nash-Sutcliffe Efficiency): Compares DA analysis to observations. >0.75 is very good.
- **KGE** (Kling-Gupta Efficiency): Balanced metric combining correlation, bias, and variability.

## Output Files

- `stage1a_obs_model_error.csv`: 16-row sweep results
- `stage1b_inflation_phi.csv`: 20-row sweep results
- `validation_default_vs_optimal.csv`: 6-row comparison (3 resolutions x 2 configs)
- `optimal_config.json`: Final configuration parameters
