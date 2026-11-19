# Report notes – TP1 MLOps & server-load forecasting

Working notes for the final report (**max 5 pages incl. figures and tables**).
Fill in as we go; `TODO` = still open. Owner in brackets: **[C]** Carolina, **[J]** Jakub, **[C+J]** together.

---

## Grading reminder

| Criterion | Weight | Where it shows up in the report |
|---|---|---|
| Problem formulation & theory | 20 % | §1, §4 |
| Data preparation & leakage-free validation | 20 % | §2, §3 |
| Baseline & model implementation | 20 % | §4 |
| Experimental comparison & metrics | 15 % | §5 |
| Error analysis, robustness, interpretation | 15 % | §6, §7 |
| Reproducibility, figures, report quality | 10 % | §8, figures checklist |

**Page budget (suggestion):** §1–2 ≈ 1 page · §3–4 ≈ 1 page · §5–6 ≈ 1.5 pages · §7 ≈ 1 page · §8 ≈ 0.5 page.

---

## 1. Problem formulation [C+J]

- **Goal:** one-step-ahead forecast of CPU utilisation of one production machine from its
  multivariate operational measurements, then monitor the model under distribution change.
- **Formulation:** let $x_t \in \mathbb{R}^p$ collect CPU, memory, memory bandwidth, cache misses,
  network in/out and disk I/O at time $t$. We learn
  $$\hat y_{t+1} = f(x_t, x_{t-1}, \dots, x_{t-L+1}), \qquad y = \texttt{cpu}.$$
- **Target:** `cpu_util_percent` of machine `m_1932`, in [0, 100].
- **Horizon:** 1 step = 1 minute ahead (`horizon_steps = 1`).
- **Sampling interval:** raw trace ≈ every 10 s, irregular (TODO [C]: verify in Q1) → averaged to
  a regular 1-min grid.
- **Why no random shuffling (Q1):** consecutive samples are strongly autocorrelated; shuffling puts
  future observations in the training set, so the model "sees the future" and the error estimate
  is optimistically biased (leakage). All splits are chronological.

## 2. Data [C]

- **Source:** Alibaba Cluster Trace v2018, `machine_usage.csv` (~4000 machines, 8 days).
  https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
- **Subset:** single machine `m_1932`, extracted with `scripts/extract_machine.py`.
- **Columns:** `cpu, mem, mem_gps, mkpi, net_in, net_out, disk` (memory, bandwidth, network and disk
  are normalised to [0, 100] by Alibaba for confidentiality).
- **Cleaning (in code, `data.py`):** values outside [0, 100] (documented invalid codes −1 / 101) → NaN;
  sort by time; drop duplicate timestamps; resample to 1-min means; gaps kept as NaN (not filled)
  so they stay visible in Q3.
- **Q1 descriptive statistics:** TODO [C] – table (count, mean, std, min, quartiles, max) per variable.
- **Q3 observations:** TODO [C]
  - periodicity (daily cycle?):
  - saturation (CPU near 100 %?):
  - missing intervals:
  - abnormal regimes:

### Q2 – explanatory variables [J]

| Variable | Why it should relate to CPU load |
|---|---|
| `net_in` | more incoming requests → more work to process |
| `net_out` | responses / data sent out reflect completed work |
| `mem` | large jobs occupy memory and CPU at the same time |
| `mem_gps` | memory bandwidth rises when the CPU actively streams data |
| `mkpi` | cache misses per 1000 instructions → CPU stalls, efficiency of the load |
| `disk` | I/O-heavy batch jobs (co-located with online services) |
| past `cpu` | load is persistent ("sticky"), strongest single predictor |

TODO [J]: confirm with correlation / lagged-correlation once real data is available.

## 3. Features & validation split

### Q4 – lag features [J]
- Lags used: `(1, 5, 15)` steps = 1, 5, 15 minutes (`config.py`).
- `make_lag_features` adds `col_lagk = col.shift(k)` (past only); `make_target` = `cpu.shift(-1)`.
- **Bias–variance trade-off:** few lags → model can't see longer dynamics (bias, underfitting);
  many lags → more parameters, collinear and noisy inputs, fewer usable rows (variance, overfitting).
  TODO [J]: show validation MAE vs. lag window if time allows.

### Q6 – chronological split [C]
- Train: TODO (e.g. days 0–5) · Validation: TODO (day 5–6) · Test: TODO (days 6–8).
- Exact time ranges: TODO
- **Hyperparameters chosen on validation only;** test set used once, at the very end.
- All learned transforms (imputer, scaler) live inside sklearn `Pipeline` → fitted on train only.
- Rows with NaN lags at the start of each split are dropped (TODO: say how many).

## 4. Models

| Model | Why | Status |
|---|---|---|
| Persistence $\hat y_{t+1} = y_t$ | reference any model must beat | code ✅ |
| Ridge (ARX) | interpretable, strong for ~linear dynamics | code ✅ |
| HistGradientBoosting | nonlinear interactions, thresholds, saturation | code ✅ |
| Iterative model for Q8 | learning curves (boosting per-iteration, or small MLP) | TODO |

### Q5 – Ridge objective [J]
$$\min_{\beta_0,\beta}\ \sum_{t}\big(y_{t+1} - \beta_0 - x_t^\top\beta\big)^2 + \alpha\,\lVert\beta\rVert_2^2$$
Closed form (standardised inputs): $\hat\beta = (X^\top X + \alpha I)^{-1} X^\top y$.
$\alpha$ shrinks weights → lower variance, stabilises collinear lag features; chosen on validation.
Pipeline: median imputer → standard scaler → Ridge.

### Q7 – why a nonlinear model may capture saturation better [J]
CPU is bounded at 100 %: near saturation the response to more load is flat, not linear.
Trees split on thresholds ("if net_in > a and mem > b …"), so they model plateaus and regime
switches that a single linear combination cannot. HistGradientBoosting also handles NaN natively.
TODO [J]: tuned hyperparameters + values.

### Q8 – learning curves [C]
TODO: train/validation loss per iteration; diagnose under/overfitting.

## 5. Results [J]

### Q9 – comparison (validation, then test)
Metrics: MAE, RMSE, $R^2$, NRMSE = RMSE / std(y_true) (NRMSE < 1 ⇒ better than predicting the mean).

| Model | MAE val | RMSE val | R² val | NRMSE val | MAE test | RMSE test | R² test |
|---|---|---|---|---|---|---|---|
| Persistence | | | | | | | |
| Ridge (α = ?) | | | | | | | |
| HistGB | | | | | | | |

- Does the ranking change with the metric? (RMSE punishes large errors more than MAE.) TODO
- Sanity check on synthetic sine + noise (not for the report): Ridge MAE 2.04, HistGB 2.07,
  persistence 2.39 → pipeline works, models beat the baseline.

### Q10 – predicted vs. true scatter [J]
TODO: calibration (points around identity line?), dispersion, outliers, behaviour near 100 %.

## 6. Error analysis [J]
TODO: residual plot / error distribution; where are the big errors (spikes, saturation, gaps)?

## 7. Drift & MLOps monitoring [C]

### Q11 – regime change
- Real change found? TODO. Otherwise inject a synthetic one (`drift.inject_regime_change`,
  e.g. workload surge on `net_in` from day 6).
- Pre-change MAE: TODO · post-change MAE: TODO.

### Q12 – drift indicator
- Indicator: rolling MAE over window TODO (e.g. 1 h); optionally input-distribution distance (KS test).
- Alert threshold: mean + 3·std of the rolling error **on the validation period**. TODO: value.

### Q13 – deployment policy [C+J]
Draft:
- **Monitor:** (1) rolling MAE vs. threshold, (2) input drift score per feature,
  (3) fraction of time CPU is saturated / share of missing inputs.
- **Recalibrate** (refit on recent window, same model/hyperparameters) when the error alert fires
  but inputs look similar.
- **Retrain** (re-tune hyperparameters, possibly new features) when input drift is persistent.
- **Reject / fall back to persistence** when the model is worse than persistence over the window.

## 8. Reproducibility [C+J]
- Code: GitHub repo, Python 3.10, `requirements.txt` (pytest pinned < 9.2).
- One config: `src/server_load/config.py` (machine id, resampling, lags, split, seed = 42).
- Data recreated from the official source with one script; checksum in README.
- TODO: final check – fresh clone → setup → run notebooks → same figures.

---

## Figures checklist (≥ 4 required)

- [ ] Multivariate time series (all variables over time) – [C], Q3
- [ ] Learning curves (train vs. validation loss) – [C], Q8
- [ ] Prediction vs. truth over time (test period) – [J]
- [ ] Predicted-vs-true scatter with identity line – [J], Q10
- [ ] Drift / rolling-error monitoring with alert threshold – [C], Q12
- [ ] (optional) Residual distribution – [J]

Save all figures to `figures/` with clear names (e.g. `q3_timeseries.png`).
