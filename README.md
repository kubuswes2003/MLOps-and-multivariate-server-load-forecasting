# TP1 – MLOps and multivariate server-load forecasting

## 👥 Work division

| Phase | **Person A – Carolina** (data & monitoring) | **Person B – Jakub** (models) |
|---|---|---|
| **A–B** Data & EDA | Q1 descriptive stats, target/horizon/sampling · Q3 time-series plots | Q2 choose ≥5 variables + justify · Q4 lag features |
| **C** Baselines | Q6 chronological split (`split.py`) | Q5 persistence + Ridge, derive objective (`models.py`) |
| **D** Advanced models | Q8 learning curves | Q7 Random Forest / Gradient Boosting · Q9 comparison table (`metrics.py`) |
| **E** Robustness | Q11 regime change · Q12 drift indicator + alert (`drift.py`) | Q10 predicted-vs-true scatter · error analysis |
| **F** Report | **Together:** Q13 deployment policy, final report (≤ 5 pages), fresh-clone reproducibility check | |

**Who owns what**

| | Person A – Carolina | Person B – Jakub |
|---|---|---|
| Notebook | `notebooks/01_eda.ipynb` | `notebooks/02_models.ipynb` |
| Code in `src/server_load/` | `data.py`, `split.py`, `drift.py` | `features.py`, `models.py`, `metrics.py` |
| Report sections | data, split, drift/monitoring | models, comparison, error analysis |

**Shared contract:** `load_machine(CONFIG)` returns a DataFrame indexed by time
(1-min steps) with columns `cpu, mem, mem_gps, mkpi, net_in, net_out, disk`.
Both of us build on this; don't change it without telling the other.

**Start here (partner):** follow [Setup](#setup-once-per-teammate) and
[Get the data](#get-the-data-once-per-teammate), then open `notebooks/02_models.ipynb`
and start with Q2 + Q4 in `features.py`.

**Git routine:** `git pull` → work → `git add .` → `git commit -m "..."` → `git push`.
Never commit anything from `data/` (it's ignored automatically).

---

Predict the next-step CPU usage of one Alibaba data-centre machine from its
multivariate operational measurements, then monitor what happens when the data
distribution changes (drift).

Data: [Alibaba cluster-trace-v2018](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018),
file `machine_usage.csv` (~4000 machines, 8 days).

## Project structure

```
├── data/                  # NOT in git — downloaded locally (see below)
├── src/server_load/
│   ├── config.py          # every experiment setting (machine, lags, split, seed)
│   ├── data.py            # read + clean + resample one machine
│   ├── features.py        # lag features, target            (Q4)
│   ├── split.py           # chronological split             (Q6)
│   ├── models.py          # persistence, Ridge, boosting    (Q5, Q7, Q8)
│   ├── metrics.py         # MAE, RMSE, R², normalised error (Q9)
│   └── drift.py           # regime change, drift alert      (Q11–Q12)
├── scripts/extract_machine.py
├── notebooks/
│   ├── 01_eda.ipynb       # Q1–Q3
│   └── 02_models.ipynb    # Q4–Q13
├── tests/
├── figures/               # plots saved for the report
└── report/                # final report (max 5 pages)
```

## Setup (once per teammate)

```bash
git clone https://github.com/kubuswes2003/MLOps-and-multivariate-server-load-forecasting.git
cd MLOps-and-multivariate-server-load-forecasting
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Get the data (once per teammate)

1. Download `machine_usage.tar.gz` (1.7 GB) into `data/`:

   ```bash
   curl -L -o data/machine_usage.tar.gz http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz
   ```

   SHA-256 (optional check): `3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a`

2. Extract one machine (streams the archive, nothing big is unpacked):

   ```bash
   python scripts/extract_machine.py --machine-id m_1932
   ```

   This writes `data/m_1932.csv`. The machine id must match `machine_id` in
   `src/server_load/config.py`.

## Reproduce

```bash
pytest
jupyter notebook notebooks/
```

All settings live in `src/server_load/config.py`; random seed = 42.

## Teamwork rules

- `git pull` before working, small commits, `git push` when you stop.
- Each person owns their notebook; shared logic goes in `src/` — tell the other
  before changing a shared function's signature.
- Fit scalers/imputers/models on **training data only**; never select a model on the test set.

Course: Artificial Intelligence – Master, Université d'Évry (TP1).
