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

### ▶️ Start here – Carolina (Person A, Windows)

1. Do the [Setup](#setup-once-per-teammate) (Windows commands).
2. Move the downloaded archive to `data\machine_usage.tar.gz`, then create the machine file:
   ```bat
   python scripts\extract_machine.py --machine-id m_1932
   ```
   → creates `data\m_1932.csv` (a few MB). **Send this file to Jakub** (WhatsApp / Drive).
3. Open your notebook (Spyder can't run `.ipynb`, use Jupyter):
   ```bat
   jupyter notebook notebooks\01_eda.ipynb
   ```
4. Run cell 1 — it loads the already-cleaned data. Then:
   - **Q1** – `df.describe().T`; define target = `cpu`, horizon = 1 step (1 min),
     raw sampling ≈ 10 s → resampled to 1 min; explain why random shuffling leaks the future.
   - **Q3** – plot every variable over time; look for daily cycles, CPU saturation near 100 %,
     gaps (NaN), abnormal periods. Save figures to `figures\`.
5. **Q6 – `split.py` is urgent:** Jakub needs the chronological split to train models.

### ▶️ Start here – Jakub (Person B, macOS)

1. Do the [Setup](#setup-once-per-teammate). Put Carolina's file at `data/m_1932.csv`.
2. Write code that doesn't need data yet: **Q4** `features.py` → **Q5** `models.py` →
   **Q9** `metrics.py` (test on a small fake DataFrame). **Q2** text in `02_models.ipynb`.

### 📦 Data – why it's not on GitHub

`data/` is git-ignored: GitHub rejects files > 100 MB (the archive is 1.7 GB) and git keeps
every committed file forever. The repo stores the *recipe* (download link + extraction
script); each of us has the data locally. Cleaning is done in code by `load_machine()`,
so nobody edits data files by hand.

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

**macOS / Linux**

```bash
git clone https://github.com/kubuswes2003/MLOps-and-multivariate-server-load-forecasting.git
cd MLOps-and-multivariate-server-load-forecasting
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Windows (cmd)** — type one line at a time; use `cd /d` to switch drives (e.g. to `E:`)

```bat
cd /d "E:\Universidad\PS M2 SAAS"
git clone https://github.com/kubuswes2003/MLOps-and-multivariate-server-load-forecasting.git
cd MLOps-and-multivariate-server-load-forecasting
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The pip upgrade is required: old pip versions fail on `pip install -e .`
Check it works: `python -c "from server_load.config import CONFIG; print(CONFIG.machine_csv)"`.

First time only, set your git identity:
`git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`.

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
