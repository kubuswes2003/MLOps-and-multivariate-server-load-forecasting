# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:50:54 2026

@author: CAROLINA
"""

# -*- coding: utf-8 -*-
"""
TP1 - Q1: descriptive statistics + target / horizon / sampling interval
Reads data/machine_cleaned.csv (produced by audit.py).
"""

from pathlib import Path
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "machine_cleaned.csv")
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)

# Selected variables (mem_gps and mkpi are ~80% missing, so excluded)
selected = ["cpu", "mem", "net_in", "net_out", "disk_io"]

# ---------------------------------------------------------------
# 1) Descriptive statistics of the selected variables
# ---------------------------------------------------------------
stats = df[selected].describe().T          # count, mean, std, min, quartiles, max
stats["missing_%"] = df[selected].isna().mean() * 100
stats = stats.round(2)
print("--- Descriptive statistics (selected variables) ---")
print(stats)
stats.to_csv(OUT / "q1_descriptive_stats.csv")

# ---------------------------------------------------------------
# 2) Sampling interval
# ---------------------------------------------------------------
dt = df["time_stamp"].diff().dropna()
interval = dt.mode()[0]                    # most frequent step = nominal interval
print("\n--- Sampling interval ---")
print("Nominal interval (seconds):", interval)
print("Share of steps equal to nominal interval: %.2f %%"
      % (100 * (dt == interval).mean()))
print(dt.value_counts().head(8))

# ---------------------------------------------------------------
# 3) Data sanity checks
# ---------------------------------------------------------------
duration = df["time_stamp"].max() - df["time_stamp"].min()
expected_rows = duration / interval + 1
print("\n--- Sanity checks ---")
print("Rows:", len(df))
print("Duration (days): %.2f" % (duration / 86400))
print("Expected rows if no gap: %d | coverage: %.2f %%"
      % (expected_rows, 100 * len(df) / expected_rows))
print("Sorted by time:", df["time_stamp"].is_monotonic_increasing)
print("Duplicate timestamps:", df["time_stamp"].duplicated().sum())
print("Gaps > 1 interval:", (dt > interval).sum())
print("Longest gap (seconds):", dt.max())
print("Machine ids in file:", df["machine_id"].unique())
print("Values outside [0, 100] left in selected columns:",
      int(((df[selected] < 0) | (df[selected] > 100)).sum().sum()))
print("Constant columns:", [c for c in selected if df[c].nunique() <= 1])