# -*- coding: utf-8 -*-
"""
Audit + cleaning of machine.csv
- Cleans sensor errors (set to NaN)
- Saves data/machine_cleaned.csv (raw machine.csv is left untouched)
- Saves 3 simple plots to figures/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "machine.csv"
CLEAN_FILE = ROOT / "data" / "machine_cleaned.csv"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

df = pd.read_csv(RAW_FILE)

# ---------------------------------------------------------------
# CLEANING: percentage columns must be in [0, 100];
# anything else (-1, 101, ...) is a sensor error -> NaN
# ---------------------------------------------------------------
pct_cols = ["cpu", "mem", "net_in", "net_out", "disk_io"]
print("--- Cleaning: out-of-range values set to NaN ---")
for col in pct_cols:
    bad = (~df[col].between(0, 100)) & df[col].notna()
    print(f"{col:8s} abnormal values: {bad.sum()}")
    df.loc[bad, col] = np.nan

for col in ["mem_gps", "mkpi"]:
    print(f"{col:8s} negative values (not modified): {(df[col] < 0).sum()}")

df.to_csv(CLEAN_FILE, index=False)
print("\nSaved cleaned data to:", CLEAN_FILE)

# ---------------------------------------------------------------
# AUDIT
# ---------------------------------------------------------------
print("\nMachine:", df["machine_id"].iloc[0])
print("Rows:", len(df), "| duration (days):",
      round((df.time_stamp.max() - df.time_stamp.min()) / 86400, 2))

print("\n--- Descriptive statistics ---")
print(df.drop(columns="machine_id").describe().T)

print("\n--- Missing rate per column ---")
print(df.isna().mean())

print("\n--- Sampling interval (seconds) ---")
dt = df["time_stamp"].diff()
print(dt.describe())
print(dt.value_counts().head(8))
print("duplicate timestamps:", df["time_stamp"].duplicated().sum())
print("gaps > 60 s:", (dt > 60).sum())
print("gaps > 600 s:", (dt > 600).sum())

# ---------------------------------------------------------------
# 3 SIMPLE PLOTS
# ---------------------------------------------------------------
df["days"] = (df["time_stamp"] - df["time_stamp"].min()) / 86400

# Plot 1: CPU and memory over time (smoothed so the trend is visible)
plt.figure(figsize=(12, 5))
plt.plot(df["days"], df["cpu"].rolling(30, min_periods=1).mean(),
         label="CPU (%)")
plt.plot(df["days"], df["mem"].rolling(30, min_periods=1).mean(),
         label="Memory (%)")
plt.title("CPU and memory usage over time (smoothed)")
plt.xlabel("Days since start")
plt.ylabel("Usage (%)")
plt.ylim(0, 100)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "1_cpu_mem_over_time.png", dpi=120)

# Plot 2: typical values and spread of each percentage metric
plt.figure(figsize=(8, 5))
data = [df[c].dropna() for c in pct_cols]
plt.boxplot(data, labels=pct_cols, showfliers=False)
plt.title("Typical values of each metric (box = middle 50% of the data)")
plt.ylabel("Usage (%)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "2_metric_ranges.png", dpi=120)

# Plot 3: % of missing data per column
plt.figure(figsize=(8, 5))
missing = df[pct_cols + ["mem_gps", "mkpi"]].isna().mean() * 100
missing.plot.bar()
plt.title("Missing data per column")
plt.ylabel("% missing")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "3_missing_data.png", dpi=120)

print("\nFigures saved in:", FIG)
plt.show()