# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:24:20 2026

@author: CAROLINA
"""

from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "machine.csv")

# disk_io: -1 and 101 are sensor errors -> NaN
bad = (~df["disk_io"].between(0, 100)) & df["disk_io"].notna()
print("abnormal disk_io values:", bad.sum())
df.loc[bad, "disk_io"] = np.nan

print("\nMachine:", df["machine_id"].iloc[0])
print("Rows:", len(df), "| duration (days):",
      round((df.time_stamp.max() - df.time_stamp.min()) / 86400, 2))

print("\n--- Q1: descriptive statistics ---")
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