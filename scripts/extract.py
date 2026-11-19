from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "machine_usage" / "machine_usage.csv"
OUT = ROOT / "data" / "machine.csv"
COLS = ["machine_id", "time_stamp", "cpu", "mem", "mem_gps",
        "mkpi", "net_in", "net_out", "disk_io"]
CHUNK = 2_000_000

# Pass 1: count rows per machine (reads only the first column)
counts = pd.Series(dtype="int64")
for i, chunk in enumerate(pd.read_csv(RAW, header=None, usecols=[0],
                                      names=["machine_id"], chunksize=CHUNK)):
    counts = counts.add(chunk["machine_id"].value_counts(), fill_value=0)
    if i % 10 == 0:
        print("pass 1, chunk", i, flush=True)

counts = counts.sort_values(ascending=False)
print(counts.head(10))

MACHINE = counts.index[0]
print("Selected machine:", MACHINE)

# Pass 2: keep only that machine
parts = []
for i, chunk in enumerate(pd.read_csv(RAW, header=None, names=COLS,
                                      chunksize=CHUNK)):
    parts.append(chunk[chunk.machine_id == MACHINE])
    if i % 10 == 0:
        print("pass 2, chunk", i, flush=True)

df = pd.concat(parts).sort_values("time_stamp").reset_index(drop=True)
df.to_csv(OUT, index=False)
print(df.shape)
print(df.head())