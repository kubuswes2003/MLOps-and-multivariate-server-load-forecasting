"""Load, clean and resample one machine from machine_usage.csv (Alibaba v2018)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from server_load.config import ProjectConfig

logger = logging.getLogger(__name__)

# machine_usage.csv has no header row; order follows cluster-trace-v2018/schema.txt.
RAW_COLUMNS = [
    "machine_id",
    "time_stamp",
    "cpu",
    "mem",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk",
]
FEATURE_COLUMNS = RAW_COLUMNS[2:]

# Columns documented as normalised to [0, 100]; -1 and 101 flag invalid readings.
BOUNDED_COLUMNS = ["cpu", "mem", "mem_gps", "net_in", "net_out", "disk"]
VALID_MIN, VALID_MAX = 0.0, 100.0


def read_machine_csv(path: Path) -> pd.DataFrame:
    """Read the raw per-machine CSV extracted from machine_usage.tar.gz.

    Args:
        path: CSV with the 9 raw columns and no header.

    Returns:
        Raw DataFrame with named columns.

    Raises:
        FileNotFoundError: If the file is missing (run scripts/extract_machine.py first).
        ValueError: If the file is empty.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/extract_machine.py first (see README)."
        )
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    if df.empty:
        raise ValueError(f"{path} is empty — wrong machine id or failed extraction?")
    logger.info("Read %d raw rows from %s", len(df), path)
    return df


def clean_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Mark invalid readings as NaN, sort by time and drop duplicate timestamps.

    Args:
        df: Raw DataFrame from read_machine_csv.

    Returns:
        Cleaned DataFrame indexed by a TimedeltaIndex (time since trace start).
    """
    out = df.drop(columns="machine_id").copy()
    for col in BOUNDED_COLUMNS:
        invalid = (out[col] < VALID_MIN) | (out[col] > VALID_MAX)
        if invalid.any():
            logger.info("%s: %d invalid values set to NaN", col, int(invalid.sum()))
        out.loc[invalid, col] = np.nan

    out = out.sort_values("time_stamp").drop_duplicates("time_stamp", keep="last")
    # The trace only gives seconds since its start, so a timedelta is the honest index.
    out.index = pd.to_timedelta(out.pop("time_stamp"), unit="s")
    out.index.name = "elapsed"
    return out


def resample_usage(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Average irregular samples onto a regular grid; gaps stay NaN on purpose (Q3).

    Args:
        df: Cleaned DataFrame with a TimedeltaIndex.
        rule: Pandas offset alias, e.g. "1min".

    Returns:
        Regularly sampled DataFrame.
    """
    return df.resample(rule).mean()


def load_machine(config: ProjectConfig) -> pd.DataFrame:
    """Full pipeline: read, clean and resample the configured machine.

    Args:
        config: Project configuration.

    Returns:
        Regularly sampled, cleaned DataFrame with FEATURE_COLUMNS.
    """
    raw = read_machine_csv(config.machine_csv)
    return resample_usage(clean_usage(raw), config.resample_rule)
