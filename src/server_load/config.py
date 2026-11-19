"""Single source of truth for every experiment setting (reproducibility, Phase F)."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"


@dataclass(frozen=True)
class ProjectConfig:
    """Experiment configuration shared by both teammates.

    Change values here, never inline in notebooks, so every figure in the
    report can be regenerated from one place.
    """

    machine_id: str = "m_1932"
    resample_rule: str = "1min"  # pandas offset alias; raw trace is ~10 s, irregular
    target: str = "cpu"
    horizon_steps: int = 1  # one-step-ahead forecast (Q1)
    lags: tuple[int, ...] = (1, 5, 15)  # in resampled steps (Q4)
    # Chronological split boundaries, as offsets from the trace start (Q6).
    train_end: str = "5D"
    val_end: str = "6D"
    random_seed: int = 42
    data_dir: Path = field(default=DATA_DIR)

    @property
    def machine_csv(self) -> Path:
        """Path of the per-machine CSV produced by scripts/extract_machine.py."""
        return self.data_dir / f"{self.machine_id}.csv"


CONFIG = ProjectConfig()
