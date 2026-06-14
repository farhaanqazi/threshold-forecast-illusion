"""Shared pytest fixtures. Adds src/ to the path and trains each model once per session."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pytest  # noqa: E402

from hcip import modeling as M  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return M.load_features()


@pytest.fixture(scope="session")
def breach_windows(df):
    """Walk-forward: breach model evaluated on each of the last 4 target months."""
    months = M.valid_target_months(df, "target_breach_next")[-4:]
    return [M.breach_fit_eval(df, m) for m in months]


@pytest.fixture(scope="session")
def breach_ho(df):
    """Single train/test holdout for calibration and slice analysis."""
    return M.breach_holdout(df)


@pytest.fixture(scope="session")
def demand_fva(df):
    return M.fva_forecast(df, "target_total_next", "total_waiting")


@pytest.fixture(scope="session")
def wait_fva(df):
    return M.fva_forecast(df, "target_pct_within_18_next", "pct_within_18wk", clip01=True)
