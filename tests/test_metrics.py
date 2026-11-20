import numpy as np
import pytest

from metrics import compute_metrics


def test_perfect():
    y = np.array([0.1, 0.2, 0.3])
    m = compute_metrics(y, y)
    assert m["r2"] == 1.0 and m["rmse"] == 0.0


def test_hand_calculation():
    m = compute_metrics(np.array([0.2, 0.2]), np.array([0.22, 0.18]))
    assert m["rmse"] == pytest.approx(0.02) and m["rrmse_pct"] == pytest.approx(10.0)


def test_missing_targets_are_ignored():
    m = compute_metrics(np.array([0.2, np.nan, 0.3]), np.array([0.2, 0.5, 0.3]))
    assert m["n"] == 2 and m["rmse"] == 0.0
