"""Kiểm tra nhanh kiểm định DM và bootstrap theo ngày.  Chạy:  python tests/test_stats.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stats_tests import dm_test, boot_days, pct_better, diff


def test_dm_detects_better_forecast_and_handles_ties():
    rng = np.random.default_rng(0)
    y = rng.normal(size=960)
    good, bad = y + rng.normal(0, 0.5, 960), y + rng.normal(0, 1.5, 960)
    dbar, stat, p = dm_test(good - y, bad - y)
    assert dbar < 0 and stat < 0 and p < 0.01
    _, stat2, p2 = dm_test(good - y, good - y)            # hai dự báo giống hệt -> không kiểm định được
    assert np.isnan(stat2) and np.isnan(p2)
    _, _, p3 = dm_test(rng.normal(size=960), rng.normal(size=960))   # cùng chất lượng -> thường không bác bỏ
    assert p3 > 0.01


def test_bootstrap_ci_contains_estimate_and_sign():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.uniform(8, 10, 40), rng.uniform(10, 12, 40)])   # cột 0 luôn nhỏ hơn
    est, lo, hi, share, nd = boot_days(X, pct_better, B=2000)
    assert lo <= est <= hi and lo > 0 and share == 0 and nd == 40
    est, lo, hi, _, _ = boot_days(X, diff, B=2000)
    assert abs(est - (X[:, 1].sum() - X[:, 0].sum())) < 1e-9 and lo > 0


if __name__ == "__main__":
    test_dm_detects_better_forecast_and_handles_ties(); test_bootstrap_ci_contains_estimate_and_sign(); print("OK")
