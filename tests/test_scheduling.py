"""Kiểm tra nhanh các bất biến của mô phỏng lập lịch trên dữ liệu nhỏ tự tạo.
Chạy:  python -m pytest tests   hoặc   python tests/test_scheduling.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src import scheduling as S
from src import timeaxis as T


def make_sessions(n=40, seed=0):
    rng = np.random.default_rng(seed)
    t0 = T.utc(C.SIM_START)
    connect = t0 + pd.to_timedelta(rng.uniform(6, 30, n), unit="h")
    dur = rng.uniform(1, 9, n)
    return pd.DataFrame({
        "session_id": range(n), "connect": connect,
        "disconnect": connect + pd.to_timedelta(dur, unit="h"),
        "kwh_delivered": rng.uniform(2, 20, n),
        # một nửa có khai báo (sai lệch so với thực tế), một nửa không
        "kwh_requested": np.where(np.arange(n) % 2 == 0, rng.uniform(2, 30, n), np.nan),
        "requested_departure": np.where(np.arange(n) % 2 == 0,
                                        connect + pd.to_timedelta(rng.uniform(1, 9, n), unit="h"), pd.NaT),
    })


def check(P, r, p_max):
    assert np.all(r.load <= p_max + 1e-6), "vượt P_max"
    assert np.all(r.delivered <= P.E + 1e-6), "cấp quá nhu cầu thật"
    assert abs(r.load.sum() * S.DT - r.delivered.sum()) < 1e-6, "tổng tải != tổng năng lượng đã cấp"


def test_invariants_and_offline_is_lower_bound():
    s = make_sessions()
    p_max = 30.0
    s["kwh_pred"], s["dur_pred"] = 8.0, 3.0
    arr = pd.Series(1.5, index=pd.date_range(T.utc(C.SIM_START), periods=48, freq="h"))
    for info in ["oracle", "declared", "predicted"]:
        P = S.Problem(s, info, defaults=(8.0, 5.0))
        for r in [S.run_priority_policy(P, "edf", p_max), S.run_priority_policy(P, "llf", p_max),
                  S.run_lp_policy(P, p_max, verbose=False), S.run_lp_policy(P, p_max, verbose=False, arrivals=arr)]:
            check(P, r, p_max)
    P = S.Problem(s, "oracle")
    off, lp = S.run_offline_optimum(P, p_max), S.run_lp_policy(P, p_max, verbose=False)
    check(P, off, p_max)
    cost = lambda r: (r.load * S.DT * P.price).sum() + C.BIG_M * (P.E - r.delivered).sum()
    assert cost(off) <= cost(lp) + 1e-3, "tối ưu biết trước phải là cận dưới của LP cuốn chiếu"


def test_partial_steps_follow_real_presence():
    # xe cắm 08:07, rời 09:08 (61 phút), cần nhiều hơn mức có thể nạp -> tối đa 7 kW x 61/60 giờ
    t0 = T.utc(C.SIM_START)
    s = pd.DataFrame({"session_id": [0], "connect": [t0 + pd.Timedelta(hours=8, minutes=7)],
                      "disconnect": [t0 + pd.Timedelta(hours=9, minutes=8)], "kwh_delivered": [20.0],
                      "kwh_requested": [np.nan], "requested_departure": [pd.NaT]})
    P = S.Problem(s, "oracle", defaults=(8.0, 5.0))
    e_max = 7.0 * 61 / 60
    assert abs(P.E[0] - e_max) < 1e-9
    k0 = P.start[0]                                            # bước 08:00-08:15: chỉ có mặt 8/15
    for r in [S.run_priority_policy(P, "fcfs", np.inf), S.run_lp_policy(P, 30.0, verbose=False),
              S.run_offline_optimum(P, 30.0)]:
        assert abs(r.delivered[0] - e_max) < 1e-6, (r.name, r.delivered[0])
        assert r.load[k0] <= 7.0 * 8 / 15 + 1e-6 and r.load[:k0].sum() == 0
        assert r.load[P.end[0] - 1] <= 7.0 * 8 / 15 + 1e-6    # bước 09:00-09:15: chỉ có mặt 8/15
    Pd = S.Problem(s, "declared", defaults=(8.0, 0.5))          # bộ lập lịch tưởng xe rời sớm hơn
    r = S.run_lp_policy(Pd, 30.0, verbose=False)
    assert r.delivered[0] <= e_max + 1e-6 and r.load[k0] <= 7.0 * 8 / 15 + 1e-6


def test_tariff_bands():
    t = pd.DatetimeIndex(["2019-03-25 03:00", "2019-03-25 12:00", "2019-03-25 18:00", "2019-03-24 18:00"])  # T2, T2, T2, CN
    price, peak = S.tariff_bands(t, C.TARIFFS["QD2892_2026"])
    pr = C.TARIFFS["QD2892_2026"]["price"]
    assert list(price) == [pr["offpeak"], pr["normal"], pr["peak"], pr["normal"]]
    assert list(peak) == [False, False, True, False]


def test_session_history_has_no_leakage():
    from src.session_model import session_features
    t = pd.Timestamp("2019-01-07 16:00", tz="UTC")
    s = pd.DataFrame({"user_id": ["u", "u", "u", None],
                      "connect": [t, t + pd.Timedelta(hours=1), t + pd.Timedelta(days=1), t],
                      "disconnect": [t + pd.Timedelta(hours=5), t + pd.Timedelta(hours=6),
                                     t + pd.Timedelta(days=1, hours=4), t + pd.Timedelta(hours=2)],
                      "kwh_delivered": [10.0, 4.0, 6.0, 3.0], "kwh_requested": np.nan,
                      "requested_departure": pd.NaT}).sort_values("connect")
    s["duration_h"] = (s.disconnect - s.connect).dt.total_seconds() / 3600
    f = session_features(s)
    # phiên thứ 2 cắm khi phiên 1 chưa kết thúc -> chưa được thấy lịch sử; phiên thứ 3 thấy cả 2
    assert list(f.loc[[0, 1, 2, 3], "u_n"]) == [0, 0, 2, 0]
    assert f.loc[2, "u_med_kwh"] == 7.0 and f.loc[2, "u_last_kwh"] == 4.0


if __name__ == "__main__":
    test_tariff_bands(); test_invariants_and_offline_is_lower_bound(); test_session_history_has_no_leakage()
    test_partial_steps_follow_real_presence()
    print("OK")
