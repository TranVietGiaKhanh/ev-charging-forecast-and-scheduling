"""Kiểm tra trục thời gian quanh ngày chuyển giờ (DST).  Chạy:  python tests/test_preprocess.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import preprocess as pp


def _raw(rows):
    # (connectionTime, disconnectTime) dạng GMT như ACN-Data
    return pd.DataFrame([{"sessionID": f"s{i}", "spaceID": "A", "stationID": "A", "userID": None, "userInputs": None,
                          "connectionTime": c, "disconnectTime": d, "doneChargingTime": d, "kWhDelivered": 5.0}
                         for i, (c, d) in enumerate(rows)])


def test_dst_hours_are_neither_merged_nor_missing():
    raw = _raw([
        ("Sun, 04 Nov 2018 08:30:00 GMT", "Sun, 04 Nov 2018 08:50:00 GMT"),   # 01:30 PDT (lần 1)
        ("Sun, 04 Nov 2018 09:30:00 GMT", "Sun, 04 Nov 2018 09:50:00 GMT"),   # 01:30 PST (lần 2, cùng giờ đồng hồ)
        ("Sun, 04 Nov 2018 07:00:00 GMT", "Sun, 04 Nov 2018 12:00:00 GMT"),   # 00:00 PDT -> 04:00 PST: 5 giờ thực
    ])
    s = pp.clean_sessions(raw)
    assert sorted(s["duration_h"].round(6)) == [round(1 / 3, 6), round(1 / 3, 6), 5.0]           # thời lượng theo giờ thực
    h = pp.build_hourly(s)
    assert len(h) == 8760 and h.index.is_unique and (pd.Series(h.index).diff().dropna() == pd.Timedelta(hours=1)).all()
    lt = h["local_time"]
    fall = h[lt.dt.normalize() == pd.Timestamp("2018-11-04")]
    spring = h[lt.dt.normalize() == pd.Timestamp("2019-03-10")]
    assert len(fall) == 25 and len(spring) == 23                              # 25 / 23 giờ địa phương, không gộp, không trống
    one_am = fall[fall["local_time"].dt.hour == 1]
    assert len(one_am) == 2 and list(one_am["N"]) == [1.0, 1.0]               # hai xe 01:30 nằm ở hai giờ thực khác nhau
    assert 2 not in set(spring["local_time"].dt.hour)                         # 02:00 ngày 10/03/2019 không tồn tại
    f = pp.build_features(h)
    assert list(f.loc[one_am.index, "hour"]) == [1, 1]                        # đặc trưng lịch theo giờ địa phương


if __name__ == "__main__":
    test_dst_hours_are_neither_merged_nor_missing(); print("OK")
