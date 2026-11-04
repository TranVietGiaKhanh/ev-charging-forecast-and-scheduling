"""
(TÙY CHỌN) Sinh dữ liệu GIẢ có cùng định dạng ACN-Data để chạy thử toàn bộ pipeline
trước khi có dữ liệu thật. KẾT QUẢ TRÊN DỮ LIỆU GIẢ KHÔNG DÙNG ĐỂ BÁO CÁO.

    python make_demo_data.py            # ghi ra data/demo/demo_acn.json
    python run_01_preprocess.py --raw-dir data/demo
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
TZ = "America/Los_Angeles"


def fmt(ts_local):
    """Đổi thời gian địa phương sang chuỗi GMT giống ACN-Data."""
    ts = ts_local.tz_localize(TZ).tz_convert("UTC")
    return ts.strftime("%a, %d %b %Y %H:%M:%S GMT")


def main(out):
    items = []
    sid = 0
    for day in pd.date_range("2019-03-01", "2019-03-31", freq="D"):
        weekend = day.dayofweek >= 5
        n = rng.poisson(8 if weekend else 38)
        for _ in range(n):
            if weekend:
                arr_h = rng.uniform(9, 17)
            elif rng.random() < 0.8:
                arr_h = np.clip(rng.normal(8.6, 1.3), 6, 14)
            else:
                arr_h = rng.uniform(12, 19)
            connect = day + pd.Timedelta(hours=float(arr_h))
            dur = float(np.clip(rng.normal(8.0, 2.5), 0.7, 14))
            disconnect = connect + pd.Timedelta(hours=dur)
            energy = float(np.clip(rng.gamma(2.5, 4.5), 1.0, 6.4 * dur * 0.95))
            charge_h = energy / rng.uniform(4.5, 6.4)
            done = connect + pd.Timedelta(hours=min(charge_h, dur))
            sid += 1
            item = {
                "_id": f"demo{sid}", "clusterID": "demo", "sessionID": f"demo_{sid}",
                "siteID": "0002", "spaceID": f"CA-{rng.integers(100, 400)}",
                "stationID": f"1-1-{rng.integers(1, 60)}", "timezone": TZ,
                "userID": int(rng.integers(1, 200)) if rng.random() < 0.6 else None,
                "connectionTime": fmt(connect), "disconnectTime": fmt(disconnect),
                "doneChargingTime": fmt(done), "kWhDelivered": round(energy, 3),
                "userInputs": None,
            }
            if item["userID"] is not None:
                item["userInputs"] = [{
                    "kWhRequested": round(energy * rng.uniform(1.0, 1.4), 2),
                    "minutesAvailable": int(dur * 60),
                    "requestedDeparture": fmt(disconnect),
                    "userID": item["userID"],
                }]
            items.append(item)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"_meta": {"total": len(items)}, "_items": items}, f)
    print(f"Đã ghi {len(items)} phiên GIẢ vào {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "demo" / "demo_acn.json"))
    main(ap.parse_args().out)
