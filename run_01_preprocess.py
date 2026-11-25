"""
Chạy:  python run_01_preprocess.py                      (dùng data/raw)
       python run_01_preprocess.py --raw-dir data/demo  (chạy thử với dữ liệu giả)
Sinh ra: data/processed/sessions.csv, hourly.csv, features.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from src import config as C
from src import preprocess as pp
from src import timeaxis as T


def main(raw_dir):
    C.DATA_PROC.mkdir(parents=True, exist_ok=True)

    print("1) Đọc dữ liệu thô")
    raw = pp.load_raw(raw_dir)

    print("\n2) Làm sạch -> bảng phiên sạc")
    sessions = pp.clean_sessions(raw)
    sessions.to_csv(C.DATA_PROC / "sessions.csv", index=False)

    print("\n3) Tổng hợp theo giờ (N: số xe đến, L: tải kW)")
    hourly = pp.build_hourly(sessions)
    hourly.to_csv(C.DATA_PROC / "hourly.csv")
    print(f"  chuỗi theo giờ: {len(hourly)} giờ thực (UTC), không trùng/không thiếu giờ khi chuyển DST "
          f"({hourly['local_time'].iloc[0]} -> {hourly['local_time'].iloc[-1]} giờ địa phương)")
    print(f"  N: trung bình {hourly['N'].mean():.2f} xe/giờ, lớn nhất {hourly['N'].max():.0f}")
    print(f"  L: trung bình {hourly['L'].mean():.2f} kW, đỉnh {hourly['L'].max():.1f} kW")

    print("\n4) Tạo đặc trưng (lag, rolling, lịch)")
    feats = pp.build_features(hourly)
    feats.to_csv(C.DATA_PROC / "features.csv")
    tr = feats.loc[T.utc(C.TRAIN_START):T.utc(C.TRAIN_END)]
    te = feats.loc[T.utc(C.TEST_START):T.utc(C.TEST_END)]
    print(f"  bảng đặc trưng: {len(feats)} hàng")
    print(f"  tập huấn luyện: {len(tr)} giờ ({tr['local_time'].iloc[0]:%d/%m} - {tr['local_time'].iloc[-1]:%d/%m})")
    print(f"  tập kiểm thử  : {len(te)} giờ ({te['local_time'].iloc[0]:%d/%m} - {te['local_time'].iloc[-1]:%d/%m})")
    print("\nĐã lưu vào", C.DATA_PROC)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(C.DATA_RAW))
    main(ap.parse_args().raw_dir)
