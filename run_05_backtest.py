"""Chạy:  python run_05_backtest.py [--workers 6]   (sau run_01; khoảng 30–40 phút)
Kiểm thử nhiều giai đoạn (rolling-origin) trên các cửa sổ config.BACKTEST_WINDOWS -> results/backtest_*.csv.
Sau đó chạy lại  python run_04_stats.py  để có kiểm định gộp mọi cửa sổ (results/stats_backtest.csv)."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # mỗi tiến trình 1 luồng BLAS, tránh hết bộ nhớ khi chạy song song
import argparse

from src import backtest

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    backtest.run(ap.parse_args().workers)
