"""Chạy:  python run_04_stats.py   (sau run_02 và run_03, kể cả  run_03_schedule.py --tariff TT16_cu)
Kiểm định Diebold–Mariano cho dự báo và bootstrap theo ngày cho các kết luận lập lịch."""
from src import stats_tests

if __name__ == "__main__":
    stats_tests.run()
