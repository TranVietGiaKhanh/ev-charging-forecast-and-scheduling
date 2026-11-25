"""Chạy:  python run_02b_session_model.py   (cần chạy run_01_preprocess.py trước; chạy trước run_03)
Dự báo kWh cần và thời gian đỗ của từng xe -> kịch bản thông tin "predicted" của run_03."""
from src import session_model

if __name__ == "__main__":
    session_model.run()
