"""
Toàn bộ tham số của đồ án nằm ở đây. Muốn đổi thí nghiệm thì chỉ sửa file này.
"""
from pathlib import Path

# ----------------------------------------------------------------- thư mục
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"          # file JSON tải từ ACN-Data
DATA_PROC = ROOT / "data" / "processed"   # sessions.csv, hourly.csv, features.csv
RESULTS = ROOT / "results"                # bảng kết quả (.csv)
FIGURES = ROOT / "figures"                # hình (.png)

# ----------------------------------------------------------------- dữ liệu
TZ = "America/Los_Angeles"                # múi giờ của trạm Caltech
# Khoảng dữ liệu dùng để huấn luyện/kiểm thử dự báo: 12 tháng, hoàn toàn trước COVID
# (Caltech đóng cửa từ 03/2020, lượng xe sạc tụt về gần 0 nên không dùng giai đoạn đó).
DATA_START = "2018-05-01 00:00"           # lấy các phiên bắt đầu từ mốc này ...
DATA_END = "2019-05-01 00:00"             # ... đến trước mốc này (8760 giờ)


MIN_KWH = 0.0                             # bỏ phiên có kWhDelivered <= MIN_KWH
# Ngày lễ liên bang Mỹ trong khoảng dữ liệu (Caltech có thể nghỉ thêm các ngày khác, ví dụ cuối tháng 12)
HOLIDAYS = ["2018-05-28", "2018-07-04", "2018-09-03", "2018-11-12", "2018-11-22",
            "2018-12-25", "2019-01-01", "2019-01-21", "2019-02-18"]

# Chia tập dữ liệu theo thời gian (KHÔNG xáo trộn)
# 7 ngày đầu (01/05-07/05/2018) chỉ dùng làm "lịch sử" để tính lag 168 giờ.
TRAIN_START = "2018-05-08 00:00"          # huấn luyện: 08/05/2018 - 21/03/2019
TRAIN_END = "2019-03-21 23:00"
TEST_START = "2019-03-22 00:00"           # kiểm thử: 22/03 - 30/04/2019 (nằm sau toàn bộ tập huấn luyện)
TEST_END = "2019-04-30 23:00"

# MÔ PHỎNG LẬP LỊCH (run_03) chạy trên đúng khoảng kiểm thử -> dự báo dùng cho LP+reserve luôn là ngoài mẫu
SIM_START = TEST_START                    # lấy các phiên cắm sạc từ mốc này ...
SIM_END = DATA_END                        # ... đến trước mốc này

# ----------------------------------------------------------------- dự báo
SEED = 42
CV_SPLITS = 4                             # số fold của TimeSeriesSplit khi chọn siêu tham số
RESERVE_MODEL = "RandomForest"            # mô hình dự báo N_t dùng để chừa công suất dự phòng

# ----------------------------------------------------------------- lập lịch
DT_H = 0.25                               # bước thời gian 15 phút = 0,25 giờ
PORT_MAX_KW = 7.0                         # công suất tối đa mỗi cổng (kW)
P_MAX_KW = 40.0                           # giới hạn công suất trạm dùng cho hình vẽ (kW)
P_MAX_LIST = [20.0, 25.0, 30.0, 35.0, 40.0]   # quét nhiều mức để thấy đánh đổi chi phí - năng lượng thiếu

# Bộ lập lịch biết gì về từng xe khi xe cắm sạc:
#   "oracle"  : biết đúng kWh cần (kWhDelivered) và giờ rời thực tế -> cận trên của những gì lập lịch online đạt được
#   "declared": dùng khai báo của người dùng (kWhRequested, requestedDeparture) nếu có; phiên không khai báo
#               dùng trung vị kWh và thời gian đỗ của các phiên TRƯỚC SIM_START. Xe vẫn rời đúng giờ thực tế
#               và ngừng nhận điện khi đã đủ kWhDelivered.
#   "predicted": kWh cần và thời gian đỗ do mô hình học máy dự báo lúc xe cắm (src/session_model.py), dựa trên
#               lịch sử của chính người dùng (userID), khai báo (nếu có), giờ đến và thứ.
INFO_SCENARIOS = ["oracle", "declared", "predicted"]

# Mô hình dự báo từng phiên (run_02b_session_model.py). Phân vị dự báo được chọn trên khoảng KIỂM ĐỊNH
# [SESSION_VAL_START, SIM_START) - 4 tuần cuối của giai đoạn huấn luyện - bằng mô phỏng LP, không nhìn tập kiểm thử.
SESSION_VAL_START = "2019-02-22 00:00"
SESSION_Q_KWH = [0.5, 0.7, 0.85, 0.95]  # phân vị cho kWh cần (cao hơn = thận trọng hơn)
SESSION_Q_DUR = [0.3, 0.15, 0.05]        # phân vị cho thời gian đỗ (thấp hơn = thận trọng hơn)
SESSION_VAL_PMAX = [30.0, 40.0]          # các mức P_max dùng khi chọn phân vị

# Biểu giá điện KHỚP VỚI DỮ LIỆU (trạm Caltech, California): Southern California Edison TOU-EV-4, biểu giá cho hệ thống
# sạc xe điện đo đếm riêng ($/kWh, theo giờ địa phương). Đây là biểu giá chính nhóm tác giả ACN dùng với dữ liệu Caltech:
#   - Lee et al., "Adaptive Charging Networks: A Framework for Smart EV Charging", IEEE Trans. Smart Grid 2021,
#     mục VI-D, Bảng III: https://arxiv.org/abs/2012.02636
#   - acnportal (ACN-Sim), file sce_tou_ev_4_march_2019.json (hiệu lực 01/03/2019):
#     https://github.com/zach401/acnportal/blob/master/acnportal/signals/tariffs/tariff_schedules/sce_tou_ev_4_march_2019.json
#   - Biểu giá gốc của SCE (liên kết ghi trong file trên): https://www1.sce.com/NR/sc3/tm2/pdf/ce141-12.pdf
# Ngày thường = thứ 2 - thứ 6; cuối tuần một giá cả ngày; không có quy tắc ngày lễ (giống acnportal).
# Phí công suất đỉnh $15,51/kW/tháng tính trên đỉnh 15 phút của tháng (xem scheduling.daily_costs).
TARIFFS = {
    "SCE_TOU_EV_4": {
        "name": "SCE TOU-EV-4",
        "seasons": [   # giờ bắt đầu từng bậc ngày thường (giờ địa phương) và giá $/kWh tương ứng
            {"name": "Mùa hè", "months": [6, 7, 8, 9], "weekday_start_h": [0, 8, 12, 18, 23],
             "weekday_price": [0.05623, 0.0925, 0.26668, 0.0925, 0.05623], "weekend_price": 0.05623},
            {"name": "Mùa đông", "months": [10, 11, 12, 1, 2, 3, 4, 5], "weekday_start_h": [0, 8, 12, 18, 23],
             "weekday_price": [0.06087, 0.07492, 0.0869, 0.07492, 0.06087], "weekend_price": 0.06087},
        ],
        "peak_hours": (12, 18),            # giờ cao điểm ngày thường (giá cao nhất của mùa)
        "demand_charge": 15.51,            # $/kW/tháng
    },
}
TARIFF = "SCE_TOU_EV_4"
CURRENCY = "USD"

# Quy hoạch tuyến tính
BIG_M = 10.0                              # phạt năng lượng thiếu hụt ($/kWh), gấp ~37 lần giá cao nhất
EPS_EARLY = 1e-5                          # phạt rất nhỏ mỗi bước ($/kWh): ưu tiên sạc sớm khi giá bằng nhau
                                          # (96 bước x 1e-5 < chênh lệch nhỏ nhất giữa hai bậc giá, ~0,012)
HORIZON_MAX_STEPS = 96                    # nhìn trước tối đa 24 giờ
# LP+forecast: phạt thiếu hụt của "xe ảo" (xe dự báo sẽ đến). Lớn hơn mọi chênh lệch giá điện (< 0,21 $/kWh) để LP
# sẵn sàng trả thêm tiền điện nhằm chừa chỗ, nhưng nhỏ hơn BIG_M để xe thật luôn được ưu tiên.
BIG_M_FORECAST = 1.0                      # $/kWh

# Công suất dự phòng theo dự báo số xe đến: R_t = gamma * P_max
RESERVE_GAMMAS = [0.1, 0.2]
RESERVE_QUANTILE = 0.75                   # "nhiều xe đến" = vượt phân vị 75% của tập huấn luyện

# Kiểm thử nhiều giai đoạn (run_05_backtest.py, rolling-origin): các cửa sổ LIÊN TIẾP theo giờ địa phương, cửa sổ cuối là
# khoảng kiểm thử chính. Với mỗi cửa sổ, mọi mô hình (dự báo, mô hình từng xe, giá trị mặc định) chỉ học từ dữ liệu TRƯỚC
# cửa sổ đó (từ TRAIN_START); phân vị của mô hình từng xe chọn trên BACKTEST_VAL_DAYS ngày ngay trước cửa sổ.
BACKTEST_WINDOWS = [("2018-10-05", "2018-11-02"), ("2018-11-02", "2018-11-30"), ("2018-11-30", "2018-12-28"),
                    ("2018-12-28", "2019-01-25"), ("2019-01-25", "2019-02-22"), ("2019-02-22", "2019-03-22"),
                    ("2019-03-22", "2019-05-01")]
BACKTEST_VAL_DAYS = 28
BACKTEST_PMAX = [25.0, 40.0]

# Kiểm định (run_04): bootstrap khối di động - độ dài khối 7 ngày giữ chu kỳ tuần và phụ thuộc giữa các ngày liền nhau
BOOT_BLOCK_DAYS = 7

PLOT_DAY = "2019-03-26"                   # ngày dùng để vẽ biểu đồ tải
