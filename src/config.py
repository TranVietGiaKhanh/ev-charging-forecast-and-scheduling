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
MONTH_START = "2019-03-01 00:00"          # lấy các phiên bắt đầu từ mốc này ...
MONTH_END = "2019-04-01 00:00"            # ... đến trước mốc này (744 giờ)
MIN_KWH = 0.0                             # bỏ phiên có kWhDelivered <= MIN_KWH
HOLIDAYS = []                             # ví dụ ["2019-03-15"]; tháng 3/2019 không có

# Chia tập dữ liệu theo thời gian (KHÔNG xáo trộn)
# 7 ngày đầu (01/03-07/03) chỉ dùng làm "lịch sử" để tính lag 168 giờ.
TRAIN_START = "2019-03-08 00:00"          # 336 giờ huấn luyện
TRAIN_END = "2019-03-21 23:00"
TEST_START = "2019-03-22 00:00"           # 240 giờ kiểm thử
TEST_END = "2019-03-31 23:00"

# ----------------------------------------------------------------- dự báo
SEED = 42
CV_SPLITS = 4                             # số fold của TimeSeriesSplit khi chọn siêu tham số
RESERVE_MODEL = "RandomForest"            # mô hình dự báo N_t dùng để chừa công suất dự phòng

# ----------------------------------------------------------------- lập lịch
DT_H = 0.25                               # bước thời gian 15 phút = 0,25 giờ
PORT_MAX_KW = 7.0                         # công suất tối đa mỗi cổng (kW)
P_MAX_KW = 40.0                           # giới hạn công suất trạm (kW)
ENERGY_SOURCE = "delivered"               # "delivered": nhu cầu = kWhDelivered
                                          # "requested": dùng kWhRequested nếu có

PRICE_PEAK = 3000.0                       # đ/kWh trong khung cao điểm
PRICE_NORMAL = 1500.0                     # đ/kWh các giờ còn lại
PEAK_START = "09:30"
PEAK_END = "11:30"

# Quy hoạch tuyến tính
BIG_M = 100_000.0                         # hệ số phạt cho năng lượng thiếu hụt (đ/kWh)
EPS_EARLY = 0.5                           # phạt rất nhỏ theo thời gian: ưu tiên sạc sớm khi giá bằng nhau
HORIZON_MAX_STEPS = 96                    # nhìn trước tối đa 24 giờ

# Công suất dự phòng theo dự báo số xe đến: R_t = gamma * P_max
RESERVE_GAMMAS = [0.1, 0.2]
RESERVE_QUANTILE = 0.75                   # "nhiều xe đến" = vượt phân vị 75% của tập huấn luyện

PLOT_DAY = "2019-03-26"                   # ngày dùng để vẽ biểu đồ tải
