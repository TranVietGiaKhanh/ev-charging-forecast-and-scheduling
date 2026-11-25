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

# Biểu giá bán lẻ điện RIÊNG CHO SẠC XE ĐIỆN 3 bậc (đ/kWh, chưa gồm VAT), mức giá theo QĐ 2892/QĐ-BCT (áp dụng từ
# 01/11/2026); khung giờ theo QĐ 963/QĐ-BCT (22/04/2026): cao điểm 17:30-22:30 thứ 2 - thứ 7 (Chủ nhật không có),
# thấp điểm 00:00-06:00 mọi ngày, còn lại là bình thường.
#   Trạm giả định đấu nối HẠ ÁP (đến 1 kV): 54 cổng AC 7 kW, công suất trạm vài chục kW, lấy điện từ lưới hạ áp của tòa nhà
#   (không có trạm biến áp riêng) -> mặc định "QD2892_2026" dùng giá hạ áp: 1.620 / 2.700 / 4.212.
#   "QD2892_trung_ap": giá trung áp (trên 1 kV, khi trạm có trạm biến áp riêng) 1.534 / 2.549 / 3.758 - để phân tích độ nhạy.
# "TT16_cu": khung giờ cũ (cao điểm 09:30-11:30 và 17:00-20:00, thấp điểm 22:00-04:00), CÙNG mức giá hạ áp - để tách riêng
#   ảnh hưởng của khung giờ.
_PEAK, _OFF = [("17:30", "22:30")], [("00:00", "06:00")]
_PRICE_HA_AP = {"offpeak": 1620.0, "normal": 2700.0, "peak": 4212.0}
TARIFFS = {
    "QD2892_2026": {"price": _PRICE_HA_AP, "peak": _PEAK, "offpeak": _OFF},
    "QD2892_trung_ap": {"price": {"offpeak": 1534.0, "normal": 2549.0, "peak": 3758.0}, "peak": _PEAK, "offpeak": _OFF},
    "TT16_cu": {"price": _PRICE_HA_AP,
                "peak": [("09:30", "11:30"), ("17:00", "20:00")], "offpeak": [("22:00", "24:00"), ("00:00", "04:00")]},
}
TARIFF = "QD2892_2026"

# Quy hoạch tuyến tính
BIG_M = 100_000.0                         # hệ số phạt cho năng lượng thiếu hụt (đ/kWh)
EPS_EARLY = 0.5                           # phạt rất nhỏ theo thời gian: ưu tiên sạc sớm khi giá bằng nhau
HORIZON_MAX_STEPS = 96                    # nhìn trước tối đa 24 giờ
# LP+forecast: phạt thiếu hụt của "xe ảo" (xe dự báo sẽ đến). Lớn hơn mọi chênh lệch giá điện (< 2.600 đ/kWh) để LP
# sẵn sàng trả thêm tiền điện nhằm chừa chỗ, nhưng nhỏ hơn BIG_M để xe thật luôn được ưu tiên.
BIG_M_FORECAST = 10_000.0

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
