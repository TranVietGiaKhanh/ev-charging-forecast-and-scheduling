# Dự báo nhu cầu sạc và tối ưu phân bổ nguồn sạc xe điện – mã nguồn tuần 6–7

## 1. Cấu trúc thư mục (cần có)

```
ev_charging_project/
├── data/
│   ├── raw/             <- ĐẶT file JSON tải từ ACN-Data vào đây (Caltech, 03/2019)
│   ├── processed/       <- tự sinh: sessions.csv, hourly.csv, features.csv
│   └── demo/            <- (tùy chọn) dữ liệu giả để chạy thử
├── src/                 <- mã nguồn dùng chung (không chạy trực tiếp)
│   ├── __init__.py
│   ├── config.py        <- MỌI tham số (ngày chia train/test, P_max, giá điện, ...)
│   ├── preprocess.py    <- làm sạch, tổng hợp theo giờ, tạo đặc trưng
│   ├── forecasting.py   <- baseline, Ridge, Decision Tree, Random Forest, MLP
│   └── scheduling.py    <- bộ mô phỏng 15 phút, FCFS/EDF/LLF, LP cuốn chiếu
├── results/             <- tự sinh: bảng kết quả .csv
├── figures/             <- tự sinh: hình .png
├── run_01_preprocess.py
├── run_02_forecast.py
├── run_03_schedule.py
├── make_demo_data.py    <- (tùy chọn) sinh dữ liệu giả
└── requirements.txt
```

## 2. Cài đặt và chạy

```bash
pip install -r requirements.txt

# đặt file JSON ACN-Data vào data/raw/ rồi chạy lần lượt:
python run_01_preprocess.py      # -> data/processed/*.csv
python run_02_forecast.py        # -> results/forecast_*.csv, figures/forecast_*.png, mlp_training_curve_*.png
python run_03_schedule.py        # -> results/schedule_*.csv, figures/schedule_*.png
```

Chạy thử khi chưa có dữ liệu thật (kết quả KHÔNG dùng để báo cáo):

```bash
python make_demo_data.py
python run_01_preprocess.py --raw-dir data/demo
python run_02_forecast.py
python run_03_schedule.py
```

Tùy chọn của bước 3: `--pmax 40` (giới hạn trạm), `--skip-lp` (bỏ LP cho nhanh), `--energy-source requested`.

## 3. Lấy dữ liệu ACN-Data

Vào trang ACN-Data (ev.caltech.edu/dataset), chọn site Caltech, khoảng thời gian 01/03/2019 – 31/03/2019,
tải file JSON và đặt vào `data/raw/`. File có dạng `{"_meta": ..., "_items": [ ... ]}`.

## 4. Cách mã hóa dữ liệu và huấn luyện

**Chuỗi theo giờ.** N = số xe bắt đầu cắm sạc trong giờ; L = tải trạm trung bình (kW). ACN-Data chỉ có tổng kWh của phiên
nên kWh được giả định cấp đều từ lúc cắm đến `doneChargingTime`, rồi cộng dồn vào từng giờ.

**Dự báo trước 24 giờ.** Mỗi hàng huấn luyện = một giờ t cần dự báo. Đặc trưng chỉ lấy từ dữ liệu cách t ít nhất 24 giờ
(lag 24h, 48h, 168h; trung bình 7 ngày cùng giờ) cộng giờ, thứ, cuối tuần, ngày lễ. 7 ngày đầu (01–07/03) chỉ làm lịch sử
nên còn 576 hàng = 336 giờ train (08–21/03) + 240 giờ test (22–31/03).

**Chia dữ liệu.** Theo thời gian, không xáo trộn. Tập test chỉ dùng một lần để báo cáo.

**Chọn siêu tham số (Ridge, Decision Tree, Random Forest).** `GridSearchCV` + `TimeSeriesSplit(4)` trên tập train:
mỗi fold huấn luyện trên đoạn quá khứ và kiểm tra trên đoạn ngay sau nó; chọn bộ tham số có MAE trung bình nhỏ nhất,
sau đó huấn luyện lại trên toàn bộ tập train.

**MLP.** Vòng lặp huấn luyện tự viết (`MLPForecaster` trong `src/forecasting.py`): tách 20% cuối của train làm validation,
huấn luyện từng epoch bằng Adam, dừng sớm nếu val-loss không giảm sau 30 epoch, ghi nhớ epoch tốt nhất, huấn luyện lại trên
toàn bộ train đúng số epoch đó, lặp với 5 hạt giống và lấy trung bình dự báo. Đường cong huấn luyện: `figures/mlp_training_curve_*.png`.

**Baseline.** Seasonal Naive (lấy lag 24h hoặc lag 168h) và trung bình lịch sử theo (giờ, thứ) tính trên tập train; không cần huấn luyện.

## 5. Mô phỏng lập lịch

* Bước 15 phút; mỗi phiên có cổng riêng, công suất tối đa `PORT_MAX_KW`; sạc công suất không đổi.
* Nhu cầu năng lượng = `kWhDelivered`, giờ rời = `disconnectTime` (giả định khai báo đúng). Bộ lập lịch chỉ thấy xe sau khi xe cắm sạc.
* Chiến lược: `FCFS-Uncontrolled` (không giới hạn trạm), `FCFS-Capped`, `EDF`, `LLF`, `LP-rolling` (giải lại LP mỗi 15 phút),
  `LP+reserve` (LP có chừa công suất dự phòng γ·P_max ở các giờ dự báo có nhiều xe đến).
* Kết quả: `schedule_summary_all.csv` (toàn bộ tháng) và `schedule_summary_test.csv` (từ 22/03, nơi dự báo là ngoài mẫu).

## 6. Lưu ý

* Số phiên sau làm sạch phụ thuộc quy tắc lọc trong `clean_sessions` (xem nhật ký in ra). Nếu bạn dùng quy tắc khác để ra 954 phiên, chỉnh `MIN_KWH` và các bước lọc cho khớp.
* Với dữ liệu thật, hãy xem dòng "đỉnh tải khi KHÔNG điều khiển" để chọn `P_MAX_KW` hợp lý (mặc định 40 kW).
* `cost_saving_vs_uncontrolled_%` chỉ có ý nghĩa khi so sánh các chiến lược cùng đáp ứng đủ năng lượng; hãy đọc kèm `pct_energy_served` và `cost_per_kWh_VND`.
* Phạt `EPS_EARLY` trong LP chỉ là cách phá hòa: khi hai thời điểm cùng giá thì ưu tiên sạc sớm.
