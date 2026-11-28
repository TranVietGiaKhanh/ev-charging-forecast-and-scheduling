# Dự báo nhu cầu sạc và tối ưu phân bổ nguồn sạc xe điện – mã nguồn tuần 6–7

## 1. Cấu trúc thư mục (cần có)

```
ev_charging_project/
├── data/
│   ├── raw/             <- ĐẶT file JSON hoặc CSV tải từ ACN-Data vào đây (Caltech, 05/2018 - 04/2019)
│   ├── processed/       <- tự sinh: sessions.csv, hourly.csv, features.csv
│   └── demo/            <- (tùy chọn) dữ liệu giả để chạy thử
├── src/                 <- mã nguồn dùng chung (không chạy trực tiếp)
│   ├── __init__.py
│   ├── config.py        <- MỌI tham số (ngày chia train/test, P_max, giá điện, ...)
│   ├── preprocess.py    <- làm sạch, tổng hợp theo giờ, tạo đặc trưng
│   ├── forecasting.py   <- baseline, Ridge, Decision Tree, Random Forest, MLP
│   ├── session_model.py <- dự báo kWh cần và thời gian đỗ của từng xe (kịch bản "predicted")
│   ├── scheduling.py    <- bộ mô phỏng 15 phút, FCFS/EDF/LLF, LP cuốn chiếu, LP+forecast
│   ├── timeaxis.py      <- giờ thực (UTC) và giờ hiển thị (địa phương)
│   ├── stats_tests.py   <- kiểm định Diebold–Mariano, bootstrap khối theo ngày
│   └── backtest.py      <- kiểm thử nhiều giai đoạn (rolling-origin)
├── results/             <- tự sinh: bảng kết quả .csv
├── figures/             <- tự sinh: hình .png
├── run_01_preprocess.py
├── run_02_forecast.py
├── run_02b_session_model.py
├── run_03_schedule.py
├── run_04_stats.py
├── run_05_backtest.py
├── make_demo_data.py    <- (tùy chọn) sinh dữ liệu giả
└── requirements.txt
```

## 2. Cài đặt và chạy

```bash
pip install -r requirements.txt

# đặt file JSON/CSV ACN-Data vào data/raw/ rồi chạy lần lượt:
python run_01_preprocess.py      # -> data/processed/*.csv
python run_02_forecast.py        # -> results/forecast_*.csv, figures/forecast_*.png, mlp_training_curve_*.png
python run_02b_session_model.py  # -> results/session_predictions.csv, session_model_metrics.csv (~2 phút)
python run_03_schedule.py        # -> results/schedule_*.csv, figures/schedule_*.png (~3 phút, chạy song song 6 tiến trình)
python run_05_backtest.py        # -> results/backtest_*.csv (kiểm thử nhiều giai đoạn, ~20 phút)
python run_04_stats.py           # -> results/stats_*.csv (kiểm định thống kê, gồm cả gộp nhiều giai đoạn nếu đã chạy run_05)
```

Chạy thử khi chưa có dữ liệu thật (kết quả KHÔNG dùng để báo cáo):

```bash
python make_demo_data.py
python run_01_preprocess.py --raw-dir data/demo
python run_02_forecast.py
python run_03_schedule.py
```

Tùy chọn của bước 3: `--pmax 30 40` (các mức giới hạn trạm), `--skip-lp` (bỏ LP cho nhanh), `--workers 6` (số tiến trình song song; giảm nếu máy ít RAM).

## 3. Lấy dữ liệu ACN-Data

Site Caltech, **12 tháng từ 01/05/2018 đến 30/04/2019** (khoảng này đặt trong `src/config.py`: `DATA_START`, `DATA_END`).
Chỉ dùng giai đoạn trước 03/2020 (tức từ 05/2018 đến 02/2020): từ 03/2020 Caltech đóng cửa vì COVID-19 nên lượng xe sạc
tụt về gần 0 và làm méo dữ liệu.

Có hai cách: tải JSON từ ev.caltech.edu/dataset (dạng `{"_meta": ..., "_items": [ ... ]}`), hoặc chạy `python download_caltech.py` (cần `ACN_API_TOKEN`, tải từng
đoạn 30 ngày ra `data/raw/caltech_*.jsonl`; vẫn đọc được các file `sessions_caltech_*.csv`. Đặt tất cả vào `data/raw/`; các file `manifest_*` được bỏ qua, phiên trùng
giữa các đoạn được loại theo `sessionID`.

## 4. Cách mã hóa dữ liệu và huấn luyện

**Chuỗi theo giờ.** N = số xe bắt đầu cắm sạc trong giờ; L = tải trạm trung bình (kW). ACN-Data chỉ có tổng kWh của phiên
nên kWh được giả định cấp đều từ lúc cắm đến `doneChargingTime`, rồi cộng dồn vào từng giờ.

**Dự báo trước 24 giờ.** Mỗi hàng huấn luyện = một giờ t cần dự báo. Đặc trưng chỉ lấy từ dữ liệu cách t ít nhất 24 giờ
(lag 24h, 48h, 168h; trung bình 7 ngày cùng giờ) cộng giờ, thứ, cuối tuần, ngày lễ. 7 ngày đầu (01–07/05/2018) chỉ làm lịch sử
nên còn 8592 hàng = 7632 giờ train (08/05/2018–21/03/2019) + 960 giờ test (22/03–30/04/2019), nếu dữ liệu đủ 12 tháng.

**Chia dữ liệu.** Theo thời gian, không xáo trộn. Tập test chỉ dùng một lần để báo cáo.

**Chọn siêu tham số (Ridge, Decision Tree, Random Forest).** `GridSearchCV` + `TimeSeriesSplit(4)` trên tập train:
mỗi fold huấn luyện trên đoạn quá khứ và kiểm tra trên đoạn ngay sau nó; chọn bộ tham số có MAE trung bình nhỏ nhất,
sau đó huấn luyện lại trên toàn bộ tập train.

**MLP.** Vòng lặp huấn luyện tự viết (`MLPForecaster` trong `src/forecasting.py`): tách 20% cuối của train làm validation,
huấn luyện từng epoch bằng Adam, dừng sớm nếu val-loss không giảm sau 30 epoch, ghi nhớ epoch tốt nhất, huấn luyện lại trên
toàn bộ train đúng số epoch đó, lặp với 5 hạt giống và lấy trung bình dự báo. Đường cong huấn luyện: `figures/mlp_training_curve_*.png`.

**Baseline.** Seasonal Naive (lấy lag 24h hoặc lag 168h) và trung bình lịch sử theo (giờ, thứ) tính trên tập train; không cần huấn luyện.

## 5. Mô phỏng lập lịch

* Chạy trên **đúng khoảng kiểm thử** `SIM_START`–`SIM_END` (22/03–30/04/2019), nên dự báo dùng cho `LP+reserve` luôn là ngoài mẫu.
* Bước 15 phút; mỗi phiên có cổng riêng, công suất tối đa `PORT_MAX_KW`; sạc công suất không đổi
  (số xe đồng thời lớn nhất là 51, ít hơn 54 cổng của Caltech). Bước đầu/cuối của phiên chỉ có xe một phần thời gian
  (VD cắm 08:07 thì bước 08:00–08:15 có 8/15): công suất tối đa ở bước đó = 7 kW × tỷ lệ thời gian thực có mặt, nên không
  cấp điện trước lúc cắm hay sau lúc rút.
* "Sự thật" của mỗi phiên: xe rời lúc `disconnectTime`, ngừng nhận điện khi đã đủ `kWhDelivered`. Bộ lập lịch chỉ thấy xe sau khi xe cắm sạc.
* **Ba kịch bản thông tin** (`INFO_SCENARIOS`):
  * `oracle`: bộ lập lịch biết đúng kWh cần và giờ rời, là cận trên của lập lịch online.
  * `declared`: dùng `kWhRequested`/`requestedDeparture` nếu người dùng khai báo (~31% phiên); phiên không khai báo dùng
    trung vị kWh và thời gian đỗ của các phiên trước `SIM_START`. Khai báo kWh trung vị gấp ~1,67 lần thực nhận và
    57% khai báo giờ rời lệch quá 1 giờ, nên đây là kịch bản thực tế hơn.
  * `predicted`: kWh cần và thời gian đỗ do mô hình học máy dự báo lúc xe cắm (`run_02b_session_model.py`). Đặc trưng:
    giờ đến, thứ, khai báo (nếu có) và **lịch sử của chính người dùng** (`userID`, có ở ~87% phiên trong khoảng mô phỏng;
    chỉ dùng các phiên đã kết thúc trước lúc cắm). Mô hình `HistGradientBoostingRegressor` với hàm mất mát phân vị;
    phân vị (`SESSION_Q_KWH`, `SESSION_Q_DUR`) được chọn bằng mô phỏng LP trên 4 tuần cuối của giai đoạn huấn luyện
    (`SESSION_VAL_START`), không nhìn tập kiểm thử. Dự báo kWh hơi cao và giờ rời hơi sớm giúp LP thận trọng.
* Chiến lược: `FCFS-Uncontrolled` (không giới hạn trạm), `FCFS-Capped`, `EDF`, `LLF`, `LP-rolling` (giải lại LP mỗi 15 phút),
  `LP+reserve` (LP chừa công suất dự phòng γ·P_max ở các giờ dự báo có nhiều xe đến),
  `LP+forecast(RandomForest)` (mỗi giờ tương lai thêm một "xe ảo" gộp N̂ xe điển hình từ dự báo số xe đến; thiếu hụt của
  xe ảo bị phạt `BIG_M_FORECAST` < `BIG_M` nên xe thật luôn được ưu tiên), `LP+forecast(perfect)` (cùng cách nhưng dùng số xe
  đến thực tế: cận trên của lợi ích mà một dự báo số xe đến có thể mang lại), và `Offline-optimal`
  (một LP cho cả giai đoạn, biết trước mọi xe; là cận dưới chi phí, chỉ chạy ở kịch bản `oracle`).
* **Quét nhiều mức P_max** (`P_MAX_LIST`, mặc định 20–40 kW) để thấy đánh đổi giữa chi phí và năng lượng không đáp ứng.
* **Biểu giá khớp với dữ liệu: SCE TOU-EV-4** (Southern California Edison, biểu giá cho hệ thống sạc xe điện đo đếm riêng,
  đơn vị USD, theo giờ địa phương California). Đây là biểu giá nhóm tác giả ACN dùng với chính dữ liệu Caltech:
  [Lee et al., IEEE TSG 2021, mục VI-D, Bảng III](https://arxiv.org/abs/2012.02636) và file
  [`sce_tou_ev_4_march_2019.json` trong acnportal](https://github.com/zach401/acnportal/blob/master/acnportal/signals/tariffs/tariff_schedules/sce_tou_ev_4_march_2019.json)
  (biểu giá gốc của SCE: <https://www1.sce.com/NR/sc3/tm2/pdf/ce141-12.pdf>).

  | Ngày thường (thứ 2–6) | Mùa hè (01/06–30/09) | Mùa đông (01/10–31/05) |
  |---|---|---|
  | 0:00–8:00, 23:00–24:00 | $0,05623/kWh | $0,06087/kWh |
  | 8:00–12:00, 18:00–23:00 | $0,0925/kWh | $0,07492/kWh |
  | 12:00–18:00 (cao điểm) | $0,26668/kWh | $0,0869/kWh |
  | Cuối tuần, cả ngày | $0,05623/kWh | $0,06087/kWh |

  Phí công suất đỉnh **$15,51/kW/tháng** trên đỉnh 15 phút của tháng, được cộng vào chi phí của mọi chiến lược
  (`scheduling.daily_costs`): phí của tháng chia đều cho số ngày của tháng, tháng chỉ có một phần trong khoảng mô phỏng thì
  chỉ tính phần ngày có mặt. LP vẫn tối ưu theo giá điện năng, còn giới hạn P_max quyết định đỉnh. Khoảng kiểm thử
  (22/03–30/04/2019) và mọi cửa sổ kiểm thử nhiều giai đoạn (từ 05/10/2018) đều thuộc mùa đông. Hệ số phạt LP tính theo USD:
  `BIG_M` = 10 $/kWh, `BIG_M_FORECAST` = 1 $/kWh, `EPS_EARLY` = 1e-5 $/kWh mỗi bước.
* Kết quả: `results/schedule_summary.csv` (mọi kịch bản × P_max), `figures/schedule_tradeoff.png` (đánh đổi theo P_max),
  `schedule_comparison.png` và `schedule_load_profile_day.png` (ở `P_MAX_KW`).
* Kiểm tra bất biến: `python tests/test_scheduling.py` (không vượt P_max, không cấp quá nhu cầu, tối ưu biết trước ≤ LP cuốn chiếu,
  lịch sử người dùng không rò rỉ thông tin tương lai).

## 6. Kiểm định thống kê (`run_04_stats.py`)

* **Dự báo:** kiểm định Diebold–Mariano trên sai số tuyệt đối. Dự báo trước 24 giờ nên sai số các giờ liền nhau tương quan:
  phương sai Newey–West 24 độ trễ, hiệu chỉnh mẫu nhỏ Harvey–Leybourne–Newbold, phân phối t.
* **Dự báo và lập lịch:** bootstrap khối di động dạng vòng (moving block bootstrap): ghép các khối `BOOT_BLOCK_DAYS` = 7 ngày
  liên tiếp, 10.000 lần, khoảng tin cậy 95% theo phân vị. Lấy theo khối vì các ngày liền nhau phụ thuộc nhau (xe đỗ qua đêm,
  chu kỳ tuần). Ngày tính theo lịch địa phương. Cột `days_differ` cho biết bao nhiêu ngày hai chiến lược thực sự khác nhau.
* **Kiểm thử nhiều giai đoạn** (`run_05_backtest.py`, cửa sổ trong `BACKTEST_WINDOWS`): 7 cửa sổ liên tiếp 05/10/2018–30/04/2019
  (cửa sổ cuối = khoảng kiểm thử chính). Mỗi cửa sổ huấn luyện lại mọi mô hình chỉ trên dữ liệu trước nó (mô hình từng xe chọn
  phân vị trên 28 ngày ngay trước), rồi mô phỏng các chiến lược chính ở `BACKTEST_PMAX`. `run_04_stats.py` gộp mọi ngày của mọi
  cửa sổ để kiểm định (`results/stats_backtest.csv`, kèm số cửa sổ cùng chiều và nhỏ nhất/lớn nhất theo cửa sổ).

## 7. Trang demo

```
python make_demo_page.py        # -> docs/demo.html
```

Một file HTML duy nhất, mở bằng trình duyệt (bấm đúp), **không cần mạng**: dữ liệu và Chart.js (`docs/vendor/`, giấy phép MIT)
được nhúng sẵn. Gồm 4 phần: dữ liệu, dự báo (chọn đại lượng/tuần/mô hình), lập lịch (chọn P_max, kịch bản thông tin, ngày),
đánh đổi theo P_max. Cần chạy run_01–run_03 (cả run_02b) trước. Chạy lại `make_demo_page.py` mỗi khi kết quả thay đổi.

## 8. Lưu ý

* Số phiên sau làm sạch phụ thuộc quy tắc lọc trong `clean_sessions` (xem nhật ký in ra); chỉnh `MIN_KWH` và các bước lọc nếu cần.
* `cost_saving_vs_uncontrolled_%` chỉ có ý nghĩa khi so sánh các chiến lược cùng đáp ứng đủ năng lượng; hãy đọc kèm `pct_energy_served` và `cost_per_kWh_USD` (gồm phí công suất đỉnh; tách riêng ở `energy_cost_USD`, `demand_cost_USD`).
* Phạt `EPS_EARLY` trong LP chỉ là cách phá hòa: khi hai thời điểm cùng giá thì ưu tiên sạc sớm.
* Ngưỡng "nhiều xe đến" của `LP+reserve` là phân vị 75% của `N` trên tập train, nay tính trên ~11 tháng thay vì 2 tuần nên khác bản cũ.
* **Thời gian:** mọi mốc lưu theo giờ thực (UTC); chỉ mục các bảng theo giờ là `timestamp_utc`, cột `local_time`/`time_local`
  là giờ hiển thị. Giờ địa phương chỉ dùng cho đặc trưng lịch, khung giờ giá điện, gom ngày và hình vẽ. Ngày 04/11/2018 có
  25 giờ địa phương, 10/03/2019 có 23 giờ, không gộp và không thiếu (`python tests/test_preprocess.py`). Các mốc trong
  `config.py` viết theo giờ địa phương và được đổi bằng `timeaxis.utc()`. Trễ (lag) tính theo giờ thực.
* Mô hình từng xe và giá trị mặc định chỉ học từ các phiên đã **kết thúc** trước mốc (nhãn kWh, thời gian đỗ mới biết khi xe rời).
* `HOLIDAYS` mới chỉ có ngày lễ liên bang Mỹ; nếu biết Caltech nghỉ thêm ngày nào (ví dụ cuối tháng 12) thì thêm vào.
