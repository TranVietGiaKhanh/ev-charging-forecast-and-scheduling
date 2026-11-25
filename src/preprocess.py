"""
BƯỚC 1 - TIỀN XỬ LÝ
  JSON ACN-Data  ->  (a) bảng phiên sạc  (b) chuỗi tổng hợp theo giờ  (c) bảng đặc trưng

Thời gian (xem src/timeaxis.py): mọi mốc được lưu theo GIỜ THỰC (UTC). Lưới giờ là các giờ UTC liên tiếp nên mỗi giờ
xuất hiện đúng một lần, kể cả ngày chuyển giờ: ngày 04/11/2018 (về giờ mùa đông) có 25 giờ địa phương, ngày 10/03/2019
(sang giờ mùa hè) có 23 giờ. Giờ địa phương chỉ dùng cho đặc trưng lịch (giờ, thứ, ngày lễ) và cột hiển thị local_time.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import timeaxis as T


# ------------------------------------------------------------------ đọc dữ liệu
def _read_items(path: Path):
    """Đọc một file: JSON {"_items": [...]}, JSON dạng list, hoặc JSON Lines."""
    text = path.read_text(encoding="utf-8-sig")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(obj, dict):
        for key in ("_items", "items", "sessions", "data"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        raise ValueError("không thấy danh sách phiên sạc trong file")
    return obj


def _read_csv(path: Path) -> pd.DataFrame:
    """Đọc file CSV do script tải dữ liệu tạo ra (sessions_*.csv). Cột userInputs là chuỗi -> đổi lại thành list."""
    df = pd.read_csv(path)

    def parse(x):
        if not isinstance(x, str) or not x.strip():
            return None
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return None

    if "userInputs" in df.columns:
        df["userInputs"] = df["userInputs"].apply(parse)
    return df


def load_raw(raw_dir) -> pd.DataFrame:
    """Đọc mọi file .json (định dạng ACN-Data) và .csv (sessions_*.csv) trong raw_dir; bỏ qua file manifest_*.
    Nhiều file (ví dụ 13 đoạn 30 ngày) được nối lại; phiên trùng sessionID sẽ bị bỏ ở bước làm sạch."""
    raw_dir = Path(raw_dir)
    existing = sorted(p.name for p in raw_dir.iterdir()) if raw_dir.exists() else []
    files = [p for p in sorted(raw_dir.glob("*"))
             if p.is_file() and not p.name.lower().startswith("manifest")
             and p.suffix.lower() in (".json", ".jsonl", ".csv")]
    if not files:
        raise FileNotFoundError(
            f"Không thấy file .json hoặc .csv nào trong {raw_dir}.\n"
            f"Các file đang có trong thư mục: {existing}\n"
            "Hãy đặt file dữ liệu ACN-Data vào thư mục này.")
    frames = []
    for f in files:
        if f.suffix.lower() == ".csv":
            df = _read_csv(f)
        else:
            df = pd.DataFrame(_read_items(f))
        frames.append(df)
        print(f"  đọc {f.name}: {len(df)} phiên")
    return pd.concat(frames, ignore_index=True)


def _to_utc(s: pd.Series) -> pd.Series:
    """Chuỗi 'Fri, 01 Mar 2019 16:30:00 GMT' (hoặc ISO có múi giờ) -> Timestamp UTC (giờ thực)."""
    # format="mixed": đọc từng chuỗi riêng (RFC 1123 của JSON và ISO có múi giờ của CSV). Nếu để pandas tự đoán một
    # định dạng chung (pandas 3.x) thì phần lớn chuỗi kiểu "Fri, 01 Jun 2018 ..." bị hiểu thành thiếu giờ và bị loại âm thầm.
    return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")


def _last_user_input(x, key):
    """userInputs là list các dict (hoặc None); lấy giá trị `key` của khai báo cuối cùng."""
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[-1], dict):
        return x[-1].get(key)
    return None


# ------------------------------------------------------------------ làm sạch
def clean_sessions(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    log = [("Ban đầu", len(df))]

    if "sessionID" in df.columns:
        df = df.drop_duplicates(subset="sessionID")
        log.append(("Sau khi bỏ trùng sessionID", len(df)))

    for col in ["connectionTime", "disconnectTime", "doneChargingTime"]:
        df[col] = _to_utc(df[col])

    df = df.dropna(subset=["connectionTime", "disconnectTime", "kWhDelivered"])
    log.append(("Sau khi bỏ dòng thiếu giờ đến/giờ rời/kWh", len(df)))

    m = (df["connectionTime"] >= T.utc(C.DATA_START)) & (df["connectionTime"] < T.utc(C.DATA_END))
    df = df[m]
    log.append((f"Sau khi lọc phiên bắt đầu trong [{C.DATA_START}, {C.DATA_END})", len(df)))

    df = df[df["disconnectTime"] > df["connectionTime"]]
    log.append(("Sau khi bỏ phiên có giờ rời <= giờ đến", len(df)))

    df = df[df["kWhDelivered"].astype(float) > C.MIN_KWH]
    log.append((f"Sau khi bỏ phiên có kWhDelivered <= {C.MIN_KWH}", len(df)))

    # khai báo của người dùng (chỉ có ở một số phiên)
    ui = df["userInputs"] if "userInputs" in df.columns else pd.Series([None] * len(df), index=df.index)
    kwh_req = pd.to_numeric(ui.apply(lambda x: _last_user_input(x, "kWhRequested")), errors="coerce")
    req_dep = _to_utc(ui.apply(lambda x: _last_user_input(x, "requestedDeparture")))

    out = pd.DataFrame({
        "session_id": df["sessionID"] if "sessionID" in df.columns else df.index,
        "space_id": df.get("spaceID"),
        "station_id": df.get("stationID"),
        "user_id": df.get("userID"),                      # chỉ có từ khoảng 10/2018 trở đi
        "connect": df["connectionTime"],
        "disconnect": df["disconnectTime"],
        "done_charging": df["doneChargingTime"],
        "kwh_delivered": df["kWhDelivered"].astype(float),
        "kwh_requested": kwh_req,
        "requested_departure": req_dep,
    })
    # nếu không có doneChargingTime thì coi như sạc đến lúc rút phích
    out["done_charging"] = out["done_charging"].fillna(out["disconnect"])
    out["done_charging"] = out["done_charging"].clip(lower=out["connect"], upper=out["disconnect"])
    out["has_declaration"] = out["kwh_requested"].notna()
    out["duration_h"] = (out["disconnect"] - out["connect"]).dt.total_seconds() / 3600
    out = out.sort_values("connect").reset_index(drop=True)

    print("\nNhật ký làm sạch:")
    for name, n in log:
        print(f"  {name:<70s} {n:>6d}")
    print(f"  -> Số phiên cuối cùng: {len(out)} "
          f"(có khai báo kWhRequested: {int(out['has_declaration'].sum())})")
    return out


# ------------------------------------------------------------------ chuỗi theo giờ
def build_hourly(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    N : số xe BẮT ĐẦU cắm sạc trong giờ t
    L : tải trạm trung bình trong giờ t (kW).
        ACN-Data không có công suất theo thời gian, chỉ có tổng kWh, nên ta GIẢ ĐỊNH
        kWh được cấp đều từ lúc cắm đến lúc sạc xong (doneChargingTime),
        rồi cộng dồn phần năng lượng rơi vào từng giờ. Vì mỗi giờ dài 1 h nên kWh = kW trung bình.
    """
    t0 = T.utc(C.DATA_START)
    n_hours = int((T.utc(C.DATA_END) - t0) / pd.Timedelta(hours=1))           # 8760 giờ thực với 12 tháng
    idx = pd.date_range(t0, periods=n_hours, freq="h", name="timestamp_utc")

    N = sessions["connect"].dt.floor("h").value_counts().reindex(idx, fill_value=0).astype(float)

    energy = np.zeros(n_hours)
    for row in sessions.itertuples():
        s = (row.connect - t0) / pd.Timedelta(minutes=1)
        e = (row.done_charging - t0) / pd.Timedelta(minutes=1)
        if e <= s:
            e = s + 1.0                                   # tránh chia cho 0
        rate = row.kwh_delivered / (e - s)                # kWh mỗi phút
        for h in range(int(s // 60), int(np.ceil(e / 60))):
            if 0 <= h < n_hours:
                overlap = min(e, (h + 1) * 60) - max(s, h * 60)
                if overlap > 0:
                    energy[h] += rate * overlap
    return pd.DataFrame({"local_time": T.local_naive(idx), "N": N.values, "L": energy}, index=idx)


# ------------------------------------------------------------------ đặc trưng
def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Dự báo trước 24 giờ  ->  mọi đặc trưng chỉ dùng dữ liệu cách thời điểm cần dự báo >= 24 giờ:
      lag24, lag48, lag168 : giá trị cùng giờ 1 ngày / 2 ngày / 1 tuần trước
      roll7                : trung bình cùng giờ của 7 ngày gần nhất (lag24 ... lag168)
      hour, dow, is_weekend, is_holiday : đặc trưng lịch, lấy theo GIỜ ĐỊA PHƯƠNG (hành vi theo đồng hồ của người dùng)
    Trễ tính theo GIỜ THỰC (dịch 24/48/168 hàng trên lưới UTC). Hệ quả: riêng 1–7 ngày sau hai lần chuyển giờ, lag24/lag168
    trỏ tới giờ địa phương lệch 1 giờ (VD 8:00 hôm nay ứng với 7:00 hoặc 9:00 hôm qua theo đồng hồ).
    """
    df = hourly.copy()
    idx = T.local(df.index)
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_holiday"] = pd.Index(T.local_day(df.index)).isin(pd.to_datetime(C.HOLIDAYS)).astype(int)
    for y in ["N", "L"]:
        for lag in (24, 48, 168):
            df[f"{y}_lag{lag}"] = df[y].shift(lag)
        df[f"{y}_roll7"] = sum(df[y].shift(24 * d) for d in range(1, 8)) / 7
    return df.dropna()          # 168 giờ đầu không đủ lịch sử -> bị loại (7 ngày đầu của khoảng dữ liệu)
