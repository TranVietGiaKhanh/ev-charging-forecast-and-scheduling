"""
Hai loại thời gian trong đồ án (giữ tách bạch):
  * GIỜ THỰC  (UTC, có múi giờ): trục thời gian nội bộ - lưới giờ, bước 15 phút, thời lượng, trễ (lag) đều tính theo giờ thực,
    nên mỗi giờ xuất hiện đúng một lần kể cả ngày chuyển giờ mùa hè/đông (DST).
  * GIỜ HIỂN THỊ (giờ địa phương America/Los_Angeles): chỉ dùng cho đặc trưng lịch (giờ trong ngày, thứ, ngày lễ),
    khung giờ giá điện, chọn "ngày" khi thống kê và khi vẽ hình.
Các mốc trong config (DATA_START, TRAIN_END, SIM_START, ...) được viết theo GIỜ ĐỊA PHƯƠNG cho dễ đọc; dùng utc() để đổi.
"""
import pandas as pd

from . import config as C


def utc(x):
    """Mốc thời gian -> Timestamp UTC. Chuỗi/Timestamp không có múi giờ được hiểu là giờ địa phương của trạm."""
    ts = pd.Timestamp(x)
    return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize(C.TZ).tz_convert("UTC")


def local(t):
    """Giờ thực (Timestamp/DatetimeIndex/Series có múi giờ) -> giờ địa phương có múi giờ."""
    if isinstance(t, pd.Series):
        return t.dt.tz_convert(C.TZ)
    return t.tz_convert(C.TZ)


def local_naive(t):
    """Giờ địa phương KHÔNG kèm múi giờ - chỉ để hiển thị (trục hình vẽ, cột time_local trong CSV)."""
    lt = local(t)
    return lt.dt.tz_localize(None) if isinstance(lt, pd.Series) else lt.tz_localize(None)


def local_day(t):
    """Ngày theo lịch địa phương (Timestamp 00:00 không múi giờ) - dùng để gom theo ngày khi thống kê."""
    return local_naive(t).dt.normalize() if isinstance(t, pd.Series) else local_naive(t).normalize()
