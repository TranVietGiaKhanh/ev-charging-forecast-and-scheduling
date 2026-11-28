"""
BƯỚC 2b - DỰ BÁO NHU CẦU CỦA TỪNG XE lúc xe vừa cắm sạc: kWh cần và thời gian đỗ (giờ)

Đặc trưng (chỉ dùng thông tin CÓ tại lúc cắm):
  hour, dow            : giờ đến (số thực) và thứ
  decl_kwh, decl_dur   : khai báo của người dùng (NaN nếu không khai báo)
  u_n, u_med_*, u_last_*: lịch sử của CHÍNH người dùng đó (userID) - chỉ gồm các phiên đã KẾT THÚC trước lúc cắm
Mô hình: HistGradientBoostingRegressor với hàm mất mát phân vị (xử lý NaN sẵn, không cần điền giá trị thiếu).
Phân vị được chọn trên 4 tuần cuối của giai đoạn huấn luyện bằng chính mô phỏng LP (không nhìn tập kiểm thử):
dự báo giờ rời THẤP hơn / kWh CAO hơn trung vị giúp LP thận trọng hơn, vì xe rời sớm mà chưa đủ điện bị phạt rất nặng.
"""
import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from . import timeaxis as T
from . import scheduling as S

FEATS = ["hour", "dow", "decl_kwh", "decl_dur", "u_n", "u_med_kwh", "u_med_dur", "u_last_kwh", "u_last_dur"]
TARGETS = {"kwh": "kwh_delivered", "dur": "duration_h"}


def session_features(s: pd.DataFrame) -> pd.DataFrame:
    """s phải được sắp theo connect. Trả về bảng đặc trưng cùng chỉ mục với s."""
    f = pd.DataFrame(index=s.index)
    lc = T.local(s["connect"])                                   # giờ đến theo đồng hồ địa phương
    f["hour"] = lc.dt.hour + lc.dt.minute / 60
    f["dow"] = lc.dt.dayofweek
    kreq = pd.to_numeric(s["kwh_requested"], errors="coerce")
    f["decl_kwh"] = kreq.where(kreq > 0)
    dd = (pd.to_datetime(s["requested_departure"], utc=True) - s["connect"]).dt.total_seconds() / 3600
    f["decl_dur"] = dd.where(dd > 0)
    h = {c: np.full(len(s), np.nan) for c in FEATS[4:]}
    h["u_n"][:] = 0
    pos = {ix: p for p, ix in enumerate(s.index)}
    for _, g in s.dropna(subset=["user_id"]).groupby("user_id"):
        con, dis = g["connect"].values, g["disconnect"].values
        kwh, dur = g["kwh_delivered"].values, g["duration_h"].values
        for j, ix in enumerate(g.index):
            past = np.flatnonzero(dis[:j] < con[j])          # chỉ phiên đã kết thúc -> không rò rỉ thông tin
            if len(past) == 0:
                continue
            p = pos[ix]
            h["u_n"][p] = len(past)
            h["u_med_kwh"][p], h["u_med_dur"][p] = np.median(kwh[past]), np.median(dur[past])
            h["u_last_kwh"][p], h["u_last_dur"][p] = kwh[past[-1]], dur[past[-1]]
    for c, v in h.items():
        f[c] = v
    return f


def fit_predict(s, f, train_mask, pred_mask, q_kwh, q_dur):
    out = {}
    for key, q in (("kwh", q_kwh), ("dur", q_dur)):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
                                          min_samples_leaf=40, random_state=C.SEED)
        m.fit(f.loc[train_mask, FEATS], s.loc[train_mask, TARGETS[key]])
        out[key] = m.predict(f.loc[pred_mask, FEATS])
    return np.clip(out["kwh"], 0.5, None), np.clip(out["dur"], 0.25, None)


def _lp_objective(sess, t0):
    """Tổng (chi phí + BIG_M * kWh thiếu) của LP cuốn chiếu trên các mức SESSION_VAL_PMAX."""
    P = S.Problem(sess, "predicted", t0=t0)
    tot = 0.0
    for p_max in C.SESSION_VAL_PMAX:
        r = S.summarize(P, S.run_lp_policy(P, p_max, verbose=False), p_max)
        tot += r["total_cost_USD"] + C.BIG_M * max(r["unmet_energy_kWh"], 0.0)
    return tot


def choose_quantiles(s, f, val0, val1, verbose=False):
    """Chọn (q_kwh, q_dur) bằng mô phỏng LP trên khoảng kiểm định [val0, val1); mô hình chỉ học từ các phiên trước val0."""
    # nhãn (kWh, thời gian đỗ) chỉ biết khi phiên đã KẾT THÚC -> chỉ học từ phiên rút sạc trước mốc
    tr_val, va = s.disconnect < val0, (s.connect >= val0) & (s.connect < val1)
    rows = []
    for qk, qd in product(C.SESSION_Q_KWH, C.SESSION_Q_DUR):
        kp, dp = fit_predict(s, f, tr_val, va, qk, qd)
        obj = _lp_objective(s[va].assign(kwh_pred=kp, dur_pred=dp), val0)
        rows.append({"q_kwh": qk, "q_dur": qd, "val_objective": round(obj)})
        if verbose:
            print(f"  q_kwh={qk:.2f} q_dur={qd:.2f} -> mục tiêu kiểm định {obj:,.0f}", flush=True)
    best = pd.DataFrame(rows).sort_values("val_objective", kind="stable").iloc[0]   # hòa -> giữ cặp đầu tiên
    return float(best.q_kwh), float(best.q_dur), rows


def run():
    s = pd.read_csv(C.DATA_PROC / "sessions.csv",
                    parse_dates=["connect", "disconnect", "done_charging", "requested_departure"])
    s = s.sort_values("connect").reset_index(drop=True)
    f = session_features(s)
    val0, sim0, sim1 = (T.utc(x) for x in (C.SESSION_VAL_START, C.SIM_START, C.SIM_END))
    va = (s.connect >= val0) & (s.connect < sim0)
    tr, te = s.disconnect < sim0, (s.connect >= sim0) & (s.connect < sim1)
    print(f"Phiên huấn luyện: {int(tr.sum())} | kiểm định chọn phân vị: {int(va.sum())} | kiểm thử: {int(te.sum())}")
    print(f"  kiểm thử: có userID {s.loc[te, 'user_id'].notna().mean():.0%}, "
          f"có lịch sử người dùng {(f.loc[te, 'u_n'] > 0).mean():.0%}")

    # ---- 1) chọn phân vị bằng mô phỏng LP trên khoảng kiểm định
    qk, qd, rows = choose_quantiles(s, f, val0, sim0, verbose=True)
    print(f"  -> chọn q_kwh={qk}, q_dur={qd}")

    # ---- 2) huấn luyện lại trên toàn bộ giai đoạn trước SIM_START, dự báo cho khoảng mô phỏng
    kp, dp = fit_predict(s, f, tr, te, qk, qd)
    km, dm = fit_predict(s, f, tr, te, 0.5, 0.5)                   # bản trung vị: chỉ để so độ chính xác
    pd.DataFrame({"session_id": s.loc[te, "session_id"].values, "kwh_pred": kp.round(6), "dur_pred": dp.round(6)}   # 6 chữ số: cùng đầu vào với run_05 (tránh nhiễu 1e-14 khi ghi/đọc CSV làm LP chọn nghiệm khác)
                 ).to_csv(C.RESULTS / "session_predictions.csv", index=False)

    # ---- 3) độ chính xác so với cách "declared" (khai báo, thiếu thì dùng trung vị quá khứ)
    y_k, y_d = s.loc[te, "kwh_delivered"].values, s.loc[te, "duration_h"].values
    kdef, ddef = s.loc[tr, "kwh_delivered"].median(), s.loc[tr, "duration_h"].median()
    fk, fd = f.loc[te, "decl_kwh"].values, f.loc[te, "decl_dur"].values
    cand = {"Trung vị quá khứ": (np.full_like(y_k, kdef), np.full_like(y_d, ddef)),
            "Khai báo (thiếu -> trung vị)": (np.where(np.isnan(fk), kdef, fk), np.where(np.isnan(fd), ddef, fd)),
            "Mô hình (trung vị)": (km, dm),
            f"Mô hình (q_kwh={qk:g}, q_dur={qd:g}) - dùng cho lập lịch": (kp, dp)}
    met = pd.DataFrame([{"method": k, "MAE_kWh": round(np.mean(np.abs(a - y_k)), 3),
                         "MAE_duration_h": round(np.mean(np.abs(b - y_d)), 3),
                         "pct_leave_before_estimate": round(100 * np.mean(y_d < b), 1),
                         "pct_need_more_than_estimate": round(100 * np.mean(y_k > a), 1)}
                        for k, (a, b) in cand.items()])
    met.to_csv(C.RESULTS / "session_model_metrics.csv", index=False)
    (C.RESULTS / "session_model_choice.json").write_text(json.dumps(
        {"q_kwh": qk, "q_dur": qd, "validation": rows}, indent=2), encoding="utf-8")
    print("\nĐộ chính xác trên khoảng kiểm thử:")
    print(met.to_string(index=False))
    print("\nĐã lưu results/session_predictions.csv, session_model_metrics.csv, session_model_choice.json")
