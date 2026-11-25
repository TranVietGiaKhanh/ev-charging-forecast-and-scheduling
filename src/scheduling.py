"""
BƯỚC 3 - MÔ PHỎNG VÀ LẬP LỊCH SẠC (bước 15 phút)

Giả định đơn giản hóa (đã nêu trong báo cáo):
  * Mỗi xe có cổng riêng (không xếp hàng chờ cổng); công suất mỗi cổng tối đa PORT_MAX_KW.
    (Số xe đồng thời lớn nhất trong dữ liệu là 51 < 54 cổng của Caltech.)
  * Sạc với công suất không đổi (bỏ qua việc pin giảm dòng khi gần đầy).
  * Bước đầu/cuối của mỗi phiên chỉ có xe một phần thời gian: công suất tối đa ở bước đó = PORT_MAX_KW x tỷ lệ thời gian
    xe thực sự có mặt (không cấp điện trước lúc cắm hay sau lúc rút).
  * "Sự thật" của mỗi phiên: xe rời lúc disconnectTime và ngừng nhận điện khi đã đủ kWhDelivered.
  * Bộ lập lịch chỉ nhìn thấy xe SAU khi xe cắm sạc, không biết trước xe sẽ đến. Nó biết gì về xe phụ thuộc
    kịch bản thông tin (config.INFO_SCENARIOS): "oracle" biết đúng sự thật, "declared" chỉ biết khai báo/ước lượng.
"""
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from . import config as C
from . import timeaxis as T

TOL = 1e-6
DT = C.DT_H
STEP = pd.Timedelta(minutes=15)


def _minutes(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def tariff_bands(step_time, tariff):
    """Trả về (giá đ/kWh, là_cao_điểm) cho từng bước. Khung giờ theo GIỜ ĐỊA PHƯƠNG: bước có múi giờ (giờ thực) được đổi
    sang giờ địa phương; bước không múi giờ coi như đã là giờ địa phương. Cao điểm chỉ áp dụng thứ 2 - thứ 7."""
    lt = T.local(step_time) if step_time.tz is not None else step_time
    tod = np.asarray(lt.hour * 60 + lt.minute)
    in_any = lambda bands: np.any([(tod >= _minutes(a)) & (tod < _minutes(b)) for a, b in bands], axis=0)
    is_peak = in_any(tariff["peak"]) & np.asarray(lt.dayofweek < 6)
    is_off = in_any(tariff["offpeak"]) & ~is_peak
    pr = tariff["price"]
    price = np.where(is_peak, pr["peak"], np.where(is_off, pr["offpeak"], pr["normal"]))
    return price, is_peak


# ================================================================== dữ liệu bài toán
class Problem:
    def __init__(self, sessions: pd.DataFrame, info="oracle", defaults=None, tariff=C.TARIFF, t0=None):
        """
        sessions : các phiên cắm sạc từ t0 (mặc định SIM_START) trở đi
        info     : "oracle", "declared" hoặc "predicted" (xem config.INFO_SCENARIOS);
                   "predicted" cần cột kwh_pred, dur_pred (do run_02b_session_model.py sinh ra)
        defaults : (kWh, số giờ đỗ) điển hình: dùng cho phiên KHÔNG khai báo ở kịch bản "declared"
                   và cho "xe ảo" của LP+forecast
        """
        self.sessions = s = sessions.reset_index(drop=True)
        self.info = info
        self.defaults = defaults
        self.t0 = T.utc(t0 or C.SIM_START)                        # trục mô phỏng = giờ thực (UTC)

        self.connect_min = ((s["connect"] - self.t0) / pd.Timedelta(minutes=1)).values
        self.start = np.floor((s["connect"] - self.t0) / STEP).astype(int).values   # bước đầu tiên có mặt
        self.end = np.ceil((s["disconnect"] - self.t0) / STEP).astype(int).values   # bước đầu tiên đã rời
        self.end = np.maximum(self.end, self.start + 1)
        self.n = len(s)
        self.pmax_i = np.full(self.n, C.PORT_MAX_KW)
        # Bước đầu và bước cuối thường chỉ có xe một phần thời gian (VD cắm 08:07 -> bước 08:00-08:15 chỉ có 8/15).
        # Công suất tối đa ở bước k = p_max_cổng * tỷ lệ thời gian xe THỰC SỰ có mặt, nên không cấp điện lúc xe chưa đến/đã rời.
        m = STEP / pd.Timedelta(minutes=1)
        disc_min = ((s["disconnect"] - self.t0) / pd.Timedelta(minutes=1)).values
        self.f_first = np.clip((np.minimum((self.start + 1) * m, disc_min) - self.connect_min) / m, 0, 1)
        self.f_last = np.clip((disc_min - np.maximum((self.end - 1) * m, self.connect_min)) / m, 0, 1)
        self.f_arrive = np.clip(((self.start + 1) * m - self.connect_min) / m, 0, 1)   # phần bước đầu bộ lập lịch biết

        # ---- sự thật: năng lượng xe nhận được tối đa
        E = s["kwh_delivered"].values.astype(float)
        cap = self.pmax_i * (disc_min - self.connect_min) / 60     # năng lượng tối đa nạp được trong thời gian thực có mặt
        self.E = np.minimum(E, cap)                                # tránh yêu cầu bất khả thi về mặt vật lý
        self.n_capped = int((E > cap + 1e-9).sum())

        # ---- những gì bộ lập lịch "thấy": E_hat (kWh cần) và end_hat (bước dự kiến rời)
        if info == "oracle":
            self.E_hat, self.end_hat = self.E.copy(), self.end.copy()
        else:
            if info == "declared":
                kwh_def, dur_def = defaults
                req = pd.to_numeric(s["kwh_requested"], errors="coerce").values
                self.E_hat = np.where(req > 0, req, kwh_def)
                dep = pd.to_datetime(s["requested_departure"], utc=True)
                dep = dep.where(dep > s["connect"], s["connect"] + pd.Timedelta(hours=dur_def))
            elif info == "predicted":
                self.E_hat = s["kwh_pred"].values.astype(float)
                dep = s["connect"] + pd.to_timedelta(s["dur_pred"], unit="h")
            else:
                raise ValueError(info)
            self.end_hat = np.maximum(np.ceil((dep - self.t0) / STEP).astype(int).values, self.start + 1)

        self.n_steps = int(max(self.end.max(), self.end_hat.max())) + 1
        self.step_time = pd.date_range(self.t0, periods=self.n_steps, freq="15min")   # giờ thực (UTC)
        self.price, self.is_peak = tariff_bands(self.step_time, C.TARIFFS[tariff])
        self.step_hour = self.step_time.floor("h")                 # để ghép với dự báo theo giờ

        self.starts_at = {}
        for i, k in enumerate(self.start):
            self.starts_at.setdefault(int(k), []).append(i)

    def avail(self, a, k):
        """Tỷ lệ thời gian các xe a THỰC SỰ có mặt trong bước k (sự thật; 1 ở các bước giữa)."""
        return np.where(k == self.start[a], self.f_first[a], np.where(k == self.end[a] - 1, self.f_last[a], 1.0))

    def plan_view(self, a, k, delivered):
        """Năng lượng còn thiếu và bước rời mà bộ lập lịch tin là đúng, cho các xe đang có mặt a tại bước k."""
        end = np.maximum(self.end_hat[a], k + 1)                   # quá giờ khai báo mà xe vẫn còn -> coi như rời ngay
        if self.info == "oracle":
            rem = self.E[a] - delivered[a]
        else:
            # khai báo đã hết nhưng xe vẫn nhận điện -> giả định còn cần thêm ít nhất 1 bước sạc
            rem = np.maximum(self.E_hat[a] - delivered[a], self.pmax_i[a] * DT)
        return rem, end


class Result:
    def __init__(self, name, load, delivered, solve_times=None, n_fallback=0):
        self.name, self.load, self.delivered = name, load, delivered
        self.solve_times = solve_times or []
        self.n_fallback = n_fallback


def _active(P, k, active, delivered):
    """Xe đang cắm (đúng theo sự thật) và vẫn còn nhận điện."""
    active += P.starts_at.get(k, [])
    return [i for i in active if P.end[i] > k and P.E[i] - delivered[i] > TOL]


# ================================================================== heuristic
def run_priority_policy(P: Problem, rule: str, p_max: float, name=None) -> Result:
    """
    Cấp công suất lần lượt theo thứ tự ưu tiên cho tới khi hết công suất trạm:
      fcfs : xe cắm trước được ưu tiên
      edf  : xe có giờ rời (theo bộ lập lịch biết) sớm nhất được ưu tiên
      llf  : xe có độ trễ cho phép nhỏ nhất được ưu tiên, laxity = (d - t) - E_còn_lại / p_max_cổng
    p_max = inf nghĩa là không giới hạn trạm (Uncontrolled). Xe ngừng nhận điện khi đủ năng lượng thật.
    """
    delivered = np.zeros(P.n)
    load = np.zeros(P.n_steps)
    active = []
    for k in range(P.n_steps):
        active = _active(P, k, active, delivered)
        if not active:
            continue
        a = np.array(active)
        rem_hat, end_hat = P.plan_view(a, k, delivered)
        if rule == "fcfs":
            order = np.argsort(P.connect_min[a], kind="stable")
        elif rule == "edf":
            order = np.lexsort((P.connect_min[a], end_hat))
        elif rule == "llf":
            lax = (end_hat - k) * DT - rem_hat / P.pmax_i[a]
            order = np.lexsort((P.connect_min[a], lax))
        else:
            raise ValueError(rule)
        left = p_max
        cap_i = P.pmax_i[a] * P.avail(a, k)
        for j in order:
            if left <= TOL:
                break
            i = a[j]
            p = min(cap_i[j], (P.E[i] - delivered[i]) / DT, left)
            delivered[i] += p * DT
            load[k] += p
            left -= p
    return Result(name or rule.upper(), load, delivered)


# ================================================================== quy hoạch tuyến tính
def _solve_lp(P: Problem, pmax, begins, ends, need, cap, k0, slack=C.BIG_M, f_begin=None, f_end=None):
    """
    LP chung cho một tập xe (công suất tối đa pmax_i); xe i được sạc ở các bước [begins_i, ends_i).
      biến  : p[i,t] >= 0 (kW), s[i] >= 0 (kWh thiếu hụt)
      min   : sum_i sum_t (giá_t + EPS*(t - begins_i)) * p[i,t] * dt  +  sum_i M_i * s[i]
      s.t.  : sum_i p[i,t] <= cap[t - k0]                 (mọi bước t)
              p[i,t] <= p_max_cổng * f[i,t]                (f = tỷ lệ có mặt: f_begin ở bước đầu, f_end ở bước cuối, còn lại 1)
              sum_t p[i,t]*dt + s[i] = need[i]             (mỗi xe)
    Trả về (x, var_sess, var_step) với x là công suất của từng biến p, hoặc None nếu solver thất bại.
    """
    n = len(need)
    w = ends - begins
    n_p = int(w.sum())
    nv = n_p + n
    var_sess = np.repeat(np.arange(n), w)
    var_step = np.concatenate([np.arange(b, e) for b, e in zip(begins, ends)])

    c = np.empty(nv)
    c[:n_p] = (P.price[var_step] + C.EPS_EARLY * (var_step - begins[var_sess])) * DT
    c[n_p:] = slack

    A_ub = csr_matrix((np.ones(n_p), (var_step - k0, np.arange(n_p))), shape=(len(cap), nv))
    rows = np.concatenate([var_sess, np.arange(n)])
    cols = np.concatenate([np.arange(n_p), n_p + np.arange(n)])
    vals = np.concatenate([np.full(n_p, DT), np.ones(n)])
    A_eq = csr_matrix((vals, (rows, cols)), shape=(n, nv))
    fr = np.ones(n_p)
    off, has = np.concatenate([[0], np.cumsum(w)[:-1]]).astype(int), w > 0
    if f_end is not None:
        fr[(off + w - 1)[has]] = f_end[has]
    if f_begin is not None:                                   # xe chỉ có 1 bước: lấy tỷ lệ nhỏ hơn
        fr[off[has]] = np.minimum(fr[off[has]], f_begin[has])
    bounds = np.column_stack([np.zeros(nv), np.concatenate([pmax[var_sess] * fr, np.full(n, np.inf)])])

    res = linprog(c, A_ub=A_ub, b_ub=cap, A_eq=A_eq, b_eq=need, bounds=bounds, method="highs")
    if res.status != 0 or res.x is None:
        return None
    return res.x[:n_p], var_sess, var_step


def run_lp_policy(P: Problem, p_max: float, R_step=None, name="LP", verbose=True, arrivals=None) -> Result:
    """
    Cuốn chiếu: tại MỖI bước 15 phút giải lại LP với thông tin bộ lập lịch đang có, chỉ áp dụng bước hiện tại.
    arrivals : (tùy chọn) dự báo số xe đến mỗi giờ (pd.Series theo giờ). Khi có, mỗi giờ tương lai trong tầm nhìn
               được thêm một "xe ảo" gộp n_h xe điển hình (P.defaults: kWh, giờ đỗ; công suất n_h * PORT_MAX_KW).
               Thiếu hụt của xe ảo bị phạt BIG_M_FORECAST < BIG_M nên LP không bao giờ hy sinh xe thật cho xe ảo,
               nhưng sẵn sàng trả thêm tiền điện để chừa chỗ cho xe sắp đến.
    """
    delivered = np.zeros(P.n)
    load = np.zeros(P.n_steps)
    active, times, n_fallback = [], [], 0
    H, sph = C.HORIZON_MAX_STEPS, int(round(1 / DT))         # sph = số bước mỗi giờ
    if arrivals is not None:
        hours = pd.date_range(P.t0, periods=P.n_steps // sph + 1, freq="h")
        n_fc = np.clip(arrivals.reindex(hours).fillna(0.0).values, 0, None)
        e_typ, d_typ = P.defaults
        d_steps = max(1, int(round(d_typ / DT)))
    for k in range(P.n_steps):
        active = _active(P, k, active, delivered)
        if not active:
            continue
        a = np.array(active)
        rem_hat, end_hat = P.plan_view(a, k, delivered)
        pm, beg, ends, need = P.pmax_i[a], np.full(len(a), k), np.minimum(end_hat, k + H), rem_hat
        slack = np.full(len(a), C.BIG_M)
        # tỷ lệ có mặt bộ lập lịch biết: oracle biết đúng cả lúc đến và lúc rời; kịch bản khác chỉ biết lúc đến
        if P.info == "oracle":
            fb_, fe_ = P.avail(a, k), np.where(ends == P.end[a], P.f_last[a], 1.0)
        else:
            fb_, fe_ = np.where(k == P.start[a], P.f_arrive[a], 1.0), np.ones(len(a))
        if arrivals is not None:
            j = np.arange(k // sph + 1, min((k + H - 1) // sph + 1, len(n_fc)))   # các giờ bắt đầu SAU bước k
            j = j[n_fc[j] > 0.01]
            fb = sph * j
            pm, beg = np.r_[pm, n_fc[j] * C.PORT_MAX_KW], np.r_[beg, fb]
            ends, need = np.r_[ends, np.minimum(fb + d_steps, min(k + H, P.n_steps))], np.r_[need, n_fc[j] * e_typ]
            slack = np.r_[slack, np.full(len(j), C.BIG_M_FORECAST)]
            fb_, fe_ = np.r_[fb_, np.ones(len(j))], np.r_[fe_, np.ones(len(j))]
        cap = np.full(int(ends.max()) - k, float(p_max))
        if R_step is not None:
            cap[1:] -= R_step[k + 1:k + len(cap)]               # chỉ chừa dự phòng cho các bước TƯƠNG LAI
        t1 = time.perf_counter()
        sol = _solve_lp(P, pm, beg, ends, need, np.maximum(cap, 0.0), k, slack, fb_, fe_)
        times.append(time.perf_counter() - t1)
        if sol is None:                                      # dự phòng: nếu solver lỗi thì dùng EDF cho bước này
            n_fallback += 1
            p = np.zeros(len(a)); left = p_max
            for j in np.lexsort((P.connect_min[a], end_hat)):
                p[j] = min(P.pmax_i[a[j]], left); left -= p[j]
        else:
            x, var_sess, var_step = sol
            p = x[var_step == k]                             # mỗi xe thật có đúng một biến ở bước k (xe ảo bắt đầu sau k)
        # sự thật: không vượt công suất theo thời gian thực có mặt, xe đầy thì ngừng nhận
        p = np.clip(p, 0, np.minimum(P.pmax_i[a] * P.avail(a, k), (P.E[a] - delivered[a]) / DT))
        if p.sum() > p_max:
            p *= p_max / p.sum()
        delivered[a] += p * DT
        load[k] = p.sum()
        if verbose and k % 960 == 0:
            print(f"    {name}: bước {k}/{P.n_steps}", flush=True)
    return Result(name, load, delivered, times, n_fallback)


def run_offline_optimum(P: Problem, p_max: float, name="Offline-optimal") -> Result:
    """
    Cận dưới về chi phí: MỘT bài LP cho cả giai đoạn, biết trước mọi xe sẽ đến, kWh cần và giờ rời thực tế.
    Không chiến lược online nào làm tốt hơn mức này (ở cùng mức năng lượng đáp ứng).
    """
    t1 = time.perf_counter()
    sol = _solve_lp(P, P.pmax_i, P.start, P.end, P.E, np.full(P.n_steps, float(p_max)), 0, f_begin=P.f_first, f_end=P.f_last)
    if sol is None:
        raise RuntimeError("LP tối ưu biết trước không giải được")
    x, var_sess, var_step = sol
    load = np.bincount(var_step, weights=x, minlength=P.n_steps)
    delivered = np.bincount(var_sess, weights=x * DT, minlength=P.n)
    return Result(name, load, delivered, [time.perf_counter() - t1])


def build_reserve(P: Problem, nhat: pd.Series, thr: float, gamma: float, p_max: float):
    """R_t = gamma * P_max ở các giờ mà số xe đến DỰ BÁO >= thr; các giờ khác (hoặc chưa có dự báo) R_t = 0."""
    nh = nhat.reindex(P.step_hour).values
    return np.where(np.nan_to_num(nh, nan=-1.0) >= thr, gamma * p_max, 0.0)


# ================================================================== chỉ số đánh giá
def summarize(P: Problem, r: Result, p_max: float):
    cost = float((r.load * DT * P.price).sum())
    delivered_total = float((r.load * DT).sum())
    return {
        "policy": r.name,
        "total_cost_VND": round(cost),
        "cost_per_kWh_VND": round(cost / delivered_total, 1) if delivered_total > 0 else np.nan,
        "peak_load_kW": round(float(r.load.max()), 2),
        "steps_over_Pmax": int((r.load > p_max + 1e-6).sum()),
        "peak_window_energy_kWh": round(float((r.load * DT)[P.is_peak].sum()), 1),
        "pct_sessions_fully_charged": round(100 * float((r.delivered >= P.E - 0.01).mean()), 2),
        "pct_energy_served": round(100 * float(r.delivered.sum() / P.E.sum()), 2),
        "unmet_energy_kWh": round(float((P.E - r.delivered).sum()), 1),
        "mean_solve_time_s": round(float(np.mean(r.solve_times)), 4) if r.solve_times else np.nan,
        "solver_fallbacks": r.n_fallback,
    }
