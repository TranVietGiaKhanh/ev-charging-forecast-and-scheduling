"""
BƯỚC 3 - MÔ PHỎNG VÀ LẬP LỊCH SẠC (bước 15 phút)

Giả định đơn giản hóa (đã nêu trong báo cáo):
  * Mỗi xe có cổng riêng (không xếp hàng chờ cổng); công suất mỗi cổng tối đa PORT_MAX_KW.
  * Sạc với công suất không đổi (bỏ qua việc pin giảm dòng khi gần đầy).
  * Xe "khai báo đúng": nhu cầu năng lượng và giờ rời lấy từ dữ liệu của chính phiên đó.
    Bộ lập lịch chỉ nhìn thấy xe SAU khi xe cắm sạc, không biết trước xe sẽ đến.
"""
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from . import config as C

TOL = 1e-6
DT = C.DT_H


# ================================================================== dữ liệu bài toán
class Problem:
    def __init__(self, sessions: pd.DataFrame, energy_source=C.ENERGY_SOURCE):
        self.sessions = sessions.reset_index(drop=True)
        s = self.sessions
        self.t0 = pd.Timestamp(C.MONTH_START)
        step = pd.Timedelta(minutes=15)

        self.connect_min = ((s["connect"] - self.t0) / pd.Timedelta(minutes=1)).values
        self.start = np.floor((s["connect"] - self.t0) / step).astype(int).values   # bước đầu tiên có mặt
        self.end = np.ceil((s["disconnect"] - self.t0) / step).astype(int).values   # bước đầu tiên đã rời
        self.end = np.maximum(self.end, self.start + 1)
        self.n = len(s)
        self.pmax_i = np.full(self.n, C.PORT_MAX_KW)

        E = s["kwh_delivered"].values.astype(float)
        if energy_source == "requested":
            req = s["kwh_requested"].values.astype(float)
            E = np.where(np.isnan(req), E, req)
        cap = self.pmax_i * (self.end - self.start) * DT           # năng lượng tối đa có thể nạp được
        self.E_raw = E
        self.E = np.minimum(E, cap)                                # tránh yêu cầu bất khả thi về mặt vật lý
        self.n_capped = int((E > cap + 1e-9).sum())

        self.n_steps = int(self.end.max()) + 1
        self.step_time = pd.date_range(self.t0, periods=self.n_steps, freq="15min")
        tod = self.step_time.hour * 60 + self.step_time.minute
        h1, m1 = map(int, C.PEAK_START.split(":"))
        h2, m2 = map(int, C.PEAK_END.split(":"))
        self.is_peak = np.asarray((tod >= h1 * 60 + m1) & (tod < h2 * 60 + m2))
        self.price = np.where(self.is_peak, C.PRICE_PEAK, C.PRICE_NORMAL)
        self.step_hour = self.step_time.floor("h")                 # để ghép với dự báo theo giờ

        self.starts_at = {}
        for i, k in enumerate(self.start):
            self.starts_at.setdefault(int(k), []).append(i)


class Result:
    def __init__(self, name, load, delivered, solve_times=None, n_fallback=0):
        self.name, self.load, self.delivered = name, load, delivered
        self.solve_times = solve_times or []
        self.n_fallback = n_fallback


# ================================================================== heuristic
def run_priority_policy(P: Problem, rule: str, p_max: float, name=None) -> Result:
    """
    Cấp công suất lần lượt theo thứ tự ưu tiên cho tới khi hết công suất trạm:
      fcfs : xe cắm trước được ưu tiên
      edf  : xe có giờ rời sớm nhất được ưu tiên
      llf  : xe có độ trễ cho phép nhỏ nhất được ưu tiên, laxity = (d - t) - E_còn_lại / p_max_cổng
    p_max = inf nghĩa là không giới hạn trạm (Uncontrolled).
    """
    rem = P.E.copy()
    load = np.zeros(P.n_steps)
    active = []
    for k in range(P.n_steps):
        active += P.starts_at.get(k, [])
        active = [i for i in active if P.end[i] > k and rem[i] > TOL]
        if not active:
            continue
        a = np.array(active)
        if rule == "fcfs":
            order = a[np.argsort(P.connect_min[a], kind="stable")]
        elif rule == "edf":
            order = a[np.lexsort((P.connect_min[a], P.end[a]))]
        elif rule == "llf":
            lax = (P.end[a] - k) * DT - rem[a] / P.pmax_i[a]
            order = a[np.lexsort((P.connect_min[a], lax))]
        else:
            raise ValueError(rule)
        left = p_max
        for i in order:
            if left <= TOL:
                break
            p = min(P.pmax_i[i], rem[i] / DT, left)
            rem[i] -= p * DT
            load[k] += p
            left -= p
    return Result(name or rule.upper(), load, P.E - rem)


# ================================================================== quy hoạch tuyến tính
def _solve_lp_step(P: Problem, k, a, rem, p_max, R_step):
    """
    Giải LP cho các xe đang có mặt (tập a) từ bước k đến hết cửa sổ đỗ của chúng.
      biến  : p[i,t] >= 0 (kW), s[i] >= 0 (kWh thiếu hụt)
      min   : sum_t sum_i (giá_t + EPS*(t-k)) * p[i,t] * dt  +  M * sum_i s[i]
      s.t.  : sum_i p[i,t] <= P_max - R_t                 (mọi bước t)
              p[i,t] <= p_max_cổng                         (biên trên của biến)
              sum_t p[i,t]*dt + s[i] = E_còn_lại[i]        (mỗi xe)
              p[i,t] chỉ tồn tại với t trong [k, d_i)      (ngoài thời gian đỗ coi như bằng 0)
    Trả về công suất của BƯỚC HIỆN TẠI k cho từng xe, hoặc None nếu solver thất bại.
    """
    n = len(a)
    h_end = min(int(P.end[a].max()), k + C.HORIZON_MAX_STEPS)
    steps = h_end - k
    ends = np.minimum(P.end[a], h_end)
    w = ends - k                                            # số bước của từng xe (>= 1)
    offs = np.concatenate([[0], np.cumsum(w)])
    n_p = int(offs[-1])
    nv = n_p + n

    var_sess = np.repeat(np.arange(n), w)
    var_step = np.concatenate([np.arange(k, e) for e in ends])

    c = np.empty(nv)
    c[:n_p] = (P.price[var_step] + C.EPS_EARLY * (var_step - k)) * DT
    c[n_p:] = C.BIG_M

    cap_rhs = np.full(steps, float(p_max))
    if R_step is not None and steps > 1:
        cap_rhs[1:] -= R_step[k + 1:h_end]                  # chỉ chừa dự phòng cho các bước TƯƠNG LAI
    cap_rhs = np.maximum(cap_rhs, 0.0)
    A_ub = csr_matrix((np.ones(n_p), (var_step - k, np.arange(n_p))), shape=(steps, nv))

    rows = np.concatenate([var_sess, np.arange(n)])
    cols = np.concatenate([np.arange(n_p), n_p + np.arange(n)])
    vals = np.concatenate([np.full(n_p, DT), np.ones(n)])
    A_eq = csr_matrix((vals, (rows, cols)), shape=(n, nv))

    ub = np.concatenate([P.pmax_i[a][var_sess], np.full(n, np.inf)])
    bounds = np.column_stack([np.zeros(nv), ub])

    res = linprog(c, A_ub=A_ub, b_ub=cap_rhs, A_eq=A_eq, b_eq=rem[a], bounds=bounds, method="highs")
    if res.status != 0 or res.x is None:
        return None
    return res.x[offs[:-1]]


def run_lp_policy(P: Problem, p_max: float, R_step=None, name="LP", verbose=True) -> Result:
    """Cuốn chiếu: tại MỖI bước 15 phút giải lại LP với năng lượng còn thiếu, chỉ áp dụng nghiệm của bước hiện tại."""
    rem = P.E.copy()
    load = np.zeros(P.n_steps)
    active, times, n_fallback = [], [], 0
    for k in range(P.n_steps):
        active += P.starts_at.get(k, [])
        active = [i for i in active if P.end[i] > k and rem[i] > TOL]
        if not active:
            continue
        a = np.array(active)
        t1 = time.perf_counter()
        p = _solve_lp_step(P, k, a, rem, p_max, R_step)
        times.append(time.perf_counter() - t1)
        if p is None:                                        # dự phòng: nếu solver lỗi thì dùng EDF cho bước này
            n_fallback += 1
            p = np.zeros(len(a)); left = p_max
            for j in np.lexsort((P.connect_min[a], P.end[a])):
                p[j] = min(P.pmax_i[a[j]], rem[a[j]] / DT, left); left -= p[j]
        p = np.clip(p, 0, np.minimum(P.pmax_i[a], rem[a] / DT))
        if p.sum() > p_max:
            p *= p_max / p.sum()
        rem[a] -= p * DT
        load[k] = p.sum()
        if verbose and k % 480 == 0:
            print(f"    {name}: bước {k}/{P.n_steps}", flush=True)
    return Result(name, load, P.E - rem, times, n_fallback)


def build_reserve(P: Problem, nhat: pd.Series, thr: float, gamma: float, p_max: float):
    """R_t = gamma * P_max ở các giờ mà số xe đến DỰ BÁO >= thr; các giờ khác (hoặc chưa có dự báo) R_t = 0."""
    nh = nhat.reindex(P.step_hour).values
    return np.where(np.nan_to_num(nh, nan=-1.0) >= thr, gamma * p_max, 0.0)


# ================================================================== chỉ số đánh giá
def summarize(P: Problem, r: Result, p_max: float, step_mask, sess_mask):
    load = r.load[step_mask]
    kwh = (r.load * DT)[step_mask]
    cost = float((r.load * DT * P.price)[step_mask].sum())
    served = r.delivered[sess_mask]
    need = P.E[sess_mask]
    delivered_total = float(kwh.sum())
    return {
        "policy": r.name,
        "total_cost_VND": round(cost),
        "cost_per_kWh_VND": round(cost / delivered_total, 1) if delivered_total > 0 else np.nan,
        "peak_load_kW": round(float(load.max()), 2),
        "steps_over_Pmax": int((load > p_max + 1e-6).sum()),
        "peak_window_energy_kWh": round(float((r.load * DT)[step_mask & P.is_peak].sum()), 1),
        "pct_sessions_fully_charged": round(100 * float((served >= need - 0.01).mean()), 2),
        "pct_energy_served": round(100 * float(served.sum() / need.sum()), 2),
        "unmet_energy_kWh": round(float((need - served).sum()), 1),
        "mean_solve_time_s": round(float(np.mean(r.solve_times)), 4) if r.solve_times else np.nan,
        "solver_fallbacks": r.n_fallback,
    }
