"""
BƯỚC 4 - KIỂM ĐỊNH THỐNG KÊ CÁC KẾT LUẬN CHÍNH

1) Dự báo: kiểm định Diebold–Mariano (DM) cho sai số tuyệt đối. Dự báo trước 24 giờ nên sai số của các giờ liền nhau
   tương quan (tối đa 23 giờ) -> phương sai dùng Newey–West với 24 độ trễ, cộng hiệu chỉnh mẫu nhỏ
   Harvey–Leybourne–Newbold (HLN) và phân phối t(T-1).
2) Dự báo và lập lịch: bootstrap KHỐI DI ĐỘNG theo ngày (moving block bootstrap, dạng vòng): mỗi lần ghép các khối
   BOOT_BLOCK_DAYS ngày LIÊN TIẾP (mặc định 7 ngày = 1 tuần) cho đủ số ngày, tính lại chỉ số trên tổng các ngày được chọn;
   khoảng tin cậy 95% = phân vị 2,5% và 97,5%. Lấy theo khối thay vì từng ngày lẻ vì các ngày liền nhau phụ thuộc nhau
   (xe đỗ qua đêm, chu kỳ tuần), nên khoảng tin cậy không bị hẹp giả tạo.
   Ngày được tính theo LỊCH ĐỊA PHƯƠNG: chi phí gán theo ngày của bước sạc, năng lượng thiếu theo ngày xe cắm.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import scheduling as S
from . import timeaxis as T

B_BOOT = 10_000


# ================================================================== công cụ
def dm_test(e1, e2, h=24):
    """H0: E|e1| = E|e2|. Trả về (trung bình d = |e1|-|e2|, thống kê DM-HLN, p hai phía). d < 0: dự báo 1 tốt hơn."""
    d = np.abs(np.asarray(e1, float)) - np.abs(np.asarray(e2, float))
    T = len(d)
    dc = d - d.mean()
    gam = [dc[k:] @ dc[:T - k] / T for k in range(h)]
    var = gam[0] + 2 * sum((1 - k / h) * gam[k] for k in range(1, h))     # Newey–West (Bartlett) -> luôn >= 0
    if var <= 1e-12:
        return float(d.mean()), np.nan, np.nan
    stat = d.mean() / np.sqrt(var / T) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(d.mean()), float(stat), float(2 * stats.t.sf(abs(stat), df=T - 1))


def boot_days(X, f, B=B_BOOT, seed=C.SEED, block=None):
    """
    X : mảng (số ngày × k) các tổng theo ngày, THEO THỨ TỰ THỜI GIAN; f : hàm của vector tổng k phần tử (nhận mảng k × B).
    block : độ dài khối (ngày) của moving block bootstrap dạng vòng; 1 = bootstrap từng ngày độc lập.
    Trả về (ước lượng, CI 2,5%, CI 97,5%, tỷ lệ bootstrap có f <= 0, số ngày hai cột khác nhau).
    """
    X = np.asarray(X, float)
    n, L = len(X), min(block or C.BOOT_BLOCK_DAYS, len(X))
    n_diff = int((np.abs(X[:, 0] - X[:, 1]) > 1e-6).sum())
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(B, int(np.ceil(n / L))))
    idx = ((starts[:, :, None] + np.arange(L)) % n).reshape(B, -1)[:, :n]     # các khối L ngày liên tiếp, nối vòng
    vals = f(X[idx].sum(axis=1).T)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(f(X.sum(axis=0)[:, None])[0]), float(lo), float(hi), float(np.mean(vals <= 0)), n_diff


pct_better = lambda S: 100 * (1 - S[0] / S[1])      # % giảm của cột 0 so với cột 1 (chi phí, sai số, kWh thiếu)
diff = lambda S: S[1] - S[0]                         # cột 1 trừ cột 0 (dương: cột 0 tốt hơn)


# ================================================================== dữ liệu theo ngày
def _sched_daily(sfx):
    prof = pd.read_csv(C.RESULTS / f"schedule_load_profiles_all{sfx}.csv", parse_dates=["time"])
    sess = pd.read_csv(C.RESULTS / f"schedule_session_level_all{sfx}.csv", parse_dates=["connect"])
    sday = T.local_day(sess["connect"])                           # ngày theo lịch địa phương
    pday = T.local_day(prof["time"]).clip(upper=sday.max())      # xe đỗ qua đêm cuối cùng -> gộp vào ngày cuối
    days = pd.Index(sorted(sday.unique()))
    dch = C.TARIFFS[C.TARIFF]["demand_charge"]
    # chi phí theo ngày = tiền điện năng + phần phí công suất đỉnh của tháng phân bổ cho ngày đó
    cost = lambda k: S.daily_costs(pday, prof["price"], prof[k], dch).sum(axis=1).reindex(days, fill_value=0.0).values
    unmet = lambda k: (sess["E_kWh"] - sess[k]).clip(lower=0).groupby(sday).sum().reindex(days, fill_value=0.0).values
    return cost, unmet, len(days)


# ================================================================== các kiểm định
def forecast_tests():
    fc = pd.read_csv(C.RESULTS / "forecast_predictions_test.csv", index_col=0, parse_dates=True)
    day = T.local_day(fc.index)
    models = [c[2:] for c in fc.columns if c.startswith("N_") and c != "N_actual"]
    base = "SeasonalNaive_24h"
    dm_rows, boot_rows = [], []
    for y in ("N", "L"):
        err = {m: fc[f"{y}_{m}"] - fc[f"{y}_actual"] for m in models}
        mae = {m: float(err[m].abs().mean()) for m in models}
        learned = [m for m in models if not m.startswith(("SeasonalNaive", "HistAvg"))]
        best = min(learned, key=mae.get)
        second = min([m for m in learned if m != best], key=mae.get)
        for a, b in [(best, base), (best, "SeasonalNaive_168h"), (best, second)]:
            dbar, st, p = dm_test(err[a], err[b])
            dm_rows.append({"target": y, "model_A": a, "model_B": b, "MAE_A": round(mae[a], 3), "MAE_B": round(mae[b], 3),
                            "mean_abs_err_diff": round(dbar, 4), "DM_HLN_stat": round(st, 3), "p_value": round(p, 4)})
            X = np.column_stack([err[a].abs().groupby(day).sum(), err[b].abs().groupby(day).sum()])
            boot_rows.append(_row("Dự báo", f"{a} so với {b}", f"{y}, tập kiểm thử", "% giảm MAE", *boot_days(X, pct_better)))
    return pd.DataFrame(dm_rows), boot_rows


def _row(group, comparison, config, metric, est, lo, hi, p_le0, n_diff):
    # CI phải cách 0 rõ ràng: khi chỉ vài ngày khác nhau, cận dưới có thể bằng 0 về số học (1e-14) mà không có ý nghĩa
    sig = "có" if (lo > 1e-6 or hi < -1e-6) else "không"
    return {"group": group, "comparison": comparison, "config": config, "metric": metric, "estimate": round(est, 3),
            "CI95_low": round(lo, 3), "CI95_high": round(hi, 3), "boot_share_le_0": round(p_le0, 4),
            "days_differ": n_diff, "significant_95": sig}


def schedule_tests():
    rows = []
    RF = f"LP+forecast({C.RESERVE_MODEL})"
    for sfx, tname in (("", C.TARIFF),):
        if not (C.RESULTS / f"schedule_session_level_all{sfx}.csv").exists():
            continue
        cost, unmet, nd = _sched_daily(sfx)
        k = lambda info, p, pol: f"{info}/{p:g}/{pol}"
        # 1) LP so với sạc không điều khiển: % tiết kiệm chi phí (oracle, 40 kW)
        X = np.column_stack([cost(k("oracle", 40, "LP-rolling")), cost(k("oracle", 40, "FCFS-Uncontrolled"))])
        rows.append(_row("Lập lịch", "LP-rolling so với FCFS-Uncontrolled", f"{tname}, oracle, 40 kW", "% tiết kiệm chi phí",
                         *boot_days(X, pct_better)))
        # 2) LP+forecast so với LP: kWh thiếu (oracle, 20–30 kW) và dự báo RF so với hoàn hảo
        for p in (20, 25, 30):
            X = np.column_stack([unmet(k("oracle", p, RF)), unmet(k("oracle", p, "LP-rolling"))])
            rows.append(_row("Giá trị dự báo xe đến", f"{RF} so với LP-rolling", f"{tname}, oracle, {p} kW",
                             "kWh thiếu giảm được", *boot_days(X, diff)))
            X = np.column_stack([unmet(k("oracle", p, "LP+forecast(perfect)")), unmet(k("oracle", p, RF))])
            rows.append(_row("Giá trị dự báo xe đến", f"LP+forecast(perfect) so với {RF}", f"{tname}, oracle, {p} kW",
                             "kWh thiếu giảm được", *boot_days(X, diff)))
        if sfx:
            continue
        # 3) predicted so với declared (LP-rolling): % giảm kWh thiếu, mọi P_max
        for p in C.P_MAX_LIST:
            X = np.column_stack([unmet(k("predicted", p, "LP-rolling")), unmet(k("declared", p, "LP-rolling"))])
            rows.append(_row("Giá trị dự báo từng xe", "predicted so với declared (LP-rolling)", f"{tname}, {p:g} kW",
                             "% giảm kWh thiếu", *boot_days(X, pct_better)))
        # 4) LLF so với LP khi chỉ có thông tin dự báo
        X = np.column_stack([unmet(k("predicted", 40, "LLF")), unmet(k("predicted", 40, "LP-rolling"))])
        rows.append(_row("Lập lịch", "LLF so với LP-rolling", f"{tname}, predicted, 40 kW", "kWh thiếu giảm được",
                         *boot_days(X, diff)))
        # 5) LP+reserve so với LP (declared, 40 kW): chi phí và kWh thiếu
        for g in C.RESERVE_GAMMAS:
            pol = f"LP+reserve(g={g})"
            X = np.column_stack([cost(k("declared", 40, pol)), cost(k("declared", 40, "LP-rolling"))])
            rows.append(_row("Lập lịch", f"{pol} so với LP-rolling", f"{tname}, declared, 40 kW", "% tiết kiệm chi phí",
                             *boot_days(X, pct_better)))
            X = np.column_stack([unmet(k("declared", 40, pol)), unmet(k("declared", 40, "LP-rolling"))])
            rows.append(_row("Lập lịch", f"{pol} so với LP-rolling", f"{tname}, declared, 40 kW", "kWh thiếu giảm được",
                             *boot_days(X, diff)))
        print(f"  lập lịch: {nd} ngày, {B_BOOT} lần bootstrap")
    return rows


def backtest_tests():
    """Kiểm định GỘP mọi cửa sổ của run_05_backtest (các ngày liên tiếp) + kết quả riêng từng cửa sổ."""
    fd = C.RESULTS / "backtest_daily.csv"
    if not fd.exists():
        return None
    d = pd.read_csv(fd, parse_dates=["day"])
    fp = pd.read_csv(C.RESULTS / "backtest_forecast_predictions.csv", index_col=0, parse_dates=True)
    rows = []

    def add(group, comparison, config, metric, X, f, win):
        est, lo, hi, share, nd = boot_days(X, f)
        per = [float(f(X[win == w].sum(axis=0)[:, None])[0]) for w in pd.unique(win)]
        r = _row(group, comparison, config, metric, est, lo, hi, share, nd)
        r.update({"windows_positive": f"{sum(v > 0 for v in per)}/{len(per)}",
                  "window_min": round(min(per), 3), "window_max": round(max(per), 3)})
        rows.append(r)

    # ---- dự báo: DM trên toàn bộ giờ của mọi cửa sổ, bootstrap khối theo ngày
    day = T.local_day(fp.index)
    models = [c[2:] for c in fp.columns if c.startswith("N_") and c != "N_actual"]
    for y in ("N", "L"):
        err = {m: fp[f"{y}_{m}"] - fp[f"{y}_actual"] for m in models}
        mae = {m: float(err[m].abs().mean()) for m in models}
        best = min([m for m in models if not m.startswith(("SeasonalNaive", "HistAvg"))], key=mae.get)
        for a_, b_ in [(best, "SeasonalNaive_24h"), ("MLP", "RandomForest")]:
            dbar, st, pv = dm_test(err[a_], err[b_])
            g = pd.DataFrame({"a": err[a_].abs(), "b": err[b_].abs(), "w": fp["window"], "d": day}).groupby("d")
            X = np.column_stack([g.a.sum(), g.b.sum()])
            add("Dự báo", f"{a_} so với {b_}", f"{y}, {len(pd.unique(fp.window))} cửa sổ (DM p = {pv:.4f})", "% giảm MAE",
                X, pct_better, g.w.first().values)

    # ---- lập lịch
    def series(tariff, info, p, pol, col):
        x = d[(d.tariff == tariff) & (d["info"] == info) & (d.p_max_kW == p) & (d.policy == pol)].sort_values("day")
        return x.set_index("day")[col], x.set_index("day")["window"]
    claims = [
        ("Lập lịch", "LP-rolling so với FCFS-Uncontrolled", C.TARIFF, 40, ("oracle", "LP-rolling"), ("oracle", "FCFS-Uncontrolled"), "cost", "% tiết kiệm chi phí", pct_better),
        ("Giá trị dự báo từng xe", "predicted so với declared (LP-rolling)", C.TARIFF, 25, ("predicted", "LP-rolling"), ("declared", "LP-rolling"), "unmet", "% giảm kWh thiếu", pct_better),
        ("Giá trị dự báo từng xe", "predicted so với declared (LP-rolling)", C.TARIFF, 40, ("predicted", "LP-rolling"), ("declared", "LP-rolling"), "unmet", "% giảm kWh thiếu", pct_better),
        ("Lập lịch", "LLF so với LP-rolling", C.TARIFF, 40, ("predicted", "LLF"), ("predicted", "LP-rolling"), "unmet", "kWh thiếu giảm được", diff),
    ]
    for tf in (C.TARIFF,):
        claims += [("Giá trị dự báo xe đến", f"LP+forecast({C.RESERVE_MODEL}) so với LP-rolling", tf, 25, ("oracle", f"LP+forecast({C.RESERVE_MODEL})"), ("oracle", "LP-rolling"), "unmet", "kWh thiếu giảm được", diff),
                   ("Giá trị dự báo xe đến", f"LP+forecast(perfect) so với LP+forecast({C.RESERVE_MODEL})", tf, 25, ("oracle", "LP+forecast(perfect)"), ("oracle", f"LP+forecast({C.RESERVE_MODEL})"), "unmet", "kWh thiếu giảm được", diff)]
    for group, comp, tf, p, A, Bspec, col, metric, f in claims:
        a_, w = series(tf, A[0], p, A[1], col)
        b_, _ = series(tf, Bspec[0], p, Bspec[1], col)
        tag = A[0] if A[0] == Bspec[0] else f"{A[0]}/{Bspec[0]}"
        add(group, comp, f"{tf}, {tag}, {p} kW", metric, np.column_stack([a_.values, b_.reindex(a_.index).values]), f, w.values)
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "stats_backtest.csv", index=False)
    return out


def run():
    dm, boot_fc = forecast_tests()
    boot = pd.DataFrame(boot_fc + schedule_tests())
    dm.to_csv(C.RESULTS / "stats_forecast_dm.csv", index=False)
    boot.to_csv(C.RESULTS / "stats_bootstrap.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nKiểm định Diebold–Mariano (sai số tuyệt đối, h = 24, hiệu chỉnh HLN):")
    print(dm.to_string(index=False))
    print(f"\nBootstrap khối {C.BOOT_BLOCK_DAYS} ngày - khoảng kiểm thử chính (khoảng tin cậy 95%):")
    print(boot.drop(columns=["group"]).to_string(index=False))
    bk = backtest_tests()
    if bk is None:
        print("\n(chưa có results/backtest_daily.csv -> bỏ kiểm định nhiều giai đoạn; chạy run_05_backtest.py trước)")
    else:
        print(f"\nKiểm định GỘP mọi cửa sổ (bootstrap khối {C.BOOT_BLOCK_DAYS} ngày):")
        print(bk.drop(columns=["group"]).to_string(index=False))
    print("\nĐã lưu results/stats_forecast_dm.csv, stats_bootstrap.csv" + ("" if bk is None else ", stats_backtest.csv"))
