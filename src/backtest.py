"""
BƯỚC 5 - KIỂM THỬ NHIỀU GIAI ĐOẠN (rolling-origin backtest)

Thay vì chỉ một khoảng kiểm thử 40 ngày, lặp lại toàn bộ quy trình trên các cửa sổ LIÊN TIẾP C.BACKTEST_WINDOWS
(cửa sổ cuối = khoảng kiểm thử chính). Với mỗi cửa sổ [w0, w1):
  * dự báo N, L: huấn luyện lại 7 mô hình chỉ trên các giờ trước w0 (từ TRAIN_START), dự báo cho cửa sổ;
  * mô hình từng xe: học từ các phiên đã kết thúc trước w0, chọn phân vị bằng mô phỏng LP trên BACKTEST_VAL_DAYS ngày
    ngay trước w0;
  * giá trị mặc định cho phiên không khai báo: trung vị của các phiên đã kết thúc trước w0;
  * mô phỏng các chiến lược cần cho kết luận chính ở các mức C.BACKTEST_PMAX.
Kết quả theo ngày (lịch địa phương) của mọi cửa sổ được ghép lại để kiểm định bằng bootstrap khối (run_04).
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import forecasting as F
from . import scheduling as S
from . import session_model as SM
from . import timeaxis as T

RF = f"LP+forecast({C.RESERVE_MODEL})"


def windows():
    return [(a, T.utc(a), T.utc(b)) for a, b in C.BACKTEST_WINDOWS]


# ================================================================== dự báo
def forecast_windows(feats):
    rows, preds = [], []
    t_lo = T.utc(C.TRAIN_START)
    for wid, w0, w1 in windows():
        train = feats[(feats.index >= t_lo) & (feats.index < w0)]
        test = feats[(feats.index >= w0) & (feats.index < w1)]
        out = {}
        for y in F.TARGETS:
            p, _, _ = F.fit_target(y, train, test, verbose=False)
            yt = test[y].values
            ref = np.mean(np.abs(p["SeasonalNaive_24h"] - yt))
            for name, v in p.items():
                mae, rmse = F.metrics(yt, v)
                rows.append({"window": wid, "target": y, "model": name, "MAE": mae, "RMSE": rmse,
                             "improve_vs_SN24_%": 100 * (ref - mae) / ref})
            out[f"{y}_actual"] = yt
            out.update({f"{y}_{k}": v for k, v in p.items()})
        df = pd.DataFrame(out, index=test.index).rename_axis("timestamp_utc")
        df.insert(0, "local_time", T.local_naive(df.index))
        df.insert(0, "window", wid)
        preds.append(df)
        print(f"  dự báo cửa sổ {wid}: {len(train)} giờ huấn luyện, {len(test)} giờ kiểm thử", flush=True)
    return pd.DataFrame(rows).round(4), pd.concat(preds)


# ================================================================== mô hình từng xe
def _session_window(args):
    s, f, wid, w0, w1 = args
    # lùi theo LỊCH ĐỊA PHƯƠNG rồi mới đổi sang UTC: lùi 28x24 giờ thực qua ngày chuyển DST sẽ lệch 1 giờ
    val0 = T.utc(pd.Timestamp(wid) - pd.Timedelta(days=C.BACKTEST_VAL_DAYS))
    qk, qd, _ = SM.choose_quantiles(s, f, val0, w0)
    tr, te = s.disconnect < w0, (s.connect >= w0) & (s.connect < w1)
    kp, dp = SM.fit_predict(s, f, tr, te, qk, qd)
    return wid, qk, qd, pd.DataFrame({"session_id": s.loc[te, "session_id"].values, "kwh_pred": kp.round(6), "dur_pred": dp.round(6)})


# ================================================================== lập lịch
def _daily(P, loads, delivered):
    """Chi phí theo ngày của bước sạc và kWh thiếu theo ngày xe cắm (lịch địa phương)."""
    sday = T.local_day(P.sessions["connect"]).values
    days = np.unique(sday)
    cost = S.daily_costs(P.step_day, P.price, loads, P.demand_charge).sum(axis=1).reindex(days, fill_value=0.0)
    unmet = pd.Series(np.clip(P.E - delivered, 0, None)).groupby(sday).sum().reindex(days, fill_value=0.0)
    return pd.DataFrame({"day": days, "cost": cost.values, "unmet": unmet.values})


def _sched_job(args):
    wid, w0, sim, defaults, tariff, info, p_max, arrivals = args
    P = S.Problem(sim, info, defaults, tariff, t0=w0)
    res = [S.run_priority_policy(P, "fcfs", np.inf, "FCFS-Uncontrolled"), S.run_priority_policy(P, "llf", p_max, "LLF"),
           S.run_lp_policy(P, p_max, None, "LP-rolling", verbose=False)]
    if info == "oracle":
        res += [S.run_offline_optimum(P, p_max)] + [S.run_lp_policy(P, p_max, None, f"LP+forecast({k})", verbose=False, arrivals=a)
                                                    for k, a in arrivals.items()]
    key = {"window": wid, "tariff": tariff, "info": info, "p_max_kW": p_max}
    summ = [{**key, **S.summarize(P, r, p_max)} for r in res]
    daily = [_daily(P, r.load, r.delivered).assign(**key, policy=r.name) for r in res]
    return summ, pd.concat(daily)


def run(workers=6):
    C.RESULTS.mkdir(exist_ok=True)
    feats = pd.read_csv(C.DATA_PROC / "features.csv", index_col=0, parse_dates=True)
    hourly = pd.read_csv(C.DATA_PROC / "hourly.csv", index_col=0, parse_dates=True)
    s = pd.read_csv(C.DATA_PROC / "sessions.csv", parse_dates=["connect", "disconnect", "done_charging", "requested_departure"])
    s = s.sort_values("connect").reset_index(drop=True)
    f = SM.session_features(s)
    W = windows()
    print(f"{len(W)} cửa sổ: " + ", ".join(f"{a}→{b}" for a, b in C.BACKTEST_WINDOWS))

    print("\n1) Dự báo N, L cho từng cửa sổ (huấn luyện lại 7 mô hình mỗi cửa sổ)")
    fm, fp = forecast_windows(feats)
    fm.to_csv(C.RESULTS / "backtest_forecast_metrics.csv", index=False)
    fp.to_csv(C.RESULTS / "backtest_forecast_predictions.csv")

    print("\n2) Mô hình từng xe cho từng cửa sổ (chọn phân vị trên 4 tuần trước cửa sổ)")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        sw = list(ex.map(_session_window, [(s, f, wid, w0, w1) for wid, w0, w1 in W]))
    pd.DataFrame([{"window": wid, "q_kwh": qk, "q_dur": qd} for wid, qk, qd, _ in sw]) \
        .to_csv(C.RESULTS / "backtest_session_quantiles.csv", index=False)
    preds = {wid: p for wid, _, _, p in sw}

    print("\n3) Mô phỏng lập lịch cho từng cửa sổ")
    jobs = []
    for wid, w0, w1 in W:
        sim = s[(s.connect >= w0) & (s.connect < w1)].merge(preds[wid], on="session_id", how="left")
        past = s[s.disconnect < w0]
        defaults = (float(past["kwh_delivered"].median()), float(past["duration_h"].median()))
        sim = sim.fillna({"kwh_pred": defaults[0], "dur_pred": defaults[1]})
        win = fp[fp.window == wid]
        arrivals = {C.RESERVE_MODEL: win[f"N_{C.RESERVE_MODEL}"],
                    "perfect": hourly.loc[(hourly.index >= w0) & (hourly.index < w1), "N"]}
        for tariff, infos in ((C.TARIFF, C.INFO_SCENARIOS),):
            for info in infos:
                for p_max in C.BACKTEST_PMAX:
                    jobs.append((wid, w0, sim, defaults, tariff, info, p_max, arrivals))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_sched_job, jobs))
    summary = pd.DataFrame([r for summ, _ in out for r in summ])
    base = summary[summary.policy == "FCFS-Uncontrolled"].set_index(["window", "tariff", "info", "p_max_kW"])["total_cost_USD"]
    summary["cost_saving_vs_uncontrolled_%"] = (100 * (1 - summary["total_cost_USD"].values /
        base.reindex(pd.MultiIndex.from_frame(summary[["window", "tariff", "info", "p_max_kW"]])).values)).round(2)
    summary.to_csv(C.RESULTS / "backtest_summary.csv", index=False)
    pd.concat([d for _, d in out]).round(4).to_csv(C.RESULTS / "backtest_daily.csv", index=False)

    # kiểm tra nhất quán: cửa sổ trùng khoảng kiểm thử chính phải cho đúng số của run_03 (cùng dữ liệu, cùng mốc)
    main_f = C.RESULTS / "schedule_summary.csv"
    if main_f.exists() and C.BACKTEST_WINDOWS[-1][0] == C.SIM_START[:10]:
        m = pd.read_csv(main_f).set_index(["info", "p_max_kW", "policy"])
        x = summary[(summary.window == C.BACKTEST_WINDOWS[-1][0]) & (summary.tariff == C.TARIFF)]
        dev = max(abs(r.unmet_energy_kWh - m.loc[(r["info"], r.p_max_kW, r.policy), "unmet_energy_kWh"]) for _, r in x.iterrows())
        print(f"\nKiểm tra nhất quán với run_03 (cửa sổ cuối): chênh kWh thiếu lớn nhất = {dev:.3f} "
              + ("-> KHỚP" if dev < 0.05 else "-> KHÔNG KHỚP, cần kiểm tra"))
    pd.set_option("display.width", 250)
    print("\nDự báo - MAE theo cửa sổ (mô hình học được tốt nhất và SN24):")
    print(fm[fm.model.isin(["SeasonalNaive_24h", "RandomForest", "MLP"])].pivot_table(
        index="window", columns=["target", "model"], values="MAE").round(3).to_string())
    k = summary[(summary.tariff == C.TARIFF) & (summary.p_max_kW == 40) & summary.policy.isin(["LLF", "LP-rolling"])]
    print("\nLập lịch (khung giờ mới, 40 kW) - kWh thiếu và % tiết kiệm theo cửa sổ:")
    print(k.pivot_table(index="window", columns=["info", "policy"], values=["unmet_energy_kWh"]).round(1).to_string())
    print("\nĐã lưu results/backtest_*.csv")
