"""
Chạy:  python run_03_schedule.py                       (cần run_01, run_02, run_02b chạy trước)
       python run_03_schedule.py --pmax 30 40 --skip-lp  (ít mức P_max, bỏ các biến thể LP cho nhanh)
       python run_03_schedule.py --tariff TT16_cu        (so sánh với khung giờ cũ)
Mô phỏng trên khoảng kiểm thử SIM_START..SIM_END (dự báo ngoài mẫu), với mọi kịch bản thông tin và mọi mức P_max.
Sinh ra: results/schedule_summary.csv, schedule_load_profiles.csv (P_MAX_KW), schedule_load_profiles_all.csv
         (mọi P_max, cho trang demo), schedule_session_level.csv,
         figures/schedule_load_profile_day.png, schedule_comparison.png, schedule_tradeoff.png
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # mỗi tiến trình 1 luồng BLAS, tránh hết bộ nhớ khi chạy song song
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src import scheduling as S
from src import timeaxis as T


def run_policies(P, p_max, skip_lp, nhat, thr, arrivals):
    res = [S.run_priority_policy(P, "fcfs", np.inf, "FCFS-Uncontrolled"),
           S.run_priority_policy(P, "fcfs", p_max, "FCFS-Capped"),
           S.run_priority_policy(P, "edf", p_max, "EDF"),
           S.run_priority_policy(P, "llf", p_max, "LLF")]
    if not skip_lp:
        res.append(S.run_lp_policy(P, p_max, None, "LP-rolling", verbose=False))
        if nhat is not None:
            for g in C.RESERVE_GAMMAS:
                R = S.build_reserve(P, nhat, thr, g, p_max)
                res.append(S.run_lp_policy(P, p_max, R, f"LP+reserve(g={g})", verbose=False))
        for lab, arr in arrivals.items():                    # LP có "xe ảo" từ dự báo số xe đến
            res.append(S.run_lp_policy(P, p_max, None, f"LP+forecast({lab})", verbose=False, arrivals=arr))
        if P.info == "oracle":                               # tối ưu biết trước chỉ có nghĩa với thông tin đầy đủ
            res.append(S.run_offline_optimum(P, p_max))
    return res


def main(p_list, skip_lp, tariff, workers=6):
    C.RESULTS.mkdir(exist_ok=True); C.FIGURES.mkdir(exist_ok=True)
    sfx = "" if tariff == C.TARIFF else f"_{tariff}"          # biểu giá khác mặc định -> không ghi đè kết quả chính
    sessions = pd.read_csv(C.DATA_PROC / "sessions.csv",
                           parse_dates=["connect", "disconnect", "done_charging", "requested_departure"])
    hourly = pd.read_csv(C.DATA_PROC / "hourly.csv", index_col=0, parse_dates=True)

    # giá trị mặc định cho phiên không khai báo: chỉ học từ dữ liệu TRƯỚC khoảng mô phỏng
    sim0, sim1 = T.utc(C.SIM_START), T.utc(C.SIM_END)          # mốc trong config là giờ địa phương -> giờ thực
    past = sessions[sessions["disconnect"] < sim0]               # chỉ phiên đã kết thúc trước khoảng mô phỏng
    defaults = (float(past["kwh_delivered"].median()), float(past["duration_h"].median()))
    sim = sessions[(sessions["connect"] >= sim0) & (sessions["connect"] < sim1)]
    pred_file = C.RESULTS / "session_predictions.csv"
    infos = [i for i in C.INFO_SCENARIOS if i != "predicted" or pred_file.exists()]
    if pred_file.exists():
        sim = sim.merge(pd.read_csv(pred_file), on="session_id", how="left")
        sim = sim.fillna({"kwh_pred": defaults[0], "dur_pred": defaults[1]})
    else:
        print("  (không thấy results/session_predictions.csv -> bỏ kịch bản 'predicted'; hãy chạy run_02b trước)")
    print(f"Mô phỏng {C.SIM_START} -> {C.SIM_END}: {len(sim)} phiên | biểu giá {tariff} | kịch bản {infos}")
    print(f"  phiên có khai báo: {int(sim['has_declaration'].sum())} | mặc định cho phiên không khai báo: "
          f"{defaults[0]:.2f} kWh, đỗ {defaults[1]:.2f} giờ")

    nhat, thr = None, None
    nhat_file = C.RESULTS / "N_hat_hourly.csv"
    if not skip_lp and nhat_file.exists():
        nh = pd.read_csv(nhat_file, index_col=0, parse_dates=True)
        nhat = nh.loc[nh["split"] == "test", "N_hat"]           # chỉ dự báo ngoài mẫu
        thr = float(hourly.loc[T.utc(C.TRAIN_START):T.utc(C.TRAIN_END), "N"].quantile(C.RESERVE_QUANTILE))
        print(f"  ngưỡng 'nhiều xe đến' = {thr:.2f} xe/giờ (phân vị {int(C.RESERVE_QUANTILE * 100)}% của tập train)")
    elif not skip_lp:
        print("  (không thấy results/N_hat_hourly.csv -> bỏ qua LP+reserve; hãy chạy run_02 trước)")
    arrivals = {}
    fc_file = C.RESULTS / "forecast_predictions_test.csv"
    if not skip_lp and fc_file.exists():
        fc = pd.read_csv(fc_file, index_col=0, parse_dates=True)
        arrivals = {C.RESERVE_MODEL: fc[f"N_{C.RESERVE_MODEL}"].clip(lower=0),   # dự báo ngoài mẫu
                    "perfect": fc["N_actual"]}                                  # dự báo hoàn hảo: cận trên giá trị dự báo

    probs = {info: S.Problem(sim, info, defaults, tariff) for info in infos}
    P = probs[infos[0]]
    print(f"  số bước 15 phút: {P.n_steps} | phiên bị giới hạn kWh vì đỗ quá ngắn: {P.n_capped}")
    # mỗi cặp (kịch bản, P_max) độc lập -> chạy song song ở các tiến trình riêng; kết quả giống hệt chạy tuần tự
    jobs = [(info, p_max) for info in infos for p_max in p_list]
    print(f"  chạy song song {len(jobs)} cấu hình ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_policies, probs[info], p_max, skip_lp, nhat, thr, arrivals) for info, p_max in jobs]
        results = [f.result() for f in futs]

    rows, keep, loads, delivered = [], {}, {}, {}
    for (info, p_max), res in zip(jobs, results):
        P = probs[info]
        print(f"\n[{info}] P_max = {p_max:g} kW")
        if nhat is not None:
            R = S.build_reserve(P, nhat, thr, C.RESERVE_GAMMAS[0], p_max)
            print(f"  số bước có chừa dự phòng: {int((R > 0).sum())}/{P.n_steps}")
        tab = pd.DataFrame([S.summarize(P, r, p_max) for r in res])
        base = tab.loc[tab.policy == "FCFS-Uncontrolled", "total_cost_VND"].iloc[0]
        tab["cost_saving_vs_uncontrolled_%"] = (100 * (base - tab["total_cost_VND"]) / base).round(2)
        tab.insert(0, "p_max_kW", p_max); tab.insert(0, "info", info)
        print(tab.drop(columns=["info", "p_max_kW", "mean_solve_time_s", "solver_fallbacks"]).to_string(index=False))
        rows.append(tab)
        loads.update({f"{info}/{p_max:g}/{r.name}": r.load for r in res})
        delivered.update({f"{info}/{p_max:g}/{r.name}": r.delivered for r in res})
        if p_max == C.P_MAX_KW:
            keep[info] = (P, res)

    summary = pd.concat(rows, ignore_index=True)
    summary.insert(2, "tariff", tariff)
    summary.to_csv(C.RESULTS / f"schedule_summary{sfx}.csv", index=False)
    # hồ sơ tải ở MỌI mức P_max (dùng cho trang demo); cột = "<kịch bản>/<P_max>/<chiến lược>"
    P0 = S.Problem(sim, C.INFO_SCENARIOS[0], defaults, tariff)
    # time = giờ thực (UTC), time_local = giờ hiển thị (địa phương)
    allp = pd.DataFrame({"time": P0.step_time, "time_local": T.local_naive(P0.step_time), "price": P0.price, "is_peak": P0.is_peak})
    allp = pd.concat([allp, pd.DataFrame({k: pd.Series(v[:len(allp)]).round(3) for k, v in loads.items()})], axis=1)
    allp.fillna(0.0).to_csv(C.RESULTS / f"schedule_load_profiles_all{sfx}.csv", index=False)
    # kWh cấp cho từng phiên ở MỌI mức P_max (dùng cho kiểm định bootstrap theo ngày, run_04)
    alls = pd.DataFrame({"session_id": P0.sessions["session_id"], "connect": P0.sessions["connect"], "E_kWh": P0.E})
    alls = pd.concat([alls, pd.DataFrame({k: np.round(v, 4) for k, v in delivered.items()})], axis=1)
    alls.to_csv(C.RESULTS / f"schedule_session_level_all{sfx}.csv", index=False)

    if C.P_MAX_KW not in p_list:
        print(f"\n(P_MAX_KW = {C.P_MAX_KW:g} không nằm trong danh sách -> bỏ qua hồ sơ tải và hình theo ngày)")
    else:
        # ---- hồ sơ tải và kết quả từng phiên ở P_MAX_KW
        P0 = keep[C.INFO_SCENARIOS[0]][0]
        prof = pd.DataFrame({"time": P0.step_time, "time_local": T.local_naive(P0.step_time), "price": P0.price})
        sess = pd.DataFrame({"session_id": P0.sessions["session_id"], "E_kWh": P0.E})
        for info, (P, res) in keep.items():
            for r in res:
                prof[f"{info}/{r.name}"] = pd.Series(r.load[:len(prof)])
                sess[f"{info}/delivered_{r.name}"] = r.delivered
        prof.fillna(0.0).to_csv(C.RESULTS / f"schedule_load_profiles{sfx}.csv", index=False)
        sess.to_csv(C.RESULTS / f"schedule_session_level{sfx}.csv", index=False)

        # ---- hình 1: hồ sơ tải một ngày (kịch bản oracle)
        P, res = keep[C.INFO_SCENARIOS[0]]
        lt = T.local_naive(P.step_time)                       # ngày vẽ chọn theo lịch địa phương, trục hình cũng vậy
        day = pd.Timestamp(C.PLOT_DAY)
        m = np.asarray((lt >= day) & (lt < day + pd.Timedelta(days=1)))
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for r in res:
            ax.plot(lt[m], r.load[m], lw=1.4, label=r.name)
        ax.axhline(C.P_MAX_KW, color="red", ls="--", lw=1, label=f"P_max = {C.P_MAX_KW:g} kW")
        ax.fill_between(lt[m], 0, 1, where=P.is_peak[m], transform=ax.get_xaxis_transform(),
                        color="orange", alpha=0.15, label="giờ cao điểm")
        ax.set_ylabel("Công suất trạm (kW)"); ax.set_title(f"Hồ sơ tải trạm ngày {C.PLOT_DAY} ({C.INFO_SCENARIOS[0]})")
        ax.legend(ncol=4, fontsize=7); fig.tight_layout()
        fig.savefig(C.FIGURES / f"schedule_load_profile_day{sfx}.png", dpi=150); plt.close(fig)

        # ---- hình 2: so sánh các chiến lược ở P_MAX_KW, theo từng kịch bản thông tin
        t = summary[summary.p_max_kW == C.P_MAX_KW]
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        for a, col, ttl in zip(ax, ["cost_per_kWh_VND", "peak_load_kW", "pct_energy_served"],
                               ["Chi phí trung bình (đ/kWh)", "Công suất đỉnh (kW)", "% năng lượng được đáp ứng"]):
            piv = t.pivot(index="policy", columns="info", values=col).reindex(t.policy.drop_duplicates())
            piv.plot.barh(ax=a, legend=a is ax[0]); a.set_title(ttl, fontsize=10); a.invert_yaxis(); a.set_ylabel("")
        ax[1].axvline(C.P_MAX_KW, color="red", ls="--")
        ax[2].set_xlim(left=max(0, t.pct_energy_served.min() - 2))
        fig.suptitle(f"P_max = {C.P_MAX_KW:g} kW"); fig.tight_layout()
        fig.savefig(C.FIGURES / f"schedule_comparison{sfx}.png", dpi=150); plt.close(fig)

    # ---- hình 3: đánh đổi theo P_max
    fig, ax = plt.subplots(2, len(infos), figsize=(6 * len(infos), 7.5), sharex=True, squeeze=False)
    for j, info in enumerate(infos):
        d = summary[(summary["info"] == info) & (summary.policy != "FCFS-Uncontrolled")]
        for pol, g in d.groupby("policy", sort=False):
            ax[0, j].plot(g.p_max_kW, g.cost_per_kWh_VND, "o-", label=pol)
            ax[1, j].plot(g.p_max_kW, g.unmet_energy_kWh, "o-", label=pol)
        ax[0, j].set_title(f"Kịch bản thông tin: {info}"); ax[0, j].set_ylabel("Chi phí trung bình (đ/kWh)")
        ax[1, j].set_ylabel("Năng lượng không đáp ứng (kWh)"); ax[1, j].set_xlabel("P_max (kW)")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.FIGURES / f"schedule_tradeoff{sfx}.png", dpi=150); plt.close(fig)
    print("\nĐã lưu results/schedule_*.csv và figures/schedule_*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmax", type=float, nargs="+", default=C.P_MAX_LIST, help="các mức giới hạn công suất trạm (kW)")
    ap.add_argument("--skip-lp", action="store_true", help="bỏ qua LP (chạy nhanh)")
    ap.add_argument("--tariff", default=C.TARIFF, choices=list(C.TARIFFS))
    ap.add_argument("--workers", type=int, default=6, help="số tiến trình chạy song song (mỗi tiến trình ~150 MB RAM)")
    a = ap.parse_args()
    main(a.pmax, a.skip_lp, a.tariff, a.workers)
