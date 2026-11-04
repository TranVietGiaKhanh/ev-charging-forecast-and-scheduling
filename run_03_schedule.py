"""
Chạy:  python run_03_schedule.py                 (cần run_01 và run_02 chạy trước)
       python run_03_schedule.py --pmax 40 --skip-lp      (bỏ các biến thể LP cho nhanh)
Sinh ra: results/schedule_summary_all.csv, schedule_summary_test.csv,
         schedule_load_profiles.csv, schedule_session_level.csv, figures/schedule_*.png
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src import scheduling as S


def main(p_max, skip_lp, energy_source):
    C.RESULTS.mkdir(exist_ok=True); C.FIGURES.mkdir(exist_ok=True)
    sessions = pd.read_csv(C.DATA_PROC / "sessions.csv", parse_dates=["connect", "disconnect", "done_charging"])
    hourly = pd.read_csv(C.DATA_PROC / "hourly.csv", index_col=0, parse_dates=True)
    P = S.Problem(sessions, energy_source)

    print(f"Số phiên: {P.n} | số bước 15 phút: {P.n_steps} | P_max = {p_max} kW | cổng = {C.PORT_MAX_KW} kW")
    print(f"Phiên bị giới hạn năng lượng do không đủ thời gian đỗ ở {C.PORT_MAX_KW} kW: {P.n_capped}")

    results = []
    print("\nChạy các chiến lược heuristic ...")
    results.append(S.run_priority_policy(P, "fcfs", np.inf, "FCFS-Uncontrolled"))
    unc_peak = results[0].load.max()
    print(f"  đỉnh tải khi KHÔNG điều khiển: {unc_peak:.1f} kW  (P_max đang dùng = {p_max} kW, "
          f"tức {100 * p_max / unc_peak:.0f}% đỉnh)")
    results.append(S.run_priority_policy(P, "fcfs", p_max, f"FCFS-Capped{p_max:g}kW"))
    results.append(S.run_priority_policy(P, "edf", p_max, "EDF"))
    results.append(S.run_priority_policy(P, "llf", p_max, "LLF"))

    if not skip_lp:
        print("\nChạy LP cuốn chiếu (giải lại mỗi 15 phút) ...")
        results.append(S.run_lp_policy(P, p_max, None, "LP-rolling"))
        nhat_file = C.RESULTS / "N_hat_hourly.csv"
        if nhat_file.exists():
            nhat = pd.read_csv(nhat_file, index_col=0, parse_dates=True)["N_hat"]
            train_N = hourly.loc[C.TRAIN_START:C.TRAIN_END, "N"]
            thr = float(train_N.quantile(C.RESERVE_QUANTILE))
            print(f"  ngưỡng 'nhiều xe đến' = {thr:.2f} xe/giờ (phân vị {int(C.RESERVE_QUANTILE * 100)}% của tập train)")
            for g in C.RESERVE_GAMMAS:
                R = S.build_reserve(P, nhat, thr, g, p_max)
                print(f"  LP + dự phòng gamma={g}: {int((R > 0).sum())} bước có chừa công suất dự phòng")
                results.append(S.run_lp_policy(P, p_max, R, f"LP+reserve(g={g})"))
        else:
            print("  (không thấy results/N_hat_hourly.csv -> bỏ qua biến thể LP có dự phòng; hãy chạy run_02 trước)")

    # ---- bảng chỉ số: toàn bộ tháng và riêng giai đoạn kiểm thử
    all_steps = np.ones(P.n_steps, bool)
    all_sess = np.ones(P.n, bool)
    test0 = pd.Timestamp(C.TEST_START)
    test_steps = np.asarray(P.step_time >= test0)
    test_sess = (P.sessions["connect"] >= test0).values

    for label, sm, ss in [("all", all_steps, all_sess), ("test", test_steps, test_sess)]:
        tab = pd.DataFrame([S.summarize(P, r, p_max, sm, ss) for r in results])
        base = tab.loc[tab.policy == "FCFS-Uncontrolled", "total_cost_VND"].iloc[0]
        tab["cost_saving_vs_uncontrolled_%"] = (100 * (base - tab["total_cost_VND"]) / base).round(2)
        tab.to_csv(C.RESULTS / f"schedule_summary_{label}.csv", index=False)
        title = "TOÀN BỘ THÁNG (954 phiên nếu dữ liệu đủ)" if label == "all" else "GIAI ĐOẠN KIỂM THỬ (từ 22/03)"
        print(f"\n=== {title} ===  ({int(ss.sum())} phiên)")
        print(tab.to_string(index=False))

    # ---- lưu hồ sơ tải và kết quả từng phiên
    prof = pd.DataFrame({"time": P.step_time, "price": P.price, **{r.name: r.load for r in results}})
    prof.to_csv(C.RESULTS / "schedule_load_profiles.csv", index=False)
    sess = pd.DataFrame({"session_id": P.sessions["session_id"], "E_required_kWh": P.E,
                         **{f"delivered_{r.name}": r.delivered for r in results}})
    sess.to_csv(C.RESULTS / "schedule_session_level.csv", index=False)

    # ---- hình 1: hồ sơ tải một ngày
    day = pd.Timestamp(C.PLOT_DAY)
    m = (prof["time"] >= day) & (prof["time"] < day + pd.Timedelta(days=1))
    plt.figure(figsize=(11, 4.2))
    for r in results:
        plt.plot(prof.loc[m, "time"], prof.loc[m, r.name], lw=1.4, label=r.name)
    plt.axhline(p_max, color="red", ls="--", lw=1, label=f"P_max = {p_max:g} kW")
    h1, m1 = map(int, C.PEAK_START.split(":")); h2, m2 = map(int, C.PEAK_END.split(":"))
    plt.axvspan(day + pd.Timedelta(hours=h1, minutes=m1), day + pd.Timedelta(hours=h2, minutes=m2),
                color="orange", alpha=0.15, label="khung giá cao điểm")
    plt.ylabel("Công suất trạm (kW)"); plt.title(f"Hồ sơ tải trạm ngày {C.PLOT_DAY}")
    plt.legend(ncol=4, fontsize=7); plt.tight_layout()
    plt.savefig(C.FIGURES / "schedule_load_profile_day.png", dpi=150); plt.close()

    # ---- hình 2: so sánh chi phí / đỉnh tải / tỷ lệ đáp ứng
    tab = pd.read_csv(C.RESULTS / "schedule_summary_all.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, col, ttl in zip(ax, ["total_cost_VND", "peak_load_kW", "pct_energy_served"],
                           ["Tổng chi phí điện (VNĐ)", "Công suất đỉnh (kW)", "% năng lượng được đáp ứng"]):
        a.barh(tab["policy"], tab[col]); a.set_title(ttl, fontsize=10); a.invert_yaxis()
    ax[1].axvline(p_max, color="red", ls="--")
    plt.tight_layout(); plt.savefig(C.FIGURES / "schedule_comparison.png", dpi=150); plt.close()
    print("\nĐã lưu bảng vào results/ và hình vào figures/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmax", type=float, default=C.P_MAX_KW, help="giới hạn công suất trạm (kW)")
    ap.add_argument("--skip-lp", action="store_true", help="bỏ qua các biến thể LP (chạy nhanh)")
    ap.add_argument("--energy-source", default=C.ENERGY_SOURCE, choices=["delivered", "requested"])
    a = ap.parse_args()
    main(a.pmax, a.skip_lp, a.energy_source)
