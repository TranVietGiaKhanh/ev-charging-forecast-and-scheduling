"""
Sinh trang demo tương tác: docs/demo.html (một file duy nhất, mở bằng trình duyệt, KHÔNG cần mạng).

Chạy:  python make_demo_page.py        (sau run_01, run_02, run_02b, run_03)
Dữ liệu lấy từ data/processed/ và results/; Chart.js được nhúng từ docs/vendor/chart.umd.min.js.
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src import timeaxis as T

OUT = C.ROOT / "docs" / "demo.html"
VENDOR = C.ROOT / "docs" / "vendor" / "chart.umd.min.js"


def rnd(a, d=2):
    return [None if pd.isna(x) else round(float(x), d) for x in a]


def build_data():
    s = pd.read_csv(C.DATA_PROC / "sessions.csv", parse_dates=["connect"])
    h = pd.read_csv(C.DATA_PROC / "hourly.csv", index_col=0, parse_dates=True)
    # dữ liệu lưu theo giờ thực (UTC); trang demo hiển thị theo giờ địa phương
    m = s.groupby(T.local_naive(s.connect).dt.to_period("M")).size()
    hl = T.local(h.index)
    wd, we = h[hl.dayofweek < 5], h[hl.dayofweek >= 5]
    hwd, hwe = hl[hl.dayofweek < 5].hour, hl[hl.dayofweek >= 5].hour

    fc = pd.read_csv(C.RESULTS / "forecast_predictions_test.csv", index_col=0, parse_dates=True)
    models = [c[2:] for c in fc.columns if c.startswith("N_") and c != "N_actual"]
    fm = pd.read_csv(C.RESULTS / "forecast_metrics.csv")

    prof = pd.read_csv(C.RESULTS / "schedule_load_profiles_all.csv", parse_dates=["time"])
    load_cols = [c for c in prof.columns if c.count("/") == 2]
    band = np.where(prof["is_peak"], "p", np.where(prof["price"] == prof["price"].min(), "o", "n"))

    summ = {"main": pd.read_csv(C.RESULTS / "schedule_summary.csv")}
    tf = C.TARIFFS[C.TARIFF]
    win = next(x for x in tf["seasons"] if pd.Timestamp(C.SIM_START).month in x["months"])   # mùa của khoảng mô phỏng
    usd = lambda x: f"${x:.4f}".replace(".", ",")
    pr = dict(zip(win["weekday_start_h"], win["weekday_price"]))
    tariff_text = (f"Giá điện {tf['name']} ({win['name'].lower()}, ngày thường): cao điểm 12–18 giờ {usd(pr[12])}, "
                   f"8–12 giờ và 18–23 giờ {usd(pr[8])}, đêm {usd(pr[0])}/kWh; cuối tuần {usd(win['weekend_price'])}/kWh; "
                   f"phí công suất đỉnh ${tf['demand_charge']:.2f}/kW/tháng".replace(".", ",") + ".")
    sim = s[(s.connect >= T.utc(C.SIM_START)) & (s.connect < T.utc(C.SIM_END))]
    sm = C.RESULTS / "session_model_metrics.csv"
    sm = pd.read_csv(sm).to_dict(orient="records") if sm.exists() else None

    return {
        "meta": {"n_sessions": len(s), "n_hours": len(h), "sim_sessions": len(sim),
                 "data_start": C.DATA_START[:10], "data_end": "2019-04-30",
                 "sim_start": C.SIM_START[:10], "port_kw": C.PORT_MAX_KW,
                 "tariff_text": tariff_text},
        "session_metrics": sm,
        "monthly": {"labels": [p.strftime("%m/%Y") for p in m.index], "values": [int(v) for v in m.values]},
        "hourly": {"N_wd": rnd(wd.groupby(hwd).N.mean()), "N_we": rnd(we.groupby(hwe).N.mean()),
                   "L_wd": rnd(wd.groupby(hwd).L.mean(), 1), "L_we": rnd(we.groupby(hwe).L.mean(), 1)},
        # shortcut: trang cộng giờ liên tục từ t0 theo giờ địa phương; đúng vì khoảng kiểm thử không có ngày chuyển DST
        "forecast": {"t0": T.local_naive(fc.index)[0].strftime("%Y-%m-%dT%H:%M"), "models": models,
                     "series": {f"{y}_{k}": rnd(fc[f"{y}_{k}"]) for y in "NL" for k in ["actual"] + models},
                     "metrics": fm.round(3).to_dict(orient="records")},
        "sched": {"t0": T.local_naive(prof["time"]).iloc[0].strftime("%Y-%m-%dT%H:%M"), "band": "".join(band),
                  "loads": {c: rnd(prof[c], 1) for c in load_cols},
                  "summary": {k: v.replace({np.nan: None}).to_dict(orient="records") for k, v in summ.items()}},
    }


def main():
    if not VENDOR.exists():
        raise SystemExit(f"Thiếu {VENDOR} (Chart.js). Tải chart.js 4.4.1 bản UMD và đặt vào đó.")
    data = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":"))
    html = (TEMPLATE.replace("__CHARTJS__", VENDOR.read_text(encoding="utf-8"))
            .replace("__DATA__", data.replace("</", "<\\/")))
    OUT.write_text(html, encoding="utf-8")
    print(f"Đã ghi {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Demo lập lịch sạc</title>
<style>
:root {
  color-scheme: light;
  --bg: #f6f6f4; --surface: #fcfcfb; --border: #e3e2dd; --grid: #ecebe7;
  --text: #0b0b0b; --text-2: #52514e; --text-3: #7a7974;
  --accent: #2a78d6;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --ref: #7a7974;
  --band-peak: rgba(235, 104, 52, 0.13); --band-off: rgba(42, 120, 214, 0.08);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface: #1a1a19; --border: #2e2e2b; --grid: #2a2a28;
    --text: #f0efec; --text-2: #c3c2b7; --text-3: #8f8e86; --accent: #3987e5;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
    --ref: #8f8e86; --band-peak: rgba(217, 89, 38, 0.2); --band-off: rgba(57, 135, 229, 0.13);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --border: #2e2e2b; --grid: #2a2a28;
  --text: #f0efec; --text-2: #c3c2b7; --text-3: #8f8e86; --accent: #3987e5;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
  --ref: #8f8e86; --band-peak: rgba(217, 89, 38, 0.2); --band-off: rgba(57, 135, 229, 0.13);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { padding: 28px 16px 8px; max-width: 1180px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--text-2); margin: 0; }
nav { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--border); }
nav .in { max-width: 1180px; margin: 0 auto; padding: 8px 16px; display: flex; gap: 6px; overflow-x: auto; }
nav a { color: var(--text-2); text-decoration: none; padding: 6px 12px; border-radius: 999px; white-space: nowrap; font-size: 14px; }
nav a:hover { background: var(--surface); color: var(--text); }
main { max-width: 1180px; margin: 0 auto; padding: 8px 16px 48px; }
section { margin-top: 28px; scroll-margin-top: 60px; }
h2 { font-size: 20px; margin: 0 0 4px; }
.lead { color: var(--text-2); margin: 0 0 14px; max-width: 80ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid2 > *, .card { min-width: 0; }
@media (max-width: 820px) { .grid2 { grid-template-columns: 1fr; } }
.card h3 { font-size: 15px; margin: 0 0 2px; }
.card .note { color: var(--text-3); font-size: 13px; margin: 0 0 8px; }
.chart { position: relative; height: 300px; }
.chart.tall { height: 380px; }
.controls { display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: center; margin-bottom: 14px; }
.ctl { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ctl > span { color: var(--text-2); font-size: 13px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { background: var(--surface); color: var(--text-2); border: 0; padding: 6px 12px; font: inherit; font-size: 14px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.seg button:focus-visible, select:focus-visible, .chip input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 2px; }
select, .btn { background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; font: inherit; font-size: 14px; }
.btn { cursor: pointer; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px 3px 8px; font-size: 13px; cursor: pointer; color: var(--text-2); background: var(--surface); user-select: none; }
.chip input { position: absolute; opacity: 0; pointer-events: none; }
.chip .sw { width: 14px; height: 3px; border-radius: 2px; }
.chip.off { opacity: 0.45; }
.chip.off .sw { background: var(--border) !important; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px; }
@media (max-width: 820px) { .tiles { grid-template-columns: repeat(2, 1fr); } }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
.tile .k { color: var(--text-2); font-size: 13px; }
.tile .v { font-size: 26px; font-weight: 650; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
.tile .d { color: var(--text-3); font-size: 13px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-2); font-weight: 600; font-size: 13px; }
tr.hl td { font-weight: 650; }
.tw { overflow-x: auto; }
.key { display: inline-flex; align-items: center; gap: 6px; color: var(--text-2); font-size: 13px; margin-right: 14px; }
.key i { width: 14px; height: 10px; border-radius: 2px; display: inline-block; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--text-2); font-size: 13px; }
.insight { border-left: 3px solid var(--accent); padding: 4px 12px; margin: 12px 0 0; color: var(--text-2); }
.insight b { color: var(--text); }
footer { max-width: 1180px; margin: 0 auto; padding: 0 16px 32px; color: var(--text-3); font-size: 13px; }
</style>
</head>
<body>
<header>
  <h1>Dự báo phụ tải và lập lịch sạc xe điện</h1>
  <p class="sub" id="sub"></p>
</header>
<nav><div class="in">
  <a href="#data">1. Dữ liệu</a><a href="#forecast">2. Dự báo</a><a href="#sched">3. Lập lịch</a>
  <a href="#tradeoff">4. Đánh đổi theo P_max</a>
</div></nav>
<main>

<section id="data">
  <h2>1. Dữ liệu</h2>
  <p class="lead" id="data-lead"></p>
  <div class="grid2">
    <div class="card"><h3>Số phiên sạc theo tháng</h3><p class="note">Từ 11/2018 lượng xe giảm khoảng một nửa.</p>
      <div class="chart"><canvas id="c-monthly" role="img" aria-label="Số phiên sạc theo tháng"></canvas></div>
      <details><summary>Xem bảng số liệu</summary><div class="tw"><table id="t-monthly"></table></div></details></div>
    <div class="card"><h3>Số xe đến trung bình theo giờ trong ngày</h3><p class="note">Trạm nơi làm việc: xe đến dồn vào 7–9 giờ sáng ngày thường.</p>
      <div class="chart"><canvas id="c-hourN" role="img" aria-label="Số xe đến trung bình theo giờ"></canvas></div></div>
  </div>
</section>

<section id="forecast">
  <h2>2. Dự báo trước 24 giờ</h2>
  <p class="lead">Dự báo số xe đến (N) và tải trạm (L) cho từng giờ trên tập kiểm thử 22/03 – 30/04/2019. Mọi đặc trưng chỉ dùng dữ liệu cách thời điểm dự báo ít nhất 24 giờ.</p>
  <div class="controls">
    <div class="ctl"><span>Đại lượng</span><div class="seg" id="f-target"></div></div>
    <div class="ctl"><span>Tuần</span><select id="f-week" aria-label="Chọn tuần"></select></div>
    <div class="ctl"><span>Mô hình</span><div class="chips" id="f-models"></div></div>
  </div>
  <div class="card"><div class="chart tall"><canvas id="c-forecast" role="img" aria-label="Dự báo so với thực tế"></canvas></div></div>
  <div class="card" style="margin-top:16px"><h3>Sai số trên toàn bộ tập kiểm thử (960 giờ)</h3>
    <p class="note">MAE: sai số tuyệt đối trung bình (càng nhỏ càng tốt). Cột cuối: tốt hơn bao nhiêu % so với “ngày mai giống hôm nay” (Seasonal Naive 24h).</p>
    <div class="tw"><table id="t-metrics"></table></div>
    <p class="insight" id="f-insight"></p></div>
</section>

<section id="sched">
  <h2>3. Lập lịch sạc</h2>
  <p class="lead" id="sched-lead"></p>
  <div class="controls">
    <div class="ctl"><span>Giới hạn trạm P_max</span><div class="seg" id="s-pmax"></div></div>
    <div class="ctl"><span>Thông tin về xe</span><div class="seg" id="s-info"></div></div>
    <div class="ctl"><span>Ngày</span><button class="btn" id="s-prev" aria-label="Ngày trước">‹</button>
      <select id="s-day" aria-label="Chọn ngày"></select><button class="btn" id="s-next" aria-label="Ngày sau">›</button></div>
  </div>
  <div class="tiles" id="s-tiles"></div>
  <div class="card">
    <h3 id="s-title"></h3>
    <p class="note"><span class="key"><i style="background:var(--band-peak)"></i>giờ cao điểm</span><span class="key"><i style="background:var(--band-off)"></i>giờ thấp điểm</span>
      <span class="key">không tô màu: giờ bình thường</span></p>
    <div class="chips" id="s-pols" style="margin-bottom:8px"></div>
    <div class="chart tall"><canvas id="c-load" role="img" aria-label="Công suất trạm theo thời gian"></canvas></div>
  </div>
  <div class="card" style="margin-top:16px"><h3 id="s-tab-title"></h3>
    <p class="note">Số liệu cho toàn bộ khoảng mô phỏng (40 ngày). Chiến lược rẻ mà sạc thiếu thì không được tính là tốt: đọc chi phí kèm với % năng lượng đáp ứng.</p>
    <div class="tw"><table id="t-sched"></table></div>
    <p class="insight" id="s-insight"></p></div>
</section>

<section id="tradeoff">
  <h2>4. Đánh đổi theo P_max</h2>
  <p class="lead">Giới hạn công suất càng thấp thì phí công suất đỉnh càng nhỏ nên chi phí mỗi kWh thấp hơn, nhưng xe bị sạc thiếu nhiều hơn. Dưới khoảng 30 kW, mọi chiến lược đều không đáp ứng đủ.</p>
  <div class="controls"><div class="ctl"><span>Thông tin về xe</span><div class="seg" id="t-info"></div></div></div>
  <div class="grid2">
    <div class="card"><h3>Chi phí trung bình ($/kWh, gồm phí công suất đỉnh)</h3><p class="note">Càng thấp càng tốt.</p><div class="chart"><canvas id="c-cost" role="img" aria-label="Chi phí theo P_max"></canvas></div></div>
    <div class="card"><h3>Năng lượng không đáp ứng (kWh)</h3><p class="note">Càng thấp càng tốt.</p><div class="chart"><canvas id="c-unmet" role="img" aria-label="Năng lượng thiếu theo P_max"></canvas></div></div>
  </div>
</section>

</main>
<footer id="foot"></footer>

<script>__CHARTJS__</script>
<script>
const D = __DATA__;
const usd = (x) => x == null ? "–" : "$" + nf(x, 4);
const nf = (x, d = 0) => x == null ? "–" : new Intl.NumberFormat("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d }).format(x);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const $ = (id) => document.getElementById(id);
const DAYS = ["CN", "T2", "T3", "T4", "T5", "T6", "T7"];
const addMin = (iso, m) => new Date(new Date(iso + ":00").getTime() + m * 60000);
const pad = (n) => String(n).padStart(2, "0");

// ---- màu cố định theo chiến lược (màu đi theo đối tượng, không đổi khi lọc)
// LP+reserve(g=0.2) có trong CSV nhưng không hiện: kết quả gần như trùng g=0.1
const POLS = ["FCFS-Uncontrolled", "FCFS-Capped", "EDF", "LLF", "LP-rolling", "LP+reserve(g=0.1)", "LP+forecast(RandomForest)",
              "LP+forecast(perfect)", "Offline-optimal"];
const PCOL = { "LP-rolling": "--s1", "LLF": "--s2", "EDF": "--s3", "FCFS-Capped": "--s4", "LP+reserve(g=0.1)": "--s5",
               "Offline-optimal": "--s6", "LP+forecast(RandomForest)": "--s7", "LP+forecast(perfect)": "--s7", "FCFS-Uncontrolled": "--ref" };
const PDASH = { "FCFS-Uncontrolled": [5, 4], "Offline-optimal": [2, 3], "LP+forecast(perfect)": [2, 3] };
const PNAME = { "FCFS-Uncontrolled": "Không điều khiển", "FCFS-Capped": "FCFS có giới hạn", "EDF": "EDF", "LLF": "LLF",
                "LP-rolling": "LP cuốn chiếu", "LP+reserve(g=0.1)": "LP + dự phòng 10%", "LP+reserve(g=0.2)": "LP + dự phòng 20%",
                "LP+forecast(RandomForest)": "LP + dự báo xe đến (RF)", "LP+forecast(perfect)": "LP + dự báo xe đến hoàn hảo",
                "Offline-optimal": "Tối ưu biết trước" };
const MCOL = { MLP: "--s1", RandomForest: "--s2", Ridge: "--s3", DecisionTree: "--s4", SeasonalNaive_24h: "--s5",
               SeasonalNaive_168h: "--s7", HistAvg_hour_dow: "--s8" };
const INFO = { oracle: "Biết đúng (oracle)", declared: "Theo khai báo (declared)", predicted: "Dự báo bằng ML (predicted)" };

const st = { target: "N", week: 0, models: new Set(["SeasonalNaive_24h", "RandomForest", "MLP"]),
             pmax: 40, info: "oracle", day: 4, pols: new Set(["FCFS-Uncontrolled", "LLF", "LP-rolling", "Offline-optimal"]),
             tinfo: "oracle" };
const charts = {};

function baseOpts(yTitle) {
  const g = css("--grid"), t2 = css("--text-2");
  return {
    responsive: true, maintainAspectRatio: false, animation: false, locale: "vi-VN",
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { labels: { color: t2, boxWidth: 14, boxHeight: 3, usePointStyle: false } },
               tooltip: { backgroundColor: css("--surface"), titleColor: css("--text"), bodyColor: t2, borderColor: css("--border"),
                          borderWidth: 1, padding: 10, boxWidth: 10, boxHeight: 3 } },
    scales: { x: { grid: { display: false }, ticks: { color: t2, maxRotation: 0, autoSkipPadding: 14 }, border: { color: g } },
              y: { beginAtZero: true, grid: { color: g }, border: { display: false }, ticks: { color: t2 },
                   title: { display: !!yTitle, text: yTitle, color: t2 } } },
  };
}
function mk(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); return charts[id]; }
function seg(el, items, cur, on) {
  el.innerHTML = "";
  items.forEach(([v, label]) => { const b = document.createElement("button"); b.type = "button"; b.textContent = label;
    b.setAttribute("aria-pressed", String(v === cur)); b.onclick = () => on(v); el.appendChild(b); });
}
function chips(el, items, set, colorOf, nameOf, on) {
  el.innerHTML = "";
  items.forEach((k) => { const l = document.createElement("label"); l.className = "chip" + (set.has(k) ? "" : " off");
    l.innerHTML = `<input type="checkbox" ${set.has(k) ? "checked" : ""}><span class="sw" style="background:var(${colorOf(k)})"></span><span>${nameOf(k)}</span>`;
    l.querySelector("input").onchange = (e) => { e.target.checked ? set.add(k) : set.delete(k); on(); }; el.appendChild(l); });
}

// ================================================================ 1. dữ liệu
function drawData() {
  const M = D.meta;
  $("sub").textContent = `Dữ liệu thật trạm sạc Caltech (ACN-Data), ${M.data_start.split("-").reverse().join("/")} – 30/04/2019: ${nf(M.n_sessions)} phiên sạc, ${nf(M.n_hours)} giờ.`;
  $("data-lead").textContent = `${nf(M.n_sessions)} phiên sạc của 54 cổng, gộp thành chuỗi theo giờ: N = số xe đến, L = công suất trạm (kW).`;
  const o = baseOpts("phiên"); o.plugins.legend.display = false;
  mk("c-monthly", { type: "bar", data: { labels: D.monthly.labels, datasets: [{ data: D.monthly.values, backgroundColor: css("--s1"),
    borderRadius: { topLeft: 4, topRight: 4 }, borderSkipped: "bottom", maxBarThickness: 36, label: "Số phiên" }] }, options: o });
  $("t-monthly").innerHTML = "<tr><th>Tháng</th><th>Số phiên</th></tr>" + D.monthly.labels.map((l, i) => `<tr><td>${l}</td><td>${nf(D.monthly.values[i])}</td></tr>`).join("");
  const hrs = [...Array(24).keys()].map((h) => h + "h");
  mk("c-hourN", { type: "line", data: { labels: hrs, datasets: [
    { label: "Ngày thường", data: D.hourly.N_wd, borderColor: css("--s1"), backgroundColor: css("--s1"), borderWidth: 2, pointRadius: 0, tension: 0.25 },
    { label: "Cuối tuần", data: D.hourly.N_we, borderColor: css("--s2"), backgroundColor: css("--s2"), borderWidth: 2, pointRadius: 0, tension: 0.25 }] },
    options: baseOpts("xe / giờ") });
}

// ================================================================ 2. dự báo
function drawForecast() {
  const F = D.forecast, nW = Math.ceil(F.series.N_actual.length / 168);
  seg($("f-target"), [["N", "Số xe đến (N)"], ["L", "Tải trạm (L)"]], st.target, (v) => { st.target = v; drawForecast(); });
  const sel = $("f-week");
  if (!sel.options.length) {
    for (let w = 0; w < nW; w++) { const a = addMin(F.t0, w * 168 * 60), b = addMin(F.t0, Math.min((w + 1) * 168, F.series.N_actual.length) * 60 - 60);
      sel.add(new Option(`${pad(a.getDate())}/${pad(a.getMonth() + 1)} – ${pad(b.getDate())}/${pad(b.getMonth() + 1)}`, w)); }
    sel.onchange = () => { st.week = +sel.value; drawForecast(); };
  }
  sel.value = st.week;
  chips($("f-models"), F.models, st.models, (m) => MCOL[m], (m) => m, drawForecast);
  const i0 = st.week * 168, i1 = Math.min(i0 + 168, F.series.N_actual.length);
  const labels = []; for (let i = i0; i < i1; i++) { const d = addMin(F.t0, i * 60); labels.push(`${DAYS[d.getDay()]} ${pad(d.getDate())}/${pad(d.getMonth() + 1)} ${pad(d.getHours())}h`); }
  const ds = [{ label: "Thực tế", data: F.series[`${st.target}_actual`].slice(i0, i1), borderColor: css("--text"), backgroundColor: css("--text"),
                borderWidth: 2, pointRadius: 0, order: 0 }];
  F.models.filter((m) => st.models.has(m)).forEach((m) => ds.push({ label: m, data: F.series[`${st.target}_${m}`].slice(i0, i1),
    borderColor: css(MCOL[m]), backgroundColor: css(MCOL[m]), borderWidth: 1.5, pointRadius: 0, borderDash: m.startsWith("Seasonal") || m.startsWith("Hist") ? [5, 4] : [] }));
  const o = baseOpts(st.target === "N" ? "xe / giờ" : "kW");
  o.scales.x.ticks.callback = function (v) { const l = this.getLabelForValue(v); return l.endsWith(" 00h") ? l.slice(0, 8) : null; };
  o.scales.x.ticks.autoSkip = false;
  mk("c-forecast", { type: "line", data: { labels, datasets: ds }, options: o });

  const rows = F.metrics.filter((r) => r.target === st.target).sort((a, b) => a.MAE - b.MAE);
  const best = rows[0].model;
  $("t-metrics").innerHTML = `<tr><th>Mô hình</th><th>MAE</th><th>RMSE</th><th>So với Seasonal Naive 24h</th></tr>` +
    rows.map((r) => `<tr class="${r.model === best ? "hl" : ""}"><td>${r.model}</td><td>${nf(r.MAE, 3)}</td><td>${nf(r.RMSE, 3)}</td>
      <td>${r["improve_vs_SN24_%"] > 0 ? "tốt hơn " : r["improve_vs_SN24_%"] < 0 ? "kém hơn " : ""}${nf(Math.abs(r["improve_vs_SN24_%"]), 1)}%</td></tr>`).join("");
  const ha = rows.find((r) => r.model === "HistAvg_hour_dow");
  $("f-insight").innerHTML = `<b>${best}</b> tốt nhất cho ${st.target}. Trung bình lịch sử <b>kém hơn cả mốc so sánh ${nf(Math.abs(ha["improve_vs_SN24_%"]), 1)}%</b>: nó lấy trung bình cả giai đoạn đông xe trước 11/2018 nên dự báo quá cao, còn các mô hình dùng giá trị ngày/tuần trước tự theo kịp mức hiện tại.`;
}

// ================================================================ 3. lập lịch
const S = D.sched, PLIST = [...new Set(S.summary.main.map((r) => r.p_max_kW))];
const nDays = Math.floor(S.band.length / 96);
const bandPlugin = { id: "bands", beforeDatasetsDraw(c, args, o) {
  if (!o.band) return; const { ctx, chartArea: a, scales: { x } } = c; const w = x.getPixelForValue(1) - x.getPixelForValue(0);
  ctx.save(); for (let i = 0; i < o.band.length; i++) { const b = o.band[i]; if (b === "n") continue;
    ctx.fillStyle = b === "p" ? css("--band-peak") : css("--band-off"); ctx.fillRect(x.getPixelForValue(i) - w / 2, a.top, w + 0.5, a.bottom - a.top); }
  ctx.restore(); } };
function row(tariff, info, p, pol) { return S.summary[tariff].find((r) => r.info === info && r.p_max_kW === p && r.policy === pol); }
function polsOf(info) { return POLS.filter((p) => S.loads[`${info}/${st.pmax}/${p}`]); }

function drawSched() {
  const M = D.meta;
  $("sched-lead").textContent = `Mô phỏng bước 15 phút trên ${nf(M.sim_sessions)} phiên của khoảng kiểm thử (từ 22/03/2019). Mỗi xe tối đa ${nf(M.port_kw)} kW; trạm không được vượt P_max. ${M.tariff_text}`;
  seg($("s-pmax"), PLIST.map((p) => [p, p + " kW"]), st.pmax, (v) => { st.pmax = v; drawSched(); });
  seg($("s-info"), Object.entries(INFO), st.info, (v) => { st.info = v; drawSched(); });
  const sel = $("s-day");
  if (!sel.options.length) {
    for (let d = 0; d < nDays; d++) { const t = addMin(S.t0, d * 1440); sel.add(new Option(`${DAYS[t.getDay()]} ${pad(t.getDate())}/${pad(t.getMonth() + 1)}/${t.getFullYear()}`, d)); }
    sel.onchange = () => { st.day = +sel.value; drawSched(); };
    $("s-prev").onclick = () => { st.day = Math.max(0, st.day - 1); drawSched(); };
    $("s-next").onclick = () => { st.day = Math.min(nDays - 1, st.day + 1); drawSched(); };
  }
  sel.value = st.day;
  const avail = polsOf(st.info);
  chips($("s-pols"), avail, st.pols, (p) => PCOL[p], (p) => PNAME[p], drawSched);

  // ---- ô số: LP cuốn chiếu so với không điều khiển (cả khoảng mô phỏng)
  const u = row("main", st.info, st.pmax, "FCFS-Uncontrolled"), lp = row("main", st.info, st.pmax, "LP-rolling"),
        llf = row("main", st.info, st.pmax, "LLF");
  $("s-tiles").innerHTML = [
    ["Đỉnh tải", `${nf(u.peak_load_kW, 0)} → ${nf(lp.peak_load_kW, 0)} kW`, `không điều khiển → LP cuốn chiếu`],
    ["Chi phí của LP cuốn chiếu", `${usd(lp.cost_per_kWh_USD)}/kWh`, `không điều khiển: ${usd(u.cost_per_kWh_USD)}/kWh`],
    ["Tiết kiệm của LP", `${nf(lp["cost_saving_vs_uncontrolled_%"], 1)}%`, `LLF: ${nf(llf["cost_saving_vs_uncontrolled_%"], 1)}%`],
    ["Năng lượng LP đáp ứng", `${nf(lp.pct_energy_served, 2)}%`, `LLF: ${nf(llf.pct_energy_served, 2)}%`],
  ].map(([k, v, d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");

  // ---- biểu đồ tải trong ngày
  const i0 = st.day * 96, i1 = i0 + 96, t = addMin(S.t0, i0 * 15);
  $("s-title").textContent = `Công suất trạm ngày ${DAYS[t.getDay()]} ${pad(t.getDate())}/${pad(t.getMonth() + 1)}/${t.getFullYear()} — ${INFO[st.info]}, P_max = ${st.pmax} kW` + (t.getDay() === 0 ? " (Chủ nhật: không có giờ cao điểm)" : "");
  const labels = [...Array(96).keys()].map((k) => `${pad(Math.floor(k / 4))}:${pad((k % 4) * 15)}`);
  const ds = avail.filter((p) => st.pols.has(p)).map((p) => ({ label: PNAME[p], data: S.loads[`${st.info}/${st.pmax}/${p}`].slice(i0, i1),
    borderColor: css(PCOL[p]), backgroundColor: css(PCOL[p]), borderWidth: p === "LP-rolling" ? 2.5 : 1.75, pointRadius: 0, stepped: true,
    borderDash: PDASH[p] || [] }));
  ds.push({ label: `P_max = ${st.pmax} kW`, data: Array(96).fill(st.pmax), borderColor: css("--s8"), borderWidth: 1, borderDash: [6, 4], pointRadius: 0 });
  const o = baseOpts("kW");
  o.scales.x.ticks.callback = function (v) { return v % (this.chart.width < 600 ? 24 : 8) === 0 ? this.getLabelForValue(v) : null; }; o.scales.x.ticks.autoSkip = false;
  o.plugins.bands = { band: S.band.slice(i0, i1) };
  o.plugins.tooltip.callbacks = { label: (c) => ` ${c.dataset.label}: ${nf(c.parsed.y, 1)} kW` };
  mk("c-load", { type: "line", data: { labels, datasets: ds }, options: o, plugins: [bandPlugin] });

  // ---- bảng tổng hợp
  $("s-tab-title").textContent = `Kết quả cả khoảng mô phỏng — ${INFO[st.info]}, P_max = ${st.pmax} kW`;
  const rows = POLS.map((p) => row("main", st.info, st.pmax, p)).filter(Boolean);
  $("t-sched").innerHTML = `<tr><th>Chiến lược</th><th>Chi phí ($/kWh)</th><th>% năng lượng đáp ứng</th><th>Thiếu (kWh)</th><th>Đỉnh tải (kW)</th><th>Sạc giờ cao điểm (kWh)</th><th>Tiết kiệm</th></tr>` +
    rows.map((r) => `<tr class="${r.policy === "LP-rolling" ? "hl" : ""}"><td><span class="sw" style="display:inline-block;width:12px;height:3px;border-radius:2px;vertical-align:middle;margin-right:8px;background:var(${PCOL[r.policy]})"></span>${PNAME[r.policy]}</td>
      <td>${usd(r.cost_per_kWh_USD)}</td><td>${nf(r.pct_energy_served, 2)}%</td><td>${nf(Math.max(0, r.unmet_energy_kWh), 1)}</td><td>${nf(r.peak_load_kW, 1)}</td>
      <td>${nf(r.peak_window_energy_kWh, 0)}</td><td>${nf(r["cost_saving_vs_uncontrolled_%"], 2)}%</td></tr>`).join("");
  const off = row("main", st.info, st.pmax, "Offline-optimal");
  const fc = row("main", st.info, st.pmax, "LP+forecast(RandomForest)");
  const unm = (r) => nf(Math.max(0, r.unmet_energy_kWh), 1);
  const fcTxt = fc ? ` Thêm "xe ảo" từ dự báo số xe đến (RF) vào LP: thiếu ${unm(lp)} → ${unm(fc)} kWh.` : "";
  const dl = row("main", "declared", st.pmax, "LP-rolling"), SM = D.session_metrics;
  if (st.info === "oracle") {
    $("s-insight").innerHTML = `Khi biết đúng kWh cần và giờ rời, <b>LP cuốn chiếu đạt ${usd(lp.cost_per_kWh_USD)}/kWh, so với ${usd(off.cost_per_kWh_USD)}/kWh của nghiệm tối ưu biết trước</b>, dù không biết trước xe nào sẽ đến. Xe đỗ trung bình 5–6 giờ nhưng chỉ cần 1–2 giờ để sạc đủ, nên còn nhiều thời gian để dời việc sạc khỏi giờ cao điểm.` +
      (lp.unmet_energy_kWh - off.unmet_energy_kWh < 1
        ? ` Ở mức P_max này LP đã đạt tối ưu, nên dự báo xe đến không còn gì để cải thiện; chọn 20–30 kW để thấy tác dụng của dự báo.`
        : ` Khoảng cách tới tối ưu biết trước (LP thiếu ${unm(lp)} kWh, tối ưu thiếu ${unm(off)} kWh) là <b>mức lợi tối đa mà bất kỳ dự báo xe đến nào có thể mang lại</b>.` + fcTxt);
  } else if (st.info === "declared") {
    $("s-insight").innerHTML = `Khi chỉ có thông tin khai báo (trên cả năm dữ liệu, kWh khai báo gấp ~1,67 lần thực tế và 57% lệch giờ rời quá 1 giờ), LP vẫn rẻ nhất nhưng <b>chỉ đáp ứng ${nf(lp.pct_energy_served, 2)}% năng lượng, so với ${nf(llf.pct_energy_served, 2)}% của LLF</b>: LP tin xe ở lâu nên hoãn việc sạc, trong khi nhiều xe rời sớm. Tối ưu hóa chỉ phát huy khi thông tin đầu vào tốt.`;
  } else {
    const m = (i) => SM ? SM[i] : null, a = m(1), b = m(2);
    $("s-insight").innerHTML = (a && b ? `Mô hình học máy dự báo nhu cầu của từng xe từ lịch sử của chính người dùng: sai số kWh ${nf(a.MAE_kWh, 2)} → <b>${nf(b.MAE_kWh, 2)} kWh</b>, sai số thời gian đỗ ${nf(a.MAE_duration_h, 2)} → <b>${nf(b.MAE_duration_h, 2)} giờ</b> so với dùng khai báo. ` : "") +
      `Nhờ đó LP sạc thiếu <b>${unm(dl)} → ${unm(lp)} kWh</b> và chi phí ${usd(dl.cost_per_kWh_USD)} → ${usd(lp.cost_per_kWh_USD)}/kWh so với kịch bản khai báo. LLF vẫn sạc đủ hơn (${nf(llf.pct_energy_served, 2)}%): thông tin tốt hơn thu hẹp nhưng chưa xóa được khoảng cách với kịch bản biết đúng.`;
  }
}

// ================================================================ 4. đánh đổi
function drawTradeoff() {
  seg($("t-info"), Object.entries(INFO), st.tinfo, (v) => { st.tinfo = v; drawTradeoff(); });
  const pols = POLS.filter((p) => p !== "FCFS-Uncontrolled" && row("main", st.tinfo, PLIST[0], p));
  const mkDs = (col) => pols.map((p) => ({ label: PNAME[p], data: PLIST.map((x) => { const v = row("main", st.tinfo, x, p)[col]; return col === "unmet_energy_kWh" ? Math.max(0, v) : v; }),
    borderColor: css(PCOL[p]), backgroundColor: css(PCOL[p]), borderWidth: 2, pointRadius: 4, pointHoverRadius: 6,
    borderDash: PDASH[p] || [] }));
  const labels = PLIST.map((p) => p + " kW");
  const o1 = baseOpts("$/kWh"); o1.scales.y.beginAtZero = false;
  o1.plugins.tooltip.callbacks = { label: (c) => ` ${c.dataset.label}: ${usd(c.parsed.y)}/kWh` };
  mk("c-cost", { type: "line", data: { labels, datasets: mkDs("cost_per_kWh_USD") }, options: o1 });
  const o2 = baseOpts("kWh"); o2.plugins.tooltip.callbacks = { label: (c) => ` ${c.dataset.label}: ${nf(c.parsed.y, 1)} kWh` };
  mk("c-unmet", { type: "line", data: { labels, datasets: mkDs("unmet_energy_kWh") }, options: o2 });
}


function drawAll() { drawData(); drawForecast(); drawSched(); drawTradeoff(); }
$("foot").textContent = "Biểu giá SCE TOU-EV-4 (Southern California Edison) – biểu giá nhóm ACN dùng với dữ liệu Caltech (Lee et al., IEEE TSG 2021; acnportal). Đơn vị USD; chi phí gồm phí công suất đỉnh. Trang sinh bởi make_demo_page.py từ results/.";
drawAll();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
