"""
Tải dữ liệu phiên sạc THẬT của site Caltech từ ACN-Data API (mặc định 01/05/2018 - 30/04/2019).

Cần API token miễn phí: đăng ký tại https://ev.caltech.edu/register rồi lấy token ở trang tài khoản.

    $env:ACN_API_TOKEN = "<token của bạn>"      # PowerShell (hoặc truyền --token)
    python download_caltech.py
    python download_caltech.py --start 2018-05-01 --end 2019-04-30

Sinh ra: data/raw/caltech_YYYYMMDD_YYYYMMDD.jsonl (mỗi đoạn 30 ngày một file, ngày cuối tính cả)
         data/manifest_caltech_<start>_<end>.json
Đoạn nào đã có file thì bỏ qua (chạy lại để tải tiếp khi bị ngắt). Muốn tải lại thì xoá file đó.
"""
import argparse
import base64
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
API = "https://ev.caltech.edu/api/v1/"
TZ = "America/Los_Angeles"


def gmt(ts_local):
    """Mốc giờ địa phương -> chuỗi GMT mà API dùng trong điều kiện where."""
    return ts_local.tz_localize(TZ).tz_convert("UTC").strftime("%a, %d %b %Y %H:%M:%S GMT")


def get_json(url, token, retries=3):
    auth = base64.b64encode(f"{token}:".encode()).decode()     # basic auth: token làm username, mật khẩu rỗng
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit("API từ chối token (401/403). Kiểm tra lại ACN_API_TOKEN.")
            if i == retries - 1:
                raise
        except urllib.error.URLError:
            if i == retries - 1:
                raise
        time.sleep(5 * (i + 1))


def fetch_chunk(site, t0, t1, token, page_size):
    """Mọi phiên có connectionTime trong [t0, t1) giờ địa phương, đi hết các trang."""
    where = f'connectionTime>="{gmt(t0)}" and connectionTime<"{gmt(t1)}"'
    url = (f"{API}sessions/{site}?" +
           urllib.parse.urlencode({"where": where, "sort": "connectionTime", "max_results": page_size}))
    items, expected = [], None
    while url:
        page = get_json(url, token)
        items.extend(page.get("_items", []))
        expected = page.get("_meta", {}).get("total", expected)
        nxt = page.get("_links", {}).get("next")
        # API trả link trang sau chưa mã hoá (còn dấu cách, ngoặc kép) -> mã hoá lại, giữ nguyên ký tự phân tách
        url = API + urllib.parse.quote(nxt["href"], safe="/?&=%") if nxt else None
    return items, expected


def main(site, start, end, chunk_days, token, page_size):
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    first, stop = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)   # end tính cả ngày đó
    chunks, total = [], 0
    t0 = first
    while t0 < stop:
        t1 = min(t0 + pd.Timedelta(days=chunk_days), stop)
        name = f"{site}_{t0:%Y%m%d}_{t1 - pd.Timedelta(days=1):%Y%m%d}"
        path = raw / f"{name}.jsonl"
        if path.exists():
            n = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            chunks.append({"chunk": name, "n": n, "expected": None, "source": "existing file"})
            print(f"  {name}: đã có ({n} phiên), bỏ qua")
        else:
            items, expected = fetch_chunk(site, t0, t1, token, page_size)
            n = len(items)
            if expected is not None and n != expected:
                print(f"  CẢNH BÁO {name}: nhận {n} phiên nhưng API báo {expected}")
            tmp = path.with_suffix(".part")     # ghi xong mới đổi tên -> file dở dang không bị coi là đã tải
            tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
            tmp.replace(path)
            chunks.append({"chunk": name, "n": n, "expected": expected, "source": "api"})
            print(f"  {name}: {n} phiên")
        total += n
        t0 = t1

    manifest = {"site": site, "start_local_date": start, "end_local_date": end, "timezone": TZ,
                "chunk_days": chunk_days, "rows_downloaded": total,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "chunks": chunks}
    mpath = ROOT / "data" / f"manifest_{site}_{start}_{end}.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTổng cộng {total} phiên (chưa bỏ trùng). Manifest: {mpath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="caltech", choices=["caltech", "jpl", "office001"])
    ap.add_argument("--start", default="2018-05-01", help="ngày đầu (giờ địa phương, khớp config.DATA_START)")
    ap.add_argument("--end", default="2019-04-30", help="ngày cuối, tính cả ngày này")
    ap.add_argument("--chunk-days", type=int, default=30)
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--token", default=os.environ.get("ACN_API_TOKEN"), help="mặc định lấy từ biến môi trường ACN_API_TOKEN")
    a = ap.parse_args()
    if not a.token:
        raise SystemExit("Thiếu API token: đặt biến môi trường ACN_API_TOKEN hoặc truyền --token.")
    main(a.site, a.start, a.end, a.chunk_days, a.token, a.page_size)
