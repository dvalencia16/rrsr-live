import json, os, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

LIVE_PATH = "data_live.json"
SST_PATH  = "sst_inputs.json"

def rsi_wilder(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-12))
    rsi = 100 - (100 / (1 + rs))
    return float(round(rsi.iloc[-1], 2))

def safe_load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def safe_write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def fetch_last_price(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            return float(round(df["Close"].iloc[-1], 2))
    except Exception:
        pass
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            return float(round(df["Close"].iloc[-1], 2))
    except Exception:
        pass
    return None

def fetch_rsi14(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty or len(df) < 20:
            return None
        return rsi_wilder(df["Close"], period=14)
    except Exception:
        return None

def build_sst(prices, rsi_map, sst_inputs):
    out = {}
    for tk, vals in sst_inputs.items():
        si_sh = vals.get("si_shares")
        flt   = vals.get("float_shares")
        avgv  = vals.get("avg_vol_30d")
        entry = vals.get("entry")
        stop  = vals.get("stop")
        t1    = vals.get("t1")
        t2    = vals.get("t2")
        rsi_val = rsi_map.get(tk)
        last    = prices.get(tk)
        def pct(x, y):
            try:
                return round(100 * x / y, 2) if x is not None and y else None
            except Exception:
                return None
        def div(x, y):
            try:
                return round(x / y, 2) if x is not None and y else None
            except Exception:
                return None
        out[tk] = {
            "rsi": rsi_val if rsi_val is not None else "data not available",
            "si_percent_float": pct(si_sh, flt) if pct(si_sh, flt) is not None else "data not available",
            "days_to_cover": div(si_sh, avgv) if div(si_sh, avgv) is not None else "data not available",
            "entry": entry if entry is not None else "data not available",
            "stop":  stop  if stop  is not None else "data not available",
            "t1":    t1    if t1    is not None else "data not available",
            "t2":    t2    if t2    is not None else "data not available",
            "last":  last  if last  is not None else "data not available",
            "sources": ["User-supplied SI/float/avgvol in sst_inputs.json"]
        }
    return out

def _to_num(x):
    try:
        return float(x)
    except Exception:
        return None

def squeeze_score(item):
    rsi = _to_num(item.get("rsi"))
    si  = _to_num(item.get("si_percent_float"))
    dtc = _to_num(item.get("days_to_cover"))
    si_norm  = min(max((si or 0.0), 0.0), 40.0) / 40.0
    dtc_norm = min(max((dtc or 0.0), 0.0), 10.0) / 10.0
    rsi_norm = 0.0 if rsi is None else max(0.0, 70.0 - rsi) / 70.0
    return round((0.5*si_norm + 0.3*dtc_norm + 0.2*rsi_norm)*100, 1)

def top5_from_sst(sst_dict):
    rows = []
    for tk, v in sst_dict.items():
        rows.append({
            "ticker": tk,
            "score": squeeze_score(v),
            "rsi": v.get("rsi"),
            "si_percent_float": v.get("si_percent_float"),
            "days_to_cover": v.get("days_to_cover"),
            "entry": v.get("entry"),
            "stop": v.get("stop"),
            "targets": [v.get("t1"), v.get("t2")],
            "sources": v.get("sources", [])
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:5]

def ensure_defaults():
    if not os.path.exists(LIVE_PATH):
        safe_write_json(LIVE_PATH, {"Prices": {}, "RSI": {}, "SST": {}, "SST_TOP5": [], "last_update_utc": ""})
    if not os.path.exists(SST_PATH):
        safe_write_json(SST_PATH, {
            "PLUG": {"si_shares": 100000000, "float_shares": 600000000, "avg_vol_30d": 25000000, "entry": 3.10, "stop": 2.72, "t1": 3.45, "t2": 3.90},
            "RIOT": {"si_shares": 36000000,  "float_shares": 260000000, "avg_vol_30d": 21000000, "entry": 8.60, "stop": 7.80, "t1": 9.40, "t2": 10.20},
            "AFRM": {"si_shares": 21000000,  "float_shares": 250000000, "avg_vol_30d": 12000000, "entry": 28.50, "stop": 26.90, "t1": 30.80, "t2": 33.00},
            "NVTS": {"si_shares": 9500000,   "float_shares": 110000000, "avg_vol_30d": 2400000,  "entry": 5.10, "stop": 4.70, "t1": 5.70, "t2": 6.30},
            "SOFI": {"si_shares": 120000000, "float_shares": 930000000, "avg_vol_30d": 46000000, "entry": 7.50, "stop": 6.90, "t1": 8.20, "t2": 8.90}
        })

def main():
    ensure_defaults()
    tickers = os.getenv("TICKERS", "SCHD,VTI,VXUS,XLV,VNQ,BND,GEMI").split(",")
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    data = safe_load_json(LIVE_PATH, {"Prices": {}, "RSI": {}, "SST": {}, "SST_TOP5": [], "last_update_utc": ""})

    prices = {}
    for tk in tickers:
        prices[tk] = fetch_last_price(tk)
        time.sleep(0.2)

    rsi_map = {}
    for tk in tickers:
        rsi_map[tk] = fetch_rsi14(tk)

    data.setdefault("Prices", {})
    for tk in tickers:
        if prices[tk] is not None:
            data["Prices"][tk] = prices[tk]
    data["RSI"] = {tk: (rsi_map[tk] if rsi_map[tk] is not None else "data not available") for tk in tickers}

    sst_inputs = safe_load_json(SST_PATH, {})
    sst = build_sst(prices, data["RSI"], sst_inputs)
    data["SST"] = sst
    data["SST_TOP5"] = top5_from_sst(sst)

    data["last_update_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    safe_write_json(LIVE_PATH, data)

if __name__ == "__main__":
    main()
