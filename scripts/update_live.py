import json, os, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

LIVE_PATH = "data_live.json"  # must exist at repo root

# ---------- Helpers ----------
def rsi_wilder(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return float(round(rsi.iloc[-1], 2))

def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def fetch_last_price(ticker):
    t = yf.Ticker(ticker)
    # intraday attempt
    try:
        df = t.history(period="1d", interval="1m")
        if not df.empty:
            return float(round(df["Close"].iloc[-1], 2))
    except Exception:
        pass
    # daily fallback
    try:
        df = t.history(period="5d", interval="1d")
        if not df.empty:
            return float(round(df["Close"].iloc[-1], 2))
    except Exception:
        pass
    return None

def fetch_rsi14(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period="6mo", interval="1d")
    if df is None or df.empty or len(df) < 20:
        return None
    return rsi_wilder(df["Close"], period=14)

# ---------- Short Squeeze inputs (you fill these) ----------
# Provide: short interest shares, float shares, 30-day avg volume, plus your entry/stop/targets.
SST_INPUTS = {
  "PLUG": {"si_shares": 100000000, "float_shares": 600000000, "avg_vol_30d": 25000000,
           "entry": 3.10, "stop": 2.72, "t1": 3.45, "t2": 3.90},
  "RIOT": {"si_shares": 36000000,  "float_shares": 260000000, "avg_vol_30d": 21000000,
           "entry": 8.60, "stop": 7.80, "t1": 9.40, "t2": 10.20},
  "AFRM": {"si_shares": 21000000,  "float_shares": 250000000, "avg_vol_30d": 12000000,
           "entry": 28.50, "stop": 26.90, "t1": 30.80, "t2": 33.00}
}



def build_sst(prices, rsi_map):
    out = {}
    for tk, vals in SST_INPUTS.items():
        si_sh = vals.get("si_shares")
        flt = vals.get("float_shares")
        avgv = vals.get("avg_vol_30d")
        entry = vals.get("entry")
        stop = vals.get("stop")
        t1 = vals.get("t1")
        t2 = vals.get("t2")

        rsi_val = rsi_map.get(tk)
        last = prices.get(tk)
def _to_num(x):
    try:
        return float(x)
    except Exception:
        return None

def squeeze_score(item):
    # Higher is better: high SI%, high DTC, low RSI
    rsi = _to_num(item.get("rsi"))
    si_pct = _to_num(item.get("si_percent_float"))
    dtc = _to_num(item.get("days_to_cover"))

    if si_pct is None and dtc is None and rsi is None:
        return 0.0

    si_norm  = min(max((si_pct or 0.0), 0.0), 40.0) / 40.0      # cap 40%
    dtc_norm = min(max((dtc or 0.0), 0.0), 10.0) / 10.0         # cap 10 days
    rsi_norm = 0.0 if rsi is None else max(0.0, 70.0 - rsi) / 70.0  # oversold better

    score = 0.5*si_norm + 0.3*dtc_norm + 0.2*rsi_norm
    return round(score*100, 1)

def top5_from_sst(sst_dict):
    rows = []
    for tk, v in sst_dict.items():
        sc = squeeze_score(v)
        rows.append({
            "ticker": tk,
            "score": sc,
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

        def safe_pct(x, y):
            try:
                return round(100 * x / y, 2) if x is not None and y else None
            except Exception:
                return None

        def safe_div(x, y):
            try:
                return round(x / y, 2) if x is not None and y else None
            except Exception:
                return None

        si_pct_float = safe_pct(si_sh, flt)
        dtc = safe_div(si_sh, avgv)

        out[tk] = {
            "rsi": rsi_val if rsi_val is not None else "data not available",
            "si_percent_float": si_pct_float if si_pct_float is not None else "data not available",
            "days_to_cover": dtc if dtc is not None else "data not available",
            "entry": entry if entry is not None else "data not available",
            "stop": stop if stop is not None else "data not available",
            "t1": t1 if t1 is not None else "data not available",
            "t2": t2 if t2 is not None else "data not available",
            "last": last if last is not None else "data not available",
            "sources": [
                "User-supplied SI/float/avgvol (free method). Edit in scripts/update_live.py"
            ]
        }
    return out

# ---------- Main ----------
def main():
    tickers = os.getenv("TICKERS", "SCHD,VTI,VXUS,XLV,VNQ,BND,GEMI").split(",")
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    data = safe_load_json(LIVE_PATH)

    # Prices
    prices = {}
    for tk in tickers:
        prices[tk] = fetch_last_price(tk)
        time.sleep(0.2)

    # RSI
    rsi_map = {}
    for tk in tickers:
        rsi_map[tk] = fetch_rsi14(tk)

    # Merge into JSON
    data.setdefault("Prices", {})
    for tk in tickers:
        if prices[tk] is not None:
            data["Prices"][tk] = prices[tk]
    data["RSI"] = {tk: (rsi_map[tk] if rsi_map[tk] is not None else "data not available") for tk in tickers}

    # SST block from your inputs
    data["SST"] = build_sst(prices, data["RSI"])

    # Timestamp
    data["last_update_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    safe_write_json(LIVE_PATH, data)

if __name__ == "__main__":
    main()
