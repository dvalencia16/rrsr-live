import json, os, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

LIVE_PATH = "data_live.json"  # existing file in your repo

# ---- Helpers ----
def rsi_wilder(series, period=14):
    # series: pandas Series of closes, ascending by time
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder’s smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return float(round(rsi.iloc[-1], 2))

def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def safe_write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def fetch_last_price(ticker):
    # Try intraday last; fallback to last close
    t = yf.Ticker(ticker)
    # 1) intraday try
    try:
        df = t.history(period="1d", interval="1m")
        if not df.empty:
            return float(round(df["Close"].iloc[-1], 2))
    except Exception:
        pass
    # 2) daily fallback
    try:
        df = t.history(period="5d", interval="1d")
        if not df.empty:
            return float(round(df["Close"].iloc[-1], 2))
    except Exception:
        pass
    return None

def fetch_rsi14(ticker):
    t = yf.Ticker(ticker)
    # Use up to 120 daily candles for smoothing
    df = t.history(period="6mo", interval="1d")
    if df is None or df.empty or len(df) < 20:
        return None
    return rsi_wilder(df["Close"], period=14)

# ---- Short Squeeze inputs (free, user-supplied) ----
# Edit this dictionary for the tickers you want tracked in the SST table.
# Provide short interest shares, float shares, and 30-day avg volume.
SST_INPUTS = {
    # "PLUG": {"si_shares": 100_000_000, "float_shares": 600_000_000, "avg_vol_30d": 25_000_000, "entry": 3.10, "stop": 2.72, "t1": 3.45, "t2": 3.90}
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
                "User-supplied SI/float/avgvol (free method). Update in scripts/update_live.py."
            ]
        }
    return out

# ---- Main ----
def main():
    tickers = os.getenv("TICKERS", "SCHD,VTI,VXUS,XLV,VNQ,BND,GEMI").split(",")
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    data = safe_load_json(LIVE_PATH)

    # 1) Update prices
    prices = {}
    for tk in tickers:
        prices[tk] = fetch_last_price(tk)
        time.sleep(0.2)  # be polite

    # 2) RSI map
    rsi_map = {}
    for tk in tickers:
        rsi_map[tk] = fetch_rsi14(tk)

    # 3) Merge into live JSON
    # Keep all existing keys; only update/add the ones we manage here.
    data.setdefault("Prices", {})
    for tk in tickers:
        if prices[tk] is not None:
            data["Prices"][tk] = prices[tk]

    data["RSI"] = {tk: (rsi_map[tk] if rsi_map[tk] is not None else "data not available") for tk in tickers}

    # 4) Build SST block from inputs
    data["SST"] = build_sst(prices, data["RSI"])

    # 5) Timestamp
    data["last_update_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 6) Write back
    safe_write_json(LIVE_PATH, data)

if __name__ == "__main__":
    main()
