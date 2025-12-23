# scripts/build_live_json.py
# Build data_live.json with prices, RSI(14), VIX, and a simple short-squeeze proxy.
# Output file: data_live.json at repo root.

import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import yfinance as yf

# ----- config -----
TICKERS = ["VTI","VXUS","SCHD","BND","BNDX","XLV","VNQ","GEMI","SPY","QQQ","^VIX"]
CORE = ["VTI","VXUS","SCHD","BND","BNDX","XLV","VNQ","GEMI","SPY","QQQ"]
TZ = timezone.utc  # JSON timestamps in UTC

def rsi_from_series(close: pd.Series, period: int = 14):
    close = close.dropna()
    if close.shape[0] < period + 1:
        return None
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).rolling(period).mean().iloc[-1]
    roll_down = pd.Series(loss, index=close.index).rolling(period).mean().iloc[-1]
    if roll_down == 0:
        return 100.0
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return float(round(rsi, 2))

def get_history(ticker: str, days: int = 60) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(period=f"{days}d", interval="1d", auto_adjust=True)
    return df

def snap_ticker(t: str) -> dict:
    try:
        df = get_history(t)
        if df.empty or "Close" not in df:
            return {
                "ticker": t, "last": None, "prev_close": None, "change": None,
                "change_pct": None, "rsi14": None, "currency": None,
                "source": "yfinance", "error": "no_history"
            }
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else None
        chg = None if prev is None else round(last - prev, 4)
        chg_pct = None if prev is None or prev == 0 else round((last/prev - 1)*100, 3)
        rsi = rsi_from_series(df["Close"])

        finfo = yf.Ticker(t).fast_info
        ccy = getattr(finfo, "currency", None)

        return {
            "ticker": t,
            "last": round(last, 4),
            "prev_close": None if prev is None else round(prev, 4),
            "change": chg,
            "change_pct": chg_pct,
            "rsi14": rsi,
            "currency": ccy,
            "source": "yfinance"
        }
    except Exception as e:
        return {
            "ticker": t, "last": None, "prev_close": None, "change": None,
            "change_pct": None, "rsi14": None, "currency": None,
            "source": "yfinance", "error": str(e)
        }

def squeeze_proxy(t: str, df: pd.DataFrame):
    # Heuristic: RSI < 35, 5d return <= -5%, and today > 0%
    try:
        if df is None or df.empty:
            return None
        close = df["Close"].dropna()
        if len(close) < 20:
            return None
        rsi = rsi_from_series(close)
        if rsi is None:
            return None
        ret_5d = float((close.iloc[-1]/close.iloc[-6] - 1) * 100) if len(close) >= 6 else None
        day_ret = float((close.iloc[-1]/close.iloc[-2] - 1) * 100) if len(close) >= 2 else None
        ok = (rsi is not None and rsi < 35) and (ret_5d is not None and ret_5d <= -5.0) and (day_ret is not None and day_ret > 0)
        if not ok:
            return None
        return {
            "ticker": t,
            "rsi14": round(rsi, 2),
            "ret_5d_pct": round(ret_5d, 2) if ret_5d is not None else None,
            "day_ret_pct": round(day_ret, 2) if day_ret is not None else None,
            "note": "Proxy squeeze flag (RSI<35, -5%/5d, >0% today)"
        }
    except Exception:
        return None

def build():
    now_utc = datetime.now(TZ).replace(microsecond=0)
    out = {
        "generated_at_utc": now_utc.isoformat(),
        "universe": TICKERS,
        "tickers": {},
        "market": {},
        "squeeze_top5": [],
        "notes": {
            "rsi_period": 14,
            "sources": ["Yahoo Finance via yfinance"],
            "disclaimer": "Education only. Not investment advice."
        }
    }

    # Snap tickers and cache history
    hist_cache = {}
    for t in TICKERS:
        out["tickers"][t] = snap_ticker(t)
        try:
            hist_cache[t] = get_history(t)
        except Exception:
            hist_cache[t] = pd.DataFrame()

    # VIX summary
    vix = out["tickers"].get("^VIX", {})
    out["market"]["vix"] = {
        "last": vix.get("last"),
        "change_pct": vix.get("change_pct"),
        "source": vix.get("source", "yfinance")
    }

    # Short-squeeze proxy
    candidates = []
    for t in CORE:
        df = hist_cache.get(t)
        flag = squeeze_proxy(t, df)
        if flag:
            # rank: more negative 5d first, then lower RSI
            rank_key = (flag["ret_5d_pct"], flag["rsi14"])
            candidates.append((rank_key, flag))
    candidates.sort(key=lambda x: (x[0][0], x[0][1]) if x[0][0] is not None and x[0][1] is not None else (999, 999))
    out["squeeze_top5"] = [c[1] for c in candidates[:5]]

    # Write JSON
    with open("data_live.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    build()
