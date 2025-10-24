import json, time, requests

def close_stooq(sym):
    try:
        j=requests.get(f"https://stooq.com/q/l/?s={sym}&i=d&f=json", timeout=12).json()
        return float(j[0]["close"])
    except: return None

def build():
    out={"ts": int(time.time())}
    out["VIX"]  = close_stooq("^vix")
    out["DXY"]  = close_stooq("dxy")
    out["US10Y"]= close_stooq("us10y")
    out["US02Y"]= close_stooq("us02y")
    out["SPREAD_2s10s_bps"] = round((out["US10Y"]-out["US02Y"])*100,1) if out["US10Y"] and out["US02Y"] else None
    out["PUT_CALL_TOTAL"] = None
    out["BREADTH_50dma_pct"] = None
    json.dump(out, open("macro.json","w"), indent=2)

if __name__=="__main__":
    build()
