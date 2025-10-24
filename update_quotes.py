import os, json, time, requests

API = os.getenv("FINNHUB_KEY")

def rsi_from_closes(cl, n=14):
    if not cl or len(cl) < n+1: return None
    gains=[max(0,cl[i]-cl[i-1]) for i in range(1,len(cl))]
    losses=[max(0,cl[i-1]-cl[i]) for i in range(1,len(cl))]
    ag=sum(gains[:n])/n; al=sum(losses[:n])/n or 1e-9
    for i in range(n+1,len(cl)):
        ch=cl[i]-cl[i-1]; g=max(0,ch); l=max(0,-ch)
        ag=(ag*(n-1)+g)/n; al=(al*(n-1)+l)/n or 1e-9
    rs=ag/al
    return round(100 - 100/(1+rs), 2)

def fh_quote(sym):
    r=requests.get("https://finnhub.io/api/v1/quote",
                   params={"symbol":sym,"token":API}, timeout=10)
    r.raise_for_status(); return r.json().get("c")

def fh_closes(sym, days=200):
    end=int(time.time()); start=end - days*86400
    r=requests.get("https://finnhub.io/api/v1/stock/candle",
                   params={"symbol":sym,"resolution":"D","from":start,"to":end,"token":API},
                   timeout=10)
    j=r.json(); return j["c"] if j.get("s")=="ok" else []

def load_holdings():
    with open("holdings.json") as f:
        h=json.load(f)
    tickers=set()
    for arr in h.values():
        for row in arr:
            tickers.add(row["ticker"])
    # ensure Fannie/Freddie included
    tickers.update({"FNMA","FMCC"})
    return sorted(tickers)

def build_quotes():
    out=[]; ts=int(time.time())
    for t in load_holdings():
        last=None; rsi=None
        try:
            last=fh_quote(t)
            cls=fh_closes(t)
            rsi=rsi_from_closes(cls) if cls else None
        except Exception:
            pass
        out.append({"ticker":t,"last":last,"rsi":rsi,"ts":ts})
    with open("quotes.json","w") as f:
        json.dump(out,f,indent=2)

if __name__=="__main__":
    build_quotes()
