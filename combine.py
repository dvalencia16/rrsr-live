import json, time, os
h=json.load(open("holdings.json"))
q={x["ticker"]: {"last":x["last"], "rsi":x["rsi"], "ts":x.get("ts")}
   for x in json.load(open("quotes.json"))} if os.path.exists("quotes.json") else {}
m=json.load(open("macro.json")) if os.path.exists("macro.json") else {}
out={"ts": int(time.time()), "holdings": h, "quotes": q, "macro": m}
json.dump(out, open("data_live.json","w"), indent=2)
