import os, time, threading
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, request

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
SCAN_KEY = os.environ.get("SCAN_KEY", "")
MAX_ALERTS = int(os.environ.get("MAX_ALERTS", 5))
MIN_WIN = float(os.environ.get("MIN_WIN_RATE", 55))
MIN_N = int(os.environ.get("MIN_SAMPLES", 8))
OFFSET = float(os.environ.get("PRICE_OFFSET", 0))
RR, LOOKBACK, STALE_MIN = 2.0, 60, 20

app = Flask(__name__)
lock = threading.Lock()
state = {"last_ts": None, "day": None, "sent": 0, "levels": [],
         "stats": None, "stats_time": 0, "last_msg": "not scanned yet"}

def tg(text):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": text}, timeout=15)
    return r.ok

def load():
    df = yf.Ticker("GC=F").history(period="59d", interval="5m")
    if df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = df.index.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(minutes=5) <= now]

def prepare(df):
    d = df.copy()
    d["rv"] = (d.Volume / d.Volume.rolling(20).mean().shift(1)).replace([np.inf], np.nan)
    tr = pd.concat([d.High - d.Low, (d.High - d.Close.shift()).abs(),
                    (d.Low - d.Close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["hi"] = d.High.rolling(LOOKBACK).max().shift(1)
    d["lo"] = d.Low.rolling(LOOKBACK).min().shift(1)
    m15 = d.Close.resample("15min").last().dropna()
    e20, e50 = m15.ewm(span=20, adjust=False).mean(), m15.ewm(span=50, adjust=False).mean()
    t = pd.Series(0, index=m15.index)
    t[(e20 > e50) & (m15 > e50)] = 1
    t[(e20 < e50) & (m15 < e50)] = -1
    d["trend"] = t.shift(1).reindex(d.index, method="ffill").fillna(0)
    return d

def arrays(d):
    return {"O": d.Open.to_numpy(), "H": d.High.to_numpy(), "L": d.Low.to_numpy(),
            "C": d.Close.to_numpy(), "RV": d.rv.to_numpy(), "ATR": d.atr.to_numpy(),
            "HI": d.hi.to_numpy(), "LO": d.lo.to_numpy(), "TR": d.trend.to_numpy()}

def detect(a, i):
    O, H, L, C, RV, HI, LO = (a[k] for k in ("O", "H", "L", "C", "RV", "HI", "LO"))
    atr, tr = a["ATR"][i], a["TR"][i]
    out = []
    if i < LOOKBACK + 26 or tr == 0 or atr != atr:
        return out
    buf = 0.15 * atr

    def add(setup, side, entry, stop, level, why):
        risk = abs(entry - stop)
        if not (0.4 * atr <= risk <= 3 * atr):
            return
        sgn = 1 if side == "BUY" else -1
        out.append(dict(setup=setup, side=side, entry=entry, stop=stop,
                        tp=entry + sgn * RR * risk, vu=entry + sgn * 0.25 * risk,
                        level=level, why=why))

    # Setup 1: rejection at key level
    s = i - 1
    rng = H[s] - L[s]
    if rng > 0 and RV[s] >= 2:
        if tr == -1 and H[s] > HI[s] and C[s] < HI[s] and \
           (H[s] - max(O[s], C[s])) >= 0.5 * rng and C[i] < L[s] and C[i] < O[i]:
            entry, stop = C[i], H[s] + buf
            if LO[i] <= entry - RR * (stop - entry):
                add("Rejection at key level", "SELL", entry, stop, HI[s],
                    f"Price swept above the recent high (LVL) on {RV[s]:.1f}x average volume, "
                    "closed back below with a long upper wick, 15m trend down, next candle "
                    "closed below the rejection low.")
        if tr == 1 and L[s] < LO[s] and C[s] > LO[s] and \
           (min(O[s], C[s]) - L[s]) >= 0.5 * rng and C[i] > H[s] and C[i] > O[i]:
            entry, stop = C[i], L[s] - buf
            if HI[i] >= entry + RR * (entry - stop):
                add("Rejection at key level", "BUY", entry, stop, LO[s],
                    f"Price swept below the recent low (LVL) on {RV[s]:.1f}x average volume, "
                    "closed back above with a long lower wick, 15m trend up, next candle "
                    "closed above the rejection high.")

    # Setup 2: breakout, low-volume pullback, resume
    for k in (1, 2, 3):
        b = i - k - 1
        pb_rv = np.nanmean(RV[b + 1:i]) if not np.all(np.isnan(RV[b + 1:i])) else 9
        resume_vol = RV[i] >= 1.2 and RV[i] > RV[i - 1]
        if tr == 1:
            lvl = HI[b]
            if C[b] > lvl and C[b] > O[b] and RV[b] >= 1.5 and min(C[b + 1:i]) > lvl \
               and min(L[b + 1:i]) <= lvl + 0.5 * atr and pb_rv < 0.9 \
               and C[i] > O[i] and C[i] > H[i - 1] and resume_vol:
                add("Breakout-pullback-resume", "BUY", C[i], min(L[b + 1:i + 1]) - buf, lvl,
                    f"Closed above LVL on {RV[b]:.1f}x volume, pulled back {k} candle(s) on low "
                    f"volume ({pb_rv:.1f}x) and held, resumed up on rising volume, 15m trend up.")
        if tr == -1:
            lvl = LO[b]
            if C[b] < lvl and C[b] < O[b] and RV[b] >= 1.5 and max(C[b + 1:i]) < lvl \
               and max(H[b + 1:i]) >= lvl - 0.5 * atr and pb_rv < 0.9 \
               and C[i] < O[i] and C[i] < L[i - 1] and resume_vol:
                add("Breakout-pullback-resume", "SELL", C[i], max(H[b + 1:i + 1]) + buf, lvl,
                    f"Closed below LVL on {RV[b]:.1f}x volume, pulled back {k} candle(s) on low "
                    f"volume ({pb_rv:.1f}x) and held, resumed down on rising volume, 15m trend down.")
    return out

def outcome(a, i, side, stop, tp, horizon=48):
    for j in range(i + 1, min(len(a["H"]), i + 1 + horizon)):
        if side == "BUY":
            if a["L"][j] <= stop: return 0
            if a["H"][j] >= tp: return 1
        else:
            if a["H"][j] >= stop: return 0
            if a["L"][j] <= tp: return 1
    return None

def backtest(d):
    a, res = arrays(d), {}
    for i in range(LOOKBACK + 26, len(d) - 1):
        for s in detect(a, i):
            o = outcome(a, i, s["side"], s["stop"], s["tp"])
            if o is not None:
                w, n = res.get(s["setup"], (0, 0))
                res[s["setup"]] = (w + o, n + 1)
    return res

def scan():
    d = load()
    if d is None or len(d) < 150:
        return "no data"
    d = prepare(d)
    ts = d.index[-1]
    if pd.Timestamp.now(tz="UTC") - ts > pd.Timedelta(minutes=STALE_MIN):
        return "market closed or data stale"
    if state["last_ts"] == ts:
        return "no new candle"
    state["last_ts"] = ts
    day = ts.strftime("%Y-%m-%d")
    if state["day"] != day:
        state.update(day=day, sent=0, levels=[])
    if state["sent"] >= MAX_ALERTS:
        return "daily cap reached"
    if state["stats"] is None or time.time() - state["stats_time"] > 3600:
        state["stats"], state["stats_time"] = backtest(d), time.time()
    a, i = arrays(d), len(d) - 1
    atr = a["ATR"][i]
    sent = 0
    for s in detect(a, i):
        if state["sent"] >= MAX_ALERTS:
            break
        w, n = state["stats"].get(s["setup"], (0, 0))
        if n < MIN_N or 100 * w / n < MIN_WIN:
            continue
        if any(abs(s["level"] - L) < 0.5 * atr for L in state["levels"]):
            continue
        P = lambda x: x + OFFSET
        t = (ts + pd.Timedelta(minutes=5)).tz_convert("Africa/Lagos").strftime("%H:%M")
        icon = "🟢" if s["side"] == "BUY" else "🔴"
        msg = (f"{icon} XAUUSD {s['side']} (5m)\nSetup: {s['setup']}\n"
               f"Entry: {P(s['entry']):.2f}\nStop loss: {P(s['stop']):.2f}\n"
               f"TP: {P(s['tp']):.2f} (1:{RR:g})\n"
               f"Skip if price already beyond: {P(s['vu']):.2f}\n"
               f"Historical win rate: {100 * w / n:.0f}% ({n} setups)\n"
               f"Reason: {s['why'].replace('LVL', f'{P(s['level']):.2f}')}\n"
               f"Candle closed: {t} Nigeria time\n"
               "Levels come from gold futures; confirm against your broker price.")
        if tg(msg):
            state["sent"] += 1
            state["levels"].append(s["level"])
            sent += 1
    return f"scanned {ts}, alerts sent: {sent}"

@app.route("/")
def home():
    return f"Gold bot running. Alerts today: {state['sent']}. Last scan: {state['last_msg']}"

@app.route("/scan")
def scan_route():
    if not SCAN_KEY or request.args.get("key") != SCAN_KEY:
        return "forbidden", 403
    with lock:
        try:
            msg = scan()
        except Exception as e:
            msg = f"error: {e}"
        state["last_msg"] = msg
    return msg

@app.route("/test")
def test_route():
    if not SCAN_KEY or request.args.get("key") != SCAN_KEY:
        return "forbidden", 403
    return "sent" if tg("✅ Gold bot test message") else "failed: check token and chat ID"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
