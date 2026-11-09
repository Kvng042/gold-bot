import os, time, threading
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, request

def env(name, default):
    return type(default)(os.environ.get(name, default))

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
SCAN_KEY = os.environ.get("SCAN_KEY", "")
MAX_ALERTS = env("MAX_ALERTS", 5)
MIN_WIN = env("MIN_WIN_RATE", 55.0)
MIN_N = env("MIN_SAMPLES", 8)
OFFSET = env("PRICE_OFFSET", 0.0)
RR = env("RR", 2.0)
LOOKBACK = env("LOOKBACK", 60)
TREND_FILTER = env("TREND_FILTER", 1)
REJ_VOL = env("REJ_VOL", 1.5)
WICK = env("WICK", 0.4)
BRK_VOL = env("BRK_VOL", 1.3)
PB_VOL = env("PB_VOL", 1.0)
RES_VOL = env("RES_VOL", 1.1)
STALE_MIN = 20
SETUPS = ["Rejection at key level", "Breakout-pullback-resume"]

app = Flask(__name__)
lock = threading.Lock()
state = {"last_ts": None, "day": None, "sent": 0, "levels": [], "stats": None,
         "stats_time": 0, "last_msg": "not scanned yet", "diag": "none yet", "blocked": []}

def P(x):
    return x + OFFSET

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
    e20 = m15.ewm(span=20, adjust=False).mean()
    e50 = m15.ewm(span=50, adjust=False).mean()
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
    if i < LOOKBACK + 26 or atr != atr:
        return out
    if TREND_FILTER and tr == 0:
        return out
    dn = (tr == -1) or not TREND_FILTER
    up = (tr == 1) or not TREND_FILTER
    dn_txt = "15m trend down" if tr == -1 else "15m trend neutral"
    up_txt = "15m trend up" if tr == 1 else "15m trend neutral"
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
    if rng > 0 and RV[s] >= REJ_VOL:
        if dn and H[s] > HI[s] and C[s] < HI[s] and \
           (H[s] - max(O[s], C[s])) >= WICK * rng and C[i] < L[s] and C[i] < O[i]:
            entry, stop = C[i], H[s] + buf
            if LO[i] <= entry - RR * (stop - entry):
                add(SETUPS[0], "SELL", entry, stop, HI[s],
                    f"Price swept above the recent high (LVL) on {RV[s]:.1f}x average volume, "
                    f"closed back below with a long upper wick, {dn_txt}, next candle "
                    "closed below the rejection low.")
        if up and L[s] < LO[s] and C[s] > LO[s] and \
           (min(O[s], C[s]) - L[s]) >= WICK * rng and C[i] > H[s] and C[i] > O[i]:
            entry, stop = C[i], L[s] - buf
            if HI[i] >= entry + RR * (entry - stop):
                add(SETUPS[0], "BUY", entry, stop, LO[s],
                    f"Price swept below the recent low (LVL) on {RV[s]:.1f}x average volume, "
                    f"closed back above with a long lower wick, {up_txt}, next candle "
                    "closed above the rejection high.")

    # Setup 2: breakout, low-volume pullback, resume
    for k in (1, 2, 3):
        b = i - k - 1
        seg = RV[b + 1:i]
        pb_rv = 9 if np.all(np.isnan(seg)) else np.nanmean(seg)
        resume_vol = RV[i] >= RES_VOL and RV[i] > RV[i - 1]
        if up:
            lvl = HI[b]
            if C[b] > lvl and C[b] > O[b] and RV[b] >= BRK_VOL and min(C[b + 1:i]) > lvl \
               and min(L[b + 1:i]) <= lvl + 0.5 * atr and pb_rv < PB_VOL \
               and C[i] > O[i] and C[i] > H[i - 1] and resume_vol:
                add(SETUPS[1], "BUY", C[i], min(L[b + 1:i + 1]) - buf, lvl,
                    f"Closed above LVL on {RV[b]:.1f}x volume, pulled back {k} candle(s) on "
                    f"lower volume ({pb_rv:.1f}x) and held, resumed up on rising volume, {up_txt}.")
        if dn:
            lvl = LO[b]
            if C[b] < lvl and C[b] < O[b] and RV[b] >= BRK_VOL and max(C[b + 1:i]) < lvl \
               and max(H[b + 1:i]) >= lvl - 0.5 * atr and pb_rv < PB_VOL \
               and C[i] < O[i] and C[i] < L[i - 1] and resume_vol:
                add(SETUPS[1], "SELL", C[i], max(H[b + 1:i + 1]) + buf, lvl,
                    f"Closed below LVL on {RV[b]:.1f}x volume, pulled back {k} candle(s) on "
                    f"lower volume ({pb_rv:.1f}x) and held, resumed down on rising volume, {dn_txt}.")
    return out

def outcome(a, i, side, stop, tp, horizon=48):
    for j in range(i + 1, min(len(a["H"]), i + 1 + horizon)):
        if side == "BUY":
            if a["L"][j] <= stop:
                return 0
            if a["H"][j] >= tp:
                return 1
        else:
            if a["H"][j] >= stop:
                return 0
            if a["L"][j] <= tp:
                return 1
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

def format_alert(s, ts, w, n, prefix=""):
    icon = "🟢" if s["side"] == "BUY" else "🔴"
    t = (ts + pd.Timedelta(minutes=65)).strftime("%H:%M")  # candle close, Nigeria time (UTC+1)
    lvl_txt = "%.2f" % P(s["level"])
    why = s["why"].replace("LVL", lvl_txt)
    return (f"{prefix}{icon} XAUUSD {s['side']} (5m)\n"
            f"Setup: {s['setup']}\n"
            f"Entry: {P(s['entry']):.2f}\n"
            f"Stop loss: {P(s['stop']):.2f}\n"
            f"TP: {P(s['tp']):.2f} (1:{RR:g})\n"
            f"Skip if price already beyond: {P(s['vu']):.2f}\n"
            f"Historical win rate: {100 * w / n:.0f}% ({n} setups)\n"
            f"Reason: {why}\n"
            f"Candle closed: {t} Nigeria time\n"
            "Levels come from gold futures; confirm against your broker price.")

def note_block(text):
    state["blocked"] = (state["blocked"] + [text])[-10:]

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
    a, i = arrays(d), len(d) - 1
    atr = a["ATR"][i]
    state["diag"] = (f"last candle {ts} UTC | 15m trend {int(a['TR'][i])} | "
                     f"relative volume {a['RV'][i]:.2f} | ATR {atr:.2f} | "
                     f"avg relative volume (last 100 candles) {np.nanmean(a['RV'][-100:]):.2f}")
    day = ts.strftime("%Y-%m-%d")
    if state["day"] != day:
        state.update(day=day, sent=0, levels=[])
    if state["sent"] >= MAX_ALERTS:
        return "daily cap reached"
    if state["stats"] is None or time.time() - state["stats_time"] > 3600:
        state["stats"], state["stats_time"] = backtest(d), time.time()
    sent = 0
    for s in detect(a, i):
        if state["sent"] >= MAX_ALERTS:
            break
        w, n = state["stats"].get(s["setup"], (0, 0))
        tag = f"{ts:%m-%d %H:%M} {s['side']} {s['setup']}"
        if n < MIN_N:
            note_block(f"{tag}: blocked, only {n} past setups (need {MIN_N})")
            continue
        if 100 * w / n < MIN_WIN:
            note_block(f"{tag}: blocked, win rate {100 * w / n:.0f}% (need {MIN_WIN:g}%)")
            continue
        if any(abs(s["level"] - L) < 0.5 * atr for L in state["levels"]):
            note_block(f"{tag}: blocked, level already alerted today")
            continue
        try:
            if tg(format_alert(s, ts, w, n)):
                state["sent"] += 1
                state["levels"].append(s["level"])
                sent += 1
            else:
                note_block(f"{tag}: Telegram send failed")
        except Exception as e:
            note_block(f"{tag}: error {e}")
    return f"scanned {ts}, alerts sent: {sent}"

def authed():
    return bool(SCAN_KEY) and request.args.get("key") == SCAN_KEY

@app.route("/")
def home():
    return f"Gold bot running. Alerts today: {state['sent']}. Last scan: {state['last_msg']}"

@app.route("/scan")
def scan_route():
    if not authed():
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
    if not authed():
        return "forbidden", 403
    return "sent" if tg("✅ Gold bot test message") else "failed: check token and chat ID"

@app.route("/testalert")
def testalert_route():
    if not authed():
        return "forbidden", 403
    ts = pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5)
    fake = dict(setup=SETUPS[0], side="BUY", entry=2000.0, stop=1996.0, tp=2008.0,
                vu=2001.0, level=1998.0,
                why="Price swept below the recent low (LVL) on 2.3x average volume, closed back above "
                    "with a long lower wick, 15m trend up, next candle closed above the rejection high.")
    try:
        ok = tg(format_alert(fake, ts, 6, 10, prefix="TEST ALERT, NOT A REAL SIGNAL\n"))
        return "sent" if ok else "failed: Telegram refused it"
    except Exception as e:
        return f"error: {e}"

@app.route("/stats")
def stats_route():
    if not authed():
        return "forbidden", 403
    s = state["stats"]
    lines = []
    if s is None:
        lines.append("Stats not computed yet. Wait for a scan, then refresh.")
    else:
        for name in SETUPS:
            w, n = s.get(name, (0, 0))
            lines.append(f"{name}: " + (f"{w} wins of {n} setups ({100 * w / n:.0f}%)" if n
                                        else "0 setups found in the last ~59 days"))
    lines.append(f"<br>Rules in use: minimum {MIN_N} setups and {MIN_WIN:g}% win rate, "
                 f"{RR:g}R target, trend filter {'on' if TREND_FILTER else 'off'}")
    lines.append(f"Volume thresholds: rejection {REJ_VOL}, wick {WICK}, breakout {BRK_VOL}, "
                 f"pullback under {PB_VOL}, resume {RES_VOL}")
    lines.append(f"<br>Latest candle: {state['diag']}")
    lines.append(f"Alerts today: {state['sent']}<br>")
    lines.append("Recent blocked signals:" if state["blocked"] else "No blocked signals yet.")
    lines.extend(state["blocked"])
    return "<br>".join(lines)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
