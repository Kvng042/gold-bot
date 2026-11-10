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
MIN_N = env("MIN_SAMPLES", 8)
MIN_WIN = env("MIN_WIN_RATE", 55.0)
OFFSET = env("PRICE_OFFSET", 0.0)
RR = env("RR", 2.0)
LOOKBACK = env("LOOKBACK", 36)
TREND_FILTER = env("TREND_FILTER", 1)
REJ_VOL = env("REJ_VOL", 1.0)
WICK = env("WICK", 0.3)
BRK_VOL = env("BRK_VOL", 1.3)
PB_VOL = env("PB_VOL", 1.0)
RES_VOL = env("RES_VOL", 1.1)
ICT_LOOK = env("ICT_LOOK", 48)
SWEEP_BACK = env("SWEEP_BACK", 10)
DISP = env("DISP", 0.5)
STALE = pd.Timedelta(minutes=20)

SETUPS = ["Rejection at key level", "Breakout-pullback-resume"]
ICT_NAME = "ICT sweep + structure shift"
SETUP_NAMES = SETUPS + [ICT_NAME]
SHORT = ("Rej", "Brk", "ICT")
PAIRS = {
    "XAUUSD": {"ict": "GC=F", "vpa": "GC=F"},
    "EURUSD": {"ict": "EURUSD=X", "vpa": "6E=F"},
    "GBPUSD": {"ict": "GBPUSD=X", "vpa": "6B=F"},
    "AUDUSD": {"ict": "AUDUSD=X", "vpa": "6A=F"},
    "NZDUSD": {"ict": "NZDUSD=X", "vpa": "6N=F"},
    "USDJPY": {"ict": "JPY=X", "vpa": "6J=F"},
    "USDCAD": {"ict": "CAD=X", "vpa": "6C=F"},
    "USDCHF": {"ict": "CHF=X", "vpa": "6S=F"},
}
INV_SYMS = {"6J=F", "6C=F", "6S=F"}  # futures quoted as USD per foreign currency: flipped to pair terms
DEC = {"XAUUSD": 2, "USDJPY": 3}
VOL_SYMS = {c["vpa"] for c in PAIRS.values() if c["vpa"]}
ALL_SYMS = sorted({s for c in PAIRS.values() for s in (c["ict"], c["vpa"]) if s})
TUNABLE = {"MIN_SAMPLES": "MIN_N", "MIN_WIN_RATE": "MIN_WIN", "MAX_ALERTS": "MAX_ALERTS",
           "PRICE_OFFSET": "OFFSET", "RR": "RR", "LOOKBACK": "LOOKBACK",
           "TREND_FILTER": "TREND_FILTER", "REJ_VOL": "REJ_VOL", "WICK": "WICK",
           "BRK_VOL": "BRK_VOL", "PB_VOL": "PB_VOL", "RES_VOL": "RES_VOL",
           "ICT_LOOK": "ICT_LOOK", "SWEEP_BACK": "SWEEP_BACK", "DISP": "DISP"}
COMMANDS = [("stats", "Results per pair, rules and blocked signals"),
            ("status", "Alerts today, last scan and feeds"),
            ("testalert", "Send a fake test alert"),
            ("settings", "Show current rules"),
            ("backtest", "Recompute past results"),
            ("scan", "Check for new candles now"),
            ("set", "Change a rule, e.g. /set REJ_VOL 1.2"),
            ("test", "Check the bot is alive"),
            ("help", "List commands")]

app = Flask(__name__)
lock = threading.Lock()
state = {"day": None, "sent": 0, "levels": {}, "last": {}, "stats": None, "stats_time": 0,
         "last_msg": "not scanned yet", "diag": "none yet", "blocked": [], "feeds": "none yet",
         "next_try": pd.Timestamp("2000-01-01", tz="UTC")}

def fmt(pair, x):
    x = x + OFFSET if pair == "XAUUSD" else x
    return f"{x:.{DEC.get(pair, 5)}f}"

def tg(text):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": text}, timeout=15)
    return r.ok

def load(sym):
    df = yf.Ticker(sym).history(period="59d", interval="5m")
    if df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = df.index.tz_convert("UTC")
    df = df[df.index + pd.Timedelta(minutes=5) <= pd.Timestamp.now(tz="UTC")]
    if sym in INV_SYMS:
        o, h, l, c = df.Open.copy(), df.High.copy(), df.Low.copy(), df.Close.copy()
        df["Open"], df["High"], df["Low"], df["Close"] = 1 / o, 1 / l, 1 / h, 1 / c
    if sym in VOL_SYMS:
        if df.Volume.tail(300).sum() == 0:
            return None
        while len(df) and df.Volume.iloc[-1] == 0:
            df = df.iloc[:-1]
    return df if len(df) >= 150 else None

def prepare(df):
    d = df.copy()
    d["rv"] = (d.Volume / d.Volume.rolling(20).mean().shift(1)).replace([np.inf, -np.inf], np.nan)
    tr = pd.concat([d.High - d.Low, (d.High - d.Close.shift()).abs(),
                    (d.Low - d.Close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["hi"] = d.High.rolling(LOOKBACK).max().shift(1)
    d["lo"] = d.Low.rolling(LOOKBACK).min().shift(1)
    d["ihi"] = d.High.rolling(ICT_LOOK).max().shift(1)
    d["ilo"] = d.Low.rolling(ICT_LOOK).min().shift(1)
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
            "HI": d.hi.to_numpy(), "LO": d.lo.to_numpy(), "IHI": d.ihi.to_numpy(),
            "ILO": d.ilo.to_numpy(), "TR": d.trend.to_numpy(), "HR": d.index.hour.to_numpy()}

def make(setup, side, entry, stop, level, why, atr):
    risk = abs(entry - stop)
    if not (0.4 * atr <= risk <= 3 * atr):
        return None
    sgn = 1 if side == "BUY" else -1
    return dict(setup=setup, side=side, entry=entry, stop=stop,
                tp=entry + sgn * RR * risk, vu=entry + sgn * 0.25 * risk,
                level=level, why=why)

def detect_vpa(a, i):
    O, H, L, C, RV, HI, LO = (a[k] for k in ("O", "H", "L", "C", "RV", "HI", "LO"))
    atr, tr = a["ATR"][i], a["TR"][i]
    out = []
    if i < max(LOOKBACK, ICT_LOOK) + 26 or atr != atr:
        return out
    if TREND_FILTER and tr == 0:
        return out
    dn = (tr == -1) or not TREND_FILTER
    up = (tr == 1) or not TREND_FILTER
    dn_txt = "15m trend down" if tr == -1 else "15m trend neutral"
    up_txt = "15m trend up" if tr == 1 else "15m trend neutral"
    buf = 0.15 * atr

    def add(*args):
        x = make(*args, atr)
        if x:
            out.append(x)

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

def detect_ict(a, i):
    out = []
    atr, tr, hr = a["ATR"][i], a["TR"][i], a["HR"][i]
    if i < max(LOOKBACK, ICT_LOOK) + 26 or atr != atr:
        return out
    if not (7 <= hr < 11 or 12 <= hr < 16):
        return out
    if TREND_FILTER and tr == 0:
        return out
    O, H, L, C, IHI, ILO = (a[k] for k in ("O", "H", "L", "C", "IHI", "ILO"))
    sess = "London" if hr < 11 else "New York"
    buf = 0.15 * atr
    m = i - 1
    body = C[m] - O[m]
    up = (tr == 1) or not TREND_FILTER
    dn = (tr == -1) or not TREND_FILTER
    up_txt = "15m trend up" if tr == 1 else "15m trend neutral"
    dn_txt = "15m trend down" if tr == -1 else "15m trend neutral"

    if up and body >= DISP * atr and L[i] > H[i - 2]:
        for s in range(m - 1, m - 1 - SWEEP_BACK, -1):
            if L[s] < ILO[s]:
                swing = np.max(H[s - 5:s])
                if C[m] > swing and (s + 1 >= m or np.max(C[s + 1:m]) <= swing):
                    entry, stop = C[i], np.min(L[s:i + 1]) - buf
                    if IHI[i] >= entry + RR * (entry - stop):
                        x = make(ICT_NAME, "BUY", entry, stop, ILO[s],
                                 f"Sell-side liquidity below LVL was swept in the {sess} session, then a "
                                 "strong bullish candle broke the nearby swing high and left a fair value "
                                 f"gap, {up_txt}.", atr)
                        if x:
                            out.append(x)
                    break
    if dn and -body >= DISP * atr and H[i] < L[i - 2]:
        for s in range(m - 1, m - 1 - SWEEP_BACK, -1):
            if H[s] > IHI[s]:
                swing = np.min(L[s - 5:s])
                if C[m] < swing and (s + 1 >= m or np.min(C[s + 1:m]) >= swing):
                    entry, stop = C[i], np.max(H[s:i + 1]) + buf
                    if ILO[i] <= entry - RR * (stop - entry):
                        x = make(ICT_NAME, "SELL", entry, stop, IHI[s],
                                 f"Buy-side liquidity above LVL was swept in the {sess} session, then a "
                                 "strong bearish candle broke the nearby swing low and left a fair value "
                                 f"gap, {dn_txt}.", atr)
                        if x:
                            out.append(x)
                    break
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

def backtest(d, kind):
    a, res = arrays(d), {}
    fn = detect_vpa if kind == "vpa" else detect_ict
    for i in range(max(LOOKBACK, ICT_LOOK) + 26, len(d) - 1):
        for s in fn(a, i):
            o = outcome(a, i, s["side"], s["stop"], s["tp"])
            if o is not None:
                w, n = res.get(s["setup"], (0, 0))
                res[s["setup"]] = (w + o, n + 1)
    return res

def load_frames():
    frames, notes = {}, []
    for sym in ALL_SYMS:
        try:
            df = load(sym)
        except Exception as e:
            df = None
            notes.append(f"{sym}: error {str(e)[:40]}")
        if df is None:
            notes.append(f"{sym}: no data")
        else:
            frames[sym] = prepare(df)
    return frames, notes

def build_tasks(frames, notes):
    tasks = []
    for pair, cfg in PAIRS.items():
        ict, vpa = cfg["ict"], cfg["vpa"]
        if ict in frames:
            tasks.append((pair, "ict", frames[ict]))
        if vpa and vpa in frames:
            if vpa != ict and ict in frames:
                r = frames[vpa].Close.iloc[-1] / frames[ict].Close.iloc[-1]
                if abs(r - 1) > 0.02:
                    notes.append(f"{pair}: futures price mismatch, VPA skipped")
                    continue
            tasks.append((pair, "vpa", frames[vpa]))
    return tasks

def compute_stats(tasks):
    st = {}
    for pair, kind, d in tasks:
        for setup, (w, n) in backtest(d, kind).items():
            st[(pair, setup)] = (w, n)
    return st

def recompute():
    frames, notes = load_frames()
    tasks = build_tasks(frames, notes)
    if not tasks:
        return False
    state["stats"], state["stats_time"] = compute_stats(tasks), time.time()
    return True

def format_alert(pair, s, ts, w, n, prefix=""):
    icon = "🟢" if s["side"] == "BUY" else "🔴"
    t = (ts + pd.Timedelta(minutes=65)).strftime("%H:%M")  # candle close, Nigeria time (UTC+1)
    why = s["why"].replace("LVL", fmt(pair, s["level"]))
    return (f"{prefix}{icon} {pair} {s['side']} (5m)\n"
            f"Setup: {s['setup']}\n"
            f"Entry: {fmt(pair, s['entry'])}\n"
            f"Stop loss: {fmt(pair, s['stop'])}\n"
            f"TP: {fmt(pair, s['tp'])} (1:{RR:g})\n"
            f"Skip if price already beyond: {fmt(pair, s['vu'])}\n"
            f"Historical win rate: {100 * w / n:.0f}% ({n} setups, {pair} only)\n"
            f"Reason: {why}\n"
            f"Candle closed: {t} Nigeria time\n"
            "Prices come from free market data and can differ from your broker; confirm before trading.")

def note_block(text):
    state["blocked"] = (state["blocked"] + [text])[-10:]

def scan_all(force=False):
    now = pd.Timestamp.now(tz="UTC")
    if not force and now < state["next_try"]:
        return f"not due yet (next check {state['next_try']:%H:%M:%S} UTC)"
    frames, notes = load_frames()
    tasks = build_tasks(frames, notes)
    state["feeds"] = f"{len(tasks)} feeds active" + (": " + "; ".join(notes) if notes else "")
    if not tasks:
        state["next_try"] = now + pd.Timedelta(minutes=5)
        return "no data"
    today = now.strftime("%Y-%m-%d")
    if state["day"] != today:
        state.update(day=today, sent=0, levels={})
    if state["stats"] is None or time.time() - state["stats_time"] > 3600:
        state["stats"], state["stats_time"] = compute_stats(tasks), time.time()
    newest = max(d.index[-1] for _, _, d in tasks)
    got_new, sent = False, 0
    for pair, kind, d in tasks:
        ts = d.index[-1]
        if now - ts > STALE:
            continue
        key = f"{pair}:{kind}"
        if state["last"].get(key) == ts:
            continue
        state["last"][key] = ts
        got_new = True
        a, i = arrays(d), len(d) - 1
        atr = a["ATR"][i]
        if pair == "XAUUSD" and kind == "vpa":
            state["diag"] = (f"{ts} UTC | 15m trend {int(a['TR'][i])} | relative volume "
                             f"{a['RV'][i]:.2f} | ATR {atr:.2f}")
        for s in (detect_vpa if kind == "vpa" else detect_ict)(a, i):
            if state["sent"] >= MAX_ALERTS:
                break
            w, n = state["stats"].get((pair, s["setup"]), (0, 0))
            tag = f"{ts:%m-%d %H:%M} {pair} {s['side']} {s['setup']}"
            if n < MIN_N:
                note_block(f"{tag}: blocked, only {n} past setups (need {MIN_N})")
                continue
            if 100 * w / n < MIN_WIN:
                note_block(f"{tag}: blocked, win rate {100 * w / n:.0f}% (need {MIN_WIN:g}%)")
                continue
            used = state["levels"].setdefault(pair, [])
            if any(abs(s["level"] - L) < 0.5 * atr for L in used):
                note_block(f"{tag}: blocked, level already alerted today")
                continue
            try:
                if tg(format_alert(pair, s, ts, w, n)):
                    state["sent"] += 1
                    used.append(s["level"])
                    sent += 1
                else:
                    note_block(f"{tag}: Telegram send failed")
            except Exception as e:
                note_block(f"{tag}: error {e}")
    if now - newest > STALE:
        state["next_try"] = now + pd.Timedelta(minutes=15)
    elif got_new:
        state["next_try"] = newest + pd.Timedelta(minutes=10, seconds=15)
    else:
        state["next_try"] = now + pd.Timedelta(seconds=60)
    return f"scanned, new candles: {got_new}, alerts sent: {sent}"

def settings_text():
    return (f"Rules: min {MIN_N} setups per pair, min {MIN_WIN:g}% win rate, {RR:g}R target, "
            f"trend filter {'on' if TREND_FILTER else 'off'}\n"
            f"VPA: lookback {LOOKBACK}, rejection vol {REJ_VOL:g}, wick {WICK:g}, "
            f"breakout {BRK_VOL:g}, pullback under {PB_VOL:g}, resume {RES_VOL:g}\n"
            f"ICT: lookback {ICT_LOOK}, sweep window {SWEEP_BACK} candles, displacement {DISP:g} ATR, "
            "sessions 07-11 and 12-16 UTC\n"
            f"Max alerts per day {MAX_ALERTS}, gold price offset {OFFSET:g}")

def stats_text():
    st, lines = state["stats"], []
    if st is None:
        lines.append("Stats not computed yet. Send /backtest.")
    else:
        lines.append("Wins/setups per pair. Rej = rejection at key level, "
                     "Brk = breakout-pullback-resume, ICT = sweep + structure shift")
        for pair in PAIRS:
            parts = []
            for name, short in zip(SETUP_NAMES, SHORT):
                w, n = st.get((pair, name), (0, 0))
                parts.append(f"{short} {w}/{n}" + (f" ({100 * w / n:.0f}%)" if n else ""))
            lines.append(f"{pair}: " + " | ".join(parts))
    lines += ["", settings_text(), "", f"Gold latest candle: {state['diag']}",
              f"Feeds: {state['feeds']}", f"Alerts today: {state['sent']}", ""]
    lines.append("Recent blocked signals:" if state["blocked"] else "No blocked signals yet.")
    lines.extend(state["blocked"])
    return "\n".join(lines)

def handle(text):
    parts = text.strip().split()
    cmd, args = parts[0].split("@")[0].lower(), parts[1:]
    try:
        if cmd in ("/start", "/help"):
            tg("Commands:\n" + "\n".join(f"/{c} - {d}" for c, d in COMMANDS))
        elif cmd == "/test":
            tg("✅ Bot is alive")
        elif cmd == "/testalert":
            ts = pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5)
            fake = dict(setup=SETUPS[0], side="BUY", entry=2000.0, stop=1996.0, tp=2008.0,
                        vu=2001.0, level=1998.0,
                        why="Price swept below the recent low (LVL) on 2.3x average volume, closed "
                            "back above with a long lower wick, 15m trend up, next candle closed "
                            "above the rejection high.")
            tg(format_alert("XAUUSD", fake, ts, 6, 10, prefix="TEST ALERT, NOT A REAL SIGNAL\n"))
        elif cmd == "/status":
            tg(f"Alerts today: {state['sent']}/{MAX_ALERTS}\nLast scan: {state['last_msg']}\n"
               f"Feeds: {state['feeds']}\nNext check: {state['next_try']:%H:%M:%S} UTC")
        elif cmd == "/stats":
            tg(stats_text())
        elif cmd == "/settings":
            tg(settings_text())
        elif cmd == "/scan":
            with lock:
                msg = scan_all(force=True)
                state["last_msg"] = msg
            tg(msg)
        elif cmd == "/backtest":
            with lock:
                ok = recompute()
            tg(stats_text() if ok else "No data right now (markets may be closed).")
        elif cmd == "/set":
            name = args[0].upper() if args else ""
            if len(args) != 2 or name not in TUNABLE:
                tg("Usage: /set NAME VALUE\nNames: " + ", ".join(TUNABLE))
                return
            var = TUNABLE[name]
            globals()[var] = type(globals()[var])(args[1])
            with lock:
                recompute()
            tg(f"{name} set to {args[1]}. This resets if the app restarts; to keep it, "
               "also change the environment variable.\n\n" + stats_text())
        else:
            tg("Unknown command. Send /help")
    except Exception as e:
        tg(f"Error: {e}")

def bg_scan():
    with lock:
        try:
            msg = scan_all()
        except Exception as e:
            msg = f"error: {e}"
        state["last_msg"] = msg

def authed():
    return bool(SCAN_KEY) and request.args.get("key") == SCAN_KEY

@app.route("/")
def home():
    return f"Bot running. Alerts today: {state['sent']}. Last scan: {state['last_msg']}"

@app.route("/scan")
def scan_route():
    if not authed():
        return "forbidden", 403
    if lock.locked():
        return "busy: a scan is already running"
    threading.Thread(target=bg_scan, daemon=True).start()
    return "scan triggered (runs in the background; send /status in Telegram)"

@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    if not SCAN_KEY or request.headers.get("X-Telegram-Bot-Api-Secret-Token") != SCAN_KEY:
        return "forbidden", 403
    msg = (request.get_json(silent=True) or {}).get("message") or {}
    text = msg.get("text", "")
    if str((msg.get("chat") or {}).get("id")) == str(CHAT_ID) and text.startswith("/"):
        threading.Thread(target=handle, args=(text,), daemon=True).start()
    return "ok"

@app.route("/setup")
def setup_route():
    if not authed():
        return "forbidden", 403
    base = f"https://api.telegram.org/bot{TOKEN}"
    url = request.url_root.replace("http://", "https://") + "telegram"
    r = requests.post(base + "/setWebhook", timeout=15,
                      json={"url": url, "secret_token": SCAN_KEY,
                            "allowed_updates": ["message"], "drop_pending_updates": True})
    requests.post(base + "/setMyCommands", timeout=15,
                  json={"commands": [{"command": c, "description": d} for c, d in COMMANDS]})
    return r.text

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
