#!/usr/bin/env python3
"""Analyze trade signals from orderbook CSVs and track daily performance.

For each signal, pulls daily OHLC via yfinance and determines whether the
Stop Loss or Target Price was hit, computes max profit / max drawdown, and for
SL-hit trades finds the next closed candle in the same direction (green for
Buy, red for Sell) as a reentry point.
"""

import csv
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILENAME_RE = re.compile(r"orderbook(\d{4}-\d{2}-\d{2})\.csv")

# Market configurations. "suffix" is appended to symbols passed to yfinance.
MARKETS = {
    "us": {"orderbook_dir": "orderbook", "report_dir": "report", "suffix": ""},
    "in": {"orderbook_dir": "orderbook_in", "report_dir": "report_in", "suffix": ".NS"},
}
DEFAULT_MARKET = "us"

# Number of extra calendar days to request beyond the signal to cover future days.
LOOKAHEAD_DAYS = 400


def parse_signal_date(filename):
    m = FILENAME_RE.search(filename)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def read_orderbook(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows


def fetch_ohlc(symbol, start_date):
    """Return a pandas DataFrame with columns [Open, High, Low, Close, Date]."""
    end = datetime.utcnow() + timedelta(days=1)
    df = yf.download(symbol, start=start_date, end=end, progress=False,
                     auto_adjust=False)
    if df is None or df.empty:
        return None
    # yfinance >=0.2 is multiindex; flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.reset_index(inplace=True)
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df.rename(columns={date_col: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    # Drop rows with missing OHLC (yfinance sometimes appends a NaN placeholder).
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def analyze_trade(row, signal_date, ohlc):
    """Return a dict with the performance result of a single trade."""
    symbol = row["Symbol"].strip().upper()
    side = row["Buy/Sell"].strip().lower()
    cbt = row.get("CBT", "").strip()
    try:
        entry = float(row["Entry"])
        sl = float(row["Stoploss"])
        tp = float(row["TargetPrice"])
    except (ValueError, KeyError):
        return None

    if ohlc is None or ohlc.empty:
        status = "No Data"
        result = {
            "signal_date": str(signal_date),
            "symbol": symbol,
            "side": side,
            "cbt": cbt,
            "entry": entry,
            "stoploss": sl,
            "target": tp,
            "status": status,
            "exit_date": "",
            "exit_price": "",
            "max_profit_pct": "",
            "max_drawdown_pct": "",
            "days_held": "",
            "pnl_pct": "",
            "last_price": "",
            "last_date": "",
        }
        return result

    # Consider days from the signal date onward (entry likely on/after signal day)
    ohlc = ohlc[ohlc["Date"] >= signal_date].copy()
    ohlc = ohlc.sort_values("Date").reset_index(drop=True)

    if ohlc.empty:
        return None

    max_profit_pct = None
    max_drawdown_pct = None
    exit_date = None
    exit_price = None
    status = "Open"

    for _, day in ohlc.iterrows():
        high = float(day["High"])
        low = float(day["Low"])
        date = day["Date"]

        pnl_if_exit = (high - entry) / entry * 100 if side == "buy" else (entry - low) / entry * 100
        dd = (entry - low) / entry * 100 if side == "buy" else (high - entry) / entry * 100

        if max_profit_pct is None or pnl_if_exit > max_profit_pct:
            max_profit_pct = pnl_if_exit
        if max_drawdown_pct is None or dd > max_drawdown_pct:
            max_drawdown_pct = dd

        # Check exits in daily order. If both conditions true same day, judge by
        # which is closer to the open (conservative default).
        hit_tp = high >= tp if side == "buy" else low <= tp
        hit_sl = low <= sl if side == "buy" else high >= sl

        if hit_tp and hit_sl:
            # ambiguous same-day: choose by which threshold is further from open
            open_px = float(day["Open"])
            if side == "buy":
                hit_tp = (tp - open_px) <= (open_px - sl)
            else:
                hit_tp = (open_px - tp) <= (sl - open_px)
            if hit_tp:
                hit_sl = False
            else:
                hit_sl = True

        if hit_tp:
            status = "TP Hit"
            exit_date = date
            exit_price = tp
            break
        if hit_sl:
            status = "SL Hit"
            exit_date = date
            exit_price = sl
            break

    if status == "Open":
        last_row = ohlc.iloc[-1]
        last_price = float(last_row["Close"])
        last_date = str(last_row["Date"])
        days_held = (last_row["Date"] - signal_date).days
        pnl = ""  # unrealized
    else:
        last_price = exit_price
        last_date = str(exit_date)
        days_held = (exit_date - signal_date).days
        pnl = (exit_price - entry) / entry * 100 if side == "buy" else (entry - exit_price) / entry * 100

    result = {
        "signal_date": str(signal_date),
        "symbol": symbol,
        "side": side,
        "cbt": cbt,
        "entry": entry,
        "stoploss": sl,
        "target": tp,
        "status": status,
        "exit_date": str(exit_date) if exit_date else "",
        "exit_price": round(exit_price, 4) if exit_price else "",
        "max_profit_pct": round(max_profit_pct, 2) if max_profit_pct is not None else "",
        "max_drawdown_pct": round(max_drawdown_pct, 2) if max_drawdown_pct is not None else "",
        "days_held": days_held if days_held is not None else "",
        "pnl_pct": round(pnl, 2) if pnl != "" else "",
        "last_price": round(last_price, 4),
        "last_date": last_date,
        "reentry_date": "",
        "reentry_side": "",
        "reentry_price": "",
    }
    if status == "SL Hit":
        add_reentry(result, ohlc, exit_date, side, entry)
    return result


def add_reentry(result, ohlc, sl_hit_date, side, entry):
    """Find the next closed candle after the SL hit in the same direction."""
    after = ohlc[ohlc["Date"] > sl_hit_date].copy().reset_index(drop=True)
    for _, day in after.iterrows():
        open_px = float(day["Open"])
        close_px = float(day["Close"])
        if close_px > open_px:  # green candle
            direction = "Green"
        elif close_px < open_px:  # red candle
            direction = "Red"
        else:
            continue  # doji, skip
        if (side == "buy" and direction == "Green") or (side == "sell" and direction == "Red"):
            result["reentry_date"] = str(day["Date"])
            result["reentry_side"] = direction
            result["reentry_price"] = round(close_px, 4)
            return


def collect(orderbook_dir):
    trades = []
    files = sorted(glob.glob(os.path.join(orderbook_dir, "orderbook*.csv")))
    for path in files:
        signal_date = parse_signal_date(os.path.basename(path))
        if signal_date is None:
            continue
        rows = read_orderbook(path)
        for row in rows:
            symbol = row.get("Symbol", "").strip().upper()
            if not symbol:
                continue
            trades.append({"path": path, "signal_date": signal_date, "row": row, "symbol": symbol})
    return trades


def main():
    market = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MARKET
    if market not in MARKETS:
        print(f"Unknown market '{market}'. Valid: {list(MARKETS)}", file=sys.stderr)
        sys.exit(1)
    cfg = MARKETS[market]
    orderbook_dir = os.path.join(BASE_DIR, cfg["orderbook_dir"])
    report_dir = os.path.join(BASE_DIR, cfg["report_dir"])
    suffix = cfg["suffix"]

    os.makedirs(report_dir, exist_ok=True)
    trades = collect(orderbook_dir)
    print(f"[{market}] Found {len(trades)} signals.", flush=True)

    # Group by symbol so we only download once per symbol.
    symbol_start = {}
    for t in trades:
        start = t["signal_date"] - timedelta(days=10)
        if t["symbol"] not in symbol_start or start < symbol_start[t["symbol"]]:
            symbol_start[t["symbol"]] = start

    ohlc_cache = {}
    for symbol, start in symbol_start.items():
        yf_symbol = symbol + suffix
        try:
            print(f"[{market}] Fetching {yf_symbol} ...", flush=True)
            df = fetch_ohlc(yf_symbol, start)
            ohlc_cache[symbol] = df
        except Exception as e:  # noqa: BLE001
            print(f"Failed to fetch {yf_symbol}: {e}", flush=True)
            ohlc_cache[symbol] = None

    results = []
    for t in trades:
        ohlc = ohlc_cache.get(t["symbol"])
        res = analyze_trade(t["row"], t["signal_date"], ohlc)
        if res:
            results.append(res)

    # Sort by signal date then symbol.
    results.sort(key=lambda r: (r["signal_date"], r["symbol"]))

    write_report(results, report_dir)


def write_report(results, report_dir):
    columns = [
        "signal_date", "symbol", "side", "cbt", "entry", "stoploss", "target",
        "status", "exit_date", "exit_price", "max_profit_pct", "max_drawdown_pct",
        "days_held", "pnl_pct", "last_price", "last_date",
    ]

    # --- daily performance CSV (all trades) ---
    perf_path = os.path.join(report_dir, "trade_performance.csv")
    with open(perf_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in columns})

    # --- sl_hit CSV (only SL-hit trades with reentry info) ---
    sl_cols = columns + ["reentry_date", "reentry_side", "reentry_price"]
    sl_hits = [r for r in results if r["status"] == "SL Hit"]
    sl_path = os.path.join(report_dir, "sl_hit.csv")
    with open(sl_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sl_cols)
        writer.writeheader()
        for r in sl_hits:
            writer.writerow({k: r.get(k, "") for k in sl_cols})

    # --- JSON for webpage ---
    data = build_json(results)
    json_path = os.path.join(report_dir, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    summary = summarize(results)
    with open(os.path.join(report_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print(f"\nTotal trades: {len(results)}")
    print(f"Written: {perf_path}")
    print(f"Written: {sl_path}")
    print(f"Written: {json_path}")


def build_json(results):
    def clean(r):
        return {
            "date": r["signal_date"],
            "symbol": r["symbol"],
            "side": r["side"],
            "cbt": r["cbt"],
            "entry": r["entry"],
            "stoploss": r["stoploss"],
            "target": r["target"],
            "status": r["status"],
            "exitDate": r["exit_date"],
            "exitPrice": r["exit_price"],
            "maxProfit": r["max_profit_pct"],
            "maxDrawdown": r["max_drawdown_pct"],
            "daysHeld": r["days_held"],
            "pnl": r["pnl_pct"],
            "lastPrice": r["last_price"],
            "lastDate": r["last_date"],
            "reentryDate": r["reentry_date"],
            "reentrySide": r["reentry_side"],
            "reentryPrice": r["reentry_price"],
        }
    return {"generatedAt": datetime.utcnow().isoformat() + "Z", "trades": [clean(r) for r in results]}


def summarize(results):
    done = [r for r in results if r["status"] in ("TP Hit", "SL Hit")]
    tp_hits = [r for r in done if r["status"] == "TP Hit"]
    sl_hits = [r for r in done if r["status"] == "SL Hit"]
    open_trades = [r for r in results if r["status"] == "Open"]

    def pct(x):
        s = 0.0
        n = 0
        for v in x:
            try:
                s += float(v)
                n += 1
            except (TypeError, ValueError):
                pass
        return round(s / n, 2) if n else 0.0

    return {
        "total": len(results),
        "tp_hits": len(tp_hits),
        "sl_hits": len(sl_hits),
        "open": len(open_trades),
        "win_rate": round(len(tp_hits) / len(done) * 100, 1) if done else 0,
        "avg_pnl": pct([r["pnl_pct"] for r in done]),
        "avg_win": pct([r["pnl_pct"] for r in tp_hits]),
        "avg_loss": pct([r["pnl_pct"] for r in sl_hits]),
        "avg_max_profit": pct([r["max_profit_pct"] for r in done]),
        "avg_max_drawdown": pct([r["max_drawdown_pct"] for r in done]),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


if __name__ == "__main__":
    main()
