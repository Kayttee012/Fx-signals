"""
FX AI Signal Scanner -- single-file version
Run with:  streamlit run fx_signals_app.py

Tabs:
  1. Scanner        - ranked BUY/SELL signals across every symbol below
  2. Chart Viewer    - candlestick chart with order block zones, trend
                       lines, entry/SL/TP drawn on it
  3. Backtest        - measures the REAL historical win rate of these
                       rules on a symbol, instead of assuming one

IMPORTANT / HONEST DISCLAIMER
------------------------------
- No trading system reliably hits an 80-90% win rate. If a tool ever
  claims that without a backtest to prove it, be skeptical of it.
- Win rate alone is meaningless without knowing the risk:reward used to
  get it -- a system can "win" 90% of trades and still lose money if
  losses are much bigger than wins. This app reports win rate AND
  average win/loss/profit factor together so you see the whole picture.
- This uses an independent data feed (Yahoo Finance), not FxPro's own
  feed -- FxPro's mobile app has no public API to connect to directly.
  Confirm live prices on FxPro before placing any real order.
- This is a decision-support and educational tool, not financial advice,
  and not a guarantee of any result. Forex trading is leveraged and can
  lose money quickly.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# CONFIG
# =========================================================

SYMBOLS = [
    ("EURUSD=X", "EURUSD"), ("GBPUSD=X", "GBPUSD"), ("USDJPY=X", "USDJPY"),
    ("USDCHF=X", "USDCHF"), ("AUDUSD=X", "AUDUSD"), ("USDCAD=X", "USDCAD"),
    ("NZDUSD=X", "NZDUSD"), ("EURGBP=X", "EURGBP"), ("EURJPY=X", "EURJPY"),
    ("GBPJPY=X", "GBPJPY"), ("EURCHF=X", "EURCHF"), ("AUDJPY=X", "AUDJPY"),
    ("CHFJPY=X", "CHFJPY"), ("EURAUD=X", "EURAUD"), ("GBPAUD=X", "GBPAUD"),
    ("AUDCAD=X", "AUDCAD"), ("AUDNZD=X", "AUDNZD"), ("NZDJPY=X", "NZDJPY"),
    ("USDZAR=X", "USDZAR"), ("USDMXN=X", "USDMXN"),
]

INTERVAL = "60m"
LOOKBACK_PERIOD = "60d"

SWING_WINDOW = 3
IMPULSE_ATR_MULT = 1.2
OB_LOOKBACK = 40

ATR_PERIOD = 14
SL_ATR_BUFFER = 0.25
RISK_REWARD = 2.0

ML_MIN_TRAIN_ROWS = 300
ML_CONFIDENCE_FLOOR = 45


# =========================================================
# DATA FEED
# =========================================================

def fetch_candles(yahoo_ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(yahoo_ticker, interval=interval, period=period,
                      progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {yahoo_ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{yahoo_ticker}: missing columns {missing}")
    df = df.dropna(subset=required)
    df.index.name = "time"
    return df


# =========================================================
# STRUCTURE: ATR, swings, order blocks, trend lines, signals
# =========================================================

def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(),
                     (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def swing_points(df: pd.DataFrame, window: int = SWING_WINDOW):
    highs, lows = df["High"].values, df["Low"].values
    n = len(df)
    sh, sl = [], []
    for i in range(window, n - window):
        seg_h = highs[i - window:i + window + 1]
        seg_l = lows[i - window:i + window + 1]
        if highs[i] == seg_h.max() and np.argmax(seg_h) == window:
            sh.append(i)
        if lows[i] == seg_l.min() and np.argmin(seg_l) == window:
            sl.append(i)
    return sh, sl


@dataclass
class OrderBlock:
    kind: str
    idx: int
    top: float
    bottom: float
    created_time: object
    mitigated: bool = False


def find_order_blocks(df: pd.DataFrame, atr_series: pd.Series,
                       lookback: int = OB_LOOKBACK) -> List[OrderBlock]:
    obs = []
    sh, sl = swing_points(df)
    start = max(SWING_WINDOW, len(df) - lookback)
    closes, opens = df["Close"].values, df["Open"].values
    highs, lows = df["High"].values, df["Low"].values

    for i in range(start, len(df) - 1):
        impulse = atr_series.iloc[i]
        if pd.isna(impulse) or impulse == 0:
            continue

        prior_highs = [h for h in sh if h < i]
        if prior_highs:
            ref_high = highs[prior_highs[-1]]
            move = closes[i] - closes[max(0, i - 3)]
            if closes[i] > ref_high and move > IMPULSE_ATR_MULT * impulse:
                for j in range(i, max(i - 6, 0), -1):
                    if closes[j] < opens[j]:
                        obs.append(OrderBlock("bullish", j, highs[j], lows[j], df.index[j]))
                        break

        prior_lows = [l for l in sl if l < i]
        if prior_lows:
            ref_low = lows[prior_lows[-1]]
            move = closes[max(0, i - 3)] - closes[i]
            if closes[i] < ref_low and move > IMPULSE_ATR_MULT * impulse:
                for j in range(i, max(i - 6, 0), -1):
                    if closes[j] > opens[j]:
                        obs.append(OrderBlock("bearish", j, highs[j], lows[j], df.index[j]))
                        break

    seen, unique = set(), []
    for ob in obs:
        key = (ob.kind, ob.idx)
        if key in seen:
            continue
        seen.add(key)
        after = df.iloc[ob.idx + 1:]
        if ob.kind == "bullish":
            ob.mitigated = bool((after["Low"] < ob.bottom).any())
        else:
            ob.mitigated = bool((after["High"] > ob.top).any())
        unique.append(ob)
    return unique


@dataclass
class TrendLine:
    kind: str
    slope: float
    intercept: float
    x_start: int
    x_end: int

    def value_at(self, x: int) -> float:
        return self.slope * x + self.intercept


def fit_trend_lines(df: pd.DataFrame, n_points: int = 4) -> List[TrendLine]:
    sh, sl = swing_points(df)
    lines = []
    if len(sl) >= n_points:
        pts = sl[-n_points:]
        x = np.array(pts)
        y = df["Low"].values[pts]
        slope, intercept = np.polyfit(x, y, 1)
        lines.append(TrendLine("support", slope, intercept, pts[0], len(df) - 1))
    if len(sh) >= n_points:
        pts = sh[-n_points:]
        x = np.array(pts)
        y = df["High"].values[pts]
        slope, intercept = np.polyfit(x, y, 1)
        lines.append(TrendLine("resistance", slope, intercept, pts[0], len(df) - 1))
    return lines


def trend_direction(df: pd.DataFrame) -> str:
    sh, sl = swing_points(df)
    if len(sh) >= 2 and len(sl) >= 2:
        hh = df["High"].values[sh[-1]] > df["High"].values[sh[-2]]
        hl = df["Low"].values[sl[-1]] > df["Low"].values[sl[-2]]
        lh = df["High"].values[sh[-1]] < df["High"].values[sh[-2]]
        ll = df["Low"].values[sl[-1]] < df["Low"].values[sl[-2]]
        if hh and hl:
            return "up"
        if lh and ll:
            return "down"
    return "range"


@dataclass
class Signal:
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    rule_score: float
    reason: str
    ob: Optional[OrderBlock] = None
    trend_lines: List[TrendLine] = field(default_factory=list)
    bar_idx: int = -1


def generate_rule_signal(symbol: str, df: pd.DataFrame,
                          upto: Optional[int] = None) -> Optional[Signal]:
    """
    If `upto` is given, only bars [0:upto+1] are considered (used by the
    backtester to avoid any lookahead into the future).
    """
    work = df.iloc[:upto + 1] if upto is not None else df
    a = atr(work)
    last_atr = a.iloc[-1]
    if pd.isna(last_atr) or last_atr == 0:
        return None

    trend = trend_direction(work)
    obs = find_order_blocks(work, a)
    lines = fit_trend_lines(work)
    price = work["Close"].iloc[-1]

    active_obs = [o for o in obs if not o.mitigated]
    if not active_obs:
        return None

    candidates = [o for o in active_obs if
                  (o.kind == "bullish" and trend in ("up", "range")) or
                  (o.kind == "bearish" and trend in ("down", "range"))]
    if not candidates:
        return None

    def distance(o):
        return abs(price - (o.top + o.bottom) / 2)

    ob = min(candidates, key=distance)
    zone_size = ob.top - ob.bottom
    if distance(ob) > max(zone_size, last_atr) * 1.5:
        return None

    score = 50.0
    reasons = []

    if ob.kind == "bullish":
        direction = "BUY"
        entry = max(price, ob.bottom)
        stop_loss = ob.bottom - SL_ATR_BUFFER * last_atr
        risk = entry - stop_loss
        take_profit = entry + RISK_REWARD * risk
        reasons.append("price near unmitigated bullish order block")
        if trend == "up":
            score += 20
            reasons.append("aligned with uptrend structure")
        supp = [l for l in lines if l.kind == "support"]
        if supp and price >= supp[-1].value_at(len(work) - 1) - last_atr:
            score += 15
            reasons.append("confluence with rising trend line support")
    else:
        direction = "SELL"
        entry = min(price, ob.top)
        stop_loss = ob.top + SL_ATR_BUFFER * last_atr
        risk = stop_loss - entry
        take_profit = entry - RISK_REWARD * risk
        reasons.append("price near unmitigated bearish order block")
        if trend == "down":
            score += 20
            reasons.append("aligned with downtrend structure")
        res = [l for l in lines if l.kind == "resistance"]
        if res and price <= res[-1].value_at(len(work) - 1) + last_atr:
            score += 15
            reasons.append("confluence with falling trend line resistance")

    score = float(np.clip(score, 0, 100))
    return Signal(symbol, direction, float(entry), float(stop_loss),
                   float(take_profit), score, "; ".join(reasons), ob, lines,
                   bar_idx=len(work) - 1)


# =========================================================
# ML CONFIDENCE FILTER
# =========================================================

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    a = atr(df)
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    feat = pd.DataFrame(index=df.index)
    feat["rsi"] = _rsi(close)
    feat["atr_pct"] = a / close
    feat["dist_ema20"] = (close - ema20) / a.replace(0, np.nan)
    feat["dist_ema50"] = (close - ema50) / a.replace(0, np.nan)
    feat["ret_5"] = close.pct_change(5)
    feat["ret_10"] = close.pct_change(10)
    feat["vol_10"] = close.pct_change().rolling(10).std()
    hi20, lo20 = df["High"].rolling(20).max(), df["Low"].rolling(20).min()
    feat["range_pos"] = (close - lo20) / (hi20 - lo20).replace(0, np.nan)
    return feat


def train_up_probability_model(df: pd.DataFrame, horizon: int = 5):
    feat = build_features(df)
    future_ret = df["Close"].pct_change(horizon).shift(-horizon)
    label = (future_ret > 0).astype(int)
    data = feat.copy()
    data["label"] = label
    data = data.dropna()
    if len(data) < ML_MIN_TRAIN_ROWS:
        return None, None
    X, y = data.drop(columns=["label"]), data["label"]
    model = RandomForestClassifier(n_estimators=200, max_depth=5,
                                    min_samples_leaf=20, random_state=42, n_jobs=-1)
    model.fit(X, y)
    latest = feat.iloc[[-1]].dropna(axis=1).reindex(columns=X.columns)
    if latest.isna().any(axis=None):
        return model, None
    return model, latest


def blended_confidence(df: pd.DataFrame, direction: str, rule_score: float) -> Tuple[float, str]:
    model, latest_row = train_up_probability_model(df)
    if model is None or latest_row is None:
        return rule_score, "ML filter skipped (insufficient history) — rule score used as-is"
    p_up = model.predict_proba(latest_row)[0][list(model.classes_).index(1)] * 100
    ml_conf = p_up if direction == "BUY" else (100 - p_up)
    final = 0.5 * rule_score + 0.5 * ml_conf
    note = f"ML P(favorable move)={ml_conf:.0f}%, blended with rule score {rule_score:.0f}%"
    return float(np.clip(final, 0, 100)), note


# =========================================================
# SCANNER
# =========================================================

def scan_all(min_confidence: float = ML_CONFIDENCE_FLOOR):
    rows, errors = [], []
    for yahoo_ticker, symbol in SYMBOLS:
        try:
            df = fetch_candles(yahoo_ticker, INTERVAL, LOOKBACK_PERIOD)
            sig = generate_rule_signal(symbol, df)
            if sig is None:
                continue
            confidence, note = blended_confidence(df, sig.direction, sig.rule_score)
            if confidence < min_confidence:
                continue
            rows.append({
                "symbol": symbol, "direction": sig.direction,
                "entry": round(sig.entry, 5), "stop_loss": round(sig.stop_loss, 5),
                "take_profit": round(sig.take_profit, 5),
                "confidence": round(confidence, 1),
                "reason": sig.reason, "ml_note": note, "as_of": df.index[-1],
            })
        except Exception as e:
            errors.append((symbol, str(e)))
    cols = ["symbol", "direction", "entry", "stop_loss", "take_profit",
            "confidence", "reason", "ml_note", "as_of"]
    result = pd.DataFrame(rows).sort_values("confidence", ascending=False) if rows else pd.DataFrame(columns=cols)
    return result, errors


# =========================================================
# BACKTESTER -- measures REAL historical win rate (no promises)
# =========================================================

def backtest_symbol(df: pd.DataFrame, min_gap_bars: int = 5):
    """
    Walks forward bar by bar. At each bar, checks if a rule signal exists
    using only data up to that bar (no lookahead). If so, simulates the
    trade forward until SL or TP is hit (or data runs out).
    Returns a DataFrame of closed trades and summary stats.
    """
    trades = []
    last_signal_bar = -10_000
    min_history = 60  # need enough bars for swings/ATR/trend lines to be meaningful

    for i in range(min_history, len(df) - 1):
        if i - last_signal_bar < min_gap_bars:
            continue
        sig = generate_rule_signal("BT", df, upto=i)
        if sig is None:
            continue

        last_signal_bar = i
        future = df.iloc[i + 1:]
        outcome, exit_price, bars_held = None, None, 0

        for j, (_, row) in enumerate(future.iterrows(), start=1):
            if sig.direction == "BUY":
                hit_sl = row["Low"] <= sig.stop_loss
                hit_tp = row["High"] >= sig.take_profit
            else:
                hit_sl = row["High"] >= sig.stop_loss
                hit_tp = row["Low"] <= sig.take_profit

            if hit_sl and hit_tp:
                # both touched in the same bar -- conservative assumption: SL hit first
                outcome, exit_price = "loss", sig.stop_loss
                bars_held = j
                break
            elif hit_sl:
                outcome, exit_price = "loss", sig.stop_loss
                bars_held = j
                break
            elif hit_tp:
                outcome, exit_price = "win", sig.take_profit
                bars_held = j
                break

        if outcome is None:
            continue  # trade never resolved within available data -- excluded, not counted either way

        risk = abs(sig.entry - sig.stop_loss)
        reward = abs(exit_price - sig.entry)
        r_multiple = (reward / risk) if outcome == "win" else -(risk / risk)
        trades.append({
            "time": df.index[i], "direction": sig.direction, "entry": sig.entry,
            "stop_loss": sig.stop_loss, "take_profit": sig.take_profit,
            "outcome": outcome, "bars_held": bars_held, "r_multiple": r_multiple,
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {}

    wins = (trades_df["outcome"] == "win").sum()
    total = len(trades_df)
    win_rate = 100 * wins / total
    avg_r = trades_df["r_multiple"].mean()
    gross_win = trades_df.loc[trades_df["outcome"] == "win", "r_multiple"].sum()
    gross_loss = -trades_df.loc[trades_df["outcome"] == "loss", "r_multiple"].sum()
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    stats = {
        "total_trades": total, "wins": int(wins), "losses": int(total - wins),
        "win_rate_pct": round(win_rate, 1), "avg_r_multiple": round(avg_r, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
    }
    return trades_df, stats


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="FX Order Block / Trend Line Scanner", layout="wide")
st.title("FX AI Signal Scanner")
st.caption(
    "Independent data feed (Yahoo Finance), not FxPro's own feed -- confirm live prices "
    "on FxPro before placing an order. Not financial advice. No win rate is assumed; "
    "use the Backtest tab to measure the real historical performance of these rules."
)

tab_scan, tab_chart, tab_bt = st.tabs(["📡 Scanner", "📈 Chart Viewer", "🧪 Backtest"])

with tab_scan:
    min_conf = st.slider("Minimum confidence to show", 0, 100, int(ML_CONFIDENCE_FLOOR))
    if st.button("Run scan", type="primary"):
        with st.spinner("Fetching candles and scoring signals..."):
            table, errors = scan_all(min_confidence=min_conf)
        if table.empty:
            st.info("No qualifying signals right now. Try lowering the confidence threshold.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)
        if errors:
            with st.expander(f"{len(errors)} symbol(s) skipped (data errors)"):
                for sym, err in errors:
                    st.write(f"**{sym}**: {err}")

with tab_chart:
    symbol_map = {sym: ticker for ticker, sym in SYMBOLS}
    chosen = st.selectbox("Symbol", list(symbol_map.keys()), key="chart_symbol")

    if st.button("Load chart", type="primary"):
        with st.spinner(f"Loading {chosen}..."):
            df = fetch_candles(symbol_map[chosen], INTERVAL, LOOKBACK_PERIOD)
            a = atr(df)
            obs = find_order_blocks(df, a)
            lines = fit_trend_lines(df)
            sig = generate_rule_signal(chosen, df)
            confidence, note = (None, None)
            if sig is not None:
                confidence, note = blended_confidence(df, sig.direction, sig.rule_score)

        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=chosen)])

        for ob in obs:
            color = "rgba(0,180,0,0.18)" if ob.kind == "bullish" else "rgba(200,0,0,0.18)"
            fig.add_shape(type="rect", x0=df.index[ob.idx], x1=df.index[-1],
                          y0=ob.bottom, y1=ob.top, fillcolor=color, line=dict(width=0), layer="below")
            fig.add_annotation(x=df.index[ob.idx], y=ob.top,
                                text=("Bull OB" if ob.kind == "bullish" else "Bear OB") + (" (mitigated)" if ob.mitigated else ""),
                                showarrow=False, yshift=10, font=dict(size=10))

        xs_int = list(range(len(df)))
        for line in lines:
            y_vals = [line.value_at(x) for x in xs_int]
            fig.add_trace(go.Scatter(x=df.index, y=y_vals, mode="lines",
                                      line=dict(dash="dot", width=1.5,
                                                color="blue" if line.kind == "support" else "orange"),
                                      name=f"{line.kind} trend line"))

        if sig is not None:
            for label, value, color in [("Entry", sig.entry, "black"),
                                          ("Stop Loss", sig.stop_loss, "red"),
                                          ("Take Profit", sig.take_profit, "green")]:
                fig.add_hline(y=value, line_dash="dash", line_color=color,
                              annotation_text=f"{label}: {value:.5f}", annotation_position="right")

        fig.update_layout(height=650, xaxis_rangeslider_visible=False,
                           margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if sig is not None:
            st.subheader(f"Signal: {sig.direction} {chosen}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entry", f"{sig.entry:.5f}")
            c2.metric("Stop Loss", f"{sig.stop_loss:.5f}")
            c3.metric("Take Profit", f"{sig.take_profit:.5f}")
            c4.metric("Confidence", f"{confidence:.0f}%" if confidence is not None else "n/a")
            st.write(f"**Rule basis:** {sig.reason}")
            if note:
                st.write(f"**ML note:** {note}")
        else:
            st.info("No active rule-based setup on this symbol right now.")

with tab_bt:
    st.write(
        "Runs the same rules against historical bars and simulates each trade "
        "to its actual SL/TP outcome, so you see a **measured** win rate -- "
        "not an assumed one."
    )
    symbol_map_bt = {sym: ticker for ticker, sym in SYMBOLS}
    bt_symbol = st.selectbox("Symbol", list(symbol_map_bt.keys()), key="bt_symbol")

    if st.button("Run backtest", type="primary"):
        with st.spinner(f"Backtesting {bt_symbol} over the last {LOOKBACK_PERIOD}..."):
            df = fetch_candles(symbol_map_bt[bt_symbol], INTERVAL, LOOKBACK_PERIOD)
            trades_df, stats = backtest_symbol(df)

        if not stats:
            st.info("No completed trades found in this window -- try a symbol with more data, "
                     "or note that few setups triggered.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", stats["total_trades"])
            c2.metric("Win rate", f"{stats['win_rate_pct']}%")
            c3.metric("Avg R-multiple", stats["avg_r_multiple"])
            c4.metric("Profit factor", stats["profit_factor"])
            st.caption(
                "R-multiple = trade result relative to risk (e.g. +2.0 means it made 2x the "
                "amount risked; -1.0 means a full stop-loss). Profit factor = gross wins ÷ gross "
                "losses in R -- above 1.0 means the approach was net profitable on this data, "
                "below 1.0 means it lost money even if win rate looks fine."
            )
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
            st.warning(
                "This is a small sample over recent history on one symbol/timeframe -- treat it "
                "as a rough signal, not proof. Past performance never guarantees future results."
            )
