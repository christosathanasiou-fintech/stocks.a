"""
Stock Analyzer — Technical + Fundamental Stock Analysis
=======================================================
Streamlit web app. Enter a stock symbol (e.g. AAPL, MSFT, TSLA),
and see: price, RSI, ATR, MACD, Bollinger Bands, SMA50/200, MA,
fundamentals (market cap, cash flow etc.), macro backdrop (Fed rates),
FOMC/ECB calendar, and an automatic trend assessment.

Run:
    pip install -r requirements.txt
    streamlit run app_en.py
"""

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# TECHNICAL INDICATORS (pure pandas/numpy)
# ----------------------------------------------------------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range — volatility measure."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands (mid, upper, lower)."""
    mid = sma(series, window)
    std = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    """Stochastic Oscillator (%K, %D) — 0-100, shows overbought/oversold."""
    low_k = df["Low"].rolling(k).min()
    high_k = df["High"].rolling(k).max()
    pct_k = 100 * (df["Close"] - low_k) / (high_k - low_k)
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d


def adx(df: pd.DataFrame, window: int = 14):
    """Average Directional Index — trend strength (0-100, >25 = strong trend)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_ = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return adx_, plus_di, minus_di


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — cumulative volume based on up/down moves."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).fillna(0).cumsum()


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Commodity Channel Index — deviation from the mean (±100 bounds)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
    return (tp - ma) / (0.015 * md)


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Williams %R — overbought/oversold (-100 to 0)."""
    high = df["High"].rolling(window).max()
    low = df["Low"].rolling(window).min()
    return -100 * (high - df["Close"]) / (high - low)


def roc(series: pd.Series, window: int = 12) -> pd.Series:
    """Rate of Change — percentage price change (momentum)."""
    return (series - series.shift(window)) / series.shift(window) * 100


def mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Money Flow Index — 'RSI with volume' (0-100)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(window).sum()
    neg_sum = neg.rolling(window).sum()
    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Chaikin Money Flow — buy/sell pressure via volume (-1 to +1)."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    mfv = mfm * df["Volume"]
    return mfv.rolling(window).sum() / df["Volume"].rolling(window).sum()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all indicators to an OHLCV DataFrame."""
    out = df.copy()
    close = out["Close"]
    # Μέσοι όροι & βασικοί
    out["SMA50"] = sma(close, 50)
    out["SMA200"] = sma(close, 200)
    out["MA20"] = sma(close, 20)
    out["EMA20"] = ema(close, 20)
    out["RSI"] = rsi(close, 14)
    out["ATR"] = atr(out, 14)
    out["MACD"], out["MACD_signal"], out["MACD_hist"] = macd(close)
    out["BB_mid"], out["BB_up"], out["BB_low"] = bollinger(close)
    # Επιπλέον δείκτες (για πιο ακριβή ανάλυση)
    out["STOCH_K"], out["STOCH_D"] = stochastic(out)
    out["ADX"], out["DI_plus"], out["DI_minus"] = adx(out)
    out["OBV"] = obv(out)
    out["OBV_ema"] = ema(out["OBV"], 20)
    out["CCI"] = cci(out)
    out["WILLR"] = williams_r(out)
    out["ROC"] = roc(close, 12)
    out["MFI"] = mfi(out)
    out["CMF"] = cmf(out)
    out["VOL_avg"] = sma(out["Volume"], 20)
    return out


# ----------------------------------------------------------------------------
# TREND DETECTION (combine signals -> score)
# ----------------------------------------------------------------------------
def trend_analysis(df: pd.DataFrame) -> dict:
    """Returns a trend score (-100 to +100) and individual signals."""
    last = df.iloc[-1]
    signals = []
    score = 0

    # 1. Θέση τιμής vs SMA200 (μακροπρόθεσμη τάση)
    if not np.isnan(last["SMA200"]):
        if last["Close"] > last["SMA200"]:
            score += 25; signals.append(("Price > SMA200", "bullish", "Long-term bullish"))
        else:
            score -= 25; signals.append(("Price < SMA200", "bearish", "Long-term bearish"))

    # 2. SMA50 vs SMA200 (golden/death cross)
    if not np.isnan(last["SMA50"]) and not np.isnan(last["SMA200"]):
        if last["SMA50"] > last["SMA200"]:
            score += 20; signals.append(("SMA50 > SMA200", "bullish", "Golden cross — bullish momentum"))
        else:
            score -= 20; signals.append(("SMA50 < SMA200", "bearish", "Death cross — bearish momentum"))

    # 3. RSI
    r = last["RSI"]
    if not np.isnan(r):
        if r > 70:
            score -= 10; signals.append((f"RSI {r:.0f}", "bearish", "Overbought — possible pullback"))
        elif r < 30:
            score += 10; signals.append((f"RSI {r:.0f}", "bullish", "Oversold — possible rebound"))
        elif r > 50:
            score += 5; signals.append((f"RSI {r:.0f}", "bullish", "Positive momentum"))
        else:
            score -= 5; signals.append((f"RSI {r:.0f}", "bearish", "Negative momentum"))

    # 4. MACD
    if not np.isnan(last["MACD"]) and not np.isnan(last["MACD_signal"]):
        if last["MACD"] > last["MACD_signal"]:
            score += 15; signals.append(("MACD > Signal", "bullish", "Bullish MACD signal"))
        else:
            score -= 15; signals.append(("MACD < Signal", "bearish", "Bearish MACD signal"))

    # 5. Bollinger position
    if not np.isnan(last["BB_up"]) and not np.isnan(last["BB_low"]):
        if last["Close"] > last["BB_up"]:
            score -= 5; signals.append(("Above BB", "bearish", "Possible excess"))
        elif last["Close"] < last["BB_low"]:
            score += 5; signals.append(("Below BB", "bullish", "Possibly oversold"))

    # 6. ADX — trend strength (reinforces the DI direction)
    if not np.isnan(last.get("ADX", np.nan)):
        if last["ADX"] > 25:
            if last["DI_plus"] > last["DI_minus"]:
                score += 12; signals.append((f"ADX {last['ADX']:.0f}", "bullish", "Strong uptrend"))
            else:
                score -= 12; signals.append((f"ADX {last['ADX']:.0f}", "bearish", "Strong downtrend"))

    # 7. Stochastic
    if not np.isnan(last.get("STOCH_K", np.nan)):
        k = last["STOCH_K"]
        if k > 80:
            score -= 6; signals.append((f"Stochastic {k:.0f}", "bearish", "Overbought zone"))
        elif k < 20:
            score += 6; signals.append((f"Stochastic {k:.0f}", "bullish", "Oversold zone"))

    # 8. MFI (money flow)
    if not np.isnan(last.get("MFI", np.nan)):
        m = last["MFI"]
        if m > 80:
            score -= 5; signals.append((f"MFI {m:.0f}", "bearish", "Excessive money inflow"))
        elif m < 20:
            score += 5; signals.append((f"MFI {m:.0f}", "bullish", "Excessive outflow — possible rebound"))

    # 9. CCI
    if not np.isnan(last.get("CCI", np.nan)):
        c = last["CCI"]
        if c > 100:
            score += 4; signals.append((f"CCI {c:.0f}", "bullish", "Strong bullish momentum"))
        elif c < -100:
            score -= 4; signals.append((f"CCI {c:.0f}", "bearish", "Strong bearish momentum"))

    # 10. Williams %R
    if not np.isnan(last.get("WILLR", np.nan)):
        w = last["WILLR"]
        if w > -20:
            score -= 4; signals.append((f"Williams %R {w:.0f}", "bearish", "Overbought"))
        elif w < -80:
            score += 4; signals.append((f"Williams %R {w:.0f}", "bullish", "Oversold"))

    # 11. ROC (momentum)
    if not np.isnan(last.get("ROC", np.nan)):
        rc = last["ROC"]
        if rc > 0:
            score += 4; signals.append((f"ROC {rc:+.1f}%", "bullish", "Positive price momentum"))
        else:
            score -= 4; signals.append((f"ROC {rc:+.1f}%", "bearish", "Negative price momentum"))

    # 12. OBV vs EMA (volume trend)
    if not np.isnan(last.get("OBV", np.nan)) and not np.isnan(last.get("OBV_ema", np.nan)):
        if last["OBV"] > last["OBV_ema"]:
            score += 5; signals.append(("OBV rising", "bullish", "Volume supports the rise"))
        else:
            score -= 5; signals.append(("OBV falling", "bearish", "Volume supports the decline"))

    # 13. CMF (Chaikin money flow)
    if not np.isnan(last.get("CMF", np.nan)):
        cm = last["CMF"]
        if cm > 0.05:
            score += 4; signals.append(("CMF positive", "bullish", "Buying pressure"))
        elif cm < -0.05:
            score -= 4; signals.append(("CMF negative", "bearish", "Selling pressure"))

    score = max(-100, min(100, score))
    if score >= 40:
        verdict, color = "STRONGLY BULLISH", "#1a9850"
    elif score >= 10:
        verdict, color = "MILDLY BULLISH", "#66bd63"
    elif score > -10:
        verdict, color = "NEUTRAL / SIDEWAYS", "#999999"
    elif score > -40:
        verdict, color = "MILDLY BEARISH", "#f46d43"
    else:
        verdict, color = "STRONGLY BEARISH", "#d73027"

    return {"score": score, "verdict": verdict, "color": color, "signals": signals}


def simple_explanation(trend: dict, last) -> str:
    """2-3 line explanation in very simple words (as if to a child)."""
    score = trend["score"]
    rsi_val = last["RSI"]

    # Βασική «ιστορία» ανάλογα με το σκορ
    if score >= 40:
        story = ("The stock has been rising steadily lately and shows strength — "
                 "like a child running who still has lots of energy.")
    elif score >= 10:
        story = ("The stock is rising mildly, but not very strongly — "
                 "like walking uphill at a slow pace.")
    elif score > -10:
        story = ("The stock is neither clearly rising nor falling — "
                 "it's moving in place, as if resting.")
    elif score > -40:
        story = ("The stock is falling mildly — "
                 "like a ball rolling slowly downhill.")
    else:
        story = ("The stock has been falling notably lately and shows weakness — "
                 "like a ball tumbling quickly down a slope.")

    # Σχόλιο RSI σε απλά λόγια
    if not np.isnan(rsi_val):
        if rsi_val > 70:
            rsi_note = " It has risen very fast, so it may need a small rest (pullback)."
        elif rsi_val < 30:
            rsi_note = " It has fallen a lot, so it may be tired and want to bounce up a bit."
        else:
            rsi_note = ""
    else:
        rsi_note = ""

    return story + rsi_note


def _all_indicator_explanations(last, close) -> list:
    """Builds an explanation for EACH indicator plus an importance score (0-10).
    Returns a list of dicts: {title, value, text, importance, kind}.
    The more extreme/clear an indicator's signal, the higher its importance,
    so the most critical ones for this asset are auto-selected."""
    out = []

    def add(title, value, text, importance, kind="neutral"):
        out.append({"title": title, "value": value, "text": text,
                    "importance": importance, "kind": kind})

    # --- RSI ---
    r = last.get("RSI", np.nan)
    if not np.isnan(r):
        if r > 70:
            add("RSI (strength index)", f"{r:.0f}",
                f"RSI is {r:.0f}, above 70 ('overbought'): it rose very fast and "
                "may need a rest or small pullback. Like a runner catching their breath.", 9, "bearish")
        elif r < 30:
            add("RSI (strength index)", f"{r:.0f}",
                f"RSI is {r:.0f}, below 30 ('oversold'): it fell a lot and may be "
                "ready to rebound. Like a spring that's been pressed and wants to snap back.", 9, "bullish")
        elif r >= 50:
            add("RSI (strength index)", f"{r:.0f}",
                f"RSI is {r:.0f}, slightly above the midpoint (50): buyers have a slight "
                "edge — healthy, without excess.", 4, "bullish")
        else:
            add("RSI (strength index)", f"{r:.0f}",
                f"RSI is {r:.0f}, slightly below the midpoint (50): sellers have a slight "
                "edge, but nothing extreme.", 4, "bearish")

    # --- SMA50 vs SMA200 ---
    s50, s200 = last.get("SMA50", np.nan), last.get("SMA200", np.nan)
    if not (np.isnan(s50) or np.isnan(s200)):
        gap = abs(s50 - s200) / s200 * 100 if s200 else 0
        imp = 8 if gap > 3 else 6
        if s50 > s200:
            add("Moving averages (SMA50 vs SMA200)", "Bullish",
                f"The 50-day average ({s50:.2f}) is above the 200-day ({s200:.2f}) — "
                "a 'golden cross', i.e. an uptrend: recent action is better than the long-term.",
                imp, "bullish")
        else:
            add("Moving averages (SMA50 vs SMA200)", "Bearish",
                f"The 50-day average ({s50:.2f}) is below the 200-day ({s200:.2f}) — "
                "a 'death cross', i.e. a downtrend: recent action is worse than the long-term.",
                imp, "bearish")

    # --- Price vs SMA200 ---
    if not np.isnan(s200):
        if close > s200:
            add("Price relative to SMA200", "Above",
                f"The price ({close:.2f}) is above the 200-day average ({s200:.2f}): "
                "a long-term bullish 'climate'.", 7, "bullish")
        else:
            add("Price relative to SMA200", "Below",
                f"The price ({close:.2f}) is below the 200-day average ({s200:.2f}): "
                "a long-term bearish 'climate'.", 7, "bearish")

    # --- MACD ---
    m, sig = last.get("MACD", np.nan), last.get("MACD_signal", np.nan)
    if not (np.isnan(m) or np.isnan(sig)):
        if m > sig:
            add("MACD (momentum)", "Bullish",
                "The MACD line is above the 'signal': momentum is turning upward.",
                7, "bullish")
        else:
            add("MACD (momentum)", "Bearish",
                "The MACD line is below the 'signal': momentum is turning downward.",
                7, "bearish")

    # --- Bollinger Bands ---
    bb_up, bb_low, bb_mid = last.get("BB_up", np.nan), last.get("BB_low", np.nan), last.get("BB_mid", np.nan)
    if not (np.isnan(bb_up) or np.isnan(bb_low)):
        if close > bb_up:
            add("Bollinger Bands (price range)", "Upper band",
                "The price broke above the upper band: it rose sharply and may 'revert' a bit downward.",
                7, "bearish")
        elif close < bb_low:
            add("Bollinger Bands (price range)", "Lower band",
                "The price fell below the lower band: it dropped sharply and may 'bounce' upward.",
                7, "bullish")
        else:
            add("Bollinger Bands (price range)", "Within range",
                f"The price is moving between the bands (around the mid {bb_mid:.2f}): normal range, "
                "no extreme move right now.", 3, "neutral")

    # --- ADX (trend strength) ---
    ax = last.get("ADX", np.nan)
    if not np.isnan(ax):
        dip, dim = last.get("DI_plus", np.nan), last.get("DI_minus", np.nan)
        if ax > 25:
            direction = "bullish" if dip > dim else "bearish"
            add("ADX (trend strength)", f"{ax:.0f}",
                f"ADX is {ax:.0f} (>25): there's a strong {direction} trend — the move has 'fuel' "
                "and isn't just noise.", 8, "bullish" if dip > dim else "bearish")
        else:
            add("ADX (trend strength)", f"{ax:.0f}",
                f"ADX is {ax:.0f} (<25): weak or no trend — the price is likely moving "
                "sideways, without a clear direction.", 5, "neutral")

    # --- Stochastic ---
    k = last.get("STOCH_K", np.nan)
    if not np.isnan(k):
        if k > 80:
            add("Stochastic (speed)", f"{k:.0f}",
                f"Stochastic is {k:.0f} (>80): overbought zone — the rise may be tiring.",
                6, "bearish")
        elif k < 20:
            add("Stochastic (speed)", f"{k:.0f}",
                f"Stochastic is {k:.0f} (<20): oversold zone — a rebound is possible.",
                6, "bullish")
        else:
            add("Stochastic (speed)", f"{k:.0f}",
                f"Stochastic is {k:.0f}: neutral zone, no extreme signal.", 2, "neutral")

    # --- MFI (money flow) ---
    mf = last.get("MFI", np.nan)
    if not np.isnan(mf):
        if mf > 80:
            add("MFI (money flow)", f"{mf:.0f}",
                f"MFI is {mf:.0f} (>80): excessive money inflow — watch for a possible pullback.",
                6, "bearish")
        elif mf < 20:
            add("MFI (money flow)", f"{mf:.0f}",
                f"MFI is {mf:.0f} (<20): excessive money outflow — a rebound is possible.",
                6, "bullish")
        else:
            add("MFI (money flow)", f"{mf:.0f}",
                f"MFI is {mf:.0f}: balanced money flow, no extreme signal.", 2, "neutral")

    # --- CCI ---
    c = last.get("CCI", np.nan)
    if not np.isnan(c):
        if c > 100:
            add("CCI (momentum)", f"{c:.0f}",
                f"CCI is {c:.0f} (>100): strong bullish momentum, but also possible overbought.", 5, "bullish")
        elif c < -100:
            add("CCI (momentum)", f"{c:.0f}",
                f"CCI is {c:.0f} (<-100): strong bearish momentum, but also possible oversold.", 5, "bearish")
        else:
            add("CCI (momentum)", f"{c:.0f}",
                f"CCI is {c:.0f}: within the normal range (±100).", 2, "neutral")

    # --- Williams %R ---
    w = last.get("WILLR", np.nan)
    if not np.isnan(w):
        if w > -20:
            add("Williams %R", f"{w:.0f}",
                f"Williams %R is {w:.0f} (>-20): overbought — possible fatigue.", 5, "bearish")
        elif w < -80:
            add("Williams %R", f"{w:.0f}",
                f"Williams %R is {w:.0f} (<-80): oversold — a rebound is possible.", 5, "bullish")

    # --- ROC (momentum) ---
    rc = last.get("ROC", np.nan)
    if not np.isnan(rc):
        strong = abs(rc) > 8
        add("ROC (price momentum)", f"{rc:+.1f}%",
            f"The price {'rose' if rc > 0 else 'fell'} {abs(rc):.1f}% in ~2 weeks — "
            f"{'strong' if strong else 'mild'} {'positive' if rc > 0 else 'negative'} momentum.",
            6 if strong else 3, "bullish" if rc > 0 else "bearish")

    # --- OBV (volume trend) ---
    ob, obe = last.get("OBV", np.nan), last.get("OBV_ema", np.nan)
    if not (np.isnan(ob) or np.isnan(obe)):
        if ob > obe:
            add("OBV (volume)", "Bullish",
                "Cumulative volume (OBV) is rising: trading volume supports the advance — a good sign.",
                5, "bullish")
        else:
            add("OBV (volume)", "Bearish",
                "Cumulative volume (OBV) is falling: volume supports the decline.", 5, "bearish")

    # --- CMF ---
    cm = last.get("CMF", np.nan)
    if not np.isnan(cm):
        if cm > 0.05:
            add("CMF (buying pressure)", f"{cm:+.2f}",
                f"CMF is {cm:+.2f} (positive): buying pressure dominates — 'smart' money is coming in.",
                4, "bullish")
        elif cm < -0.05:
            add("CMF (buying pressure)", f"{cm:+.2f}",
                f"CMF is {cm:+.2f} (negative): selling pressure dominates.", 4, "bearish")

    # --- ATR (volatility) ---
    a = last.get("ATR", np.nan)
    if not np.isnan(a):
        pct = a / close * 100 if close else 0
        if pct > 4:
            level, extra, imp = "high", "It moves sharply — higher risk but also opportunity.", 5
        elif pct > 2:
            level, extra, imp = "moderate", "Normal daily fluctuations.", 3
        else:
            level, extra, imp = "low", "It moves calmly, with small daily changes.", 3
        add("ATR (volatility)", f"{a:.2f}",
            f"ATR is {a:.2f} (~{pct:.1f}% of price): {level} volatility — how 'jumpy' "
            f"the price is. {extra}", imp, "neutral")

    return out


def explain_indicators(last, df, top_n: int = 5) -> list:
    """Returns only the top_n most important indicators for this asset,
    as a list of (title, value, explanation). Analysis uses ALL indicators,
    but only the most critical ones are shown right now."""
    allx = _all_indicator_explanations(last, last["Close"])
    # Ταξινόμηση κατά σημαντικότητα (φθίνουσα), κρατάμε τους top_n
    allx.sort(key=lambda d: d["importance"], reverse=True)
    top = allx[:top_n]
    return [(d["title"], d["value"], d["text"]) for d in top]


def buy_hold_sell(score: int) -> dict:
    """Converts the trend score into Buy / Hold / Sell percentages (sum to 100)."""
    # Όσο πιο ανοδικό το σκορ, τόσο μεγαλύτερο το Buy· όσο πιο καθοδικό, τόσο το Sell.
    # Το Hold είναι μεγαλύτερο όταν το σκορ είναι κοντά στο μηδέν (αβεβαιότητα).
    s = max(-100, min(100, score))
    buy = max(0, s)               # 0..100
    sell = max(0, -s)             # 0..100
    hold = 100 - abs(s)           # larger near 0
    total = buy + hold + sell
    if total == 0:
        buy, hold, sell = 0, 100, 0
        total = 100
    return {
        "Buy": round(buy / total * 100),
        "Hold": round(hold / total * 100),
        "Sell": round(sell / total * 100),
    }


# ----------------------------------------------------------------------------
# FUNDAMENTAL — company data via yfinance
# ----------------------------------------------------------------------------
def fmt_num(n):
    """Formats large numbers (T/B/M)."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.2f}"


@st.cache_data(ttl=900, show_spinner=False)
def load_stock(ticker: str, period: str):
    """Downloads history + fundamentals, with automatic retries
    in case of temporary Yahoo rate-limiting. Cached 15 minutes."""
    import time

    hist = None
    last_err = None
    # Έως 3 προσπάθειες με αυξανόμενη αναμονή (1s, 2s, 4s)
    for attempt in range(3):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, auto_adjust=False)
            if hist is not None and not hist.empty:
                break  # επιτυχία
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)  # 1, 2, 4 seconds

    # Αν μετά τις προσπάθειες δεν έχουμε δεδομένα, επιστρέφουμε άδειο + λόγο
    if hist is None or hist.empty:
        return None, {}, None, "rate_limit_or_invalid"

    # Θεμελιώδη (δεν χαλάει η ροή αν αποτύχουν)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    cashflow = None
    try:
        cashflow = tk.cashflow
    except Exception:
        cashflow = None
    return hist, info, cashflow, "ok"


@st.cache_data(ttl=1800, show_spinner=False)
def load_news(ticker: str, limit: int = 6):
    """Recent news for the symbol via yfinance. Cached 30 minutes."""
    items = []
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        raw = []
    for art in raw[:limit]:
        # Το yfinance επιστρέφει είτε επίπεδα κλειδιά είτε μέσα σε 'content'
        content = art.get("content", art) if isinstance(art, dict) else {}
        title = content.get("title") or art.get("title")
        if not title:
            continue
        # Πηγή
        provider = ""
        prov = content.get("provider") or {}
        if isinstance(prov, dict):
            provider = prov.get("displayName", "")
        provider = provider or art.get("publisher", "")
        # Σύνδεσμος
        link = ""
        cu = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        if isinstance(cu, dict):
            link = cu.get("url", "")
        link = link or art.get("link", "")
        # Ημερομηνία
        pub = content.get("pubDate") or art.get("providerPublishTime", "")
        items.append({"title": title, "provider": provider, "link": link, "pub": pub})
    return items


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro():
    """Macro backdrop via yfinance (no FRED key): 10Y yield, VIX, DXY."""
    out = {}
    for name, sym in [("10Y Treasury", "^TNX"), ("VIX (fear)", "^VIX"), ("Dollar Index", "DX-Y.NYB")]:
        try:
            h = yf.Ticker(sym).history(period="1mo")
            if not h.empty:
                cur = h["Close"].iloc[-1]
                prev = h["Close"].iloc[0]
                chg = (cur - prev) / prev * 100
                out[name] = (cur, chg)
        except Exception:
            pass
    return out


# ----------------------------------------------------------------------------
# FOMC / ECB CALENDAR (static — official dates)
# ----------------------------------------------------------------------------
FOMC_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
    "2027-07-28", "2027-09-22", "2027-11-03", "2027-12-15",
]
ECB_DATES = [
    "2025-01-30", "2025-03-06", "2025-04-17", "2025-06-05",
    "2025-07-24", "2025-09-11", "2025-10-30", "2025-12-18",
    "2026-01-29", "2026-03-12", "2026-04-16", "2026-06-04",
    "2026-07-23", "2026-09-10", "2026-10-29", "2026-12-17",
    "2027-02-04", "2027-03-18", "2027-04-22", "2027-06-10",
    "2027-07-22", "2027-09-09", "2027-10-28", "2027-12-16",
]


def next_meeting(dates):
    today = dt.date.today()
    upcoming = [dt.date.fromisoformat(d) for d in dates if dt.date.fromisoformat(d) >= today]
    return min(upcoming) if upcoming else None


# ----------------------------------------------------------------------------
# HIGH-IMPACT EVENTS (fixed dates + impact explanation)
# We don't predict the reaction — we explain WHAT TO WATCH per event type.
# ----------------------------------------------------------------------------
def build_upcoming_events():
    """Builds a list of upcoming high-impact events, sorted by date."""
    events = []
    for d in FOMC_DATES:
        events.append({
            "date": d, "flag": "🇺🇸", "name": "FOMC — Fed rate decision",
            "impact": "Very high",
            "affects": "Stocks (esp. tech), gold, the dollar, bonds",
            "note": ("If the Fed is more 'hawkish' than expected (higher rates or a "
                     "strict message), historically stocks & gold get pressured and the dollar strengthens. "
                     "If more 'dovish' (lower rates/soft message), usually "
                     "the opposite happens. What matters is the deviation from the forecast, not the number itself."),
        })
    for d in ECB_DATES:
        events.append({
            "date": d, "flag": "🇪🇺", "name": "ECB — rate decision",
            "impact": "High",
            "affects": "European & Greek stocks, the euro, European bonds",
            "note": ("Directly affects banks & European stocks. Higher-than-expected "
                     "rates usually help banks but pressure indebted companies and the "
                     "broader risk mood."),
        })
    # Ταξινόμηση & κράτημα μόνο των μελλοντικών
    today = dt.date.today()
    events = [e for e in events if dt.date.fromisoformat(e["date"]) >= today]
    events.sort(key=lambda e: e["date"])
    return events


# Recurring high-impact events (what to watch — general explanation)
RECURRING_EVENTS_INFO = [
    ("📊 US Inflation report (CPI)", "Monthly",
     "One of the most critical numbers. Higher-than-expected inflation → fear of "
     "higher rates → pressure on stocks. Lower → relief for markets."),
    ("💼 US Non-Farm Payrolls (NFP)", "1st Friday of each month",
     "Shows the health of the labor market. Very strong data can paradoxically pressure "
     "stocks (fear of higher rates); very weak data raises recession fears."),
    ("🏢 Earnings season", "Every quarter",
     "Companies report profits. Results vs analyst forecasts move "
     "the specific stock sharply — check the earnings date in the 'Fundamentals' tab."),
    ("🛢️ Oil inventories (EIA)", "Every Wednesday",
     "Strongly affects oil & energy stocks. Larger-than-expected "
     "inventories usually push the oil price down."),
]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0; }
    .subtitle { color: #888; margin-top: 0; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    /* Hide only the top-right hamburger menu, toolbar and footer. */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    div[data-testid="stToolbar"] { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
    header[data-testid="stHeader"] { background: transparent; }
    /* Keep the settings sidebar permanently open: hide the collapse (X / arrow)
       button so it can't be closed, and force the sidebar to stay visible. */
    div[data-testid="stSidebarCollapseButton"],
    button[data-testid="stBaseButton-headerNoPadding"],
    div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
    section[data-testid="stSidebar"] {
        visibility: visible !important;
        transform: none !important;
        min-width: 300px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📈 Stock Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Technical + fundamental stock analysis with a macro backdrop</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")

    # Markets: name -> (Yahoo suffix, example symbols)
    MARKETS = {
        "🇺🇸 USA (NYSE/Nasdaq)": ("", ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]),
        "🇬🇷 Greece (ATHEX)":      (".AT", ["ETE.AT", "OPAP.AT", "AEGN.AT", "MYTIL.AT", "EUROB.AT"]),
        "🇩🇪 Germany (Xetra)":   (".DE", ["BMW.DE", "SAP.DE", "VOW3.DE", "SIE.DE"]),
        "🇫🇷 France (Euronext)":  (".PA", ["MC.PA", "AIR.PA", "OR.PA", "BNP.PA"]),
        "🇳🇱 Netherlands (Euronext)": (".AS", ["ASML.AS", "INGA.AS", "HEIA.AS"]),
        "🇬🇧 London (LSE)":      (".L", ["HSBA.L", "BP.L", "SHEL.L", "VOD.L"]),
        "🇯🇵 Tokyo (TSE)":        (".T", ["7203.T", "6758.T", "9984.T", "8306.T"]),
        "🇦🇺 Sydney (ASX)":       (".AX", ["BHP.AX", "CBA.AX", "CSL.AX", "WBC.AX"]),
        "🛢️ Commodities": ("LIST:COMMODITY", []),
        "📈 ETFs / Indices": ("LIST:ETF", []),
        "₿ Crypto": ("LIST:CRYPTO", []),
        "📜 Bonds (bonds & ETFs)": ("LIST:BONDS", []),
        "✏️ Other / type it myself": (None, []),
    }

    # Λίστες φιλικών ονομάτων -> σύμβολα Yahoo
    COMMODITIES = {
        "🥇 Gold": "GC=F",
        "🥈 Silver": "SI=F",
        "🛢️ WTI Crude Oil": "CL=F",
        "🛢️ Brent Crude": "BZ=F",
        "🔥 Natural Gas": "NG=F",
        "🟤 Copper": "HG=F",
        "🪙 Platinum": "PL=F",
        "🌾 Wheat": "ZW=F",
        "🌽 Corn": "ZC=F",
        "☕ Coffee": "KC=F",
    }
    CRYPTO = {
        "₿ Bitcoin": "BTC-USD",
        "Ξ Ethereum": "ETH-USD",
        "◎ Solana": "SOL-USD",
        "✕ XRP": "XRP-USD",
        "🐕 Dogecoin": "DOGE-USD",
        "🔷 Cardano": "ADA-USD",
        "🔺 Avalanche": "AVAX-USD",
        "🔗 Chainlink": "LINK-USD",
    }
    BONDS = {
        "🇺🇸 US 10Y Treasury (yield)": "^TNX",
        "🇺🇸 US 30Y Treasury (yield)": "^TYX",
        "🇺🇸 US 5Y Treasury (yield)": "^FVX",
        "🇺🇸 US 13-week Treasury (yield)": "^IRX",
        "📦 TLT — US 20+ year (ETF)": "TLT",
        "📦 IEF — US 7-10 year (ETF)": "IEF",
        "📦 SHY — US 1-3 year (ETF)": "SHY",
        "📦 AGG — Total bond market (ETF)": "AGG",
        "📦 LQD — Corporate bonds (ETF)": "LQD",
        "📦 HYG — High-yield bonds (ETF)": "HYG",
    }
    ETFS = {
        "🇺🇸 SPY — S&P 500 (USA)": "SPY",
        "🇺🇸 VOO — S&P 500 (Vanguard)": "VOO",
        "🇺🇸 IVV — S&P 500 (iShares)": "IVV",
        "🇺🇸 QQQ — Nasdaq 100 (tech)": "QQQ",
        "🇺🇸 DIA — Dow Jones 30": "DIA",
        "🇺🇸 IWM — Russell 2000 (small caps)": "IWM",
        "🇺🇸 VTI — Total US market": "VTI",
        "🌍 VT — Whole world (Vanguard)": "VT",
        "🌍 VWCE.DE — FTSE All-World (EU/€)": "VWCE.DE",
        "🇪🇺 VUAA.DE — S&P 500 (EU/€, acc)": "VUAA.DE",
        "🇬🇧 VUAA.L — S&P 500 (London)": "VUAA.L",
        "🇪🇺 CSPX.L — S&P 500 (iShares, €)": "CSPX.L",
        "🇪🇺 EUNL.DE — MSCI World (iShares)": "EUNL.DE",
        "🌏 EEM — Emerging markets": "EEM",
        "🇪🇺 EZU — Eurozone": "EZU",
        "📊 ^GSPC — S&P 500 index (raw)": "^GSPC",
        "📊 ^NDX — Nasdaq 100 index (raw)": "^NDX",
        "📊 ^DJI — Dow Jones index (raw)": "^DJI",
        "📊 ^GDAXI — Germany DAX index": "^GDAXI",
        "📊 GD.AT — ATHEX Composite index": "GD.AT",
    }
    PICK_LISTS = {
        "LIST:COMMODITY": ("commodity", COMMODITIES,
                           "Commodities have no fundamentals — technical analysis only."),
        "LIST:ETF": ("ETF / index", ETFS,
                     "ETFs are 'baskets' of many stocks. The ^ ones are indices. Technical analysis works fully."),
        "LIST:CRYPTO": ("cryptocurrency", CRYPTO,
                        "Cryptocurrencies have no fundamentals — technical analysis only. Note: very high volatility."),
        "LIST:BONDS": ("bond / ETF", BONDS,
                       "Yields (^) show the return; ETFs (📦) trade like stocks."),
    }

    market = st.selectbox("Market / Exchange", list(MARKETS.keys()), index=0)
    suffix, examples = MARKETS[market]

    if suffix in PICK_LISTS:
        kind_label, pick_dict, note = PICK_LISTS[suffix]
        choice = st.selectbox(f"Choose {kind_label}", list(pick_dict.keys()), index=0)
        ticker = pick_dict[choice]
        st.caption(note)
    else:
        is_manual = suffix is None
        raw = st.text_input(
            "Symbol" if is_manual else "Stock symbol",
            value=examples[0] if examples else "AAPL",
            help=("Type any Yahoo Finance symbol: stock, ETF, index, "
                  "futures — with the correct market suffix if needed."
                  if is_manual else
                  "Type just the symbol — the market suffix is added automatically."),
        ).strip().upper()

        # Αυτόματη προσθήκη κατάληξης (αν δεν την έχει ήδη γράψει ο χρήστης)
        if suffix and raw and not raw.endswith(suffix):
            # αν έγραψε ήδη κάποια άλλη κατάληξη (π.χ. .DE) σεβόμαστε αυτό που έγραψε
            if "." not in raw:
                ticker = raw + suffix
            else:
                ticker = raw
        else:
            ticker = raw

        if is_manual:
            st.caption("💡 Here you can analyze **anything** on Yahoo Finance — "
                       "any stock, ETF or index in the world.")

        if examples:
            st.caption("Popular: " + " · ".join(examples))

    period = st.selectbox(
        "Time period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3,
        format_func=lambda p: {
            "1mo": "1 month", "3mo": "3 months", "6mo": "6 months",
            "1y": "1 year", "2y": "2 years", "5y": "5 years", "max": "Max",
        }.get(p, p),
    )
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
    if suffix:
        st.caption(f"🔎 Will analyze: **{ticker}**")
    st.divider()
    st.caption("Price source: Yahoo Finance · Macro: TNX/VIX/DXY")
    st.caption("⚠️ For educational purposes only — not investment advice.")

if analyze or ticker:
    try:
        with st.spinner(f"Loading data for {ticker}… (with automatic retries)"):
            hist, info, cashflow, status = load_stock(ticker, period)

        if hist is None or hist.empty:
            st.error(
                f"⚠️ No data found for '{ticker}' right now.\n\n"
                "If the symbol is correct (e.g. NVDA, AAPL), most likely "
                "**Yahoo Finance temporarily blocked you** because too many requests were made quickly "
                "(rate-limit). It's not your fault nor the symbol's."
            )
            cretry, cinfo = st.columns([1, 2])
            with cretry:
                if st.button("🔄 Clear cache & retry"):
                    st.cache_data.clear()
                    st.rerun()
            with cinfo:
                st.caption(
                    "Or wait 30-60 seconds and press Analyze again. "
                    "If it persists, close the app (Ctrl+C) and reopen it."
                )
            st.divider()
            st.caption(
                "Also check: 🇯🇵 Tokyo symbols are **numbers** (7203.T = Toyota) · "
                "🇬🇷 Greece uses the **.AT** suffix (ETE.AT)."
            )
            st.stop()

        df = compute_indicators(hist)
        last = df.iloc[-1]
        trend = trend_analysis(df)
        name = info.get("longName") or info.get("shortName") or ticker
        currency = info.get("currency", "USD")

        # --- Κεφαλίδα: τιμή + τάση ---
        price = last["Close"]
        prev = df["Close"].iloc[-2] if len(df) > 1 else price
        chg = (price - prev) / prev * 100

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1.4])
        c1.markdown(f"### {name}")
        c1.caption(f"{ticker} · {info.get('sector', '')} {info.get('industry', '')}")
        c2.metric("Price", f"{price:.2f} {currency}", f"{chg:+.2f}%")
        c3.metric("RSI (14)", f"{last['RSI']:.0f}" if not np.isnan(last['RSI']) else "—")
        c4.markdown(
            f"<div style='padding:8px 14px;border-radius:10px;background:{trend['color']}22;"
            f"border:1px solid {trend['color']};text-align:center'>"
            f"<div style='font-size:0.75rem;color:#888'>TREND (score {trend['score']:+d})</div>"
            f"<div style='font-weight:800;color:{trend['color']}'>{trend['verdict']}</div></div>",
            unsafe_allow_html=True,
        )

        # --- ΑΠΛΗ ΕΞΗΓΗΣΗ + BUY/HOLD/SELL ΠΙΤΑ ---
        st.markdown("")  # μικρό κενό
        sum_left, sum_right = st.columns([1.4, 1])

        with sum_left:
            st.markdown("#### 💡 In simple words")
            st.markdown(
                f"<div style='padding:14px 18px;border-radius:12px;background:{trend['color']}15;"
                f"border-left:4px solid {trend['color']};font-size:1.05rem;line-height:1.6'>"
                f"{simple_explanation(trend, last)}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "This is an automatic summary of the technical indicators (RSI, MACD, moving averages etc.) — "
                "not a personal opinion or prediction."
            )

        with sum_right:
            bhs = buy_hold_sell(trend["score"])
            pie = go.Figure(data=[go.Pie(
                labels=["Buy", "Hold", "Sell"],
                values=[bhs["Buy"], bhs["Hold"], bhs["Sell"]],
                hole=0.5,
                marker=dict(colors=["#1a9850", "#bdbdbd", "#d73027"],
                            line=dict(color="#ffffff", width=2)),
                textinfo="percent",
                textposition="outside",
                textfont=dict(size=13),
                sort=False,
                direction="clockwise",
            )])
            # Ποια ετικέτα κυριαρχεί
            dominant = max(bhs, key=bhs.get)
            dom_label = {"Buy": "Buy", "Hold": "Hold", "Sell": "Sell"}[dominant]
            dom_color = {"Buy": "#1a9850", "Hold": "#999999", "Sell": "#d73027"}[dominant]
            pie.update_layout(
                height=340,
                margin=dict(t=60, b=60, l=20, r=20),
                template="plotly_white",
                title=dict(text="Analysis signal", x=0.5, xanchor="center", y=0.97, font=dict(size=15)),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                            xanchor="center", x=0.5, font=dict(size=12)),
                annotations=[dict(text=f"<b>{dom_label}</b>", x=0.5, y=0.5,
                                  font=dict(size=17, color=dom_color), showarrow=False)],
            )
            st.plotly_chart(pie, use_container_width=True)

        st.divider()

        # --- Tabs ---
        tab_chart, tab_tech, tab_fund, tab_news, tab_macro = st.tabs(
            ["📊 Chart", "🔧 Technical signals", "🏢 Fundamentals", "📰 News", "🌍 Macro & Events"]
        )

        # ===== TAB 1: Chart =====
        with tab_chart:
            chart_type = st.radio(
                "Chart type",
                ["Candlesticks", "Line"],
                horizontal=True,
                index=0,
            )

            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                row_heights=[0.5, 0.17, 0.17, 0.16], vertical_spacing=0.06,
                subplot_titles=("Price + Bollinger + SMA", "Volume", "RSI", "MACD"),
            )
            # Price: κεριά ή απλή γραμμή, ανάλογα με την επιλογή
            if chart_type.startswith("Candl"):
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    name="Price", increasing_line_color="#1a9850", decreasing_line_color="#d73027",
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["Close"], name="Price (close)",
                    line=dict(color="#111111", width=1.6),
                ), row=1, col=1)
            for col, color, w in [("SMA50", "#2166ac", 1.2), ("SMA200", "#b2182b", 1.2), ("MA20", "#f1a340", 1)]:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color, width=w)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB up", line=dict(color="#888", width=0.7, dash="dot"), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_low"], name="BB low", line=dict(color="#888", width=0.7, dash="dot"), fill="tonexty", fillcolor="rgba(120,120,120,0.08)", showlegend=False), row=1, col=1)
            # Volume
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#aaa"), row=2, col=1)
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#762a83")), row=3, col=1)
            fig.add_hline(y=70, line=dict(color="#d73027", width=0.7, dash="dash"), row=3, col=1)
            fig.add_hline(y=30, line=dict(color="#1a9850", width=0.7, dash="dash"), row=3, col=1)
            # MACD
            colors = ["#1a9850" if v >= 0 else "#d73027" for v in df["MACD_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist", marker_color=colors), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#2166ac", width=1)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="#f1a340", width=1)), row=4, col=1)

            fig.update_layout(
                height=860, template="plotly_white", xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.06,
                            xanchor="center", x=0.5, font=dict(size=11)),
                margin=dict(t=95, b=25),
            )
            # Μικρότερη γραμματοσειρά για τους τίτλους των υπο-γραφημάτων ώστε να μην πατάνε
            fig.update_annotations(font_size=13)
            st.plotly_chart(fig, use_container_width=True)

        # ===== TAB 2: Technical signals =====
        with tab_tech:
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("SMA50", f"{last['SMA50']:.2f}" if not np.isnan(last['SMA50']) else "—")
            cc2.metric("SMA200", f"{last['SMA200']:.2f}" if not np.isnan(last['SMA200']) else "—")
            cc3.metric("ATR (14)", f"{last['ATR']:.2f}" if not np.isnan(last['ATR']) else "—",
                       help="Volatility measure — average move range")
            cc4.metric("MACD", f"{last['MACD']:.2f}" if not np.isnan(last['MACD']) else "—")

            st.subheader("The most important indicators right now")
            st.caption("Analysis uses 14+ indicators; here are the 5 most critical for this selection, in simple words.")

            for title, value, explanation in explain_indicators(last, df):
                st.markdown(
                    f"<div style='padding:12px 16px;margin-bottom:10px;border-radius:10px;"
                    f"background:#e2e5ea;border-left:3px solid #2166ac'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<span style='font-weight:700;font-size:1.02rem;color:#000000'>{title}</span>"
                    f"<span style='font-weight:700;color:#2166ac'>{value}</span></div>"
                    f"<div style='margin-top:6px;color:#333;line-height:1.55'>{explanation}</div></div>",
                    unsafe_allow_html=True,
                )

            st.info(
                f"**Overall assessment: {trend['verdict']}** (score {trend['score']:+d}/100). "
                "The score combines all the indicators above. "
                "It's not a prediction — it's a summary of the current technical signals."
            )

        # ===== TAB 3: Fundamentals =====
        with tab_fund:
            if not info:
                st.warning("No fundamental data (commodity, crypto, ETF or index — not a company). Technical analysis still applies normally.")
            else:
                f1, f2, f3 = st.columns(3)
                f1.metric("Market Cap", fmt_num(info.get("marketCap")))
                f2.metric("P/E (trailing)", f"{info.get('trailingPE'):.1f}" if info.get("trailingPE") else "—")
                f3.metric("EPS", f"{info.get('trailingEps'):.2f}" if info.get("trailingEps") else "—")

                f4, f5, f6 = st.columns(3)
                f4.metric("Revenue (TTM)", fmt_num(info.get("totalRevenue")))
                f5.metric("Free Cash Flow", fmt_num(info.get("freeCashflow")))
                f6.metric("Operating CF", fmt_num(info.get("operatingCashflow")))

                f7, f8, f9 = st.columns(3)
                f7.metric("Dividend yield", f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "—")
                f8.metric("Beta", f"{info.get('beta'):.2f}" if info.get("beta") else "—")
                f9.metric("Profit Margin", f"{info.get('profitMargins')*100:.1f}%" if info.get("profitMargins") else "—")

                target = info.get("targetMeanPrice")
                if target:
                    upside = (target - price) / price * 100
                    st.markdown(
                        f"🎯 **Mean analyst target:** {target:.2f} {currency} "
                        f"({upside:+.1f}% from current) · Recommendation: **{info.get('recommendationKey', '—')}**"
                    )

                if cashflow is not None and not cashflow.empty:
                    with st.expander("📋 Cash Flow table (annual)"):
                        st.dataframe((cashflow / 1e6).round(0).rename_axis("in millions").style.format("{:,.0f}"),
                                     use_container_width=True)

                summary = info.get("longBusinessSummary")
                if summary:
                    with st.expander("ℹ️ Company description"):
                        st.write(summary)

        # (old macro tab)
        # ===== TAB 4: News =====
        with tab_news:
            st.subheader(f"📰 Recent news — {ticker}")
            news = load_news(ticker)
            if not news:
                st.info("No recent news found for this symbol right now. "
                        "Commodities, indices and some non-US stocks often have fewer or no news on Yahoo.")
            else:
                for n in news:
                    # Ημερομηνία σε αναγνώσιμη μορφή
                    when = ""
                    pub = n.get("pub", "")
                    try:
                        if isinstance(pub, (int, float)):
                            when = dt.datetime.fromtimestamp(pub).strftime("%d %b %Y")
                        elif isinstance(pub, str) and pub:
                            when = pub[:10]
                    except Exception:
                        when = ""
                    title = n["title"]
                    provider = n.get("provider", "")
                    link = n.get("link", "")
                    meta = " · ".join([x for x in [provider, when] if x])
                    if link:
                        st.markdown(
                            f"<div style='padding:12px 16px;margin-bottom:10px;border-radius:10px;"
                            f"background:#e2e5ea;border-left:3px solid #2166ac'>"
                            f"<a href='{link}' target='_blank' style='font-weight:700;color:#111;"
                            f"text-decoration:none;font-size:1.02rem'>{title}</a>"
                            f"<div style='margin-top:5px;color:#555;font-size:0.85rem'>{meta} ↗</div></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"**{title}**  \n<span style='color:#666;font-size:0.85rem'>{meta}</span>",
                                    unsafe_allow_html=True)
            st.caption("Source: Yahoo Finance. News opens in a new tab.")

        # ===== TAB 5: Macro & Events =====
        with tab_macro:
            st.subheader("Macroeconomic backdrop")
            macro = load_macro()
            if macro:
                mcols = st.columns(len(macro))
                for col, (k, (val, ch)) in zip(mcols, macro.items()):
                    col.metric(k, f"{val:.2f}", f"{ch:+.1f}% / month")
            st.caption("10Y yield ↑ → pressure on stocks · VIX ↑ → fear/volatility · DXY ↑ → strong dollar")

            st.divider()
            st.subheader("📅 Upcoming high-impact events")
            events = build_upcoming_events()
            today = dt.date.today()
            if events:
                for e in events[:6]:
                    edate = dt.date.fromisoformat(e["date"])
                    days = (edate - today).days
                    when_txt = "today" if days == 0 else ("tomorrow" if days == 1 else f"in {days} days")
                    st.markdown(
                        f"<div style='padding:12px 16px;margin-bottom:10px;border-radius:10px;"
                        f"background:#e2e5ea;border-left:3px solid #b2182b'>"
                        f"<div style='display:flex;justify-content:space-between'>"
                        f"<span style='font-weight:700;color:#111'>{e['flag']} {e['name']}</span>"
                        f"<span style='color:#b2182b;font-weight:700'>{edate.strftime('%d %b %Y')} · {when_txt}</span></div>"
                        f"<div style='margin-top:4px;font-size:0.85rem;color:#444'>"
                        f"Impact: <b>{e['impact']}</b> · Affects: {e['affects']}</div>"
                        f"<div style='margin-top:6px;color:#333;line-height:1.5'>{e['note']}</div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No scheduled meetings in the calendar.")
            st.caption("ℹ️ The 2025-2026 dates are official; 2027 dates are indicative (officially announced ~1 year ahead).")

            st.divider()
            st.subheader("🔁 Recurring events that move the markets")
            st.caption("They don't have a fixed date here, but they're worth watching each month:")
            for name, freq, note in RECURRING_EVENTS_INFO:
                with st.expander(f"{name}  ({freq})"):
                    st.write(note)

            st.info(
                "⚠️ Important: nobody knows in advance whether an event will move the stock up "
                "or down — it depends on whether the outcome is better or worse than the "
                "market's forecast. The explanations above show what to watch, not what will happen."
            )

    except Exception as e:
        st.error(f"Error during analysis: {e}")
        st.caption("Try another symbol or try again shortly (possible Yahoo rate-limit).")
