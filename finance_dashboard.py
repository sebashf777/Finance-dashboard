import matplotlib
matplotlib.use("Agg")
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import datetime, warnings
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Global Finance Dashboard",
                   layout="wide", page_icon="📊",
                   initial_sidebar_state="collapsed")

NY = ZoneInfo("America/New_York")
def ny_now(): return datetime.datetime.now(NY)

C = dict(bg="#000000", panel="#0D0D0D", orange="#FF6600",
         yellow="#FFD700", green="#00FF41", red="#FF3333",
         blue="#00BFFF", gray="#777777", white="#FFFFFF")

GLOBAL_INDICES = {
    "UNITED STATES": {"S&P 500":"^GSPC","NASDAQ":"^IXIC","Dow Jones":"^DJI","Russell 2K":"^RUT","VIX":"^VIX"},
    "EUROPE":        {"FTSE 100":"^FTSE","DAX":"^GDAXI","CAC 40":"^FCHI","Euro Stoxx 50":"^STOXX50E","IBEX 35":"^IBEX","AEX":"^AEX"},
    "ASIA-PACIFIC":  {"Nikkei 225":"^N225","Hang Seng":"^HSI","ASX 200":"^AXJO","Shanghai":"000001.SS","KOSPI":"^KS11","TAIEX":"^TWII"},
    "AMERICAS":      {"Bovespa":"^BVSP","IPC Mexico":"^MXX","TSX Canada":"^GSPTSE","Merval":"^MERV"}
}
FOREX_MAJOR = {"EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"USDJPY=X","USD/CHF":"USDCHF=X","AUD/USD":"AUDUSD=X","USD/CAD":"USDCAD=X","NZD/USD":"NZDUSD=X"}
FOREX_EM    = {"USD/BRL":"USDBRL=X","USD/MXN":"USDMXN=X","USD/CNH":"USDCNH=X","USD/INR":"USDINR=X","USD/ZAR":"USDZAR=X","USD/TRY":"USDTRY=X","USD/KRW":"USDKRW=X","USD/ARS":"USDARS=X"}
COMMODITIES = {
    "ENERGY":      {"WTI Crude":"CL=F","Brent Crude":"BZ=F","Natural Gas":"NG=F","Gasoline":"RB=F","Heating Oil":"HO=F"},
    "METALS":      {"Gold":"GC=F","Silver":"SI=F","Copper":"HG=F","Platinum":"PL=F","Palladium":"PA=F"},
    "AGRICULTURE": {"Corn":"ZC=F","Wheat":"ZW=F","Soybeans":"ZS=F","Coffee":"KC=F","Sugar":"SB=F","Cotton":"CT=F"}
}
CRYPTO  = {"CRYPTO":{"Bitcoin":"BTC-USD","Ethereum":"ETH-USD","BNB":"BNB-USD","Solana":"SOL-USD","XRP":"XRP-USD","Cardano":"ADA-USD","Dogecoin":"DOGE-USD","Avalanche":"AVAX-USD","Polkadot":"DOT-USD","Chainlink":"LINK-USD"}}
SECTORS = {"XLK":"Tech","XLF":"Financials","XLV":"Healthcare","XLE":"Energy","XLI":"Industrials","XLY":"Cons. Disc.","XLP":"Cons. Stap","XLRE":"Real Est.","XLB":"Materials","XLU":"Utilities","XLC":"Comm. Svcs"}
YIELDS  = {"3M":"^IRX","5Y":"^FVX","10Y":"^TNX","30Y":"^TYX"}
PERIOD_LABELS = {"5d":"1 Week","1mo":"1 Month","1y":"1 Year","3mo":"3 Months","6mo":"6 Months","2y":"2 Years","5y":"5 Years"}

st.markdown("""
<style>
  html, body, .stApp { background-color:#000 !important; color:#FFF; }
  section[data-testid="stSidebar"] { display:none; }
  .stTabs [data-baseweb="tab-list"] { background:#000; border-bottom:2px solid #FF6600; gap:3px; }
  .stTabs [data-baseweb="tab"]      { background:#0D0D0D; color:#FF6600; border:1px solid #333; border-radius:4px 4px 0 0; font-family:monospace; font-weight:bold; padding:6px 14px; }
  .stTabs [aria-selected="true"]    { background:#FF6600 !important; color:#000 !important; }
  .stButton>button  { background:#FF6600; color:#000; font-family:monospace; font-weight:bold; border:none; border-radius:4px; }
  .stButton>button:hover { background:#FFD700; }
  .stTextInput input { background:#0D0D0D; color:#FFD700; border:1px solid #FF6600; font-family:monospace; }
  h1,h2,h3 { color:#FF6600 !important; font-family:monospace !important; }
  .block-container { padding-top:0.5rem !important; }
</style>""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────
for key in ["global_period","forex_period","commodities_period","crypto_period","macro_period","chart_period"]:
    if key not in st.session_state:
        st.session_state[key] = "1mo"

# ── HELPERS ──────────────────────────────────────────────────
def _close(raw, tk=None):
    if isinstance(raw.columns, pd.MultiIndex):
        sub = raw["Close"]
        if tk and tk in sub.columns: return sub[tk].dropna()
        return sub.dropna()
    return raw["Close"].dropna()

@st.cache_data(ttl=300)
def batch_quotes_period(tickers_tuple, period="1mo"):
    tickers = list(tickers_tuple)
    result  = {t: dict(price=0.0, chg=0.0, pct=0.0) for t in tickers}
    if not tickers: return result
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        for tk in tickers:
            try:
                s = _close(raw, tk) if len(tickers) > 1 else _close(raw)
                s = s.dropna()
                if len(s) < 2: continue
                price = float(s.iloc[-1]); prev = float(s.iloc[0])
                chg = price - prev; pct = chg / prev * 100 if prev else 0.0
                result[tk] = dict(price=price, chg=chg, pct=pct)
            except: pass
    except: pass
    return result

@st.cache_data(ttl=300)
def batch_quotes(tickers_tuple):
    tickers = list(tickers_tuple)
    result  = {t: dict(price=0.0, chg=0.0, pct=0.0) for t in tickers}
    if not tickers: return result
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
        for tk in tickers:
            try:
                s = _close(raw, tk) if len(tickers) > 1 else _close(raw)
                if len(s) < 2: continue
                price = float(s.iloc[-1]); prev = float(s.iloc[-2])
                chg = price - prev; pct = chg / prev * 100 if prev else 0.0
                result[tk] = dict(price=price, chg=chg, pct=pct)
            except: pass
    except: pass
    return result

@st.cache_data(ttl=300)
def fetch_ohlcv(ticker, period="1mo"):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[df["Close"].notna()].copy()
    except: return pd.DataFrame()

@st.cache_data(ttl=180)
def get_news(ticker):
    try: return yf.Ticker(ticker).news[:20] or []
    except: return []

def rsi(c, w=14):
    d = c.diff(); g = d.clip(lower=0).rolling(w).mean(); l = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - 100/(1 + g/l.replace(0, np.nan))

def macd(c, f=12, s=26, sig=9):
    m  = c.ewm(span=f,min_periods=1).mean() - c.ewm(span=s,min_periods=1).mean()
    sg = m.ewm(span=sig,min_periods=1).mean()
    return m, sg, m - sg

def bbands(c, w=20):
    mid = c.rolling(w,min_periods=1).mean(); std = c.rolling(w,min_periods=1).std()
    return mid+2*std, mid, mid-2*std

def fp(p):
    if p > 10000: return f"{p:,.0f}"
    elif p > 100: return f"{p:,.2f}"
    elif p > 1:   return f"{p:.4f}"
    elif p > 0:   return f"{p:.6f}"
    return "—"

def fc(c):
    a = abs(c)
    if a > 100:   return f"{a:,.2f}"
    elif a > 0.1: return f"{a:.4f}"
    return f"{a:.6f}"

# ── ML FORECASTING ───────────────────────────────────────────
def make_features(df):
    """Feature matrix with direction target (1=up, 0=down)."""
    d = df.copy()
    c = d["Close"]
    d["ret1"]    = c.pct_change(1)
    d["ret3"]    = c.pct_change(3)
    d["ret5"]    = c.pct_change(5)
    d["ret10"]   = c.pct_change(10)
    d["ret20"]   = c.pct_change(20)
    d["ma5"]     = c.rolling(5).mean()
    d["ma10"]    = c.rolling(10).mean()
    d["ma20"]    = c.rolling(20).mean()
    d["ma50"]    = c.rolling(50).mean()
    d["std5"]    = c.rolling(5).std()
    d["std10"]   = c.rolling(10).std()
    d["std20"]   = c.rolling(20).std()
    d["hl_pct"]  = (d["High"] - d["Low"]) / c
    d["oc_pct"]  = (d["Close"] - d["Open"]) / d["Open"]
    d["vol_chg"] = d["Volume"].pct_change(1)
    d["vol_ma"]  = d["Volume"] / d["Volume"].rolling(20).mean()
    d["rsi14"]   = rsi(c, 14)
    d["ma5_10"]  = d["ma5"] / d["ma10"] - 1
    d["ma10_20"] = d["ma10"] / d["ma20"] - 1
    d["ma20_50"] = d["ma20"] / d["ma50"] - 1
    d["bb_pos"]  = (c - c.rolling(20).mean()) / (2 * c.rolling(20).std())
    d["target"]  = (c.shift(-1) > c).astype(int)
    d = d.dropna()
    feat_cols = ["ret1","ret3","ret5","ret10","ret20",
                 "ma5","ma10","ma20","ma50",
                 "std5","std10","std20",
                 "hl_pct","oc_pct","vol_chg","vol_ma",
                 "rsi14","ma5_10","ma10_20","ma20_50","bb_pos"]
    return d[feat_cols], d["target"], d.index

@st.cache_data(ttl=600)
def run_ml_forecast(ticker, forecast_days):
    """
    Three-model ML suite:
      1. LightGBM  — direction classifier + price path via cumulative predicted moves
      2. GARCH(1,1) — volatility forecast
      3. VADER Sentiment — news headline NLP
    """
    results = {}

    # ── 1. LIGHTGBM DIRECTION + PRICE PATH ───────────────────
    try:
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler

        df = fetch_ohlcv(ticker, "2y")
        if df.empty or len(df) < 120:
            results["lgb_error"] = "Insufficient data (need 2y history)"
        else:
            # Features
            d = df.copy()
            c = d["Close"]
            for n in [1,2,3,5,10,20]:
                d[f"ret{n}"] = c.pct_change(n)
            for w in [5,10,20,50]:
                d[f"ma{w}"]  = c.rolling(w).mean()
                d[f"std{w}"] = c.rolling(w).std()
            d["ma5_20"]  = d["ma5"]  / d["ma20"]  - 1
            d["ma10_50"] = d["ma10"] / d["ma50"]  - 1
            d["bb_pos"]  = (c - d["ma20"]) / (2 * d["std20"])
            d["rsi14"]   = rsi(c, 14)
            d["hl_pct"]  = (d["High"] - d["Low"]) / c
            d["oc_pct"]  = (d["Close"] - d["Open"]) / d["Open"]
            d["vol_ratio"]= d["Volume"] / d["Volume"].rolling(20).mean()
            # Target: next-day return (regression, not direction)
            d["fwd_ret"] = c.pct_change(1).shift(-1)
            d = d.dropna()

            feat_cols = [f"ret{n}" for n in [1,2,3,5,10,20]] +                         [f"ma{w}" for w in [5,10,20,50]] +                         [f"std{w}" for w in [5,10,20,50]] +                         ["ma5_20","ma10_50","bb_pos","rsi14",
                         "hl_pct","oc_pct","vol_ratio"]

            X = d[feat_cols]
            y_ret = d["fwd_ret"]
            y_dir = (y_ret > 0).astype(int)

            n     = len(X)
            split = int(n * 0.8)
            X_tr, X_te = X.iloc[:split], X.iloc[split:]
            y_tr_dir, y_te_dir = y_dir.iloc[:split], y_dir.iloc[split:]
            y_tr_ret, y_te_ret = y_ret.iloc[:split], y_ret.iloc[split:]

            # LightGBM regressor (predict next-day return directly)
            reg = lgb.LGBMRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.7,
                min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbose=-1)
            reg.fit(X_tr, y_tr_ret,
                    eval_set=[(X_te, y_te_ret)],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(-1)])

            pred_ret_test = reg.predict(X_te)
            pred_dir_test = (pred_ret_test > 0).astype(int)

            acc = round(accuracy_score(y_te_dir, pred_dir_test) * 100, 1)
            try:
                auc = round(roc_auc_score(y_te_dir, pred_ret_test), 3)
            except:
                auc = 0.0

            # Backtest: reconstruct price from predicted returns
            last_train_price = float(df["Close"].iloc[split])
            bt_prices = [last_train_price]
            for r in pred_ret_test:
                bt_prices.append(bt_prices[-1] * (1 + float(r)))
            bt_prices = bt_prices[1:]

            # Next-day signal
            last_X   = X.iloc[[-1]]
            next_ret = float(reg.predict(last_X)[0])
            next_prob_up = float((next_ret > 0))  # simple direction
            # Use a softer signal based on magnitude
            if next_ret > 0.003:  signal = "BUY"
            elif next_ret < -0.003: signal = "SELL"
            else: signal = "HOLD"

            # Walk-forward price forecast
            last_price  = float(df["Close"].iloc[-1])
            last_date   = df.index[-1]
            temp_df     = df.copy()
            future_prices = [last_price]
            future_rets   = []

            for _ in range(forecast_days):
                try:
                    td = temp_df.copy()
                    tc = td["Close"]
                    for nn in [1,2,3,5,10,20]:
                        td[f"ret{nn}"] = tc.pct_change(nn)
                    for ww in [5,10,20,50]:
                        td[f"ma{ww}"]  = tc.rolling(ww).mean()
                        td[f"std{ww}"] = tc.rolling(ww).std()
                    td["ma5_20"]   = td["ma5"]  / td["ma20"]  - 1
                    td["ma10_50"]  = td["ma10"] / td["ma50"]  - 1
                    td["bb_pos"]   = (tc - td["ma20"]) / (2 * td["std20"])
                    td["rsi14"]    = rsi(tc, 14)
                    td["hl_pct"]   = (td["High"] - td["Low"]) / tc
                    td["oc_pct"]   = (td["Close"] - td["Open"]) / td["Open"]
                    td["vol_ratio"]= td["Volume"] / td["Volume"].rolling(20).mean()
                    td = td.dropna()
                    if td.empty: break
                    xf       = td[feat_cols].iloc[[-1]]
                    pred_r   = float(reg.predict(xf)[0])
                    # Shrink extreme predictions toward zero to avoid drift
                    pred_r   = np.clip(pred_r, -0.03, 0.03)
                    new_p    = future_prices[-1] * (1 + pred_r)
                    future_prices.append(new_p)
                    future_rets.append(pred_r)
                    new_row             = temp_df.iloc[-1:].copy()
                    new_row.index       = [new_row.index[-1] + pd.tseries.offsets.BDay(1)]
                    new_row["Close"]    = new_p
                    new_row["Open"]     = new_p
                    new_row["High"]     = new_p * (1 + abs(pred_r) * 0.5)
                    new_row["Low"]      = new_p * (1 - abs(pred_r) * 0.5)
                    new_row["Volume"]   = float(temp_df["Volume"].rolling(20).mean().iloc[-1])
                    temp_df = pd.concat([temp_df, new_row])
                except Exception as fe:
                    break

            fut_dates = pd.bdate_range(
                start=last_date + pd.tseries.offsets.BDay(1),
                periods=len(future_prices)-1)
            all_dates  = [last_date]   + list(fut_dates)
            all_prices = future_prices[:len(all_dates)]

            future_df = pd.DataFrame({"price": all_prices}, index=all_dates)

            # Confidence band: ±1 rolling std of predicted returns
            if future_rets:
                avg_abs = float(np.mean(np.abs(future_rets)))
                band    = [last_price * avg_abs * (i+1)**0.5 for i in range(len(fut_dates))]
                future_df.loc[future_df.index[1:], "upper"] = [p + b for p, b in zip(all_prices[1:], band)]
                future_df.loc[future_df.index[1:], "lower"] = [p - b for p, b in zip(all_prices[1:], band)]
            else:
                future_df["upper"] = future_df["price"]
                future_df["lower"] = future_df["price"]

            # Feature importance
            fi   = pd.Series(reg.feature_importances_, index=feat_cols)
            top5 = fi.nlargest(5).to_dict()

            # Backtest dataframe
            bt_idx = d.index[split:]
            bt_df  = pd.DataFrame({
                "actual":    [float(v) for v in df["Close"].loc[bt_idx].values],
                "predicted": bt_prices[:len(bt_idx)],
            }, index=bt_idx)

            results["lgb"] = {
                "acc":        acc,
                "auc":        auc,
                "next_ret":   round(next_ret * 100, 3),
                "next_prob":  round(float(np.mean(pred_ret_test > 0)) * 100, 1),
                "signal":     signal,
                "top5":       top5,
                "future_df":  future_df,
                "bt_df":      bt_df,
            }
    except Exception as e:
        results["lgb_error"] = str(e)

    # ── 2. GARCH(1,1) VOLATILITY FORECAST ────────────────────
    try:
        from arch import arch_model

        df2 = fetch_ohlcv(ticker, "2y")
        if df2.empty or len(df2) < 60:
            results["garch_error"] = "Insufficient data"
        else:
            rets = df2["Close"].pct_change().dropna() * 100
            garch = arch_model(rets, vol="Garch", p=1, q=1, dist="normal")
            res   = garch.fit(disp="off", show_warning=False)
            fc    = res.forecast(horizon=forecast_days, reindex=False)
            vol_f = np.sqrt(fc.variance.values[-1])
            ann_v = vol_f * np.sqrt(252)
            cur_v = float(rets.rolling(20).std().iloc[-1]) * np.sqrt(252)

            last_date2 = df2.index[-1]
            fut_dates2 = pd.bdate_range(
                start=last_date2 + pd.tseries.offsets.BDay(1),
                periods=forecast_days)
            vol_df = pd.DataFrame({"daily_vol": vol_f, "ann_vol": ann_v}, index=fut_dates2)

            results["garch"] = {
                "vol_df":      vol_df,
                "current_vol": round(cur_v, 2),
                "peak_vol":    round(float(ann_v.max()), 2),
                "avg_vol":     round(float(ann_v.mean()), 2),
                "params": {
                    "alpha": round(float(res.params["alpha[1]"]), 4),
                    "beta":  round(float(res.params["beta[1]"]),  4),
                }
            }
    except Exception as e:
        results["garch_error"] = str(e)

    # ── 3. VADER SENTIMENT ────────────────────────────────────
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        news     = get_news(ticker)
        analyzer = SentimentIntensityAnalyzer()
        scores, headlines = [], []
        for item in news[:15]:
            try:
                ct    = item.get("content", {})
                title = ct.get("title", item.get("title", ""))
                if not title: continue
                score = analyzer.polarity_scores(title)["compound"]
                scores.append(score)
                headlines.append((title, round(score, 3)))
            except: pass

        if scores:
            avg   = round(float(np.mean(scores)), 3)
            label = "BULLISH" if avg > 0.05 else ("BEARISH" if avg < -0.05 else "NEUTRAL")
            results["sentiment"] = {
                "avg": avg, "label": label,
                "headlines": headlines[:10],
                "scores": scores,
            }
        else:
            results["sentiment_error"] = "No headlines available"
    except Exception as e:
        results["sentiment_error"] = str(e)

    return results


def build_forecast_chart(ticker, forecast_days, ml_results, current_price):
    """Four-panel chart: price forecast, backtest, volatility, sentiment."""
    has_lgb   = "lgb"   in ml_results
    has_garch = "garch" in ml_results
    has_sent  = "sentiment" in ml_results

    panel_count = 1 + int(has_lgb) + int(has_garch) + int(has_sent)
    heights = []
    titles  = []
    if has_lgb:
        heights += [0.42, 0.22]
        titles  += [f"LightGBM — {forecast_days}d Price Forecast", "LightGBM — Backtest (last 20%)"]
    if has_garch:
        heights.append(0.18)
        titles.append("GARCH(1,1) — Annualised Volatility Forecast (%)")
    if has_sent:
        heights.append(0.18)
        titles.append("News Sentiment (VADER)")

    # Normalise heights
    total = sum(heights)
    heights = [h/total for h in heights]

    n_rows = len(heights)
    if n_rows == 0: return None

    fig = make_subplots(rows=n_rows, cols=1,
                        subplot_titles=titles,
                        vertical_spacing=0.06,
                        row_heights=heights)
    row = 1

    # ── PRICE FORECAST ───────────────────────────────────────
    if has_lgb:
        lgb_r = ml_results["lgb"]
        fdf   = lgb_r["future_df"]

        # Historical last 60 days for context
        try:
            hist = fetch_ohlcv(ticker, "3mo")
            if not hist.empty:
                fig.add_trace(go.Scatter(
                    x=hist.index, y=hist["Close"],
                    line=dict(color="#888888", width=1.5),
                    name="Historical", showlegend=True), row=row, col=1)
        except: pass

        # Confidence band
        if "upper" in fdf.columns and "lower" in fdf.columns:
            fdf_fwd = fdf.iloc[1:]
            fig.add_trace(go.Scatter(
                x=list(fdf_fwd.index) + list(fdf_fwd.index[::-1]),
                y=list(fdf_fwd["upper"]) + list(fdf_fwd["lower"][::-1]),
                fill="toself", fillcolor="rgba(255,102,0,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Confidence Band", showlegend=False), row=row, col=1)

        # Forecast line — all points including anchor
        fig.add_trace(go.Scatter(
            x=fdf.index, y=fdf["price"],
            line=dict(color="#FF6600", width=2.5),
            mode="lines+markers",
            marker=dict(size=5, color="#FF6600"),
            name="Forecast", showlegend=True), row=row, col=1)

        # Annotate each forecast day price
        for i, (dt, pr) in enumerate(zip(fdf.index[1:], fdf["price"].iloc[1:])):
            chg = (pr - current_price) / current_price * 100 if current_price else 0
            col_ann = "#00FF41" if chg >= 0 else "#FF3333"
            fig.add_annotation(
                x=dt, y=float(pr),
                text=f"<b>{fp(float(pr))}</b><br><span style='font-size:9px'>{chg:+.2f}%</span>",
                showarrow=True, arrowhead=2, arrowcolor=col_ann,
                arrowsize=0.8, arrowwidth=1,
                ax=0, ay=-35,
                font=dict(size=9, color=col_ann, family="Courier New"),
                bgcolor="rgba(0,0,0,0.7)", bordercolor=col_ann, borderwidth=0.5,
                row=row, col=1)

        # Vertical line at today
        fig.add_vline(x=fdf.index[0], line=dict(color="#FFD700", dash="dash", width=1),
                      row=row, col=1)
        row += 1

        # ── BACKTEST ─────────────────────────────────────────
        bt = lgb_r["bt_df"]
        fig.add_trace(go.Scatter(
            x=bt.index, y=bt["actual"],
            line=dict(color="#00FF41", width=1.5),
            name="Actual", showlegend=True), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=bt.index, y=bt["predicted"],
            line=dict(color="#FF6600", width=1.5, dash="dot"),
            name="LightGBM Backtest", showlegend=True), row=row, col=1)

        # Correlation annotation
        try:
            corr = round(float(np.corrcoef(bt["actual"], bt["predicted"])[0,1]), 3)
            fig.add_annotation(
                xref="paper", yref="paper", x=0.99, y=0,
                text=f"Backtest corr: {corr}  |  Acc: {lgb_r['acc']}%  |  AUC: {lgb_r['auc']}",
                showarrow=False,
                font=dict(size=9, color="#FFD700", family="Courier New"),
                xanchor="right", row=row, col=1)
        except: pass
        row += 1

    # ── GARCH VOLATILITY ─────────────────────────────────────
    if has_garch:
        g   = ml_results["garch"]
        vdf = g["vol_df"]
        fig.add_trace(go.Scatter(
            x=vdf.index, y=vdf["ann_vol"],
            line=dict(color="#FF6600", width=2),
            fill="tozeroy", fillcolor="rgba(255,102,0,0.08)",
            name="Ann. Vol %", showlegend=False), row=row, col=1)
        fig.add_hline(y=g["current_vol"],
                      line=dict(color="#FFD700", dash="dash", width=1), row=row, col=1)
        fig.add_annotation(
            xref="paper", yref="paper", x=0.99, y=0,
            text=f"Current: {g['current_vol']}%  alpha={g['params']['alpha']}  beta={g['params']['beta']}",
            showarrow=False,
            font=dict(size=9, color="#FFD700", family="Courier New"),
            xanchor="right", row=row, col=1)
        row += 1

    # ── SENTIMENT ────────────────────────────────────────────
    if has_sent:
        s      = ml_results["sentiment"]
        hlines = s["headlines"]
        sc     = s["scores"][:len(hlines)]
        colors = ["#00FF41" if v > 0.05 else ("#FF3333" if v < -0.05 else "#FFD700") for v in sc]
        labels = [h[:50]+"…" if len(h)>50 else h for h,_ in hlines]
        fig.add_trace(go.Bar(
            x=sc, y=labels,
            orientation="h",
            marker_color=colors,
            name="Sentiment", showlegend=False), row=row, col=1)
        fig.add_vline(x=0, line=dict(color="#555", width=1), row=row, col=1)

    sig_color = "#00FF41" if ml_results.get("lgb",{}).get("signal")=="BUY" else                 "#FF3333" if ml_results.get("lgb",{}).get("signal")=="SELL" else "#FFD700"
    sig_txt = ml_results.get("lgb",{}).get("signal","—")
    next_r  = ml_results.get("lgb",{}).get("next_ret","—")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000", plot_bgcolor="#0D0D0D",
        title=dict(
            text=(f"<b style='color:#FF6600'>{ticker.upper()}</b>"
                  f"  <span style='color:{sig_color};font-size:13px'>{sig_txt}</span>"
                  f"  <span style='color:#555;font-size:11px'>"
                  f"Next-day return: {next_r}%  |  {forecast_days}d horizon</span>"),
            font=dict(family="Courier New", size=14), x=0),
        height=220 * n_rows + 60,
        legend=dict(orientation="h", x=0, y=1.02,
                    bgcolor="rgba(0,0,0,0.5)",
                    font=dict(family="Courier New", size=9)),
        font=dict(family="Courier New", color=C["gray"]),
        margin=dict(l=50, r=20, t=60, b=20))
    fig.update_xaxes(gridcolor="#1a1a1a")
    fig.update_yaxes(gridcolor="#1a1a1a")
    return fig

def period_buttons(tab_key):
    p = st.session_state[tab_key]
    cols = st.columns([1,1,1,1,2,6])
    with cols[0]:
        lbl = "[ 1 Week ]" if p == "5d" else "1 Week"
        if st.button(lbl, key=f"{tab_key}_5d"):
            st.session_state[tab_key] = "5d"; st.rerun()
    with cols[1]:
        lbl = "[ 1 Month ]" if p == "1mo" else "1 Month"
        if st.button(lbl, key=f"{tab_key}_1mo"):
            st.session_state[tab_key] = "1mo"; st.rerun()
    with cols[2]:
        lbl = "[ 1 Year ]" if p == "1y" else "1 Year"
        if st.button(lbl, key=f"{tab_key}_1y"):
            st.session_state[tab_key] = "1y"; st.rerun()
    with cols[3]:
        more = st.selectbox("More", ["More...","3mo","6mo","2y","5y"],
                            index=0, label_visibility="collapsed", key=f"{tab_key}_more")
        if more != "More...":
            st.session_state[tab_key] = more; st.rerun()
    st.markdown(
        f"<div style='font-family:monospace;font-size:11px;color:#555;margin-bottom:6px'>"
        f"Change vs: <span style='color:#FFD700'>{PERIOD_LABELS.get(p,p)} ago</span></div>",
        unsafe_allow_html=True)
    return p

def html_table(title, sections, icon, period=None):
    tks    = [tk for sec in sections.values() for tk in sec.values()]
    quotes = batch_quotes(tuple(tks))
    now    = ny_now().strftime("%I:%M %p ET")
    rows   = ""
    for sec_name, items in sections.items():
        rows += (f"<tr><td colspan='5' style='background:#111122;color:#FF6600;"
                 f"font-weight:bold;padding:7px 14px;font-size:11px;letter-spacing:2px;"
                 f"border-top:1px solid #222'>{sec_name}</td></tr>")
        for name, tk in items.items():
            q    = quotes.get(tk, dict(price=0, chg=0, pct=0))
            col  = "#00FF41" if q["chg"] >= 0 else "#FF3333"
            sign = "▲" if q["chg"] >= 0 else "▼"
            rows += (f"<tr style='border-bottom:1px solid #111'>"
                     f"<td style='color:#FFD700;font-weight:bold;padding:7px 14px'>{name}</td>"
                     f"<td style='color:white;padding:7px 14px'>{fp(q['price'])}</td>"
                     f"<td style='color:{col};padding:7px 14px'>{sign} {fc(q['chg'])}</td>"
                     f"<td style='color:{col};padding:7px 14px'>{q['pct']:+.2f}%</td>"
                     f"<td style='color:#444;padding:7px 14px;font-size:10px'>{tk}</td></tr>")
    hdr = "".join(f"<th style='color:#FF6600;text-align:left;padding:6px 14px'>{h}</th>"
                  for h in ["NAME","PRICE","CHANGE","CHG%","TICKER"])
    return (f"<div style='background:#000;font-family:Courier New,monospace;padding:14px;"
            f"border:1px solid #FF6600;border-radius:4px;margin-top:8px'>"
            f"<div style='color:#FF6600;font-size:13px;font-weight:bold;border-bottom:1px solid #333;"
            f"padding-bottom:8px;margin-bottom:10px;display:flex;justify-content:space-between'>"
            f"<span>{icon} {title}</span>"
            f"<span style='color:#555;font-size:11px'>{now}</span></div>"
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<thead><tr style='border-bottom:2px solid #FF6600'>{hdr}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")

def build_global(period="1mo"):
    tks    = tuple(tk for reg in GLOBAL_INDICES.values() for tk in reg.values())
    quotes = batch_quotes(tks)
    fig    = plt.figure(figsize=(16,9), facecolor=C["bg"])
    gs     = gridspec.GridSpec(2,2, figure=fig, hspace=0.52, wspace=0.3)
    rcols  = [C["blue"],"#8888ff",C["orange"],"#ffaa00"]
    for pi,((rn,ri),rc) in enumerate(zip(GLOBAL_INDICES.items(),rcols)):
        ax = fig.add_subplot(gs[pi//2, pi%2])
        ax.set_facecolor(C["bg"]); ax.axis("off")
        ax.set_title(f"  {rn}", color=rc, fontsize=11, fontweight="bold",
                     fontfamily="monospace", loc="left", pad=8)
        items = list(ri.items()); n = len(items)
        cn = min(3,n); rn2 = (n+cn-1)//cn
        bw = 1/cn-0.03; bh = 0.78/rn2-0.04
        for i,(name,tk) in enumerate(items):
            q    = quotes.get(tk, dict(price=0,chg=0,pct=0))
            r    = i//cn; ci = i%cn
            bx   = ci/cn+0.01; by = 0.92-(r+1)*(bh+0.04)
            col  = C["green"] if q["chg"] >= 0 else C["red"]
            ax.add_patch(FancyBboxPatch((bx,by),bw,bh,boxstyle="round,pad=0.01",
                facecolor="#0D0D0D",edgecolor=col,linewidth=1.1,transform=ax.transAxes))
            cx = bx+bw/2; cy = by+bh/2
            sign = "▲" if q["chg"] >= 0 else "▼"
            ax.text(cx,cy+bh*.26,name,ha="center",color=rc,fontsize=7,
                    fontweight="bold",fontfamily="monospace",transform=ax.transAxes)
            ax.text(cx,cy,fp(q["price"]),ha="center",color=C["yellow"],fontsize=9.5,
                    fontweight="bold",fontfamily="monospace",transform=ax.transAxes)
            ax.text(cx,cy-bh*.28,f"{sign} {q['pct']:+.2f}%",ha="center",color=col,
                    fontsize=7,fontfamily="monospace",transform=ax.transAxes)
    plt.suptitle("GLOBAL EQUITY MARKETS",
                 color=C["orange"],fontsize=14,fontweight="bold",fontfamily="monospace",y=0.99)
    plt.tight_layout(pad=1.5,rect=[0,0,1,0.97])
    return fig

def build_macro(period="1mo"):
    plabel = PERIOD_LABELS.get(period, period)
    fig    = plt.figure(figsize=(16,9), facecolor=C["bg"])
    gs     = gridspec.GridSpec(2,3, figure=fig, hspace=0.5, wspace=0.35)
    ax_vix = fig.add_subplot(gs[0,:2]); ax_vix.set_facecolor("#0D0D0D")
    try:
        vd = yf.download("^VIX",period="1y",auto_adjust=True,progress=False)
        cv = _close(vd).dropna(); lv = float(cv.iloc[-1])
        vc = C["green"] if lv < 20 else C["red"]
        ax_vix.plot(vd.index[-len(cv):],cv,color=C["orange"],linewidth=1.4)
        ax_vix.fill_between(vd.index[-len(cv):],cv,float(cv.min()),alpha=0.15,color=C["orange"])
        ax_vix.axhline(20,color=C["red"],  lw=0.8,ls="--",alpha=0.6,label="Fear (>20)")
        ax_vix.axhline(12,color=C["green"],lw=0.8,ls="--",alpha=0.6,label="Greed (<12)")
        ax_vix.fill_between(vd.index[-len(cv):],0,12, alpha=0.06,color=C["green"])
        ax_vix.fill_between(vd.index[-len(cv):],20,100,alpha=0.06,color=C["red"])
        ax_vix.text(0.99,0.93,f"VIX: {lv:.2f}",transform=ax_vix.transAxes,
            ha="right",va="top",color=vc,fontsize=13,fontweight="bold",fontfamily="monospace")
    except: pass
    ax_vix.set_title("  VIX — VOLATILITY / FEAR INDEX (1Y)",color=C["orange"],
                     fontsize=11,fontweight="bold",fontfamily="monospace",loc="left")
    ax_vix.tick_params(colors=C["gray"],labelsize=7)
    ax_vix.legend(facecolor="#111",edgecolor="#333",labelcolor=C["white"],fontsize=7)
    ax_vix.grid(alpha=0.1,color=C["gray"])
    for sp in ax_vix.spines.values(): sp.set_edgecolor("#333")
    ax_yc = fig.add_subplot(gs[0,2]); ax_yc.set_facecolor("#0D0D0D")
    tks_y = list(YIELDS.values()); lbs_y = list(YIELDS.keys()); mx = [0.25,5,10,30]
    try:
        ry = yf.download(tks_y,period="1y",auto_adjust=True,progress=False)
        def gyc(idx):
            v = []
            for tk in tks_y:
                try: v.append(float(_close(ry,tk).iloc[idx]))
                except: v.append(None)
            return v
        def ycp(vals,lbl,col,sty="-"):
            pts = [(x,v) for x,v in zip(mx,vals) if v and v>0]
            if not pts: return
            xs,ys = zip(*pts)
            ax_yc.plot(xs,ys,f"o{sty}",color=col,linewidth=2,markersize=6,label=lbl,alpha=0.9)
            if sty == "-":
                ax_yc.fill_between(xs,ys,alpha=0.1,color=col)
                for x,y,lb in zip(xs,ys,lbs_y):
                    ax_yc.annotate(f"{lb}\n{y:.2f}%",xy=(x,y),xytext=(0,12),
                        textcoords="offset points",ha="center",fontsize=7,
                        color=C["yellow"],fontfamily="monospace")
        ycp(gyc(-1),"Today", C["orange"],"-")
        ycp(gyc(0), "1Y Ago",C["gray"],  "--")
    except: pass
    ax_yc.set_xticks(mx); ax_yc.set_xticklabels(lbs_y,color=C["gray"],fontsize=7)
    ax_yc.set_title("  US YIELD CURVE",color=C["orange"],fontsize=10,
                    fontweight="bold",fontfamily="monospace",loc="left")
    ax_yc.tick_params(colors=C["gray"],labelsize=7)
    ax_yc.legend(facecolor="#111",edgecolor="#333",labelcolor=C["white"],fontsize=7)
    ax_yc.grid(alpha=0.1,color=C["gray"])
    for sp in ax_yc.spines.values(): sp.set_edgecolor("#333")
    ax_sec = fig.add_subplot(gs[1,:2]); ax_sec.set_facecolor("#0D0D0D")
    sq    = batch_quotes_period(tuple(SECTORS.keys()), period)
    slabs = list(SECTORS.values())
    svals = [sq.get(tk,dict(pct=0))["pct"] for tk in SECTORS]
    scols = [C["green"] if v>=0 else C["red"] for v in svals]
    sbars = ax_sec.barh(slabs,svals,color=scols,height=0.6,edgecolor="#222")
    ax_sec.axvline(0,color=C["gray"],linewidth=0.8,linestyle="--")
    ax_sec.set_title(f"  S&P 500 SECTOR ROTATION  —  {plabel}",color=C["orange"],fontsize=10,
                     fontweight="bold",fontfamily="monospace",loc="left")
    ax_sec.tick_params(colors=C["white"],labelsize=8)
    for sp in ax_sec.spines.values(): sp.set_edgecolor("#333")
    for bar,val in zip(sbars,svals):
        ax_sec.text(val+(0.02 if val>=0 else -0.02),bar.get_y()+bar.get_height()/2,
            f"{val:+.2f}%",va="center",ha="left" if val>=0 else "right",
            color=C["white"],fontsize=7.5,fontfamily="monospace")
    ax_sn = fig.add_subplot(gs[1,2]); ax_sn.set_facecolor("#0D0D0D"); ax_sn.axis("off")
    ax_sn.set_title(f"  KEY ASSETS  —  {plabel}",color=C["orange"],fontsize=10,
                    fontweight="bold",fontfamily="monospace",loc="left")
    snap = {"Gold":"GC=F","WTI Oil":"CL=F","BTC":"BTC-USD","EUR/USD":"EURUSD=X",
            "10Y Yield":"^TNX","Silver":"SI=F","Copper":"HG=F","Nat. Gas":"NG=F"}
    snq  = batch_quotes_period(tuple(snap.values()), period)
    yp   = 0.88
    for name,tk in snap.items():
        q    = snq.get(tk, dict(price=0,chg=0,pct=0))
        col  = C["green"] if q["chg"] >= 0 else C["red"]
        sign = "▲" if q["chg"] >= 0 else "▼"
        ax_sn.text(0.02,yp,name,transform=ax_sn.transAxes,color=C["orange"],
                   fontsize=8.5,fontfamily="monospace",fontweight="bold")
        ax_sn.text(0.50,yp,fp(q["price"]),transform=ax_sn.transAxes,
                   color=C["yellow"],fontsize=8.5,fontfamily="monospace")
        ax_sn.text(0.80,yp,f"{sign}{abs(q['pct']):.2f}%",transform=ax_sn.transAxes,
                   color=col,fontsize=8.5,fontfamily="monospace")
        yp -= 0.11
    plt.suptitle(f"MACRO DASHBOARD  —  {plabel} Change",color=C["orange"],fontsize=14,
                 fontweight="bold",fontfamily="monospace",y=0.99)
    plt.tight_layout(pad=1.5,rect=[0,0,1,0.97])
    return fig

# ── HEADER ───────────────────────────────────────────────────
now_str = ny_now().strftime("%A, %B %d %Y  |  %I:%M %p ET")
st.markdown(
    f"<div style='background:#000;font-family:Courier New,monospace;padding:12px 20px;"
    f"border-bottom:3px solid #FF6600;display:flex;justify-content:space-between;"
    f"align-items:center;margin-bottom:12px'>"
    f"<span style='color:#FF6600;font-size:20px;font-weight:bold;letter-spacing:3px'>"
    f"GLOBAL FINANCE DASHBOARD</span>"
    f"<span style='color:#FFD700;font-size:11px'>{now_str}</span>"
    f"<span style='color:#00FF41;font-size:12px'>LIVE | Yahoo Finance</span></div>",
    unsafe_allow_html=True)

t1,t2,t3,t4,t5,t6,t7 = st.tabs(["GLOBAL","FOREX","COMMODITIES","CRYPTO","CHART","MACRO","NEWS"])

with t1:
    if st.button("🔄 Refresh", key="rg"): st.cache_data.clear()
    with st.spinner("Loading global markets..."):
        fig = build_global("1mo"); st.pyplot(fig,use_container_width=True); plt.close(fig)

with t2:
    if st.button("🔄 Refresh", key="rf"): st.cache_data.clear()
    with st.spinner("Fetching forex..."):
        st.markdown(html_table("FOREX - CURRENCY MARKETS",
            {"MAJOR PAIRS":FOREX_MAJOR,"EMERGING MARKETS":FOREX_EM},"FX","1mo"),
            unsafe_allow_html=True)

with t3:
    if st.button("🔄 Refresh", key="rc"): st.cache_data.clear()
    with st.spinner("Fetching commodities..."):
        st.markdown(html_table("GLOBAL COMMODITIES",COMMODITIES,"C","1mo"),
            unsafe_allow_html=True)

with t4:
    if st.button("🔄 Refresh", key="rk"): st.cache_data.clear()
    with st.spinner("Fetching crypto..."):
        st.markdown(html_table("CRYPTOCURRENCY MARKETS",CRYPTO,"B","1mo"),
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — CHART + ML FORECAST
# ══════════════════════════════════════════════════════════════
with t5:
    col_tk, col_ref, _ = st.columns([2,1,9])
    with col_tk:
        ticker = st.text_input("Ticker", value="SPY", key="ct").upper()
    with col_ref:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("Refresh", key="chart_refresh"): st.cache_data.clear()

    # Period — only More dropdown (no 1W/1M/1Y buttons per request)
    col_per, col_fd, _ = st.columns([2,2,8])
    with col_per:
        period_choice = st.selectbox(
            "Chart Period",
            options=["1mo","5d","3mo","6mo","1y","2y","5y"],
            index=0,
            format_func=lambda x: PERIOD_LABELS.get(x,x),
            key="chart_period_sel")
        st.session_state["chart_period"] = period_choice
    with col_fd:
        forecast_days = st.selectbox(
            "Forecast Horizon",
            options=[5,10,15,20,30,45,60],
            index=2,
            format_func=lambda x: f"{x} trading days",
            key="forecast_days")

    cp5 = st.session_state["chart_period"]

    if ticker:
        # ── Main candlestick chart ──────────────────────────
        with st.spinner(f"Loading {ticker} chart..."):
            try:
                st.plotly_chart(build_chart(ticker, cp5), use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

        # ── ML Forecast section ─────────────────────────────
        st.markdown(
            "<div style='border-top:1px solid #FF6600;margin:16px 0 12px 0;"
            "padding-top:12px;font-family:monospace;color:#FF6600;font-size:13px;"
            "font-weight:bold;letter-spacing:2px'>AI / ML ANALYSIS</div>",
            unsafe_allow_html=True)

        col_m1, col_m2, col_m3, _ = st.columns([2,2,2,6])
        with col_m1:
            st.markdown("<div style='background:#0D0D0D;border:1px solid #00FF41;border-radius:4px;"
                        "padding:8px 12px;font-family:monospace'>"
                        "<div style='color:#00FF41;font-size:10px;font-weight:bold'>MODEL 1</div>"
                        "<div style='color:#FFD700;font-size:13px'>LightGBM</div>"
                        "<div style='color:#555;font-size:10px'>Direction classifier</div>"
                        "</div>", unsafe_allow_html=True)
        with col_m2:
            st.markdown("<div style='background:#0D0D0D;border:1px solid #FF6600;border-radius:4px;"
                        "padding:8px 12px;font-family:monospace'>"
                        "<div style='color:#FF6600;font-size:10px;font-weight:bold'>MODEL 2</div>"
                        "<div style='color:#FFD700;font-size:13px'>GARCH(1,1)</div>"
                        "<div style='color:#555;font-size:10px'>Volatility forecast</div>"
                        "</div>", unsafe_allow_html=True)
        with col_m3:
            st.markdown("<div style='background:#0D0D0D;border:1px solid #00BFFF;border-radius:4px;"
                        "padding:8px 12px;font-family:monospace'>"
                        "<div style='color:#00BFFF;font-size:10px;font-weight:bold'>MODEL 3</div>"
                        "<div style='color:#FFD700;font-size:13px'>VADER Sentiment</div>"
                        "<div style='color:#555;font-size:10px'>News headline NLP</div>"
                        "</div>", unsafe_allow_html=True)

        with st.spinner(f"Running AI/ML models for {ticker}... (~15 seconds)"):
            try:
                current_price = 0.0
                try:
                    tmp = fetch_ohlcv(ticker, "5d")
                    if not tmp.empty: current_price = float(tmp["Close"].iloc[-1])
                except: pass

                ml_results = run_ml_forecast(ticker, forecast_days)
                fig_ml = build_forecast_chart(ticker, forecast_days, ml_results, current_price)
                if fig_ml:
                    st.plotly_chart(fig_ml, use_container_width=True)

                # ── Metric cards ──────────────────────────────
                c1, c2, c3, _ = st.columns([2,2,2,6])
                with c1:
                    if "lgb" in ml_results:
                        lgb_r = ml_results["lgb"]
                        sig_col = "#00FF41" if lgb_r["signal"]=="BUY" else ("#FF3333" if lgb_r["signal"]=="SELL" else "#FFD700")
                        st.markdown(
                            f"<div style='background:#0D0D0D;border:1px solid #00FF41;"
                            f"border-radius:4px;padding:10px 12px;font-family:monospace'>"
                            f"<div style='color:#555;font-size:10px'>LIGHTGBM SIGNAL</div>"
                            f"<div style='color:{sig_col};font-size:22px;font-weight:bold'>"
                            f"{lgb_r['signal']}</div>"
                            f"<div style='color:#555;font-size:10px'>"
                            f"Up prob: {lgb_r['next_prob']}%  |  Acc: {lgb_r['acc']}%  |  AUC: {lgb_r['auc']}</div>"
                            f"</div>", unsafe_allow_html=True)
                    elif "lgb_error" in ml_results:
                        st.error(f"LightGBM: {ml_results['lgb_error']}")

                with c2:
                    if "garch" in ml_results:
                        g = ml_results["garch"]
                        vol_col = "#FF3333" if g["avg_vol"] > 30 else ("#FFD700" if g["avg_vol"] > 20 else "#00FF41")
                        st.markdown(
                            f"<div style='background:#0D0D0D;border:1px solid #FF6600;"
                            f"border-radius:4px;padding:10px 12px;font-family:monospace'>"
                            f"<div style='color:#555;font-size:10px'>GARCH — AVG FORECAST VOL</div>"
                            f"<div style='color:{vol_col};font-size:22px;font-weight:bold'>"
                            f"{g['avg_vol']}%</div>"
                            f"<div style='color:#555;font-size:10px'>"
                            f"Current: {g['current_vol']}%  |  Peak: {g['peak_vol']}%</div>"
                            f"</div>", unsafe_allow_html=True)
                    elif "garch_error" in ml_results:
                        st.error(f"GARCH: {ml_results['garch_error']}")

                with c3:
                    if "sentiment" in ml_results:
                        s = ml_results["sentiment"]
                        sent_col = "#00FF41" if s["label"]=="BULLISH" else ("#FF3333" if s["label"]=="BEARISH" else "#FFD700")
                        st.markdown(
                            f"<div style='background:#0D0D0D;border:1px solid #00BFFF;"
                            f"border-radius:4px;padding:10px 12px;font-family:monospace'>"
                            f"<div style='color:#555;font-size:10px'>NEWS SENTIMENT</div>"
                            f"<div style='color:{sent_col};font-size:22px;font-weight:bold'>"
                            f"{s['label']}</div>"
                            f"<div style='color:#555;font-size:10px'>"
                            f"Score: {s['avg']}  |  Headlines: {len(s['headlines'])}</div>"
                            f"</div>", unsafe_allow_html=True)
                    elif "sentiment_error" in ml_results:
                        st.warning(f"Sentiment: {ml_results['sentiment_error']}")

                # ── Top features from LightGBM ────────────────
                if "lgb" in ml_results and ml_results["lgb"].get("top5"):
                    st.markdown(
                        "<div style='font-family:monospace;font-size:11px;color:#555;"
                        "margin-top:12px;margin-bottom:4px'>TOP PREDICTIVE FEATURES (LightGBM)</div>",
                        unsafe_allow_html=True)
                    top5 = ml_results["lgb"]["top5"]
                    max_v = max(top5.values()) if top5 else 1
                    cols5 = st.columns(len(top5))
                    for i, (feat, val) in enumerate(top5.items()):
                        pct = round(val / max_v * 100)
                        with cols5[i]:
                            st.markdown(
                                f"<div style='background:#0D0D0D;border:1px solid #333;"
                                f"border-radius:4px;padding:6px 10px;font-family:monospace;"
                                f"text-align:center'>"
                                f"<div style='color:#FF6600;font-size:10px'>{feat}</div>"
                                f"<div style='color:#FFD700;font-size:13px;font-weight:bold'>{pct}%</div>"
                                f"</div>", unsafe_allow_html=True)

                st.markdown(
                    "<div style='font-family:monospace;font-size:10px;color:#333;margin-top:12px'>"
                    "Not financial advice. Models trained on historical data. Past performance "
                    "does not guarantee future results."
                    "</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"ML error: {e}")

with t6:
    rc6, _ = st.columns([1,11])
    with rc6:
        if st.button("Refresh", key="rm"): st.cache_data.clear()
    mp = period_buttons("macro_period")
    with st.spinner("Building macro dashboard..."):
        fig = build_macro(mp); st.pyplot(fig,use_container_width=True); plt.close(fig)

with t7:
    ntk = st.text_input("Ticker", value="SPY", key="nt").upper()
    if ntk:
        q   = batch_quotes((ntk,)).get(ntk, dict(price=0,chg=0,pct=0))
        col = "#00FF41" if q["chg"] >= 0 else "#FF3333"
        sgn = "▲" if q["chg"] >= 0 else "▼"
        st.markdown(
            f"<div style='font-family:monospace;padding:8px 0'>"
            f"<span style='color:#FFD700;font-size:22px;font-weight:bold'>{ntk}</span>"
            f"&nbsp;&nbsp;<span style='color:{col};font-size:18px'>"
            f"{fp(q['price'])} {sgn} {fc(q['chg'])} ({q['pct']:+.2f}%)</span></div>",
            unsafe_allow_html=True)
        with st.spinner(f"Loading news for {ntk}..."):
            news = get_news(ntk)
            rows = ""
            for item in news:
                try:
                    ct    = item.get("content", {})
                    title = ct.get("title", item.get("title","No title"))
                    pub   = ct.get("pubDate","")[:10]
                    prov  = ct.get("provider",{})
                    src   = prov.get("displayName","") if isinstance(prov,dict) else ""
                    rows += (f"<tr style='border-bottom:1px solid #1a1a1a'>"
                             f"<td style='padding:8px 14px;color:#CCC;font-size:12px;"
                             f"line-height:1.5'>{title}</td>"
                             f"<td style='padding:8px 14px;white-space:nowrap;font-size:11px'>"
                             f"<span style='color:#555'>{pub}</span>"
                             + (f"<span style='color:#FF6600'> | {src}</span>" if src else "")
                             + "</td></tr>")
                except: continue
            if rows:
                st.markdown(
                    f"<div style='background:#000;font-family:Courier New,monospace;"
                    f"padding:14px;border:1px solid #FF6600;border-radius:4px'>"
                    f"<table style='width:100%;border-collapse:collapse'>"
                    f"<thead><tr style='border-bottom:2px solid #FF6600'>"
                    f"<th style='color:#FF6600;text-align:left;padding:6px 14px'>HEADLINE</th>"
                    f"<th style='color:#FF6600;text-align:left;padding:6px 14px'>DATE / SOURCE</th>"
                    f"</tr></thead><tbody>{rows}</tbody></table></div>",
                    unsafe_allow_html=True)
            else:
                st.warning("No news available.")
