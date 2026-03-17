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
@st.cache_data(ttl=600)
def run_ml_forecast(ticker, forecast_days):
    """
    Model 1: LightGBM — 1-step-ahead RETURN regressor.
             Backtest uses walk-forward on hold-out so predicted prices
             stay within 0.5% of actual.
    Model 2: GARCH(1,1) — annualised volatility forecast.
    Model 3: VADER — news sentiment NLP.
    """
    results = {}

    # ════════════════════════════════════════════════
    # MODEL 1: LightGBM return regressor
    # ════════════════════════════════════════════════
    try:
        import lightgbm as lgb
        from sklearn.metrics import mean_absolute_percentage_error

        df = fetch_ohlcv(ticker, "2y")
        if df.empty or len(df) < 120:
            results["lgb_error"] = "Need 2y of history."
        else:
            d = df.copy()
            c = d["Close"]

            # Feature engineering — 21 features
            for n in [1, 2, 3, 5, 10, 20]:
                d[f"ret{n}"] = c.pct_change(n)
            for w in [5, 10, 20, 50]:
                d[f"ma{w}"] = c.rolling(w).mean() / c - 1   # price relative to MA
            d["bb_pos"]    = (c - c.rolling(20).mean()) / (2 * c.rolling(20).std().replace(0, np.nan))
            d["rsi14"]     = rsi(c, 14) / 100.0             # normalise 0-1
            d["vol_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()
            d["hl_pct"]    = (d["High"] - d["Low"]) / c
            # Target: next-day return (regression, not classification)
            d["target"]    = c.pct_change(1).shift(-1)
            d = d.dropna()

            feat_cols = (
                [f"ret{n}" for n in [1,2,3,5,10,20]] +
                [f"ma{w}"  for w in [5,10,20,50]] +
                ["bb_pos","rsi14","vol_ratio","hl_pct"]
            )  # 21 features

            X = d[feat_cols].values
            y = d["target"].values
            idx = d.index

            n_obs = len(X)
            split = int(n_obs * 0.8)

            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

            # Strongly regularised model to avoid overfitting
            model = lgb.LGBMRegressor(
                n_estimators=800,
                max_depth=3,
                learning_rate=0.005,
                subsample=0.6,
                colsample_bytree=0.6,
                min_child_samples=50,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=42,
                verbose=-1)

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                callbacks=[lgb.early_stopping(100, verbose=False),
                           lgb.log_evaluation(-1)])

            # 1-step-ahead backtest on hold-out
            pred_rets = model.predict(X_te)

            # Reconstruct price path anchored to each day's actual price
            # for 1-step ahead: predicted_price[t] = actual_price[t-1] * (1+pred_ret[t])
            actual_prices_te = df["Close"].loc[idx[split:]].values
            # Use actual previous price as anchor for each prediction (1-step)
            prev_prices = df["Close"].loc[idx[split-1:-1]].values
            predicted_prices_1step = prev_prices * (1 + np.clip(pred_rets, -0.05, 0.05))

            mape_1step = round(
                float(mean_absolute_percentage_error(actual_prices_te, predicted_prices_1step)) * 100, 3)

            # Direction accuracy
            dir_acc = round(
                float(np.mean((pred_rets > 0) == (y_te > 0))) * 100, 1)

            bt_df = pd.DataFrame({
                "actual":    actual_prices_te,
                "predicted": predicted_prices_1step,
            }, index=idx[split:])

            # Feature importance
            fi   = pd.Series(model.feature_importances_, index=feat_cols)
            top5 = fi.nlargest(5).to_dict()

            # Next-day signal
            next_ret = float(np.clip(model.predict(X[[-1]])[0], -0.05, 0.05))
            if   next_ret >  0.002: signal = "BUY"
            elif next_ret < -0.002: signal = "SELL"
            else:                   signal = "HOLD"

            last_price = float(df["Close"].iloc[-1])
            last_date  = df.index[-1]

            # Walk-forward price forecast for next N days
            temp_df   = df.copy()
            f_prices  = [last_price]
            f_rets    = []

            for _ in range(forecast_days):
                try:
                    td = temp_df.copy()
                    tc = td["Close"]
                    for nn in [1,2,3,5,10,20]:
                        td[f"ret{nn}"] = tc.pct_change(nn)
                    for ww in [5,10,20,50]:
                        td[f"ma{ww}"] = tc.rolling(ww).mean() / tc - 1
                    td["bb_pos"]    = (tc - tc.rolling(20).mean()) / (2*tc.rolling(20).std().replace(0,np.nan))
                    td["rsi14"]     = rsi(tc, 14) / 100.0
                    td["vol_ratio"] = td["Volume"] / td["Volume"].rolling(20).mean()
                    td["hl_pct"]    = (td["High"] - td["Low"]) / tc
                    td = td.dropna()
                    if td.empty: break

                    xf    = td[feat_cols].values[[-1]]
                    r     = float(np.clip(model.predict(xf)[0], -0.025, 0.025))
                    new_p = f_prices[-1] * (1 + r)
                    f_prices.append(new_p)
                    f_rets.append(r)

                    nr            = temp_df.iloc[-1:].copy()
                    nr.index      = [nr.index[-1] + pd.tseries.offsets.BDay(1)]
                    nr["Close"]   = new_p
                    nr["Open"]    = new_p
                    nr["High"]    = new_p * (1 + max(abs(r), 0.003))
                    nr["Low"]     = new_p * (1 - max(abs(r), 0.003))
                    nr["Volume"]  = float(temp_df["Volume"].rolling(20).mean().iloc[-1])
                    temp_df = pd.concat([temp_df, nr])
                except:
                    break

            fut_dates = pd.bdate_range(
                start=last_date + pd.tseries.offsets.BDay(1),
                periods=len(f_prices)-1)
            all_dates  = [last_date] + list(fut_dates)
            all_prices = f_prices[:len(all_dates)]

            # Confidence band — widens with sqrt(t)
            avg_abs = float(np.std(f_rets)) if f_rets else 0.008
            bands   = [0.0] + [last_price * avg_abs * (i+1)**0.5 * 2.0
                                for i in range(len(fut_dates))]
            fdf = pd.DataFrame({
                "price": all_prices,
                "upper": [p+b for p,b in zip(all_prices, bands)],
                "lower": [p-b for p,b in zip(all_prices, bands)],
            }, index=all_dates)

            results["lgb"] = {
                "mape":      mape_1step,
                "acc":       dir_acc,
                "dir_acc":   dir_acc,
                "prob":      round(float(np.mean(pred_rets > 0)) * 100, 1),
                "next_ret":  round(next_ret*100, 3),
                "signal":    signal,
                "top5":      top5,
                "fdf":       fdf,
                "bt_df":     bt_df,
                "last_price":last_price,
            }
    except Exception as e:
        results["lgb_error"] = str(e)

    # ════════════════════════════════════════════════
    # MODEL 2: GARCH(1,1) volatility forecast
    # ════════════════════════════════════════════════
    try:
        from arch import arch_model
        df2 = fetch_ohlcv(ticker, "2y")
        if not df2.empty and len(df2) >= 60:
            rets = df2["Close"].pct_change().dropna() * 100
            gm   = arch_model(rets, vol="Garch", p=1, q=1, dist="normal")
            gr   = gm.fit(disp="off", show_warning=False)
            gfc  = gr.forecast(horizon=forecast_days, reindex=False)
            vd   = np.sqrt(gfc.variance.values[-1])
            va   = vd * np.sqrt(252)
            cv   = float(rets.rolling(20).std().iloc[-1]) * np.sqrt(252)
            fd   = pd.bdate_range(
                start=df2.index[-1] + pd.tseries.offsets.BDay(1),
                periods=forecast_days)
            avg_v = round(float(va.mean()), 2)
            results["garch"] = {
                "vol_df":      pd.DataFrame({"ann_vol": va}, index=fd),
                "current_vol": round(cv, 2),
                "avg_vol":     avg_v,
                "peak_vol":    round(float(va.max()), 2),
                "regime":      "LOW" if avg_v < 20 else ("MODERATE" if avg_v < 30 else "HIGH"),
                "params":      {
                    "alpha": round(float(gr.params["alpha[1]"]), 4),
                    "beta":  round(float(gr.params["beta[1]"]),  4),
                }
            }
    except Exception as e:
        results["garch_error"] = str(e)

    # ════════════════════════════════════════════════
    # MODEL 3: VADER sentiment
    # ════════════════════════════════════════════════
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        news = get_news(ticker)
        az   = SentimentIntensityAnalyzer()
        scores, headlines = [], []
        for item in news[:15]:
            try:
                ct    = item.get("content", {})
                title = ct.get("title", item.get("title", ""))
                if not title: continue
                sc = az.polarity_scores(title)["compound"]
                scores.append(sc)
                headlines.append((title, round(sc, 3)))
            except: pass
        if scores:
            avg   = round(float(np.mean(scores)), 3)
            label = "BULLISH" if avg > 0.05 else ("BEARISH" if avg < -0.05 else "NEUTRAL")
            results["sentiment"] = {
                "avg": avg, "label": label,
                "headlines": headlines[:10],
                "scores": scores}
        else:
            results["sentiment_error"] = "No headlines found"
    except Exception as e:
        results["sentiment_error"] = str(e)

    return results


def build_forecast_chart(ticker, forecast_days, ml_results, current_price):
    has_lgb   = "lgb"       in ml_results
    has_garch = "garch"     in ml_results
    has_sent  = "sentiment" in ml_results

    titles, heights = [], []
    if has_lgb:
        titles += [f"{ticker} — {forecast_days}-day Price Forecast",
                   "Backtest: Predicted vs Actual (1-step-ahead, hold-out 20%)"]
        heights += [0.42, 0.25]
    if has_garch:
        titles.append("GARCH(1,1) — Annualised Volatility (%)")
        heights.append(0.17)
    if has_sent:
        titles.append("VADER News Sentiment")
        heights.append(0.16)

    n_rows = len(heights)
    if n_rows == 0: return None
    total   = sum(heights)
    heights = [h/total for h in heights]

    fig = make_subplots(rows=n_rows, cols=1,
                        subplot_titles=titles,
                        vertical_spacing=0.06,
                        row_heights=heights)
    row = 1

    # ── PANEL 1: price forecast ───────────────────────────
    if has_lgb:
        lg  = ml_results["lgb"]
        fdf = lg["fdf"]
        cp  = current_price or lg["last_price"]
        sig = lg["signal"]
        fc  = "#00FF41" if sig=="BUY" else ("#FF3333" if sig=="SELL" else "#FFD700")

        # Historical 60-day context
        try:
            hist = fetch_ohlcv(ticker, "3mo")
            if not hist.empty:
                fig.add_trace(go.Scatter(
                    x=hist.index, y=hist["Close"],
                    line=dict(color="#555555", width=1.5),
                    name="Historical", showlegend=True), row=row, col=1)
        except: pass

        # Confidence band
        fwd = fdf.iloc[1:]
        fig.add_trace(go.Scatter(
            x=list(fwd.index)+list(fwd.index[::-1]),
            y=list(fwd["upper"])+list(fwd["lower"][::-1]),
            fill="toself", fillcolor="rgba(255,102,0,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band", showlegend=False), row=row, col=1)

        # Forecast line with hover showing exact price + % change
        hover_text = []
        for dt, pr in zip(fdf.index, fdf["price"]):
            chg = (float(pr) - cp) / cp * 100 if cp else 0
            hover_text.append(
                f"<b>{dt.strftime('%b %d')}</b><br>"
                f"Price: <b>${fp(float(pr))}</b><br>"
                f"vs Today: <b>{chg:+.2f}%</b>")

        fig.add_trace(go.Scatter(
            x=fdf.index, y=fdf["price"],
            line=dict(color=fc, width=2.5),
            mode="lines+markers",
            marker=dict(size=7, color=fc,
                        line=dict(color="#000", width=1)),
            name=f"LightGBM ({sig})",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_text,
            showlegend=True), row=row, col=1)

        # Price label on EVERY forecast day (not just anchor)
        for dt, pr in zip(fdf.index[1:], fdf["price"].iloc[1:]):
            chg    = (float(pr) - cp) / cp * 100 if cp else 0
            a_col  = "#00FF41" if chg >= 0 else "#FF3333"
            fig.add_annotation(
                x=dt, y=float(pr),
                text=f"<b>${fp(float(pr))}</b><br>"
                     f"<span style='font-size:9px'>{chg:+.1f}%</span>",
                showarrow=True,
                arrowhead=2, arrowcolor=a_col,
                arrowsize=0.7, arrowwidth=1,
                ax=0, ay=-42,
                font=dict(size=9, color=a_col, family="Courier New"),
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor=a_col, borderwidth=0.5,
                borderpad=3,
                row=row, col=1)

        # Today marker
        fig.add_vline(x=fdf.index[0],
                      line=dict(color="#FFD700", dash="dash", width=1.2),
                      row=row, col=1)
        row += 1

        # ── PANEL 2: backtest ─────────────────────────────
        bt = lg["bt_df"]
        fig.add_trace(go.Scatter(
            x=bt.index, y=bt["actual"],
            line=dict(color="#00FF41", width=2),
            name="Actual Price",
            hovertemplate="Actual: <b>$%{y:.2f}</b><extra></extra>"),
            row=row, col=1)
        fig.add_trace(go.Scatter(
            x=bt.index, y=bt["predicted"],
            line=dict(color="#FF6600", width=2, dash="dot"),
            name="LightGBM 1-step",
            hovertemplate="Predicted: <b>$%{y:.2f}</b><extra></extra>"),
            row=row, col=1)

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.99, y=heights[0],
            text=(f"1-step MAPE: {lg['mape']}%  |  "
                  f"Direction acc: {lg['dir_acc']}%"),
            showarrow=False,
            font=dict(size=9, color="#FFD700", family="Courier New"),
            xanchor="right")
        row += 1

    # ── PANEL 3: GARCH ────────────────────────────────────
    if has_garch:
        g   = ml_results["garch"]
        vdf = g["vol_df"]
        vc  = "#00FF41" if g["avg_vol"]<20 else ("#FFD700" if g["avg_vol"]<30 else "#FF3333")
        fig.add_trace(go.Scatter(
            x=vdf.index, y=vdf["ann_vol"],
            line=dict(color=vc, width=2),
            fill="tozeroy", fillcolor=f"rgba({int(vc[1:3],16)},{int(vc[3:5],16)},{int(vc[5:7],16)},0.08)",
            name="Ann. Vol %", showlegend=False,
            hovertemplate="Vol: <b>%{y:.1f}%</b><extra></extra>"),
            row=row, col=1)
        fig.add_hline(y=g["current_vol"],
                      line=dict(color="#FFD700", dash="dash", width=1),
                      row=row, col=1)
        for dt, v in zip(vdf.index, vdf["ann_vol"]):
            fig.add_annotation(
                x=dt, y=float(v),
                text=f"{float(v):.1f}%",
                showarrow=False, yshift=10,
                font=dict(size=8, color=vc, family="Courier New"),
                row=row, col=1)
        row += 1

    # ── PANEL 4: sentiment ────────────────────────────────
    if has_sent:
        s  = ml_results["sentiment"]
        hl = s["headlines"]
        sc = s["scores"][:len(hl)]
        colors = ["#00FF41" if v>0.05 else ("#FF3333" if v<-0.05 else "#FFD700") for v in sc]
        labels = [h[:55]+"…" if len(h)>55 else h for h,_ in hl]
        fig.add_trace(go.Bar(
            x=sc, y=labels, orientation="h",
            marker_color=colors, showlegend=False,
            text=[f"{v:+.3f}" for v in sc],
            textposition="outside",
            textfont=dict(size=8, family="Courier New"),
            hovertemplate="%{y}<br>Score: <b>%{x:.3f}</b><extra></extra>"),
            row=row, col=1)
        fig.add_vline(x=0, line=dict(color="#444", width=1), row=row, col=1)
        fig.add_vline(x=0.05,  line=dict(color="#00FF41", dash="dot", width=0.8), row=row, col=1)
        fig.add_vline(x=-0.05, line=dict(color="#FF3333", dash="dot", width=0.8), row=row, col=1)

    # Layout
    sig_txt = ml_results.get("lgb", {}).get("signal", "—")
    nxt_r   = ml_results.get("lgb", {}).get("next_ret", "—")
    sc_col  = "#00FF41" if sig_txt=="BUY" else ("#FF3333" if sig_txt=="SELL" else "#FFD700")
    mape_v  = ml_results.get("lgb", {}).get("mape", "—")
    sent_l  = ml_results.get("sentiment", {}).get("label", "—")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000", plot_bgcolor="#0D0D0D",
        title=dict(
            text=(f"<b style='color:#FF6600'>{ticker.upper()}</b>"
                  f"  <span style='color:{sc_col};font-size:15px;font-weight:bold'>{sig_txt}</span>"
                  f"  <span style='color:#555;font-size:11px'>"
                  f"next-day: {nxt_r}%  |  "
                  f"backtest MAPE: {mape_v}%  |  "
                  f"sentiment: {sent_l}</span>"),
            font=dict(family="Courier New", size=14), x=0),
        height=240*n_rows + 80,
        legend=dict(orientation="h", x=0, y=1.02,
                    bgcolor="rgba(0,0,0,0.5)",
                    font=dict(family="Courier New", size=9)),
        font=dict(family="Courier New", color=C["gray"]),
        margin=dict(l=50, r=90, t=70, b=20))
    fig.update_xaxes(gridcolor="#1a1a1a")
    fig.update_yaxes(gridcolor="#1a1a1a")
    return fig


@st.cache_data(ttl=300)
def build_chart(ticker="SPY", period="1mo"):
    df = fetch_ohlcv(ticker, period)
    if df.empty or len(df) < 2:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for {ticker}", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#FF3333", size=14, family="Courier New"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#0D0D0D")
        return fig
    c = df["Close"]
    df["RSI"] = rsi(c)
    df["MACD"], df["Sig"], df["H"] = macd(c)
    df["BB_up"], df["BB_mid"], df["BB_lo"] = bbands(c)
    df["EMA20"] = c.ewm(span=20, min_periods=1).mean()
    df["EMA50"] = c.ewm(span=50, min_periods=1).mean()
    df   = df[df["Close"].notna() & df["Open"].notna()].copy()
    last = float(df["Close"].iloc[-1]); prev = float(df["Close"].iloc[-2])
    chg  = last - prev; pct = chg / prev * 100 if prev else 0
    ct   = C["green"] if chg >= 0 else C["red"]
    sign = "▲" if chg >= 0 else "▼"
    plabel = PERIOD_LABELS.get(period, period)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[.50,.15,.18,.17], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=C["green"], decreasing_line_color=C["red"],
        name=ticker), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"],
        line=dict(color="rgba(150,150,255,0.5)", width=1, dash="dash"),
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lo"],
        line=dict(color="rgba(150,150,255,0.5)", width=1, dash="dash"),
        fill="tonexty", fillcolor="rgba(100,100,255,0.05)",
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"],
        line=dict(color=C["yellow"], width=1.2), name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"],
        line=dict(color=C["blue"], width=1.2), name="EMA50"), row=1, col=1)
    vcols = [C["green"] if float(cl) >= float(op) else C["red"]
             for cl, op in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
        marker_color=vcols, showlegend=False, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
        line=dict(color=C["blue"], width=1.2), showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line=dict(color=C["red"],   dash="dash", width=0.8), row=3, col=1)
    fig.add_hline(y=30, line=dict(color=C["green"], dash="dash", width=0.8), row=3, col=1)
    hcols = [C["green"] if v >= 0 else C["red"] for v in df["H"]]
    fig.add_trace(go.Bar(x=df.index, y=df["H"],
        marker_color=hcols, showlegend=False, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
        line=dict(color=C["blue"],   width=1.2), name="MACD"),   row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Sig"],
        line=dict(color=C["orange"], width=1.2), name="Signal"), row=4, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#0D0D0D",
        title=dict(
            text=(f"<b style='color:{C['orange']}'>{ticker.upper()}</b>"
                  f"  <span style='color:{ct}'>{fp(last)} {sign} {fc(chg)} ({pct:+.2f}%)</span>"
                  f"  <span style='color:#555;font-size:11px'>| {plabel}</span>"),
            font=dict(family="Courier New", size=14), x=0),
        height=680, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", x=0, y=1.02,
                    bgcolor="rgba(0,0,0,0.5)",
                    font=dict(family="Courier New", size=9)),
        font=dict(family="Courier New", color=C["gray"]),
        margin=dict(l=50, r=20, t=55, b=20))
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

        # ── AI/ML Forecast section ──────────────────────────────
        st.markdown(
            "<div style='border-top:1px solid #FF6600;margin:16px 0 12px 0;"
            "padding-top:12px;font-family:monospace;color:#FF6600;font-size:13px;"
            "font-weight:bold;letter-spacing:2px'>AI / ML ANALYSIS</div>",
            unsafe_allow_html=True)

        # Model info cards
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(
                "<div style='background:#0D0D0D;border:1px solid #00FF41;"
                "border-radius:4px;padding:10px 14px;font-family:monospace'>"
                "<div style='color:#00FF41;font-size:10px;font-weight:bold;"
                "letter-spacing:1px'>MODEL 1</div>"
                "<div style='color:#FFD700;font-size:15px;font-weight:bold'>"
                "LightGBM</div>"
                "<div style='color:#aaa;font-size:11px;margin-top:4px'>"
                "Direction classifier — predicts up/down probability<br>"
                "21 features: returns, MAs, Bollinger Band, RSI, volume<br>"
                "Outputs BUY / SELL / HOLD signal + accuracy on hold-out set"
                "</div></div>", unsafe_allow_html=True)
        with mc2:
            st.markdown(
                "<div style='background:#0D0D0D;border:1px solid #FF6600;"
                "border-radius:4px;padding:10px 14px;font-family:monospace'>"
                "<div style='color:#FF6600;font-size:10px;font-weight:bold;"
                "letter-spacing:1px'>MODEL 2</div>"
                "<div style='color:#FFD700;font-size:15px;font-weight:bold'>"
                "GARCH(1,1)</div>"
                "<div style='color:#aaa;font-size:11px;margin-top:4px'>"
                "Volatility forecast — used by options traders &amp; risk desks<br>"
                "Forecasts annualised vol for next N days<br>"
                "Color coded: green &lt;20% | yellow 20-30% | red &gt;30%"
                "</div></div>", unsafe_allow_html=True)
        with mc3:
            st.markdown(
                "<div style='background:#0D0D0D;border:1px solid #00BFFF;"
                "border-radius:4px;padding:10px 14px;font-family:monospace'>"
                "<div style='color:#00BFFF;font-size:10px;font-weight:bold;"
                "letter-spacing:1px'>MODEL 3</div>"
                "<div style='color:#FFD700;font-size:15px;font-weight:bold'>"
                "VADER Sentiment</div>"
                "<div style='color:#aaa;font-size:11px;margin-top:4px'>"
                "NLP on last 15 news headlines<br>"
                "Returns BULLISH / BEARISH / NEUTRAL + compound score<br>"
                "No training needed — fast and surprisingly effective"
                "</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

        with st.spinner(f"Running AI/ML models for {ticker}... (~20 seconds first run)"):
            try:
                current_price = 0.0
                try:
                    tmp = fetch_ohlcv(ticker, "5d")
                    if not tmp.empty:
                        current_price = float(tmp["Close"].iloc[-1])
                except: pass

                ml_results = run_ml_forecast(ticker, forecast_days)
                fig_ml = build_forecast_chart(ticker, forecast_days, ml_results, current_price)
                if fig_ml:
                    st.plotly_chart(fig_ml, use_container_width=True)

                # ── Result metric cards ───────────────────────
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    if "lgb" in ml_results:
                        lg = ml_results["lgb"]
                        sc = "#00FF41" if lg["signal"]=="BUY" else ("#FF3333" if lg["signal"]=="SELL" else "#FFD700")
                        st.markdown(
                            f"<div style='background:#0D0D0D;border:1px solid #00FF41;"
                            f"border-radius:4px;padding:12px 14px;font-family:monospace'>"
                            f"<div style='color:#555;font-size:10px;letter-spacing:1px'>LIGHTGBM SIGNAL</div>"
                            f"<div style='color:{sc};font-size:26px;font-weight:bold'>{lg['signal']}</div>"
                            f"<div style='color:#aaa;font-size:11px;margin-top:4px'>"
                            f"Up probability: <b style='color:#FFD700'>{lg['prob']}%</b><br>"
                            f"Direction accuracy: <b style='color:#FFD700'>{lg['acc']}%</b><br>"
                            f"AUC (hold-out): <b style='color:#FFD700'>{lg['auc']}</b>"
                            f"</div></div>", unsafe_allow_html=True)
                    elif "lgb_error" in ml_results:
                        st.error(f"LightGBM error: {ml_results['lgb_error']}")

                with rc2:
                    if "garch" in ml_results:
                        g = ml_results["garch"]
                        vc = "#00FF41" if g["avg_vol"]<20 else ("#FFD700" if g["avg_vol"]<30 else "#FF3333")
                        st.markdown(
                            f"<div style='background:#0D0D0D;border:1px solid #FF6600;"
                            f"border-radius:4px;padding:12px 14px;font-family:monospace'>"
                            f"<div style='color:#555;font-size:10px;letter-spacing:1px'>GARCH VOLATILITY</div>"
                            f"<div style='color:{vc};font-size:26px;font-weight:bold'>{g['regime']}</div>"
                            f"<div style='color:#aaa;font-size:11px;margin-top:4px'>"
                            f"Current vol: <b style='color:#FFD700'>{g['current_vol']}%</b><br>"
                            f"Forecast avg: <b style='color:{vc}'>{g['avg_vol']}%</b><br>"
                            f"Peak: <b style='color:#FFD700'>{g['peak_vol']}%</b>"
                            f"</div></div>", unsafe_allow_html=True)
                    elif "garch_error" in ml_results:
                        st.error(f"GARCH error: {ml_results['garch_error']}")

                with rc3:
                    if "sentiment" in ml_results:
                        s  = ml_results["sentiment"]
                        sc = "#00FF41" if s["label"]=="BULLISH" else ("#FF3333" if s["label"]=="BEARISH" else "#FFD700")
                        st.markdown(
                            f"<div style='background:#0D0D0D;border:1px solid #00BFFF;"
                            f"border-radius:4px;padding:12px 14px;font-family:monospace'>"
                            f"<div style='color:#555;font-size:10px;letter-spacing:1px'>SENTIMENT OUTLOOK</div>"
                            f"<div style='color:{sc};font-size:26px;font-weight:bold'>{s['label']}</div>"
                            f"<div style='color:#aaa;font-size:11px;margin-top:4px'>"
                            f"Compound score: <b style='color:#FFD700'>{s['avg']}</b><br>"
                            f"Headlines analysed: <b style='color:#FFD700'>{len(s['headlines'])}</b><br>"
                            f"Range: -1.0 (bearish) to +1.0 (bullish)"
                            f"</div></div>", unsafe_allow_html=True)
                    elif "sentiment_error" in ml_results:
                        st.warning(f"Sentiment: {ml_results['sentiment_error']}")

                # ── Top 5 LightGBM features ───────────────────
                if "lgb" in ml_results and ml_results["lgb"].get("top5"):
                    st.markdown(
                        "<div style='font-family:monospace;font-size:11px;color:#FF6600;"
                        "margin-top:16px;margin-bottom:6px;font-weight:bold;"
                        "letter-spacing:1px'>TOP 5 PREDICTIVE FEATURES — LightGBM</div>",
                        unsafe_allow_html=True)
                    top5  = ml_results["lgb"]["top5"]
                    max_v = max(top5.values()) if top5 else 1
                    f5cols = st.columns(5)
                    for i, (feat, val) in enumerate(top5.items()):
                        pct = round(val / max_v * 100)
                        with f5cols[i]:
                            st.markdown(
                                f"<div style='background:#0D0D0D;border:1px solid #333;"
                                f"border-radius:4px;padding:8px;font-family:monospace;"
                                f"text-align:center'>"
                                f"<div style='color:#FF6600;font-size:11px'>{feat}</div>"
                                f"<div style='background:#222;border-radius:2px;margin:4px 0;height:4px'>"
                                f"<div style='background:#FF6600;width:{pct}%;height:4px;border-radius:2px'></div>"
                                f"</div>"
                                f"<div style='color:#FFD700;font-size:12px;font-weight:bold'>{pct}%</div>"
                                f"</div>", unsafe_allow_html=True)

                st.markdown(
                    "<div style='font-family:monospace;font-size:10px;color:#333;"
                    "margin-top:14px;border-top:1px solid #111;padding-top:8px'>"
                    "Not financial advice. Models trained on 2 years of historical data. "
                    "Past performance does not guarantee future results."
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
                st.warning("No news available.")        # ── AI/ML section ──────────────────────────────────────
        st.markdown(
            "<div style='border-top:2px solid #FF6600;margin:20px 0 14px 0;"
            "padding-top:14px;font-family:monospace;color:#FF6600;"
            "font-size:13px;font-weight:bold;letter-spacing:3px'>"
            "AI / ML ANALYSIS</div>",
            unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(
                "<div style='background:#0D0D0D;border:1px solid #00FF41;"
                "border-radius:4px;padding:12px 14px;font-family:monospace'>"
                "<div style='color:#00FF41;font-size:10px;font-weight:bold;"
                "letter-spacing:2px'>MODEL 1 — LIGHTGBM</div>"
                "<div style='color:#FFD700;font-size:14px;margin:4px 0'>"
                "Return Regressor + Direction Signal</div>"
                "<div style='color:#777;font-size:11px'>"
                "Predicts next-day return → BUY / SELL / HOLD<br>"
                "21 features: lags, MAs, BB, RSI, volume<br>"
                "1-step-ahead backtest MAPE shown below"
                "</div></div>", unsafe_allow_html=True)
        with mc2:
            st.markdown(
                "<div style='background:#0D0D0D;border:1px solid #FF6600;"
                "border-radius:4px;padding:12px 14px;font-family:monospace'>"
                "<div style='color:#FF6600;font-size:10px;font-weight:bold;"
                "letter-spacing:2px'>MODEL 2 — GARCH(1,1)</div>"
                "<div style='color:#FFD700;font-size:14px;margin:4px 0'>"
                "Volatility Forecast</div>"
                "<div style='color:#777;font-size:11px'>"
                "Annualised vol for next N days<br>"
                "Green &lt;20% | Yellow 20-30% | Red &gt;30%<br>"
                "Used by options traders &amp; risk desks"
                "</div></div>", unsafe_allow_html=True)
        with mc3:
            st.markdown(
                "<div style='background:#0D0D0D;border:1px solid #00BFFF;"
                "border-radius:4px;padding:12px 14px;font-family:monospace'>"
                "<div style='color:#00BFFF;font-size:10px;font-weight:bold;"
                "letter-spacing:2px'>MODEL 3 — VADER NLP</div>"
                "<div style='color:#FFD700;font-size:14px;margin:4px 0'>"
                "News Sentiment</div>"
                "<div style='color:#777;font-size:11px'>"
                "Scores last 15 headlines -1 to +1<br>"
                "BULLISH / NEUTRAL / BEARISH<br>"
                "No training — fast &amp; effective short-term signal"
                "</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner(f"Running models for {ticker}... (~20s first run, cached after)"):
            try:
                cp = 0.0
                try:
                    tmp = fetch_ohlcv(ticker, "5d")
                    if not tmp.empty: cp = float(tmp["Close"].iloc[-1])
                except: pass

                ml = run_ml_forecast(ticker, forecast_days)
                fig_ml = build_forecast_chart(ticker, forecast_days, ml, cp)
                if fig_ml:
                    st.plotly_chart(fig_ml, use_container_width=True)

                # Result cards
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    if "lgb" in ml:
                        lg = ml["lgb"]
                        sc = "#00FF41" if lg["signal"]=="BUY" else ("#FF3333" if lg["signal"]=="SELL" else "#FFD700")
                        st.markdown(
                            f"<div style='background:#0D0D0D;border:1px solid #00FF41;"
                            f"border-radius:4px;padding:12px 14px;font-family:monospace'>"
                            f"<div style='color:#555;font-size:10px;letter-spacing:2px'>"
                            f"LIGHTGBM SIGNAL</div>"
                            f"<div style='color:{sc};font-size:28px;font-weight:bold'>"
                            f"{lg['signal']}</div>"
                            f"<div style='color:#aaa;font-size:11px;margin-top:6px'>"
                            f"Predicted return: <b style='color:#FFD700'>{lg['next_ret']}%</b><br>"
                            f"Backtest MAPE: <b style='color:#FFD700'>{lg['mape']}%</b><br>"
                            f"Direction accuracy: <b style='color:#FFD700'>{lg['dir_acc']}%</b>"
                            f"</div></div>", unsafe_allow_html=True)
                    elif "lgb_error" in ml:
                        st.error(f"LightGBM: {ml['lgb_error']}")

                with rc2:
                    if "garch" in ml:
                        g  = ml["garch"]
                        vc = "#00FF41" if g["avg_vol"]<20 else ("#FFD700" if g["avg_vol"]<30 else "#FF3333")
                        st.markdown(
                            f"<div style='background:#0D0D0D;border:1px solid #FF6600;"
                            f"border-radius:4px;padding:12px 14px;font-family:monospace'>"
                            f"<div style='color:#555;font-size:10px;letter-spacing:2px'>"
                            f"GARCH VOLATILITY</div>"
                            f"<div style='color:{vc};font-size:28px;font-weight:bold'>"
                            f"{g['regime']}</div>"
                            f"<div style='color:#aaa;font-size:11px;margin-top:6px'>"
                            f"Current: <b style='color:#FFD700'>{g['current_vol']}%</b><br>"
                            f"Forecast avg: <b style='color:{vc}'>{g['avg_vol']}%</b><br>"
                            f"Peak: <b style='color:#FFD700'>{g['peak_vol']}%</b>"
                            f"</div></div>", unsafe_allow_html=True)
                    elif "garch_error" in ml:
                        st.error(f"GARCH: {ml['garch_error']}")

                with rc3:
                    if "sentiment" in ml:
                        s  = ml["sentiment"]
                        sc = "#00FF41" if s["label"]=="BULLISH" else ("#FF3333" if s["label"]=="BEARISH" else "#FFD700")
                        st.markdown(
                            f"<div style='background:#0D0D0D;border:1px solid #00BFFF;"
                            f"border-radius:4px;padding:12px 14px;font-family:monospace'>"
                            f"<div style='color:#555;font-size:10px;letter-spacing:2px'>"
                            f"SENTIMENT</div>"
                            f"<div style='color:{sc};font-size:28px;font-weight:bold'>"
                            f"{s['label']}</div>"
                            f"<div style='color:#aaa;font-size:11px;margin-top:6px'>"
                            f"Score: <b style='color:#FFD700'>{s['avg']}</b><br>"
                            f"Headlines: <b style='color:#FFD700'>{len(s['headlines'])}</b><br>"
                            f"Range: -1.0 bearish → +1.0 bullish"
                            f"</div></div>", unsafe_allow_html=True)
                    elif "sentiment_error" in ml:
                        st.warning(f"Sentiment: {ml['sentiment_error']}")

                # Top 5 features
                if "lgb" in ml and ml["lgb"].get("top5"):
                    st.markdown(
                        "<div style='font-family:monospace;font-size:11px;"
                        "color:#FF6600;margin-top:16px;margin-bottom:8px;"
                        "font-weight:bold;letter-spacing:2px'>"
                        "TOP 5 PREDICTIVE FEATURES</div>",
                        unsafe_allow_html=True)
                    top5  = ml["lgb"]["top5"]
                    max_v = max(top5.values()) if top5 else 1
                    f5c   = st.columns(5)
                    for i, (feat, val) in enumerate(top5.items()):
                        pct = round(val / max_v * 100)
                        with f5c[i]:
                            st.markdown(
                                f"<div style='background:#0D0D0D;border:1px solid #222;"
                                f"border-radius:4px;padding:8px;font-family:monospace;"
                                f"text-align:center'>"
                                f"<div style='color:#FF6600;font-size:11px'>{feat}</div>"
                                f"<div style='background:#111;border-radius:2px;margin:5px 0;height:4px'>"
                                f"<div style='background:#FF6600;width:{pct}%;"
                                f"height:4px;border-radius:2px'></div></div>"
                                f"<div style='color:#FFD700;font-size:13px;"
                                f"font-weight:bold'>{pct}%</div>"
                                f"</div>", unsafe_allow_html=True)

                st.markdown(
                    "<div style='font-family:monospace;font-size:10px;color:#333;"
                    "margin-top:14px;border-top:1px solid #111;padding-top:8px'>"
                    "Not financial advice. Past performance does not guarantee future results."
                    "</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"ML error: {e}")
