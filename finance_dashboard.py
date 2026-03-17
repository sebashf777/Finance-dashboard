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
    "EUROPE":         {"FTSE 100":"^FTSE","DAX":"^GDAXI","CAC 40":"^FCHI","Euro Stoxx 50":"^STOXX50E","IBEX 35":"^IBEX","AEX":"^AEX"},
    "ASIA-PACIFIC":   {"Nikkei 225":"^N225","Hang Seng":"^HSI","ASX 200":"^AXJO","Shanghai":"000001.SS","KOSPI":"^KS11","TAIEX":"^TWII"},
    "AMERICAS":       {"Bovespa":"^BVSP","IPC Mexico":"^MXX","TSX Canada":"^GSPTSE","Merval":"^MERV"}
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
  .stTextInput input, .stSelectbox select { background:#0D0D0D; color:#FFD700; border:1px solid #FF6600; font-family:monospace; }
  h1,h2,h3 { color:#FF6600 !important; font-family:monospace !important; }
  div[data-testid="stMetricValue"] { color:#FFD700 !important; font-family:monospace; }
  div[data-testid="stMetricLabel"] { color:#FF6600 !important; font-family:monospace; }
  .block-container { padding-top:0.5rem !important; }
</style>""", unsafe_allow_html=True)

def _close(raw, tk=None):
    if isinstance(raw.columns, pd.MultiIndex):
        sub = raw["Close"]
        if tk and tk in sub.columns: return sub[tk].dropna()
        return sub.dropna()
    return raw["Close"].dropna()

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
    m = c.ewm(span=f,min_periods=1).mean() - c.ewm(span=s,min_periods=1).mean()
    sg = m.ewm(span=sig,min_periods=1).mean()
    return m, sg, m - sg

def bbands(c, w=20):
    mid = c.rolling(w,min_periods=1).mean(); std = c.rolling(w,min_periods=1).std()
    return mid+2*std, mid, mid-2*std

def fp(p):
    if p>10000: return f"{p:,.0f}"
    elif p>100: return f"{p:,.2f}"
    elif p>1:   return f"{p:.4f}"
    elif p>0:   return f"{p:.6f}"
    return "—"

def fc(c):
    a=abs(c)
    if a>100:   return f"{a:,.2f}"
    elif a>0.1: return f"{a:.4f}"
    return f"{a:.6f}"

def html_table(title, sections, icon="◈"):
    tks    = [tk for sec in sections.values() for tk in sec.values()]
    quotes = batch_quotes(tuple(tks))
    now    = ny_now().strftime("%I:%M %p ET")
    rows   = ""
    for sec_name, items in sections.items():
        rows += f"<tr><td colspan='5' style='background:#111122;color:#FF6600;font-weight:bold;padding:7px 14px;font-size:11px;letter-spacing:2px;border-top:1px solid #222'>{sec_name}</td></tr>"
        for name, tk in items.items():
            q    = quotes.get(tk, dict(price=0,chg=0,pct=0))
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
            f"<span>{icon} {title}</span><span style='color:#555;font-size:11px'>{now}</span></div>"
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<thead><tr style='border-bottom:2px solid #FF6600'>{hdr}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")

def build_global():
    tks    = tuple(tk for reg in GLOBAL_INDICES.values() for tk in reg.values())
    quotes = batch_quotes(tks)
    fig    = plt.figure(figsize=(16,9), facecolor=C["bg"])
    gs     = gridspec.GridSpec(2,2, figure=fig, hspace=0.52, wspace=0.3)
    rcols  = [C["blue"],"#8888ff",C["orange"],"#ffaa00"]
    for pi,((rn,ri),rc) in enumerate(zip(GLOBAL_INDICES.items(),rcols)):
        ax = fig.add_subplot(gs[pi//2, pi%2])
        ax.set_facecolor(C["bg"]); ax.axis("off")
        ax.set_title(f"  {rn}", color=rc, fontsize=11, fontweight="bold", fontfamily="monospace", loc="left", pad=8)
        items=list(ri.items()); n=len(items)
        cn=min(3,n); rn2=(n+cn-1)//cn
        bw=1/cn-0.03; bh=0.78/rn2-0.04
        for i,(name,tk) in enumerate(items):
            q=quotes.get(tk,dict(price=0,chg=0,pct=0))
            r=i//cn; ci=i%cn
            bx=ci/cn+0.01; by=0.92-(r+1)*(bh+0.04)
            col=C["green"] if q["chg"]>=0 else C["red"]
            ax.add_patch(FancyBboxPatch((bx,by),bw,bh,boxstyle="round,pad=0.01",
                facecolor="#0D0D0D",edgecolor=col,linewidth=1.1,transform=ax.transAxes))
            cx=bx+bw/2; cy=by+bh/2
            sign="▲" if q["chg"]>=0 else "▼"
            ax.text(cx,cy+bh*.26,name,ha="center",color=rc,fontsize=7,fontweight="bold",fontfamily="monospace",transform=ax.transAxes)
            ax.text(cx,cy,fp(q["price"]),ha="center",color=C["yellow"],fontsize=9.5,fontweight="bold",fontfamily="monospace",transform=ax.transAxes)
            ax.text(cx,cy-bh*.28,f"{sign} {q['pct']:+.2f}%",ha="center",color=col,fontsize=7,fontfamily="monospace",transform=ax.transAxes)
    plt.suptitle("GLOBAL EQUITY MARKETS",color=C["orange"],fontsize=14,fontweight="bold",fontfamily="monospace",y=0.99)
    plt.tight_layout(pad=1.5,rect=[0,0,1,0.97])
    return fig

@st.cache_data(ttl=300)
def build_chart(ticker="SPY", period="1mo"):
    df = fetch_ohlcv(ticker, period)
    if df.empty or len(df)<2:
        fig=go.Figure()
        fig.add_annotation(text=f"No data for {ticker}",xref="paper",yref="paper",
            x=0.5,y=0.5,showarrow=False,font=dict(color="#FF3333",size=14,family="Courier New"))
        fig.update_layout(template="plotly_dark",paper_bgcolor="#000",plot_bgcolor="#0D0D0D")
        return fig
    c=df["Close"]
    df["RSI"]=rsi(c); df["MACD"],df["Sig"],df["H"]=macd(c)
    df["BB_up"],df["BB_mid"],df["BB_lo"]=bbands(c)
    df["EMA20"]=c.ewm(span=20,min_periods=1).mean()
    df["EMA50"]=c.ewm(span=50,min_periods=1).mean()
    df=df[df["Close"].notna()&df["Open"].notna()].copy()
    last=float(df["Close"].iloc[-1]); prev=float(df["Close"].iloc[-2])
    chg=last-prev; pct=chg/prev*100 if prev else 0
    ct=C["green"] if chg>=0 else C["red"]; sign="▲" if chg>=0 else "▼"
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[.50,.15,.18,.17],vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],
        increasing_line_color=C["green"],decreasing_line_color=C["red"],name=ticker),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_up"],line=dict(color="rgba(150,150,255,0.5)",width=1,dash="dash"),showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["BB_lo"],line=dict(color="rgba(150,150,255,0.5)",width=1,dash="dash"),fill="tonexty",fillcolor="rgba(100,100,255,0.05)",showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["EMA20"],line=dict(color=C["yellow"],width=1.2),name="EMA20"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["EMA50"],line=dict(color=C["blue"],width=1.2),name="EMA50"),row=1,col=1)
    vcols=[C["green"] if float(cl)>=float(op) else C["red"] for cl,op in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],marker_color=vcols,showlegend=False,opacity=0.7),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],line=dict(color=C["blue"],width=1.2),showlegend=False),row=3,col=1)
    fig.add_hline(y=70,line=dict(color=C["red"],  dash="dash",width=0.8),row=3,col=1)
    fig.add_hline(y=30,line=dict(color=C["green"],dash="dash",width=0.8),row=3,col=1)
    hcols=[C["green"] if v>=0 else C["red"] for v in df["H"]]
    fig.add_trace(go.Bar(x=df.index,y=df["H"],marker_color=hcols,showlegend=False,opacity=0.7),row=4,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["MACD"],line=dict(color=C["blue"],  width=1.2),name="MACD"),row=4,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["Sig"], line=dict(color=C["orange"],width=1.2),name="Signal"),row=4,col=1)
    period_label = PERIOD_LABELS.get(period, period)
    fig.update_layout(
        template="plotly_dark",paper_bgcolor="#000",plot_bgcolor="#0D0D0D",
        title=dict(
            text=f"<b style='color:{C['orange']}'>{ticker.upper()}</b>  <span style='color:{ct}'>{fp(last)} {sign} {fc(chg)} ({pct:+.2f}%)</span>  <span style='color:#555;font-size:11px'>| {period_label}</span>",
            font=dict(family="Courier New",size=14),x=0),
        height=680,xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",x=0,y=1.02,bgcolor="rgba(0,0,0,0.5)",font=dict(family="Courier New",size=9)),
        font=dict(family="Courier New",color=C["gray"]),
        margin=dict(l=50,r=20,t=55,b=20))
    fig.update_xaxes(gridcolor="#1a1a1a")
    fig.update_yaxes(gridcolor="#1a1a1a")
    return fig

def build_macro():
    fig = plt.figure(figsize=(16,9), facecolor=C["bg"])
    gs  = gridspec.GridSpec(2,3, figure=fig, hspace=0.5, wspace=0.35)
    ax_vix = fig.add_subplot(gs[0,:2]); ax_vix.set_facecolor("#0D0D0D")
    try:
        vd=yf.download("^VIX",period="1y",auto_adjust=True,progress=False)
        cv=_close(vd).dropna(); lv=float(cv.iloc[-1])
        vc=C["green"] if lv<20 else C["red"]
        ax_vix.plot(vd.index[-len(cv):],cv,color=C["orange"],linewidth=1.4)
        ax_vix.fill_between(vd.index[-len(cv):],cv,float(cv.min()),alpha=0.15,color=C["orange"])
        ax_vix.axhline(20,color=C["red"],  lw=0.8,ls="--",alpha=0.6,label="Fear (>20)")
        ax_vix.axhline(12,color=C["green"],lw=0.8,ls="--",alpha=0.6,label="Greed (<12)")
        ax_vix.fill_between(vd.index[-len(cv):],0,12, alpha=0.06,color=C["green"])
        ax_vix.fill_between(vd.index[-len(cv):],20,100,alpha=0.06,color=C["red"])
        ax_vix.text(0.99,0.93,f"VIX: {lv:.2f}",transform=ax_vix.transAxes,
            ha="right",va="top",color=vc,fontsize=13,fontweight="bold",fontfamily="monospace")
    except: pass
    ax_vix.set_title("  VIX — VOLATILITY / FEAR INDEX (1Y)",color=C["orange"],fontsize=11,fontweight="bold",fontfamily="monospace",loc="left")
    ax_vix.tick_params(colors=C["gray"],labelsize=7)
    ax_vix.legend(facecolor="#111",edgecolor="#333",labelcolor=C["white"],fontsize=7)
    ax_vix.grid(alpha=0.1,color=C["gray"])
    for sp in ax_vix.spines.values(): sp.set_edgecolor("#333")
    ax_yc=fig.add_subplot(gs[0,2]); ax_yc.set_facecolor("#0D0D0D")
    tks_y=list(YIELDS.values()); lbs_y=list(YIELDS.keys()); mx=[0.25,5,10,30]
    try:
        ry=yf.download(tks_y,period="1y",auto_adjust=True,progress=False)
        def gyc(idx):
            v=[]
            for tk in tks_y:
                try: v.append(float(_close(ry,tk).iloc[idx]))
                except: v.append(None)
            return v
        def ycp(vals,lbl,col,sty="-"):
            pts=[(x,v) for x,v in zip(mx,vals) if v and v>0]
            if not pts: return
            xs,ys=zip(*pts)
            ax_yc.plot(xs,ys,f"o{sty}",color=col,linewidth=2,markersize=6,label=lbl,alpha=0.9)
            if sty=="-":
                ax_yc.fill_between(xs,ys,alpha=0.1,color=col)
                for x,y,lb in zip(xs,ys,lbs_y):
                    ax_yc.annotate(f"{lb}\n{y:.2f}%",xy=(x,y),xytext=(0,12),
                        textcoords="offset points",ha="center",fontsize=7,color=C["yellow"],fontfamily="monospace")
        ycp(gyc(-1),"Today", C["orange"],"-")
        ycp(gyc(0), "1Y Ago",C["gray"],  "--")
    except: pass
    ax_yc.set_xticks(mx); ax_yc.set_xticklabels(lbs_y,color=C["gray"],fontsize=7)
    ax_yc.set_title("  US YIELD CURVE",color=C["orange"],fontsize=10,fontweight="bold",fontfamily="monospace",loc="left")
    ax_yc.tick_params(colors=C["gray"],labelsize=7)
    ax_yc.legend(facecolor="#111",edgecolor="#333",labelcolor=C["white"],fontsize=7)
    ax_yc.grid(alpha=0.1,color=C["gray"])
    for sp in ax_yc.spines.values(): sp.set_edgecolor("#333")
    ax_sec=fig.add_subplot(gs[1,:2]); ax_sec.set_facecolor("#0D0D0D")
    sq=batch_quotes(tuple(SECTORS.keys()))
    slabs=list(SECTORS.values()); svals=[sq.get(tk,dict(pct=0))["pct"] for tk in SECTORS]
    scols=[C["green"] if v>=0 else C["red"] for v in svals]
    sbars=ax_sec.barh(slabs,svals,color=scols,height=0.6,edgecolor="#222")
    ax_sec.axvline(0,color=C["gray"],linewidth=0.8,linestyle="--")
    ax_sec.set_title("  S&P 500 SECTOR ROTATION",color=C["orange"],fontsize=10,fontweight="bold",fontfamily="monospace",loc="left")
    ax_sec.tick_params(colors=C["white"],labelsize=8)
    for sp in ax_sec.spines.values(): sp.set_edgecolor("#333")
    for bar,val in zip(sbars,svals):
        ax_sec.text(val+(0.02 if val>=0 else -0.02),bar.get_y()+bar.get_height()/2,
            f"{val:+.2f}%",va="center",ha="left" if val>=0 else "right",
            color=C["white"],fontsize=7.5,fontfamily="monospace")
    ax_sn=fig.add_subplot(gs[1,2]); ax_sn.set_facecolor("#0D0D0D"); ax_sn.axis("off")
    ax_sn.set_title("  KEY ASSETS SNAPSHOT",color=C["orange"],fontsize=10,fontweight="bold",fontfamily="monospace",loc="left")
    snap={"Gold":"GC=F","WTI Oil":"CL=F","BTC":"BTC-USD","EUR/USD":"EURUSD=X","10Y Yield":"^TNX","Silver":"SI=F","Copper":"HG=F","Nat. Gas":"NG=F"}
    snq=batch_quotes(tuple(snap.values()))
    yp=0.88
    for name,tk in snap.items():
        q=snq.get(tk,dict(price=0,chg=0,pct=0))
        col=C["green"] if q["chg"]>=0 else C["red"]; sign="▲" if q["chg"]>=0 else "▼"
        ax_sn.text(0.02,yp,name,transform=ax_sn.transAxes,color=C["orange"],fontsize=8.5,fontfamily="monospace",fontweight="bold")
        ax_sn.text(0.50,yp,fp(q["price"]),transform=ax_sn.transAxes,color=C["yellow"],fontsize=8.5,fontfamily="monospace")
        ax_sn.text(0.80,yp,f"{sign}{abs(q['pct']):.2f}%",transform=ax_sn.transAxes,color=col,fontsize=8.5,fontfamily="monospace")
        yp-=0.11
    plt.suptitle("MACRO DASHBOARD",color=C["orange"],fontsize=14,fontweight="bold",fontfamily="monospace",y=0.99)
    plt.tight_layout(pad=1.5,rect=[0,0,1,0.97])
    return fig

# ── SESSION STATE ────────────────────────────────────────────
if "chart_period" not in st.session_state:
    st.session_state["chart_period"] = "1mo"

# ── HEADER ───────────────────────────────────────────────────
now_str = ny_now().strftime("%A, %B %d %Y  |  %I:%M %p ET")
st.markdown(
    f"<div style='background:#000;font-family:Courier New,monospace;padding:12px 20px;"
    f"border-bottom:3px solid #FF6600;display:flex;justify-content:space-between;"
    f"align-items:center;margin-bottom:12px'>"
    f"<span style='color:#FF6600;font-size:20px;font-weight:bold;letter-spacing:3px'>GLOBAL FINANCE DASHBOARD</span>"
    f"<span style='color:#FFD700;font-size:11px'>{now_str}</span>"
    f"<span style='color:#00FF41;font-size:12px'>LIVE | Yahoo Finance</span></div>",
    unsafe_allow_html=True)

t1,t2,t3,t4,t5,t6,t7 = st.tabs(["GLOBAL","FOREX","COMMODITIES","CRYPTO","CHART","MACRO","NEWS"])

with t1:
    if st.button("Refresh",key="rg"): st.cache_data.clear()
    with st.spinner("Loading global markets..."):
        fig=build_global(); st.pyplot(fig,use_container_width=True); plt.close(fig)

with t2:
    if st.button("Refresh",key="rf"): st.cache_data.clear()
    with st.spinner("Fetching forex..."):
        st.markdown(html_table("FOREX - CURRENCY MARKETS",{"MAJOR PAIRS":FOREX_MAJOR,"EMERGING MARKETS":FOREX_EM},"FX"),unsafe_allow_html=True)

with t3:
    if st.button("Refresh",key="rc"): st.cache_data.clear()
    with st.spinner("Fetching commodities..."):
        st.markdown(html_table("GLOBAL COMMODITIES",COMMODITIES,"C"),unsafe_allow_html=True)

with t4:
    if st.button("Refresh",key="rk"): st.cache_data.clear()
    with st.spinner("Fetching crypto..."):
        st.markdown(html_table("CRYPTOCURRENCY MARKETS",CRYPTO,"B"),unsafe_allow_html=True)

with t5:
    col_tk, col_ref, _ = st.columns([2, 1, 9])
    with col_tk:
        ticker = st.text_input("Ticker", value="SPY", key="ct").upper()
    with col_ref:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("Refresh", key="chart_refresh"):
            st.cache_data.clear()

    st.markdown("<div style='margin-bottom:4px;color:#FF6600;font-family:monospace;font-size:12px;font-weight:bold'>SELECT PERIOD</div>", unsafe_allow_html=True)

    pb1, pb2, pb3, pb4, _ = st.columns([1, 1, 1, 2, 7])

    with pb1:
        lbl1 = "[ 1 Week ]" if st.session_state["chart_period"] == "5d" else "1 Week"
        if st.button(lbl1, key="p5d"):
            st.session_state["chart_period"] = "5d"
            st.rerun()

    with pb2:
        lbl2 = "[ 1 Month ]" if st.session_state["chart_period"] == "1mo" else "1 Month"
        if st.button(lbl2, key="p1mo"):
            st.session_state["chart_period"] = "1mo"
            st.rerun()

    with pb3:
        lbl3 = "[ 1 Year ]" if st.session_state["chart_period"] == "1y" else "1 Year"
        if st.button(lbl3, key="p1y"):
            st.session_state["chart_period"] = "1y"
            st.rerun()

    with pb4:
        more_choice = st.selectbox(
            "More periods",
            options=["More...", "3mo", "6mo", "2y", "5y"],
            index=0,
            label_visibility="collapsed",
            key="more_period"
        )
        if more_choice != "More...":
            st.session_state["chart_period"] = more_choice
            st.rerun()

    current = st.session_state["chart_period"]
    st.markdown(
        f"<div style='font-family:monospace;font-size:11px;color:#555;margin-bottom:8px'>"
        f"Selected: <span style='color:#FFD700'>{PERIOD_LABELS.get(current, current)}</span></div>",
        unsafe_allow_html=True)

    if ticker:
        with st.spinner(f"Building chart for {ticker} — {PERIOD_LABELS.get(current, current)}..."):
            try:
                st.plotly_chart(build_chart(ticker, current), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

with t6:
    if st.button("Refresh",key="rm"): st.cache_data.clear()
    with st.spinner("Building macro dashboard..."):
        fig=build_macro(); st.pyplot(fig,use_container_width=True); plt.close(fig)

with t7:
    ntk = st.text_input("Ticker", value="SPY", key="nt").upper()
    if ntk:
        q   = batch_quotes((ntk,)).get(ntk, dict(price=0,chg=0,pct=0))
        col = "#00FF41" if q["chg"] >= 0 else "#FF3333"
        sgn = "▲" if q["chg"] >= 0 else "▼"
        st.markdown(
            f"<div style='font-family:monospace;padding:8px 0'>"
            f"<span style='color:#FFD700;font-size:22px;font-weight:bold'>{ntk}</span>&nbsp;&nbsp;"
            f"<span style='color:{col};font-size:18px'>{fp(q['price'])} {sgn} {fc(q['chg'])} ({q['pct']:+.2f}%)</span></div>",
            unsafe_allow_html=True)
        with st.spinner(f"Loading news for {ntk}..."):
            news = get_news(ntk)
            rows = ""
            for item in news:
                try:
                    ct    = item.get("content", {})
                    title = ct.get("title", item.get("title", "No title"))
                    pub   = ct.get("pubDate", "")[:10]
                    prov  = ct.get("provider", {})
                    src   = prov.get("displayName", "") if isinstance(prov, dict) else ""
                    rows += (f"<tr style='border-bottom:1px solid #1a1a1a'>"
                             f"<td style='padding:8px 14px;color:#CCC;font-size:12px;line-height:1.5'>{title}</td>"
                             f"<td style='padding:8px 14px;white-space:nowrap;font-size:11px'>"
                             f"<span style='color:#555'>{pub}</span>"
                             + (f"<span style='color:#FF6600'> | {src}</span>" if src else "")
                             + "</td></tr>")
                except:
                    continue
            if rows:
                st.markdown(
                    f"<div style='background:#000;font-family:Courier New,monospace;padding:14px;"
                    f"border:1px solid #FF6600;border-radius:4px'>"
                    f"<table style='width:100%;border-collapse:collapse'>"
                    f"<thead><tr style='border-bottom:2px solid #FF6600'>"
                    f"<th style='color:#FF6600;text-align:left;padding:6px 14px'>HEADLINE</th>"
                    f"<th style='color:#FF6600;text-align:left;padding:6px 14px'>DATE / SOURCE</th>"
                    f"</tr></thead><tbody>{rows}</tbody></table></div>",
                    unsafe_allow_html=True)
            else:
                st.warning("No news available.")
