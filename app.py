import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=" Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #1e2130; border-radius: 12px; padding: 1rem 1.5rem;
        border: 1px solid #2d3250;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #0072ff; border-radius: 16px;
        padding: 1.5rem; text-align: center;
    }
    .up   { color: #00e676; font-weight: 700; }
    .down { color: #ff5252; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Helper: fetch data ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300)
def fetch_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return info
    except Exception:
        return {}


# ── Prophet prediction ─────────────────────────────────────────────────────────
def predict_prophet(df: pd.DataFrame, days_ahead: int = 1):
    try:
        from prophet import Prophet
    except ImportError:
        return None, None, "Prophet not installed."

    prophet_df = df[["Close"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    # prophet requires tz-naive timestamps
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,
    )
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=days_ahead, freq="B")  # business days
    forecast = m.predict(future)
    return m, forecast, None


# ── Linear Regression prediction ──────────────────────────────────────────────
def predict_linear(df: pd.DataFrame, days_ahead: int = 1):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler

    close = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    window = min(60, len(scaled) - 1)
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window : i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    model = LinearRegression()
    model.fit(X, y)

    last_window = scaled[-window:, 0].reshape(1, -1)
    preds_scaled = []
    current = last_window.copy()
    for _ in range(days_ahead):
        p = model.predict(current)[0]
        preds_scaled.append(p)
        current = np.append(current[:, 1:], [[p]], axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds


# ── Moving-average baseline ────────────────────────────────────────────────────
def predict_ma(df: pd.DataFrame, window: int = 20) -> float:
    return float(df["Close"].rolling(window).mean().iloc[-1])


# ── Technical indicators ───────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    bb_mid           = df["Close"].rolling(20).mean()
    bb_std           = df["Close"].rolling(20).std()
    df["BB_Upper"]   = bb_mid + 2 * bb_std
    df["BB_Lower"]   = bb_mid - 2 * bb_std
    df["BB_Mid"]     = bb_mid
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    exchange = st.selectbox(
        "Exchange / Market",
        ["🇺🇸 US (NYSE / NASDAQ)", "🇮🇳 India — NSE", "🇮🇳 India — BSE"],
    )
    suffix_map = {
        "🇺🇸 US (NYSE / NASDAQ)": "",
        "🇮🇳 India — NSE": ".NS",
        "🇮🇳 India — BSE": ".BO",
    }
    suffix = suffix_map[exchange]
    is_indian = suffix in (".NS", ".BO")

    default_ticker = "RELIANCE" if is_indian else "AAPL"
    raw_ticker = st.text_input(
        "Stock Symbol (without exchange suffix)",
        value=st.session_state.get("raw_ticker", default_ticker),
        max_chars=15,
        help="Enter the base symbol only — the exchange suffix is added automatically.",
    ).upper().strip()
    # Strip any manually typed suffix so we don't double-append
    for s in (".NS", ".BO"):
        if raw_ticker.endswith(s):
            raw_ticker = raw_ticker[: -len(s)]
    ticker = raw_ticker + suffix

    period_map = {
        "3 Months": "3mo", "6 Months": "6mo",
        "1 Year": "1y",   "2 Years": "2y", "5 Years": "5y",
    }
    period_label = st.selectbox("Historical Data Period", list(period_map.keys()), index=2)
    period       = period_map[period_label]

    model_choice = st.selectbox(
        "Prediction Model",
        ["Linear Regression (ML)", "Prophet (Time-Series)", "Both"],
        help="If Prophet fails to install, Linear Regression will still work."
    )

    days_ahead = st.slider("Days to Predict Ahead", 1, 30, 1)

    st.markdown("---")
    if is_indian:
        st.markdown("**Popular Indian Stocks**")
        cols = st.columns(3)
        popular_india = [
            "RELIANCE", "TCS", "INFY",
            "HDFCBANK", "ICICIBANK", "SBIN",
            "WIPRO",   "HINDUNILVR", "LT",
            "BAJFINANCE", "AXISBANK", "MARUTI",
        ]
        for i, t in enumerate(popular_india):
            if cols[i % 3].button(t, key=f"btn_{t}", use_container_width=True):
                raw_ticker = t
                ticker = raw_ticker + suffix
        st.caption(f"Suffix applied automatically: `{suffix}`")
    else:
        st.markdown("**Popular US Stocks**")
        cols = st.columns(3)
        popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BRK-B", "JPM", "V", "JNJ"]
        for i, t in enumerate(popular):
            if cols[i % 3].button(t, key=f"btn_{t}", use_container_width=True):
                raw_ticker = t
                ticker = raw_ticker

    st.markdown("---")
    run_btn = st.button("🔍 Analyze & Predict", type="primary", use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📈 Stock Price Predictor</div>', unsafe_allow_html=True)
st.markdown("AI-powered stock price forecasting using Prophet & Machine Learning")
st.divider()

if not run_btn and "last_ticker" not in st.session_state:
    st.info("👈 Enter a stock ticker in the sidebar and click **Analyze & Predict** to get started.")
    st.stop()

if run_btn:
    st.session_state["last_ticker"]  = ticker
    st.session_state["last_is_indian"] = is_indian
    st.session_state["raw_ticker"] = raw_ticker

ticker    = st.session_state.get("last_ticker", ticker)
is_indian = st.session_state.get("last_is_indian", is_indian)

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker}**..."):
    df   = fetch_data(ticker, period)
    info = fetch_info(ticker)

if df.empty:
    st.error(f"No data found for ticker `{ticker}`. Please check the symbol and try again.")
    st.stop()

df = add_indicators(df)

# ── Company Header ─────────────────────────────────────────────────────────────
company_name = info.get("longName", ticker)
sector       = info.get("sector", "N/A")
industry     = info.get("industry", "N/A")
exchange_name = info.get("exchange", "")

# Detect currency from yfinance info; fall back based on exchange
ccy_symbol = info.get("currency", "INR" if is_indian else "USD")
ccy_display = "₹" if ccy_symbol in ("INR", "GBp") else ("$" if ccy_symbol == "USD" else ccy_symbol + " ")

st.markdown(f"### {company_name} &nbsp; `{ticker}`")
st.caption(f"Exchange: **{exchange_name}** | Sector: **{sector}** | Industry: **{industry}** | Currency: **{ccy_symbol}**")

# ── KPI Metrics ───────────────────────────────────────────────────────────────
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2])
day_change = last_close - prev_close
day_pct    = (day_change / prev_close) * 100

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last Close",    f"{ccy_display}{last_close:,.2f}")
m2.metric("Day Change",    f"{ccy_display}{day_change:+,.2f}", f"{day_pct:+.2f}%")
m3.metric("52W High",      f"{ccy_display}{float(df['Close'].max()):,.2f}")
m4.metric("52W Low",       f"{ccy_display}{float(df['Close'].min()):,.2f}")
rsi_val = df["RSI"].iloc[-1]
m5.metric("RSI (14)",      f"{rsi_val:.1f}",
          delta="Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"),
          delta_color="inverse" if rsi_val > 70 else ("off" if rsi_val < 30 else "normal"))

st.divider()

# ── Run predictions ────────────────────────────────────────────────────────────
prophet_pred_tomorrow = None
lr_pred_tomorrow      = None
prophet_forecast      = None
prophet_error         = None

with st.spinner("Running predictions..."):
    use_prophet = model_choice in ["Prophet (Time-Series)", "Both"]
    use_lr      = model_choice in ["Linear Regression (ML)", "Both"]

    if use_prophet:
        _, prophet_forecast, prophet_error = predict_prophet(df, days_ahead=days_ahead + 30)
        if prophet_forecast is not None:
            prophet_pred_tomorrow = float(prophet_forecast["yhat"].iloc[-(days_ahead + 29)])

    if use_lr:
        lr_preds = predict_linear(df, days_ahead=days_ahead)
        lr_pred_tomorrow = float(lr_preds[days_ahead - 1])

# ── Prediction Results ─────────────────────────────────────────────────────────
label = "Tomorrow" if days_ahead == 1 else f"In {days_ahead} days"
st.markdown(f"### 🎯 Price Prediction — {label}")

pred_cols = []
if prophet_pred_tomorrow is not None:
    pred_cols.append(("Prophet", prophet_pred_tomorrow))
if lr_pred_tomorrow is not None:
    pred_cols.append(("Linear Regression", lr_pred_tomorrow))
if pred_cols:
    avg_pred = np.mean([v for _, v in pred_cols])
    pred_cols.append(("Ensemble Avg", avg_pred))

if pred_cols:
    pcols = st.columns(len(pred_cols))
    for i, (model_name, pred_val) in enumerate(pred_cols):
        chg     = pred_val - last_close
        chg_pct = (chg / last_close) * 100
        direction = "▲" if chg >= 0 else "▼"
        color     = "#00e676" if chg >= 0 else "#ff5252"
        with pcols[i]:
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size:0.85rem;color:#aaa;margin-bottom:0.4rem">{model_name}</div>
                <div style="font-size:2rem;font-weight:800;color:{color}">{ccy_display}{pred_val:,.2f}</div>
                <div style="font-size:1rem;color:{color}">{direction} {abs(chg):,.2f} ({chg_pct:+.2f}%)</div>
                <div style="font-size:0.75rem;color:#888;margin-top:0.3rem">vs Last Close {ccy_display}{last_close:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

if prophet_error:
    st.warning(
        f"⚠️ {prophet_error}\n\n"
        f"**Fix:** Run this in your terminal:\n"
        f"```\n"
        f"pip install pystan==2.19.1.1\n"
        f"pip install prophet\n"
        f"```\n\n"
        f"Or use conda (Windows): `conda install -c conda-forge prophet`\n\n"
        f"The app will work fine with **Linear Regression** in the meantime."
    )

st.divider()

# ── Charts ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Price & Forecast", "📉 Technical Indicators", "📋 Raw Data"])

# ── Tab 1: Price + Forecast ────────────────────────────────────────────────────
with tab1:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        name="Price", increasing_line_color="#00e676",
        decreasing_line_color="#ff5252",
    ), row=1, col=1)

    # Moving averages
    for ma, color in [("MA20", "#ffd54f"), ("MA50", "#81d4fa"), ("MA200", "#ce93d8")]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ma], name=ma,
            line=dict(color=color, width=1.2), opacity=0.85,
        ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Upper"], name="BB Upper",
        line=dict(color="rgba(100,181,246,0.4)", dash="dot", width=1),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Lower"], name="BB Lower",
        fill="tonexty", fillcolor="rgba(100,181,246,0.05)",
        line=dict(color="rgba(100,181,246,0.4)", dash="dot", width=1),
    ), row=1, col=1)

    # Prophet forecast band
    if prophet_forecast is not None:
        hist_end = df.index[-1]
        if hasattr(hist_end, "tzinfo") and hist_end.tzinfo is not None:
            hist_end_naive = hist_end.tz_localize(None)
        else:
            hist_end_naive = hist_end
        fcast_future = prophet_forecast[
            prophet_forecast["ds"] > pd.Timestamp(hist_end_naive)
        ]
        if not fcast_future.empty:
            fig.add_trace(go.Scatter(
                x=fcast_future["ds"], y=fcast_future["yhat_upper"],
                name="Prophet Upper", line=dict(color="rgba(0,114,255,0)", width=0),
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=fcast_future["ds"], y=fcast_future["yhat_lower"],
                name="Forecast Range",
                fill="tonexty", fillcolor="rgba(0,114,255,0.15)",
                line=dict(color="rgba(0,114,255,0)", width=0),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=fcast_future["ds"], y=fcast_future["yhat"],
                name="Prophet Forecast",
                line=dict(color="#0072ff", width=2, dash="dash"),
            ), row=1, col=1)

    # Linear regression future points
    if lr_pred_tomorrow is not None:
        future_dates = pd.bdate_range(start=df.index[-1], periods=days_ahead + 1)[1:]
        lr_preds_all = predict_linear(df, days_ahead=days_ahead)
        fig.add_trace(go.Scatter(
            x=future_dates, y=lr_preds_all,
            name="LR Prediction",
            mode="markers+lines",
            marker=dict(color="#ff6d00", size=8, symbol="diamond"),
            line=dict(color="#ff6d00", width=2, dash="dot"),
        ), row=1, col=1)

    # Volume
    colors_vol = ["#00e676" if c >= o else "#ff5252"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors_vol, opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text=f"Price ({ccy_symbol})", row=1, col=1, gridcolor="#1e2130")
    fig.update_yaxes(title_text="Volume",      row=2, col=1, gridcolor="#1e2130")
    fig.update_xaxes(gridcolor="#1e2130")
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Technical Indicators ────────────────────────────────────────────────
with tab2:
    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("RSI (14)", "MACD"),
        vertical_spacing=0.12,
    )

    # RSI
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#ce93d8", width=1.5),
    ), row=1, col=1)
    fig2.add_hline(y=70, line_dash="dot", line_color="#ff5252", row=1, col=1)
    fig2.add_hline(y=30, line_dash="dot", line_color="#00e676", row=1, col=1)
    fig2.add_hrect(y0=70, y1=100, fillcolor="rgba(255,82,82,0.05)", line_width=0, row=1, col=1)
    fig2.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,230,118,0.05)", line_width=0, row=1, col=1)

    # MACD
    macd_colors = ["#00e676" if v >= 0 else "#ff5252"
                   for v in (df["MACD"] - df["MACD_Signal"])]
    fig2.add_trace(go.Bar(
        x=df.index, y=df["MACD"] - df["MACD_Signal"],
        name="MACD Histogram", marker_color=macd_colors, opacity=0.7,
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], name="MACD",
        line=dict(color="#0072ff", width=1.5),
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"], name="Signal",
        line=dict(color="#ff6d00", width=1.5),
    ), row=2, col=1)

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig2.update_xaxes(gridcolor="#1e2130")
    fig2.update_yaxes(gridcolor="#1e2130")
    st.plotly_chart(fig2, use_container_width=True)

    # RSI signal
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### RSI Signal")
        if rsi_val > 70:
            st.error(f"⚠️ Overbought (RSI = {rsi_val:.1f}) — potential pullback")
        elif rsi_val < 30:
            st.success(f"✅ Oversold (RSI = {rsi_val:.1f}) — potential bounce")
        else:
            st.info(f"ℹ️ Neutral (RSI = {rsi_val:.1f})")

    with col_b:
        st.markdown("#### MACD Signal")
        macd_last   = float(df["MACD"].iloc[-1])
        signal_last = float(df["MACD_Signal"].iloc[-1])
        if macd_last > signal_last:
            st.success("✅ Bullish crossover — MACD above Signal line")
        else:
            st.error("⚠️ Bearish crossover — MACD below Signal line")


# ── Tab 3: Raw Data ────────────────────────────────────────────────────────────
with tab3:
    display_df = df[["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI"]].copy()
    display_df = display_df.tail(100).sort_index(ascending=False)
    display_df = display_df.round(2)
    st.dataframe(display_df, use_container_width=True, height=500)

    csv = display_df.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Download Data as CSV",
        data=csv,
        file_name=f"{ticker}_stock_data.csv",
        mime="text/csv",
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: This app is for educational purposes only. "
    "Stock predictions are inherently uncertain and should NOT be used as financial advice. "
    "Always consult a qualified financial advisor before making investment decisions."
)
