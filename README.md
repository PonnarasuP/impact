# 📈 Stock Price Predictor

An AI-powered web app for predicting tomorrow's stock prices using machine learning & time-series forecasting.

## Features

✅ **Multiple Prediction Models**
- Prophet (Facebook) — time-series decomposition with trends & seasonality  
- Linear Regression — ML sliding-window regression
- Ensemble — average of both models

✅ **Global Stock Support**
- 🇺🇸 US markets (NYSE, NASDAQ, OTC)
- 🇮🇳 Indian markets (NSE, BSE)
- Any ticker supported by yfinance

✅ **Rich Visualizations**
- Interactive candlestick charts with Bollinger Bands
- Technical indicators (RSI, MACD)
- Forecast confidence bands
- Volume analysis

✅ **Customizable Predictions**
- Predict 1–30 days ahead
- Choose 3mo to 5y historical data
- 12 popular stock quick-picks per market

---

## Installation

### Quick Start

**Step 1:** Install Python 3.9+ if you don't have it  
https://www.python.org/downloads/

**Step 2:** Install dependencies

```bash
pip install -r requirements.txt
```

If Prophet fails on Windows (common issue), run:

```bash
pip install pystan==2.19.1.1
pip install prophet
```

**Alternative for Windows** — use Conda (recommended for Prophet):

```bash
conda install -c conda-forge prophet
pip install streamlit yfinance plotly pandas scikit-learn
```

**Step 3:** Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## Troubleshooting

### Error: "Prophet not installed"

Prophet can be tricky on Windows due to C++ compiler requirements.

**Option A:** Skip Prophet (app still works with Linear Regression)
- The error message shows commands to install it
- Or just use "Linear Regression (ML)" or toggle between them

**Option B:** Install via Conda (easiest)
```bash
conda install -c conda-forge prophet
```

**Option C:** Install dependencies manually
```bash
pip install pystan==2.19.1.1
pip install prophet==1.1.5
```

### Error: "No data found for ticker..."

The ticker symbol wasn't found. Make sure:
- For Indian stocks: select the exchange (NSE/BSE) first — it auto-adds the suffix
- For US stocks: use standard symbols like AAPL, MSFT, GOOGL
- Ticker symbol exists and trades on the selected exchange

### Error: "ModuleNotFoundError: No module named 'streamlit'"

Run: `pip install -r requirements.txt`

---

## Popular Ticker Examples

**US Stocks:** AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, JNJ  
**Indian NSE:** RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, WIPRO, LT  
**Indian BSE:** Same symbols as NSE, just select BSE in the app

---

## How It Works

### Prophet Model
- **Method:** Additive time-series decomposition (trend + yearly/weekly seasonality)
- **Pros:** Handles holidays, trend changes, and seasonality well
- **Cons:** Requires C++ compiler to install; slower on large datasets

### Linear Regression Model
- **Method:** 60-day sliding window → normalized → linear fit → future projection
- **Pros:** Fast, reliable, always works
- **Cons:** Assumes linear trends; less accurate for volatile stocks

### Ensemble
- **Result:** Average of both model predictions

---

## Important Disclaimer

⚠️ **Educational Use Only**

This app is a demonstration of machine learning for stock price forecasting. Stock market predictions are inherently uncertain and **should NOT be used as financial advice**.

**Always:**
- Do your own research
- Consult a qualified financial advisor
- Never invest more than you can afford to lose
- Consider the broader market context, news, and fundamentals

Past performance ≠ future results.

---

## Technical Stack

- **Frontend:** Streamlit (web UI)
- **Data:** yfinance (stock prices)
- **Forecasting:** Prophet, scikit-learn  
- **Visualization:** Plotly
- **Data:** Pandas, NumPy

---

## Project Structure

```
c:\Project_Work\General_works\AI-Work\impact\
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── install_deps.bat      # Windows installer script
├── run_app.bat           # Windows launcher script
└── .gitignore            # (optional) Git ignore file
```

---

## Next Steps / Ideas

- [ ] Add LSTM deep learning model
- [ ] Support for intraday predictions (hourly data)
- [ ] Volatility forecasting (Bollinger Band width)
- [ ] Portfolio tracking (multiple stocks)
- [ ] Email alerts for price predictions
- [ ] Database storage of historical predictions
- [ ] Fine-tuning model parameters via UI

---

## License

Created for educational purposes. Feel free to modify and extend!

---

**Questions?** Check the app's sidebar help buttons or consult the yfinance documentation.
