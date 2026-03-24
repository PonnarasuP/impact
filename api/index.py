from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(title="Stock Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    exchange: str = Field("US", pattern="^(US|NSE|BSE)$")
    period: str = Field("1y", pattern="^(3mo|6mo|1y|2y|5y)$")
    days_ahead: int = Field(1, ge=1, le=30)


class PredictResponse(BaseModel):
    symbol: str
    exchange: str
    fetched_ticker: str
    currency: str
    last_close: float
    predicted_close: float
    change: float
    change_pct: float
    generated_at_utc: str


def normalize_ticker(symbol: str, exchange: str) -> str:
    cleaned = symbol.strip().upper().replace(" ", "")
    for suffix in (".NS", ".BO"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]

    if exchange == "NSE":
        return f"{cleaned}.NS"
    if exchange == "BSE":
        return f"{cleaned}.BO"
    return cleaned


def predict_linear_from_close(close_series: pd.Series, days_ahead: int) -> float:
    close = close_series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    window = min(60, len(scaled) - 1)
    if window < 2:
        raise ValueError("Not enough data points to build a model.")

    x_vals, y_vals = [], []
    for i in range(window, len(scaled)):
        x_vals.append(scaled[i - window : i, 0])
        y_vals.append(scaled[i, 0])

    x_arr, y_arr = np.array(x_vals), np.array(y_vals)
    if len(x_arr) == 0:
        raise ValueError("Not enough history to train the model.")

    model = LinearRegression()
    model.fit(x_arr, y_arr)

    current = scaled[-window:, 0].reshape(1, -1)
    future_scaled = []
    for _ in range(days_ahead):
        pred = model.predict(current)[0]
        future_scaled.append(pred)
        current = np.append(current[:, 1:], [[pred]], axis=1)

    predicted = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()[-1]
    return float(predicted)


@app.get("/")
def root() -> dict:
    return {
        "service": "stock-predictor-api",
        "status": "ok",
        "docs": "/api/docs",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    ticker = normalize_ticker(payload.symbol, payload.exchange)

    data = yf.download(ticker, period=payload.period, auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        raise HTTPException(status_code=500, detail="Close price not present in source data.")

    close = data["Close"].dropna()
    if len(close) < 20:
        raise HTTPException(status_code=400, detail="Not enough data points for prediction.")

    try:
        predicted_close = predict_linear_from_close(close, payload.days_ahead)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    last_close = float(close.iloc[-1])
    change = predicted_close - last_close
    change_pct = (change / last_close) * 100 if last_close else 0.0

    currency = "INR" if payload.exchange in ("NSE", "BSE") else "USD"

    return PredictResponse(
        symbol=payload.symbol.upper(),
        exchange=payload.exchange,
        fetched_ticker=ticker,
        currency=currency,
        last_close=round(last_close, 4),
        predicted_close=round(predicted_close, 4),
        change=round(change, 4),
        change_pct=round(change_pct, 4),
        generated_at_utc=datetime.utcnow().isoformat() + "Z",
    )
