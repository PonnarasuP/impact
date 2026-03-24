import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


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


def response(handler: BaseHTTPRequestHandler, status: int, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return response(self, 400, {"error": "Missing JSON body."})

            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)

            symbol = str(payload.get("symbol", "")).strip().upper()
            exchange = str(payload.get("exchange", "US")).strip().upper()
            period = str(payload.get("period", "1y")).strip()
            days_ahead = int(payload.get("days_ahead", 1))

            if not symbol:
                return response(self, 400, {"error": "symbol is required."})
            if exchange not in ("US", "NSE", "BSE"):
                return response(self, 400, {"error": "exchange must be one of US/NSE/BSE."})
            if period not in ("3mo", "6mo", "1y", "2y", "5y"):
                return response(self, 400, {"error": "period must be one of 3mo/6mo/1y/2y/5y."})
            if days_ahead < 1 or days_ahead > 30:
                return response(self, 400, {"error": "days_ahead must be between 1 and 30."})

            ticker = normalize_ticker(symbol, exchange)
            data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if data is None or data.empty:
                return response(self, 404, {"error": f"No data found for ticker: {ticker}"})

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            close = data.get("Close")
            if close is None:
                return response(self, 500, {"error": "Close price not present in source data."})

            close = close.dropna()
            if len(close) < 20:
                return response(self, 400, {"error": "Not enough data points for prediction."})

            predicted_close = predict_linear_from_close(close, days_ahead)
            last_close = float(close.iloc[-1])
            change = predicted_close - last_close
            change_pct = (change / last_close) * 100 if last_close else 0.0
            currency = "INR" if exchange in ("NSE", "BSE") else "USD"

            return response(
                self,
                200,
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "fetched_ticker": ticker,
                    "currency": currency,
                    "last_close": round(last_close, 4),
                    "predicted_close": round(predicted_close, 4),
                    "change": round(change, 4),
                    "change_pct": round(change_pct, 4),
                    "generated_at_utc": datetime.utcnow().isoformat() + "Z",
                },
            )

        except ValueError as exc:
            return response(self, 400, {"error": str(exc)})
        except Exception as exc:
            return response(self, 500, {"error": f"Internal error: {str(exc)}"})
