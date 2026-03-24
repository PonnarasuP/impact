import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = {
            "service": "stock-predictor-api",
            "status": "ok",
            "time_utc": datetime.utcnow().isoformat() + "Z",
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
