"""
prometheus_metrics.py — Prometheus metrics exporter
Zaalima Development · Project 2 · Computer Vision

Exports model latency, throughput, and defect detection
counts as Prometheus-compatible metrics.

Usage:
    python src/prometheus_metrics.py   # starts metrics server on :9090
"""

import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

_metrics = {
    "requests_total": 0,
    "inference_seconds_sum": 0.0,
    "detections_total": {
        "crazing": 0, "inclusion": 0, "patches": 0,
        "pitted_surface": 0, "rolled-in_scale": 0, "scratches": 0,
    },
    "model_load_time_seconds": 0.0,
    "start_time": time.time(),
}


def record_inference(latency_s: float, detections: list):
    _metrics["requests_total"] += 1
    _metrics["inference_seconds_sum"] += latency_s
    for det in detections:
        cls = det.get("class_name", "unknown")
        if cls in _metrics["detections_total"]:
            _metrics["detections_total"][cls] += 1


def format_prometheus() -> str:
    uptime = time.time() - _metrics["start_time"]
    avg_lat = (
        _metrics["inference_seconds_sum"] / _metrics["requests_total"]
        if _metrics["requests_total"] > 0 else 0
    )
    lines = [
        "# HELP defect_requests_total Total inference requests",
        "# TYPE defect_requests_total counter",
        f'defect_requests_total {_metrics["requests_total"]}',
        "",
        "# HELP defect_inference_latency_avg_seconds Average inference latency",
        "# TYPE defect_inference_latency_avg_seconds gauge",
        f"defect_inference_latency_avg_seconds {avg_lat:.6f}",
        "",
        "# HELP defect_uptime_seconds Service uptime",
        "# TYPE defect_uptime_seconds counter",
        f"defect_uptime_seconds {uptime:.1f}",
        "",
        "# HELP defect_detections_total Detections per class",
        "# TYPE defect_detections_total counter",
    ]
    for cls, count in _metrics["detections_total"].items():
        lines.append(f'defect_detections_total{{class="{cls}"}} {count}')
    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = format_prometheus().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # silence


def start_metrics_server(port: int = 9090):
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"📊 Prometheus metrics available at http://localhost:{port}/metrics")
    return server


if __name__ == "__main__":
    server = start_metrics_server()
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()
