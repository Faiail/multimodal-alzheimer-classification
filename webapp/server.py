"""
Multimodal Alzheimer's Disease Classification Web Server.
Serves the Interactive Web Application, REST APIs for 5,002 patient scans,
on-the-fly 3D slice rendering (Axial, Coronal, Sagittal for MRI/PET/Fusion),
and multi-model prediction inference across all 11 tested architectures.
"""

import json
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.data_manager import DataManager
from webapp.prediction_engine import PredictionEngine

# Initialize core services
data_mgr = DataManager(PROJECT_ROOT)
pred_engine = PredictionEngine(PROJECT_ROOT)


class WebAppHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for local cross-origin development and iframe embedding
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # ----------------------------------------------------
        # API Routes
        # ----------------------------------------------------
        if path == "/api/stats":
            self._send_json(data_mgr.get_stats())
            return

        elif path == "/api/models":
            self._send_json(pred_engine.get_models_overview())
            return

        elif path == "/api/showcase":
            showcase = data_mgr.get_curated_showcase()
            for item in showcase:
                meta_row = data_mgr.df[data_mgr.df["scan_id"] == item["scan_id"]].iloc[0]
                item["predictions_summary"] = pred_engine.compute_scan_predictions(meta_row)
            self._send_json(showcase)
            return

        elif path == "/api/scans":
            query = params.get("q", [""])[0]
            diagnosis = params.get("diagnosis", ["ALL"])[0]
            modality = params.get("modality", ["ALL"])[0]
            limit = int(params.get("limit", [50])[0])
            offset = int(params.get("offset", [0])[0])

            res = data_mgr.filter_scans(
                query=query, diagnosis=diagnosis, modality=modality, limit=limit, offset=offset
            )
            self._send_json(res)
            return

        elif path.startswith("/api/scan/"):
            subpath = path[len("/api/scan/") :]

            # Check if slice rendering endpoint: /api/scan/<id>/slice
            if "/slice" in subpath:
                scan_id = subpath.split("/slice")[0]
                scan_meta = data_mgr.get_scan_by_id(scan_id)
                if not scan_meta:
                    self._send_error_json(f"Scan '{scan_id}' not found", HTTPStatus.NOT_FOUND)
                    return

                safetensor_file = scan_meta["safetensor_file"]
                modality = params.get("modality", ["mri"])[0]
                plane = params.get("plane", ["axial"])[0]
                slice_idx = int(params.get("slice_idx", [64])[0])
                colormap = params.get("colormap", ["inferno"])[0]
                alpha = float(params.get("alpha", [0.5])[0])
                window = float(params.get("window", [1.0])[0])
                level = float(params.get("level", [0.0])[0])

                img_bytes = data_mgr.render_slice(
                    safetensor_file=safetensor_file,
                    modality=modality,
                    plane=plane,
                    slice_idx=slice_idx,
                    colormap=colormap,
                    alpha=alpha,
                    window=window,
                    level=level,
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(img_bytes)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(img_bytes)
                return

            else:
                # Get detailed scan info and model predictions
                scan_id = subpath
                scan_meta = data_mgr.get_scan_by_id(scan_id)
                if not scan_meta:
                    self._send_error_json(f"Scan '{scan_id}' not found", HTTPStatus.NOT_FOUND)
                    return

                meta_row = data_mgr.df[data_mgr.df["scan_id"] == scan_meta["scan_id"]].iloc[0]
                preds = pred_engine.compute_scan_predictions(meta_row)
                full_response = {**scan_meta, **preds}
                self._send_json(full_response)
                return

        # ----------------------------------------------------
        # Static Webapp UI Route
        # ----------------------------------------------------
        elif path == "/" or path == "/index.html":
            index_path = PROJECT_ROOT / "webapp/index.html"
            if index_path.exists():
                content = index_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        # Fallback to standard static file serving from webapp directory
        self.directory = str(PROJECT_ROOT / "webapp")
        super().do_GET()

    def _send_json(self, data: Any, status: int = HTTPStatus.OK):
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, msg: str, status: int):
        self._send_json({"error": msg, "status": status}, status)


def run_server(port: int = 8050, host: str = "0.0.0.0"):
    server_addr = (host, port)
    server = ThreadingHTTPServer(server_addr, WebAppHandler)
    print("=" * 65)
    print(f" Multimodal Alzheimer's Classification Web Application")
    print(f" URL:  http://localhost:{port}")
    print(f" Serving 5,002 scans & 11 evaluated models")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal Alzheimer Classification Web Server")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind server (default: 8050)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)
