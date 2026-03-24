"""
live_viewer.py — Web-based live camera viewer + classification display
Runs on: Laptop (Mac)

Purpose:
    Receives raw camera frames and classification results from the ESP,
    serves a web page at http://localhost:9090 that shows:
    1. Live camera feed (auto-refreshing)
    2. Current classification result with confidence
    3. Preprocessed 32x32 view (what the CNN sees)

Usage:
    1. Run: python live_viewer.py
    2. Open browser: http://localhost:9090
    3. Run live_stream_esp.py on the ESP

Requirements:
    pip install Pillow numpy

Original code by project team.
"""

import socket
import struct
import threading
import os
import io
import time
import base64
import numpy as np
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler

# ─── Configuration ───
ESP_LISTEN_PORT = 8080   # receive frames from ESP
WEB_PORT = 9090          # serve web page here
THRESHOLD = 128          # must match ESP preprocessing

# ─── Shared state (thread-safe via GIL) ───
latest_frame_b64 = ""           # base64-encoded JPEG of raw camera frame
latest_processed_b64 = ""       # base64-encoded JPEG of 32x32 preprocessed
latest_classification = "waiting..."
latest_confidence = 0.0
latest_probs = {}
latest_inference_ms = 0
latest_status = "waiting"       # "waiting", "capturing", "classifying", "done"
frame_count = 0
classify_count = 0


def process_raw_bmp(data):
    """Convert raw camera BMP to displayable JPEG + preprocessed 32x32 view."""
    global latest_frame_b64, latest_processed_b64

    try:
        img = Image.open(io.BytesIO(bytes(data)))

        # Raw frame → JPEG for browser display (upscale for visibility)
        raw_display = img.resize((256, 256), Image.NEAREST)
        buf = io.BytesIO()
        raw_display.save(buf, format="JPEG", quality=80)
        latest_frame_b64 = base64.b64encode(buf.getvalue()).decode()

        # Preprocessed view: convert to grayscale, resize to 32x32, threshold
        gray = img.convert("L")
        small = gray.resize((32, 32), Image.NEAREST)
        arr = np.array(small)
        arr = np.where(arr >= THRESHOLD, 255, 0).astype(np.uint8)
        processed = Image.fromarray(arr, mode="L")

        # Upscale for visibility
        proc_display = processed.resize((256, 256), Image.NEAREST)
        buf2 = io.BytesIO()
        proc_display.save(buf2, format="JPEG", quality=80)
        latest_processed_b64 = base64.b64encode(buf2.getvalue()).decode()

    except Exception as e:
        print(f"[ERR]  Frame processing: {e}")


def process_classification(data):
    """Parse classification result from ESP."""
    global latest_classification, latest_confidence, latest_probs, latest_inference_ms, classify_count

    try:
        text = data.decode()
        parts = text.split("|")
        latest_classification = parts[0]
        latest_confidence = float(parts[1])
        # Parse individual probabilities
        if len(parts) > 2:
            for pair in parts[2].split(","):
                name, val = pair.split(":")
                latest_probs[name] = float(val)
        if len(parts) > 3:
            latest_inference_ms = int(parts[3])
        classify_count += 1
    except Exception as e:
        print(f"[ERR]  Classification parse: {e}")


# ─── ESP Receiver Thread ───
def esp_receiver():
    """Listen for incoming frames and classifications from ESP."""
    global frame_count, latest_status

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", ESP_LISTEN_PORT))
    server.listen(5)
    server.settimeout(1.0)

    print(f"[OK]   ESP listener on port {ESP_LISTEN_PORT}")

    while True:
        try:
            client, addr = server.accept()
            client.settimeout(10)

            # Read header: 1 byte type + 4 byte length
            header = b""
            while len(header) < 5:
                chunk = client.recv(5 - len(header))
                if not chunk:
                    break
                header += chunk

            if len(header) < 5:
                client.close()
                continue

            msg_type = header[0]
            payload_len = struct.unpack('<I', header[1:5])[0]

            # Read payload
            data = b""
            while len(data) < payload_len:
                chunk = client.recv(min(4096, payload_len - len(data)))
                if not chunk:
                    break
                data += chunk
            client.close()

            if msg_type == 0x01:
                # Raw camera frame just arrived — ESP is now classifying
                frame_count += 1
                latest_status = "classifying"
                process_raw_bmp(data)
                if frame_count % 10 == 1:
                    print(f"[FRAME] #{frame_count} received ({len(data)} bytes)")
            elif msg_type == 0x02:
                # Classification result ready
                process_classification(data)
                latest_status = "done"
                print(f"[CLASS] {latest_classification} ({latest_confidence:.0%}) | {latest_inference_ms}ms")
            elif msg_type == 0x03:
                # Status update (kept for compatibility)
                latest_status = data.decode()

        except socket.timeout:
            continue
        except Exception as e:
            if "Errno 54" not in str(e):  # suppress connection reset
                print(f"[ERR]  Receiver: {e}")


# ─── Web Server ───
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RPS Live Classifier</title>
    <style>
        body {
            background: #1a1a2e; color: #eee; font-family: 'Courier New', monospace;
            margin: 0; padding: 20px; text-align: center;
        }
        h1 { color: #e94560; margin-bottom: 5px; }
        .container { display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; }
        .panel {
            background: #16213e; border-radius: 12px; padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .panel h2 { color: #0f3460; margin-top: 0; }
        img { border-radius: 8px; border: 2px solid #0f3460; }
        .result {
            font-size: 48px; font-weight: bold; margin: 20px 0;
            text-transform: uppercase; letter-spacing: 3px;
        }
        .rock { color: #e94560; }
        .paper { color: #4ecca3; }
        .scissors { color: #f9a825; }
        .waiting { color: #666; }
        .confidence { font-size: 24px; color: #aaa; }
        .bar-container { width: 300px; margin: 8px auto; text-align: left; }
        .bar-label { display: inline-block; width: 80px; }
        .bar-outer {
            display: inline-block; width: 180px; height: 20px;
            background: #0f3460; border-radius: 4px; vertical-align: middle;
        }
        .bar-inner { height: 100%; border-radius: 4px; transition: width 0.3s; }
        .bar-rock { background: #e94560; }
        .bar-paper { background: #4ecca3; }
        .bar-scissors { background: #f9a825; }
        .bar-pct { display: inline-block; width: 45px; text-align: right; }
        .stats { color: #666; font-size: 14px; margin-top: 15px; }
        .help { color: #555; font-size: 12px; max-width: 600px; margin: 15px auto; }

        /* Status indicator */
        .status-bar {
            display: inline-block; padding: 8px 24px; border-radius: 20px;
            font-size: 16px; font-weight: bold; letter-spacing: 2px;
            margin-bottom: 15px; transition: all 0.3s;
        }
        .status-waiting { background: #333; color: #666; }
        .status-capturing {
            background: #1b4332; color: #40c057;
            animation: pulse 0.8s infinite alternate;
        }
        .status-classifying {
            background: #3d1f00; color: #f9a825;
            animation: pulse 0.6s infinite alternate;
        }
        .status-done { background: #16213e; color: #4ecca3; }
        @keyframes pulse { from { opacity: 1; } to { opacity: 0.5; } }

        /* Camera border glow on capture */
        .cam-capturing { border-color: #40c057 !important; box-shadow: 0 0 15px #40c057; }
        .cam-classifying { border-color: #f9a825 !important; box-shadow: 0 0 15px #f9a825; }
    </style>
</head>
<body>
    <h1>Rock Paper Scissors - Live Classifier</h1>
    <p style="color:#666">XIAO ESP32S3 Sense - Single-shot CNN Inference</p>

    <div class="status-bar status-waiting" id="status_badge">WAITING FOR ESP</div>

    <div class="container">
        <div class="panel">
            <h2 style="color:#4ecca3">Camera Feed</h2>
            <img id="raw" width="256" height="256" alt="Waiting for ESP...">
            <div class="stats">Live from ESP camera</div>
        </div>
        <div class="panel">
            <h2 style="color:#f9a825">CNN Input (32x32)</h2>
            <img id="proc" width="256" height="256" alt="Waiting...">
            <div class="stats">What the neural network sees</div>
        </div>
    </div>

    <div class="panel" style="display:inline-block; margin-top:20px; min-width:400px;">
        <div class="result" id="class_name">waiting...</div>
        <div class="confidence" id="conf_text"></div>
        <div style="margin-top:15px">
            <div class="bar-container">
                <span class="bar-label">rock</span>
                <span class="bar-outer"><span class="bar-inner bar-rock" id="bar_rock" style="width:0%"></span></span>
                <span class="bar-pct" id="pct_rock">0%</span>
            </div>
            <div class="bar-container">
                <span class="bar-label">paper</span>
                <span class="bar-outer"><span class="bar-inner bar-paper" id="bar_paper" style="width:0%"></span></span>
                <span class="bar-pct" id="pct_paper">0%</span>
            </div>
            <div class="bar-container">
                <span class="bar-label">scissors</span>
                <span class="bar-outer"><span class="bar-inner bar-scissors" id="bar_scissors" style="width:0%"></span></span>
                <span class="bar-pct" id="pct_scissors">0%</span>
            </div>
        </div>
        <div class="stats" id="stats_text">Waiting for ESP connection...</div>
    </div>

    <div class="help">
        Center your hand in the camera feed above.<br>
        The status badge shows: CAPTURING (green) → CLASSIFYING (yellow) → result.
    </div>

    <script>
        function update() {
            fetch('/status')
                .then(r => r.json())
                .then(d => {
                    if (d.raw_frame) {
                        document.getElementById('raw').src = 'data:image/jpeg;base64,' + d.raw_frame;
                    }
                    if (d.processed_frame) {
                        document.getElementById('proc').src = 'data:image/jpeg;base64,' + d.processed_frame;
                    }

                    // Update status badge
                    let badge = document.getElementById('status_badge');
                    let rawImg = document.getElementById('raw');
                    rawImg.className = '';
                    if (d.status === 'capturing') {
                        badge.textContent = 'CAPTURING';
                        badge.className = 'status-bar status-capturing';
                        rawImg.className = 'cam-capturing';
                    } else if (d.status === 'classifying') {
                        badge.textContent = 'CLASSIFYING...';
                        badge.className = 'status-bar status-classifying';
                        rawImg.className = 'cam-classifying';
                    } else if (d.status === 'done') {
                        badge.textContent = 'RESULT READY';
                        badge.className = 'status-bar status-done';
                    } else {
                        badge.textContent = 'WAITING FOR ESP';
                        badge.className = 'status-bar status-waiting';
                    }

                    let el = document.getElementById('class_name');
                    el.textContent = d.classification || 'waiting...';
                    el.className = 'result ' + (d.classification || 'waiting');

                    document.getElementById('conf_text').textContent =
                        d.confidence > 0 ? Math.round(d.confidence * 100) + '% confidence' : '';

                    for (let cls of ['rock', 'paper', 'scissors']) {
                        let pct = Math.round((d.probs[cls] || 0) * 100);
                        document.getElementById('bar_' + cls).style.width = pct + '%';
                        document.getElementById('pct_' + cls).textContent = pct + '%';
                    }

                    document.getElementById('stats_text').textContent =
                        'Frames: ' + d.frame_count + ' | Classifications: ' + d.classify_count +
                        ' | Inference: ' + d.inference_ms + 'ms';
                })
                .catch(e => {});
            setTimeout(update, 300);
        }
        update();
    </script>
</body>
</html>"""


class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/status":
            data = {
                "raw_frame": latest_frame_b64,
                "processed_frame": latest_processed_b64,
                "classification": latest_classification,
                "confidence": latest_confidence,
                "probs": latest_probs,
                "inference_ms": latest_inference_ms,
                "status": latest_status,
                "frame_count": frame_count,
                "classify_count": classify_count,
            }
            import json
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress HTTP access logs


# ─── Main ───
if __name__ == "__main__":
    print("=" * 50)
    print("  RPS Live Viewer")
    print("=" * 50)
    print(f"  ESP listener:  port {ESP_LISTEN_PORT}")
    print(f"  Web viewer:    http://localhost:{WEB_PORT}")
    print(f"  Open the URL above in your browser!")
    print("=" * 50)

    # Start ESP receiver in background thread
    t = threading.Thread(target=esp_receiver, daemon=True)
    t.start()

    # Start web server (main thread)
    httpd = HTTPServer(("0.0.0.0", WEB_PORT), WebHandler)
    print(f"[OK]   Web server running on http://localhost:{WEB_PORT}")
    print(f"[INFO] Now run live_stream_esp.py on the ESP")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Server stopped")
        httpd.server_close()
