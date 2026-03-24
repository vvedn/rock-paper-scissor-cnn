"""
live_stream_esp.py — Live camera stream + CNN classification
Runs on: ESP32S3 (XIAO ESP32S3 Sense)

Purpose:
    Streams camera frames to the laptop web server AND runs CNN inference.
    Uses majority voting (3 captures) to reduce noise misclassification.
    Sends both raw camera frames (for live preview) and classification results.

Adapted from:
    - Camera init: camera_test.py (course-provided, pin config)
    - WiFi: Wifi.py (Sharil Tumin, MIT License)
    - CNN inference: final_submission.py (project code)

Protocol (ESP → Laptop):
    Each message: [1 byte type][4 byte length][payload]
    Type 0x01 = raw camera BMP frame (for live preview)
    Type 0x02 = classification result (JSON string)
"""

import gc
import socket
import struct
from time import sleep, ticks_ms, ticks_diff

# ─── Configuration ───
SSID = b'V\xe2\x80\x99s iPhone'
PASSWORD = "temppwdesp"
LAPTOP_IP = "172.20.10.2"
LAPTOP_PORT = 8080
STREAM_INTERVAL = 0  # seconds between frames (0 = as fast as possible)

# ─── Step 1: Connect to WiFi ───
gc.collect()
print(f"[INFO] Free memory: {gc.mem_free()} bytes")

from Wifi import Sta  # Provided WiFi library (Sharil Tumin, MIT License)

print(f"[INFO] Connecting to WiFi...")
wif = Sta(SSID, PASSWORD)
wif.connect()
wif.wait()

status = wif.status()
if not status:
    print("[FAIL] WiFi connection failed.")
    raise SystemExit

MY_IP = status[0]
print(f"[OK]   Connected. ESP IP: {MY_IP}")

# ─── Step 2: Load model weights ───
print("[INFO] Loading model weights...")
gc.collect()
mem_before = gc.mem_free()

import model_data
CLASSES = model_data.CLASSES
W1 = model_data.W1
b1 = model_data.b1
W2 = model_data.W2
b2 = model_data.b2
W3 = model_data.W3
b3 = model_data.b3
W4 = model_data.W4
b4 = model_data.b4

gc.collect()
print(f"[OK]   Model loaded. Free mem: {gc.mem_free()} bytes")

# ─── Step 3: Initialize camera ───
# Adapted from provided camera_test.py — identical pin config
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling

CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],
    "vsync_pin": 38,
    "href_pin": 47,
    "sda_pin": 40,
    "scl_pin": 39,
    "pclk_pin": 13,
    "xclk_pin": 10,
    "xclk_freq": 20000000,
    "powerdown_pin": -1,
    "reset_pin": -1,
}

print("[INFO] Initializing camera...")
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)
print("[OK]   Camera ready")

# ─── Step 4: Preprocessing + CNN functions ───
# (Same as final_submission.py — needed for on-device inference)

def preprocess_raw(bmp_data, threshold=128):
    """Convert raw camera BMP to flat list of 1024 floats for CNN.
    Handles 128x128 24-bit RGB top-down BMP from OV2640."""
    in_offset = 54  # known from camera testing
    in_w = 128
    in_h = 128
    in_row_size = 128 * 3
    scale = 4  # 128 / 32

    pixels = [0.0] * 1024
    for y in range(32):
        for x in range(32):
            src_row = y * scale
            src_col = x * scale
            src_off = in_offset + src_row * in_row_size + src_col * 3
            b = bmp_data[src_off] & 0xFF
            g = bmp_data[src_off + 1] & 0xFF
            r = bmp_data[src_off + 2] & 0xFF
            gray = (r + g + b) // 3
            pixels[y * 32 + x] = 1.0 if gray >= threshold else 0.0

    # Normalize orientation: rotate so hand enters from the left
    pixels = normalize_orientation(pixels)
    return pixels


def normalize_orientation(pixels):
    """Rotate 32x32 flat pixel array so hand (black pixels, value < 0.5)
    enters from the top edge. Counts black pixels on each edge and
    rotates accordingly. Must match train_cnn.py normalize_orientation()."""
    # Count black pixels on each edge
    top = sum(1 for x in range(32) if pixels[x] < 0.5)           # row 0
    bottom = sum(1 for x in range(32) if pixels[31*32 + x] < 0.5) # row 31
    left = sum(1 for y in range(32) if pixels[y*32] < 0.5)        # col 0
    right = sum(1 for y in range(32) if pixels[y*32 + 31] < 0.5)  # col 31

    counts = [('left', left), ('right', right), ('top', top), ('bottom', bottom)]
    dominant = max(counts, key=lambda c: c[1])[0]

    if dominant == 'top':
        return pixels  # already correct
    elif dominant == 'bottom':
        # 180 degree rotation
        return list(reversed(pixels))
    elif dominant == 'left':
        # 90 CW: left→top. old[y][x] → new[x][31-y]
        out = [0.0] * 1024
        for y in range(32):
            for x in range(32):
                out[x * 32 + (31 - y)] = pixels[y * 32 + x]
        return out
    elif dominant == 'right':
        # 90 CCW: right→top. old[y][x] → new[31-x][y]
        out = [0.0] * 1024
        for y in range(32):
            for x in range(32):
                out[(31 - x) * 32 + y] = pixels[y * 32 + x]
        return out
    return pixels


def conv2d(pixels, W, b, in_h, in_w, in_c, n_filters, fh, fw):
    out_h = in_h - fh + 1
    out_w = in_w - fw + 1
    out = [0.0] * (out_h * out_w * n_filters)
    for f in range(n_filters):
        for i in range(out_h):
            for j in range(out_w):
                val = b[f]
                for fi in range(fh):
                    for fj in range(fw):
                        for c in range(in_c):
                            px_idx = ((i + fi) * in_w + (j + fj)) * in_c + c
                            val += pixels[px_idx] * W[fi][fj][c][f]
                out_idx = (i * out_w + j) * n_filters + f
                out[out_idx] = val if val > 0 else 0.0
    return out, out_h, out_w


def maxpool2d(pixels, in_h, in_w, n_c):
    out_h = in_h // 2
    out_w = in_w // 2
    out = [0.0] * (out_h * out_w * n_c)
    for i in range(out_h):
        for j in range(out_w):
            for c in range(n_c):
                max_val = -999999.0
                for di in range(2):
                    for dj in range(2):
                        idx = ((i * 2 + di) * in_w + (j * 2 + dj)) * n_c + c
                        if pixels[idx] > max_val:
                            max_val = pixels[idx]
                out[i * out_w * n_c + j * n_c + c] = max_val
    return out, out_h, out_w


def dense(inputs, W, b, n_in, n_out, relu=True):
    out = [0.0] * n_out
    for j in range(n_out):
        val = b[j]
        for i in range(n_in):
            val += inputs[i] * W[i][j]
        out[j] = (val if val > 0 else 0.0) if relu else val
    return out


def softmax(x):
    max_val = max(x)
    exp_vals = []
    total = 0.0
    for v in x:
        val = v - max_val
        if val < -20:
            val = -20
        ev = 1.0 + val / 256.0
        for _ in range(8):
            ev = ev * ev
        exp_vals.append(ev)
        total += ev
    return [v / total for v in exp_vals]


def predict(pixels):
    x = pixels
    x, h, w = conv2d(x, W1, b1, 32, 32, 1, 8, 3, 3)
    x, h, w = maxpool2d(x, h, w, 8)
    x, h, w = conv2d(x, W2, b2, h, w, 8, 16, 3, 3)
    x, h, w = maxpool2d(x, h, w, 16)
    x = dense(x, W3, b3, 576, 32, relu=True)
    x = dense(x, W4, b4, 32, len(CLASSES), relu=False)
    probs = softmax(x)
    best_idx = probs.index(max(probs))
    return CLASSES[best_idx], probs[best_idx], probs


# ─── Step 5: Send frame to laptop ───
def send_to_laptop(msg_type, payload):
    """Send a typed message to the laptop web server.
    msg_type: 1 = raw BMP frame, 2 = classification JSON
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect((LAPTOP_IP, LAPTOP_PORT))
        # Header: 1 byte type + 4 byte length
        header = bytes([msg_type]) + struct.pack('<I', len(payload))
        s.sendall(header)
        # Send payload in chunks (memoryview-safe)
        sent = 0
        while sent < len(payload):
            chunk = payload[sent:sent+2048]
            s.send(bytes(chunk) if isinstance(chunk, memoryview) else chunk)
            sent += len(chunk)
        s.close()
        return True
    except Exception as e:
        print(f"[NET]  Send failed: {e}")
        try:
            s.close()
        except:
            pass
        return False


# ─── Step 6: Main loop — stream + classify ───
print(f"\n{'='*40}")
print("  LIVE STREAM + CLASSIFIER")
print(f"{'='*40}")
print(f"  Streaming to {LAPTOP_IP}:{LAPTOP_PORT}")
print(f"  Single image per classification")
print(f"  Start live_viewer.py on your Mac first!")
print(f"{'='*40}\n")

frame_count = 0

while True:
    try:
        gc.collect()

        # Send status: capturing (type 0x03)
        send_to_laptop(0x03, b"capturing")

        # Capture raw frame
        raw = cam.capture()
        frame_count += 1

        # Send raw frame for live preview (type 0x01)
        send_to_laptop(0x01, raw)

        # Send status: classifying (type 0x03)
        send_to_laptop(0x03, b"classifying")

        # Preprocess and classify
        t0 = ticks_ms()
        pixels = preprocess_raw(raw, 128)
        del raw
        gc.collect()

        cls_name, conf, probs = predict(pixels)
        del pixels
        gc.collect()

        inf_ms = ticks_diff(ticks_ms(), t0)

        # Build result string
        prob_str = ",".join(f"{CLASSES[i]}:{probs[i]:.2f}" for i in range(len(CLASSES)))
        result = f"{cls_name}|{conf:.2f}|{prob_str}|{inf_ms}"

        # Send classification result (type 0x02)
        send_to_laptop(0x02, result.encode())

        print(f"[#{frame_count}] {cls_name} ({conf:.0%}) | {prob_str} | {inf_ms}ms")

        if STREAM_INTERVAL > 0:
            sleep(STREAM_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Stopped by user")
        break
    except MemoryError:
        print("[ERR]  Memory! Collecting...")
        gc.collect()
        sleep(2)
    except Exception as e:
        print(f"[ERR]  {e}")
        gc.collect()
        sleep(1)
