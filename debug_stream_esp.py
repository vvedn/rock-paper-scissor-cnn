"""
debug_stream_esp.py — Stream preprocessed 32x32 images + classification to laptop
Runs on: ESP32S3

Purpose:
    Capture, preprocess (same as final_submission.py), classify,
    then send the 32x32 BMP + result to the laptop so you can SEE
    what the CNN actually receives and compare it to training data.

Adapted from:
    - esp_image_server.py (WiFi + camera + push architecture)
    - final_submission.py (preprocessing + inference)
"""

import gc
import socket
import struct
from time import sleep, ticks_ms, ticks_diff

# ─── Configuration ───
SSID = "V\u2019s iPhone"
PASSWORD = "temppwdesp"
LAPTOP_IP = "172.20.10.2"
LAPTOP_PORT = 8080
CAPTURE_INTERVAL = 3

# ─── Step 1: Load model ───
print("[INFO] Loading model weights...")
gc.collect()
import model_data
CLASSES = model_data.CLASSES
W1 = model_data.W1; b1 = model_data.b1
W2 = model_data.W2; b2 = model_data.b2
W3 = model_data.W3; b3 = model_data.b3
W4 = model_data.W4; b4 = model_data.b4
print(f"[OK]   Classes: {CLASSES}")

# ─── Step 2: Connect WiFi ───
gc.collect()
from Wifi import Sta
print(f"[INFO] Connecting to WiFi...")
wif = Sta(SSID, PASSWORD)
wif.connect()
wif.wait()
status = wif.status()
if not status:
    print("[FAIL] WiFi failed")
    raise SystemExit
print(f"[OK]   Connected. ESP IP: {status[0]}")

# ─── Step 3: Init camera ───
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],
    "vsync_pin": 38, "href_pin": 47, "sda_pin": 40, "scl_pin": 39,
    "pclk_pin": 13, "xclk_pin": 10, "xclk_freq": 20000000,
    "powerdown_pin": -1, "reset_pin": -1,
}
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)
print("[OK]   Camera ready")

# ─── Step 4: Preprocessing function ───
# The provided image_preprocessing.py assumes 96x96 grayscale BMP,
# but the camera actually outputs 128x128 24-bit RGB BMP (49,206 bytes).
# This function handles the actual camera format.

def preprocess_camera_bmp(bmp_data, threshold=128):
    """Convert 128x128 RGB BMP from camera → 32x32 grayscale thresholded BMP.

    Camera output: 128x128, 24-bit RGB, no palette, pixel data at offset 54
    Target output: 32x32, 8-bit grayscale, with palette, thresholded to 0/255

    Returns a proper 32x32 8-bit BMP bytearray.
    """
    # Parse input BMP header
    in_offset = int.from_bytes(bmp_data[10:14], 'little')  # pixel data offset
    in_w = int.from_bytes(bmp_data[18:22], 'little')       # width
    in_h_raw = int.from_bytes(bmp_data[22:26], 'little')   # height (unsigned!)
    in_bpp = int.from_bytes(bmp_data[28:30], 'little')     # bits per pixel
    # CRITICAL FIX: MicroPython int.from_bytes returns UNSIGNED.
    # BMP height is signed (negative = top-down). Convert manually.
    if in_h_raw >= 0x80000000:
        in_h_raw = in_h_raw - 0x100000000  # e.g., 4294967168 → -128
    in_h = abs(in_h_raw)  # now correctly 128
    bottom_up = in_h_raw > 0  # now correctly False for top-down
    bytes_per_pixel = in_bpp // 8

    # Input row size (padded to 4 bytes)
    in_row_size = ((in_w * bytes_per_pixel + 3) // 4) * 4

    # Output: 32x32 8-bit grayscale BMP
    NEW_W = 32
    NEW_H = 32
    HEADER = 14
    DIB = 40
    PALETTE = 256 * 4  # 8-bit needs a palette
    OUT_ROW_SIZE = 32  # 32 pixels, already aligned to 4 bytes
    out_pixel_size = OUT_ROW_SIZE * NEW_H
    out_file_size = HEADER + DIB + PALETTE + out_pixel_size

    out = bytearray(out_file_size)

    # BMP header
    out[0:2] = b'BM'
    out[2:6] = out_file_size.to_bytes(4, 'little')
    out[10:14] = (HEADER + DIB + PALETTE).to_bytes(4, 'little')

    # DIB header
    out[14:18] = DIB.to_bytes(4, 'little')
    out[18:22] = NEW_W.to_bytes(4, 'little')
    out[22:26] = NEW_H.to_bytes(4, 'little')
    out[26:28] = b'\x01\x00'  # planes
    out[28:30] = b'\x08\x00'  # 8 bpp
    out[34:38] = out_pixel_size.to_bytes(4, 'little')

    # Grayscale palette (0=black, 255=white)
    palette_start = HEADER + DIB
    for i in range(256):
        idx = palette_start + i * 4
        out[idx] = i      # blue
        out[idx+1] = i    # green
        out[idx+2] = i    # red
        out[idx+3] = 0    # reserved

    # Resize + convert to grayscale + threshold
    out_pixel_start = HEADER + DIB + PALETTE
    scale_x = in_w / NEW_W
    scale_y = in_h / NEW_H

    for new_y in range(NEW_H):
        for new_x in range(NEW_W):
            old_x = int(new_x * scale_x)
            old_y = int(new_y * scale_y)

            # Handle row order:
            # bottom_up (positive height): row 0 = bottom → direct mapping
            # top_down (negative height): row 0 = top → flip for bottom-up output
            if bottom_up:
                src_row = old_y
            else:
                src_row = (in_h - 1 - old_y)

            src_offset = in_offset + src_row * in_row_size + old_x * bytes_per_pixel

            if src_offset + bytes_per_pixel > len(bmp_data):
                pixel = 255
            elif bytes_per_pixel >= 3:
                # IMPORTANT: cam.capture() returns memoryview with SIGNED bytes
                # (-128 to 127). Must convert to unsigned (0-255) with & 0xFF.
                b_val = bmp_data[src_offset] & 0xFF
                g_val = bmp_data[src_offset + 1] & 0xFF
                r_val = bmp_data[src_offset + 2] & 0xFF
                gray = (r_val + g_val + b_val) // 3
                pixel = 255 if gray >= threshold else 0
            else:
                gray = bmp_data[src_offset] & 0xFF
                pixel = 255 if gray >= threshold else 0

            out[out_pixel_start + new_y * OUT_ROW_SIZE + new_x] = pixel

    return out

# ─── Step 5: CNN forward pass (same as final_submission.py) ───
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
                out[(i * out_w + j) * n_c + c] = max_val
    return out, out_h, out_w

def dense(inputs, W, b, n_in, n_out, apply_relu=True):
    out = [0.0] * n_out
    for j in range(n_out):
        val = b[j]
        for i in range(n_in):
            val += inputs[i] * W[i][j]
        if apply_relu:
            out[j] = val if val > 0 else 0.0
        else:
            out[j] = val
    return out

def softmax(x):
    max_val = max(x)
    exp_vals = [0.0] * len(x)
    total = 0.0
    for i in range(len(x)):
        val = x[i] - max_val
        if val < -20:
            val = -20
        v = 1.0 + val / 256.0
        for _ in range(8):
            v = v * v
        exp_vals[i] = v
        total += v
    for i in range(len(exp_vals)):
        exp_vals[i] /= total
    return exp_vals

def predict(pixels_32x32):
    x = pixels_32x32
    x, h, w = conv2d(x, W1, b1, 32, 32, 1, 8, 3, 3)
    x, h, w = maxpool2d(x, h, w, 8)
    x, h, w = conv2d(x, W2, b2, h, w, 8, 16, 3, 3)
    x, h, w = maxpool2d(x, h, w, 16)
    x = dense(x, W3, b3, 576, 32, apply_relu=True)
    n_classes = len(CLASSES)
    x = dense(x, W4, b4, 32, n_classes, apply_relu=False)
    probs = softmax(x)
    best_idx = 0
    best_prob = probs[0]
    for i in range(1, len(probs)):
        if probs[i] > best_prob:
            best_prob = probs[i]
            best_idx = i
    return CLASSES[best_idx], best_prob, probs

# ─── Step 6: Fixed threshold ───
# Using fixed threshold of 128 (midpoint) to match training data
# The adaptive threshold was causing all-white images
THRESHOLD = 128
print(f"[OK]   Using fixed threshold: {THRESHOLD}")

# ─── Step 7: Main loop — capture, classify, stream to laptop ───
print(f"\n[INFO] Streaming to {LAPTOP_IP}:{LAPTOP_PORT}")
print("[INFO] Start debug_viewer.py on your Mac first!\n")

count = 0
while True:
    try:
        gc.collect()
        t0 = ticks_ms()

        # Capture
        raw_img = cam.capture()

        # Preprocess: resize 128x128 RGB → 32x32 grayscale + threshold
        small_bmp = preprocess_camera_bmp(raw_img, THRESHOLD)
        del raw_img
        gc.collect()

        # Extract pixels for CNN
        HEADER_SIZE = 14 + 40 + 256 * 4
        pixels = [0.0] * 1024
        for row in range(32):
            bmp_row = 31 - row
            offset = HEADER_SIZE + bmp_row * 32
            for col in range(32):
                pixels[row * 32 + col] = small_bmp[offset + col] / 255.0

        # Classify
        class_name, confidence, all_probs = predict(pixels)
        del pixels
        gc.collect()

        t_total = ticks_diff(ticks_ms(), t0)
        count += 1

        prob_str = " | ".join(f"{CLASSES[i]}:{all_probs[i]:.0%}" for i in range(len(CLASSES)))
        print(f"[#{count:3d}] {class_name:>8s} ({confidence:.0%}) | {prob_str} | {t_total}ms")

        # Send preprocessed 32x32 BMP + classification to laptop
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((LAPTOP_IP, LAPTOP_PORT))

            # Protocol: 4 bytes BMP length + BMP data + class name string + newline
            result_str = f"{class_name} {confidence:.2f}"
            result_bytes = result_str.encode()

            # Send: [4B bmp_len][bmp_data][4B result_len][result_data]
            bmp_len = len(small_bmp)
            s.sendall(struct.pack('>I', bmp_len))
            s.sendall(small_bmp)
            s.sendall(struct.pack('>I', len(result_bytes)))
            s.sendall(result_bytes)
            s.close()
        except Exception as e:
            print(f"  [WARN] Could not send to laptop: {e}")

        del small_bmp
        gc.collect()
        sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("\n[STOP] Stopped")
        break
    except MemoryError:
        print("[ERR] MemoryError")
        gc.collect()
        sleep(3)
    except Exception as e:
        print(f"[ERR] {e}")
        gc.collect()
        sleep(3)
