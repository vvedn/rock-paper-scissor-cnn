"""
esp_image_server.py — Dataset Collection: ESP Side
Runs on: ESP32S3 (XIAO ESP32S3 Sense)

Purpose:
    Capture a 96x96 BMP image every few seconds and PUSH it to the
    laptop's TCP server. The ESP acts as a CLIENT, not a server.
    This avoids MicroPython's broken accept() on ESP32S3.

Adapted from provided working examples:
    - Camera init: camera_test.py (pin config, init, set_bmp_out)
    - WiFi: Wifi.py Sta class (Sharil Tumin, MIT License)

Changes from originals:
    - Combined camera + WiFi into one script
    - ESP connects to laptop (client mode) instead of running a server
    - Sends raw 96x96 BMP; resize happens on laptop
"""

import gc
import socket
from time import sleep

# ─── Configuration ───
SSID = "V\u2019s iPhone"  # curly apostrophe to match iPhone's actual SSID
PASSWORD = "temppwdesp"

# *** CHANGE THIS to your Mac's IP address ***
# Find it on Mac: System Settings → Wi-Fi → Details → IP Address
# Or run: ipconfig getifaddr en0 (or ipconfig getifaddr en1)
LAPTOP_IP = "172.20.10.2"
LAPTOP_PORT = 8080

CAPTURE_INTERVAL = 5  # seconds between captures

# ─── Step 1: Connect to WiFi ───
gc.collect()
print(f"[INFO] Free memory: {gc.mem_free()} bytes")

from Wifi import Sta  # Provided WiFi library (Sharil Tumin, MIT License)

print(f"[INFO] Connecting to '{SSID}'...")
wif = Sta(SSID, PASSWORD)
wif.connect()
wif.wait()

status = wif.status()
if not status:
    print("[FAIL] WiFi connection failed. Check SSID/password.")
    raise SystemExit

MY_IP = status[0]
print(f"[OK]   Connected. ESP IP: {MY_IP}")

# ─── Step 2: Initialize camera ───
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
cam.set_bmp_out(True)  # 96x96 grayscale BMP output
print("[OK]   Camera ready")

# ─── Step 3: Main loop — capture and push to laptop ───
print(f"\n[INFO] Will push images to {LAPTOP_IP}:{LAPTOP_PORT}")
print(f"[INFO] Capturing every {CAPTURE_INTERVAL}s")
print("[INFO] Start collect_dataset.py on your Mac first!\n")

count = 0

while True:
    try:
        # Capture image
        gc.collect()
        raw_img = cam.capture()
        count += 1
        print(f"[CAP]  #{count} captured ({len(raw_img)} bytes)", end=" ")

        # Connect to laptop and send
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((LAPTOP_IP, LAPTOP_PORT))

        # Send 4-byte length header then image data
        img_len = len(raw_img)
        s.sendall(img_len.to_bytes(4, 'big'))
        s.sendall(raw_img)
        s.close()

        print(f"→ sent to laptop")

    except OSError as e:
        print(f"→ FAILED ({e})")
        print("[TIP]  Is collect_dataset.py running on your Mac?")
    except Exception as e:
        print(f"→ ERROR ({e})")

    sleep(CAPTURE_INTERVAL)
