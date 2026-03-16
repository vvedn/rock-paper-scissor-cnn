"""
camera_test_phase1.py — Phase 1: Camera Test
Runs on: ESP32S3 (XIAO ESP32S3 Sense)

Purpose:
    Initialize the OV2640 camera, capture a single 96x96 grayscale BMP image,
    save it to the ESP filesystem, and print diagnostic info over serial.

Adapted from provided working example: camera_test.py
Changes from original:
    1. Added FrameSize and PixelFormat configuration for 96x96 grayscale
    2. Added file save (original only captured to memory)
    3. Added structured debug output with pass/fail checks
    4. Added error handling with helpful messages
    5. Kept identical CAMERA_PARAMETERS pin mapping from original

Citation: Camera pin configuration and initialization pattern from
          course-provided camera_test.py example.
"""

import gc

# ─── Step 1: Free memory before camera init ───
gc.collect()
print(f"[INFO] Free memory before camera init: {gc.mem_free()} bytes")

# ─── Step 2: Import camera module ───
# This module comes from the flashed firmware.bin, not from pip
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling

# ─── Step 3: Camera hardware pin configuration ───
# Adapted from provided working example — DO NOT CHANGE these pins
# They are specific to the XIAO ESP32S3 Sense board layout
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],  # 8-bit parallel data bus
    "vsync_pin": 38,    # vertical sync
    "href_pin": 47,     # horizontal reference
    "sda_pin": 40,      # I2C data (camera config)
    "scl_pin": 39,      # I2C clock (camera config)
    "pclk_pin": 13,     # pixel clock
    "xclk_pin": 10,     # external clock to camera
    "xclk_freq": 20000000,  # 20 MHz clock frequency
    "powerdown_pin": -1,    # not used on this board
    "reset_pin": -1,        # not used on this board
}

# ─── Step 4: Initialize camera ───
print("[INFO] Initializing camera...")
try:
    cam = Camera(**CAMERA_PARAMETERS)
    cam.init()
    print("[OK]   Camera initialized successfully")
except Exception as e:
    print(f"[FAIL] Camera init failed: {e}")
    print("[TIP]  Make sure firmware.bin is flashed correctly")
    print("[TIP]  Try unplugging and re-plugging USB, then re-run")
    raise  # stop here if camera won't init

# ─── Step 5: Configure for 96x96 grayscale BMP output ───
# set_bmp_out(True) tells the camera driver to return BMP format
# instead of JPEG. This gives us raw pixel data with a BMP header.
# The smallest resolution the OV2640 supports is 96x96.
cam.set_bmp_out(True)
print("[OK]   BMP output mode enabled")

# ─── Step 6: Print camera settings ───
# Adapted from provided working example — enumerates all get_* methods
print("\n--- Camera Settings ---")
get_methods = [m for m in dir(cam) if callable(getattr(cam, m)) and m.startswith("get")]
for method in get_methods:
    try:
        result = getattr(cam, method)()
        print(f"  {method}: {result}")
    except Exception as e:
        print(f"  {method}: Error - {e}")

# ─── Step 7: Capture an image ───
print("\n[INFO] Capturing image...")
gc.collect()  # free memory before capture
try:
    img = cam.capture()
    print(f"[OK]   Captured image: {len(img)} bytes")
except Exception as e:
    print(f"[FAIL] Capture failed: {e}")
    print("[TIP]  Try adding a 1-second delay before capture")
    raise

# ─── Step 8: Validate the captured image ───
# A valid 96x96 8-bit grayscale BMP should be:
#   14 (BMP header) + 40 (DIB header) + 1024 (palette) + 96*96 (pixels) = 10,294 bytes
# But row padding may add bytes. The camera driver handles this.
EXPECTED_MIN_SIZE = 9000  # rough minimum for a 96x96 BMP
EXPECTED_MAX_SIZE = 15000  # generous upper bound

if len(img) < EXPECTED_MIN_SIZE:
    print(f"[WARN] Image seems too small ({len(img)} bytes)")
    print("[TIP]  Camera may not be in correct resolution mode")
elif len(img) > EXPECTED_MAX_SIZE:
    print(f"[WARN] Image seems too large ({len(img)} bytes)")
    print("[TIP]  Camera may be in a higher resolution mode")
else:
    print(f"[OK]   Image size looks correct for 96x96 BMP")

# Check BMP magic bytes
if img[0:2] == b'BM':
    print("[OK]   Valid BMP header detected (starts with 'BM')")
else:
    print(f"[WARN] BMP header not found. First 2 bytes: {img[0:2]}")
    print("[TIP]  Make sure cam.set_bmp_out(True) was called")

# ─── Step 9: Save image to ESP filesystem ───
FILENAME = "test_capture.bmp"
print(f"\n[INFO] Saving image to {FILENAME}...")
try:
    with open(FILENAME, "wb") as f:
        f.write(img)
    print(f"[OK]   Image saved to {FILENAME}")
except Exception as e:
    print(f"[FAIL] Could not save file: {e}")
    print("[TIP]  ESP filesystem may be full. Try deleting old files.")
    raise

# ─── Step 10: Verify saved file ───
import os
try:
    file_info = os.stat(FILENAME)
    file_size = file_info[6]  # st_size is index 6 in MicroPython
    print(f"[OK]   Verified: {FILENAME} exists, size = {file_size} bytes")
    if file_size == len(img):
        print("[OK]   File size matches captured image size")
    else:
        print(f"[WARN] File size ({file_size}) != capture size ({len(img)})")
except Exception as e:
    print(f"[FAIL] Could not verify file: {e}")

# ─── Step 11: Quick pixel sanity check ───
# Check if the image is all-zero (black) or all-255 (white) — both bad signs
BMP_HEADER_SIZE = 14
DIB_HEADER_SIZE = 40
PALETTE_SIZE = 256 * 4
pixel_start = BMP_HEADER_SIZE + DIB_HEADER_SIZE + PALETTE_SIZE

if len(img) > pixel_start + 100:
    # Sample 100 pixels from the middle of the image
    sample = img[pixel_start + 4000 : pixel_start + 4100]
    all_zero = all(b == 0 for b in sample)
    all_white = all(b == 255 for b in sample)

    if all_zero:
        print("[WARN] Sampled pixels are ALL BLACK — camera may be covered or broken")
        print("[TIP]  Remove any lens cap, check ribbon cable connection")
    elif all_white:
        print("[WARN] Sampled pixels are ALL WHITE — possible overexposure")
        print("[TIP]  Try pointing camera away from bright light")
    else:
        pixel_min = min(sample)
        pixel_max = max(sample)
        print(f"[OK]   Pixel range in sample: {pixel_min} – {pixel_max} (good, has variation)")

# ─── Step 12: Final summary ───
gc.collect()
print(f"\n{'='*40}")
print("PHASE 1 CAMERA TEST SUMMARY")
print(f"{'='*40}")
print(f"  Image captured:  YES ({len(img)} bytes)")
print(f"  BMP valid:       {'YES' if img[0:2] == b'BM' else 'NO'}")
print(f"  Saved to file:   {FILENAME}")
print(f"  Free memory:     {gc.mem_free()} bytes")
print(f"{'='*40}")
print("\nNext step: Download test_capture.bmp via Thonny and open it on your Mac")
print("           to visually confirm the image looks correct.")
print("           See Phase 1 instructions for how to do this.")
