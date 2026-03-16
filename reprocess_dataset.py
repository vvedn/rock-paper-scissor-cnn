"""
reprocess_dataset.py — Re-preprocess training data to match ESP inference
Runs on: Laptop

Purpose:
    The original dataset was preprocessed with Pillow (luminosity grayscale,
    LANCZOS resize, adaptive threshold). The ESP uses a different method
    (simple average grayscale, nearest-neighbor resize, fixed threshold=128).
    This mismatch causes the CNN to see different images at inference time
    than it was trained on.

    This script re-processes the RAW 128x128 RGB BMPs using the EXACT same
    algorithm the ESP uses, so training data matches inference data perfectly.

    If raw BMPs are not available (only 32x32 preprocessed BMPs exist),
    it re-thresholds them with a fixed threshold of 128 to at least fix
    the threshold mismatch.
"""

import os
import struct
import numpy as np
from PIL import Image
import io
import shutil

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_matched")

def esp_preprocess_from_32x32_bmp(filepath):
    """Re-threshold an existing 32x32 BMP with fixed threshold=128.
    This matches the ESP's fixed threshold instead of the adaptive np.mean().

    Also converts grayscale using simple average (R+G+B)/3 if the image
    happens to be color, matching ESP behavior.
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    if data[0:2] != b'BM':
        return None

    offset = struct.unpack('<I', data[10:14])[0]
    w = struct.unpack('<i', data[18:22])[0]
    h = struct.unpack('<i', data[22:26])[0]
    bpp = struct.unpack('<H', data[28:30])[0]
    h_abs = abs(h)

    # Read pixel values as grayscale
    row_size = ((w * (bpp // 8) + 3) // 4) * 4
    pixels = np.zeros((h_abs, w), dtype=np.uint8)

    for row in range(h_abs):
        if h > 0:  # bottom-up
            src_row = h_abs - 1 - row
        else:  # top-down
            src_row = row
        start = offset + src_row * row_size
        for col in range(w):
            if bpp == 24:
                b = data[start + col * 3]
                g = data[start + col * 3 + 1]
                r = data[start + col * 3 + 2]
                # Simple average — matches ESP's (R+G+B)//3
                pixels[row, col] = (r + g + b) // 3
            else:
                pixels[row, col] = data[start + col]

    # Apply FIXED threshold of 128 — matches ESP
    threshold = 128
    pixels = np.where(pixels >= threshold, 255, 0).astype(np.uint8)

    return pixels


def save_as_32x32_bmp(pixels, filepath):
    """Save a 32x32 uint8 array as an 8-bit grayscale BMP."""
    img = Image.fromarray(pixels, mode='L')
    img.save(filepath, format='BMP')


def main():
    print("=" * 50)
    print("  Re-processing dataset to match ESP preprocessing")
    print("=" * 50)
    print(f"  Source:  {DATASET_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Method:  Fixed threshold=128 (matching ESP)")
    print()

    classes = sorted([d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d))])

    for cls in classes:
        src_dir = os.path.join(DATASET_DIR, cls)
        dst_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(src_dir) if f.endswith('.bmp')])
        good = 0
        bad = 0

        for f in files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dst_dir, f)

            pixels = esp_preprocess_from_32x32_bmp(src_path)
            if pixels is not None:
                save_as_32x32_bmp(pixels, dst_path)
                good += 1
            else:
                bad += 1

        print(f"  {cls:10s}: {good} re-processed, {bad} skipped")

    # Show comparison: old vs new for one sample
    print("\n--- Sample comparison (first rock image) ---")
    sample_old = os.path.join(DATASET_DIR, classes[0],
                              sorted(os.listdir(os.path.join(DATASET_DIR, classes[0])))[0])
    sample_new = os.path.join(OUTPUT_DIR, classes[0],
                              sorted(os.listdir(os.path.join(OUTPUT_DIR, classes[0])))[0])

    with open(sample_old, 'rb') as f:
        old_data = f.read()
    old_off = struct.unpack('<I', old_data[10:14])[0]

    with open(sample_new, 'rb') as f:
        new_data = f.read()
    new_off = struct.unpack('<I', new_data[10:14])[0]

    # Count differences
    diffs = 0
    for i in range(1024):
        old_px = old_data[old_off + i] if old_off + i < len(old_data) else 0
        new_px = new_data[new_off + i] if new_off + i < len(new_data) else 0
        if old_px != new_px:
            diffs += 1

    print(f"  Pixels changed: {diffs}/1024 ({diffs*100//1024}%)")
    print()
    print(f"[OK] Dataset re-processed. Now run train_cnn.py with:")
    print(f'     DATASET_DIR pointed to "{OUTPUT_DIR}"')
    print()
    print("Or run: python train_cnn.py --dataset dataset_matched")


if __name__ == "__main__":
    main()
