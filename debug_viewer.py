"""
debug_viewer.py — Receive preprocessed 32x32 images + classification from ESP
Runs on: Laptop (Mac)

Purpose:
    Shows what the CNN on the ESP is actually seeing, side-by-side with
    training data samples, so you can debug classification issues.
"""

import socket
import struct
import os
import sys
import io

LISTEN_PORT = 8080
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_captures")
os.makedirs(DEBUG_DIR, exist_ok=True)


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def bmp_to_ascii(bmp_data, label=""):
    """Convert 32x32 8-bit BMP to ASCII art."""
    header_size = 14 + 40 + 256 * 4
    lines = []
    if label:
        lines.append(f"  --- {label} ---")

    black_count = 0
    total = 32 * 32

    for row in range(32):
        bmp_row = 31 - row  # BMP is bottom-up
        offset = header_size + bmp_row * 32
        line = "  "
        for col in range(32):
            px = bmp_data[offset + col]
            if px < 128:
                line += "██"
                black_count += 1
            else:
                line += "  "
        lines.append(line)

    lines.append(f"  black: {black_count}/{total} ({black_count/total*100:.0f}%)")
    return "\n".join(lines)


def load_training_sample(class_name):
    """Load first training image for a class, for comparison."""
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(class_dir):
        return None
    files = sorted([f for f in os.listdir(class_dir) if f.endswith('.bmp')])
    if not files:
        return None
    with open(os.path.join(class_dir, files[len(files)//2]), 'rb') as f:
        return f.read()


def main():
    local_ip = get_local_ip()
    print("=" * 70)
    print("  RPS Debug Viewer")
    print("=" * 70)
    print(f"  Listening on {local_ip}:{LISTEN_PORT}")
    print(f"  Saving captures to {DEBUG_DIR}")
    print(f"  Showing ESP's preprocessed image vs training samples")
    print("=" * 70)

    # Pre-load one training sample per class for comparison
    training_samples = {}
    for cls in ['rock', 'paper', 'scissors']:
        sample = load_training_sample(cls)
        if sample:
            training_samples[cls] = sample
            print(f"  Loaded training sample for '{cls}'")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", LISTEN_PORT))
    server.listen(1)

    count = 0

    print(f"\n  Waiting for ESP to send frames...\n")

    while True:
        try:
            conn, addr = server.accept()
            conn.settimeout(10)

            # Read BMP
            bmp_len_data = conn.recv(4)
            if len(bmp_len_data) < 4:
                conn.close()
                continue
            bmp_len = struct.unpack('>I', bmp_len_data)[0]

            bmp_data = b''
            while len(bmp_data) < bmp_len:
                chunk = conn.recv(min(4096, bmp_len - len(bmp_data)))
                if not chunk:
                    break
                bmp_data += chunk

            # Read classification result
            result_len_data = conn.recv(4)
            result_str = ""
            if len(result_len_data) == 4:
                result_len = struct.unpack('>I', result_len_data)[0]
                result_data = conn.recv(result_len)
                result_str = result_data.decode('utf-8', errors='replace')

            conn.close()
            count += 1

            # Parse result
            parts = result_str.split()
            predicted_class = parts[0] if parts else "unknown"
            confidence = float(parts[1]) if len(parts) > 1 else 0.0

            # Save the captured image
            save_path = os.path.join(DEBUG_DIR, f"debug_{count:04d}_{predicted_class}.bmp")
            with open(save_path, 'wb') as f:
                f.write(bmp_data)

            # Display
            print("=" * 70)
            print(f"  Frame #{count} | ESP says: {predicted_class} ({confidence:.0%})")
            print("=" * 70)
            print()
            print("  WHAT THE ESP CNN SEES (preprocessed 32x32):")
            print(bmp_to_ascii(bmp_data, f"ESP capture #{count}"))
            print()

            # Show training sample for comparison
            if predicted_class in training_samples:
                print(f"  TRAINING SAMPLE FOR '{predicted_class}':")
                print(bmp_to_ascii(training_samples[predicted_class],
                                   f"training {predicted_class}"))
                print()

            # Show all training samples for comparison
            print("  TRAINING SAMPLES (for reference):")
            for cls in ['rock', 'paper', 'scissors']:
                if cls in training_samples:
                    print(bmp_to_ascii(training_samples[cls], f"training: {cls}"))
                    print()

            print(f"  Saved to: {save_path}")
            print()

        except KeyboardInterrupt:
            print("\n[STOP] Viewer stopped")
            break
        except Exception as e:
            print(f"[ERR] {e}")

    server.close()
    print(f"\nTotal frames received: {count}")
    print(f"Debug images saved in: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
