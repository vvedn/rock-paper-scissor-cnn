"""
collect_dataset.py — Dataset Collection: Laptop Side
Runs on: Mac (Apple Silicon)

Purpose:
    Run a TCP server that receives BMP images pushed from the ESP,
    resizes them to 32x32 grayscale, and saves to labeled folders.

Workflow:
    1. Run this script FIRST on your Mac
    2. Choose a label (r/p/s) — all incoming images get that label
    3. Run esp_image_server.py on the ESP — it pushes images every 5s
    4. Press Ctrl+C to stop, then pick a new label
    5. Press q to quit

Requirements (install on Mac):
    pip install Pillow

Original code by project author. No borrowed code in this file.
"""

import socket
import sys
import time
import io
import struct
from pathlib import Path
from PIL import Image

# ─── Configuration ───
SERVER_PORT = 8080  # must match LAPTOP_PORT in esp_image_server.py
DATASET_DIR = Path(__file__).parent / "dataset"

# Labels and their keyboard keys
LABELS = {
    "r": "rock",
    "p": "paper",
    "s": "scissors",
}


def setup_folders():
    """Create dataset/rock, dataset/paper, dataset/scissors folders."""
    for label in LABELS.values():
        folder = DATASET_DIR / label
        folder.mkdir(parents=True, exist_ok=True)
        existing = len(list(folder.glob("*.bmp")))
        print(f"  {label:10s} → {folder}  ({existing} images already)")


def count_images(label):
    """Count existing images in a label folder."""
    return len(list((DATASET_DIR / label).glob("*.bmp")))


def get_next_filename(label):
    """Generate the next sequential filename for a label."""
    folder = DATASET_DIR / label
    idx = count_images(label)
    return folder / f"{label}_{idx:04d}.bmp"


def receive_one_image(server_sock):
    """Wait for ESP to connect and push one image. Returns BMP bytes or None."""
    server_sock.settimeout(30.0)
    try:
        client, addr = server_sock.accept()
    except socket.timeout:
        return None

    try:
        # Read 4-byte length header
        header = b""
        while len(header) < 4:
            chunk = client.recv(4 - len(header))
            if not chunk:
                client.close()
                return None
            header += chunk

        img_len = struct.unpack('>I', header)[0]

        # Read the image data
        data = b""
        while len(data) < img_len:
            chunk = client.recv(min(4096, img_len - len(data)))
            if not chunk:
                break
            data += chunk
        client.close()

        if len(data) < 100 or data[0:2] != b'BM':
            print(f"[WARN] Invalid BMP: {len(data)} bytes")
            return None

        # Resize to 32x32 grayscale using Pillow
        img = Image.open(io.BytesIO(data))
        img = img.convert("L")  # ensure grayscale
        img = img.resize((32, 32), Image.LANCZOS)

        # Apply threshold to create high-contrast black & white image
        # This makes hand shapes much clearer for CNN training
        # Pixels above threshold → white (255), below → black (0)
        import numpy as np
        arr = np.array(img)
        threshold = np.mean(arr)  # adaptive threshold based on image brightness
        arr = np.where(arr >= threshold, 255, 0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

        # Save back to BMP bytes
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        return buf.getvalue()

    except Exception as e:
        print(f"[ERR]  {e}")
        try:
            client.close()
        except:
            pass
        return None


def get_local_ip():
    """Get the Mac's IP on the local network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "unknown"


def main():
    local_ip = get_local_ip()

    print("=" * 50)
    print("  RPS Dataset Collector (Server Mode)")
    print("=" * 50)
    print(f"\n  Listening on port: {SERVER_PORT}")
    print(f"  Your Mac IP:      {local_ip}")
    print(f"  Dataset dir:      {DATASET_DIR}\n")
    print(f"  *** Set LAPTOP_IP = \"{local_ip}\" in esp_image_server.py ***\n")

    setup_folders()

    # Start TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", SERVER_PORT))
    server.listen(5)

    print(f"\n  Server started on {local_ip}:{SERVER_PORT}")
    print("\n  Controls:")
    print("    r = label incoming images as ROCK")
    print("    p = label incoming images as PAPER")
    print("    s = label incoming images as SCISSORS")
    print("    Ctrl+C = stop current label, pick new one")
    print("    q = quit\n")

    total_saved = 0

    try:
        while True:
            counts = {label: count_images(label) for label in LABELS.values()}
            status = " | ".join(f"{k}: {v}" for k, v in counts.items())
            print(f"\n[{status}]  Total saved: {total_saved}")

            key = input("  Label for incoming images? [r]ock / [p]aper / [s]cissors / q=quit: ").strip().lower()

            if key == "q":
                break

            if key not in LABELS:
                print(f"  Unknown key '{key}'. Use r/p/s/q.")
                continue

            label = LABELS[key]
            print(f"\n[RUN]  Saving all incoming images as '{label}'")
            print(f"       Waiting for ESP to push images...")
            print(f"       Press Ctrl+C to stop and pick a new label.\n")

            session_saved = 0
            try:
                while True:
                    bmp_data = receive_one_image(server)

                    if bmp_data is None:
                        print("  [WAIT] No image received (ESP may not be running yet)...")
                        continue

                    filepath = get_next_filename(label)
                    with open(filepath, "wb") as f:
                        f.write(bmp_data)

                    session_saved += 1
                    total_saved += 1
                    print(f"  [{label}] #{session_saved} SAVED → {filepath.name} "
                          f"({len(bmp_data)} bytes)")

            except KeyboardInterrupt:
                pass

            print(f"\n[DONE] Saved {session_saved} images as '{label}' in this session")

    except KeyboardInterrupt:
        pass

    server.close()
    print(f"\n{'='*50}")
    print(f"  Collection complete. Total saved: {total_saved}")
    print(f"{'='*50}")
    counts = {label: count_images(label) for label in LABELS.values()}
    for label, cnt in counts.items():
        print(f"  {label:10s}: {cnt} images")


if __name__ == "__main__":
    main()
