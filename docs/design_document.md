# Rock-Paper-Scissors CNN Classifier — Design Document

## 1. Project Overview

This project implements an end-to-end machine learning pipeline for classifying rock-paper-scissors hand gestures using a Seeed Studio XIAO ESP32S3 Sense microcontroller. Images are captured by the on-board OV2640 camera, preprocessed into 32x32 binary (black/white) bitmaps, and classified by a lightweight convolutional neural network (CNN) running entirely on the ESP32S3 in MicroPython. The classification result is printed over serial to the host PC in near real-time (~5 seconds per inference).

## 2. System Architecture

The system is divided between two platforms:

```
 ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
 │     XIAO ESP32S3 Sense          │     │     Apple Silicon Mac (Laptop)  │
 │                                 │     │                                 │
 │  OV2640 Camera                  │     │  Dataset Collection             │
 │    ↓ capture (128x128 RGB BMP)  │     │    collect_dataset.py           │
 │  Preprocessing                  │ WiFi│      ↓                          │
 │    ↓ resize to 32x32            │────→│  Training                       │
 │    ↓ grayscale conversion       │     │    train_cnn.py                 │
 │    ↓ binary threshold (128)     │     │      ↓                          │
 │  CNN Forward Pass               │     │  Weight Export                  │
 │    ↓ conv2d → pool → dense      │     │    model_data.py → upload to   │
 │  Classification                 │←────│    ESP via Thonny               │
 │    ↓ print result over serial   │     │                                 │
 └─────────────────────────────────┘     └─────────────────────────────────┘
```

### 2.1 ESP32S3 Responsibilities
- Camera initialization and image capture (128x128 24-bit RGB BMP)
- Image preprocessing: resize 128x128 → 32x32, RGB → grayscale, binary threshold
- CNN inference: pure MicroPython forward pass using pre-trained weights
- Output: print classification result ("rock", "paper", or "scissors") with confidence percentages over USB serial
- WiFi connectivity for dataset collection phase (push images to laptop)

### 2.2 Laptop Responsibilities
- Receive and label training images from ESP via WiFi (TCP socket)
- Dataset management: organize into class folders, clean, augment
- CNN training: pure NumPy implementation (no TensorFlow/PyTorch dependency)
- Weight export: convert trained parameters to Python lists for ESP import
- Debug visualization: receive preprocessed frames from ESP to verify pipeline

## 3. Data Pipeline

### 3.1 Image Collection
The ESP captures 128x128 RGB BMP images and pushes them via TCP to the laptop every 5 seconds. The laptop's `collect_dataset.py` receives each image, converts to 32x32 grayscale using Pillow, applies adaptive thresholding, and saves to labeled folders (`dataset/rock/`, `dataset/paper/`, `dataset/scissors/`).

The ESP acts as a TCP **client** (not server), connecting to the laptop's listening socket. This architecture was chosen because MicroPython's `socket.accept()` on ESP32S3 has a known issue where it fails to return after `cam.capture()` allocates a large memory buffer.

### 3.2 Dataset Statistics
- **rock**: 108 images
- **paper**: 105 images
- **scissors**: 104 images
- **Total**: 317 raw images, augmented to 750 for training (600 train / 150 validation)

### 3.3 Preprocessing
Each image undergoes the following transformation:
1. **Capture**: 128x128 pixels, 24-bit RGB, BMP format (49,206 bytes)
2. **Grayscale conversion**: `gray = (R + G + B) / 3` (simple average)
3. **Resize**: 128x128 → 32x32 via nearest-neighbor sampling (every 4th pixel)
4. **Binary threshold**: pixels >= 128 → white (255), pixels < 128 → black (0)
5. **Normalize**: 0.0 (black/hand) or 1.0 (white/background) for CNN input

A critical lesson learned was ensuring the **same preprocessing** runs on both the laptop (training) and ESP (inference). An early version used Pillow's luminosity-weighted grayscale and LANCZOS resampling on the laptop but simple-average grayscale and nearest-neighbor on the ESP. This mismatch was resolved by re-preprocessing the training dataset with `reprocess_dataset.py` to match the ESP's exact algorithm.

### 3.4 Data Augmentation
To expand the 317-image dataset, the following augmentations are applied during training:
- Horizontal flip
- Vertical flip
- Random 1-2 pixel shifts (up/down/left/right)
- Binary erosion (shrink shapes by 1 pixel)
- Salt-and-pepper noise (1% of pixels flipped)

## 4. Memory Constraints

| Resource | Available | Used | Notes |
|----------|-----------|------|-------|
| RAM | ~8 MB (PSRAM) | ~500 KB for model + image buffer | Comfortable margin |
| Flash | 8 MB | ~450 KB for model_data.py + code | Plenty of room |
| Raw capture buffer | 49,206 bytes | Allocated by camera driver | Freed after preprocessing |
| 32x32 pixel array | 1,024 bytes | Input to CNN | Minimal |
| CNN weights | ~19,800 float parameters | ~79 KB as float32 | Fits in PSRAM |

## 5. Communication Protocols

### 5.1 Dataset Collection (ESP → Laptop)
- Transport: TCP over WiFi (phone hotspot, 2.4 GHz)
- Protocol: 4-byte little-endian length header + raw BMP payload
- Direction: ESP connects to laptop (client → server)
- Port: 8080

### 5.2 Inference Output
- Transport: USB serial (Thonny IDE)
- Format: Human-readable text with class name, confidence, and timing

## 6. Key Technical Challenges Encountered

1. **MicroPython socket.accept() failure**: After camera capture allocates a large buffer, `accept()` never returns. Solved by reversing the architecture — ESP pushes to laptop instead of laptop pulling from ESP.

2. **Signed byte memoryview**: `cam.capture()` returns a `memoryview` with signed bytes (-128 to 127). Pixel values like 200 appeared as -56, corrupting grayscale conversion. Fixed by masking with `& 0xFF`.

3. **Top-down BMP orientation**: The camera outputs BMP with negative height (-128), indicating top-down row order. The preprocessing initially assumed bottom-up, producing garbled or blank images. Fixed by detecting the height sign and adjusting row indexing.

4. **SSID encoding mismatch**: The iPhone hotspot broadcasts "V\u2019s iPhone" with a UTF-8 curly apostrophe (`\xe2\x80\x99`), but the code used a straight apostrophe. WiFi connections silently failed until the exact bytes were matched.

5. **Train/inference preprocessing mismatch**: Pillow's `convert("L")` uses luminosity weighting (0.299R + 0.587G + 0.114B) while the ESP uses simple averaging ((R+G+B)/3). This caused the CNN to see different images at inference time. Solved by reprocessing training data with the ESP's exact algorithm.
