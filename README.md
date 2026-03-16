# Rock-Paper-Scissors CNN Classifier — XIAO ESP32S3 Sense

A complete embedded machine learning pipeline that captures hand gesture images on an ESP32S3 microcontroller, trains a convolutional neural network on a laptop, and runs real-time inference on the device.

## Hardware

- **Board**: Seeed Studio XIAO ESP32S3 Sense
- **Camera**: OV2640 (on-board)
- **Firmware**: MicroPython (custom build with camera support — `firmware.bin`)
- **Laptop**: Apple Silicon Mac (M1/M2/M3)

## Project Structure

```
RPSCnn/
├── ESP Files (upload to board via Thonny)
│   ├── final_submission.py      Main inference script — capture, preprocess, classify
│   ├── esp_image_server.py      Dataset collection — push images to laptop
│   ├── debug_stream_esp.py      Debug — stream preprocessed frames to laptop
│   ├── camera_test.py           Camera initialization test (course-provided)
│   ├── camera_test_phase1.py    Extended camera test with validation
│   ├── wifi_test_phase2.py      WiFi connection test
│   ├── Wifi.py                  WiFi library (Sharil Tumin, MIT License)
│   ├── image_preprocessing.py   BMP preprocessing utilities (course-provided)
│   ├── model_data.py            Trained CNN weights (auto-generated)
│   ├── model_weights.bin        Compact binary weights (auto-generated)
│   └── boot.py                  MicroPython boot config
│
├── Laptop Files (run on Mac)
│   ├── train_cnn.py             CNN training (pure NumPy)
│   ├── collect_dataset.py       Receive and label images from ESP
│   ├── debug_viewer.py          Visualize what the ESP CNN sees
│   ├── reprocess_dataset.py     Align training data with ESP preprocessing
│   └── export_binary_weights.py Convert weights to compact binary format
│
├── dataset/                     Training images (32x32 BMP, ~300 per class)
│   ├── rock/
│   ├── paper/
│   └── scissors/
│
├── docs/
│   ├── design_document.md       System architecture and design decisions
│   └── cnn_architecture.md      CNN layer explanations and improvements
│
├── firmware.bin                 MicroPython firmware for ESP32S3
└── README.md                    This file
```

## Quick Start

### 1. Flash Firmware

```bash
pip install esptool
esptool.py --chip esp32s3 --port /dev/tty.usbmodem* erase_flash
esptool.py --chip esp32s3 --port /dev/tty.usbmodem* write_flash 0 firmware.bin
```

### 2. Test Camera

Open Thonny, select **MicroPython (ESP32)** interpreter, upload `camera_test.py`, and run it. You should see a captured image size printed.

### 3. Test WiFi

Edit `wifi_test_phase2.py` with your WiFi SSID and password (use a phone hotspot for simplicity — ESP32 only supports 2.4 GHz). Upload `Wifi.py` and `wifi_test_phase2.py`, run, and confirm an IP address is assigned.

### 4. Collect Dataset

**On Mac:**
```bash
python collect_dataset.py
```

**On ESP (via Thonny):**
Upload and run `esp_image_server.py` (edit `LAPTOP_IP` to your Mac's IP first).

Press `r`, `p`, or `s` on the Mac to label incoming images as rock, paper, or scissors. Collect ~300 images per class.

### 5. Train CNN

```bash
python train_cnn.py
```

This trains a pure NumPy CNN and exports weights to `model_data.py`.

### 6. Run Inference

Upload `final_submission.py` and `model_data.py` to the ESP via Thonny. Run `final_submission.py`. The serial output will show classifications:

```
rock (87%)  | paper:5% | rock:87% | scissors:8% | 5200ms
```

## CNN Architecture

```
Input:    32x32x1 (binary grayscale)
Conv2D:   8 filters, 3x3, ReLU      → 30x30x8
MaxPool:  2x2                        → 15x15x8
Conv2D:   16 filters, 3x3, ReLU     → 13x13x16
MaxPool:  2x2                        → 6x6x16
Flatten:  576
Dense:    32 neurons, ReLU
Dense:    3 neurons, Softmax         → [rock, paper, scissors]

Total parameters: ~19,800
Validation accuracy: ~89%
```

See `docs/cnn_architecture.md` for detailed layer explanations and improvement suggestions.

## Image Preprocessing Pipeline

```
Camera capture (128x128 RGB BMP, 49 KB)
    → Grayscale: (R + G + B) / 3
    → Resize: 128x128 → 32x32 (nearest-neighbor, every 4th pixel)
    → Threshold: pixels >= 128 → white, < 128 → black
    → Normalize: 0.0 (black/hand) or 1.0 (white/background)
```

## Requirements

### ESP32S3
- MicroPython firmware with camera module support (`firmware.bin`)
- Thonny IDE for file upload and serial monitoring

### Laptop (Mac)
- Python 3.9+
- NumPy (`pip install numpy`)
- Pillow (`pip install Pillow`) — for dataset collection only

No TensorFlow, PyTorch, or other ML frameworks required. The CNN is implemented in pure NumPy for training and pure MicroPython for inference.

## Known Issues

- **Inference speed**: ~5 seconds per classification (pure Python math). Could be reduced to <100 ms with a C firmware implementation.
- **WiFi reliability**: iPhone hotspots may auto-disable after idle periods. Keep the Personal Hotspot settings screen open during data collection.
- **SSID encoding**: iPhones use a curly apostrophe (U+2019) in names like "V\u2019s iPhone". The SSID must match the exact bytes.
- **Signed memoryview**: `cam.capture()` returns signed bytes. All pixel access must use `& 0xFF` to convert to unsigned.

## Citations

See individual source files for per-file citations. Summary:

| Source | License | Usage |
|--------|---------|-------|
| `Wifi.py` by Sharil Tumin | MIT License | WiFi STA connection on ESP |
| `camera_test.py` | Course-provided | Camera pin configuration and initialization pattern |
| `image_preprocessing.py` | Course-provided | Reference for BMP preprocessing functions |
| `camera` MicroPython module | Community firmware | Camera driver (compiled into `firmware.bin`) |
| NumPy | BSD License | CNN training on laptop |
| Pillow | HPND License | Image format handling during dataset collection |
| Claude (Anthropic) | AI assistant | Code development assistance — see individual file headers |

## License

Course project — refer to course policies for distribution terms.
