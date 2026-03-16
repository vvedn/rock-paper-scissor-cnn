# CNN— Rock-Paper-Scissors Classifier

This project implements a CNN for classifying rock-paper-scissors hand gestures using a Seeed Studio XIAO ESP32S3 Sense microcontroller. Images are captured, preprocessed into 32x32 binary (black/white) bitmaps, and classified by a lightweight convolutional neural network (CNN) running entirely on the ESP32S3 in MicroPython. The classification result is printed over serial to the host PC in near real-time (~5 seconds per inference).

## Data Pipeline

### Image Collection
The ESP captures 128x128 RGB BMP images and pushes them via TCP to the laptop every 5 seconds. The laptop's `collect_dataset.py` receives each image, converts to 32x32 grayscale using Pillow, applies adaptive thresholding, and saves to labeled folders (`dataset/rock/`, `dataset/paper/`, `dataset/scissors/`).

The ESP acts as a TCP **client** (not server), connecting to the laptop's listening socket. This architecture was chosen because MicroPython's `socket.accept()` on ESP32S3 has a known issue where it fails to return after `cam.capture()` allocates a large memory buffer.

### Dataset Statistics
- **rock**: 108 images
- **paper**: 105 images
- **scissors**: 104 images
- **Total**: 317 raw images, augmented to 750 for training (600 train / 150 validation)

### Preprocessing
Each image undergoes the following transformation:
1. **Capture**: 128x128 pixels, 24-bit RGB, BMP format (49,206 bytes)
2. **Grayscale conversion**: `gray = (R + G + B) / 3` (simple average)
3. **Resize**: 128x128 → 32x32 via nearest-neighbor sampling (every 4th pixel)
4. **Binary threshold**: pixels >= 128 → white (255), pixels < 128 → black (0)
5. **Normalize**: 0.0 (black/hand) or 1.0 (white/background) for CNN input

A critical lesson learned was ensuring the **same preprocessing** runs on both the laptop (training) and ESP (inference). An early version used Pillow's luminosity-weighted grayscale and LANCZOS resampling on the laptop but simple-average grayscale and nearest-neighbor on the ESP. This mismatch was resolved by re-preprocessing the training dataset with `reprocess_dataset.py` to match the ESP's exact algorithm.
## Architecture Overview

The classifier uses a compact convolutional neural network with two convolutional layers followed by two fully connected (dense) layers. The architecture is designed to fit within ESP32S3 memory constraints while maintaining sufficient capacity to distinguish three hand gesture classes.

```
Input (32x32x1)
    │
    ├─ Conv2D: 8 filters, 3x3 kernel, ReLU activation
    │  Output: 30x30x8    (8 feature maps)
    │  Parameters: (3x3x1 + 1) x 8 = 80
    │
    ├─ MaxPool2D: 2x2 stride 2
    │  Output: 15x15x8    (spatial reduction)
    │  Parameters: 0
    │
    ├─ Conv2D: 16 filters, 3x3 kernel, ReLU activation
    │  Output: 13x13x16   (16 feature maps)
    │  Parameters: (3x3x8 + 1) x 16 = 1,168
    │
    ├─ MaxPool2D: 2x2 stride 2
    │  Output: 6x6x16     (spatial reduction)
    │  Parameters: 0
    │
    ├─ Flatten
    │  Output: 576         (6 x 6 x 16)
    │
    ├─ Dense: 32 neurons, ReLU activation
    │  Output: 32
    │  Parameters: (576 + 1) x 32 = 18,464
    │
    └─ Dense: 3 neurons, Softmax activation
       Output: 3           (rock, paper, scissors probabilities)
       Parameters: (32 + 1) x 3 = 99

Total trainable parameters: ~19,811
```

## Layer-by-Layer Explanation

### Layer 1 — Conv2D (8 filters, 3x3)
**Purpose:** Extract low-level features such as edges, corners, and boundaries.

Each of the 8 filters slides a 3x3 window across the 32x32 input image, computing the dot product at each position. The ReLU activation (`max(0, x)`) zeroes out negative responses, keeping only strong feature detections. With 8 filters, the network learns 8 different edge orientations — sufficient to capture the outline differences between a closed fist (rock), flat hand (paper), and V-shape (scissors).

**Why 8 filters:** Fewer than 8 loses too much shape information. More than 8 increases memory usage without significant accuracy gain for only 3 simple classes.

### Layer 2 — MaxPool2D (2x2)
**Purpose:** Reduce spatial dimensions by half (30x30 → 15x15), making the network invariant to small shifts in hand position.

Max pooling takes the maximum value in each 2x2 window. This achieves two goals: (1) reduces computation for subsequent layers, and (2) provides translational robustness — if the hand moves a few pixels, the pooled features remain similar.

### Layer 3 — Conv2D (16 filters, 3x3)
**Purpose:** Extract higher-level features by combining the low-level edges from Layer 1.

Operating on the 8 feature maps from the first convolution, these 16 filters learn combinations of edges — for example, a "curved boundary" (rock) versus a "straight edge with parallel lines" (scissors). Each filter has a 3x3x8 kernel, connecting it to all 8 input channels.

**Why 16 filters:** Doubles the feature depth to capture more complex patterns. The spatial size is already small (15x15 → 13x13), so the memory cost is modest.

### Layer 4 — MaxPool2D (2x2)
**Purpose:** Further reduce spatial dimensions (13x13 → 6x6) before the dense layers.

After this pooling, each of the 16 feature maps is only 6x6, producing 576 values total when flattened. This is a manageable input size for a dense layer on the ESP32.

### Layer 5 — Dense (32 neurons, ReLU)
**Purpose:** Learn non-linear combinations of all spatial features to form a compact class representation.

This layer connects all 576 flattened features to 32 neurons. It is the "decision-making" layer — it learns which combinations of detected features correspond to each gesture. ReLU activation allows the network to learn complex decision boundaries.

**Why 32 neurons:** Sufficient for 3 classes. Using 64 or 128 would add ~18–37 KB of weights, straining ESP memory for minimal accuracy improvement.

### Layer 6 — Dense (3 neurons, Softmax)
**Purpose:** Output class probabilities.

The final layer has one neuron per class. Softmax normalizes the outputs so they sum to 1.0, producing interpretable probabilities. The predicted class is the one with the highest probability.

**ESP softmax implementation:** To avoid overflow from `exp()` on large values, the ESP uses a soft approximation: `max(0, x+1)` followed by normalization, which is more numerically stable in MicroPython.

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 0.001 (with decay to 0.0003) | Standard starting rate; decay prevents oscillation in later epochs |
| Epochs | 30–60 | Enough for convergence; early stopping at best validation accuracy |

## Performance

| Metric | Value |
|--------|-------|
| Training accuracy | 99.8% |
| Validation accuracy | 81–89% |
| Inference time on ESP | ~5 seconds per classification |
| Model size (Python lists) | ~424 KB on flash |
| Model size (binary float32) | ~79 KB |

### Per-Class Accuracy (Best Run)
| Class | Accuracy |
|-------|----------|
| Paper | 94.6% |
| Rock | 91.1% |
| Scissors | 81.6% |

Scissors has the lowest accuracy because its V-shape is sometimes confused with the spread fingers of paper, particularly when the two fingers are close together.

## Potential Improvements

### Data Quality Improvements
- **Collect more images** — 300+ per class with controlled lighting and plain backgrounds. The current ~105 per class is near the minimum. More data is the single most impactful improvement.
- **Better pose diversity** — Include both hands, multiple angles (front, side, 45 degrees), and varying distances (20–40 cm from camera).
- **Consistent framing** — Ensure the entire hand is visible in every image. Cropped hands confuse the classifier.
- **Data cleaning** — Remove images where the gesture is ambiguous, the hand is out of frame, or the background has strong edges that survive thresholding.
- **Add dropout** (e.g., 25% after pooling layers) — Reduces overfitting, which is evident from the gap between 99.8% train and 81% validation accuracy.
- **Batch normalization** — Would stabilize training, but adds complexity to the ESP forward pass.
- **Deeper network** — A third convolutional layer (32 filters) could improve feature extraction, but would add ~15,000 parameters and slow ESP inference.
- **Binary weight quantization** — Convert float32 weights to int8 (4x smaller, faster multiply). Would require quantization-aware training or post-training calibration.
- **TFLite Micro** — Google's TensorFlow Lite for Microcontrollers runs optimized inference kernels in C. Would require building custom firmware, but inference would be 50–100x faster.
