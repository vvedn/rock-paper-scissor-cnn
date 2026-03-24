"""
train_cnn.py — Phase 3: Train a CNN to classify Rock vs Paper (vs Scissors later)
Runs on: Laptop (Apple Silicon Mac)

Purpose:
    Load 32x32 grayscale thresholded BMP images from dataset/ folders,
    train a small CNN using pure NumPy (no TensorFlow/PyTorch needed),
    evaluate accuracy, and export weights as a Python file for the ESP32.

Why pure NumPy:
    1. No framework install issues on Apple Silicon
    2. The exact same math runs on the ESP in MicroPython
    3. We fully understand every layer — easy to explain in class

CNN Architecture:
    Input:    32x32x1 (grayscale, values 0.0 or 1.0 after threshold)
    Conv2D:   8 filters, 3x3, ReLU      → 30x30x8
    MaxPool:  2x2                        → 15x15x8
    Conv2D:   16 filters, 3x3, ReLU     → 13x13x16
    MaxPool:  2x2                        → 6x6x16
    Flatten:  576
    Dense:    32, ReLU
    Dense:    N_classes, softmax

Original code by project team. No external code adapted.
"""

import numpy as np
import os
import struct
import random
import json
from datetime import datetime

# ─── Configuration ───
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
MODEL_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data.py")
WEIGHTS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.json")
IMG_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 16
VAL_SPLIT = 0.2  # 20% for validation

# ─── BMP Loading (no Pillow needed) ───
def load_bmp_grayscale(filepath):
    """Load a 32x32 8-bit grayscale BMP file into a numpy array.
    Returns pixel values normalized to 0.0-1.0"""
    with open(filepath, 'rb') as f:
        data = f.read()

    if data[0:2] != b'BM':
        return None

    offset = struct.unpack('<I', data[10:14])[0]
    w = struct.unpack('<i', data[18:22])[0]
    h = struct.unpack('<i', data[22:26])[0]
    h_abs = abs(h)

    if w != IMG_SIZE or h_abs != IMG_SIZE:
        return None

    row_size = ((w + 3) // 4) * 4  # BMP rows padded to 4 bytes
    pixels = np.zeros((h_abs, w), dtype=np.float32)

    for row in range(h_abs):
        if h > 0:  # bottom-up BMP
            src_row = h_abs - 1 - row
        else:  # top-down BMP
            src_row = row
        start = offset + src_row * row_size
        for col in range(w):
            pixels[row, col] = data[start + col] / 255.0

    return pixels


def normalize_orientation(img_2d):
    """Rotate a 32x32 binary image so the hand always enters from the top.

    Strategy: count black pixels on each edge (top/bottom/left/right).
    The edge with the most black pixels is where the arm enters.
    Rotate so that edge becomes the TOP side.

    This ensures consistent orientation regardless of how the camera
    is positioned relative to the hand.

    Works with both numpy arrays and plain 2D lists.
    """
    if isinstance(img_2d, np.ndarray):
        h, w = img_2d.shape[:2]
        if img_2d.ndim == 3:
            img = img_2d[:, :, 0]
        else:
            img = img_2d

        # Count black pixels (value < 0.5) on each edge
        top = np.sum(img[0, :] < 0.5)
        bottom = np.sum(img[-1, :] < 0.5)
        left = np.sum(img[:, 0] < 0.5)
        right = np.sum(img[:, -1] < 0.5)

        counts = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
        dominant = max(counts, key=counts.get)

        # Rotate so dominant edge becomes TOP
        if dominant == 'top':
            rotated = img  # already correct
        elif dominant == 'bottom':
            rotated = np.rot90(img, 2)  # 180 degrees
        elif dominant == 'left':
            rotated = np.rot90(img, -1)  # 90 CW: left→top
        elif dominant == 'right':
            rotated = np.rot90(img, 1)  # 90 CCW: right→top

        if img_2d.ndim == 3:
            return rotated[:, :, np.newaxis]
        return rotated
    else:
        # Plain list version (for ESP MicroPython compatibility testing)
        return img_2d


# ─── Data Loading ───
def load_dataset(dataset_dir):
    """Load all BMP images from class subfolders.
    Returns images array (N, 32, 32, 1) and labels array (N,)"""
    classes = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))
                      and not d.startswith('.')])

    # Only include classes that have images
    classes = [c for c in classes
               if any(f.endswith('.bmp') for f in os.listdir(os.path.join(dataset_dir, c)))]

    print(f"Classes found: {classes}")

    images = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        files = sorted([f for f in os.listdir(class_dir) if f.endswith('.bmp')])
        loaded = 0
        for f in files:
            img = load_bmp_grayscale(os.path.join(class_dir, f))
            if img is not None:
                # Normalize orientation so hand always enters from left
                img = normalize_orientation(img)
                images.append(img)
                labels.append(class_idx)
                loaded += 1
        print(f"  {class_name}: {loaded} images loaded (class {class_idx})")

    images = np.array(images, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels, dtype=np.int32)

    return images, labels, classes


# ─── Data Augmentation ───
def augment_image(img):
    """Apply random augmentations to a 32x32 image.
    Input shape: (32, 32, 1)

    More aggressive augmentation to make model robust to:
    - slight position differences between training (LANCZOS) and ESP (nearest-neighbor)
    - lighting/threshold variations
    - hand orientation differences
    """
    result = img.copy()
    h, w, c = result.shape

    # NO rotation augmentation — orientation is normalized before training
    # so all images have the hand entering from the left side

    # Random horizontal flip (50% chance)
    if random.random() > 0.5:
        result = result[:, ::-1, :]

    # Random vertical flip (20% chance)
    if random.random() > 0.8:
        result = result[::-1, :, :]

    # Random shift by 1-3 pixels (more range for robustness)
    dx = random.randint(-3, 3)
    dy = random.randint(-3, 3)
    if dx != 0 or dy != 0:
        shifted = np.zeros_like(result)
        x1s = max(0, dx)
        x1e = min(w, w + dx)
        x2s = max(0, -dx)
        x2e = min(w, w - dx)
        y1s = max(0, dy)
        y1e = min(h, h + dy)
        y2s = max(0, -dy)
        y2e = min(h, h - dy)
        shifted[y1s:y1e, x1s:x1e, :] = result[y2s:y2e, x2s:x2e, :]
        result = shifted

    # Random pixel erosion/dilation (simulates threshold differences)
    # Randomly flip ~5% of pixels — makes model robust to threshold mismatch
    if random.random() > 0.3:
        flip_mask = np.random.random(result.shape) < 0.05
        result = np.where(flip_mask, 1.0 - result, result)

    # Random noise (small)
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.03, result.shape).astype(np.float32)
        result = np.clip(result + noise, 0, 1)

    return result


def augment_dataset(images, labels, target_per_class=500):
    """Augment dataset so each class has at least target_per_class images."""
    classes = np.unique(labels)
    aug_images = list(images)
    aug_labels = list(labels)

    for c in classes:
        class_mask = labels == c
        class_images = images[class_mask]
        current_count = len(class_images)

        if current_count >= target_per_class:
            continue

        needed = target_per_class - current_count
        print(f"  Class {c}: {current_count} images, augmenting {needed} more")

        for i in range(needed):
            src = class_images[i % current_count]
            aug = augment_image(src)
            aug_images.append(aug)
            aug_labels.append(c)

    return np.array(aug_images), np.array(aug_labels)


# ─── CNN Layers (forward pass) ───

def conv2d_forward(x, W, b):
    """2D convolution. x: (H,W,Cin), W: (Fh,Fw,Cin,Cout), b: (Cout,)
    Returns: (H-Fh+1, W-Fw+1, Cout)"""
    H, W_in, Cin = x.shape
    Fh, Fw, _, Cout = W.shape
    out_h = H - Fh + 1
    out_w = W_in - Fw + 1
    out = np.zeros((out_h, out_w, Cout), dtype=np.float32)

    for f in range(Cout):
        for i in range(out_h):
            for j in range(out_w):
                patch = x[i:i+Fh, j:j+Fw, :]
                out[i, j, f] = np.sum(patch * W[:, :, :, f]) + b[f]
    return out


def conv2d_forward_fast(x, W, b):
    """Optimized conv2d using im2col approach."""
    H, W_in, Cin = x.shape
    Fh, Fw, _, Cout = W.shape
    out_h = H - Fh + 1
    out_w = W_in - Fw + 1

    # im2col: extract all patches into a 2D matrix
    cols = np.zeros((out_h * out_w, Fh * Fw * Cin), dtype=np.float32)
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            cols[idx] = x[i:i+Fh, j:j+Fw, :].flatten()
            idx += 1

    # Reshape filters to 2D
    W_2d = W.reshape(Fh * Fw * Cin, Cout)

    # Matrix multiply + bias
    out = cols @ W_2d + b
    return out.reshape(out_h, out_w, Cout)


def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(np.float32)


def maxpool2d(x, size=2):
    """2x2 max pooling. x: (H,W,C) → (H//2, W//2, C)"""
    H, W, C = x.shape
    out_h = H // size
    out_w = W // size
    out = np.zeros((out_h, out_w, C), dtype=np.float32)
    mask = np.zeros_like(x)  # for backprop

    for i in range(out_h):
        for j in range(out_w):
            patch = x[i*size:(i+1)*size, j*size:(j+1)*size, :]
            out[i, j, :] = patch.reshape(size*size, C).max(axis=0)
            # Store mask for backprop
            for c in range(C):
                p = patch[:, :, c]
                max_idx = np.unravel_index(p.argmax(), p.shape)
                mask[i*size + max_idx[0], j*size + max_idx[1], c] = 1

    return out, mask


def softmax(x):
    """Softmax activation"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def cross_entropy_loss(pred, label):
    """Cross-entropy loss for single sample"""
    return -np.log(pred[label] + 1e-8)


# ─── CNN Model Class ───
class TinyCNN:
    """Small CNN for rock/paper/scissors classification.
    Designed to fit in ESP32S3 memory (~20KB of weights)."""

    def __init__(self, n_classes):
        self.n_classes = n_classes

        # He initialization for conv/dense layers
        # Conv1: 3x3x1 → 8 filters
        self.W1 = np.random.randn(3, 3, 1, 8).astype(np.float32) * np.sqrt(2.0 / 9)
        self.b1 = np.zeros(8, dtype=np.float32)

        # Conv2: 3x3x8 → 16 filters
        self.W2 = np.random.randn(3, 3, 8, 16).astype(np.float32) * np.sqrt(2.0 / 72)
        self.b2 = np.zeros(16, dtype=np.float32)

        # Dense1: 576 → 32
        self.W3 = np.random.randn(576, 32).astype(np.float32) * np.sqrt(2.0 / 576)
        self.b3 = np.zeros(32, dtype=np.float32)

        # Dense2: 32 → n_classes
        self.W4 = np.random.randn(32, n_classes).astype(np.float32) * np.sqrt(2.0 / 32)
        self.b4 = np.zeros(n_classes, dtype=np.float32)

        total = (3*3*1*8 + 8) + (3*3*8*16 + 16) + (576*32 + 32) + (32*n_classes + n_classes)
        print(f"  Total parameters: {total:,}")

    def forward(self, x):
        """Forward pass. x shape: (32, 32, 1). Returns class probabilities."""
        # Conv1 + ReLU + Pool
        self.z1 = conv2d_forward_fast(x, self.W1, self.b1)   # 30x30x8
        self.a1 = relu(self.z1)
        self.p1, self.mask1 = maxpool2d(self.a1)               # 15x15x8

        # Conv2 + ReLU + Pool
        self.z2 = conv2d_forward_fast(self.p1, self.W2, self.b2)  # 13x13x16
        self.a2 = relu(self.z2)
        self.p2, self.mask2 = maxpool2d(self.a2)                    # 6x6x16

        # Flatten
        self.flat = self.p2.flatten()  # 576

        # Dense1 + ReLU
        self.z3 = self.flat @ self.W3 + self.b3   # 32
        self.a3 = relu(self.z3)

        # Dense2 + Softmax
        self.z4 = self.a3 @ self.W4 + self.b4     # n_classes
        self.probs = softmax(self.z4)

        return self.probs

    def backward(self, x, label, lr):
        """Backward pass with SGD update. Returns loss."""
        probs = self.probs.copy()
        loss = cross_entropy_loss(probs, label)

        # Output layer gradient (softmax + cross-entropy combined)
        d_z4 = probs.copy()
        d_z4[label] -= 1  # shape: (n_classes,)

        # Dense2 gradients
        d_W4 = np.outer(self.a3, d_z4)
        d_b4 = d_z4
        d_a3 = d_z4 @ self.W4.T  # shape: (32,)

        # ReLU gradient
        d_z3 = d_a3 * relu_derivative(self.z3)

        # Dense1 gradients
        d_W3 = np.outer(self.flat, d_z3)
        d_b3 = d_z3
        d_flat = d_z3 @ self.W3.T  # shape: (576,)

        # Unflatten → 6x6x16
        d_p2 = d_flat.reshape(self.p2.shape)

        # MaxPool2 backward
        d_a2 = np.zeros_like(self.a2)
        pH, pW, pC = d_p2.shape
        for i in range(pH):
            for j in range(pW):
                for c in range(pC):
                    d_a2[i*2:(i+1)*2, j*2:(j+1)*2, c] = \
                        self.mask2[i*2:(i+1)*2, j*2:(j+1)*2, c] * d_p2[i, j, c]

        # ReLU gradient
        d_z2 = d_a2 * relu_derivative(self.z2)

        # Conv2 backward
        d_W2 = np.zeros_like(self.W2)
        d_b2 = np.zeros_like(self.b2)
        d_p1 = np.zeros_like(self.p1)
        Fh, Fw, Cin, Cout = self.W2.shape
        out_h, out_w = d_z2.shape[0], d_z2.shape[1]

        for f in range(Cout):
            d_b2[f] = np.sum(d_z2[:, :, f])
            for i in range(out_h):
                for j in range(out_w):
                    d_W2[:, :, :, f] += self.p1[i:i+Fh, j:j+Fw, :] * d_z2[i, j, f]
                    d_p1[i:i+Fh, j:j+Fw, :] += self.W2[:, :, :, f] * d_z2[i, j, f]

        # MaxPool1 backward
        d_a1 = np.zeros_like(self.a1)
        pH, pW, pC = self.p1.shape
        for i in range(pH):
            for j in range(pW):
                for c in range(pC):
                    d_a1[i*2:(i+1)*2, j*2:(j+1)*2, c] = \
                        self.mask1[i*2:(i+1)*2, j*2:(j+1)*2, c] * d_p1[i, j, c]

        # ReLU gradient
        d_z1 = d_a1 * relu_derivative(self.z1)

        # Conv1 backward
        d_W1 = np.zeros_like(self.W1)
        d_b1 = np.zeros_like(self.b1)
        Fh, Fw, Cin, Cout = self.W1.shape
        out_h, out_w = d_z1.shape[0], d_z1.shape[1]

        for f in range(Cout):
            d_b1[f] = np.sum(d_z1[:, :, f])
            for i in range(out_h):
                for j in range(out_w):
                    d_W1[:, :, :, f] += x[i:i+Fh, j:j+Fw, :] * d_z1[i, j, f]

        # SGD update
        self.W4 -= lr * d_W4
        self.b4 -= lr * d_b4
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1

        return loss

    def predict(self, x):
        """Return predicted class index"""
        probs = self.forward(x)
        return np.argmax(probs)

    def export_weights(self, filepath_py, filepath_json, class_names):
        """Export weights as a Python file for ESP32 and JSON for backup."""
        weights = {
            'W1': self.W1.tolist(), 'b1': self.b1.tolist(),
            'W2': self.W2.tolist(), 'b2': self.b2.tolist(),
            'W3': self.W3.tolist(), 'b3': self.b3.tolist(),
            'W4': self.W4.tolist(), 'b4': self.b4.tolist(),
            'classes': class_names,
            'img_size': IMG_SIZE,
        }

        # Save JSON backup
        with open(filepath_json, 'w') as f:
            json.dump(weights, f)
        print(f"[OK] Weights saved to {filepath_json}")

        # Save as Python file for ESP32 (MicroPython can import this)
        with open(filepath_py, 'w') as f:
            f.write('"""CNN weights for ESP32 inference. Auto-generated."""\n\n')
            f.write(f'CLASSES = {class_names}\n')
            f.write(f'IMG_SIZE = {IMG_SIZE}\n\n')

            for name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4']:
                arr = weights[name]
                f.write(f'{name} = {repr(arr)}\n\n')

        print(f"[OK] ESP model file saved to {filepath_py}")


# ─── Training Loop ───
def train():
    print("=" * 50)
    print("  RPS CNN Trainer (Pure NumPy)")
    print("=" * 50)

    # Load data
    print("\n[1] Loading dataset...")
    images, labels, classes = load_dataset(DATASET_DIR)
    n_classes = len(classes)
    print(f"    Total: {len(images)} images, {n_classes} classes")

    if n_classes < 2:
        print("[FAIL] Need at least 2 classes with images!")
        return

    # Augment to balance classes
    print("\n[2] Augmenting dataset...")
    images, labels = augment_dataset(images, labels, target_per_class=350)
    print(f"    After augmentation: {len(images)} images")

    # Shuffle
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    # Train/val split
    split = int(len(images) * (1 - VAL_SPLIT))
    X_train, X_val = images[:split], images[split:]
    y_train, y_val = labels[:split], labels[split:]
    print(f"    Train: {len(X_train)}, Validation: {len(X_val)}")

    # Init model
    print("\n[3] Initializing CNN...")
    model = TinyCNN(n_classes)

    # Training
    print(f"\n[4] Training for {EPOCHS} epochs...")
    print(f"    Learning rate: {LEARNING_RATE}")
    print(f"    Batch size: {BATCH_SIZE} (SGD on individual samples)\n")

    best_val_acc = 0
    best_weights = None

    for epoch in range(EPOCHS):
        # Learning rate decay: reduce by half every 15 epochs
        lr = LEARNING_RATE * (0.5 ** (epoch // 15))

        # Shuffle training data each epoch
        train_idx = np.arange(len(X_train))
        np.random.shuffle(train_idx)

        epoch_loss = 0
        correct = 0

        for i in train_idx:
            x = X_train[i]
            y = y_train[i]

            # Forward
            probs = model.forward(x)
            pred = np.argmax(probs)
            if pred == y:
                correct += 1

            # Backward
            loss = model.backward(x, y, lr)
            epoch_loss += loss

        train_acc = correct / len(X_train) * 100
        avg_loss = epoch_loss / len(X_train)

        # Validation
        val_correct = 0
        for i in range(len(X_val)):
            pred = model.predict(X_val[i])
            if pred == y_val[i]:
                val_correct += 1
        val_acc = val_correct / len(X_val) * 100

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Deep copy weights
            best_weights = {
                'W1': model.W1.copy(), 'b1': model.b1.copy(),
                'W2': model.W2.copy(), 'b2': model.b2.copy(),
                'W3': model.W3.copy(), 'b3': model.b3.copy(),
                'W4': model.W4.copy(), 'b4': model.b4.copy(),
            }
            marker = " ← best"
        else:
            marker = ""

        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | loss: {avg_loss:.4f} | "
              f"train: {train_acc:.1f}% | val: {val_acc:.1f}%{marker}")

    # Restore best weights
    if best_weights:
        for k, v in best_weights.items():
            setattr(model, k, v)
        print(f"\n[OK] Restored best model (val accuracy: {best_val_acc:.1f}%)")

    # Confusion matrix on validation set
    print(f"\n[5] Validation Results:")
    confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
    for i in range(len(X_val)):
        pred = model.predict(X_val[i])
        confusion[y_val[i], pred] += 1

    # Print confusion matrix
    print(f"\n    Confusion Matrix (rows=actual, cols=predicted):")
    header = "         " + "  ".join(f"{c:>8s}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = f"  {c:>6s}  " + "  ".join(f"{confusion[i,j]:>8d}" for j in range(n_classes))
        print(row)

    # Per-class accuracy
    print(f"\n    Per-class accuracy:")
    for i, c in enumerate(classes):
        total = confusion[i].sum()
        correct = confusion[i, i]
        acc = correct / total * 100 if total > 0 else 0
        print(f"      {c}: {correct}/{total} = {acc:.1f}%")

    overall = sum(confusion[i, i] for i in range(n_classes)) / len(X_val) * 100
    print(f"\n    Overall validation accuracy: {overall:.1f}%")

    # Export
    print(f"\n[6] Exporting model...")
    model.export_weights(MODEL_OUTPUT, WEIGHTS_JSON, classes)

    print(f"\n{'='*50}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best validation accuracy: {best_val_acc:.1f}%")
    print(f"  Model exported to: {MODEL_OUTPUT}")
    print(f"{'='*50}")

    if best_val_acc < 70:
        print("\n[WARN] Accuracy below 70% — suggestions:")
        print("  • Collect more images (300+ per class)")
        print("  • Make sure rock/paper look different (fist vs flat hand)")
        print("  • Clean dataset — delete blurry or ambiguous images")
        print("  • Try more epochs (set EPOCHS = 50)")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    train()
