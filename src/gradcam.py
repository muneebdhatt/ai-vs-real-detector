"""Grad-CAM heatmaps for the MobileNetV2 detector.

Usage:
    python src/gradcam.py --image sample.jpg --out report/gradcam_sample.png
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MAIN_PATH, IMG_SIZE
from predict import load_image


def find_base(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    raise ValueError("MobileNet base not found in model")


def gradcam(model, img):
    """img: float32 array (H, W, 3) in 0-255. Returns heatmap (H, W) in 0-1 and P(real)."""
    base = find_base(model)
    last_conv = base.get_layer("out_relu").output if "out_relu" in [l.name for l in base.layers] \
        else base.layers[-1].output
    feat_model = tf.keras.Model(base.inputs, [last_conv, base.output])

    x = tf.convert_to_tensor(img[None, ...])
    # replicate the pre-base layers of the model (rescaling only; augmentation is inactive at inference)
    x = x / 127.5 - 1.0
    with tf.GradientTape() as tape:
        conv_out, base_out = feat_model(x, training=False)
        tape.watch(conv_out)
        pooled = tf.reduce_mean(base_out, axis=(1, 2))
        # run the model's head on the pooled features
        head = pooled
        for layer in model.layers[-2:]:      # Dropout, Dense
            head = layer(head, training=False)
        score = head[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]
    return cam, float(score.numpy()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default=MAIN_PATH)
    ap.add_argument("--out", default="gradcam.png")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    img = load_image(args.image)
    cam, p_real = gradcam(model, img)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(img.astype("uint8")); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(img.astype("uint8")); ax[1].imshow(cam, cmap="jet", alpha=0.45)
    ax[1].set_title(f"Grad-CAM (P(real)={p_real:.2f})"); ax[1].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
