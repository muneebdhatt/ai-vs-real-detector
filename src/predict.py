"""Predict whether one image (or every image in a folder) is REAL or AI-generated.

Usage:
    python src/predict.py --image sample.jpg
    python src/predict.py --folder some_dir --model models/mobilenetv2.keras
"""
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MAIN_PATH, IMG_SIZE

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_image(path):
    # Use the same resize as training (tf.image.resize, bilinear, no antialiasing)
    # so predictions match evaluate.py and the website.
    arr = np.asarray(Image.open(path).convert("RGB"), dtype="float32")
    return tf.image.resize(arr, (IMG_SIZE, IMG_SIZE), method="bilinear").numpy()


def predict_paths(model, paths):
    x = np.stack([load_image(p) for p in paths])
    p_real = model.predict(x, verbose=0).ravel()
    return p_real


def label(p_real):
    # The model outputs P(REAL). Confidence is the probability of the predicted class.
    return ("REAL", p_real) if p_real >= 0.5 else ("AI-GENERATED", 1 - p_real)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--folder")
    ap.add_argument("--model", default=MAIN_PATH)
    args = ap.parse_args()
    if not args.image and not args.folder:
        ap.error("give --image or --folder")

    model = tf.keras.models.load_model(args.model)
    if args.image:
        paths = [args.image]
    else:
        paths = [os.path.join(args.folder, f) for f in sorted(os.listdir(args.folder))
                 if f.lower().endswith(EXTS)]
    for path, p in zip(paths, predict_paths(model, paths)):
        name, conf = label(float(p))
        print(f"{os.path.basename(path)}: {name} ({conf * 100:.1f}% confidence)")


if __name__ == "__main__":
    main()
