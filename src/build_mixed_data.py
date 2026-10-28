"""Build a more varied TRAINING set that mixes three sources:

  * CIFAKE (subsample)                           real = CIFAR-10 photos, fake = Stable Diffusion, 32x32
  * COCO photos  (real, high resolution)         Hugging Face: Multimodal-Fatima/COCO_captions_validation
  * Midjourney v6 images (fake, high resolution) Hugging Face: Photoroom/midjourney-v6-recap

Why: a model trained only on tiny CIFAKE images does not generalise (it scored 50% on the
real-world test set). Adding sharp, high-resolution real and AI images, from both classes at
both resolutions, stops the model from using "blurry vs sharp" as a shortcut.

Your real-world test images are NEVER used for training: this script finds which COCO and
Midjourney rows they came from (realworld_test/sources.csv and the fixed random seed used
to pick them) and skips those rows.

Usage (Colab, after `pip install -q datasets`):
    python src/build_mixed_data.py
    # then train on it:
    TRAIN_DIR=data/mixed MODEL_TAG=mixed python src/train.py --model mobilenet --epochs 8 --finetune-epochs 10 --robust-aug

Output layout:  data/mixed/train/REAL , data/mixed/train/FAKE   (the validation split is taken
from it automatically by train.py).
"""
import argparse
import csv
import io
import os
import random
import re
import shutil
import sys

from PIL import Image, ImageOps
from datasets import Image as HFImage
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, ROOT, REALWORLD_DIR

MJ_DS, MJ_TOTAL = "Photoroom/midjourney-v6-recap", 1235432
COCO_DS = "Multimodal-Fatima/COCO_captions_validation"
IMG_EXT = (".jpg", ".jpeg", ".png")


def mj_test_rows():
    """Rows of the Midjourney dataset that the real-world TEST images were taken from.
    Reproduces the seeded random choice used when the test set was collected."""
    rng = random.Random(42)
    offsets = rng.sample(range(0, MJ_TOTAL - 10, 10), 90)
    return {o + k for o in offsets for k in range(5)}


def coco_test_rows():
    path = os.path.join(REALWORLD_DIR, "sources.csv")
    rows = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("label") == "real":
                m = re.search(r"row (\d+)", r.get("source", ""))
                if m:
                    rows.add(int(m.group(1)))
    if not rows:
        raise SystemExit(f"No COCO row numbers found in {path}. Use the sources.csv that came with realworld_test.zip.")
    return rows


def save_square(raw, path, size=256):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im = ImageOps.fit(im, (size, size), Image.LANCZOS, centering=(0.5, 0.5))
    im.save(path, "JPEG", quality=93)


def stream_images(name, split, keep, n, skip_rows, stride=1):
    """Yield (row_index, raw_bytes) for n usable rows, skipping the test rows."""
    ds = load_dataset(name, split=split, streaming=True).select_columns(["image"])
    ds = ds.cast_column("image", HFImage(decode=False))
    got = 0
    for i, row in enumerate(ds):
        if got >= n:
            return
        if i in skip_rows or i % stride != 0:
            continue
        raw = row["image"]["bytes"]
        if raw is None:
            continue
        yield i, raw
        got += 1


def fetch(name, split, out_dir, prefix, n, skip_rows, stride=1):
    os.makedirs(out_dir, exist_ok=True)
    got = 0
    for i, raw in stream_images(name, split, None, n, skip_rows, stride):
        try:
            save_square(raw, os.path.join(out_dir, f"{prefix}_{i:07d}.jpg"))
            got += 1
        except Exception as e:
            print("skip row", i, e)
        if got % 500 == 0 and got:
            print(f"  {prefix}: {got}/{n}", flush=True)
    print(f"{prefix}: saved {got} images")
    return got


def link_cifake(cifake_dir, out_root, per_class, seed=0):
    rng = random.Random(seed)
    for cls in ("REAL", "FAKE"):
        src = os.path.join(cifake_dir, "train", cls)
        files = sorted(f for f in os.listdir(src) if f.lower().endswith(IMG_EXT))
        rng.shuffle(files)
        dst = os.path.join(out_root, "train", cls)
        os.makedirs(dst, exist_ok=True)
        for f in files[:per_class]:
            os.symlink(os.path.join(src, f), os.path.join(dst, "cifake_" + f))
        print(f"cifake {cls}: linked {min(per_class, len(files))} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifake-dir", default=DATA_DIR, help="folder containing train/REAL and train/FAKE")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "mixed"))
    ap.add_argument("--n-per-source", type=int, default=4800, help="COCO photos and Midjourney images to use")
    ap.add_argument("--cifake-per-class", type=int, default=10000)
    ap.add_argument("--mj-stride", type=int, default=2, help="take every Nth Midjourney row (rows share prompts)")
    args = ap.parse_args()

    coco_skip, mj_skip = coco_test_rows(), mj_test_rows()
    print(f"Excluding {len(coco_skip)} COCO rows and {len(mj_skip)} Midjourney rows used by the test set")

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    link_cifake(args.cifake_dir, args.out, args.cifake_per_class)

    print("Downloading COCO photos (REAL)...")
    n_real = fetch(COCO_DS, "validation", os.path.join(args.out, "train", "REAL"), "coco",
                   args.n_per_source, coco_skip)
    print("Downloading Midjourney v6 images (FAKE)...")
    n_fake = fetch(MJ_DS, "train", os.path.join(args.out, "train", "FAKE"), "mj",
                   args.n_per_source, mj_skip, stride=args.mj_stride)

    for cls in ("REAL", "FAKE"):
        d = os.path.join(args.out, "train", cls)
        print(f"{cls}: {len(os.listdir(d))} images in {d}")
    print("\nNext:\n  TRAIN_DIR=%s MODEL_TAG=mixed python src/train.py --model mobilenet --epochs 8 --finetune-epochs 10 --robust-aug" % args.out)


if __name__ == "__main__":
    main()
    # The streaming download threads can keep the process alive after the work is done,
    # which would leave a Colab cell spinning forever. Everything is saved by now, so exit.
    sys.stdout.flush()
    os._exit(0)
