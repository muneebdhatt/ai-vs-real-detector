"""Train the baseline CNN and the MobileNetV2 transfer model.

Usage:
    python src/train.py --model baseline --epochs 15
    python src/train.py --model mobilenet --epochs 5 --finetune-epochs 8
    python src/train.py --model mobilenet --epochs 1 --finetune-epochs 1 --max-batches 5   # smoke test
"""
import argparse
import json
import os
import tensorflow as tf
from config import MODELS_DIR, REPORT_DIR, BASELINE_PATH, MAIN_PATH, SEED
from data import get_train_val
from models import build_baseline, build_mobilenet, unfreeze_top

METRICS = [
    "accuracy",
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]


def callbacks(path):
    return [
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=METRICS)


def merge(h1, h2=None):
    out = {k: [float(v) for v in vals] for k, vals in h1.history.items()}
    if h2:
        for k, vals in h2.history.items():
            out.setdefault(k, []).extend(float(v) for v in vals)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "mobilenet"], required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=0, help="limit batches per epoch (smoke tests)")
    args = p.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    train, val = get_train_val()
    kw = {}
    if args.max_batches:
        train, val = train.take(args.max_batches), val.take(args.max_batches)

    if args.model == "baseline":
        model = build_baseline()
        compile_model(model, 1e-3)
        hist = model.fit(train, validation_data=val, epochs=args.epochs,
                         callbacks=callbacks(BASELINE_PATH))
        history = merge(hist)
        out_path = BASELINE_PATH
    else:
        model, base = build_mobilenet(trainable_base=False)
        compile_model(model, 1e-3)
        print("Phase 1: training the head with a frozen base")
        h1 = model.fit(train, validation_data=val, epochs=args.epochs,
                       callbacks=callbacks(MAIN_PATH))
        print("Phase 2: fine-tuning the top of the base at a low learning rate")
        unfreeze_top(base, 30)
        compile_model(model, 1e-5)
        h2 = model.fit(train, validation_data=val, epochs=args.finetune_epochs,
                       callbacks=callbacks(MAIN_PATH))
        history = merge(h1, h2)
        out_path = MAIN_PATH

    if not os.path.exists(out_path):   # e.g. very short smoke runs
        model.save(out_path)
    with open(os.path.join(REPORT_DIR, f"history_{args.model}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
