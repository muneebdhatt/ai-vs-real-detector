"""Evaluate a saved model on the CIFAKE test set and on your own real-world set.

Usage:
    python src/evaluate.py --model mobilenet
    python src/evaluate.py --model baseline
Writes report/results_<model>.json and confusion-matrix / ROC images.
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from config import BASELINE_PATH, MAIN_PATH, REPORT_DIR, REALWORLD_DIR, CLASS_NAMES, SUFFIX
from data import get_test, get_realworld

PATHS = {"baseline": BASELINE_PATH, "mobilenet": MAIN_PATH}


def predict_all(model, ds):
    y_true, y_prob = [], []
    for x, y in ds:
        y_prob.extend(model.predict(x, verbose=0).ravel().tolist())
        y_true.extend(y.numpy().ravel().tolist())
    return np.array(y_true).astype(int), np.array(y_prob)


def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else None
    return out


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_prob, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(PATHS), required=True)
    p.add_argument("--max-batches", type=int, default=0)
    args = p.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    model = tf.keras.models.load_model(PATHS[args.model])
    results = {}

    sets = {"cifake_test": get_test()}
    rw_real = os.path.join(REALWORLD_DIR, "real")
    rw_ai = os.path.join(REALWORLD_DIR, "ai")
    if all(os.path.isdir(d) and os.listdir(d) for d in (rw_real, rw_ai)):
        sets["realworld_test"] = get_realworld()
    else:
        print("No images in realworld_test/real and realworld_test/ai yet - skipping that set.")

    for name, ds in sets.items():
        if args.max_batches:
            ds = ds.take(args.max_batches)
        y_true, y_prob = predict_all(model, ds)
        r = metrics(y_true, y_prob)
        results[name] = r
        plot_cm(r["confusion_matrix"], f"{args.model}{SUFFIX} - {name}",
                os.path.join(REPORT_DIR, f"cm_{args.model}{SUFFIX}_{name}.png"))
        if r["roc_auc"] is not None:
            plot_roc(y_true, y_prob, f"{args.model}{SUFFIX} - {name}",
                     os.path.join(REPORT_DIR, f"roc_{args.model}{SUFFIX}_{name}.png"))
        print(name, json.dumps({k: v for k, v in r.items() if k != "confusion_matrix"}))

    if "realworld_test" in results:
        gap = results["cifake_test"]["accuracy"] - results["realworld_test"]["accuracy"]
        results["generalization_gap_accuracy"] = float(gap)
        print(f"Generalization gap (CIFAKE acc - real-world acc): {gap:.3f}")

    with open(os.path.join(REPORT_DIR, f"results_{args.model}{SUFFIX}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
