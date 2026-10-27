"""Shared settings. Change values here, not in the other scripts."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset layout expected (CIFAKE):
#   DATA_DIR/train/REAL, DATA_DIR/train/FAKE, DATA_DIR/test/REAL, DATA_DIR/test/FAKE
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data", "cifake"))
REALWORLD_DIR = os.environ.get("REALWORLD_DIR", os.path.join(ROOT, "realworld_test"))  # realworld_test/real, realworld_test/ai

MODELS_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "report")
WEB_MODEL_DIR = os.path.join(ROOT, "web", "model")

IMG_SIZE = 128          # both models use 128x128 input
BATCH_SIZE = 64
VAL_SPLIT = 0.1
SEED = 42

# Class order matters: alphabetical folder names -> FAKE=0, REAL=1.
# Model output is P(REAL). Keep this consistent in predict.py and the web app.
CLASS_NAMES = ["FAKE", "REAL"]

BASELINE_PATH = os.path.join(MODELS_DIR, "baseline_cnn.keras")
MAIN_PATH = os.path.join(MODELS_DIR, "mobilenetv2.keras")
