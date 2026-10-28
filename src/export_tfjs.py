"""Export the trained Keras model to TensorFlow.js for the Vercel site.

Why this is more than one converter call: the training model is a Keras 3 model with random
augmentation layers, and TensorFlow.js cannot load Keras 3 exports. So this script:
 1. Rebuilds the inference graph (Rescaling -> MobileNetV2 -> GlobalAvgPool -> Dense) in tf_keras
    (Keras 2), copies the trained weights across, and checks the outputs match the original model.
 2. Saves it as models/mobilenetv2_inference.h5
 3. Converts it with tensorflowjs_converter into web/model/ (model.json + .bin shards).

Requires:  pip install tensorflowjs tf_keras
Usage:     python src/export_tfjs.py
"""
import os
import subprocess
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MAIN_PATH, MODELS_DIR, WEB_MODEL_DIR, IMG_SIZE, SUFFIX

INFERENCE_PATH = os.path.join(MODELS_DIR, f"mobilenetv2_inference{SUFFIX}.h5")


def build_inference_model():
    import tf_keras  # Keras 2, the format TensorFlow.js understands

    m = tf.keras.models.load_model(MAIN_PATH)
    base3 = next(l for l in m.layers if isinstance(l, tf.keras.Model) and "mobilenet" in l.name.lower())
    head3 = m.get_layer("p_real")

    base2 = tf_keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=None)
    base2.set_weights(base3.get_weights())

    inp = tf_keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = tf_keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)
    x = base2(x)
    x = tf_keras.layers.GlobalAveragePooling2D()(x)
    out = tf_keras.layers.Dense(1, activation="sigmoid", name="p_real")(x)
    inf = tf_keras.Model(inp, out, name="mobilenetv2_inference")
    inf.get_layer("p_real").set_weights(head3.get_weights())

    # The exported copy must reproduce the trained model's predictions exactly.
    xs = np.random.default_rng(0).uniform(0, 255, (8, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    diff = float(np.abs(m.predict(xs, verbose=0) - inf.predict(xs, verbose=0)).max())
    print(f"Max difference between trained model and export copy: {diff:.2e}")
    assert diff < 1e-4, "export copy differs from the trained model - do not deploy"
    os.makedirs(MODELS_DIR, exist_ok=True)
    inf.save(INFERENCE_PATH)


def main():
    build_inference_model()
    os.makedirs(WEB_MODEL_DIR, exist_ok=True)
    cmd = ["tensorflowjs_converter", "--input_format=keras",
           "--output_format=tfjs_layers_model", INFERENCE_PATH, WEB_MODEL_DIR]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    total = sum(os.path.getsize(os.path.join(WEB_MODEL_DIR, f)) for f in os.listdir(WEB_MODEL_DIR))
    print(f"Exported to {WEB_MODEL_DIR} ({total / 1e6:.1f} MB)")
    print("Next: serve web/ locally and compare a few images with python src/predict.py")


if __name__ == "__main__":
    main()
