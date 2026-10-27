"""Model definitions: custom CNN baseline and MobileNetV2 transfer model.

Both models take raw 0-255 RGB images of size IMG_SIZE x IMG_SIZE and do their
own preprocessing inside the graph, so the same file works in Python and after
TensorFlow.js export (no separate normalisation code needed in the browser).
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE

AUGMENT = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augment")


def build_baseline():
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inp)
    x = AUGMENT(x)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_real")(x)
    return models.Model(inp, out, name="baseline_cnn")


def build_mobilenet(trainable_base=False):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = trainable_base

    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = AUGMENT(inp)
    # MobileNetV2 expects inputs in [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_real")(x)
    model = models.Model(inp, out, name="mobilenetv2_detector")
    return model, base


def unfreeze_top(base, n_layers=30):
    """Fine-tune the last n_layers of the base (BatchNorm layers stay frozen)."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
