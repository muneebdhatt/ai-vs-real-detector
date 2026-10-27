"""Dataset loading helpers."""
import os
import tensorflow as tf
from config import (DATA_DIR, REALWORLD_DIR, IMG_SIZE, BATCH_SIZE,
                    VAL_SPLIT, SEED, CLASS_NAMES)


def _load(directory, subset=None, shuffle=True, validation_split=None):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,           # FAKE=0, REAL=1
        image_size=(IMG_SIZE, IMG_SIZE),   # CIFAKE 32x32 is upscaled here
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        validation_split=validation_split,
        subset=subset,
    )


def get_train_val(data_dir=DATA_DIR):
    train_dir = os.path.join(data_dir, "train")
    train = _load(train_dir, subset="training", validation_split=VAL_SPLIT)
    val = _load(train_dir, subset="validation", validation_split=VAL_SPLIT)
    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE)


def get_test(data_dir=DATA_DIR):
    test = _load(os.path.join(data_dir, "test"), shuffle=False)
    return test.prefetch(tf.data.AUTOTUNE)


def get_realworld():
    """Your own test set: realworld_test/real and realworld_test/ai.
    Folder names are mapped to the same labels as CIFAKE (ai->FAKE, real->REAL)."""
    ds = tf.keras.utils.image_dataset_from_directory(
        REALWORLD_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["ai", "real"],        # ai=0 (FAKE), real=1 (REAL)
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return ds.prefetch(tf.data.AUTOTUNE)
