"""Dataset loading helpers."""
import os
import tensorflow as tf
from config import (DATA_DIR, TRAIN_DIR, REALWORLD_DIR, IMG_SIZE, BATCH_SIZE,
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


def robust_aug(x, y):
    """Random resolution loss and JPEG compression, applied to the training set only.

    CIFAKE images are tiny and blurry, so a model can learn "blurry = real". Randomly
    blurring and recompressing images stops it relying on sharpness or compression cues.
    One random setting is drawn per batch; each effect is applied to half of the batches.
    """
    x = tf.cast(x, tf.float32)

    def blur(img):
        s = tf.random.uniform([], 32, IMG_SIZE + 1, dtype=tf.int32)
        small = tf.image.resize(img, (s, s), method="bilinear", antialias=True)
        return tf.image.resize(small, (IMG_SIZE, IMG_SIZE), method="bilinear")

    def jpeg(img):
        img8 = tf.cast(tf.clip_by_value(img, 0.0, 255.0), tf.uint8)

        def one(im):   # variable-quality JPEG only works on a single image, so map over the batch
            q = tf.random.uniform([], 40, 96, dtype=tf.int32)
            return tf.image.adjust_jpeg_quality(im, q)

        out = tf.map_fn(one, img8, fn_output_signature=tf.uint8, parallel_iterations=16)
        return tf.cast(out, tf.float32)

    x = tf.cond(tf.random.uniform([]) < 0.5, lambda: blur(x), lambda: x)
    x = tf.cond(tf.random.uniform([]) < 0.5, lambda: jpeg(x), lambda: x)
    return x, y


def get_train_val(data_dir=TRAIN_DIR, robust=False):
    train_dir = os.path.join(data_dir, "train")
    train = _load(train_dir, subset="training", validation_split=VAL_SPLIT)
    val = _load(train_dir, subset="validation", validation_split=VAL_SPLIT)
    if robust:
        train = train.map(robust_aug, num_parallel_calls=tf.data.AUTOTUNE)
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
