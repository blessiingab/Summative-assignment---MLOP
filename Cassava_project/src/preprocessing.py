from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from .config import IMAGE_SIZE

def preprocess_image(image_path):
    """Read an image as RGB and return a float32 batch in the 0-255 range."""
    image = Image.open(image_path).convert("RGB").resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)

def decode_path(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, antialias=False)
    return tf.cast(image, tf.float32), label
