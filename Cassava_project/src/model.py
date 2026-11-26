import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .config import IMAGE_SIZE, NUM_CLASSES, KERAS_MODEL_PATH

def build_model(weights="imagenet"):
    base = keras.applications.EfficientNetB0(
        include_top=False, weights=weights,
        input_shape=(*IMAGE_SIZE, 3)
    )
    base.trainable = False
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="leaf_image")
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.10)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name="cassava_efficientnetb0")

def load_model(model_path=KERAS_MODEL_PATH, compile=False):
    return keras.models.load_model(model_path, compile=compile)
