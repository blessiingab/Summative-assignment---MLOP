import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from .config import (
    IMAGE_SIZE, NUM_CLASSES, CLASS_NAMES, KERAS_MODEL_PATH,
    H5_MODEL_PATH, MODEL_DIR
)
from .datacleaning import validate_image_folder

def retrain(data_dir, epochs=3, batch_size=16, validation_split=0.2):
    """
    Retrain the saved model using new labelled images.

    Expected upload layout:
      data/retrain/
        Cassava Bacterial Blight (CBB)/
        Cassava Brown Streak Disease (CBSD)/
        Cassava Green Mottle (CGM)/
        Cassava Mosaic Disease (CMD)/
        Healthy/

    Uploading data does not start training by itself. Run this function or
    its CLI command to trigger retraining after the upload and validation.
    """
    data_dir = Path(data_dir)
    report = validate_image_folder(data_dir)
    if report["invalid"] or report["unknown_folders"]:
        raise ValueError(
            f"Fix invalid images/unknown folders first. "
            f"Invalid={len(report['invalid'])}, unknown={report['unknown_folders']}"
        )
    if len(report["valid"]) < 10:
        raise ValueError("At least 10 valid labelled images are required for retraining.")

    ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training",
        seed=42,
        shuffle=True
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        shuffle=True
    )
    autotune = tf.data.AUTOTUNE
    ds = ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    if not KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Pre-trained model not found: {KERAS_MODEL_PATH}")
    model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)

    # Fine-tune only the last layers of the existing pre-trained model.
    base = next(layer for layer in model.layers if isinstance(layer, keras.Model))
    base.trainable = True
    for layer in base.layers[:-25]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"]
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    history = model.fit(
        ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(KERAS_MODEL_PATH))
    model.save(str(H5_MODEL_PATH))
    print("Updated model saved:", KERAS_MODEL_PATH)
    print("Updated H5 copy saved:", H5_MODEL_PATH)
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger cassava model retraining")
    parser.add_argument("--data", default="data/retrain", help="Uploaded labelled image folder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    retrain(args.data, epochs=args.epochs, batch_size=args.batch_size)
