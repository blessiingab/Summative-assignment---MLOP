import argparse
import numpy as np
from .config import CLASS_NAMES, KERAS_MODEL_PATH
from .model import load_model
from .preprocessing import preprocess_image

def predict_image(image_path, model=None):
    model = model or load_model(KERAS_MODEL_PATH, compile=False)
    batch = preprocess_image(image_path)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_id = int(np.argmax(probabilities))
    return {
        "class_id": predicted_id,
        "class_name": CLASS_NAMES[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to one leaf image")
    args = parser.parse_args()
    print(predict_image(args.image))
