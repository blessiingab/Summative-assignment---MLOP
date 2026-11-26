from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RETRAIN_DIR = DATA_DIR / "retrain"
IMAGE_SIZE = (260, 260)
NUM_CLASSES = 5
CLASS_NAMES = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}
KERAS_MODEL_PATH = MODEL_DIR / "cassava_leaf_model.keras"
H5_MODEL_PATH = MODEL_DIR / "cassava_leaf_model.h5"
