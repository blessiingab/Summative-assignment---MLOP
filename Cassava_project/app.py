from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from src.config import CLASS_NAMES, KERAS_MODEL_PATH, PROJECT_ROOT, RETRAIN_DIR
from starlette.concurrency import run_in_threadpool

STATIC_DIR = PROJECT_ROOT / "static"
DATA_DIR = PROJECT_ROOT / "data"
METRICS_PATH = PROJECT_ROOT / "reports" / "evaluation_metrics.json"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BATCH_BYTES = 100 * 1024 * 1024
MAX_BATCH_FILES = 250
ALLOWED_IMAGE_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}

CLASS_COUNTS = {
    "Cassava Bacterial Blight (CBB)": 1087,
    "Cassava Brown Streak Disease (CBSD)": 2189,
    "Cassava Green Mottle (CGM)": 2386,
    "Cassava Mosaic Disease (CMD)": 13158,
    "Healthy": 2577,
}

app = FastAPI(
    title="Cassava Leaf Disease Classifier",
    description="Image prediction and labelled-image model retraining API.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_started_at = time.monotonic()
_started_at_wall = datetime.now(timezone.utc)
_model = None
_model_lock = threading.Lock()
_retraining_lock = threading.Lock()
_upload_in_progress = False
_retraining_state = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "metrics": None,
}


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from src.model import load_model

            _model = load_model(KERAS_MODEL_PATH, compile=False)
        return _model


def _predict_image(image_path: Path, model):
    from src.prediction import predict_image

    return predict_image(image_path, model=model)


def _validate_image(data: bytes) -> tuple[str, str]:
    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=415,
            detail="Use a JPEG, PNG, or WEBP image.",
        )
    return image_format, ALLOWED_IMAGE_FORMATS[image_format]


def _copy_retraining_state() -> dict:
    with _retraining_lock:
        return dict(_retraining_state)


def _run_retraining() -> None:
    global _model
    try:
        from src.retrain import retrain

        history = retrain(RETRAIN_DIR, epochs=3, batch_size=16)
        metrics = {
            name: float(values[-1])
            for name, values in history.history.items()
            if values
        }
        with _model_lock:
            _model = None
        with _retraining_lock:
            _retraining_state.update(
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=None,
                metrics=metrics,
            )
    except Exception as exc:  # noqa: BLE001 - publish any worker failure through the status API
        with _retraining_lock:
            _retraining_state.update(
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc),
            )


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    model_available = KERAS_MODEL_PATH.is_file()
    return {
        "status": "ok" if model_available else "model_missing",
        "model_available": model_available,
        "model_loaded": _model is not None,
        "uptime_seconds": round(time.monotonic() - _started_at, 1),
        "started_at": _started_at_wall.isoformat(),
    }


@app.get("/api/insights")
def insights() -> dict:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    total = sum(CLASS_COUNTS.values())
    return {
        "dataset": {
            "name": "Cassava Leaf Disease Classification",
            "image_count": total,
            "class_counts": CLASS_COUNTS,
        },
        "evaluation": metrics,
        "interpretations": [
            {
                "title": "Strong class imbalance",
                "text": (
                    f"CMD accounts for {CLASS_COUNTS['Cassava Mosaic Disease (CMD)'] / total:.1%} "
                    "of training images, so accuracy alone can hide weak performance on rare classes."
                ),
            },
            {
                "title": "Rare bacterial blight examples",
                "text": (
                    f"CBB has only {CLASS_COUNTS['Cassava Bacterial Blight (CBB)']:,} "
                    "images (about 5% of the dataset); macro-averaged recall is useful to monitor "
                    "because it gives every disease class equal weight."
                ),
            },
            {
                "title": "Healthy is not the majority",
                "text": (
                    f"Healthy leaves are {CLASS_COUNTS['Healthy'] / total:.1%} of the sample. "
                    "The model must distinguish healthy leaves from four disease classes, not "
                    "just predict a healthy/diseased split."
                ),
            },
        ],
    }


@app.post("/api/predict")
async def predict(file: Annotated[UploadFile, File()]) -> dict:
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Choose an image to classify.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 10 MB or smaller.")

    _, suffix = _validate_image(content)
    prediction_dir = DATA_DIR / "inference"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=prediction_dir, suffix=suffix, delete=False
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)
        model = await run_in_threadpool(_get_model)
        return await run_in_threadpool(_predict_image, temporary_path, model)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/api/retraining/uploads", status_code=201)
async def upload_retraining_images(
    files: Annotated[list[UploadFile], File()],
) -> dict:
    global _upload_in_progress
    if not files:
        raise HTTPException(status_code=400, detail="Select labelled images to upload.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload no more than {MAX_BATCH_FILES} images at a time.",
        )
    with _retraining_lock:
        if _retraining_state["status"] == "running" or _upload_in_progress:
            raise HTTPException(
                status_code=409,
                detail="Wait for the current upload or retraining operation to finish.",
            )
        _upload_in_progress = True

    staging_dir = DATA_DIR / f".retraining-upload-{uuid.uuid4().hex}"
    staged_files: list[tuple[str, Path]] = []
    total_bytes = 0
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for upload in files:
            relative_name = (upload.filename or "").replace("\\", "/")
            path_parts = PurePosixPath(relative_name).parts
            if len(path_parts) < 2 or any(part in {".", ".."} for part in path_parts):
                raise HTTPException(
                    status_code=400,
                    detail="Upload a folder per class, with images inside each class folder.",
                )
            class_name = path_parts[-2]
            if class_name not in CLASS_NAMES.values():
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown class folder: {class_name}",
                )

            content = await upload.read(MAX_IMAGE_BYTES + 1)
            if not content:
                raise HTTPException(status_code=400, detail="An uploaded image is empty.")
            if len(content) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Each image must be 10 MB or smaller.",
                )
            total_bytes += len(content)
            if total_bytes > MAX_BATCH_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="The total upload must be 100 MB or smaller.",
                )

            image_format, suffix = _validate_image(content)
            if image_format == "WEBP":
                raise HTTPException(
                    status_code=415,
                    detail="Training images must be JPEG or PNG.",
                )
            class_staging_dir = staging_dir / class_name
            class_staging_dir.mkdir(parents=True, exist_ok=True)
            staged_path = class_staging_dir / f"{uuid.uuid4().hex}{suffix}"
            staged_path.write_bytes(content)
            staged_files.append((class_name, staged_path))

        for class_name, staged_path in staged_files:
            destination_dir = RETRAIN_DIR / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            staged_path.replace(destination_dir / staged_path.name)
    finally:
        with _retraining_lock:
            _upload_in_progress = False
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        for upload in files:
            await upload.close()

    return {
        "uploaded_count": len(staged_files),
        "class_counts": {
            class_name: sum(1 for name, _ in staged_files if name == class_name)
            for class_name, _ in staged_files
        },
        "message": "Images saved. Start retraining when all five class folders are represented.",
    }


@app.get("/api/retraining/status")
def retraining_status() -> dict:
    return _copy_retraining_state()


@app.post("/api/retraining", status_code=202)
def start_retraining(background_tasks: BackgroundTasks) -> dict:
    if not KERAS_MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="The pre-trained model file is missing.")
    if not RETRAIN_DIR.is_dir():
        raise HTTPException(status_code=400, detail="Upload labelled training images first.")

    from src.datacleaning import validate_image_folder

    report = validate_image_folder(RETRAIN_DIR)
    if report["invalid"] or report["unknown_folders"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Fix {len(report['invalid'])} invalid images and remove "
                f"unknown folders: {report['unknown_folders']}."
            ),
        )
    class_counts = {
        class_name: sum(
            1
            for image_path in report["valid"]
            if Path(image_path).parent.name == class_name
        )
        for class_name in CLASS_NAMES.values()
    }
    if sum(class_counts.values()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Upload at least 10 valid labelled images before retraining.",
        )
    missing_classes = [name for name, count in class_counts.items() if not count]
    if missing_classes:
        raise HTTPException(
            status_code=400,
            detail=f"Add at least one image for each class. Missing: {missing_classes}",
        )

    with _retraining_lock:
        if _retraining_state["status"] == "running":
            raise HTTPException(status_code=409, detail="Retraining is already in progress.")
        if _upload_in_progress:
            raise HTTPException(status_code=409, detail="Wait for the current image upload to finish.")
        _retraining_state.update(
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            error=None,
            metrics=None,
        )
    background_tasks.add_task(_run_retraining)
    return {
        "status": "running",
        "training_images": sum(class_counts.values()),
        "class_counts": class_counts,
        "message": "Fine-tuning started. Check the retraining status for progress.",
    }
