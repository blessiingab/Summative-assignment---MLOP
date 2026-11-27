# Cassava Leaf Disease Classifier

An end-to-end machine-learning application for classifying cassava leaf images into five classes. It combines an EfficientNetB0 image classifier, a FastAPI service, and a responsive browser dashboard for prediction, model evaluation, and labelled-data retraining.

**Deployment:** [http://13.49.75.124/](http://13.49.75.124/)


## Contents

- [Features](#features)
- [Classes and evaluation](#classes-and-evaluation)
- [Project structure](#project-structure)
- [Run locally on Windows](#run-locally-on-windows)
- [Use the API](#use-the-api)
- [Upload data and retrain](#upload-data-and-retrain)
- [Run with Docker](#run-with-docker)
- [Deploy to a Linux VM](#deploy-to-a-linux-vm)
- [Configuration and limitations](#configuration-and-limitations)

## Features

- Classify one JPEG, PNG, or WEBP cassava leaf image and return the predicted class, confidence, and class probabilities.
- View service health, validation metrics, and the training-image distribution in the dashboard.
- Upload labelled JPEG or PNG images and trigger asynchronous fine-tuning from the dashboard or API.
- Inspect the API interactively at `/docs`.
- Run locally with Uvicorn reload support or package the service with Docker Compose.

## Model approach

The classifier uses an ImageNet-initialized EfficientNetB0 backbone with a five-class softmax output. Images are converted to RGB and resized to 260 x 260 pixels. During model training, augmentation applies horizontal/vertical flips, small rotations, and zooms before the backbone; the classification head uses global average pooling, dropout, and a regularized 128-unit dense layer.

The retraining pipeline starts from the saved model, fine-tunes the final 25 backbone layers while keeping batch-normalization layers frozen, and uses Adam with a low learning rate, categorical cross-entropy with label smoothing, a 20% validation split, and early stopping. The implementation is in [`Cassava_project/src/model.py`](Cassava_project/src/model.py), [`Cassava_project/src/preprocessing.py`](Cassava_project/src/preprocessing.py), and [`Cassava_project/src/retrain.py`](Cassava_project/src/retrain.py).

## Classes and evaluation

The model predicts one of the following classes:

| Class | Training images |
|---|---:|
| Cassava Bacterial Blight (CBB) | 1,087 |
| Cassava Brown Streak Disease (CBSD) | 2,189 |
| Cassava Green Mottle (CGM) | 2,386 |
| Cassava Mosaic Disease (CMD) | 13,158 |
| Healthy | 2,577 |
| **Total** | **21,397** |

The class counts show a strong imbalance, particularly for CMD. The dashboard therefore reports macro-averaged as well as weighted metrics.

The checked-in evaluation report (`Cassava_project/reports/evaluation_metrics.json`) contains:

| Metric | Score |
|---|---:|
| Accuracy | 76.26% |
| Macro precision | 63.09% |
| Macro recall | 59.90% |
| Macro F1 | 60.33% |
| Weighted precision | 77.26% |
| Weighted recall | 76.26% |
| Weighted F1 | 76.10% |
| Validation loss | 0.808 |

These are the saved evaluation results, not a guarantee of performance on new field images. Fine-tuning changes the model and may change its results.

The notebook used for model development and evaluation is [Cassava Leaf Disease Classification.ipynb](Cassava_project/notebook/Cassava%20Leaf%20Disease%20Classification.ipynb).

## Project structure

```text
.
├── README.md
└── Cassava_project/
    ├── app.py                 # FastAPI routes and application state
    ├── dev.py                 # Local Uvicorn development launcher
    ├── requirements.txt
    ├── Dockerfile
    ├── compose.yaml
    ├── models/
    │   ├── cassava_leaf_model.keras
    │   └── cassava_leaf_model.h5
    ├── notebook/
    │   └── Cassava Leaf Disease Classification.ipynb
    ├── reports/
    │   └── evaluation_metrics.json
    ├── src/
    │   ├── config.py          # Paths, image size, and class labels
    │   ├── model.py           # EfficientNetB0 model loading/building
    │   ├── preprocessing.py   # Image decoding and preprocessing
    │   ├── prediction.py      # Prediction helper and CLI
    │   ├── datacleaning.py    # Label-folder and image validation
    │   └── retrain.py         # Fine-tuning pipeline and CLI
    └── static/
        ├── index.html
        ├── styles.css
        └── app.js
```

`data/` is used at runtime for temporary inference files and uploaded retraining images. It is not the source of the checked-in class-count summary.

## Run locally on Windows

Python 3.11 is recommended for local parity with the Docker image. TensorFlow must have a wheel available for the Python version you use.

From the repository root, create and activate a virtual environment, install dependencies, and start the development launcher:

```powershell
Set-Location .\Cassava_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python dev.py
```

If PowerShell blocks script activation, run the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe dev.py
```

The development launcher serves `app:app` at `http://127.0.0.1:8000`, watches project code, and excludes `.venv`, `data`, and `models` from reload watching. Uvicorn access logs remain enabled for requests. Press **Ctrl+C** in the terminal to stop the server.

Open:

- Dashboard: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

The first prediction can take longer because the model is loaded lazily when it is first needed.

## Use the API

All endpoints are served from the same origin as the dashboard.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Dashboard |
| `GET` | `/api/health` | Service uptime and model-file availability |
| `GET` | `/api/insights` | Saved evaluation metrics and class counts |
| `POST` | `/api/predict` | Predict one image using multipart field `file` |
| `POST` | `/api/retraining/uploads` | Save labelled images using repeated `files` fields |
| `POST` | `/api/retraining` | Validate uploaded data and start fine-tuning |
| `GET` | `/api/retraining/status` | Read the current training state and final metrics |
| `GET` | `/docs` | Swagger UI |

### Predict an image

The prediction endpoint accepts JPEG, PNG, or WEBP images up to 10 MB:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/predict `
  -F "file=@C:\path\to\cassava-leaf.jpg"
```

The response includes `class_id`, `class_name`, `confidence`, and a `probabilities` object with a score for every class.

### Check service health

```powershell
curl.exe http://127.0.0.1:8000/api/health
```

The health response reports whether the model artifact exists, whether it has been loaded, and the server's uptime. A health check does not run a prediction.

## Upload data and retrain

Each image must be inside a folder whose name exactly matches one of the five class labels. For example:

```text
new-cassava-images/
├── Cassava Bacterial Blight (CBB)/
├── Cassava Brown Streak Disease (CBSD)/
├── Cassava Green Mottle (CGM)/
├── Cassava Mosaic Disease (CMD)/
└── Healthy/
```

### Dashboard workflow

1. Select the directory containing the five class folders using **Choose a labelled dataset folder**.
2. Select **Upload labelled images**. Uploading saves the files but does not start training.
3. After at least 10 valid images in total and at least one image in every class have been uploaded, select **Start retraining**.
4. The dashboard checks status while training runs and displays the final metrics or any training error.

Images must be JPEG or PNG for retraining. The API limits each image to 10 MB, each request to 100 MB, and each upload batch to 250 files. Labels are inferred from the parent folder, so check folder names and labels before uploading. Uploaded images remain on the server under `Cassava_project/data/retrain/`.

### API upload example

The upload endpoint accepts repeated `files` fields. The submitted filename must retain the class directory:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/retraining/uploads `
  -F "files=@C:\datasets\Healthy\leaf.jpg;filename=dataset/Healthy/leaf.jpg"
```

Repeat the `-F "files=..."` argument for additional images and classes. Upload at least 10 valid images total with every class represented before starting training.

Start training and read its status:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/retraining
curl.exe http://127.0.0.1:8000/api/retraining/status
```

Fine-tuning runs in the background. The current pipeline uses a 20% validation split, a maximum of three epochs, batch size 16, and early stopping. On completion, it saves updated `.keras` and `.h5` model files under `models/` and clears the in-memory prediction-model cache so subsequent predictions use the updated model.

Training state and coordination are held in application memory. Run a single application process/replica; multiple workers or replicas do not share the training lock or status.

## Run with Docker

Docker and the Compose plugin are optional for local development. From the repository root:

```powershell
Set-Location .\Cassava_project
docker compose up --build -d
docker compose ps
docker compose logs -f app
```

The Compose service maps host port `80` to container port `8000` and mounts `./data` and `./models` so uploads and model updates are stored on the host. Ensure `models/` contains the checked-in model files: the bind mount can hide files copied into the container image if the host directory is empty.

Check the service and stop it with:

```powershell
curl.exe http://localhost/api/health
docker compose down
```

On Linux, use `sudo` with Docker commands if your account is not in the Docker group.

TensorFlow makes the Docker image and its build cache large. For EC2, allow at least 30 GiB for the root EBS volume. If a build reports `no space left on device`, check `df -hT /` and `sudo docker system df`; `docker compose down` stops containers but does not remove build cache. After increasing an EBS volume, grow the root partition and filesystem using the [AWS Linux volume expansion guide](https://docs.aws.amazon.com/ebs/latest/userguide/recognize-expanded-volume-linux.html). `sudo docker builder prune -af` can reclaim unused build cache, but the next build will need to recreate it.

## Deploy to a Linux VM

The same Compose setup can be used on a Linux VM such as AWS EC2:

1. Install Docker Engine and the Docker Compose plugin.
2. Clone the repository and enter `Summative-assignment---MLOP/Cassava_project`.
3. Confirm the model files are present in `models/`, then run `sudo docker compose up --build -d`.
4. Allow inbound HTTP on port 80 in the VM firewall/security group.
5. Check `http://<server-address>/api/health` and open `http://<server-address>/`.

The Compose service uses one app instance and plain HTTP. For a public deployment, add HTTPS termination, authentication, suitable resource limits, backups, and monitoring. Uploaded training data and model updates are stored in the mounted host directories; back them up if they must survive host or disk loss.

## Configuration and limitations

- Dependencies are listed in [`Cassava_project/requirements.txt`](Cassava_project/requirements.txt). The Docker image uses Python 3.11.
- On native Windows, TensorFlow runs on CPU; its oneDNN and CPU-feature messages are informational. TensorFlow GPU support on Windows requires a separate supported setup such as WSL2.
- Prediction accepts WEBP, but retraining uploads accept JPEG and PNG only.
- Retraining updates the local model files. Keep backups of `models/` before training if you need to preserve the original model.
- The API has no built-in authentication or multi-replica training coordination.
- Evaluation scores are historical report values and should not be treated as a guarantee for a particular farm, camera, or field condition.
