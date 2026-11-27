# Cassava Leaf Disease Classifier

An end-to-end image classification project for identifying five cassava leaf classes. The repository includes an EfficientNetB0 model, notebook evaluation, a FastAPI prediction/retraining API, and a plain HTML/CSS/JavaScript dashboard.

## What is included

- Single-image prediction with the class probabilities.
- Health endpoint with API uptime and model artifact status.
- Dashboard visualizations for the five-class dataset and measured evaluation metrics.
- Bulk labelled-image upload and a button-triggered fine-tuning process using the saved model.
- FastAPI's interactive endpoint documentation at `/docs`.
- Docker packaging for a cloud container deployment.

## Model and evaluation

The model is an EfficientNetB0 transfer-learning classifier with five classes: Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD), Cassava Green Mottle (CGM), Cassava Mosaic Disease (CMD), and Healthy.

The notebook is [Cassava_project/notebook/Cassava Leaf Disease Classification.ipynb](Cassava_project/notebook/Cassava%20Leaf%20Disease%20Classification.ipynb). The checked-in evaluation report contains:

| Metric | Score |
|---|---:|
| Accuracy | 76.26% |
| Macro precision | 63.09% |
| Macro recall | 59.90% |
| Macro F1 | 60.33% |
| Weighted F1 | 76.10% |
| Validation loss | 0.808 |

The 21,397-image training distribution is imbalanced: 13,158 CMD images, 2,577 Healthy, 2,386 CGM, 2,189 CBSD, and 1,087 CBB. Macro metrics and per-class predictions are included so that performance on the less common classes is not hidden by accuracy.

## Run locally

Requirements: Python 3.10 or newer. TensorFlow may require additional memory and takes time to load the first prediction.

```powershell
cd Cassava_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the UI or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the API docs. The trained `.keras` model is in `Cassava_project/models/`.

## Prediction API

`POST /api/predict` accepts one multipart image under the `file` field (JPEG, PNG, or WEBP, up to 10 MB). A successful response contains the predicted class, confidence, and a probability for each class.

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/predict -F "file=@leaf.jpg"
```

`GET /api/health` reports service uptime and model availability. `GET /api/insights` returns the notebook's evaluation scores and dataset distribution.

## Upload labelled data and retrain

Select a directory with this structure in the dashboard. Directory upload preserves the class folder names:

```text
new-cassava-images/
├── Cassava Bacterial Blight (CBB)/
├── Cassava Brown Streak Disease (CBSD)/
├── Cassava Green Mottle (CGM)/
├── Cassava Mosaic Disease (CMD)/
└── Healthy/
```

Images must be JPEG or PNG and no larger than 10 MB each. Uploading saves them under `Cassava_project/data/retrain/<class>/`; it does not begin training automatically. Once at least 10 valid images and at least one image from each class are present, press **Start retraining**. The API fine-tunes the checked-in model, uses early stopping, saves updated `.keras` and `.h5` artifacts, and invalidates the prediction model cache when complete.

The same flow is available through:

- `POST /api/retraining/uploads` with repeated `files` multipart fields; filenames must retain the class directory (for example `dataset/Healthy/leaf.jpg`).
- `POST /api/retraining` to validate the saved files and start asynchronous fine-tuning.
- `GET /api/retraining/status` to read the training state and final training metrics.

Uploads are limited to 250 files and 100 MB per request. Keep uploaded data labelled correctly; training inherits the labels from the directory names.

## Docker and cloud deployment

The Docker setup lives in `Cassava_project/Dockerfile`, `Cassava_project/.dockerignore`, and `Cassava_project/compose.yaml`. From `Cassava_project`, start or update the app with:

```bash
sudo docker compose up -d --build
sudo docker compose ps
curl http://localhost/api/health
```

Compose publishes host port `80` to container port `8000` and bind-mounts `./data` and `./models`, preserving uploaded images and model updates on the instance. Keep a single app replica; retraining state and coordination are in-process.

### AWS EC2 deployment

1. Use an Ubuntu EC2 instance and install Docker Engine with the Compose plugin.
2. Before building, make sure the root EBS volume is at least 30 GiB. If you increased the volume size in AWS, expand the root partition and filesystem on the instance:

   ```bash
   sudo growpart /dev/nvme0n1 1
   sudo resize2fs /dev/nvme0n1p1
   df -hT /
   ```

3. Clone the repository and enter `Cassava_project`:

   ```bash
   git clone https://github.com/blessiingab/Summative-assignment---MLOP.git
   cd Summative-assignment---MLOP/Cassava_project
   ```

4. Build and start the service:

   ```bash
   sudo docker compose up -d --build
   ```

5. In the EC2 security group, allow inbound HTTP on port `80`; restrict SSH on port `22` to your IP. Open `http://YOUR_EC2_PUBLIC_IP/` and verify `http://YOUR_EC2_PUBLIC_IP/api/health`.

To deploy updates, pull the new commit and run `sudo docker compose up -d --build` again. Keep the instance's EBS volume; the Compose bind mounts store uploads and models under `Cassava_project/data` and `Cassava_project/models`. Back these up if they must survive instance or volume loss.

### Render deployment

The repository includes [render.yaml](render.yaml), so the quickest setup is:

1. Push the repository to GitHub.
2. In Render, choose **New > Blueprint** and connect this GitHub repository.
3. Render will detect `render.yaml` and create a Docker web service named `cassava-leaf-classifier`.
4. Review the service plan and region, then select **Apply**.
5. Wait for the Docker build to finish. The build context is `Cassava_project`, and the container listens on port `8000`.
6. Open the Render-generated URL. Use `/api/health` to verify the model artifact.

The checked-in model is included in the Docker image, so no environment variables or secret values are required for prediction. Use one instance and no persistent disk for the initial deployment: the app is memory-backed and retraining state is not shared across replicas. If Render restarts or redeploys the service, uploaded retraining data and retrained model files in the container filesystem can be lost.

For retraining data and retrained `.keras`/`.h5` files to survive restarts, add a paid Render disk under **Disks**:

- Mount `/app/data` for uploaded training images.
- Mount `/app/models` only after updating the app to seed the shipped model into an initially empty model disk. Mounting `/app/models` directly hides the model copied into the Docker image.

Until that disk initialization is implemented, a disk is recommended only for `/app/data`; keep the checked-in model in `/app/models` and allow retraining to use the existing image model.

## Flood/load testing

The API is compatible with Locust or other HTTP load-test clients. Point test traffic at `POST /api/predict` with multipart images and record users, request rate, failures, and latency percentiles from the load-test report. Compare runs at different replica counts on the target cloud platform; prediction latency depends on the selected instance, accelerator, model warm-up, and concurrency. No load-test measurements or public service URL are included because no cloud deployment or benchmark has been run for this repository.

## Demo

Video demo: add the submitted YouTube link here after recording the prediction and retraining workflow.
