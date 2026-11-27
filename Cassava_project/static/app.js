const $ = (selector) => document.querySelector(selector);
const classNames = [
  "Cassava Bacterial Blight (CBB)",
  "Cassava Brown Streak Disease (CBSD)",
  "Cassava Green Mottle (CGM)",
  "Cassava Mosaic Disease (CMD)",
  "Healthy",
];
let selectedPrediction = null;
let selectedDataset = [];
let trainingPollTimer = null;

function formatDuration(seconds) {
  const total = Math.floor(seconds || 0);
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return days ? `${days}d ${hours}h` : hours ? `${hours}h ${minutes}m` : `${minutes}m ${total % 60}s`;
}

async function parseApiResponse(response, action) {
  const body = await response.text();
  let result;
  try {
    result = JSON.parse(body);
  } catch {
    if (!response.ok) {
      const contentType = response.headers.get("content-type") || "";
      const detail = contentType.includes("text/plain") ? body.trim() : "";
      throw new Error(detail
        ? `${action} failed: ${detail}`
        : `${action} failed (HTTP ${response.status}).`);
    }
    throw new Error(`${action} returned an invalid response.`);
  }
  if (!response.ok) {
    throw new Error(result.detail || `${action} failed (HTTP ${response.status}).`);
  }
  return result;
}

function renderInsights(data) {
  $("#dataset-summary").textContent =
    `${data.dataset.name} · ${data.dataset.image_count.toLocaleString()} labelled training images`;
  $("#dataset-total").textContent = `${data.dataset.image_count.toLocaleString()} images`;
  $("#validation-accuracy").textContent = `${(data.evaluation.accuracy * 100).toFixed(1)}%`;

  const metricNames = [
    ["Accuracy", "accuracy"],
    ["Macro precision", "precision_macro"],
    ["Macro recall", "recall_macro"],
    ["Macro F1", "f1_macro"],
  ];
  $("#metrics-grid").replaceChildren(...metricNames.map(([label, key]) => {
    const item = document.createElement("div");
    item.className = "metric";
    const caption = document.createElement("span");
    caption.textContent = label;
    const value = document.createElement("strong");
    value.textContent = `${(data.evaluation[key] * 100).toFixed(1)}%`;
    item.append(caption, value);
    return item;
  }));

  const counts = Object.entries(data.dataset.class_counts);
  const maximum = Math.max(...counts.map(([, count]) => count));
  $("#class-chart").replaceChildren(...counts.map(([name, count]) => {
    const row = document.createElement("div");
    row.className = "chart-row";
    const label = document.createElement("span");
    label.className = "chart-label";
    label.title = name;
    label.textContent = name;
    const track = document.createElement("div");
    track.className = "chart-track";
    const bar = document.createElement("span");
    bar.style.width = `${(count / maximum) * 100}%`;
    track.append(bar);
    const amount = document.createElement("span");
    amount.className = "chart-count";
    amount.textContent = count.toLocaleString();
    row.append(label, track, amount);
    return row;
  }));

  $("#interpretations").replaceChildren(...data.interpretations.map((insight) => {
    const item = document.createElement("div");
    item.className = "interpretation";
    const text = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = `${insight.title}. `;
    text.append(title, document.createTextNode(insight.text));
    item.append(text);
    return item;
  }));

  $("#class-key-list").replaceChildren(...classNames.map((name) => {
    const pill = document.createElement("span");
    pill.className = "class-pill";
    pill.textContent = name;
    return pill;
  }));
}

async function refreshHealth() {
  const status = $("#service-status");
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    status.textContent = response.ok ? "API online" : "Model unavailable";
    status.parentElement.classList.toggle("online", response.ok);
    status.parentElement.classList.toggle("offline", !response.ok);
    $("#api-uptime").textContent = formatDuration(health.uptime_seconds);
    $("#model-status").textContent = health.model_available ? "Available" : "Missing";
  } catch {
    status.textContent = "API offline";
    status.parentElement.classList.remove("online");
    status.parentElement.classList.add("offline");
    $("#model-status").textContent = "Unknown";
  }
}

async function loadInsights() {
  try {
    const response = await fetch("/api/insights");
    if (!response.ok) throw new Error("Could not load evaluation details.");
    renderInsights(await response.json());
  } catch (error) {
    $("#dataset-summary").textContent = error.message;
  }
}

$("#prediction-file").addEventListener("change", (event) => {
  selectedPrediction = event.target.files[0] || null;
  $("#predict-button").disabled = !selectedPrediction;
  $("#prediction-result").classList.add("hidden");
  $("#prediction-message").textContent = "";
  if (!selectedPrediction) {
    $("#prediction-file-label").textContent = "Choose a leaf image";
    $("#prediction-preview-wrap").classList.add("hidden");
    return;
  }
  $("#prediction-file-label").textContent = selectedPrediction.name;
  $("#prediction-preview").src = URL.createObjectURL(selectedPrediction);
  $("#prediction-preview-wrap").classList.remove("hidden");
});

$("#predict-button").addEventListener("click", async () => {
  if (!selectedPrediction) return;
  const button = $("#predict-button");
  button.disabled = true;
  button.classList.add("loading");
  button.firstChild.textContent = "Classifying… ";
  $("#prediction-message").textContent = "";
  const form = new FormData();
  form.append("file", selectedPrediction);
  try {
    const response = await fetch("/api/predict", { method: "POST", body: form });
    const result = await parseApiResponse(response, "Prediction");
    $("#result-class").textContent = result.class_name;
    $("#result-confidence").textContent = `${(result.confidence * 100).toFixed(1)}%`;
    const scores = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
    $("#probability-list").replaceChildren(...scores.map(([name, score]) => {
      const row = document.createElement("div");
      row.className = "probability-row";
      const label = document.createElement("span");
      label.textContent = name;
      const value = document.createElement("span");
      value.textContent = `${(score * 100).toFixed(1)}%`;
      const track = document.createElement("div");
      track.className = "probability-track";
      const bar = document.createElement("span");
      bar.style.width = `${Math.max(0, Math.min(score * 100, 100))}%`;
      track.append(bar);
      row.append(label, value, track);
      return row;
    }));
    $("#prediction-result").classList.remove("hidden");
  } catch (error) {
    $("#prediction-message").textContent = error.message;
  } finally {
    button.disabled = !selectedPrediction;
    button.classList.remove("loading");
    button.firstChild.textContent = "Classify leaf ";
  }
});

$("#dataset-files").addEventListener("change", (event) => {
  selectedDataset = Array.from(event.target.files || []);
  $("#dataset-file-label").textContent = selectedDataset.length
    ? `${selectedDataset.length} image${selectedDataset.length === 1 ? "" : "s"} selected`
    : "Choose a labelled dataset folder";
  $("#dataset-file-hint").textContent = selectedDataset.length
    ? "Files are grouped by their parent class folder."
    : "Folder structure: class name / image";
  $("#upload-button").disabled = selectedDataset.length === 0;
});

$("#upload-button").addEventListener("click", async () => {
  if (!selectedDataset.length) return;
  const button = $("#upload-button");
  button.disabled = true;
  button.classList.add("loading");
  button.firstChild.textContent = "Uploading… ";
  const form = new FormData();
  selectedDataset.forEach((file) => {
    form.append("files", file, file.webkitRelativePath || file.name);
  });
  try {
    const response = await fetch("/api/retraining/uploads", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Upload failed.");
    $("#training-status-text").textContent = `${result.uploaded_count} images saved. Ready to retrain.`;
    $("#training-status").classList.add("online");
  } catch (error) {
    $("#training-status-text").textContent = error.message;
    $("#training-status").classList.add("failed");
  } finally {
    button.disabled = selectedDataset.length === 0;
    button.classList.remove("loading");
    button.firstChild.textContent = "Upload labelled images ";
  }
});

async function refreshTrainingStatus() {
  const response = await fetch("/api/retraining/status");
  if (!response.ok) return;
  const state = await response.json();
  const status = $("#training-status");
  const text = $("#training-status-text");
  status.classList.toggle("online", state.status === "completed");
  status.classList.toggle("failed", state.status === "failed");
  if (state.status === "running") {
    text.textContent = "Fine-tuning model… this can take a few minutes.";
    $("#retrain-button").disabled = true;
    if (trainingPollTimer === null) {
      trainingPollTimer = setInterval(() => refreshTrainingStatus().catch(() => {}), 10000);
    }
  } else {
    if (trainingPollTimer !== null) {
      clearInterval(trainingPollTimer);
      trainingPollTimer = null;
    }
    $("#retrain-button").disabled = false;
    if (state.status === "completed") {
      text.textContent = `Retraining complete · ${new Date(state.finished_at).toLocaleTimeString()}`;
      if (state.metrics) {
        $("#training-metrics").textContent = Object.entries(state.metrics)
          .map(([name, value]) => `${name}: ${Number(value).toFixed(4)}`).join(" · ");
        $("#training-metrics").classList.remove("hidden");
      }
      refreshHealth();
    } else if (state.status === "failed") {
      text.textContent = `Retraining failed: ${state.error}`;
    } else if (!text.textContent.includes("images saved")) {
      text.textContent = "Ready when you are";
    }
  }
}

$("#retrain-button").addEventListener("click", async () => {
  const button = $("#retrain-button");
  button.disabled = true;
  try {
    const response = await fetch("/api/retraining", { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Could not start retraining.");
    $("#training-status").classList.remove("failed");
    $("#training-status-text").textContent =
      `Training started with ${result.training_images} labelled images.`;
    await refreshTrainingStatus();
  } catch (error) {
    $("#training-status").classList.add("failed");
    $("#training-status-text").textContent = error.message;
    button.disabled = false;
  }
});

loadInsights();
refreshHealth();
refreshTrainingStatus().catch(() => {});
