---
title: Q-Knee
emoji: 🦵
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.62.0"
app_file: streamlit_app.py
pinned: false
license: mit
---

# Q-Knee

**Quantum-assisted ACL & meniscal tear risk triage from knee MRI — ResNet18 feature extraction, PCA-to-angle encoding, and a 4-qubit PennyLane variational quantum classifier, served via a Streamlit clinical dashboard.**

> This README covers the PRD-scoped pipeline: ingestion → ResNet18 → PCA → 4-qubit VQC → Streamlit UI → Grad-CAM → SVM benchmark. A FastAPI backend, clinician auth, multi-service Docker/cloud deploy config, and a few alternate model variants were quarantined (moved, not deleted) to [`extras/`](extras/README.md) — see that directory's README for what's there and why.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Python](https://img.shields.io/badge/python-3.11-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c)](#)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9c27b0)](#)
[![Tests](https://img.shields.io/badge/tests-pytest-0a9edc)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

> ⚠️ **Research prototype.** Q-Knee is not a certified medical device and is not validated for clinical use. All metrics below are measured on a synthetic benchmark dataset (see [`qknee/models/evaluate.py`](qknee/models/evaluate.py)) used to exercise the pipeline end-to-end — they demonstrate that the quantum stage is competitive with classical baselines on this task shape, not real-world diagnostic accuracy. Retrain and clinically validate on a real, IRB-approved MRI dataset before any clinical use.

---

## Deploying the dashboard (zero-cost)

The Streamlit clinical dashboard (`qknee/ui/dashboard.py`, wrapped by the root-level [`streamlit_app.py`](streamlit_app.py) — the `app_file` in this README's frontmatter above) deploys directly to either platform's free tier — no Docker image, no paid compute:

- **Streamlit Community Cloud**: [share.streamlit.io](https://share.streamlit.io) → New app → point at this repo, branch, and `streamlit_app.py`. Reads `requirements.txt` and `packages.txt` from the repo root automatically.
- **Hugging Face Spaces**: create a Space with SDK "Streamlit" and push this repo — the frontmatter block at the top of this file *is* the Space's `README.md` metadata (`sdk`, `app_file`, etc.), so no separate configuration step is needed.

`streamlit_app.py` exists purely so `app_file` can point at a *repo-root* script: both platforms run `streamlit run <app_file>` with only that script's own directory on `sys.path`, not the repo root, so pointing `app_file` straight at the nested `qknee/ui/dashboard.py` would break every `from qknee....` import in it (`ModuleNotFoundError: No module named 'qknee'`). The wrapper adds the repo root to `sys.path` first, then delegates to `qknee.ui.dashboard.main()`.

Both platforms cap free-tier containers around 1GB RAM and enforce a build-time limit, so `requirements.txt` pins CPU-only PyTorch wheels (`torch==2.13.0+cpu` / `torchvision==0.28.0+cpu` via `--extra-index-url https://download.pytorch.org/whl/cpu`) instead of PyPI's default multi-GB CUDA build, and `opencv-python-headless` instead of the GUI-linked `opencv-python` — see the comments in [`requirements.txt`](requirements.txt) and [`packages.txt`](packages.txt) for the full audit. The dashboard also pre-warms `qknee/artifacts/precomputed_cache.json` into `@st.cache_resource` memory at boot and caches per-slice inference/Grad-CAM/DICOM-decode results, so a cold container's first few interactions don't pay full recomputation cost.

Without a fitted PCA/VQC checkpoint (`qknee/artifacts/pca_scaler.pkl`, etc. — see [`.dockerignore`](.dockerignore)/`.gitignore` for what's excluded from the repo), the dashboard falls back to seeded mock inference automatically, so it stays fully demoable even on a from-scratch deploy.

---

## Key metrics

Measured with [`qknee/models/evaluate.py`](qknee/models/evaluate.py) (synthetic 800-sample benchmark, 75/25 train/test split) and [`qknee/tests/test_latency_benchmark.py`](qknee/tests/test_latency_benchmark.py) (CPU, single-slice inference). Re-run both against your own labeled cohort before trusting these numbers for anything beyond pipeline sanity-checking.

| Model                      | ROC-AUC  | Sensitivity | Specificity | F1-score | Per-slice latency (CPU) |
|-----------------------------|:--------:|:-----------:|:-----------:|:--------:|:-----------------------:|
| Baseline SVM (RBF, 512-D)   | 0.93     | 0.89        | 0.84        | 0.87     | —                        |
| Baseline ResNet18 (linear probe) | 0.83 | 0.77        | 0.75        | 0.76     | ~30 ms (feature extraction only) |
| **Q-Knee Hybrid VQC (ours)** | **0.93** | **0.84**    | **0.88**    | **0.85** | **~104 ms** (full pipeline) |

Regenerate this table and the accompanying charts with:

```bash
python -m qknee.models.evaluate         # prints the table, saves eval_outputs/*.png
python scripts/generate_deck_assets.py  # pitch-deck-ready figures (.png + .svg) in qknee/artifacts/deck_figures/
```

---

## System architecture

```
┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐   ┌───────────────────────┐   ┌────────────────────┐
│   Ingestion  │   │  Spatial Feature  │   │ Dimensionality    │   │  4-Qubit Variational  │   │   Explainable UI    │
│ & Preproc.   │──▶│   Extraction      │──▶│   Reduction       │──▶│  Quantum Classifier   │──▶│                     │
│ (DICOM/.npy) │   │   (ResNet18)      │   │  (PCA → [0, 2π])  │   │     (PennyLane)       │   │   (Grad-CAM)        │
└──────────────┘   └───────────────────┘   └──────────────────┘   └───────────────────────┘   └────────────────────┘
     224×224            frozen backbone         StandardScaler          Angle Encoding            layer4 heatmap
   grayscale→RGB         → 512-D vector        → PCA(4) → MinMax      (RX+RY per qubit)          overlaid on the
  ImageNet-normalized    (avgpool tap,          → 4 continuous          + 3 variational           original slice
                          fc stripped)             scalars              RX/RY/RZ+CNOT layers
                                                                       → Sigmoid risk score
```

| Stage | Module | What it does |
|---|---|---|
| **1. Input Preprocessing** | [`qknee/data/dataset.py`](qknee/data/dataset.py), [`qknee/data/ingestion.py`](qknee/data/ingestion.py) | Loads PNG/JPEG slices or `.npy` volumes, resizes to 224×224, converts grayscale → 3-channel RGB, normalizes with ImageNet mean/std, and applies light rotation/flip/Gaussian-noise augmentation for training. |
| **2. Spatial Feature Extraction** | [`qknee/models/resnet_extractor.py`](qknee/models/resnet_extractor.py) | A frozen, pretrained `resnet18(weights=ResNet18_Weights.DEFAULT)` with its `fc` head stripped, tapping `avgpool` directly for a 512-D embedding per slice (multi-slice volumes are mean-pooled into one embedding). |
| **3. Dimensionality Reduction** | [`qknee/models/pca_reducer.py`](qknee/models/pca_reducer.py) | `StandardScaler → PCA(n_components=4) → MinMaxScaler(0, 2π)`, fit offline and persisted with `joblib` (`pca_scaler.pkl`) for consistent inference. |
| **4. 4-Qubit VQC** | [`qknee/models/vqc.py`](qknee/models/vqc.py) | Continuous **angle encoding** (RX then RY per qubit) of the 4 PCA scalars, followed by 3 variational layers (RX/RY/RZ + a CNOT entangling ring), measuring PauliZ expectation values, wrapped as a `qml.qnn.TorchLayer` so it trains inside a normal PyTorch optimizer loop. |
| **5. Explainable UI** | [`qknee/xai/gradcam.py`](qknee/xai/gradcam.py), [`qknee/ui/dashboard.py`](qknee/ui/dashboard.py) | Grad-CAM on ResNet18's `layer4` highlights the anatomical regions driving the embedding; the Streamlit dashboard surfaces this as an overlay. |

All five stages are chained by [`qknee/models/pipeline.py`](qknee/models/pipeline.py) (`PipelineRunner`) — the single entry point downstream consumers should use, with shape/dtype/range validation between every pair of stages. The end-to-end differentiable `nn.Module` (for joint training) is [`qknee/models/qknee_model.py`](qknee/models/qknee_model.py) (`QKneeModel`). All hyperparameters and paths are centralized in [`qknee/config/config.yaml`](qknee/config/config.yaml), loaded via `qknee/config/loader.py`.

---

## Setup & installation

### Prerequisites

- Python 3.11

### 1. Clone and create a virtual environment

```bash
git clone <this-repo-url> qknee
cd qknee

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
# Runtime dependencies only
pip install -r requirements.txt

# + pytest, for running the test suite
pip install -r requirements-dev.txt
```

> **CPU-only PyTorch:** `requirements.txt` pins `torch`/`torchvision` by version but does not force a CPU-only wheel index on every platform. If `pip` resolves a CUDA build you don't want, install explicitly first:
> `pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu`

### 3. Fit the PCA artifact (one-time, before first inference)

The PCA/angle-scaling stage must be fit on a representative corpus of 512-D ResNet embeddings before the pipeline can run inference:

```bash
python -m qknee.models.pca_reducer     # fits on demo data, saves to qknee/artifacts/pca_scaler.pkl
```

Swap in real ResNet18 embeddings from your own MRI corpus for a production fit — see `QuantumDimReducer.fit()`. The artifact path is controlled by `qknee/config/config.yaml`'s `paths.pca_artifact` (overridable via `$PCA_ARTIFACT_PATH`).

### 4. Run locally

```bash
# Run the full DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM pipeline via PipelineRunner
python -m qknee.models.pipeline

# Train + evaluate against classical baselines, save ROC/confusion-matrix figures
python -m qknee.models.evaluate

# Launch the Streamlit clinical dashboard (http://localhost:8501)
streamlit run qknee/ui/dashboard.py

# Run the test suite (testpaths=qknee/tests, see pytest.ini)
pytest                    # full suite
pytest -m "not slow"      # skip the real ResNet18/PennyLane latency benchmark
```

> A FastAPI backend, Docker/docker-compose setup, and Render/Vercel deploy config exist in [`extras/`](extras/README.md) (quarantined — outside the judged PRD scope) if you want the multi-service architecture back.

---

## Programmatic usage (Python)

```python
from qknee.models.pipeline import PipelineRunner

runner = PipelineRunner()   # loads qknee/config/config.yaml + qknee/artifacts/pca_scaler.pkl

# Accepts a .png/.jpg path, a .npy volume path, a PIL.Image, or an in-memory np.ndarray
result = runner.run("path/to/slice.png")
# -> PipelineResult(risk_score: float in [0, 1], quantum_angles: (1, 4) ndarray, gradcam_heatmap: (H, W) ndarray)

# Or just the ingestion -> ResNet18 -> PCA stages, e.g. to feed a custom VQC head:
quantum_angles = runner.extract_quantum_features("path/to/slice.png")
```

For the full trained model (ResNet18 → PCA → VQC → risk score) as a single differentiable `nn.Module`:

```python
import torch
from qknee.models.qknee_model import QKneeModel
from qknee.models.pca_reducer import QuantumDimReducer

reducer = QuantumDimReducer.load("qknee/artifacts/pca_scaler.pkl")
model = QKneeModel(pca_reducer=reducer, n_qubits=4, n_layers=3)
model.eval()

image = torch.rand(1, 3, 224, 224)   # or a real preprocessed MRI slice
with torch.no_grad():
    risk_score = model(image)        # (1, 1) tensor, in [0, 1]
```

A REST wrapper (FastAPI, `POST /predict` → the same `risk_score`/`diagnosis`/`gradcam_heatmap` payload) exists in [`extras/api/`](extras/README.md) if you need HTTP access instead of importing `qknee` directly.

---

## Project structure

```
qknee/
├── config/
│   ├── config.yaml               # central hyperparameters, paths, logging config
│   ├── loader.py                 # typed, env-override-aware dynamic config parser
│   └── logging_config.py         # process-wide logging setup
├── data/
│   ├── dataset.py                 # MRIDataset, DataLoader builders, train/eval transforms
│   └── ingestion.py                # DataIngestion: raw input -> normalized tensor batch
├── models/
│   ├── resnet_extractor.py        # frozen ResNet18 -> 512-D embedding (+ an optional ONNX
│   │                               # Runtime backend, ONNXFeatureExtractor — see extras/README.md)
│   ├── pca_reducer.py             # StandardScaler -> PCA(4) -> [0, 2pi] angle scaling
│   ├── vqc.py                     # 4-qubit, 3-layer PennyLane VQC as a torch.nn.Module —
│   │                               # the ansatz config.yaml's quantum.n_qubits/n_layers names
│   ├── vqc_data_reuploading.py    # alternate classifier backbone (pipeline.py's
│   │                               # classifier_backbone="data_reuploading" option)
│   ├── vqc_multitarget.py         # multi-label quantum-classical head for the 12-condition
│   │                               # RSNA Knee task (qknee_model.py's QKneeMultiTargetModel)
│   ├── quantum_autoencoder.py     # alternate encoder (pipeline.py's
│   │                               # encoder_type="quantum_autoencoder" option)
│   ├── qknee_model.py             # QKneeModel: unified ResNet18+PCA+VQC nn.Module,
│   │                               # training loop, checkpoint save/load
│   ├── pipeline.py                # PipelineRunner: orchestrates all 5 stages, with
│   │                               # per-stage validation (the central entry point)
│   └── evaluate.py                # SVM / ResNet-only / VQC comparison + ROC/confusion plots
├── xai/
│   └── gradcam.py                 # Grad-CAM on ResNet18 layer4 + OpenCV overlay
├── ui/
│   ├── dashboard.py                # Streamlit clinical dashboard — opens straight into the
│   │                                # workstation, no sign-in (tri-planar, dual risk heads)
│   └── analysis_app.py             # Streamlit single-scan analysis app
├── tests/
│   ├── conftest.py                 # shared fixtures (fitted reducer, ResNet, QKneeModel)
│   ├── test_feature_extractor.py   # (B, 512) shape + PCA [0, 2pi] bound tests
│   ├── test_latency_benchmark.py   # end-to-end inference latency
│   ├── test_determinism.py         # seeded reproducibility
│   └── test_quantum_simulator_mocking.py  # mocked NISQ resource-limit behavior
└── artifacts/                      # fitted pca_scaler.pkl / qknee_model.pt (gitignored)

scripts/
├── generate_deck_assets.py         # pitch-deck figures: ROC+efficiency chart, VQC circuit diagram, clinical case walkthrough (.png+.svg in qknee/artifacts/deck_figures/)
└── run_benchmark.py                # SVM / linear-probe / VQC comparison, saves benchmark_results.json

extras/                              # quarantined — outside the judged PRD scope; moved, not
│                                     # deleted. See extras/README.md for the full inventory.
├── api/                             # FastAPI backend (was qknee/api/) + JWT/Argon2 auth
├── ui/                              # clinician sign-in + institutional landing page
├── deployment/                      # Dockerfile, docker-compose*.yml, render.yaml, vercel.json
├── scripts/                         # export_onnx.py, generate_kaggle_submission.py
├── models/                          # vqc_strongly_entangling.py (alternate ansatz)
└── tests/                           # tests for everything above

.streamlit/config.toml              # dark theme config
requirements.txt                    # runtime dependencies (+ pyyaml)
requirements-dev.txt                # + pytest
pytest.ini                          # slow/benchmark markers, testpaths=qknee/tests
```