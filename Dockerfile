# syntax=docker/dockerfile:1
#
# Q-Knee container image.
#
# One shared multi-stage image serves BOTH the FastAPI backend (api.py) and
# the Streamlit frontend (app.py) — they share the exact same heavy ML
# dependency stack (PyTorch, PennyLane, OpenCV, scikit-learn), so building
# two separate images would just duplicate multiple GB of identical layers
# for no benefit. docker-compose.yml runs two containers *from this one
# image*, each with a different `command:` override.
#
# Build:   docker build -t qknee:latest .
# Run API: docker run -p 8000:8000 qknee:latest
# Run UI:  docker run -p 8501:8501 qknee:latest streamlit run app.py --server.port=8501 --server.address=0.0.0.0


# --------------------------------------------------------------------------- #
# Stage 1: builder
#   Installs all Python dependencies into an isolated virtualenv. Kept
#   separate from the runtime stage so build-only tooling (a C compiler,
#   pip's wheel cache, apt package lists) never ends up in the final image.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

# build-essential: some scientific-Python deps (e.g. scipy, which PennyLane
# and scikit-learn depend on) fall back to compiling from source on
# platforms without a prebuilt manylinux wheel; keeping this in the builder
# stage only (not runtime) avoids bloating the final image with a toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv (rather than installing into the system site-packages)
# so the whole dependency tree can be copied as one clean directory into the
# runtime stage below.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .

# Install PyTorch/torchvision from the official CPU-only wheel index first.
# This is the single most important line for image size and NISQ-simulator
# CPU deployments: the default PyPI wheel can pull in multi-GB NVIDIA CUDA
# libraries that are dead weight on a CPU-only host (PennyLane's
# `default.qubit` simulator and this project's ResNet18 backbone both run
# on CPU here — there is no GPU workload to justify the CUDA runtime).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt


# --------------------------------------------------------------------------- #
# Stage 2: runtime
#   Minimal image: just the Python runtime, the prebuilt virtualenv, and the
#   application source. No compiler, no pip cache, no apt package lists.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

# libgl1 + libglib2.0-0: OpenCV's Python wheel (`opencv-python-headless`,
# per requirements.txt) still dynamically links against a small set of
# system graphics/glib libraries even in headless mode. Everything else
# (Qt, GTK, X11) is intentionally NOT installed here — that's what makes
# `opencv-python-headless` the right choice over plain `opencv-python` for
# a container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user — standard container hardening practice; nothing
# in this app needs root privileges at runtime.
RUN useradd --create-home --uid 1000 qknee
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Application source. Anything matched by .dockerignore (datasets, notebook
# checkpoints, local pca_scaler.pkl / *.pt artifacts, __pycache__, the git
# history) is excluded from the build context entirely.
COPY --chown=qknee:qknee . .

# --- CPU-performance tuning for NISQ simulation + ResNet18 inference ---
# PennyLane's `default.qubit` simulator and PyTorch's CPU backend will both
# otherwise try to use every core they can see, which causes severe
# oversubscription/thrashing once two containers (api + frontend) share a
# host. Pinning explicit thread counts keeps latency predictable; override
# via `docker run -e TORCH_NUM_THREADS=... -e OMP_NUM_THREADS=...` to match
# the deployment host's core count.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4 \
    PCA_ARTIFACT_PATH=/app/models/pca_scaler.pkl \
    MODEL_CHECKPOINT_PATH=/app/models/qknee_model.pt

USER qknee

# Both services' ports are declared here since this one image can run
# either role; docker-compose.yml maps only the relevant port per service.
EXPOSE 8000 8501

# Default command runs the FastAPI backend; docker-compose.yml overrides
# this with the Streamlit `command:` for the frontend service.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
