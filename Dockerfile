# syntax=docker/dockerfile:1
#
# Q-Knee container image.
#
# One shared multi-stage image serves BOTH the FastAPI backend
# (qknee/api/server.py) and the Streamlit frontend (qknee/ui/dashboard.py) —
# they share the exact same heavy ML dependency stack (PyTorch, PennyLane,
# OpenCV, scikit-learn), so building two separate images would just
# duplicate multiple GB of identical layers for no benefit.
# docker-compose.yml runs two containers *from this one image*, each with a
# different `command:` override.
#
# Build:   docker build -t qknee:latest .
# Run API: docker run -p 8000:8000 qknee:latest
# Run UI:  docker run -p 8501:8501 qknee:latest streamlit run qknee/ui/dashboard.py --server.port=8501 --server.address=0.0.0.0


# --------------------------------------------------------------------------- #
# Stage 1: builder
#   Installs all Python dependencies into an isolated virtualenv. Kept
#   separate from the runtime stage so build-only tooling (a C compiler,
#   pip's wheel cache, apt package lists) never ends up in the final image.
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS builder

# build-essential: some scientific-Python deps (e.g. scipy, which PennyLane
# and scikit-learn depend on) fall back to compiling from source on
# platforms without a prebuilt manylinux wheel; keeping this in the builder
# stage only (not runtime) avoids bloating the final image with a toolchain.
#
# --mount=type=cache on /var/cache/apt and /var/lib/apt/lists (BuildKit-only,
# hence the `# syntax=` pragma at the top of this file) persists apt's
# downloaded .deb files and package lists across builds *without* baking
# them into any image layer — a rebuild on a warm BuildKit cache re-hits
# this package from local disk instead of the network, while the final
# layer stays exactly as clean as the old `rm -rf /var/lib/apt/lists/*`
# one-liner produced.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends build-essential

# Isolated virtualenv (rather than installing into the system site-packages)
# so the whole dependency tree can be copied as one clean directory into the
# runtime stage below.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .

# pip itself: split into its own layer so a `requirements.txt`/torch-version
# edit below never re-triggers a pip self-upgrade, and vice versa.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# PyTorch/torchvision, from the official CPU-only wheel index, in its own
# layer. This is the single most important line for image size and
# NISQ-simulator CPU deployments: the default PyPI wheel can pull in
# multi-GB NVIDIA CUDA libraries that are dead weight on a CPU-only host
# (PennyLane's `default.qubit` simulator and this project's ResNet18
# backbone both run on CPU here — there is no GPU workload to justify the
# CUDA runtime). Kept in its own `RUN` (not chained with the
# `requirements.txt` install below) specifically so an unrelated edit to
# `requirements.txt` doesn't invalidate this layer and force re-downloading
# these large wheels every time.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Remaining Python dependencies, last — this is the layer most likely to
# change from build to build (a new package, a version bump), so it's
# ordered after the much larger, much more stable torch/torchvision layer
# above rather than before it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt


# --------------------------------------------------------------------------- #
# Stage 2: runtime
#   Minimal image: just the Python runtime, the prebuilt virtualenv, and the
#   application source. No compiler, no pip cache, no apt package lists.
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS runtime

# libgl1 + libglib2.0-0: OpenCV's Python wheel (`opencv-python-headless`,
# per requirements.txt) still dynamically links against a small set of
# system graphics/glib libraries even in headless mode. Everything else
# (Qt, GTK, X11) is intentionally NOT installed here — that's what makes
# `opencv-python-headless` the right choice over plain `opencv-python` for
# a container.
#
# This RUN depends on nothing above it but the base image, so it's ordered
# first in this stage — it changes only if these two package names/versions
# change, which is far rarer than a `requirements.txt`/app-code edit, so it
# stays cached across nearly every rebuild.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0

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
    PCA_ARTIFACT_PATH=/app/qknee/artifacts/pca_scaler.pkl \
    MODEL_CHECKPOINT_PATH=/app/qknee/artifacts/qknee_model.pt

USER qknee

# Both services' ports are declared here since this one image can run
# either role; docker-compose.yml maps only the relevant port per service.
EXPOSE 8000 8501

# Default command runs the FastAPI backend; docker-compose.yml overrides
# this with the Streamlit `command:` for the frontend service.
CMD ["uvicorn", "qknee.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
