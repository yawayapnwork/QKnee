"""
Q-Knee REST API: FastAPI wrapper exposing `qknee.models.pipeline.PipelineRunner`
(DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM) to the Streamlit frontend
(qknee/ui) or any other HTTP client.

Run with:
    uvicorn qknee.api.server:app --reload --port 8000

Endpoints:
    GET  /health              - liveness/readiness probe, reports whether the
                                 real model backend loaded or the API is
                                 running in mock mode. Unauthenticated.
    POST /api/v1/auth/signup  - register a new user (hashed password,
                                 role-assigned). See qknee.api.auth.
    POST /api/v1/auth/login   - authenticate and receive a JWT bearer token.
    GET  /api/v1/auth/me      - the authenticated caller's own profile.
    POST /predict              - accepts a DICOM (.dcm/.dicom) or NumPy (.npy)
                                 MRI slice/volume upload, returns risk score,
                                 diagnosis, and a base64-encoded Grad-CAM
                                 heatmap overlay (PNG). Requires a bearer
                                 token for a `radiologist`/`triage_nurse`
                                 account (see qknee.api.auth.require_role).
    POST /explain              - same upload contract and auth requirement as
                                 /predict, but returns just the
                                 explainability payload (Grad-CAM heatmap +
                                 risk score for context) — for a caller that
                                 only needs the visual explanation, not the
                                 full diagnosis response.
    POST /report                - same upload contract and auth requirement as
                                 /predict, but returns a formal one-page
                                 radiology-style PDF report (reportlab)
                                 instead of a JSON payload.

Startup memory/latency: torch, torchvision, pennylane, matplotlib, and
reportlab are all deliberately kept out of this module's top-level
imports — see `get_backend()`'s docstring and the `TYPE_CHECKING` block
below. `uvicorn qknee.api.server:app` reaches "Application startup
complete" without importing any of them; the first `/predict`, `/explain`,
or `/report` request pays their one-time import + model-load cost instead,
which is what keeps a cold boot under Render's free-tier 512MB ceiling.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

# Set before ANY matplotlib import — including a transitive one triggered
# by `get_backend()`'s lazy torch/pennylane/gradcam import chain, or by
# `/report`'s lazy reportlab import — rather than only inside the code
# path that happens to import matplotlib today. A container's default
# config dir is often read-only or non-tmpfs storage, and matplotlib
# builds its font cache there on first import; on a cold boot that can
# stall for tens of seconds. This is a plain env-var write (no import,
# no filesystem access yet), so it costs nothing at module-import time
# and is in place well before `get_backend()`'s own `matplotlib.use("Agg")`
# call ever runs. `setdefault` so an operator-supplied override wins.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qknee.api.auth import INFERENCE_ROLES, UserResponse, require_role
from qknee.api.auth import router as auth_router
from qknee.api.auth import user_store
from qknee.config.loader import load_config, redact_connection_string
from qknee.config.logging_config import get_logger, setup_logging

# --------------------------------------------------------------------------- #
# Deliberately NOT imported at module scope: torch, torchvision, pennylane
# (all pulled in transitively by `qknee.models.pipeline`/`qknee.data.
# ingestion`/`qknee.xai.gradcam`), matplotlib, and reportlab. Together these
# account for the overwhelming majority of this service's *cold* RSS —
# importing any of them is what turns a <150MB, <1s "the process is up and
# routing requests" boot into a multi-hundred-MB, multi-second one, which
# blows Render's free-tier 512MB ceiling well before any real request
# arrives. Every one of them is deferred to inside the function/method that
# actually needs it (`get_backend()` below for the ML stack; the `/report`
# handler for reportlab), so `uvicorn qknee.api.server:app` reaches
# "Application startup complete" having imported none of them — see
# `get_backend()`'s docstring for the lazy-singleton that triggers the
# import on first `/predict`/`/explain` request instead.
#
# `cv2`/`numpy` are ALSO not module-scope imports (unlike before) for the
# same reason, one level down: Vercel's serverless deployment installs
# only `requirements-vercel.txt`/`qknee/api/requirements.txt` (see those
# files), which excludes opencv/numpy entirely — a bare `import cv2` at
# module scope would make `qknee.api.server` fail to import at all on that
# target. `_HEAVY_IMAGING_AVAILABLE` below records whether they resolved,
# so `get_backend()` can pick a pixel-processing-capable backend
# (`QKneeBackend`) when they did, or a router/proxy/cache backend when
# they didn't — see `ProxyBackend`/`CachedFallbackBackend`.
#
# `TYPE_CHECKING`-only imports below give this module real type hints for
# the deferred names without importing anything at runtime — a static
# type checker sees them, `python -c "import qknee.api.server"` never
# executes this branch.
# --------------------------------------------------------------------------- #
if TYPE_CHECKING:
    import numpy as np

    from qknee.data.ingestion import DataIngestion, IngestionError
    from qknee.models.pipeline import PipelineRunner

try:
    import cv2
    import numpy as np

    _HEAVY_IMAGING_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _HEAVY_IMAGING_AVAILABLE = False

setup_logging()
logger = get_logger("qknee.api")

_config = load_config()
PCA_ARTIFACT_PATH = _config.paths.pca_artifact
TEAR_RISK_THRESHOLD = _config.api.tear_risk_threshold

# --------------------------------------------------------------------------- #
# Serverless routing config — read once at import time (plain env vars,
# no heavy dependency involved), used by `get_backend()` to decide which
# backend implementation to build:
#
#   USE_MOCK_FALLBACK=true  - force the precomputed-cache backend
#                              (`CachedFallbackBackend`) even if the full
#                              ML stack happens to be importable. Useful
#                              for a low-resource preview deployment that
#                              has the packages installed but doesn't want
#                              to pay ResNet18/PennyLane's cold-load cost.
#   BACKEND_API_URL          - base URL of a full Render/Docker deployment
#                              of this same API (e.g.
#                              "https://qknee-api.onrender.com"). When set,
#                              `/predict`/`/explain`/`/report` proxy the
#                              upload there instead of running inference
#                              locally — the intended Vercel configuration,
#                              since Vercel never carries the ML stack.
#
# Precedence in `get_backend()`: BACKEND_API_URL (proxy) > USE_MOCK_FALLBACK
# or missing heavy imaging deps (precomputed-cache) > full local
# QKneeBackend. A serverless deployment should set BACKEND_API_URL; a
# demo/preview deployment with neither set still degrades gracefully to
# the cache instead of crashing on a missing torch/pennylane import.
# --------------------------------------------------------------------------- #
USE_MOCK_FALLBACK = os.getenv("USE_MOCK_FALLBACK", "false").strip().lower() in ("1", "true", "yes")
BACKEND_API_URL = (os.getenv("BACKEND_API_URL") or "").strip().rstrip("/") or None

# The Streamlit Community Cloud deployment of the clinical workstation
# (`qknee/ui/dashboard.py`) this API serves — surfaced from `GET /` so a
# visitor hitting the bare API root gets sent to the actual UI instead of
# a bare JSON blob with nowhere to go. Overridable for local/staging
# frontends via `$STREAMLIT_FRONTEND_URL`.
STREAMLIT_FRONTEND_URL = (os.getenv("STREAMLIT_FRONTEND_URL") or "https://q-knee.streamlit.app").strip()

_REPO_ROOT = Path(__file__).resolve().parents[2]
PRECOMPUTED_CACHE_PATH = _REPO_ROOT / "qknee" / "artifacts" / "precomputed_cache.json"


# --------------------------------------------------------------------------- #
# CacheService — unified Redis-or-in-memory cache facade
# --------------------------------------------------------------------------- #

class CacheService:
    """Unified async cache facade in front of two backends:

        - Redis (`redis.asyncio`), when `$REDIS_URL` is set — connection is
          attempted lazily (on first cache access, not at import time) with
          exponential backoff across `max_retries` attempts; a failure at
          any point (missing `redis` package, connection refused, timeout)
          logs a warning and permanently degrades this instance to the
          in-memory backend for the rest of the process's lifetime, rather
          than retrying on every subsequent call or raising into the
          request path.
        - An in-process `dict` with per-key TTL, when `$REDIS_URL` is unset
          (the default, correct configuration for a single-node free-tier
          deployment — Render, Streamlit Cloud, Vercel) or once Redis has
          been given up on as above.

    Values are pickled before being written to Redis (so a numpy Grad-CAM
    array round-trips exactly) — this cache only ever stores payloads this
    process itself computed and wrote, never externally-supplied data, so
    `pickle`'s arbitrary-code-execution-on-load risk does not apply to the
    values it reads back.
    """

    _MAX_MEMORY_ENTRIES = 256

    def __init__(self, redis_url: str = "", default_ttl_seconds: int = 3600, namespace: str = "qknee", max_retries: int = 3) -> None:
        self._redis_url = redis_url
        self._default_ttl = default_ttl_seconds
        self._namespace = namespace
        self._max_retries = max_retries

        self._redis_client: Optional[Any] = None
        self._redis_unavailable = not bool(redis_url)
        self._connect_lock = asyncio.Lock()

        self._memory: Dict[str, Tuple[Optional[float], Any]] = {}
        self._memory_lock = threading.Lock()

        if self._redis_unavailable:
            logger.info("CacheService: REDIS_URL not set — using in-process TTL cache.")

    @property
    def backend_name(self) -> str:
        """`"redis"` once a live connection has been established, else
        `"in-memory"` — read by `/health` so an operator can confirm which
        backend is actually active (e.g. after a Redis outage silently
        degraded a running process)."""
        return "redis" if self._redis_client is not None else "in-memory"

    async def _ensure_redis(self) -> Optional[Any]:
        if self._redis_unavailable:
            return None
        if self._redis_client is not None:
            return self._redis_client

        async with self._connect_lock:
            # Re-check inside the lock: another coroutine may have already
            # connected (or given up) while this one was waiting.
            if self._redis_client is not None or self._redis_unavailable:
                return self._redis_client

            try:
                import redis.asyncio as aioredis
            except ImportError:
                logger.warning(
                    "REDIS_URL is set but the 'redis' package is not installed; "
                    "degrading to in-process TTL caching."
                )
                self._redis_unavailable = True
                return None

            redacted = redact_connection_string(self._redis_url)
            backoff = 0.25
            for attempt in range(1, self._max_retries + 1):
                try:
                    client = aioredis.from_url(
                        self._redis_url, socket_connect_timeout=2.0, socket_timeout=2.0,
                    )
                    await client.ping()
                    self._redis_client = client
                    logger.info("CacheService connected to Redis at %s (attempt %d/%d).", redacted, attempt, self._max_retries)
                    return client
                except Exception as exc:  # noqa: BLE001 - any connectivity failure retries/degrades, never raises
                    logger.warning(
                        "Redis connection attempt %d/%d to %s failed: %s",
                        attempt, self._max_retries, redacted, exc,
                    )
                    if attempt < self._max_retries:
                        await asyncio.sleep(backoff)
                        backoff *= 2

            logger.warning(
                "Could not connect to Redis at %s after %d attempts; degrading to in-process "
                "TTL caching for the rest of this process's lifetime.", redacted, self._max_retries,
            )
            self._redis_unavailable = True
            return None

    async def get(self, key: str) -> Optional[Any]:
        full_key = f"{self._namespace}:{key}"
        client = await self._ensure_redis()
        if client is not None:
            try:
                import pickle

                raw = await client.get(full_key)
                if raw is not None:
                    return pickle.loads(raw)
                return None
            except Exception as exc:  # noqa: BLE001 - a live-but-flaky Redis falls back per-call, not permanently
                logger.warning("Redis GET failed (%s); serving this lookup from the in-memory cache instead.", exc)
        return self._memory_get(full_key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = self._default_ttl if ttl is None else ttl
        full_key = f"{self._namespace}:{key}"
        client = await self._ensure_redis()
        if client is not None:
            try:
                import pickle

                await client.set(full_key, pickle.dumps(value), ex=ttl)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis SET failed (%s); writing this entry to the in-memory cache instead.", exc)
        self._memory_set(full_key, value, ttl)

    async def get_or_set(self, key: str, compute_fn, ttl: Optional[int] = None) -> Any:
        """Returns the cached value for `key` if present, else calls the
        zero-argument `compute_fn` (synchronous — the caller wraps any
        blocking pipeline/Grad-CAM work), caches its result, and returns
        it. The single call site every expensive lookup in this module
        (Grad-CAM inference, precomputed-cache reads) should go through."""
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = compute_fn()
        await self.set(key, value, ttl=ttl)
        return value

    # -- in-memory TTL dict: both the pure fallback and Redis's own
    # per-call degrade path share this implementation. --
    def _memory_get(self, full_key: str) -> Optional[Any]:
        with self._memory_lock:
            entry = self._memory.get(full_key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at is not None and time.monotonic() > expires_at:
                del self._memory[full_key]
                return None
            return value

    def _memory_set(self, full_key: str, value: Any, ttl: Optional[int]) -> None:
        expires_at = (time.monotonic() + ttl) if ttl else None
        with self._memory_lock:
            self._memory[full_key] = (expires_at, value)
            if len(self._memory) > self._MAX_MEMORY_ENTRIES:
                # Evict the oldest entry (insertion-ordered dict) — a
                # simple bound so a long-running single-node process never
                # grows this dict unboundedly, not a true LRU.
                oldest_key = next(iter(self._memory))
                del self._memory[oldest_key]


cache_service = CacheService(
    redis_url=_config.storage.redis_url,
    default_ttl_seconds=_config.storage.cache_ttl_seconds,
)


# --------------------------------------------------------------------------- #
# Response schema
# --------------------------------------------------------------------------- #

class PredictionResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Predicted tear risk probability, in [0, 1].")
    diagnosis: str = Field(..., description="'Tear Detected' if risk_score >= 0.5, else 'Normal'.")
    gradcam_heatmap: str = Field(..., description="Base64-encoded PNG of the Grad-CAM overlay on the input slice.")
    backend: str = Field(..., description="'live' if PipelineRunner ran, 'mock' if a fallback was used.")


class HealthResponse(BaseModel):
    status: str
    backend_ready: bool
    detail: Optional[str] = None
    mode: str = Field(
        ...,
        description="Which backend implementation /predict, /explain, and /report route through: "
                    "'not-loaded' (lazy-inits on first request), 'live-or-mock' (QKneeBackend — full "
                    "local ML stack, Render/Docker), 'proxy' (ProxyBackend — forwards to BACKEND_API_URL), "
                    "or 'cache-fallback' (CachedFallbackBackend — precomputed_cache.json, serverless).",
    )
    user_store_backend: str = Field(..., description="'SQLAlchemyUserRepository' or 'LocalFileUserRepository'.")
    cache_backend: str = Field(..., description="'redis' or 'in-memory' — see qknee.api.server.CacheService.")


class ExplanationResponse(BaseModel):
    gradcam_heatmap: str = Field(..., description="Base64-encoded PNG of the Grad-CAM overlay on the input slice.")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Predicted tear risk probability, in [0, 1] — provided for context alongside the heatmap.")
    backend: str = Field(..., description="'live' if PipelineRunner ran, 'mock' if a fallback was used.")


# --------------------------------------------------------------------------- #
# Backend loading (mirrors app.py's mock-fallback pattern, so the API stays
# usable for frontend development even without a fitted PCA artifact)
# --------------------------------------------------------------------------- #

class QKneeBackend:
    """Thin FastAPI-facing wrapper around `PipelineRunner` — the canonical
    DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM engine. Delegates all
    actual inference to one `PipelineRunner` instance built at construction
    time, and falls back to a deterministic mock only when that engine
    fails to initialize (e.g. no fitted PCA artifact yet), so the API stays
    usable for frontend development without a trained backend.

    Construction (not just this class's definition) is where the heavy
    torch/torchvision/pennylane import chain actually happens — see
    `get_backend()` below, which is what defers *building* one of these
    until the first `/predict`/`/explain` request instead of at module
    import time."""

    def __init__(self, pca_artifact_path: Path = PCA_ARTIFACT_PATH):
        self.backend_ready = False
        self.load_error: Optional[str] = None
        self.runner: Optional["PipelineRunner"] = None
        self._ingestion: Optional["DataIngestion"] = None

        # Imported lazily, right here, rather than at module scope —
        # `qknee.data.ingestion` transitively imports `qknee.data.dataset`,
        # which imports torch/torchvision/pandas at ITS OWN module level —
        # so even a "just parse the upload, don't run the model" `/predict`
        # call would otherwise force that whole chain in. `sys.modules`
        # caches the actual import after the first call, so every
        # subsequent `QKneeBackend(...)` construction (or any other local
        # import of the same names elsewhere in this class) is just a
        # dict lookup, not a re-import.
        #
        # Deliberately INSIDE this try block (not above it, as this class
        # briefly had it): `get_backend()` only ever constructs
        # `QKneeBackend` when the router has already decided the full ML
        # stack should be importable (see `_HEAVY_IMAGING_AVAILABLE` and
        # its selection logic), but environments drift — a partial
        # install, a missing native wheel for the host's platform, etc.
        # An `ImportError` here must degrade to MOCK mode exactly like a
        # missing PCA artifact does, not crash `get_backend()` and take
        # every `/predict`/`/explain`/`/report` request down with it.
        try:
            from qknee.data.ingestion import DataIngestion
            from qknee.models.pipeline import PipelineRunner

            self._ingestion = DataIngestion(train=False)
            self.runner = PipelineRunner(config=_config, pca_artifact_path=pca_artifact_path)
            self.backend_ready = True
            logger.info("QKneeBackend loaded live PipelineRunner from %s", pca_artifact_path)
        except ImportError as exc:
            self.load_error = f"Heavy ML dependency unavailable: {exc}"
            logger.warning("QKneeBackend starting in MOCK mode: %s", self.load_error)
        except Exception as exc:  # noqa: BLE001 - PipelineValidationError or any unexpected load failure
            self.load_error = str(exc)
            logger.warning("QKneeBackend starting in MOCK mode: %s", self.load_error)
        finally:
            # Loading (or failing to load and discarding) ResNet18 +
            # VQC/PCA weights is the heaviest allocation this process ever
            # does outside of serving an actual request — collect promptly
            # rather than leaving it for whenever the next GC cycle runs.
            gc.collect()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_dicom_slice(raw_bytes: bytes) -> np.ndarray:
        """Parses DICOM bytes into a calibrated grayscale pixel array.

        Beyond a bare `pixel_array` read, this:
            - Reads with `force=True` so anonymized/non-conformant exports
              missing the 128-byte preamble + 'DICM' magic (common from some
              clinical PACS/de-identification pipelines) still parse instead
              of raising `InvalidDicomError`.
            - Applies the Modality LUT (`RescaleSlope`/`RescaleIntercept`, or
              a full `ModalityLUTSequence` if present) via pydicom, so CT-style
              DICOMs return calibrated intensities (e.g. Hounsfield units)
              instead of raw stored pixel values. A no-op when neither is
              present in the dataset.
            - Inverts `MONOCHROME1` datasets (where 0 = white, max = black)
              so intensity semantics match the far more common `MONOCHROME2`
              convention (0 = black) that the rest of the pipeline assumes.
        """
        import io

        import pydicom
        from pydicom.pixels import apply_modality_lut

        try:
            dataset = pydicom.dcmread(io.BytesIO(raw_bytes), force=True)
            array = apply_modality_lut(dataset.pixel_array, dataset)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Failed to parse DICOM file: {exc}") from exc

        if getattr(dataset, "PhotometricInterpretation", None) == "MONOCHROME1":
            array = np.asarray(array)
            array = array.max() - array
            logger.debug("Inverted MONOCHROME1 DICOM slice to MONOCHROME2 intensity convention.")

        return array

    def load_slice(self, raw_bytes: bytes, filename: str) -> np.ndarray:
        """Parses uploaded bytes (.dcm/.dicom or .npy) into a single 2D
        grayscale slice array. Multi-slice/multi-frame volumes (3D `(D,H,W)`
        or 4D `(D,H,W,C)` color arrays) are reduced to their central slice
        via `DataIngestion`, the same ingestion path used everywhere else in
        the pipeline — so a color multi-frame DICOM or a 4D `.npy` volume is
        decomposed consistently instead of being rejected outright."""
        suffix = Path(filename).suffix.lower()

        if suffix in (".dcm", ".dicom"):
            array = self._load_dicom_slice(raw_bytes)

        elif suffix == ".npy":
            import io

            try:
                array = np.load(io.BytesIO(raw_bytes))
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Failed to parse .npy file: {exc}") from exc

        else:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix}'. Upload a .dcm/.dicom or .npy file.",
            )

        return self._to_central_2d_slice(np.asarray(array))

    def _to_central_2d_slice(self, array: np.ndarray) -> np.ndarray:
        """Reduces a 2D/3D/4D array to one representative 2D grayscale slice.

        2D arrays pass through unchanged. 3D `(D, H, W)` and 4D `(D, H, W, C)`
        arrays are decomposed via `DataIngestion._array_to_pil_slices` (the
        same slice-extraction + color-averaging logic used for `.npy` volume
        ingestion elsewhere in the pipeline) and reduced to the central slice
        — the anatomical midpoint of the stack, consistent with
        `PipelineRunner.run()`'s own central-slice Grad-CAM selection.
        """
        if array.ndim == 2:
            return array

        if self._ingestion is None:
            # `QKneeBackend.__init__` failed to import/construct
            # `DataIngestion` (see its ImportError handling) — this
            # instance is running in MOCK mode with no ingestion engine at
            # all, so a multi-slice upload can't be decomposed. A 2D
            # upload still works fine (the branch above never reaches
            # here); only a genuinely volumetric upload hits this.
            raise HTTPException(
                status_code=503,
                detail="Multi-slice volume decomposition is unavailable: the ML backend failed to load "
                       f"({self.load_error or 'unknown reason'}). Upload a single 2D slice instead.",
            )

        from qknee.data.ingestion import IngestionError

        try:
            slices = self._ingestion.load_slices_as_pil(array)
        except IngestionError as exc:
            raise HTTPException(status_code=422, detail=f"Failed to decompose array into slices: {exc}") from exc

        return np.array(slices[len(slices) // 2])

    @staticmethod
    def _normalize_uint8(slice_2d: np.ndarray) -> np.ndarray:
        slice_2d = slice_2d.astype(np.float32)
        min_val, max_val = float(slice_2d.min()), float(slice_2d.max())
        if max_val > min_val:
            slice_2d = (slice_2d - min_val) / (max_val - min_val)
        else:
            slice_2d = np.zeros_like(slice_2d)
        return (slice_2d * 255).astype(np.uint8)

    def predict(self, raw_bytes: bytes, filename: str, auth_header: Optional[str] = None) -> PredictionResponse:
        # `auth_header` is part of the common backend surface (see
        # `ProxyBackend.predict`, which actually needs it to forward the
        # caller's bearer token upstream) — unused here since this backend
        # runs inference in-process rather than over HTTP.
        del auth_header
        slice_2d = self.load_slice(raw_bytes, filename)
        display_slice = self._normalize_uint8(slice_2d)

        if self.backend_ready:
            risk_score, heatmap_b64 = self._predict_live(display_slice)
            backend = "live"
        else:
            risk_score, heatmap_b64 = self._predict_mock(display_slice)
            backend = "mock"

        diagnosis = "Tear Detected" if risk_score >= TEAR_RISK_THRESHOLD else "Normal"

        return PredictionResponse(
            risk_score=risk_score,
            diagnosis=diagnosis,
            gradcam_heatmap=heatmap_b64,
            backend=backend,
        )

    def _predict_live(self, display_slice: np.ndarray) -> Tuple[float, str]:
        from qknee.xai.gradcam import overlay_heatmap

        try:
            result = self.runner.run(display_slice)  # DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM
            overlay = overlay_heatmap(result.gradcam_heatmap, display_slice)
            heatmap_b64 = self._encode_png_base64(overlay)
            return result.risk_score, heatmap_b64
        except Exception as exc:  # noqa: BLE001 - PipelineValidationError or any unexpected failure
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        finally:
            # `result`'s tensors are already pulled out into `risk_score`/
            # `heatmap_b64` (plain float/str) by this point — collect the
            # ResNet18 forward + Grad-CAM backward pass's freed activation
            # graph now rather than leaving it for the next GC cycle.
            # `PipelineRunner.run()` already does this internally too; the
            # extra call here is cheap insurance around the encode step
            # above, which runs after that internal collection.
            gc.collect()

    def _predict_mock(self, display_slice: np.ndarray) -> Tuple[float, str]:
        import hashlib

        from qknee.xai.gradcam import overlay_heatmap

        digest = hashlib.sha256(display_slice.tobytes()).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)
        risk_score = float(rng.uniform(0.05, 0.95))

        # Fake "heatmap": a soft radial gradient, just so the response shape
        # (and the frontend's image-decoding path) is exercised in mock mode.
        h, w = display_slice.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2, w / 2
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        fake_heatmap = np.clip(1 - radius / radius.max(), 0, 1)
        overlay = overlay_heatmap(fake_heatmap, display_slice)
        return risk_score, self._encode_png_base64(overlay)

    @staticmethod
    def _encode_png_base64(bgr_image: np.ndarray) -> str:
        success, encoded = cv2.imencode(".png", bgr_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode Grad-CAM heatmap as PNG")
        return base64.b64encode(encoded.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- #
# Lightweight serverless backends — CachedFallbackBackend and ProxyBackend
# both implement the same `.predict(raw_bytes, filename) -> PredictionResponse`
# surface as `QKneeBackend`, so `get_backend()`'s callers (`/predict`,
# `/explain`) don't need to know which one is active. Neither imports
# numpy/cv2/torch/pennylane — both work with only `requirements-vercel.txt`
# installed, which is the entire point: on Vercel, this router/mock/proxy
# layer must run without the ~5GB CUDA/PyTorch/PennyLane dependency chain
# that blew through the 500MB serverless function size limit.
# --------------------------------------------------------------------------- #

class CachedFallbackBackend:
    """Serves `/predict`/`/explain` from `qknee/artifacts/precomputed_cache.json`
    — the same offline-generated "Judge Fast-Path" cache
    `qknee.ui.dashboard`/`qknee.ui.analysis_app` use — instead of running
    any real inference. Selected by `get_backend()` when `$USE_MOCK_FALLBACK`
    is set, or when the heavy ML stack (numpy/cv2/torch/pennylane) isn't
    importable at all and no `$BACKEND_API_URL` is configured to proxy to.

    Deterministic: the same uploaded bytes always select the same cached
    case (content-hashed, stdlib `hashlib` only), so repeat calls on one
    file are stable rather than randomly cycling cases. Each case's
    Grad-CAM overlay is already a base64-encoded PNG string in the cache
    file, so this never needs to decode/re-encode an image — no cv2 or
    numpy required anywhere in this class.
    """

    backend_ready = False  # never "live" — always serving canned data

    def __init__(self, cache_path: Path = PRECOMPUTED_CACHE_PATH) -> None:
        self._cache_path = cache_path
        self._cases: list = []
        self.load_error: Optional[str] = None

        if not cache_path.exists():
            self.load_error = f"No precomputed cache found at {cache_path}"
            logger.warning(
                "CachedFallbackBackend: %s — run `python scripts/generate_demo_cache.py` to build one; "
                "/predict and /explain will 503 until it exists.", self.load_error,
            )
            return

        try:
            import json

            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self._cases = payload.get("cases", [])
        except (OSError, ValueError) as exc:
            self.load_error = f"Failed to read precomputed cache at {cache_path}: {exc}"
            logger.warning("CachedFallbackBackend: %s", self.load_error)
            return

        if not self._cases:
            self.load_error = f"Precomputed cache at {cache_path} has no cases recorded"
            logger.warning("CachedFallbackBackend: %s", self.load_error)
        else:
            logger.info("CachedFallbackBackend serving %d precomputed case(s) from %s", len(self._cases), cache_path)

    def predict(self, raw_bytes: bytes, filename: str, auth_header: Optional[str] = None) -> "PredictionResponse":
        del auth_header  # part of the common backend surface; unused — see QKneeBackend.predict
        if not self._cases:
            raise HTTPException(
                status_code=503,
                detail=f"No precomputed cache available for fallback inference ({self.load_error}). "
                       "Configure BACKEND_API_URL to proxy to a full ML backend instead.",
            )

        # Deterministic case selection: same content -> same case, so a
        # repeat upload doesn't appear to "flip" its answer between calls.
        digest = hashlib.sha256(raw_bytes).digest()
        index = int.from_bytes(digest[:4], "big") % len(self._cases)
        case = self._cases[index]

        risk_score = float(case.get("risk_score", 0.5))
        diagnosis = "Tear Detected" if risk_score >= TEAR_RISK_THRESHOLD else "Normal"
        heatmap_b64 = case.get("heatmap_base64", "")

        return PredictionResponse(
            risk_score=risk_score,
            diagnosis=diagnosis,
            gradcam_heatmap=heatmap_b64,
            backend=f"cache-fallback/{case.get('case_id', 'unknown')}",
        )


class ProxyBackend:
    """Forwards `/predict`/`/explain`/`/report` uploads to a full Render/
    Docker deployment of this same API (`$BACKEND_API_URL`) over HTTP,
    instead of running any inference locally — the intended Vercel
    configuration: this process never imports numpy/cv2/torch/pennylane at
    all, it's a pure HTTP router.

    `httpx` (already in `requirements-vercel.txt`) does the forwarding —
    the Authorization header from the *original* inbound request is
    forwarded byte-for-byte (see `predict()`'s `auth_header` argument),
    relying on both deployments sharing the same `$JWT_SECRET` so a token
    issued by one is valid on the other.
    """

    backend_ready = True  # "ready" in the sense that requests will be served (proxied), not run locally
    load_error: Optional[str] = None

    def __init__(self, backend_api_url: str, timeout: float = 30.0) -> None:
        self.backend_api_url = backend_api_url
        self.timeout = timeout
        logger.info("ProxyBackend configured — forwarding inference requests to %s", backend_api_url)

    def _forward(self, path: str, raw_bytes: bytes, filename: str, auth_header: Optional[str]) -> "httpx.Response":  # type: ignore[name-defined]
        import httpx

        headers = {"Authorization": auth_header} if auth_header else {}
        try:
            response = httpx.post(
                f"{self.backend_api_url}{path}",
                files={"file": (filename, raw_bytes, "application/octet-stream")},
                headers=headers,
                timeout=self.timeout,
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502, detail=f"Failed to reach backend API at {self.backend_api_url}: {exc}",
            ) from exc

        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response

    def predict(self, raw_bytes: bytes, filename: str, auth_header: Optional[str] = None) -> "PredictionResponse":
        response = self._forward("/predict", raw_bytes, filename, auth_header)
        return PredictionResponse(**response.json())

    def explain(self, raw_bytes: bytes, filename: str, auth_header: Optional[str] = None) -> "ExplanationResponse":
        response = self._forward("/explain", raw_bytes, filename, auth_header)
        return ExplanationResponse(**response.json())

    def report(self, raw_bytes: bytes, filename: str, auth_header: Optional[str] = None) -> bytes:
        """Returns the raw PDF bytes from the upstream `/report` response —
        this process never has reportlab/PIL/numpy to build one itself."""
        response = self._forward("/report", raw_bytes, filename, auth_header)
        return response.content


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Q-Knee Inference API",
    description="Quantum-assisted ACL/meniscal tear risk scoring from MRI slices.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    # Starlette's default (True) — set explicitly so a Vercel rewrite that
    # adds/drops a trailing slash (e.g. `/health` vs `/health/`) still
    # resolves to the same route instead of a bare 404.
    redirect_slashes=True,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

# Lazy singleton: `backend` starts `None` — no heavy backend is built, and
# therefore no torch/torchvision/pennylane import happens, at module
# import time. `get_backend()` below builds (and caches) the one
# process-wide instance on the FIRST call, which only ever happens from
# inside `/predict`/`/explain`/`/report` — never from `/health`, and never
# from simply importing this module — so `uvicorn qknee.api.server:app`
# reaches "Application startup complete" having paid none of that cost.
BackendType = Any  # QKneeBackend | ProxyBackend | CachedFallbackBackend — see get_backend()
backend: Optional[BackendType] = None
_backend_lock = threading.Lock()


def get_backend() -> BackendType:
    """Returns the process-wide backend, building it on the first call
    (thread-safe: `/predict`/`/explain`/`/report` are `async def`s but the
    sync construction below still runs behind this lock, so two requests
    racing to be "first" never build two backends). Every call after the
    first is just a `None` check.

    Picks one of three implementations, in this precedence order:

        1. `ProxyBackend`     - `$BACKEND_API_URL` is set. Forwards every
                                 request to a full Render/Docker deployment
                                 over HTTP; never imports numpy/cv2/torch/
                                 pennylane. The intended Vercel config.
        2. `CachedFallbackBackend` - `$USE_MOCK_FALLBACK` is set, OR the
                                 heavy imaging stack (numpy/cv2) failed to
                                 import at all (i.e. only
                                 `requirements-vercel.txt` is installed).
                                 Serves deterministic canned responses from
                                 `precomputed_cache.json` — also
                                 numpy/cv2/torch/pennylane-free.
        3. `QKneeBackend`      - everything else (Render/Docker, the full
                                 `requirements.txt` installed): runs real
                                 inference, with its own internal live/mock
                                 fallback if the PCA artifact isn't fitted.

    `matplotlib.use("Agg")` is selected *before* triggering `QKneeBackend`'s
    torch/pennylane import chain, in case anything on it — now, or added
    later — imports matplotlib: the non-interactive Agg backend must be
    chosen before matplotlib's own first import, or it probes for a GUI
    toolkit that doesn't exist on a headless container, which can hang
    and/or trigger an expensive font-cache rebuild. Skipped silently if
    matplotlib isn't installed at all (never installed on Vercel).
    """
    global backend
    if backend is not None:
        return backend

    with _backend_lock:
        if backend is None:
            if BACKEND_API_URL:
                backend = ProxyBackend(BACKEND_API_URL)
            elif USE_MOCK_FALLBACK or not _HEAVY_IMAGING_AVAILABLE:
                if USE_MOCK_FALLBACK:
                    logger.info("USE_MOCK_FALLBACK=true — serving from the precomputed cache.")
                else:
                    logger.warning(
                        "numpy/cv2 not importable (lightweight/serverless install) and no "
                        "BACKEND_API_URL configured — serving from the precomputed cache."
                    )
                backend = CachedFallbackBackend()
            else:
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                except ImportError:
                    pass
                backend = QKneeBackend()
    return backend


def _planned_backend_mode() -> str:
    """Which backend `get_backend()` *would* build, computed from plain
    env vars/import-availability flags only — never actually triggers the
    lazy construction. Used by `/health` so a readiness probe can report
    the intended routing mode even before the first `/predict` request."""
    if BACKEND_API_URL:
        return "proxy"
    if USE_MOCK_FALLBACK or not _HEAVY_IMAGING_AVAILABLE:
        return "cache-fallback"
    return "live-or-mock"


@app.get("/", tags=["System"])
async def root(request: Request):
    """Landing payload for the bare API root — without this, `GET /` 404s
    (`{"detail":"Not Found"}`) since FastAPI/Starlette only match routes
    that are explicitly declared, and no `@app.get("/")` existed. Returns
    a structured JSON gateway summary to API clients/`curl`, or a minimal
    HTML welcome page (with a live link to the Streamlit workstation and
    `/docs`) to a browser — negotiated off the request's `Accept` header,
    so this single route serves both without a redirect."""
    payload = {
        "system": "Q-Knee Diagnostic API Gateway",
        "status": "operational",
        "version": app.version,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
        "health": "/health",
        "frontend_workstation": STREAMLIT_FRONTEND_URL,
        "client_endpoints": {
            "predict": "/predict",
            "explain": "/explain",
            "report": "/report",
        },
    }

    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        from fastapi.responses import HTMLResponse

        html = f"""
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Q-Knee Diagnostic API Gateway</title>
            <style>
                body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background: #F8FAFC;
                        color: #0F172A; max-width: 640px; margin: 4rem auto; padding: 0 1.5rem; }}
                h1 {{ font-size: 1.4rem; }}
                a {{ color: #0284C7; }}
                code {{ background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px;
                        padding: 0.1rem 0.35rem; font-family: 'SF Mono', Consolas, monospace; }}
                ul {{ line-height: 1.9; }}
            </style>
        </head>
        <body>
            <h1>Q-Knee Diagnostic API Gateway</h1>
            <p>Status: <strong>operational</strong> &middot; Version: <code>{payload["version"]}</code></p>
            <p>This is the inference API. For the interactive clinical workstation, go to
               <a href="{STREAMLIT_FRONTEND_URL}">{STREAMLIT_FRONTEND_URL}</a>.</p>
            <ul>
                <li><a href="/docs">Swagger UI (/docs)</a></li>
                <li><a href="/redoc">ReDoc (/redoc)</a></li>
                <li><a href="/health">Health check (/health)</a></li>
            </ul>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    return payload


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health() -> HealthResponse:
    """Reports whether the live QKneeModel backend loaded successfully,
    which of the three backend implementations is (or will be) active,
    plus which storage backends are actually active for the user store and
    the cache — useful to confirm a configured `$DATABASE_URL`/`$REDIS_URL`
    actually connected, rather than silently degrading to the local
    fallback.

    Deliberately reads the module-level `backend` singleton directly
    rather than calling `get_backend()`: this is Render's (and any other
    host's) readiness/liveness probe, hit immediately and repeatedly after
    boot, so it must never itself be what triggers the one-time
    torch/torchvision/pennylane import + ResNet18/VQC load that the first
    real `/predict`/`/explain`/`/report` request pays for."""
    if backend is None:
        return HealthResponse(
            status="ok",
            backend_ready=False,
            detail="Model backend not yet loaded (lazy-initializes on the first /predict, /explain, or /report request).",
            mode=_planned_backend_mode(),
            user_store_backend=type(user_store).__name__,
            cache_backend=cache_service.backend_name,
        )
    return HealthResponse(
        status="ok",
        backend_ready=backend.backend_ready,
        detail=backend.load_error,
        mode=_planned_backend_mode(),
        user_store_backend=type(user_store).__name__,
        cache_backend=cache_service.backend_name,
    )


def _predict_cache_key(raw_bytes: bytes) -> str:
    """Content-addressed cache key for one uploaded slice/volume's
    prediction — identical bytes always hash to the same key, so a
    repeat upload of the same file (whether via `/predict`, `/explain`,
    or `/report`, all three share this key) is served from `CacheService`
    instead of re-running the ResNet18 -> PCA -> VQC -> Grad-CAM
    pipeline."""
    return f"predict:{hashlib.sha256(raw_bytes).hexdigest()}"


def _decode_png_base64(png_base64: str) -> "np.ndarray":
    """Inverse of `QKneeBackend._encode_png_base64` — decodes a cached/
    fresh `PredictionResponse.gradcam_heatmap` back into a `(H, W, 3)` BGR
    array, for `/report` to hand to `generate_radiology_report` (which
    wants a raw overlay array, not a base64 PNG string).

    Only ever called from the `QKneeBackend` branch of `/report` (see
    below), which itself only runs when `_HEAVY_IMAGING_AVAILABLE` is
    True — the guard here is defense-in-depth, not the primary gate."""
    if not _HEAVY_IMAGING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PDF report generation requires numpy/opencv, which are not installed in this "
                   "deployment. Configure BACKEND_API_URL to proxy /report to a full backend instead.",
        )
    png_bytes = base64.b64decode(png_base64)
    return cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    request: Request,
    file: UploadFile = File(..., description="DICOM (.dcm/.dicom) or NumPy (.npy) MRI slice/volume"),
    current_user: UserResponse = Depends(require_role(INFERENCE_ROLES)),
) -> PredictionResponse:
    """Runs one MRI slice (or the middle slice of a volume) through the
    Q-Knee pipeline and returns the tear-risk score, diagnosis label, and a
    base64-encoded Grad-CAM overlay for visual explainability. Requires a
    bearer token for a `radiologist`/`triage_nurse` account — 401 with no
    token, 403 for a `guest_demo` token.

    The expensive part of this call (ResNet18 forward pass + Grad-CAM
    backward pass) is routed through `CacheService`, keyed on the
    uploaded file's content hash — a repeat upload of the same slice
    (including one that arrives via `/explain` instead) is served from
    cache rather than re-run.

    On a lightweight/serverless deployment (`get_backend()` returns a
    `ProxyBackend`), this call is forwarded to `$BACKEND_API_URL` instead
    of running locally — the inbound `Authorization` header is passed
    through unchanged so the upstream deployment's own auth gate sees the
    same bearer token."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    logger.info("POST /predict by user=%r role=%r file=%r", current_user.username, current_user.role, file.filename)
    cache_key = _predict_cache_key(raw_bytes)
    cached = await cache_service.get(cache_key)
    if cached is not None:
        logger.debug("POST /predict cache hit (key=%s)", cache_key)
        return PredictionResponse(**cached)

    auth_header = request.headers.get("authorization")
    result = get_backend().predict(raw_bytes, file.filename or "upload", auth_header=auth_header)
    await cache_service.set(cache_key, result.model_dump(), ttl=_config.storage.cache_ttl_seconds)
    return result


@app.post("/explain", response_model=ExplanationResponse, tags=["Inference"])
async def explain(
    request: Request,
    file: UploadFile = File(..., description="DICOM (.dcm/.dicom) or NumPy (.npy) MRI slice/volume"),
    current_user: UserResponse = Depends(require_role(INFERENCE_ROLES)),
) -> ExplanationResponse:
    """Runs one MRI slice (or the middle slice of a volume) through the
    Q-Knee pipeline and returns just its Grad-CAM explainability heatmap
    (plus the risk score for context) — the explanation-focused
    counterpart to /predict. Shares /predict's exact upload contract, auth
    requirement, file parsing, error handling, `CacheService` entry (same
    content-hash key — an /predict then /explain round-trip on the same
    file only ever runs the pipeline once), and proxy-forwarding behavior
    on a lightweight/serverless deployment, so a client can point either
    endpoint at the same file with the same bearer token."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    logger.info("POST /explain by user=%r role=%r file=%r", current_user.username, current_user.role, file.filename)
    cache_key = _predict_cache_key(raw_bytes)
    cached = await cache_service.get(cache_key)
    if cached is not None:
        logger.debug("POST /explain cache hit (key=%s)", cache_key)
        prediction = PredictionResponse(**cached)
    else:
        auth_header = request.headers.get("authorization")
        prediction = get_backend().predict(raw_bytes, file.filename or "upload", auth_header=auth_header)
        await cache_service.set(cache_key, prediction.model_dump(), ttl=_config.storage.cache_ttl_seconds)

    return ExplanationResponse(
        gradcam_heatmap=prediction.gradcam_heatmap,
        risk_score=prediction.risk_score,
        backend=prediction.backend,
    )


@app.post("/report", tags=["Inference"])
async def report(
    request: Request,
    file: UploadFile = File(..., description="DICOM (.dcm/.dicom) or NumPy (.npy) MRI slice/volume"),
    current_user: UserResponse = Depends(require_role(INFERENCE_ROLES)),
):
    """Runs (or reuses a cached) prediction for one MRI slice/volume and
    returns a formal one-page radiology-style PDF report — the
    reportlab-backed counterpart to /predict's JSON payload. Shares
    /predict's exact upload contract, auth requirement, and `CacheService`
    entry (same content-hash key — a /predict or /explain call on the same
    file first means this endpoint never re-runs the pipeline either).

    `reportlab` (via `qknee.xai.report_generator`) is imported lazily,
    right here, so a server that never receives a /report request never
    pays its import cost — matching /predict's and /explain's own
    lazy-loaded torch/torchvision/pennylane.

    Building a PDF needs numpy/PIL/reportlab regardless of backend, so
    this endpoint has no local fallback on a lightweight/serverless
    deployment: a `ProxyBackend` forwards the whole request (and its raw
    PDF response) to `$BACKEND_API_URL` unchanged; a `CachedFallbackBackend`
    (no proxy configured, heavy imaging unavailable) 503s with a clear
    explanation instead of crashing on a missing import."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    logger.info("POST /report by user=%r role=%r file=%r", current_user.username, current_user.role, file.filename)

    active_backend = get_backend()

    if isinstance(active_backend, ProxyBackend):
        from fastapi.responses import Response

        auth_header = request.headers.get("authorization")
        pdf_bytes = active_backend.report(raw_bytes, file.filename or "upload", auth_header=auth_header)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="qknee_diagnostic_report.pdf"'},
        )

    if not isinstance(active_backend, QKneeBackend):
        # CachedFallbackBackend: precomputed_cache.json has no raw image
        # array to build a PDF from, and this process may not even have
        # numpy/PIL/reportlab installed (serverless/lightweight install) —
        # there's no local fallback for report generation, only proxying
        # (handled above) or a clear error.
        raise HTTPException(
            status_code=503,
            detail="PDF report generation requires a full ML backend (numpy/opencv/reportlab), "
                   "which is not active in this deployment (serving from the precomputed cache). "
                   "Configure BACKEND_API_URL to proxy /report to a full backend instead.",
        )

    from fastapi.responses import Response

    from qknee.xai.report_generator import generate_radiology_report

    display_slice = active_backend._normalize_uint8(active_backend.load_slice(raw_bytes, file.filename or "upload"))

    cache_key = _predict_cache_key(raw_bytes)
    cached = await cache_service.get(cache_key)
    if cached is not None:
        logger.debug("POST /report cache hit (key=%s)", cache_key)
        prediction = PredictionResponse(**cached)
    else:
        prediction = active_backend.predict(raw_bytes, file.filename or "upload")
        await cache_service.set(cache_key, prediction.model_dump(), ttl=_config.storage.cache_ttl_seconds)

    pdf_bytes = generate_radiology_report(
        output_path=None,
        mri_slice=display_slice,
        gradcam_overlay=_decode_png_base64(prediction.gradcam_heatmap),
        prediction_results={
            # This API scores one unified risk (no separate ACL/MCL/
            # meniscus heads — see PredictionResponse); report_generator
            # renders the fields it isn't given ("mcl_risk", "meniscus_risk",
            # "pauli_z_expectations", per-stage latencies) as "N/A" rather
            # than fabricating them.
            "acl_risk": prediction.risk_score,
            "backend": prediction.backend,
        },
        metadata={
            "modality": "MRI Knee",
            "clinical_indication": f"Q-Knee API /report — {prediction.diagnosis}",
        },
    )
    # The PIL images reportlab's Canvas wraps around `display_slice`/
    # `gradcam_overlay`, and the in-memory PNG/PDF byte buffers
    # `generate_radiology_report` builds internally, are all done being
    # used past this point — `pdf_bytes` has already been captured above.
    gc.collect()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="qknee_diagnostic_report.pdf"'},
    )


if __name__ == "__main__":
    import uvicorn

    # Render (and most other PaaS free tiers) assign the listen port at
    # runtime via `$PORT` and route traffic only to that port — a
    # hardcoded/config-only port silently fails to bind where the
    # platform actually expects it. `$PORT`, when set, always wins;
    # `config.yaml`'s `api.port` (8000 by default) is the local-dev
    # fallback when `$PORT` is unset.
    port = int(os.getenv("PORT", _config.api.port))
    uvicorn.run("qknee.api.server:app", host=_config.api.host, port=port, reload=True)
