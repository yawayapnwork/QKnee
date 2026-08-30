"""
Central Q-Knee orchestration: DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM.

`PipelineRunner` is the single entry point downstream consumers (the FastAPI
server, the Streamlit dashboards, batch scoring scripts) should use instead
of wiring the five stages together by hand. It:

    1. Loads all hyperparameters from `qknee/config/config.yaml`
       (via `qknee.config.loader.load_config`).
    2. Validates the tensor shape/dtype/range handed between every pair of
       adjacent stages, raising `PipelineValidationError` with the offending
       stage name on mismatch, instead of letting a shape error surface deep
       inside PyTorch/PennyLane/sklearn.
    3. Logs stage-by-stage progress and timing through the shared logger
       (`qknee.config.logging_config`) rather than printing.

Usage:
    from qknee.models.pipeline import PipelineRunner

    runner = PipelineRunner()                    # loads pca_scaler.pkl per config.yaml
    result = runner.run("slice.png")              # PipelineResult(risk_score=..., heatmap=...)

    # Or drive individual stages (e.g. to reuse one quantum-feature vector
    # across multiple trained VQC heads, as the clinical dashboard does):
    quantum_angles = runner.extract_quantum_features("slice.png")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pennylane as qml
import torch
from PIL import Image

from qknee.config.loader import QKneeConfig, load_config
from qknee.config.logging_config import get_logger
from qknee.data.ingestion import DataIngestion, IngestionError
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.qknee_model import PCAProjectionLayer
from qknee.models.quantum_autoencoder import QuantumAutoencoder
from qknee.models.resnet_extractor import ONNXFeatureExtractor, ResNet18FeatureExtractor
from qknee.models.vqc import VQCClassifier, angle_encoding, variational_block
from qknee.models.vqc_data_reuploading import DataReuploadingVQC
from qknee.xai.gradcam import GradCAM, TargetFn, get_default_target_layer, overlay_heatmap

logger = get_logger(__name__)

InputType = Union[str, Path, np.ndarray, Image.Image]

# Dynamic backbone-selection flags (Stage 3 / Stage 4) — see `PipelineRunner`.
EncoderType = Literal["pca", "quantum_autoencoder"]
ClassifierBackbone = Literal["vqc", "data_reuploading"]

# --------------------------------------------------------------------------- #
# Decoupled ONNX export artifacts (see `scripts/export_onnx.py` and
# `HybridONNXInferenceEngine` below). Named here (not just in the export
# script) so the export side and the load side share one source of truth
# for where these three files live.
# --------------------------------------------------------------------------- #
DEFAULT_RESNET_ONNX_PATH = Path("qknee/artifacts/resnet_feature_extractor.onnx")
DEFAULT_VQC_WEIGHTS_PATH = Path("qknee/artifacts/qknee_vqc_weights.pt")
DEFAULT_CIRCUIT_PARAMS_PATH = Path("qknee/artifacts/circuit_params.json")

QuantumBackend = Literal["pennylane", "qiskit"]


class PipelineValidationError(RuntimeError):
    """Raised when the tensor handed between two pipeline stages fails
    shape/dtype/range validation, naming the two stages involved."""


@dataclass(frozen=True)
class PipelineResult:
    """Output of a full `PipelineRunner.run()` call."""

    risk_score: float                 # in [0, 1]
    quantum_angles: np.ndarray        # (1, n_qubits), in [0, 2*pi]
    gradcam_heatmap: Optional[np.ndarray]  # (H, W) in [0, 1], or None if skip_gradcam=True


_VQC_PREFIX = "vqc."


def load_vqc_weights(
    vqc: VQCClassifier,
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> None:
    """Loads `vqc`'s weights in place from a checkpoint produced by
    `qknee.models.qknee_model.save_checkpoint`.

    Standalone counterpart of `PipelineRunner`'s internal checkpoint loading,
    reusable by any caller managing its own `VQCClassifier` instance
    directly — e.g. the clinical dashboard's separate ACL/meniscus heads,
    which aren't wrapped in a `PipelineRunner`.

    Accepts two on-disk shapes for the checkpoint dict:
        - `vqc_state_dict`   (preferred): the VQC's own state dict,
          unprefixed (e.g. `"readout.weight"`) — written directly by
          current `save_checkpoint()`.
        - `model_state_dict` (fallback): the full joint `QKneeModel`
          state dict, namespaced (e.g. `"vqc.readout.weight"`) — the
          `vqc.` prefix is stripped so it loads cleanly into a
          standalone `VQCClassifier`.

    Raises `PipelineValidationError` naming the missing/mismatched keys
    instead of letting a bad checkpoint surface as a raw
    `RuntimeError`/`KeyError` from `torch.load`/`load_state_dict`.
    """
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise PipelineValidationError(f"[VQC checkpoint] failed to read {checkpoint_path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise PipelineValidationError(
            f"[VQC checkpoint] expected a dict at {checkpoint_path}, got {type(checkpoint)}"
        )

    vqc_state_dict = checkpoint.get("vqc_state_dict")
    if vqc_state_dict is None:
        model_state_dict = checkpoint.get("model_state_dict")
        if model_state_dict is None:
            raise PipelineValidationError(
                f"[VQC checkpoint] {checkpoint_path} has neither 'vqc_state_dict' nor "
                "'model_state_dict' - this checkpoint was not produced by "
                "qknee.models.qknee_model.save_checkpoint() and cannot be loaded."
            )
        vqc_state_dict = {
            key[len(_VQC_PREFIX):]: value
            for key, value in model_state_dict.items()
            if key.startswith(_VQC_PREFIX)
        }
        if not vqc_state_dict:
            raise PipelineValidationError(
                f"[VQC checkpoint] {checkpoint_path}'s 'model_state_dict' has no keys "
                f"prefixed '{_VQC_PREFIX}' - cannot recover VQC weights from it."
            )
        logger.debug(
            "[VQC checkpoint] no 'vqc_state_dict' in %s; recovered %d VQC tensors "
            "from 'model_state_dict' by stripping the '%s' prefix.",
            checkpoint_path, len(vqc_state_dict), _VQC_PREFIX,
        )

    checkpoint_n_qubits = checkpoint.get("n_qubits")
    if checkpoint_n_qubits is not None and checkpoint_n_qubits != vqc.n_qubits:
        raise PipelineValidationError(
            f"[VQC checkpoint] {checkpoint_path} was saved with n_qubits={checkpoint_n_qubits}, "
            f"but the target VQCClassifier expects n_qubits={vqc.n_qubits}"
        )
    checkpoint_n_layers = checkpoint.get("n_layers")
    if checkpoint_n_layers is not None and checkpoint_n_layers != vqc.n_layers:
        raise PipelineValidationError(
            f"[VQC checkpoint] {checkpoint_path} was saved with n_layers={checkpoint_n_layers}, "
            f"but the target VQCClassifier expects n_layers={vqc.n_layers}"
        )

    try:
        vqc.load_state_dict(vqc_state_dict, strict=True)
    except RuntimeError as exc:
        raise PipelineValidationError(
            f"[VQC checkpoint] state dict from {checkpoint_path} does not match "
            f"VQCClassifier(n_qubits={vqc.n_qubits}, n_layers={vqc.n_layers}): {exc}"
        ) from exc


class PipelineRunner:
    """Chains DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM into one
    validated, config-driven inference pipeline.

    Args:
        config: A pre-loaded `QKneeConfig`; defaults to `load_config()`.
        pca_artifact_path: Overrides `config.paths.pca_artifact`. Only used
            when `encoder_type="pca"`.
        vqc_checkpoint_path: Optional path to a trained classifier
            `state_dict` (works for either `classifier_backbone`, since
            `VQCClassifier` and `DataReuploadingVQC` share the same
            `n_qubits`/`n_layers`/`state_dict` shape); if omitted, the
            classifier is randomly initialized (matches the pre-refactor
            demo behavior of app.py/qknee_frontend.py).
        encoder_type: `"pca"` (default) loads a fitted
            `QuantumDimReducer` artifact, exactly as before.
            `"quantum_autoencoder"` instead builds a trainable
            `QuantumAutoencoder` (SWAP-test Hilbert-space compression) —
            see `qknee.models.quantum_autoencoder`.
        classifier_backbone: `"vqc"` (default) builds a `VQCClassifier`
            (single-shot angle encoding), exactly as before.
            `"data_reuploading"` instead builds a `DataReuploadingVQC`
            (re-encodes the input at every layer) — see
            `qknee.models.vqc_data_reuploading`.
        quantum_autoencoder_checkpoint_path: Optional path to a trained
            `QuantumAutoencoder` `state_dict`. Only used when
            `encoder_type="quantum_autoencoder"`; if omitted (or the
            path doesn't exist), the autoencoder is randomly initialized.
        device: Torch device string; defaults to `config.device` or
            CUDA-if-available.
    """

    def __init__(
        self,
        config: Optional[QKneeConfig] = None,
        pca_artifact_path: Optional[Union[str, Path]] = None,
        vqc_checkpoint_path: Optional[Union[str, Path]] = None,
        encoder_type: EncoderType = "pca",
        classifier_backbone: ClassifierBackbone = "vqc",
        quantum_autoencoder_checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        if encoder_type not in ("pca", "quantum_autoencoder"):
            raise ValueError(f"encoder_type must be 'pca' or 'quantum_autoencoder', got {encoder_type!r}")
        if classifier_backbone not in ("vqc", "data_reuploading"):
            raise ValueError(
                f"classifier_backbone must be 'vqc' or 'data_reuploading', got {classifier_backbone!r}"
            )

        self.config = config or load_config()
        self.encoder_type: EncoderType = encoder_type
        self.classifier_backbone: ClassifierBackbone = classifier_backbone
        self.device = torch.device(
            device or self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- Stage 1: ingestion ---
        self.ingestion = DataIngestion(train=False)

        # --- Stage 2: ResNet18 (eager PyTorch, or ONNX Runtime per
        # config.resnet.backend_engine — both expose the same
        # extract_features()/forward()/__call__ interface, so the rest of
        # the pipeline (and Grad-CAM's backbone gradient hooks, which only
        # work with the PyTorch backend) doesn't need to know which is active) ---
        try:
            if self.config.resnet.backend_engine == "onnx":
                onnx_path = self.config.resnet.onnx_path
                if not onnx_path.exists():
                    raise PipelineValidationError(
                        f"resnet.backend_engine='onnx' but no ONNX model found at {onnx_path}. "
                        "Export one first via `python scripts/export_onnx.py`."
                    )
                self.feature_extractor = ONNXFeatureExtractor(onnx_path=onnx_path)
            else:
                self.feature_extractor = ResNet18FeatureExtractor(
                    freeze_backbone=self.config.resnet.freeze_backbone
                )
                self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
        except PipelineValidationError:
            raise
        except Exception as exc:
            raise PipelineValidationError(f"Failed to initialize ResNet18 backbone: {exc}") from exc

        # --- Stage 3: dimensionality reduction (PCA or Quantum Autoencoder) ---
        self.reducer: Optional[QuantumDimReducer] = None
        self.pca_layer: Optional[PCAProjectionLayer] = None
        self.quantum_autoencoder: Optional[QuantumAutoencoder] = None

        if encoder_type == "pca":
            pca_artifact_path = Path(pca_artifact_path or self.config.paths.pca_artifact)
            if not pca_artifact_path.exists():
                raise PipelineValidationError(
                    f"PCA artifact not found at {pca_artifact_path}. Fit and save a "
                    "QuantumDimReducer (qknee/models/pca_reducer.py) before running the pipeline."
                )
            try:
                self.reducer = QuantumDimReducer.load(pca_artifact_path)
            except Exception as exc:
                raise PipelineValidationError(
                    f"Failed to load PCA artifact from {pca_artifact_path}: {exc}"
                ) from exc

            if self.reducer.n_components != self.config.quantum.n_qubits:
                raise PipelineValidationError(
                    f"PCA artifact produces {self.reducer.n_components}-D output but "
                    f"config.quantum.n_qubits={self.config.quantum.n_qubits}; these must match."
                )

            # Differentiable re-expression of the fitted (sklearn) reducer, used
            # only by explain() so Grad-CAM can backprop a risk score all the
            # way through the PCA stage into the ResNet backbone. classify()
            # still uses self.reducer.transform() directly for the numerically
            # exact, non-differentiable inference path.
            try:
                self.pca_layer = PCAProjectionLayer.from_reducer(self.reducer).to(self.device)
            except Exception as exc:
                raise PipelineValidationError(f"Failed to build differentiable PCA layer for Grad-CAM: {exc}") from exc
        else:  # encoder_type == "quantum_autoencoder"
            try:
                self.quantum_autoencoder = QuantumAutoencoder(
                    feature_dim=self.config.resnet.feature_dim,
                    n_latent_qubits=self.config.quantum.n_qubits,
                )
            except Exception as exc:
                raise PipelineValidationError(f"Failed to initialize QuantumAutoencoder: {exc}") from exc

            qae_checkpoint_path = Path(quantum_autoencoder_checkpoint_path) if quantum_autoencoder_checkpoint_path else None
            if qae_checkpoint_path and qae_checkpoint_path.exists():
                try:
                    self.quantum_autoencoder.load_state_dict(torch.load(qae_checkpoint_path, map_location=self.device))
                    logger.info("Loaded trained QuantumAutoencoder weights from %s", qae_checkpoint_path)
                except Exception as exc:
                    raise PipelineValidationError(
                        f"Failed to load QuantumAutoencoder checkpoint from {qae_checkpoint_path}: {exc}"
                    ) from exc
            else:
                logger.warning(
                    "No QuantumAutoencoder checkpoint found at %s; using randomly initialized weights.",
                    qae_checkpoint_path,
                )
            self.quantum_autoencoder.to(self.device)
            self.quantum_autoencoder.eval()

        # --- Stage 4: classifier (VQC or Data-Re-Uploading VQC) ---
        classifier_cls = VQCClassifier if classifier_backbone == "vqc" else DataReuploadingVQC
        self.vqc = classifier_cls(
            n_qubits=self.config.quantum.n_qubits,
            n_layers=self.config.quantum.n_layers,
        )
        checkpoint_path = Path(vqc_checkpoint_path or self.config.paths.model_checkpoint)
        if checkpoint_path.exists():
            self._load_vqc_checkpoint(checkpoint_path)
            logger.info("Loaded trained %s weights from %s", classifier_cls.__name__, checkpoint_path)
        else:
            logger.warning(
                "No %s checkpoint found at %s; using randomly initialized weights.",
                classifier_cls.__name__, checkpoint_path,
            )
        self.vqc.to(self.device)
        self.vqc.eval()  # standard PyTorch inference mode: disables dropout/BatchNorm updates

        # --- Stage 5: Grad-CAM ---
        # ONNX Runtime is inference-only (no autograd graph to hook), so
        # Grad-CAM (which backprops through the ResNet backbone) only
        # works with the "pytorch" backend; explain() raises a clear error
        # under "onnx" instead of failing deep inside a hook registration.
        self.gradcam_target_layer = (
            get_default_target_layer(self.feature_extractor)
            if self.config.resnet.backend_engine != "onnx" else None
        )

        logger.info(
            "PipelineRunner ready (device=%s, encoder_type=%s, classifier_backbone=%s, n_qubits=%d, n_layers=%d)",
            self.device,
            encoder_type,
            classifier_backbone,
            self.config.quantum.n_qubits,
            self.config.quantum.n_layers,
        )

    # ------------------------------------------------------------------ #
    # Checkpoint loading
    # ------------------------------------------------------------------ #
    def _load_vqc_checkpoint(self, checkpoint_path: Path) -> None:
        load_vqc_weights(self.vqc, checkpoint_path, device=self.device)

    # ------------------------------------------------------------------ #
    # Stage-level validation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_stage_output(
        tensor: Union[torch.Tensor, np.ndarray],
        expected_ndim: int,
        expected_last_dim: Optional[int],
        stage_name: str,
    ) -> None:
        shape = tuple(tensor.shape)
        if len(shape) != expected_ndim:
            raise PipelineValidationError(
                f"[{stage_name}] expected a {expected_ndim}D output, got shape {shape}"
            )
        if expected_last_dim is not None and shape[-1] != expected_last_dim:
            raise PipelineValidationError(
                f"[{stage_name}] expected last dimension {expected_last_dim}, got shape {shape}"
            )
        array = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
        if not np.all(np.isfinite(array)):
            raise PipelineValidationError(f"[{stage_name}] output contains NaN/Inf values")

    # ------------------------------------------------------------------ #
    # Individual stages (public, so callers can compose their own flow —
    # e.g. reusing one quantum-feature vector across multiple VQC heads)
    # ------------------------------------------------------------------ #
    def ingest(self, source: InputType) -> torch.Tensor:
        """Stage 1: raw input -> `(1, S, 3, 224, 224)` tensor batch."""
        try:
            batch = self.ingestion.preprocess(source)
        except IngestionError as exc:
            raise PipelineValidationError(f"[DataIngestion] {exc}") from exc
        self._validate_stage_output(batch, expected_ndim=5, expected_last_dim=None, stage_name="DataIngestion")
        return batch

    def extract_resnet_features(self, batch: torch.Tensor) -> np.ndarray:
        """Stage 2: `(1, S, 3, 224, 224)` batch -> `(1, 512)` ResNet18 embedding."""
        try:
            with torch.no_grad():
                features = self.feature_extractor.extract_features(batch.to(self.device))
        except Exception as exc:
            raise PipelineValidationError(f"[ResNet18] forward pass failed: {exc}") from exc

        self._validate_stage_output(
            features, expected_ndim=2, expected_last_dim=self.config.resnet.feature_dim, stage_name="ResNet18"
        )
        return features.cpu().numpy()

    def reduce_to_quantum_angles(self, features_512d: np.ndarray) -> np.ndarray:
        """Stage 3: `(1, 512)` ResNet features -> `(1, n_qubits)` angles in
        `[0, 2*pi]`, via whichever `encoder_type` this runner was built
        with (`"pca"` -> `QuantumDimReducer.transform`, or
        `"quantum_autoencoder"` -> `QuantumAutoencoder`'s forward pass)."""
        stage_name = "PCA" if self.encoder_type == "pca" else "QuantumAutoencoder"
        try:
            if self.encoder_type == "pca":
                angles = self.reducer.transform(features_512d)
            else:
                with torch.no_grad():
                    features_tensor = torch.from_numpy(features_512d).float().to(self.device)
                    latent_angles = self.quantum_autoencoder.compress(features_tensor)
                angles = latent_angles.cpu().numpy()
        except Exception as exc:
            raise PipelineValidationError(f"[{stage_name}] transform failed: {exc}") from exc

        self._validate_stage_output(
            angles, expected_ndim=2, expected_last_dim=self.config.quantum.n_qubits, stage_name=stage_name
        )
        low, high = self.config.pca.angle_range
        if angles.min() < low - 1e-6 or angles.max() > high + 1e-6:
            raise PipelineValidationError(
                f"[{stage_name}] output {angles.min():.4f}..{angles.max():.4f} outside expected range [{low}, {high}]"
            )
        return angles

    def classify(self, quantum_angles: np.ndarray, vqc: Optional[Union[VQCClassifier, DataReuploadingVQC]] = None) -> float:
        """Stage 4: `(1, n_qubits)` angles -> scalar risk probability in `[0, 1]`."""
        model = vqc or self.vqc
        angles_tensor = torch.from_numpy(quantum_angles).float().to(self.device)
        try:
            with torch.no_grad():
                risk = model(angles_tensor)
        except Exception as exc:
            raise PipelineValidationError(f"[VQC] forward pass failed: {exc}") from exc

        self._validate_stage_output(risk, expected_ndim=2, expected_last_dim=1, stage_name="VQC")
        risk_value = float(risk.item())
        if not 0.0 <= risk_value <= 1.0:
            raise PipelineValidationError(f"[VQC] risk score {risk_value} outside expected range [0, 1]")
        return risk_value

    def _risk_target_fn(self, vqc: Optional[Union[VQCClassifier, DataReuploadingVQC]] = None) -> TargetFn:
        """Builds a Grad-CAM `target_fn` that continues the forward pass from
        a ResNet embedding through the differentiable PCA layer and `vqc`
        (defaulting to `self.vqc`) to the scalar predicted risk probability —
        so backpropagating from it highlights the image regions that
        actually drove *that* risk score, not just the embedding's energy.
        """
        model = vqc or self.vqc

        def risk_target(resnet_output: torch.Tensor) -> torch.Tensor:
            if self.encoder_type == "pca":
                angles = self.pca_layer(resnet_output)          # (B, n_qubits), differentiable
            else:
                angles = self.quantum_autoencoder.compress(resnet_output)  # natively differentiable
            risk = model(angles)                      # (B, 1)
            return risk.squeeze()                     # 0-dim scalar for .backward()

        return risk_target

    def explain(
        self,
        single_slice_tensor: torch.Tensor,
        vqc: Optional[Union[VQCClassifier, DataReuploadingVQC]] = None,
        target_fn: Optional[TargetFn] = None,
    ) -> np.ndarray:
        """Stage 5: `(1, 3, 224, 224)` single-slice tensor -> `(H, W)` Grad-CAM
        heatmap in `[0, 1]`, backpropagated from the predicted tear-risk
        probability (not embedding energy), so the heatmap explains the
        actual prediction.

        Args:
            single_slice_tensor: `(1, 3, 224, 224)` preprocessed slice.
            vqc: Optional VQC head to target instead of `self.vqc` — e.g. the
                dashboard's separate ACL/meniscus heads, so each condition's
                heatmap reflects that condition's own prediction.
            target_fn: Optional full override of the backprop target, for
                callers that want something other than risk-score Grad-CAM
                (e.g. embedding-energy, by passing `lambda x: x.pow(2).sum()`).
        """
        if single_slice_tensor.dim() != 4:
            raise PipelineValidationError(
                f"[GradCAM] expects a single-slice (1, 3, 224, 224) tensor, got {tuple(single_slice_tensor.shape)}"
            )
        if self.gradcam_target_layer is None:
            raise PipelineValidationError(
                "[GradCAM] not available with resnet.backend_engine='onnx' (ONNX Runtime is "
                "inference-only, no autograd graph to backprop through). Build this "
                "PipelineRunner with backend_engine='pytorch' to use explain()/Grad-CAM."
            )
        resolved_target_fn = target_fn or self._risk_target_fn(vqc)
        try:
            with GradCAM(self.feature_extractor, self.gradcam_target_layer) as cam:
                heatmap = cam.generate(single_slice_tensor.to(self.device), target_fn=resolved_target_fn)
        except Exception as exc:
            raise PipelineValidationError(f"[GradCAM] generation failed: {exc}") from exc

        self._validate_stage_output(heatmap, expected_ndim=2, expected_last_dim=None, stage_name="GradCAM")
        return heatmap

    # ------------------------------------------------------------------ #
    # Convenience: ingestion + ResNet + PCA in one call (mirrors the
    # pre-refactor MRIQuantumPipeline.extract_quantum_features API)
    # ------------------------------------------------------------------ #
    def extract_quantum_features(self, source: InputType) -> np.ndarray:
        batch = self.ingest(source)
        features = self.extract_resnet_features(batch)
        return self.reduce_to_quantum_angles(features)

    # ------------------------------------------------------------------ #
    # Full orchestration
    # ------------------------------------------------------------------ #
    def run(self, source: InputType, skip_gradcam: bool = False) -> PipelineResult:
        """Runs the full DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM
        chain on one input and returns a `PipelineResult`.

        Args:
            source: MRI slice path, volume path, in-memory array, or PIL image.
            skip_gradcam: If True, omits Stage 5 (useful for latency-sensitive
                batch scoring where explainability isn't needed per-call).
        """
        logger.debug("PipelineRunner.run: starting for source=%r", source)

        batch = self.ingest(source)
        features_512d = self.extract_resnet_features(batch)
        quantum_angles = self.reduce_to_quantum_angles(features_512d)
        risk_score = self.classify(quantum_angles)

        heatmap = None
        if not skip_gradcam:
            # Grad-CAM needs a single (1, 3, 224, 224) slice; use the central
            # slice of the volume (multi-slice volumes are averaged upstream
            # for classification, but Grad-CAM visualizes one representative
            # slice) — the anatomical midpoint is a more representative choice
            # than an arbitrary edge slice for a multi-slice MRI stack.
            central_slice_index = batch.shape[1] // 2
            single_slice = batch[:, central_slice_index]
            heatmap = self.explain(single_slice)

        logger.info("PipelineRunner.run: risk_score=%.4f, source=%r", risk_score, source)
        return PipelineResult(risk_score=risk_score, quantum_angles=quantum_angles, gradcam_heatmap=heatmap)


# --------------------------------------------------------------------------- #
# HybridONNXInferenceEngine — the runtime counterpart of
# `scripts/export_onnx.py`'s decoupled export.
#
# Why decoupled: `torch.onnx.export()` cannot trace `VQCClassifier` as a
# single graph. `qml.qnn.TorchLayer` wraps a PennyLane QNode whose
# `default.qubit` simulator represents the quantum state vector as a
# complex128 tensor; when the TorchScript-based ONNX exporter traces
# through that QNode call, the resulting op trace contains raw complex-
# arithmetic ops with no ONNX equivalent, and the export fails with
# `RuntimeError: Unknown number type: complex` (reproduced by
# `scripts/export_onnx.py`'s `demonstrate_vqc_export_failure()`). This is
# the "graph-tracing mismatch" between an eager-mode quantum simulator and
# ONNX's static, real-valued-tensor op set — it isn't a bug to work around
# with a different opset or exporter flag, it's a fundamental representation
# gap, so the fix is architectural: only export the classical, ordinary-
# tensor-arithmetic half of the model (ResNet18 + the PCA-projection
# bottleneck) to ONNX; keep the quantum half as raw parameter tensors
# (`qknee_vqc_weights.pt`) plus a JSON description of the circuit structure
# (`circuit_params.json`), and re-execute it directly through PennyLane (or
# Qiskit Aer) at inference time — never through a traced ONNX graph.
# --------------------------------------------------------------------------- #

def _resolve_quantum_device(backend: QuantumBackend, n_qubits: int):
    """Builds the PennyLane device `HybridONNXInferenceEngine` runs the
    circuit on. `"pennylane"` (default) uses the same `default.qubit`
    state-vector simulator the rest of this project trains against.
    `"qiskit"` instead delegates simulation to Qiskit Aer via the optional
    `pennylane-qiskit` plugin — useful for validating against IBM's own
    simulator/noise-model stack, at the cost of an extra dependency."""
    if backend == "qiskit":
        try:
            return qml.device("qiskit.aer", wires=n_qubits, backend="aer_simulator")
        except Exception as exc:
            raise PipelineValidationError(
                "quantum_backend='qiskit' requires the optional 'pennylane-qiskit' plugin "
                "(and 'qiskit'/'qiskit-aer'): pip install pennylane-qiskit qiskit-aer. "
                f"Original error: {exc}"
            ) from exc
    return qml.device(load_config().quantum.device, wires=n_qubits)


class HybridONNXInferenceEngine:
    """Loads the two decoupled export artifacts (`resnet_feature_extractor.onnx`
    + `qknee_vqc_weights.pt`/`circuit_params.json`) and reproduces
    `QKneeModel`'s full forward pass — ResNet18 -> PCA(4) -> 4-qubit VQC ->
    sigmoid risk score — without ever loading PyTorch's autograd machinery
    for the classical half (ONNX Runtime instead) or tracing the quantum
    half through ONNX at all (a raw PennyLane/Qiskit QNode instead).

    Args:
        resnet_onnx_path: Path to the exported ResNet18+PCA ONNX graph
            (`(B, 3, 224, 224) -> (B, n_qubits)` angles in `[0, 2*pi]`).
        vqc_weights_path: Path to the `.pt` file holding the quantum
            circuit's rotation weights and the classical readout layer's
            weight/bias (`scripts/export_onnx.py::export_vqc_weights_and_circuit_params`).
        circuit_params_path: Optional path to the circuit's structural
            description (`circuit_params.json`); if given, its
            `n_qubits`/`n_layers` are cross-checked against
            `vqc_weights_path`'s so a mismatched pair of export artifacts
            fails loudly at construction time instead of silently
            producing wrong predictions.
        quantum_backend: `"pennylane"` (default, `default.qubit`) or
            `"qiskit"` (Qiskit Aer via `pennylane-qiskit`) — see
            `_resolve_quantum_device`.
        onnx_providers: ONNX Runtime execution providers, in priority
            order; defaults to GPU-if-available, else CPU (same policy as
            `ONNXFeatureExtractor`).
        intra_op_num_threads: CPU threads for the ONNX Runtime session's
            intra-op parallelism; defaults to all available cores.
    """

    def __init__(
        self,
        resnet_onnx_path: Union[str, Path] = DEFAULT_RESNET_ONNX_PATH,
        vqc_weights_path: Union[str, Path] = DEFAULT_VQC_WEIGHTS_PATH,
        circuit_params_path: Optional[Union[str, Path]] = DEFAULT_CIRCUIT_PARAMS_PATH,
        quantum_backend: QuantumBackend = "pennylane",
        onnx_providers: Optional[List[str]] = None,
        intra_op_num_threads: Optional[int] = None,
    ) -> None:
        import onnxruntime as ort

        resnet_onnx_path = Path(resnet_onnx_path)
        vqc_weights_path = Path(vqc_weights_path)
        if not resnet_onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX feature extractor not found at {resnet_onnx_path}. Export one first via "
                "`python scripts/export_onnx.py`."
            )
        if not vqc_weights_path.exists():
            raise FileNotFoundError(
                f"VQC weights not found at {vqc_weights_path}. Export them first via "
                "`python scripts/export_onnx.py`."
            )

        # --- Classical half: ONNX Runtime session (ResNet18 + PCA(4)) ---
        if onnx_providers is None:
            available = ort.get_available_providers()
            onnx_providers = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else [])
            onnx_providers.append("CPUExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_op_num_threads or (os.cpu_count() or 1)
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(resnet_onnx_path), sess_options=session_options, providers=onnx_providers,
        )
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

        # --- Quantum half: raw weight tensors + a from-scratch (non-Torch,
        # non-ONNX) PennyLane/Qiskit QNode, executed directly at inference
        # time instead of through any traced graph ---
        weights_blob = torch.load(vqc_weights_path, map_location="cpu")
        self.n_qubits = int(weights_blob["n_qubits"])
        self.n_layers = int(weights_blob["n_layers"])
        self.quantum_weights = weights_blob["quantum_weights"].detach().cpu().numpy()
        self.readout_weight = weights_blob["readout_weight"].detach().cpu().numpy()  # (1, n_qubits)
        self.readout_bias = weights_blob["readout_bias"].detach().cpu().numpy()      # (1,)

        self.circuit_params: Optional[Dict] = None
        if circuit_params_path is not None and Path(circuit_params_path).exists():
            with open(circuit_params_path, "r", encoding="utf-8") as handle:
                self.circuit_params = json.load(handle)
            if (
                self.circuit_params.get("n_qubits") != self.n_qubits
                or self.circuit_params.get("n_layers") != self.n_layers
            ):
                raise PipelineValidationError(
                    f"circuit_params.json (n_qubits={self.circuit_params.get('n_qubits')}, "
                    f"n_layers={self.circuit_params.get('n_layers')}) disagrees with "
                    f"{vqc_weights_path} (n_qubits={self.n_qubits}, n_layers={self.n_layers}) "
                    "— these two export artifacts must come from the same trained VQC."
                )

        self.quantum_backend = quantum_backend
        self._circuit = self._build_inference_circuit(quantum_backend)

        logger.info(
            "HybridONNXInferenceEngine ready (onnx_provider=%s, quantum_backend=%s, n_qubits=%d, n_layers=%d)",
            self.session.get_providers()[0], quantum_backend, self.n_qubits, self.n_layers,
        )

    def _build_inference_circuit(self, quantum_backend: QuantumBackend):
        """Builds a bare PennyLane QNode with `interface=None` (plain numpy
        in/out, no autograd) — deliberately *not* `qml.qnn.TorchLayer`,
        since this engine is inference-only and skipping Torch's autograd
        bookkeeping is a genuine (if modest) CPU speedup on top of avoiding
        the ONNX-tracing problem entirely."""
        device = _resolve_quantum_device(quantum_backend, self.n_qubits)
        wires = list(range(self.n_qubits))
        n_layers = self.n_layers

        @qml.qnode(device, interface=None)
        def circuit(inputs, weights):
            angle_encoding(inputs, wires)
            for layer in range(n_layers):
                variational_block(weights[layer], wires)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        return circuit

    def extract_angles(self, batch: torch.Tensor) -> np.ndarray:
        """Classical stage: `(B, 3, 224, 224)` -> `(B, n_qubits)` angles in
        `[0, 2*pi]`, via the ONNX Runtime session (ResNet18 + PCA(4))."""
        input_array = batch.detach().cpu().numpy().astype(np.float32)
        angles = self.session.run([self._output_name], {self._input_name: input_array})[0]
        return angles

    def run_quantum_circuit(self, angles: np.ndarray) -> np.ndarray:
        """Quantum stage: `(B, n_qubits)` angles -> `(B, n_qubits)` Pauli-Z
        expectation values, one QNode evaluation per row.

        This per-sample loop *is* the "optimized CPU loop": `default.qubit`
        (or Qiskit Aer) simulates each circuit with vectorized C/numpy
        tensor contractions internally, so the Python-level loop over the
        batch dimension is cheap relative to each simulation — matching how
        the dashboard/API actually serve one slice at a time, rather than
        claiming a batched-throughput number no live request would see.
        """
        angles = np.asarray(angles, dtype=np.float64)
        expvals = np.stack([
            np.asarray(self._circuit(angles[i], self.quantum_weights), dtype=np.float64)
            for i in range(angles.shape[0])
        ])
        return expvals

    def classify(self, expvals: np.ndarray) -> np.ndarray:
        """Classical readout: `(B, n_qubits)` expectation values -> `(B, 1)`
        risk probabilities, via the exported `Linear(n_qubits, 1) + Sigmoid`
        (plain numpy — the readout layer is tiny, no need for ONNX Runtime
        here)."""
        logits = expvals @ self.readout_weight.T + self.readout_bias  # (B, 1)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, batch: torch.Tensor) -> np.ndarray:
        """Full decoupled forward pass: `(B, 3, 224, 224)` -> `(B, 1)` risk
        probabilities, matching `QKneeModel.forward`'s output exactly (up
        to floating-point export/quantization error — see
        `scripts/export_onnx.py`'s `validate_decoupled_export`)."""
        angles = self.extract_angles(batch)
        expvals = self.run_quantum_circuit(angles)
        return self.classify(expvals)

    def __call__(self, batch: torch.Tensor) -> np.ndarray:
        return self.predict(batch)


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()

    artifact_path = Path(load_config().paths.pca_artifact)
    if not artifact_path.exists():
        logger.info("No PCA artifact found at %s - fitting a dummy one for smoke-testing pipeline.py", artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.random.seed(0)
        dummy_512d = np.random.randn(500, 512).astype(np.float32)
        QuantumDimReducer().fit(dummy_512d).save(artifact_path)

    runner = PipelineRunner(pca_artifact_path=artifact_path)

    dummy_slice = np.random.randint(0, 255, size=(224, 224), dtype=np.uint8)
    result = runner.run(dummy_slice)
    assert 0.0 <= result.risk_score <= 1.0
    assert result.quantum_angles.shape == (1, 4)
    assert result.gradcam_heatmap is not None and result.gradcam_heatmap.ndim == 2
    logger.info("PipelineRunner smoke test passed: risk_score=%.4f", result.risk_score)
