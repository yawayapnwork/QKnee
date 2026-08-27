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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from qknee.config.loader import QKneeConfig, load_config
from qknee.config.logging_config import get_logger
from qknee.data.ingestion import DataIngestion, IngestionError
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.qknee_model import PCAProjectionLayer
from qknee.models.resnet_extractor import ResNet18FeatureExtractor
from qknee.models.vqc import VQCClassifier
from qknee.xai.gradcam import GradCAM, TargetFn, get_default_target_layer, overlay_heatmap

logger = get_logger(__name__)

InputType = Union[str, Path, np.ndarray, Image.Image]


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
        pca_artifact_path: Overrides `config.paths.pca_artifact`.
        vqc_checkpoint_path: Optional path to a trained `VQCClassifier`
            `state_dict`; if omitted, the VQC is randomly initialized
            (matches the pre-refactor demo behavior of app.py/qknee_frontend.py).
        device: Torch device string; defaults to `config.device` or
            CUDA-if-available.
    """

    def __init__(
        self,
        config: Optional[QKneeConfig] = None,
        pca_artifact_path: Optional[Union[str, Path]] = None,
        vqc_checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or load_config()
        self.device = torch.device(
            device or self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- Stage 1: ingestion ---
        self.ingestion = DataIngestion(train=False)

        # --- Stage 2: ResNet18 ---
        try:
            self.feature_extractor = ResNet18FeatureExtractor(
                freeze_backbone=self.config.resnet.freeze_backbone
            )
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
        except Exception as exc:
            raise PipelineValidationError(f"Failed to initialize ResNet18 backbone: {exc}") from exc

        # --- Stage 3: PCA ---
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

        # --- Stage 4: VQC ---
        self.vqc = VQCClassifier(
            n_qubits=self.config.quantum.n_qubits,
            n_layers=self.config.quantum.n_layers,
        )
        checkpoint_path = Path(vqc_checkpoint_path or self.config.paths.model_checkpoint)
        if checkpoint_path.exists():
            self._load_vqc_checkpoint(checkpoint_path)
            logger.info("Loaded trained VQC weights from %s", checkpoint_path)
        else:
            logger.warning(
                "No VQC checkpoint found at %s; using randomly initialized weights.",
                checkpoint_path,
            )
        self.vqc.to(self.device)
        self.vqc.eval()  # standard PyTorch inference mode: disables dropout/BatchNorm updates

        # --- Stage 5: Grad-CAM ---
        self.gradcam_target_layer = get_default_target_layer(self.feature_extractor)

        logger.info(
            "PipelineRunner ready (device=%s, pca_artifact=%s, n_qubits=%d, n_layers=%d)",
            self.device,
            pca_artifact_path,
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
                features = self.feature_extractor(batch.to(self.device))
        except Exception as exc:
            raise PipelineValidationError(f"[ResNet18] forward pass failed: {exc}") from exc

        self._validate_stage_output(
            features, expected_ndim=2, expected_last_dim=self.config.resnet.feature_dim, stage_name="ResNet18"
        )
        return features.cpu().numpy()

    def reduce_to_quantum_angles(self, features_512d: np.ndarray) -> np.ndarray:
        """Stage 3: `(1, 512)` ResNet features -> `(1, n_qubits)` angles in `[0, 2*pi]`."""
        try:
            angles = self.reducer.transform(features_512d)
        except Exception as exc:
            raise PipelineValidationError(f"[PCA] transform failed: {exc}") from exc

        self._validate_stage_output(
            angles, expected_ndim=2, expected_last_dim=self.config.quantum.n_qubits, stage_name="PCA"
        )
        low, high = self.config.pca.angle_range
        if angles.min() < low - 1e-6 or angles.max() > high + 1e-6:
            raise PipelineValidationError(
                f"[PCA] output {angles.min():.4f}..{angles.max():.4f} outside expected range [{low}, {high}]"
            )
        return angles

    def classify(self, quantum_angles: np.ndarray, vqc: Optional[VQCClassifier] = None) -> float:
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

    def _risk_target_fn(self, vqc: Optional[VQCClassifier] = None) -> TargetFn:
        """Builds a Grad-CAM `target_fn` that continues the forward pass from
        a ResNet embedding through the differentiable PCA layer and `vqc`
        (defaulting to `self.vqc`) to the scalar predicted risk probability —
        so backpropagating from it highlights the image regions that
        actually drove *that* risk score, not just the embedding's energy.
        """
        model = vqc or self.vqc

        def risk_target(resnet_output: torch.Tensor) -> torch.Tensor:
            angles = self.pca_layer(resnet_output)   # (B, n_qubits), differentiable
            risk = model(angles)                      # (B, 1)
            return risk.squeeze()                     # 0-dim scalar for .backward()

        return risk_target

    def explain(
        self,
        single_slice_tensor: torch.Tensor,
        vqc: Optional[VQCClassifier] = None,
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
            # Grad-CAM needs a single (1, 3, 224, 224) slice; use the first
            # slice of the batch (multi-slice volumes are averaged upstream
            # for classification, but Grad-CAM visualizes one representative slice).
            single_slice = batch[:, 0]
            heatmap = self.explain(single_slice)

        logger.info("PipelineRunner.run: risk_score=%.4f, source=%r", risk_score, source)
        return PipelineResult(risk_score=risk_score, quantum_angles=quantum_angles, gradcam_heatmap=heatmap)


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
