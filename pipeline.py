"""
Master inference pipeline: raw MRI slice/volume -> 4-D quantum-ready vector.

Chains together the three previously-built stages:
    1. Ingestion & preprocessing  (mri_dataset.build_transforms)
    2. ResNet18 feature extraction (resnet_feature_extractor.ResNet18FeatureExtractor)
    3. PCA -> [0, 2*pi] reduction   (quantum_dim_reduction.QuantumDimReducer)

Exposes a single entry point for downstream consumers:

    pipeline = MRIQuantumPipeline(pca_artifact_path="pca_scaler.pkl")
    processed_vector = pipeline.extract_quantum_features(image_path)
    # -> np.ndarray of shape (1, 4), values in [0, 2*pi]

`extract_quantum_features` accepts, interchangeably:
    - a path (str/Path) to a PNG/JPEG slice
    - a path (str/Path) to a .npy volume (multi-slice; averaged into one
      embedding before PCA)
    - an in-memory np.ndarray (2D single slice, or 3D/4D volume)
    - a PIL.Image.Image

The PCA/scaler artifact must already be fitted (see quantum_dim_reduction.py)
on a representative corpus of 512-D ResNet features before this pipeline is
used for inference; this class only loads and applies it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from mri_dataset import IMAGE_EXTENSIONS, VOLUME_EXTENSIONS, build_transforms
from quantum_dim_reduction import QuantumDimReducer
from resnet_feature_extractor import ResNet18FeatureExtractor

logger = logging.getLogger(__name__)

InputType = Union[str, Path, np.ndarray, Image.Image]


class PipelineError(RuntimeError):
    """Raised when any stage of the pipeline fails on a given input."""


class MRIQuantumPipeline:
    """End-to-end MRI slice/volume -> 4-D quantum feature vector pipeline.

    Args:
        pca_artifact_path: Path to the joblib-persisted, pre-fitted
            `QuantumDimReducer` (see quantum_dim_reduction.py).
        device: torch device string for the ResNet18 backbone
            ("cpu", "cuda", or "cuda:0" etc). Defaults to CUDA if available.
    """

    def __init__(
        self,
        pca_artifact_path: str | Path = "pca_scaler.pkl",
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        try:
            self.feature_extractor = ResNet18FeatureExtractor(freeze_backbone=True)
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
        except Exception as exc:
            raise PipelineError(f"Failed to initialize ResNet18 backbone: {exc}") from exc

        pca_artifact_path = Path(pca_artifact_path)
        if not pca_artifact_path.exists():
            raise PipelineError(
                f"PCA artifact not found at {pca_artifact_path}. "
                "Fit and save a QuantumDimReducer (see quantum_dim_reduction.py) "
                "before running inference."
            )
        try:
            self.reducer = QuantumDimReducer.load(pca_artifact_path)
        except Exception as exc:
            raise PipelineError(
                f"Failed to load PCA artifact from {pca_artifact_path}: {exc}"
            ) from exc

        self.eval_transform = build_transforms(train=False)

        logger.info(
            "MRIQuantumPipeline ready (device=%s, pca_artifact=%s)",
            self.device,
            pca_artifact_path,
        )

    # ------------------------------------------------------------------ #
    # Stage 1: ingestion -> normalized (S, 3, 224, 224) tensor of slices
    # ------------------------------------------------------------------ #
    def _load_slices_as_pil(self, source: InputType) -> list[Image.Image]:
        """Normalizes any accepted input type into a list of grayscale PIL
        images (one per slice; length 1 for a single 2D image)."""

        if isinstance(source, Image.Image):
            return [source.convert("L")]

        if isinstance(source, np.ndarray):
            return self._array_to_pil_slices(source)

        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise PipelineError(f"Input file does not exist: {path}")

            ext = path.suffix.lower()
            if ext in VOLUME_EXTENSIONS:
                try:
                    volume = np.load(path)
                except (ValueError, OSError, EOFError) as exc:
                    raise PipelineError(f"Corrupted/unreadable .npy volume {path}: {exc}") from exc
                return self._array_to_pil_slices(volume)

            if ext in IMAGE_EXTENSIONS:
                try:
                    with Image.open(path) as img:
                        img.load()
                        return [img.convert("L")]
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    raise PipelineError(f"Corrupted/unreadable image {path}: {exc}") from exc

            raise PipelineError(
                f"Unsupported file extension '{ext}' for {path}. "
                f"Expected one of {IMAGE_EXTENSIONS | VOLUME_EXTENSIONS}"
            )

        raise PipelineError(
            f"Unsupported input type {type(source)}. Expected str, Path, "
            "np.ndarray, or PIL.Image.Image."
        )

    @staticmethod
    def _array_to_pil_slices(array: np.ndarray) -> list[Image.Image]:
        """Converts a 2D single-slice array or 3D/4D volume array into a
        list of 8-bit grayscale PIL images, min-max normalized per slice."""
        array = np.asarray(array)

        if array.ndim == 2:
            slices = [array]
        elif array.ndim in (3, 4):
            slices = [np.asarray(array[i]) for i in range(array.shape[0])]
        else:
            raise PipelineError(
                f"Expected a 2D slice or 3D/4D volume array, got shape {array.shape}"
            )

        pil_slices = []
        for slice_array in slices:
            if slice_array.ndim == 3 and slice_array.shape[-1] in (3, 4):
                slice_array = slice_array[..., :3].mean(axis=-1)
            slice_array = slice_array.astype(np.float32)

            min_val, max_val = float(slice_array.min()), float(slice_array.max())
            if max_val > min_val:
                slice_array = (slice_array - min_val) / (max_val - min_val)
            else:
                slice_array = np.zeros_like(slice_array)

            slice_array = (slice_array * 255.0).astype(np.uint8)
            pil_slices.append(Image.fromarray(slice_array, mode="L"))

        return pil_slices

    def _preprocess(self, source: InputType) -> torch.Tensor:
        """Returns a (1, S, 3, 224, 224) tensor ready for the ResNet18 backbone."""
        pil_slices = self._load_slices_as_pil(source)
        if not pil_slices:
            raise PipelineError("Input produced zero usable slices.")

        try:
            tensors = [self.eval_transform(img) for img in pil_slices]
        except Exception as exc:
            raise PipelineError(f"Preprocessing/transform stage failed: {exc}") from exc

        stacked = torch.stack(tensors, dim=0)  # (S, 3, 224, 224)
        return stacked.unsqueeze(0)  # (1, S, 3, 224, 224)

    # ------------------------------------------------------------------ #
    # Stage 2: ResNet18 -> 512-D embedding
    # ------------------------------------------------------------------ #
    def _extract_resnet_features(self, batch: torch.Tensor) -> np.ndarray:
        """Runs the (1, S, 3, 224, 224) batch through ResNet18 and returns a
        (1, 512) numpy array (volume-averaged if S > 1)."""
        try:
            with torch.no_grad():
                batch = batch.to(self.device)
                features = self.feature_extractor(batch)  # (1, 512) via forward_volume/forward_slice
        except Exception as exc:
            raise PipelineError(f"ResNet18 forward pass failed: {exc}") from exc

        return features.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Stage 3: PCA -> [0, 2*pi]
    # ------------------------------------------------------------------ #
    def _reduce_to_quantum_vector(self, features_512d: np.ndarray) -> np.ndarray:
        try:
            return self.reducer.transform(features_512d)  # (1, 4)
        except Exception as exc:
            raise PipelineError(f"PCA/quantum-scaling stage failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def extract_quantum_features(
        self,
        source: InputType,
        as_tensor: bool = False,
        verbose: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Runs the full ingestion -> ResNet18 -> PCA pipeline on one input.

        Args:
            source: MRI slice path (.png/.jpg/.jpeg), volume path (.npy),
                in-memory np.ndarray, or PIL.Image.Image.
            as_tensor: If True, returns a torch.Tensor instead of np.ndarray.
            verbose: If True, prints shape/range diagnostics of the output
                vector.

        Returns:
            Array/tensor of shape (1, 4) with values in [0, 2*pi], ready for
            a 4-qubit Angle Encoding VQC.

        Raises:
            PipelineError: If any stage fails (missing/corrupted file,
                unsupported format, shape mismatch, etc). The original
                exception is chained for debugging.
        """
        try:
            batch = self._preprocess(source)
            features_512d = self._extract_resnet_features(batch)
            quantum_vector = self._reduce_to_quantum_vector(features_512d)
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"Unexpected pipeline failure for input {source!r}: {exc}") from exc

        if verbose:
            self._print_diagnostics(source, quantum_vector)

        if as_tensor:
            return torch.from_numpy(quantum_vector).float()
        return quantum_vector

    @staticmethod
    def _print_diagnostics(source: InputType, vector: np.ndarray) -> None:
        label = source if isinstance(source, (str, Path)) else type(source).__name__
        print(f"[MRIQuantumPipeline] source={label}")
        print(f"  shape : {vector.shape}")
        print(f"  values: {np.array2string(vector.flatten(), precision=4, floatmode='fixed')}")
        print(f"  range : [{vector.min():.4f}, {vector.max():.4f}]  (target: [0.0000, {2*np.pi:.4f}])")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Fit a throwaway PCA artifact from dummy data if one doesn't exist yet,
    # purely so this smoke test is runnable standalone.
    artifact_path = Path("pca_scaler.pkl")
    if not artifact_path.exists():
        print("No pca_scaler.pkl found — fitting a dummy one for smoke-testing pipeline.py")
        from quantum_dim_reduction import QuantumDimReducer

        np.random.seed(0)
        dummy_512d = np.random.randn(500, 512).astype(np.float32)
        QuantumDimReducer().fit(dummy_512d).save(artifact_path)

    pipeline = MRIQuantumPipeline(pca_artifact_path=artifact_path)

    # --- Smoke test 1: in-memory single-slice array ---
    dummy_slice = np.random.randint(0, 255, size=(224, 224), dtype=np.uint8)
    vector = pipeline.extract_quantum_features(dummy_slice)
    assert vector.shape == (1, 4)

    # --- Smoke test 2: in-memory multi-slice volume array ---
    dummy_volume = np.random.randint(0, 255, size=(8, 224, 224), dtype=np.uint8)
    vector_volume = pipeline.extract_quantum_features(dummy_volume, as_tensor=True)
    assert isinstance(vector_volume, torch.Tensor)
    assert tuple(vector_volume.shape) == (1, 4)

    # --- Smoke test 3: missing file -> PipelineError, not a crash ---
    try:
        pipeline.extract_quantum_features("does_not_exist.png")
        print("ERROR: expected PipelineError for missing file")
        sys.exit(1)
    except PipelineError as exc:
        print(f"\nCorrectly raised PipelineError for missing file: {exc}")

    print("\nAll pipeline.py smoke tests passed.")
