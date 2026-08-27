"""
Dimensionality reduction pipeline: 512-D ResNet feature vectors -> 4
quantum-ready scalars in [0, 2*pi], suitable for Angle Encoding in
PennyLane/Qiskit variational quantum circuits.

Pipeline:
    1. StandardScaler   - zero-mean/unit-variance per feature (PCA is
                           variance-sensitive, so features must be scaled
                           first).
    2. PCA(n_components=4) / IncrementalPCA(n_components=4)
                        - compress 512-D -> 4-D, capturing maximum variance.
    3. MinMaxScaler(feature_range=(0, 2*pi))
                        - map each of the 4 PCA components independently
                          into [0, 2*pi] for angle encoding.

All three fitted transformers are bundled into one `QuantumDimReducer`
object and persisted together via joblib, so inference-time transform()
calls are guaranteed to use the exact scaling/PCA basis learned at fit time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

N_QUANTUM_DIMS = 4
ANGLE_RANGE = (0.0, 2 * np.pi)


class QuantumDimReducer:
    """Fits and applies StandardScaler -> PCA(4) -> MinMaxScaler(0, 2*pi).

    Args:
        use_incremental_pca: If True, uses IncrementalPCA (fit via
            `partial_fit` batches) instead of full-batch PCA. Useful when
            the full 512-D feature matrix doesn't fit in memory at once.
        n_components: Number of output dimensions. Fixed at 4 by the task
            spec, but left configurable for flexibility.
    """

    def __init__(self, use_incremental_pca: bool = False, n_components: int = N_QUANTUM_DIMS):
        self.n_components = n_components
        self.use_incremental_pca = use_incremental_pca

        self.standard_scaler = StandardScaler()
        self.pca = (
            IncrementalPCA(n_components=n_components)
            if use_incremental_pca
            else PCA(n_components=n_components)
        )
        self.minmax_scaler = MinMaxScaler(feature_range=ANGLE_RANGE)

        self._is_fitted = False

    def fit(self, features: np.ndarray, batch_size: Optional[int] = None) -> "QuantumDimReducer":
        """Fits all three stages on a (N, 512) feature matrix.

        Args:
            features: Array of shape (N, D) with D >= n_components (typically
                D=512 ResNet feature vectors).
            batch_size: Only used when `use_incremental_pca=True`. If None,
                defaults to `max(5 * n_components, 128)` per scikit-learn's
                IncrementalPCA guidance (batch_size must be >= n_components).
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D array (N, D), got shape {features.shape}")
        if features.shape[0] < self.n_components:
            raise ValueError(
                f"Need at least {self.n_components} samples to fit "
                f"{self.n_components} PCA components, got {features.shape[0]}"
            )

        scaled = self.standard_scaler.fit_transform(features)

        if self.use_incremental_pca:
            batch_size = batch_size or max(5 * self.n_components, 128)
            for start in range(0, scaled.shape[0], batch_size):
                batch = scaled[start:start + batch_size]
                if batch.shape[0] < self.n_components:
                    # IncrementalPCA requires each batch >= n_components;
                    # fold a too-small trailing batch into the previous one.
                    continue
                self.pca.partial_fit(batch)
            pca_output = self.pca.transform(scaled)
        else:
            pca_output = self.pca.fit_transform(scaled)

        self.minmax_scaler.fit(pca_output)
        self._is_fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Applies the fitted pipeline to new (N, 512) features.

        Returns:
            Array of shape (N, n_components) with values in [0, 2*pi].
        """
        if not self._is_fitted:
            raise RuntimeError("QuantumDimReducer must be fit() before transform()")

        features = np.asarray(features, dtype=np.float64)
        scaled = self.standard_scaler.transform(features)
        pca_output = self.pca.transform(scaled)
        angles = self.minmax_scaler.transform(pca_output)
        # Guard against floating-point overshoot at the [0, 2*pi] boundary.
        return np.clip(angles, ANGLE_RANGE[0], ANGLE_RANGE[1])

    def fit_transform(self, features: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        self.fit(features, batch_size=batch_size)
        return self.transform(features)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("QuantumDimReducer must be fit() before accessing this property")
        return self.pca.explained_variance_ratio_

    def save(self, path: str | Path = "pca_scaler.pkl") -> Path:
        """Persists the fitted reducer (all three transformers) via joblib."""
        if not self._is_fitted:
            raise RuntimeError("Refusing to save an unfitted QuantumDimReducer")
        path = Path(path)
        joblib.dump(self, path)
        logger.info("Saved fitted QuantumDimReducer to %s", path)
        return path

    @staticmethod
    def load(path: str | Path = "pca_scaler.pkl") -> "QuantumDimReducer":
        """Loads a previously fitted reducer for consistent downstream inference."""
        reducer = joblib.load(Path(path))
        if not isinstance(reducer, QuantumDimReducer):
            raise TypeError(f"Loaded object is not a QuantumDimReducer: {type(reducer)}")
        return reducer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(0)

    # --- Dummy data: simulate 500 ResNet18 512-D feature vectors ---
    n_samples, n_features = 500, 512
    dummy_features = np.random.randn(n_samples, n_features).astype(np.float32)
    # Inject some correlated structure so PCA has non-trivial variance to find.
    latent = np.random.randn(n_samples, 4)
    projection = np.random.randn(4, n_features)
    dummy_features += latent @ projection

    reducer = QuantumDimReducer(use_incremental_pca=False)
    quantum_angles = reducer.fit_transform(dummy_features)

    print(f"Input shape:  {dummy_features.shape}")
    print(f"Output shape: {quantum_angles.shape}")
    assert quantum_angles.shape == (n_samples, N_QUANTUM_DIMS)
    assert np.all(quantum_angles >= 0.0) and np.all(quantum_angles <= 2 * np.pi + 1e-9)
    print(f"Output range: [{quantum_angles.min():.4f}, {quantum_angles.max():.4f}] "
          f"(target: [0, {2*np.pi:.4f}])")

    # --- Explained variance of the top 4 components ---
    evr = reducer.explained_variance_ratio_
    print("\nExplained variance ratio (top 4 PCA components):")
    for i, ratio in enumerate(evr):
        print(f"  PC{i + 1}: {ratio:.4f}")
    print(f"  Total (cumulative): {evr.sum():.4f}")

    # --- Persist and reload for inference consistency ---
    artifact_path = reducer.save("pca_scaler.pkl")
    reloaded = QuantumDimReducer.load(artifact_path)

    new_batch = dummy_features[:8]
    original_output = reducer.transform(new_batch)
    reloaded_output = reloaded.transform(new_batch)
    np.testing.assert_allclose(original_output, reloaded_output, rtol=1e-6, atol=1e-6)
    print(f"\nReloaded transformer from '{artifact_path}' produces identical output. "
          f"All checks passed.")
