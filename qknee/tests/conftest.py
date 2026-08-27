"""
Shared pytest fixtures for the Q-Knee test suite.

Session-scoped fixtures that construct real ResNet18/PennyLane objects are
kept expensive-once: the ResNet18 backbone and PennyLane QNode are built a
single time per test session and reused across tests, since re-instantiating
either is slow (model download / QNode compilation) relative to the actual
test assertions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from qknee.models.qknee_model import QKneeModel
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.resnet_extractor import ResNet18FeatureExtractor
from qknee.models.vqc import VQCClassifier

RESNET_FEATURE_DIM = 512
N_QUBITS = 4


@pytest.fixture(scope="session")
def resnet_extractor() -> ResNet18FeatureExtractor:
    torch.manual_seed(0)
    model = ResNet18FeatureExtractor(freeze_backbone=True)
    model.eval()
    return model


@pytest.fixture(scope="session")
def fitted_reducer() -> QuantumDimReducer:
    """A QuantumDimReducer fit once on a fixed synthetic 512-D corpus, shared
    read-only across tests (fitting involves an SVD and is not free)."""
    rng = np.random.default_rng(42)
    dummy_features = rng.normal(size=(300, RESNET_FEATURE_DIM)).astype(np.float32)
    return QuantumDimReducer().fit(dummy_features)


@pytest.fixture(scope="session")
def qknee_model(fitted_reducer: QuantumDimReducer) -> QKneeModel:
    torch.manual_seed(0)
    model = QKneeModel(pca_reducer=fitted_reducer, n_qubits=N_QUBITS, n_layers=3)
    model.eval()
    return model


@pytest.fixture
def dummy_image_batch() -> torch.Tensor:
    """Deterministic (B, 3, 224, 224) batch — fixed seed so pixel content
    (and therefore any downstream numeric assertions) is reproducible."""
    generator = torch.Generator().manual_seed(123)
    return torch.rand(4, 3, 224, 224, generator=generator)


@pytest.fixture
def dummy_resnet_features() -> np.ndarray:
    """Deterministic (N, 512) array standing in for real ResNet18 embeddings."""
    rng = np.random.default_rng(7)
    return rng.normal(size=(64, RESNET_FEATURE_DIM)).astype(np.float32)


@pytest.fixture
def dummy_slice_2d() -> np.ndarray:
    """Deterministic (224, 224) uint8 array standing in for one MRI slice."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 255, size=(224, 224), dtype=np.uint8)


@pytest.fixture(scope="session")
def pca_artifact_path(tmp_path_factory: pytest.TempPathFactory, fitted_reducer: QuantumDimReducer) -> Path:
    """Persists the shared `fitted_reducer` to a session-scoped temp file, so
    `PipelineRunner`/API tests can point at a real, valid PCA artifact
    without re-fitting one per test."""
    path = tmp_path_factory.mktemp("artifacts") / "pca_scaler.pkl"
    fitted_reducer.save(path)
    return path


@pytest.fixture(scope="session")
def missing_checkpoint_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A path that is guaranteed not to exist, so tests that build a
    `PipelineRunner`/`QKneeBackend` get deterministic randomly-initialized
    VQC weights instead of accidentally picking up a real checkpoint left
    on disk at the default `config.yaml` location."""
    return tmp_path_factory.mktemp("no-checkpoint") / "qknee_model.pt"


@pytest.fixture(scope="session")
def pipeline_runner(pca_artifact_path: Path, missing_checkpoint_path: Path):
    """A real `PipelineRunner` (ResNet18 + fitted PCA + randomly-initialized
    VQC), built once per session — constructing it loads a full ResNet18
    backbone and compiles a PennyLane QNode, which isn't free."""
    from qknee.models.pipeline import PipelineRunner

    torch.manual_seed(0)
    return PipelineRunner(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=missing_checkpoint_path)
