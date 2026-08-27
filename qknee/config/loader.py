"""
Dynamic configuration parser for Q-Knee.

Loads `qknee/config/config.yaml` into a typed, dot-accessible `QKneeConfig`
tree of dataclasses. "Dynamic" in the sense that:

    - Any leaf value can be overridden by an environment variable, without
      changing code: the dotted YAML path `paths.pca_artifact` maps to
      `$PCA_ARTIFACT_PATH` (last path segment, upper-cased). This is how the
      Docker deployment repoints the API/UI containers at a bind-mounted
      checkpoint (see docker-compose.yml) without rebuilding the image.
    - An alternate YAML file can be supplied at call time (`load_config(path)`)
      for tests/experiments without touching the shipped default.

Usage:
    from qknee.config.loader import load_config

    config = load_config()
    print(config.quantum.n_qubits)      # 4
    print(config.paths.pca_artifact)    # Path(...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

# Maps a dotted YAML path to the environment variable that may override its
# leaf value. Only leaves that legacy env-var contracts (Docker, docker-compose,
# api.py/app.py's `os.environ.get(...)` calls) already depend on are listed;
# everything else is config-file-only by design.
_ENV_OVERRIDES: Dict[str, str] = {
    "paths.data_root": "DATA_ROOT_PATH",
    "paths.pca_artifact": "PCA_ARTIFACT_PATH",
    "paths.model_checkpoint": "MODEL_CHECKPOINT_PATH",
}


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing, malformed, or fails validation."""


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path
    pca_artifact: Path
    model_checkpoint: Path
    eval_output_dir: Path
    deck_output_dir: Path


@dataclass(frozen=True)
class AugmentationConfig:
    random_rotation_degrees: float
    horizontal_flip_prob: float
    gaussian_noise_std: float


@dataclass(frozen=True)
class DataConfig:
    image_size: Tuple[int, int]
    imagenet_mean: List[float]
    imagenet_std: List[float]
    image_extensions: List[str]
    volume_extensions: List[str]
    batch_size: int
    num_workers: int
    train_augmentation: AugmentationConfig


@dataclass(frozen=True)
class ResNetConfig:
    feature_dim: int
    freeze_backbone: bool


@dataclass(frozen=True)
class PCAConfig:
    n_components: int
    use_incremental_pca: bool
    angle_range: Tuple[float, float]


@dataclass(frozen=True)
class QuantumConfig:
    n_qubits: int
    n_layers: int
    device: str
    diff_method: str


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    n_epochs: int
    log_every: int
    optimizer: str
    val_holdout_fraction: float
    pca_fit_max_samples: int
    max_train_samples: Optional[int]


@dataclass(frozen=True)
class GradCAMConfig:
    alpha: float
    colormap: str


@dataclass(frozen=True)
class APIConfig:
    host: str
    port: int
    cors_origins: List[str]
    tear_risk_threshold: float


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str
    datefmt: str


@dataclass(frozen=True)
class QKneeConfig:
    paths: PathsConfig
    data: DataConfig
    resnet: ResNetConfig
    pca: PCAConfig
    quantum: QuantumConfig
    training: TrainingConfig
    gradcam: GradCAMConfig
    api: APIConfig
    logging: LoggingConfig
    device: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Mutates and returns `raw` with any `_ENV_OVERRIDES` env vars applied."""
    for dotted_path, env_var in _ENV_OVERRIDES.items():
        override = os.environ.get(env_var)
        if override is None:
            continue

        *parents, leaf = dotted_path.split(".")
        node = raw
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = override

    return raw


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config {path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigError(f"Expected a top-level mapping in {path}, got {type(parsed)}")
    return parsed


def _require(section: Dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required key '{section_name}.{key}' in config.yaml")
    return section[key]


def _build_config(raw: Dict[str, Any]) -> QKneeConfig:
    try:
        paths_raw = raw["paths"]
        paths = PathsConfig(
            data_root=Path(_require(paths_raw, "data_root", "paths")),
            pca_artifact=Path(_require(paths_raw, "pca_artifact", "paths")),
            model_checkpoint=Path(_require(paths_raw, "model_checkpoint", "paths")),
            eval_output_dir=Path(_require(paths_raw, "eval_output_dir", "paths")),
            deck_output_dir=Path(_require(paths_raw, "deck_output_dir", "paths")),
        )

        data_raw = raw["data"]
        augmentation = AugmentationConfig(**data_raw["train_augmentation"])
        data = DataConfig(
            image_size=tuple(data_raw["image_size"]),
            imagenet_mean=list(data_raw["imagenet_mean"]),
            imagenet_std=list(data_raw["imagenet_std"]),
            image_extensions=list(data_raw["image_extensions"]),
            volume_extensions=list(data_raw["volume_extensions"]),
            batch_size=int(data_raw["batch_size"]),
            num_workers=int(data_raw["num_workers"]),
            train_augmentation=augmentation,
        )

        resnet = ResNetConfig(**raw["resnet"])
        pca_raw = raw["pca"]
        pca = PCAConfig(
            n_components=int(pca_raw["n_components"]),
            use_incremental_pca=bool(pca_raw["use_incremental_pca"]),
            angle_range=tuple(pca_raw["angle_range"]),
        )
        quantum = QuantumConfig(**raw["quantum"])
        training = TrainingConfig(**raw["training"])
        gradcam = GradCAMConfig(**raw["gradcam"])
        api = APIConfig(**raw["api"])
        logging_cfg = LoggingConfig(**raw["logging"])
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Malformed config.yaml section: {exc}") from exc

    if pca.n_components != quantum.n_qubits:
        raise ConfigError(
            f"pca.n_components ({pca.n_components}) must equal "
            f"quantum.n_qubits ({quantum.n_qubits})"
        )

    return QKneeConfig(
        paths=paths,
        data=data,
        resnet=resnet,
        pca=pca,
        quantum=quantum,
        training=training,
        gradcam=gradcam,
        api=api,
        logging=logging_cfg,
        device=raw.get("device"),
        raw=raw,
    )


@lru_cache(maxsize=None)
def _load_cached(config_path: str) -> QKneeConfig:
    raw = _read_yaml(Path(config_path))
    raw = _apply_env_overrides(raw)
    return _build_config(raw)


def load_config(config_path: Optional[Path] = None) -> QKneeConfig:
    """Loads and validates `config.yaml` (or an alternate path) into a typed
    `QKneeConfig`. Results are cached per resolved path, so repeated calls
    across modules are cheap and share one instance.
    """
    resolved = str((config_path or DEFAULT_CONFIG_PATH).resolve())
    return _load_cached(resolved)
