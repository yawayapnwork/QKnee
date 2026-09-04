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

import copy
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import yaml

BackendEngine = Literal["pytorch", "onnx"]
MultiTargetHeadType = Literal["multi_observable", "ensemble"]
VolumetricAggregation = Literal["mean", "attention", "topk_max"]
DicomBackend = Literal["pydicom", "sitk", "monai"]

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

# Maps a dotted YAML path to the environment variable that may override its
# leaf value. Only leaves that legacy env-var contracts (Docker, docker-compose,
# api.py/app.py's `os.environ.get(...)` calls) already depend on are listed;
# everything else is config-file-only by design.
_ENV_OVERRIDES: Dict[str, str] = {
    "paths.data_root": "DATA_ROOT_PATH",
    "paths.pca_artifact": "PCA_ARTIFACT_PATH",
    "paths.model_checkpoint": "MODEL_CHECKPOINT_PATH",
    "paths.acl_checkpoint": "ACL_CHECKPOINT_PATH",
    "paths.meniscus_checkpoint": "MENISCUS_CHECKPOINT_PATH",
    "paths.rsna_series_dir": "RSNA_SERIES_DIR",
    "api.jwt_secret_key": "QKNEE_JWT_SECRET_KEY",
    # Optional PostgreSQL/Redis connection strings for `qknee.api.auth`'s
    # user repository and `qknee.api.server`'s CacheService, respectively.
    # Both are read with `os.environ.get` (see `_apply_env_overrides`
    # below), so an explicitly-empty `DATABASE_URL=`/`REDIS_URL=` (as set
    # by a single-node cloud runtime with no managed Postgres/Redis
    # attached — Render, Streamlit Cloud, Vercel) still overrides the
    # YAML value to `""`, which both modules treat as "use the local
    # fallback," never as an error.
    "storage.database_url": "DATABASE_URL",
    "storage.redis_url": "REDIS_URL",
}


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing, malformed, or fails validation."""


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path
    pca_artifact: Path
    model_checkpoint: Path
    acl_checkpoint: Path
    meniscus_checkpoint: Path
    eval_output_dir: Path
    deck_output_dir: Path
    checkpoint_dir: Path
    rsna_series_dir: Optional[Path] = None


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
class IngestionConfig:
    # Defaults to "pydicom" directly (not via _build_config) so a bare
    # IngestionConfig()/QKneeConfig's own default_factory below stays valid
    # even for a config.yaml/override dict that omits the whole `ingestion:`
    # section — same optional-section treatment as StorageConfig.
    backend: DicomBackend = "pydicom"  # "pydicom" (default) -> pydicom, per-file; "sitk" -> SimpleITK; "monai" -> monai.transforms.LoadImage — see qknee.data.ingestion.DataIngestion


@dataclass(frozen=True)
class ResNetConfig:
    feature_dim: int
    freeze_backbone: bool
    backend_engine: BackendEngine  # "pytorch" -> ResNet18FeatureExtractor, "onnx" -> ONNXFeatureExtractor
    onnx_path: Path                # used only when backend_engine == "onnx"
    volumetric_aggregation: VolumetricAggregation  # "mean" | "attention" | "topk_max"
    topk_k: int                    # used only when volumetric_aggregation == "topk_max"


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
    multi_target_head: MultiTargetHeadType  # "multi_observable" | "ensemble" — see qknee.models.vqc_multitarget


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    n_epochs: int
    log_every: int
    optimizer: str
    val_holdout_fraction: float
    pca_fit_max_samples: int
    max_train_samples: Optional[int]
    early_stopping_patience: int
    early_stopping_min_delta: float


@dataclass(frozen=True)
class GradCAMConfig:
    alpha: float
    colormap: str


@dataclass(frozen=True)
class EvaluationConfig:
    synthetic_n_samples: int
    test_size: float
    random_seed: int
    classical_max_iter: int
    figure_dpi: int
    deck_figure_dpi: int


@dataclass(frozen=True)
class APIConfig:
    host: str
    port: int
    cors_origins: List[str]
    tear_risk_threshold: float
    # Both default so a config.yaml predating auth.py's addition still loads;
    # jwt_secret_key's shipped default is dev-only (see config.yaml) and is
    # meant to be overridden via $QKNEE_JWT_SECRET_KEY in any real deployment.
    jwt_secret_key: str = "INSECURE-DEV-ONLY-CHANGE-ME-VIA-QKNEE_JWT_SECRET_KEY-ENV-VAR"
    access_token_expire_minutes: int = 60


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str
    datefmt: str


@dataclass(frozen=True)
class StorageConfig:
    """Optional distributed-storage backends for `qknee.api.auth` (user
    repository) and `qknee.api.server` (`CacheService`) — both default to
    a fully local, single-node fallback (JSON-file user store, in-process
    TTL cache) so the API boots and runs identically on a free-tier
    single-node runtime (Render, Streamlit Cloud, Vercel) with neither
    variable set, and transparently upgrades to Postgres/Redis when one
    is attached (e.g. a Docker Compose stack, or a managed Postgres/Redis
    add-on on a paid host).
    """

    database_url: str = ""  # env: DATABASE_URL — e.g. postgresql+asyncpg://... or sqlite:///./users.db; "" = local JSON store
    redis_url: str = ""     # env: REDIS_URL — e.g. redis://default:...@host:6379; "" = in-process TTL cache
    local_users_path: Path = Path("qknee/artifacts/users.json")  # LocalFileUserRepository fallback store
    cache_ttl_seconds: int = 3600  # default TTL for CacheService entries (Redis or in-memory)


@dataclass(frozen=True)
class QKneeConfig:
    paths: PathsConfig
    data: DataConfig
    resnet: ResNetConfig
    pca: PCAConfig
    quantum: QuantumConfig
    training: TrainingConfig
    gradcam: GradCAMConfig
    evaluation: EvaluationConfig
    api: APIConfig
    logging: LoggingConfig
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
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

    # `api.cors_origins` is a List[str], not a scalar, so it can't reuse the
    # plain string-assignment loop above (which would hand the dataclass a
    # bare string where a list is expected). `$QKNEE_CORS_ORIGINS`, when
    # set, is a comma-separated origin list — this lets a multi-cloud
    # deployment (Vercel API + Streamlit Cloud UI + a local dev frontend)
    # widen/narrow the allowed origins per-environment without editing
    # config.yaml or rebuilding the image.
    cors_override = os.environ.get("QKNEE_CORS_ORIGINS")
    if cors_override is not None:
        origins = [origin.strip() for origin in cors_override.split(",") if origin.strip()]
        raw.setdefault("api", {})["cors_origins"] = origins

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
            acl_checkpoint=Path(_require(paths_raw, "acl_checkpoint", "paths")),
            meniscus_checkpoint=Path(_require(paths_raw, "meniscus_checkpoint", "paths")),
            eval_output_dir=Path(_require(paths_raw, "eval_output_dir", "paths")),
            deck_output_dir=Path(_require(paths_raw, "deck_output_dir", "paths")),
            checkpoint_dir=Path(_require(paths_raw, "checkpoint_dir", "paths")),
            rsna_series_dir=Path(paths_raw["rsna_series_dir"]) if paths_raw.get("rsna_series_dir") else None,
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

        resnet_raw = raw["resnet"]
        resnet = ResNetConfig(
            feature_dim=int(resnet_raw["feature_dim"]),
            freeze_backbone=bool(resnet_raw["freeze_backbone"]),
            backend_engine=resnet_raw.get("backend_engine", "pytorch"),
            onnx_path=Path(resnet_raw.get("onnx_path", "qknee/artifacts/resnet18_extractor.onnx")),
            volumetric_aggregation=resnet_raw.get("volumetric_aggregation", "mean"),
            topk_k=int(resnet_raw.get("topk_k", 5)),
        )
        pca_raw = raw["pca"]
        pca = PCAConfig(
            n_components=int(pca_raw["n_components"]),
            use_incremental_pca=bool(pca_raw["use_incremental_pca"]),
            angle_range=tuple(pca_raw["angle_range"]),
        )
        quantum = QuantumConfig(**raw["quantum"])
        training = TrainingConfig(**raw["training"])
        gradcam = GradCAMConfig(**raw["gradcam"])
        evaluation = EvaluationConfig(**raw["evaluation"])
        api = APIConfig(**raw["api"])
        logging_cfg = LoggingConfig(**raw["logging"])

        # Optional section: absent entirely from a config.yaml predating
        # this field (or from a caller-supplied override dict that only
        # touches other sections) still builds a fully-defaulted
        # StorageConfig rather than raising — every one of its fields
        # already defaults to "use the local single-node fallback".
        storage_raw = raw.get("storage") or {}
        storage = StorageConfig(
            database_url=str(storage_raw.get("database_url") or ""),
            redis_url=str(storage_raw.get("redis_url") or ""),
            local_users_path=Path(storage_raw.get("local_users_path", "qknee/artifacts/users.json")),
            cache_ttl_seconds=int(storage_raw.get("cache_ttl_seconds", 3600)),
        )

        # Optional section, same treatment as `storage` above: absent
        # entirely from a config.yaml predating this field still builds a
        # fully-defaulted IngestionConfig (backend="pydicom") rather than
        # raising.
        ingestion_raw = raw.get("ingestion") or {}
        ingestion = IngestionConfig(backend=ingestion_raw.get("backend", "pydicom"))
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Malformed config.yaml section: {exc}") from exc

    if pca.n_components != quantum.n_qubits:
        raise ConfigError(
            f"pca.n_components ({pca.n_components}) must equal "
            f"quantum.n_qubits ({quantum.n_qubits})"
        )
    if resnet.backend_engine not in ("pytorch", "onnx"):
        raise ConfigError(
            f"resnet.backend_engine must be 'pytorch' or 'onnx', got {resnet.backend_engine!r}"
        )
    if ingestion.backend not in ("pydicom", "sitk", "monai"):
        raise ConfigError(
            f"ingestion.backend must be 'pydicom', 'sitk', or 'monai', got {ingestion.backend!r}"
        )
    if resnet.volumetric_aggregation not in ("mean", "attention", "topk_max"):
        raise ConfigError(
            "resnet.volumetric_aggregation must be 'mean', 'attention', or 'topk_max', "
            f"got {resnet.volumetric_aggregation!r}"
        )
    if resnet.topk_k < 1:
        raise ConfigError(f"resnet.topk_k must be >= 1, got {resnet.topk_k}")
    if quantum.multi_target_head not in ("multi_observable", "ensemble"):
        raise ConfigError(
            f"quantum.multi_target_head must be 'multi_observable' or 'ensemble', got {quantum.multi_target_head!r}"
        )

    return QKneeConfig(
        paths=paths,
        data=data,
        resnet=resnet,
        pca=pca,
        quantum=quantum,
        training=training,
        gradcam=gradcam,
        evaluation=evaluation,
        api=api,
        logging=logging_cfg,
        ingestion=ingestion,
        storage=storage,
        device=raw.get("device"),
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Connection-string redaction — shared by qknee.api.auth (DATABASE_URL) and
# qknee.api.server (REDIS_URL) so neither ever logs a plaintext credential.
# --------------------------------------------------------------------------- #

def redact_connection_string(url: str) -> str:
    """Returns `url` with any embedded password replaced by `***`, safe to
    write to logs. Returns `"<unset>"` for an empty/falsy `url`, and
    `"<unparseable>"` (rather than raising) for a string that isn't a
    parseable URL at all.
    """
    if not url:
        return "<unset>"
    try:
        parsed = urlsplit(url)
        if not parsed.password:
            return url
        redacted_netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
        return urlunsplit(parsed._replace(netloc=redacted_netloc))
    except ValueError:
        return "<unparseable>"


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


# --------------------------------------------------------------------------- #
# Dynamic dictionary merging — CLI/programmatic config overrides
# --------------------------------------------------------------------------- #

def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges `overrides` into a deep copy of `base` and returns
    the result — `base` and `overrides` are both left unmodified.

    Nested dicts merge key-by-key (so `overrides = {"training": {"n_epochs": 50}}`
    only touches `training.n_epochs`, leaving every other `training.*` key
    from `base` untouched); any other value in `overrides` (including an
    explicit `None`) replaces the corresponding leaf in `base` outright.
    This is the "dynamic dictionary merging" `load_config_with_overrides`
    below applies CLI-supplied overrides through, so a caller only needs to
    supply the handful of leaves it actually wants to change, shaped like a
    fragment of `config.yaml` itself.

    Args:
        base: The starting nested dict (typically a parsed `config.yaml`).
        overrides: A nested dict of the same shape (or a subset of it)
            whose leaf values should win.

    Returns:
        A new merged dict; `base`/`overrides` are never mutated.
    """
    merged = copy.deepcopy(base)

    def _merge_into(destination: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(destination.get(key), dict):
                _merge_into(destination[key], value)
            else:
                destination[key] = value

    _merge_into(merged, overrides)
    return merged


def load_config_with_overrides(
    overrides: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> QKneeConfig:
    """Loads `config.yaml` (env-var overrides still applied first, same as
    `load_config`), deep-merges `overrides` on top via `deep_merge`, and
    builds a fresh `QKneeConfig` from the result — the full
    `_build_config` validation (e.g. `pca.n_components == quantum.n_qubits`)
    still runs against the merged config, so an override that breaks a
    cross-field invariant fails loudly here rather than silently producing
    an inconsistent config downstream.

    Unlike `load_config`, this is **not** cached — each call can supply
    different overrides and must produce an independent `QKneeConfig` — so
    it's meant for one-off CLI/script entry points (e.g.
    `scripts/train.py`'s `--epochs`/`--batch_size`/`--learning_rate`
    flags) rather than the module-level `_config = load_config()` pattern
    used throughout the rest of the codebase.

    Args:
        overrides: A nested dict shaped like (a subset of) `config.yaml`,
            e.g. `{"training": {"n_epochs": 50, "learning_rate": 0.02},
            "data": {"batch_size": 16}}`. `None`/`{}` behaves exactly like
            `load_config(config_path)`.
        config_path: Optional alternate YAML path; defaults to the shipped
            `config.yaml`.

    Returns:
        A `QKneeConfig` reflecting `config.yaml` + env-var overrides +
        `overrides`, in that precedence order (later wins).
    """
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    raw = _read_yaml(resolved_path)
    raw = _apply_env_overrides(raw)
    if overrides:
        raw = deep_merge(raw, overrides)
    return _build_config(raw)
