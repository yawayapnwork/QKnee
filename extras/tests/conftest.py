"""
`extras/api/server.py` (and every test in this directory) still imports
`qknee.api.auth`/`qknee.api.server` -- the module path this code lived at
*before* the `qknee/api -> extras/api` quarantine move documented in
`extras/README.md` and diagnosed in `AUDIT.md` §A1/A2/F1/F2. That rename
was never propagated into the import statements themselves, so
`extras/api/server.py` -- and therefore this entire test directory -- has
been unimportable (`ModuleNotFoundError: No module named 'qknee.api'')
since that move, independent of anything in this change.

Renaming the module path for real (updating every deployment config,
docstring, and the package layout itself) is a separate, larger decision
than the fix this test directory needs (AUDIT.md P1 #10, not this task) --
so this conftest does the minimal, test-only thing instead: it registers
`extras.api`/`extras.api.auth`/`extras.api.server` under the legacy
`qknee.api.*` names in `sys.modules` *before* anything in this directory
imports them, so `from qknee.api.auth import ...` resolves to the real
`extras/api/auth.py` without touching any production file. This changes
nothing about what actually runs -- `qknee.api.auth` and `extras.api.auth`
end up being the exact same module object.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# `extras.api.auth` (imported below) resolves its JWT signing secret at
# MODULE IMPORT TIME and refuses to import at all without a valid one by
# default (AUDIT.md P1 #8: no committed/default secret exists anymore --
# see `qknee.api.auth.resolve_jwt_secret`). This test suite explicitly
# configures its own throwaway-but-strong secret before that import, the
# same way a real deployment must configure a real one -- it does NOT rely
# on the local-dev insecure-secret opt-in, so these tests also exercise the
# "valid configured secret" path on every run.
os.environ.setdefault("QKNEE_ENV", "test")
os.environ.setdefault("QKNEE_JWT_SECRET_KEY", "test-suite-only-secret-" + "b" * 40)

import extras.api.auth as _extras_auth

_qknee_api_pkg = sys.modules.get("qknee.api") or types.ModuleType("qknee.api")
_qknee_api_pkg.auth = _extras_auth
sys.modules["qknee.api"] = _qknee_api_pkg
sys.modules["qknee.api.auth"] = _extras_auth

import qknee  # noqa: E402 -- must come after the sys.modules setup above

qknee.api = _qknee_api_pkg

import extras.api.server as _extras_server  # noqa: E402 -- needs qknee.api.auth registered first

_qknee_api_pkg.server = _extras_server
sys.modules["qknee.api.server"] = _extras_server


# --------------------------------------------------------------------------- #
# Fixtures `test_api_server.py`/`test_auth.py`/etc. expect (`pca_artifact_path`,
# `missing_checkpoint_path`) -- mirrors `qknee/tests/conftest.py`'s fixtures
# of the same name (fixtures don't cross from a sibling directory's own
# conftest.py, so this directory needs its own copy rather than importing
# qknee/tests', which also isn't a package).
# --------------------------------------------------------------------------- #

RESNET_FEATURE_DIM = 512


@pytest.fixture(scope="session")
def pca_artifact_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from qknee.models.pca_reducer import QuantumDimReducer

    rng = np.random.default_rng(42)
    dummy_features = rng.normal(size=(300, RESNET_FEATURE_DIM)).astype(np.float32)
    reducer = QuantumDimReducer().fit(dummy_features)

    path = tmp_path_factory.mktemp("artifacts") / "pca_scaler.pkl"
    reducer.save(path)
    return path


@pytest.fixture(scope="session")
def missing_checkpoint_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("no-checkpoint") / "qknee_model.pt"


@pytest.fixture
def dummy_slice_2d() -> np.ndarray:
    """Deterministic (224, 224) uint8 array standing in for one MRI slice."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 255, size=(224, 224), dtype=np.uint8)
