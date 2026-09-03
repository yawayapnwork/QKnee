"""
Pre-deployment sanity check for the Q-Knee Streamlit app (Streamlit
Community Cloud / Hugging Face Spaces).

Runs a sequential set of deployment gates — required-file presence,
artifact integrity, a real inference pass through `QKneePipeline`, a real
PDF-generation pass through `qknee.xai.report_generator`, Streamlit UI
path-resolution, and a repeated-inference memory-stability smoke test —
and prints a structured pass/fail report. Exits 0 only if every gate
passes, non-zero otherwise, so it can be dropped straight into a CI job
or a pre-push hook.

The FastAPI-specific gates (import/route-mount check, subprocess
`uvicorn` cold-boot timing) were quarantined to `extras/` along with
`qknee/api/` — see `extras/README.md`.

Usage:
    python scripts/verify_deployment.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Windows terminals commonly default stdout to a legacy codepage (e.g.
# cp1252) that can't encode the pass/fail markers below; reconfigure to
# UTF-8 with a replace fallback so the report never crashes mid-print
# instead of just showing a slightly uglier glyph.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARTIFACTS_DIR = ROOT_DIR / "qknee" / "artifacts"
ASSET_DIRS = [
    ARTIFACTS_DIR / "deck_figures",
    ARTIFACTS_DIR / "demo_cache",
    ARTIFACTS_DIR / "heatmaps",
]


class GateResult(NamedTuple):
    name: str
    passed: bool
    detail: str
    duration_s: float


def _run_gate(name: str, fn: Callable[[], str]) -> GateResult:
    """Runs one gate function, capturing its outcome/timing instead of
    letting an unexpected exception kill the whole sanity check —
    everything downstream still runs, and the failure shows up as one
    line in the final report rather than a raw traceback."""
    start = time.perf_counter()
    try:
        detail = fn()
        duration = time.perf_counter() - start
        return GateResult(name=name, passed=True, detail=detail, duration_s=duration)
    except Exception as exc:  # noqa: BLE001 - any gate failure is reported, not raised
        duration = time.perf_counter() - start
        detail = f"{type(exc).__name__}: {exc}"
        if _VERBOSE:
            detail += "\n" + textwrap_indent(traceback.format_exc())
        return GateResult(name=name, passed=False, detail=detail, duration_s=duration)


def textwrap_indent(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


_VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


# --------------------------------------------------------------------------- #
# Gate 1: required root deployment files
# --------------------------------------------------------------------------- #

def check_required_files() -> str:
    """`app.py` is the conventional Streamlit Cloud entry-point name; this
    repo instead ships `streamlit_app.py` (see README frontmatter's
    `app_file: streamlit_app.py`) as the repo-root wrapper that adds the
    repo root to `sys.path` before delegating to `qknee.ui.dashboard.main`.
    Either name satisfies this gate; a fresh `app.py` is preferred if
    present, since that's the name this gate was written to expect."""
    missing: List[str] = []

    entry_point = None
    for candidate in ("app.py", "streamlit_app.py"):
        path = ROOT_DIR / candidate
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            entry_point = candidate
            break
    if entry_point is None:
        missing.append("app.py (or streamlit_app.py)")

    for filename in ("packages.txt", "requirements.txt"):
        path = ROOT_DIR / filename
        if not path.exists():
            missing.append(filename)
            continue
        if path.stat().st_size == 0:
            missing.append(f"{filename} (empty)")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            missing.append(f"{filename} (not valid UTF-8: {exc})")
            continue
        non_comment_lines = [
            line for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not non_comment_lines:
            missing.append(f"{filename} (no non-comment entries)")

    if missing:
        raise AssertionError(f"missing/invalid: {', '.join(missing)}")

    return f"entry point: {entry_point}; packages.txt and requirements.txt present and non-empty"


# --------------------------------------------------------------------------- #
# Gate 2: artifact directories
# --------------------------------------------------------------------------- #

def check_artifact_assets() -> str:
    """Every file directly under each of the three artifact directories
    must exist, be non-empty, and be readable (open + read one byte) —
    catches a truncated/corrupt asset that `.exists()` alone would miss."""
    total_files = 0
    checked_dirs = []
    for directory in ASSET_DIRS:
        if not directory.is_dir():
            raise AssertionError(f"missing directory: {directory.relative_to(ROOT_DIR)}")

        files = sorted(p for p in directory.iterdir() if p.is_file())
        if not files:
            raise AssertionError(f"no assets found in {directory.relative_to(ROOT_DIR)}")

        for file_path in files:
            if file_path.stat().st_size == 0:
                raise AssertionError(f"empty asset: {file_path.relative_to(ROOT_DIR)}")
            try:
                with open(file_path, "rb") as handle:
                    handle.read(1)
            except OSError as exc:
                raise AssertionError(f"unreadable asset {file_path.relative_to(ROOT_DIR)}: {exc}") from exc

        total_files += len(files)
        checked_dirs.append(f"{directory.name} ({len(files)})")

    return f"{total_files} asset(s) verified across {', '.join(checked_dirs)}"


# --------------------------------------------------------------------------- #
# Gate 3: QKneePipeline inference pass on synthetic tensors
# --------------------------------------------------------------------------- #

def check_pipeline_inference() -> str:
    """Constructs a real `QKneePipeline` (fitting a throwaway PCA artifact
    first if none is committed, exactly like `pipeline.py`'s own
    `__main__` smoke test) and runs it end-to-end on a synthetic random
    slice — this is a real forward pass through ResNet18 -> PCA -> VQC ->
    GradCAM, not a mock, so it catches a broken checkpoint/config/
    dependency before it reaches production."""
    import numpy as np

    from qknee.config.loader import load_config
    from qknee.models.pca_reducer import QuantumDimReducer
    from qknee.models.pipeline import QKneePipeline

    config = load_config()
    artifact_path = Path(config.paths.pca_artifact)
    fitted_dummy_artifact = False
    if not artifact_path.exists():
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        dummy_512d = rng.standard_normal((500, config.resnet.feature_dim)).astype(np.float32)
        QuantumDimReducer().fit(dummy_512d).save(artifact_path)
        fitted_dummy_artifact = True

    pipeline = QKneePipeline(config=config, pca_artifact_path=artifact_path)

    rng = np.random.default_rng(42)
    synthetic_slice = rng.integers(0, 256, size=(224, 224), dtype=np.uint8)

    result = pipeline.predict_volume(synthetic_slice, skip_gradcam=False)

    if not (0.0 <= result.risk_score <= 1.0):
        raise AssertionError(f"risk_score {result.risk_score} outside [0, 1]")
    if result.quantum_angles.shape != (1, config.quantum.n_qubits):
        raise AssertionError(
            f"quantum_angles shape {result.quantum_angles.shape} != (1, {config.quantum.n_qubits})"
        )
    if result.gradcam_heatmap is None or result.gradcam_heatmap.ndim != 2:
        raise AssertionError("gradcam_heatmap missing or not 2D")

    note = " (fitted a throwaway PCA artifact — none was committed)" if fitted_dummy_artifact else ""
    return f"risk_score={result.risk_score:.4f}, quantum_angles={result.quantum_angles.shape}{note}"


# --------------------------------------------------------------------------- #
# Gate 4: PDF report generation pass
# --------------------------------------------------------------------------- #

def check_pdf_generation() -> str:
    """Runs `generate_radiology_report` end-to-end on synthetic slice/
    heatmap arrays and a representative prediction payload, and checks the
    returned bytes are a well-formed PDF (correct magic bytes, non-trivial
    size)."""
    import numpy as np

    from qknee.xai.report_generator import generate_radiology_report

    rng = np.random.default_rng(7)
    mri_slice = rng.integers(0, 256, size=(224, 224), dtype=np.uint8)
    gradcam_overlay = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    pdf_bytes = generate_radiology_report(
        output_path=None,
        mri_slice=mri_slice,
        gradcam_overlay=gradcam_overlay,
        prediction_results={
            "acl_risk": 0.72,
            "mcl_risk": 0.18,
            "meniscus_risk": 0.41,
            "pauli_z_expectations": [0.3, -0.5, 0.1, 0.8],
            "resnet_latency_ms": 24.1,
            "pca_latency_ms": 1.2,
            "quantum_latency_ms": 6.4,
            "total_latency_ms": 31.7,
            "backend": "verify_deployment",
        },
        metadata={
            "modality": "MRI Knee",
            "clinical_indication": "Deployment sanity check",
            "scan_date": "1970-01-01",
        },
    )

    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise AssertionError(f"expected bytes, got {type(pdf_bytes)}")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise AssertionError("output does not start with the PDF magic bytes ('%PDF-')")
    if len(pdf_bytes) < 1024:
        raise AssertionError(f"suspiciously small PDF output ({len(pdf_bytes)} bytes)")

    return f"generated {len(pdf_bytes):,} byte PDF"


# --------------------------------------------------------------------------- #
# Gate 5: Streamlit UI modules resolve their asset paths absolutely,
# independent of the process's current working directory.
# --------------------------------------------------------------------------- #

def check_ui_path_resolution() -> str:
    """Imports `qknee.ui.dashboard`/`analysis_app` with the process's CWD
    deliberately pointed somewhere other than the repo root (a scratch
    temp dir), and asserts each module's own `_REPO_ROOT`/`_ARTIFACTS_DIR`
    constants still resolve to the real repo root and a real, existing
    `qknee/artifacts` directory. `streamlit run` (and every cloud
    runtime's various ways of invoking it) doesn't guarantee CWD is the
    repo root, so a bare relative `Path("qknee/artifacts/...")` would
    silently resolve to nothing in exactly the deployment scenario this
    gate reproduces; these modules instead anchor on
    `Path(__file__).resolve().parents[2]`, which this gate confirms
    actually holds.
    """
    import importlib
    import os
    import sys as _sys
    import tempfile

    module_names = ["qknee.ui.dashboard", "qknee.ui.analysis_app"]
    original_cwd = os.getcwd()
    checked = []
    scratch = tempfile.TemporaryDirectory()
    try:
        os.chdir(scratch.name)
        for module_name in module_names:
            _sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)

            repo_root = getattr(module, "_REPO_ROOT", None)
            artifacts_dir = getattr(module, "_ARTIFACTS_DIR", None)
            if repo_root is None or artifacts_dir is None:
                raise AssertionError(f"{module_name} has no _REPO_ROOT/_ARTIFACTS_DIR constant")
            if repo_root.resolve() != ROOT_DIR.resolve():
                raise AssertionError(f"{module_name}._REPO_ROOT={repo_root} != actual repo root {ROOT_DIR}")
            if not artifacts_dir.is_dir():
                raise AssertionError(f"{module_name}._ARTIFACTS_DIR={artifacts_dir} does not exist")
            precomputed_cache = getattr(module, "PRECOMPUTED_CACHE_PATH", None)
            if precomputed_cache is not None and not precomputed_cache.is_absolute():
                raise AssertionError(f"{module_name}.PRECOMPUTED_CACHE_PATH={precomputed_cache} is not absolute")
            checked.append(module_name.rsplit(".", 1)[-1])
    finally:
        # Restore CWD *before* the scratch dir is cleaned up — on Windows,
        # deleting a directory that's still the process's current working
        # directory raises `PermissionError: [WinError 32]`, since the OS
        # holds an implicit open handle to it.
        os.chdir(original_cwd)
        scratch.cleanup()
        for module_name in module_names:
            _sys.modules.pop(module_name, None)

    return f"CWD-independent path resolution verified for: {', '.join(checked)}"


# --------------------------------------------------------------------------- #
# Gate 6: repeated-inference memory stability (leak smoke test)
# --------------------------------------------------------------------------- #

def check_inference_memory_stability() -> str:
    """Runs `QKneePipeline.predict_volume()` several times in a row on one
    long-lived pipeline instance (the same reuse pattern
    `qknee.api.server`'s lazily-built singleton follows across requests),
    tracking Python-heap growth via the stdlib, cross-platform
    `tracemalloc` module between iterations (`gc.collect()`-ing after
    each) — a portable smoke check that `PipelineRunner.run()`'s
    `_release_memory()` hook is actually keeping per-request
    tensors/activation graphs from accumulating across a long-running
    process, not a byte-exact memory budget.
    """
    import gc
    import tracemalloc

    import numpy as np

    from qknee.config.loader import load_config
    from qknee.models.pipeline import QKneePipeline

    config = load_config()
    pipeline = QKneePipeline(config=config, pca_artifact_path=config.paths.pca_artifact)

    n_iterations = 8
    rng = np.random.default_rng(99)
    tracemalloc.start()
    try:
        snapshots_kb: List[float] = []
        for _ in range(n_iterations):
            synthetic_slice = rng.integers(0, 256, size=(224, 224), dtype=np.uint8)
            result = pipeline.predict_volume(synthetic_slice, skip_gradcam=False)
            if not (0.0 <= result.risk_score <= 1.0):
                raise AssertionError(f"risk_score {result.risk_score} outside [0, 1]")
            del result
            gc.collect()
            current, _ = tracemalloc.get_traced_memory()
            snapshots_kb.append(current / 1024)
    finally:
        tracemalloc.stop()

    # Ignore the first couple of iterations (legitimate one-time lazy
    # allocations — e.g. PennyLane device setup, first-call buffer
    # sizing) and compare the FINAL pass against the steady-state
    # baseline established by the middle iterations, rather than a raw
    # first-vs-last diff that would flag normal warm-up growth.
    steady_state = snapshots_kb[2:-1] or snapshots_kb[:-1] or snapshots_kb
    baseline_kb = sum(steady_state) / len(steady_state)
    final_kb = snapshots_kb[-1]
    growth_kb = final_kb - baseline_kb
    growth_pct = (growth_kb / baseline_kb * 100) if baseline_kb else 0.0

    # A generous threshold — this is a smoke check for gross leaks (an
    # un-released tensor/activation graph accumulating every call), not a
    # tight budget; ordinary allocator noise run-to-run is expected.
    if growth_kb > 0 and growth_pct > 25.0:
        raise AssertionError(
            f"heap grew {growth_pct:.1f}% ({growth_kb:.0f}KB) from the {baseline_kb:.0f}KB "
            f"steady-state baseline to the final of {n_iterations} inference passes — "
            f"possible memory leak. Per-iteration traced heap (KB): {[f'{v:.0f}' for v in snapshots_kb]}"
        )

    return (
        f"{n_iterations} inference passes, traced heap stable "
        f"(baseline {baseline_kb:.0f}KB -> final {final_kb:.0f}KB, {growth_pct:+.1f}%)"
    )


# --------------------------------------------------------------------------- #
# Report + orchestration
# --------------------------------------------------------------------------- #

GATES: List[tuple] = [
    ("Required root deployment files", check_required_files),
    ("Artifact directory integrity", check_artifact_assets),
    ("QKneePipeline inference pass", check_pipeline_inference),
    ("PDF report generation", check_pdf_generation),
    ("Streamlit UI path resolution (CWD-independent)", check_ui_path_resolution),
    ("Repeated-inference memory stability", check_inference_memory_stability),
]


def _print_header() -> None:
    print("=" * 72)
    print("Q-Knee Pre-Deployment Verification")
    print("=" * 72)


def _print_result(index: int, total: int, result: GateResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    marker = "✓" if result.passed else "✗"
    print(f"[{index}/{total}] {marker} {status}  {result.name}  ({result.duration_s:.2f}s)")
    detail_prefix = "      -> " if result.passed else "      !! "
    for line in result.detail.splitlines() or [""]:
        print(f"{detail_prefix}{line}")
        detail_prefix = "      "


def main() -> int:
    _print_header()

    results: List[GateResult] = []
    for index, (name, fn) in enumerate(GATES, start=1):
        result = _run_gate(name, fn)
        results.append(result)
        _print_result(index, len(GATES), result)
        print()

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print("-" * 72)
    print(f"Summary: {passed}/{len(results)} gate(s) passed, {failed} failed")
    print("-" * 72)

    if failed:
        print("DEPLOYMENT GATE: FAIL")
        for result in results:
            if not result.passed:
                print(f"  - {result.name}: {result.detail.splitlines()[0]}")
        return 1

    print("DEPLOYMENT GATE: PASS - all checks green, safe to deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
