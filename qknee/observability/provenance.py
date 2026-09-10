"""Unified prediction-provenance vocabulary, used by `extras/api/server.py`'s
FastAPI/Next.js contract so a prediction's origin is described with exactly
one set of words everywhere it's shown to a viewer.

Fixes AUDIT.md P1 #5 (D2 -- no prominent mock/live indicator on the Next.js
workstation) and P1 #7 (B3/C4b -- a checkpoint-fallback silently downgrades
every score with only a `logger.warning`, and C4c -- the Streamlit
"SIMULATION MODE" badge is ambiguous between "the quantum *simulator* backend
is active, as expected" and "this result is fabricated").

Three independent axes, because they answer three different questions a
viewer might ask:

    `Provenance`       -- "where did this response come from, structurally?"
                          (LIVE / PRECOMPUTED DEMO / MOCK-FALLBACK / CACHED /
                          PROXY) -- see `PROVENANCE_LABELS`.
    `ModelSource`       -- "when a real forward pass ran, whose weights did
                          it use?" (a genuinely trained checkpoint, or the
                          random-initialization fallback from
                          `qknee.models.qknee_model.load_best_checkpoint_or_init`/
                          `qknee.models.pipeline.PipelineRunner`). `None` when
                          no model ran at all (pure mock).
    `QuantumExecution`  -- "did a real quantum-circuit-simulator execution
                          happen for this response?" Deliberately never
                          spelled "SIMULATION MODE" — PennyLane's
                          `default.qubit` *simulator* is this project's
                          always-expected, non-degraded quantum backend (see
                          README.md/RESULTS.md: no physical QPU is claimed
                          anywhere), so "simulator ran" and "this is
                          degraded/fake" must never share one ambiguous label.

`classify()` is the single function `extras/api/server.py` calls to
derive all three from the same raw signals
(a `backend` tag string, whether a trained checkpoint was actually loaded,
and whether real per-qubit measurements exist) — see its docstring for the
exact precedence rules, most importantly the AUDIT.md C4b fix: a real
forward pass through **untrained, randomly-initialized** weights is
reported as `MOCK_FALLBACK`/`RANDOM_FALLBACK`, never silently reported as
`LIVE` just because a real tensor pipeline happened to execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Provenance = Literal["live", "precomputed_demo", "mock_fallback", "cached", "proxy"]
ModelSource = Literal["trained_checkpoint", "random_fallback"]
QuantumExecution = Literal["quantum_simulator", "unavailable"]

# Display strings — the ONLY place any UI surface should get these words
# from. Never hardcode "LIVE"/"MOCK"/etc. as a literal string anywhere else;
# import these dicts (or `ProvenanceInfo`'s `*_label` properties) instead, so
# Next.js and Streamlit can never drift into different terminology for the
# same underlying state.
PROVENANCE_LABELS: dict[Provenance, str] = {
    "live": "LIVE",
    "precomputed_demo": "PRECOMPUTED DEMO",
    "mock_fallback": "MOCK/FALLBACK",
    "cached": "CACHED",
    "proxy": "PROXY",
}

MODEL_SOURCE_LABELS: dict[ModelSource, str] = {
    "trained_checkpoint": "TRAINED MODEL",
    "random_fallback": "MODEL FALLBACK (untrained weights)",
}

QUANTUM_EXECUTION_LABELS: dict[QuantumExecution, str] = {
    "quantum_simulator": "QUANTUM SIMULATOR",
    "unavailable": "QUANTUM TELEMETRY UNAVAILABLE",
}

# Provenance values a UI must render as visually unmistakable (loud
# amber/red styling, never blended in alongside a routine "LIVE" badge) —
# AUDIT.md P1 #5 requirement 5 ("Make MOCK/FALLBACK visually unmistakable").
UNMISTAKABLE_PROVENANCE: frozenset[Provenance] = frozenset({"mock_fallback"})


@dataclass(frozen=True)
class ProvenanceInfo:
    """Everything a UI needs to render one prediction's provenance badge —
    computed once (via `classify()`) and threaded through unchanged from
    backend to API response to frontend/Streamlit render call, so the label
    a viewer sees always matches the actual execution path, never a stale
    or re-derived guess."""

    provenance: Provenance
    model_source: Optional[ModelSource]
    quantum_execution: QuantumExecution

    @property
    def provenance_label(self) -> str:
        return PROVENANCE_LABELS[self.provenance]

    @property
    def model_source_label(self) -> Optional[str]:
        return MODEL_SOURCE_LABELS[self.model_source] if self.model_source else None

    @property
    def quantum_execution_label(self) -> str:
        return QUANTUM_EXECUTION_LABELS[self.quantum_execution]

    @property
    def is_trustworthy(self) -> bool:
        """False whenever this response must never be presented as an
        ordinary successful "LIVE" result — either no real inference ran
        (`mock_fallback`) or its provenance can't be personally vouched for
        by this process (`proxy`, when the upstream response predates this
        schema). Used by callers to decide whether a result may replace an
        existing on-screen result silently vs. needing an explicit,
        visible transition (AUDIT.md P1 #7 requirement 6: never silently
        transition live -> mock)."""
        return self.provenance not in ("mock_fallback",)

    def as_dict(self) -> dict:
        return {
            "provenance": self.provenance,
            "provenance_label": self.provenance_label,
            "model_source": self.model_source,
            "model_source_label": self.model_source_label,
            "quantum_execution": self.quantum_execution,
            "quantum_execution_label": self.quantum_execution_label,
        }


def classify(
    *,
    backend_tag: str,
    model_checkpoint_loaded: Optional[bool],
    quantum_expectations_present: bool,
) -> ProvenanceInfo:
    """Derives a `ProvenanceInfo` from the raw signals every backend
    implementation already has lying around:

        backend_tag: a short tag identifying which code path produced this
            response -- `"live"` (a real `PipelineRunner.run()` call),
            `"mock"` (no model ran at all -- a seeded pseudo-random
            generator stood in), a `"cache-fallback/<case_id>"` string (a
            precomputed/demo case replayed verbatim), `"cached/..."` /
            `"cached-fastpath/..."` (Streamlit's own precomputed-cache
            sidebar paths), or `"api/..."` (Streamlit calling out to the
            FastAPI backend over HTTP -- treated as `proxy` here since this
            process didn't run inference itself). Anything else
            unrecognized is conservatively classified `mock_fallback`
            rather than risk mislabeling an unknown path as `live`.
        model_checkpoint_loaded: `True` if a real trained checkpoint's
            weights were loaded before this forward pass ran, `False` if
            the model ran on randomly-initialized weights instead (a
            missing/invalid checkpoint), `None` if no model forward pass
            happened for this response at all (pure `mock`/precomputed
            paths).
        quantum_expectations_present: whether real per-qubit Pauli-Z
            measurements exist for this response (i.e. a quantum circuit
            simulator actually executed, live or precomputed).

    The one precedence rule that fixes AUDIT.md C4b: a `"live"` backend_tag
    with `model_checkpoint_loaded=False` is reported as `mock_fallback`
    provenance (with `model_source="random_fallback"`), never as `live` --
    a real tensor pipeline executing on meaningless random weights is not a
    trustworthy result and must not wear the same badge as one that is.
    """
    if backend_tag == "live":
        if model_checkpoint_loaded is False:
            provenance: Provenance = "mock_fallback"
        else:
            provenance = "live"
    elif backend_tag == "mock":
        provenance = "mock_fallback"
    elif backend_tag.startswith("cache-fallback/") or backend_tag.startswith("cached") or backend_tag.startswith("cached-fastpath"):
        provenance = "precomputed_demo"
    elif backend_tag.startswith("api/"):
        provenance = "proxy"
    elif backend_tag == "proxy":
        provenance = "proxy"
    else:
        provenance = "mock_fallback"

    model_source: Optional[ModelSource]
    if model_checkpoint_loaded is None:
        model_source = None
    elif model_checkpoint_loaded:
        model_source = "trained_checkpoint"
    else:
        model_source = "random_fallback"

    quantum_execution: QuantumExecution = "quantum_simulator" if quantum_expectations_present else "unavailable"

    return ProvenanceInfo(provenance=provenance, model_source=model_source, quantum_execution=quantum_execution)
