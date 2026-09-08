# Q-Knee — Architecture

This document is the single authoritative description of what actually runs
in this repository today, and which of its two user-facing interfaces is
the real production/judged deliverable. It exists because AUDIT.md
(finding H1) identified a real contradiction: `frontend/` (a Next.js app
that calls a FastAPI backend) exists, is functional, and is fully wired up
— but neither `README.md` nor `extras/README.md` mentioned it, while both
described the FastAPI backend it depends on as quarantined/out of judged
scope. This file resolves that contradiction with one answer, backed by
repository evidence, and every other doc (`README.md`, `extras/README.md`)
now points here rather than re-describing it independently.

## The one-line answer

**The Streamlit dashboard is the primary, judged, production interface.
The Next.js frontend + FastAPI backend is a secondary, exploratory
interface, quarantined out of judged scope, that talks to the exact same
ML pipeline over HTTP instead of in-process.** Both are documented below;
neither has been deleted.

### The evidence for that ranking

- `README.md`'s YAML frontmatter (`sdk: streamlit`, `app_file:
  streamlit_app.py`) is Hugging Face Spaces / Streamlit Community Cloud's
  own submission-metadata format — this repo is *structured* as a
  Streamlit-first submission, not an afterthought.
- `README.md` says explicitly: *"This README covers the PRD-scoped
  pipeline: ingestion → ResNet18 → PCA → 4-qubit VQC → **Streamlit UI** →
  Grad-CAM → SVM benchmark."*
- `extras/README.md` documents, with git history preserved, that the
  FastAPI backend + its auth stack were deliberately moved out of judged
  scope (`qknee/api/` → `extras/api/`) because the PRD scopes a Streamlit
  UI, not a second HTTP service.
- `scripts/verify_deployment.py` (the repo's own pre-deploy sanity check)
  targets the Streamlit app only, and says so in its own docstring: *"The
  FastAPI-specific gates ... were quarantined to `extras/`."*
- Nothing in this repository — no CI config, no `vercel.json` pointing at
  a real deployed project, no README section — claims the Next.js/FastAPI
  pair has ever been deployed end-to-end. `extras/README.md`'s "Deployment
  targets" table records this explicitly: Render/Vercel config is
  internally consistent and import-clean (validated by
  `extras/tests/test_deployment_config.py`) but has not been deployed to a
  live account as part of this project's work.

`frontend/` was simply never folded into that documented decision — it sits
at the repo root, untouched by the `extras/` quarantine move, calling the
quarantined API by default (`https://qknee-api.onrender.com`). That is the
actual bug this document fixes: not that the wrong system was chosen, but
that the second system was never written down anywhere.

---

## Both request paths, end to end

Every request, on either interface, ends up running the exact same five
pipeline stages (`qknee/models/pipeline.py`'s `PipelineRunner` —
see `README.md`'s "System architecture" section for the full stage table).
Neither interface reimplements any of it.

### Primary: Streamlit (in-process)

```
Browser
   │  streamlit run streamlit_app.py  (or HF Spaces / Streamlit Cloud)
   ▼
qknee/ui/dashboard.py   ← the actual application: no network hop to a
   │                       separate backend process; PipelineRunner and the
   │                       ACL/MCL/Meniscus VQC heads are constructed once,
   │                       in-process, at Streamlit's own cold start
   ▼
qknee.models.pipeline.PipelineRunner
   │
   ▼
DataIngestion → ResNet18FeatureExtractor → QuantumDimReducer (PCA) →
VQCClassifier (PennyLane) → qknee.xai.gradcam
```

### Secondary/exploratory: Next.js + FastAPI (HTTP)

```
frontend/  (Next.js/React, deployed separately — e.g. Vercel)
   │  fetch(`${NEXT_PUBLIC_API_URL}/predict`, ...)   — frontend/lib/api.ts
   ▼
extras/api/server.py  (FastAPI, "uvicorn extras.api.server:app")
   │  QKneeBackend.predict() — extras/api/server.py
   ▼
qknee.models.pipeline.PipelineRunner        ← the SAME class as above
   │
   ▼
DataIngestion → ResNet18FeatureExtractor → QuantumDimReducer (PCA) →
VQCClassifier (PennyLane) → qknee.xai.gradcam
```

This is the exact chain the task that produced this document asked to have
documented:

```
Next.js frontend
   ↓
FastAPI API
   ↓
Q-Knee ML pipeline
   ↓
ResNet18 / PCA / VQC / Grad-CAM
```

It is real and it works (`extras/tests/test_deployment_config.py` proves
`extras.api.server` imports cleanly and serves `/predict`) — it is just not
the interface this project is judged on.

**No ML logic lives in `frontend/`.** `frontend/lib/api.ts` is a typed HTTP
client only: it builds `FormData`, calls `fetch`, and parses the JSON
response into the TypeScript types in `frontend/lib/types.ts`. There is no
ResNet/PCA/VQC/Grad-CAM code anywhere under `frontend/`, and none should
ever be added there — a Next.js app doing tensor math would duplicate,
not share, `qknee/models/`.

---

## Why the two interfaces show different things — and why that's not a bug

Both interfaces call `PipelineRunner`, but they present **different
numbers of risk scores**, on purpose:

| | Streamlit dashboard | Next.js / FastAPI |
|---|---|---|
| Risk scores shown | **Three**: ACL, MCL, Medial Meniscus — each its own, independently-instantiated `VQCClassifier()` (`qknee/ui/dashboard.py`'s `load_backend()`), sharing one ResNet18+PCA feature pipeline | **One** unified risk score, from `PipelineRunner`'s own single `VQCClassifier` (`config.paths.model_checkpoint`) |
| Checkpoint status | Per-head, via `qknee.observability.model_health.full_status` — MCL is a permanent, explicitly-labeled research placeholder (no checkpoint has ever existed for it); ACL/Meniscus show "UNAVAILABLE" honestly when their specific checkpoint files aren't present | Single status for the one head it serves, same `model_health` module, surfaced in `GET /health`'s `model_status` field |
| Provenance labels | `qknee.observability.provenance.classify` → LIVE / PRECOMPUTED DEMO / MOCK-FALLBACK / CACHED / PROXY, rendered via `render_provenance_badge` | The exact same `provenance`/`provenance_label`/`model_source`/`quantum_execution` fields, computed by the same function, in every `PredictionResponse` |

The three-heads-vs-one-head difference is a real architectural fact (the
Streamlit dashboard was built to triage three named conditions; the API
predates/serves a simpler single-score contract) — it is not silent,
because both surfaces now derive their trustworthiness signals (is this
live, is this a real trained checkpoint, is the quantum simulator really
running) from the same two shared modules:

- `qknee/observability/provenance.py` — one `classify()` function, called
  by both `extras/api/server.py` and `qknee/ui/dashboard.py`, producing
  identical `LIVE`/`PRECOMPUTED DEMO`/`MOCK/FALLBACK`/`CACHED`/`PROXY`
  vocabulary on both surfaces. Neither surface invents its own wording.
- `qknee/observability/model_health.py` — one `inspect_vqc_checkpoint()` /
  `full_status()` / `quick_status()` set of functions, called by both
  `GET /health`'s `model_status` field and the dashboard's "Model Health"
  sidebar panel, so "is there a real trained checkpoint" is answered
  identically everywhere it's asked.

Neither surface will ever silently present a randomly-initialized model's
output as a trustworthy prediction — see AUDIT.md P1 #6/#7 for the fixes
that made this true, and `extras/tests/test_jwt_security.py`,
`qknee/tests/test_provenance.py`, and `qknee/tests/test_model_health.py`
for the regression tests.

## Benchmark numbers: one source, reproduced in three places

The only real, ground-truth benchmark this project has is
`qknee/artifacts/kaggle_benchmark_summary.json` (real RSNA Knee data,
n=58 studies), generated by `qknee/models/evaluate.py` and narrated in
`RESULTS.md`. Three places currently *state* these numbers:

1. `RESULTS.md` (the canonical narrative, with full methodology/caveats).
2. `README.md`'s "Key metrics" section (states it reproduces `RESULTS.md`
   verbatim; the two are cross-checked in this repo's own review history).
3. `frontend/components/landing/BenchmarksTable.tsx` — a **static
   reproduction** of the same three numbers (0.6574 / 0.5596 / 0.5579
   macro-AUC), with a code comment naming its exact source file and an
   on-page disclaimer that the hybrid VQC trails both classical baselines.
   This is a real fix over AUDIT.md's original finding (C4a): the
   landing page used to show fabricated `0.884`/`0.912` numbers under a
   "Verified Clinical Benchmarks" heading, attributed to a "Stanford MRNet
   validation cohort" that this project never evaluated on. Both the fake
   numbers and the MRNet attribution are gone (`frontend/components/landing/
   Hero.tsx` now says "Real RSNA Knee ground truth, n=58").

**Known residual risk, not a current contradiction**: `BenchmarksTable.tsx`
is a compile-time constant, not a live fetch from
`kaggle_benchmark_summary.json` — if that file is regenerated with new
numbers, someone has to update the three places by hand. This is called
out explicitly (rather than silently accepted) as a follow-up: a build-time
or request-time fetch from the same JSON file would close this gap, but
doing so is outside this document's scope (documentation/architecture, not
a new data-fetching feature).

---

## Directory map

| Directory | What it is | Judged scope? |
|---|---|---|
| `qknee/models/`, `qknee/data/`, `qknee/xai/`, `qknee/config/` | The ML pipeline itself (ResNet18 → PCA → VQC → Grad-CAM) | **Yes** — shared by both interfaces |
| `qknee/ui/` | The Streamlit dashboard (primary interface) | **Yes** |
| `streamlit_app.py` | Repo-root wrapper so HF Spaces/Streamlit Cloud can boot the dashboard | **Yes** |
| `qknee/observability/` | Shared provenance/model-health vocabulary used by both interfaces | **Yes** (supports both) |
| `extras/api/` | The FastAPI backend (was `qknee/api/`) | No — quarantined |
| `extras/ui/` | The clinician-auth Streamlit views the FastAPI backend used before quarantine | No — quarantined |
| `extras/deployment/` | Docker/Render/Vercel config for the FastAPI backend | No — quarantined |
| `frontend/` | The Next.js client for the FastAPI backend | No — was undocumented; now explicitly secondary/exploratory (this document) |

---

## Exactly which command starts what

| Interface | Command | Notes |
|---|---|---|
| **Streamlit dashboard (primary)** | `streamlit run qknee/ui/dashboard.py` (or `streamlit run streamlit_app.py` from the repo root, matching how HF Spaces/Streamlit Cloud actually boot it) | Needs `pip install -r requirements.txt` and a fitted PCA artifact — see `README.md` §"Setup & installation". Falls back to seeded mock inference automatically if the checkpoint/PCA artifact are missing. |
| **FastAPI backend (secondary)** | `uvicorn extras.api.server:app --reload --port 8000` | Needs `pip install -r requirements.txt -r extras/api/requirements.txt` and a JWT secret (`extras/api/.env.example`) — see `extras/README.md` §"Security configuration". |
| **Next.js frontend (secondary)** | `cd frontend && npm install && npm run dev` | Reads `NEXT_PUBLIC_API_URL` (defaults to the FastAPI backend above's would-be Render URL); see `frontend/.env.local.example`. |

`README.md` §"Run locally" documents these same three commands for a new
developer, in this order, with the same primary/secondary labeling.
