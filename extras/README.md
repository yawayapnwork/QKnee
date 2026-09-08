# Extras — quarantined, not part of the judged PRD scope

Everything under this directory was moved out of `qknee/` (and the repo
root) because it falls outside the hackathon PRD's scoped pipeline:

    ingestion -> ResNet18 -> PCA -> 4-qubit VQC -> Streamlit UI -> Grad-CAM -> SVM benchmark

Nothing here was deleted — it's moved, with git history preserved via
`git mv`, so it can be restored (`git mv extras/<path> <original path>`)
if a later effort wants to bring the multi-service (API + auth + cloud
deploy) architecture back.

**Import paths are NOT preserved.** Code that lived at `qknee/api/...`
now lives at `extras/api/...` — anything that did `from qknee.api...`
will `ModuleNotFoundError` from here on. Restoring functionality means
moving the files back, not just running them from this location.

## What's here and why

- **`api/`** (was `qknee/api/`) — the FastAPI server (`server.py`,
  `auth.py` — JWT/Argon2/SQLAlchemy user store, `requirements.txt`,
  `users.json`). The PRD scopes a Streamlit UI, not a second HTTP
  service.
- **`ui/`** (was `qknee/ui/auth_view.py`, `qknee/ui/landing_page.py`) —
  the clinician sign-in flow and institutional marketing landing page.
  Both called the FastAPI backend above (`/api/v1/auth/*`, `/health`),
  so they're quarantined together with it. `qknee/ui/dashboard.py` was
  edited (not moved — it's core PRD scope) to drop its dependency on
  both and open directly into the diagnostic workstation, unauthenticated.
- **`deployment/`** (was repo root `Dockerfile`, `docker-compose.yml`,
  `docker-compose.override.yml`, `render.yaml`, `vercel.json`,
  `requirements-vercel.txt`) — multi-service container/cloud deploy
  config for the API + UI architecture above. The PRD's judged deploy
  target is `streamlit run streamlit_app.py`.
- **`scripts/export_onnx.py`** — exports a decoupled ONNX Runtime graph
  for `ResNet18FeatureExtractor`. `qknee/models/resnet_extractor.py`'s
  `ONNXFeatureExtractor` and `qknee/models/pipeline.py`'s
  `backend_engine="onnx"` path / `HybridONNXInferenceEngine` were
  **left in place** (not quarantined) since they're part of files the
  PRD scope marks do-not-touch. The default config (`config.yaml`'s
  `resnet.backend_engine: "pytorch"`) never touches this path, so this
  is dead code in the judged demo, just not deleted.
- **`tests/test_hybrid_onnx_engine.py`** — exercises the export/import
  round-trip (`export_onnx.py` -> `ONNXFeatureExtractor`/
  `HybridONNXInferenceEngine`) directly, so it moved with the export
  script rather than staying in `qknee/tests/`.
- **`scripts/generate_kaggle_submission.py`** — builds an RSNA Knee
  Kaggle competition submission CSV. Its test (`test_rsna_kaggle_submission.py`)
  moved with it.
- **`models/vqc_strongly_entangling.py`** — an alternate VQC ansatz
  (`qml.StronglyEntanglingLayers`), not the one `config.yaml`'s
  `quantum.n_qubits: 4` / `quantum.n_layers: 3` names (`qknee/models/vqc.py`).
  Unlike `vqc_multitarget.py`/`vqc_data_reuploading.py`/
  `quantum_autoencoder.py` (which stayed in `qknee/models/` — see below),
  nothing in the do-not-touch core pipeline imports this one, so it moved
  cleanly.
- **`tests/`** — the test files whose only subject is the code above:
  `test_api_server.py`, `test_auth.py`, `test_auth_and_navigation.py`,
  `test_auth_view.py`, `test_landing_page.py`, `test_rsna_kaggle_submission.py`,
  `test_hybrid_onnx_engine.py`. `pytest.ini`'s `testpaths = qknee/tests`
  already excludes this directory from the main test run — no config
  change was needed.

## Security configuration (JWT secret, environment)

`extras/api/auth.py` no longer ships any fallback JWT signing secret — see
AUDIT.md P1 #8 (the old committed default in `config.yaml`, E1, is gone
entirely). Before running `extras/api/server.py` (directly, via
`docker-compose`, or on Render):

1. Copy `extras/api/.env.example` to `.env` (git-ignored) and set
   `QKNEE_JWT_SECRET_KEY` to a real, random, >= 32-character secret:
   `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
2. `$QKNEE_ENV` controls strictness — unset or anything other than
   `development`/`dev`/`local`/`test` is treated as **production**, which
   makes a missing or weak secret a hard startup failure
   (`InsecureJWTConfigurationError`), never a warning-and-continue. Local
   Docker Compose sets `QKNEE_ENV=development` +
   `QKNEE_ALLOW_INSECURE_JWT_SECRET=1` so a fresh clone can boot without
   generating a secret first — **never set that combination anywhere
   reachable outside your own machine.**
3. Render's `render.yaml` sets `QKNEE_ENV=production` explicitly and
   leaves `QKNEE_JWT_SECRET_KEY` as `sync: false` (must be entered manually
   in the Render dashboard — Render never auto-populates or defaults it).

See `qknee.api.auth.resolve_jwt_secret`'s docstring for the exact
precedence/validation rules (minimum length, rejected placeholder values,
the two-opt-in local-dev escape hatch), and `extras/tests/test_jwt_security.py`
for the regression tests covering every branch of that logic.

## What was requested but deliberately NOT moved (see PR discussion)

Three explicitly-requested removals turned out to be load-bearing
dependencies of files on the do-not-touch list, and were kept in place
after checking with the repo owner rather than guessed at:

- **`qknee/models/vqc_multitarget.py`, `vqc_data_reuploading.py`,
  `quantum_autoencoder.py`** stay in `qknee/models/`. `qknee_model.py`
  imports `vqc_multitarget` at module scope for `QKneeMultiTargetModel`;
  `pipeline.py` imports `quantum_autoencoder`/`vqc_data_reuploading` for
  its optional `encoder_type`/`classifier_backbone` variants; and
  `qknee/models/__init__.py` imports `vqc_data_reuploading` unconditionally
  at package-import time — moving any of the three would break
  `import qknee.models` entirely, taking the core PRD-scoped pipeline
  down with it.
- **ONNX code in `resnet_extractor.py`/`pipeline.py`** stays, for the
  same reason (see `scripts/export_onnx.py` above) — only the export
  script moved.

If a future pass wants the PRD-only surface fully minimal, that means
editing `qknee_model.py`/`pipeline.py`/`qknee/models/__init__.py`
directly (dropping `QKneeMultiTargetModel`, the alternate
encoder/classifier backbones, and the ONNX branch), not just moving
files — a larger, more invasive change than this pass made.
