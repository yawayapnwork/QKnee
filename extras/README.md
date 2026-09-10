# Extras — the FastAPI backend

> **See [`ARCHITECTURE.md`](../ARCHITECTURE.md)** at the repo root for the
> authoritative picture of how this backend and [`frontend/`](../frontend/)
> (the Next.js clinical workstation that calls it) relate — together they
> are the judged interface.

This directory holds the FastAPI backend (`server.py`, JWT/Argon2 auth),
its deployment config, and a few scripts/tests whose only subject is code
here — kept separate from `qknee/` (the shared ML pipeline both this API
and any other caller use) so the ML package has no FastAPI/auth
dependency baked in.

## What's here and why

- **`api/`** — the FastAPI server (`server.py`, `auth.py` —
  JWT/Argon2/SQLAlchemy user store, `requirements.txt`, `users.json`).
- **`deployment/`** — container/cloud deploy config: `Dockerfile`,
  `docker-compose.yml`, `docker-compose.override.yml`, `render.yaml`,
  `vercel.json`, `requirements-vercel.txt`.
- **`scripts/export_onnx.py`** — exports a decoupled ONNX Runtime graph
  for `ResNet18FeatureExtractor`. `qknee/models/resnet_extractor.py`'s
  `ONNXFeatureExtractor` and `qknee/models/pipeline.py`'s
  `backend_engine="onnx"` path / `HybridONNXInferenceEngine` live in
  `qknee/` itself (not here) since they're part of the core pipeline; the
  default config (`config.yaml`'s `resnet.backend_engine: "pytorch"`)
  never touches this path, so it's opt-in, not part of the default
  inference path.
- **`tests/test_hybrid_onnx_engine.py`** — exercises the export/import
  round-trip (`export_onnx.py` -> `ONNXFeatureExtractor`/
  `HybridONNXInferenceEngine`) directly, so it lives with the export
  script rather than in `qknee/tests/`.
- **`scripts/generate_kaggle_submission.py`** — builds an RSNA Knee
  Kaggle competition submission CSV. Its test
  (`test_rsna_kaggle_submission.py`) lives with it.
- **`tests/`** — the test files whose only subject is the code above:
  `test_api_server.py`, `test_auth.py`, `test_rsna_kaggle_submission.py`,
  `test_hybrid_onnx_engine.py`, `test_jwt_security.py`,
  `test_deployment_config.py`. `pytest.ini`'s `testpaths = qknee/tests`
  excludes this directory from the main ML-pipeline test run — run these
  with `pytest extras/tests`.

## Security configuration (JWT secret, environment)

`extras/api/auth.py` ships no fallback JWT signing secret. Before running
`extras/api/server.py` (directly, via `docker-compose`, or on Render):

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

See `extras.api.auth.resolve_jwt_secret`'s docstring for the exact
precedence/validation rules (minimum length, rejected placeholder values,
the two-opt-in local-dev escape hatch), and
`extras/tests/test_jwt_security.py` for the regression tests covering
every branch of that logic.

## Deployment targets

| Target | Entrypoint | Status |
|---|---|---|
| Render (`extras/deployment/render.yaml`) | `uvicorn extras.api.server:app` | Primary backend deploy target. `extras/tests/test_deployment_config.py` proves the app imports cleanly and `/health` is real. |
| Vercel API (`extras/deployment/vercel.json`) | `extras/api/server.py` (thin proxy/cache-fallback; see `requirements-vercel.txt`) | Alternate backend deploy target. |
| Vercel frontend (`frontend/vercel.json`) | `next build` / `next start` | Primary frontend deploy target. `NEXT_PUBLIC_API_URL` points at the Render service (`qknee-api` -> `https://qknee-api.onrender.com`). |
| Docker Compose (`extras/deployment/docker-compose.yml`) | `uvicorn extras.api.server:app` (api) | Local/self-hosted backend deploy. `docker compose config` (run from `extras/deployment/`) confirms the merged config is valid and every `build.context`/bind-mount path resolves relative to the repo root (`../..`, not `extras/deployment/` itself). |

**Caveat, stated plainly**: as of the last review pass, the image itself
was not built and run end-to-end against a live Render/Vercel account in
this environment (no Docker daemon available) — the config has been
validated for internal consistency and import-cleanliness, not for an
actual live deploy. Confirm a real deploy before treating this as
verified-in-production.
