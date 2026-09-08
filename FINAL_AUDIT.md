# Q-Knee — FINAL Adversarial Audit

**Posture**: hostile hackathon judge + senior ML engineer. Nothing in AUDIT.md,
ARCHITECTURE.md, or prior session summaries was trusted — every finding below
was re-derived by reading the current source, running the current test
suites, and (where a claim couldn't be substantiated) classifying it as
**UNSUPPORTED** rather than assuming it's true. No code was modified during
this audit.

**Bottom line**: the two most severe issues found are in the **primary,
judged interface** (the Streamlit dashboard), not the quarantined one. Both
are reproducible today, on this exact checkout, with no special setup:

1. On first load, with no file uploaded, the dashboard silently diagnoses
   pure Gaussian noise as if it were an MRI (§FAIL-1).
2. The top-level "LIVE" provenance badge does not know that all three risk
   heads on this checkout are checkpoint-less — it can and does show green
   "LIVE" while every gauge below it says "UNAVAILABLE" (§FAIL-2).

---

## Subsystem verdicts

| Subsystem | Verdict | Notes |
|---|---|---|
| ML pipeline core (`qknee/models/`, `qknee/data/`, `qknee/xai/`) | **PASS** | 390/390 tests pass. PCA fit is train-only (no leakage found). |
| Streamlit dashboard — MRI viewer/plane controls | **PASS** (with one WARNING) | Real controls; see FAIL-3 for the one dishonest plane claim. |
| Streamlit dashboard — provenance/trust signals | **FAIL** | FAIL-1, FAIL-2 below. |
| Next.js/FastAPI backend — inference/provenance | **PASS** | Live-vs-mock gating, checkpoint gating, Grad-CAM/base-image separation all verified correct in current code. |
| Next.js frontend — schema/types | **WARNING** | FAIL-4 (missing `model_status` field — doesn't break anything, but silently regresses a shipped backend feature to invisible on this surface). |
| Next.js frontend — cosmetic controls | **WARNING** | FAIL-5 (Window/Level buttons fabricate WL/WW numbers not tied to real windowing math). |
| Benchmark numbers | **PASS** | Verified byte-for-byte against `kaggle_benchmark_summary.json`. |
| Clinical-claim hedging | **WARNING** | FAIL-6 (Hero.tsx headline is unhedged marketing copy above the fold; disclaimer is present but visually subordinate). |
| Evaluation methodology transparency | **WARNING** | FAIL-7: single 75/25 split on n=58, not disclosed as such in RESULTS.md — classified **UNSUPPORTED**, not proven wrong, but not substantiated either. |
| Deployment config (`extras/deployment/*`) | **PASS** | Re-verified fresh: no stale `qknee.api.*` executable references, Render URL/frontend default agree, `docker compose config` validates cleanly. |
| Packaging metadata (`pyproject.toml`) | **FAIL** | FAIL-8: never touched by the prior deployment-config fix pass; contains 3 stale/broken references. |
| Secrets/JWT | **PASS** | `resolve_jwt_secret` re-verified: no bypass, no committed default, demo account server-side-blocked from inference. |
| Frontend build/typecheck | **PASS** | `next build` and `tsc --noEmit` both clean. |
| Frontend lint | **FAIL** | FAIL-9: `npm run lint` is non-functional (no ESLint config exists; interactive prompt, exit code 1 in a non-interactive shell). |
| Error handling (API → client) | **PASS** (one WARNING) | FAIL-10 (Streamlit's `run_api_inference` failure path is silent to the user, logged server-side only). |

---

## FAIL findings

### FAIL-1 — Default app state runs full "diagnosis" on random noise, disclosed only by one small caption
- **File/function**: `qknee/ui/dashboard.py:1413-1416`, inside `render_diagnostic_tab`
- **Exact code**:
  ```python
  if not uploaded_files:
      st.sidebar.caption("No file uploaded — viewing a synthetic demo volume (24 slices).")
      rng = np.random.default_rng(123)
      volume = rng.normal(loc=128, scale=40, size=(24, 224, 224)).clip(0, 255)
  ```
- **Problem**: this is pure Gaussian noise, not synthetic anatomy, and it is fed through the **identical** downstream path as a real DICOM upload: tri-planar reslicing with a clinical crosshair overlay (`theme.draw_clinical_crosshair`), `run_live_inference`/`run_mock_inference` producing a real risk score, a real Grad-CAM heatmap, and quantum telemetry — all rendered with the same visual weight as a genuine scan. The only disclosure anywhere in the app is one small `st.sidebar.caption` line; the main viewport, the Diagnostic Triage Card, and the provenance badge give no indication the input is noise. Opening the app and clicking through with zero uploads produces a fabricated-looking clinical result.
- **Severity**: **HIGH** — this is the primary, judged interface, in its default (first-impression) state.
- **Recommended fix**: either don't run inference at all on the synthetic placeholder (disable the triage card / show an explicit "upload a scan to begin" empty state instead of a scored result), or carry a hard `provenance="mock_fallback"`-equivalent tag through this exact path so the badge and an `st.error` both fire, matching how a real inference failure is already handled elsewhere in this same file.

### FAIL-2 — Provenance badge can say "LIVE" while every risk score on screen is "UNAVAILABLE"
- **File/function**: `qknee/ui/dashboard.py:717-736`, `render_provenance_badge`
- **Exact code**:
  ```python
  info = provenance_module.classify(
      backend_tag=result.backend,
      model_checkpoint_loaded=None,  # per-head status is shown separately, see render_risk_gauge
      quantum_expectations_present=result.pauli_z_expectations is not None,
  )
  ```
- **Problem**: `classify()` (`qknee/observability/provenance.py:164-168`) only downgrades `"live"` → `mock_fallback` when `model_checkpoint_loaded is False`. Passing `None` here means the downgrade path can never trigger for this badge, regardless of whether any head actually has a trained checkpoint. **Verified against this exact checkout**: `qknee/artifacts/acl_vqc.pt` and `meniscus_vqc.pt` do not exist, and MCL has no checkpoint path configured at all —
  ```
  primary available
  acl unavailable   No checkpoint found at qknee\artifacts\acl_vqc.pt.
  meniscus unavailable   No checkpoint found at qknee\artifacts\meniscus_vqc.pt.
  mcl unavailable   No MCL checkpoint path is configured...
  ```
  (reproduced live via `qknee.observability.model_health.full_status()`). Because the shared ResNet18/PCA pipeline itself loads fine (`backend_ready=True`, since the *primary* unified checkpoint exists), `run_live_inference` tags `backend="live"`. `render_risk_gauge` correctly shows "UNAVAILABLE" for all three heads (the per-head gating from AUDIT P1#6 works correctly) — but the badge above them says green **LIVE**, and `info.is_trustworthy` is `True` for `provenance="live"`, so the `st.error(...)` guard at line 735 never fires. A user sees a trustworthy-looking "LIVE" badge directly above three gauges that all say the model was never trained.
- **Severity**: **HIGH** — reproduces today, no special setup, in the judged interface, at exactly the UI element AUDIT P1#5/#6/#7 were meant to fix.
- **Recommended fix**: derive `model_checkpoint_loaded` from `result.checkpoint_status` (already computed and attached to `InferenceResult`) — e.g. pass `False` when every entry in `checkpoint_status` is `False`/missing, `True` when at least one head is trained, so `classify()` can correctly downgrade the badge when zero heads are trustworthy.

### FAIL-3 — Dashboard presents Coronal/Sagittal as real anatomical planes; the API it shares a pipeline with explicitly refuses to
- **File/function**: `qknee/ui/dashboard.py:601-615` (`get_slice`), used by the "Synchronized Orthogonal Viewport" section of `render_diagnostic_tab`
- **Exact code**:
  ```python
  elif view == "Coronal":
      return volume[:, index, :]
  elif view == "Sagittal":
      return volume[:, :, index]
  ```
- **Problem**: this indexes the *same* array's in-plane pixel-resolution axes and presents the result as "Coronal"/"Sagittal" — with a rendered clinical crosshair and per-plane slice-depth captions implying real anatomical views. `extras/api/server.py:809-816` documents, in its own code, exactly why this is not a real anatomical view for a typical single-series MRI (*"reslicing... along the in-plane pixel axes... isn't a real anatomical view"*) and marks Coronal/Sagittal `available: false` for this reason. The two interfaces that are supposed to share one ML pipeline and one honesty standard (per `ARCHITECTURE.md`) disagree on this exact architectural fact, with zero caveat on the dashboard side.
- **Severity**: **MEDIUM-HIGH** — a direct, current cross-interface inconsistency (audit category 6/9/15), not just an unsupported claim in one place.
- **Recommended fix**: either add the same "not a real anatomical view for single-series input" caveat to the dashboard's Synchronized Orthogonal Viewport caption, or gate Coronal/Sagittal there the same way the API does.

### FAIL-4 — `HealthResponse.model_status` exists in the API but has no representation in the frontend's types
- **File/function**: `frontend/lib/types.ts` (`HealthResponse` interface) vs. `extras/api/server.py:508` (`model_status: Dict[str, str]`)
- **Problem**: the Python `HealthResponse` has 8 fields including `model_status`; the TypeScript interface has 7 — `model_status` is absent entirely. `frontend/lib/api.ts`'s `fetchHealth()` is typed against the incomplete interface, so nothing on the Next.js side can ever surface "MCL unavailable" / "ACL unavailable" the way the Streamlit sidebar's Model Health panel does. This is a schema mismatch (category 13) and, downstream, a cross-interface inconsistency (category 6/7): the Next.js frontend has strictly less trust-signal visibility than the interface it's supposed to mirror.
- **Severity**: **MEDIUM** — no crash (TS simply can't see the field; nothing reads or misreads it), but a real, silent feature gap.
- **Recommended fix**: add `model_status: Record<string, string>` to `HealthResponse` in `frontend/lib/types.ts` and surface it in `CommandBar.tsx` or a dedicated status panel.

### FAIL-5 — MRI viewport's "Window/Level" presets fabricate real-looking WL/WW numbers for a plain CSS filter
- **File/function**: `frontend/components/workstation/MriViewport.tsx:18-21, 47`
- **Exact code**:
  ```ts
  const WINDOWS = [
    { id: "soft-tissue", label: "Soft Tissue", wl: 40, ww: 400 },
    { id: "bone", label: "Bone", wl: 500, ww: 2000 },
  ] as const;
  ...
  const filter = windowPreset === "bone" ? "contrast(1.4) brightness(1.1)" : "contrast(1.05) brightness(1)";
  ```
- **Problem**: `wl`/`ww` are displayed in the UI (`WL{w.wl}/WW{w.ww}`) as if they were real DICOM windowing parameters, but they are never consumed — the actual visual effect is two hardcoded, unrelated CSS `filter` strings. A radiologist reading "WL500/WW2000" would reasonably expect real pixel-intensity windowing math, not a canned brightness/contrast bump. This is not fully cosmetic (the button does change the image), which makes it worse than a no-op control: it's a control that does *something*, but not the specific, numerically-labeled thing it claims to do.
- **Severity**: **LOW-MEDIUM**.
- **Recommended fix**: either implement real WL/WW pixel remapping, or drop the fabricated numeric labels and call the control what it is ("Contrast preset").

### FAIL-6 — Landing page headline is unhedged clinical-sounding marketing copy above the fold
- **File**: `frontend/components/landing/Hero.tsx:20-25` vs. the disclaimer at line 47
- **Problem**: `"Q-Knee **Diagnostic Platform**"` / `"Accelerating orthopedic knee MRI triage..."` reads as a confident clinical-product pitch. The hedge ("Investigational research prototype... Confirmatory radiologist over-read required") exists but is small text below the CTA buttons, not adjacent to the headline itself.
- **Severity**: **LOW-MEDIUM** — not a false claim (nothing here is untrue), but a hostile judge can reasonably call the framing order "clinical-washing." Every other clinical-sounding string in the codebase checked during this audit (`README.md`, `TriageCard.tsx`, `ReportExport.tsx`) is immediately and adequately hedged.
- **Recommended fix**: move the disclaimer (or a short form of it) adjacent to the headline, not just below the CTA row.

### FAIL-7 — Evaluation methodology's split robustness is UNSUPPORTED, not proven wrong
- **File**: `RESULTS.md` §1 vs. `qknee/models/evaluate.py`'s `run_kaggle_macro_auc_benchmark`
- **Problem**: PCA/StandardScaler fitting is correctly train-only (verified: `reducer.fit_transform(X_train)` / `reducer.transform(X_test)`, no leakage). However, the benchmark is a **single** 75/25 split (`test_size=0.25`, `seed=0`) on n=58 studies (~14-15 in the held-out test set), with the same split reused across every architecture/condition compared. `RESULTS.md` never states the split ratio, that it's a single non-cross-validated split, or that per-condition AUCs at this test-set size are highly sensitive to which 14 studies land in it. The document's own "directional, not statistically robust" hedge is present and honest in spirit — but "directional" doesn't by itself communicate "a different random seed could plausibly reorder these three models."
- **Severity**: **MEDIUM** — classified **UNSUPPORTED** (not disproven; RESULTS.md's numbers are real and correctly computed) rather than a fabrication. This is a transparency gap in the writeup, not a code bug.
- **Recommended fix**: state the split methodology explicitly in RESULTS.md §1, and/or report a cross-validated estimate.

### FAIL-8 — `pyproject.toml` was never touched by the deployment-path fix and still points at the pre-quarantine module
- **File**: `pyproject.toml:38-39, 98, 109`
- **Exact code**:
  ```toml
  # in `requirements-vercel.txt` / `qknee/api/requirements.txt`), where
  # `qknee/api/server.py` runs as a thin router/mock/proxy layer, never the
  ...
  qknee-server = "qknee.api.server:main"
  ...
  [tool.vercel]
  entrypoint = "qknee.api.server:app"
  ```
- **Problem**: three separate stale references to the pre-quarantine `qknee.api.*` path, none of which were caught by the earlier deployment-configuration audit (which only checked `extras/deployment/*`, `Dockerfile`, and `docker-compose*.yml`). Compounding this: `extras/api/server.py` has **no `def main()`** at all — even correcting the module path to `extras.api.server` wouldn't fix the console-script entry, because the attribute it names doesn't exist. Separately, `[tool.setuptools.packages.find]` (line 103) only includes `qknee*` — so even a corrected `extras.api.server:main` reference would not be installable as a console script from a built wheel of this package, since `extras/` is never packaged. `qknee-train` and `qknee-verify` (the other two `[project.scripts]` entries) were checked and are genuinely valid (`scripts/train.py:595`, `scripts/verify_deployment.py:415` both have real `main()` functions) — this is specific to `qknee-server`.
- **Severity**: **MEDIUM** — not on the runtime path any currently-documented deployment target uses (Render/Vercel both use their own YAML/JSON config, not this file's metadata), but it is exactly the class of bug the prior deployment audit was supposed to eliminate project-wide, and `pip install -e . && qknee-server` — a completely reasonable thing for a new contributor to try — fails immediately.
- **Recommended fix**: either fix the entry point (`extras.api.server:app` for `[tool.vercel]`; delete `qknee-server` from `[project.scripts]` since no `main()` exists to point it at, or add one) and the two prose comments, or remove the dead `[tool.vercel]` block and `qknee-server` script entirely if this packaging metadata isn't actually part of any real deploy path (also update `[tool.setuptools.packages.find]` to include `extras*` if `extras.api.server` is ever meant to be pip-installable).

### FAIL-9 — `npm run lint` does not run in a non-interactive environment
- **File**: `frontend/package.json`'s `lint` script (`next lint`); no ESLint config file exists anywhere under `frontend/`
- **Reproduction**:
  ```
  $ npm run lint
  ? How would you like to configure ESLint? https://nextjs.org/docs/basic-features/eslint
  ❯ Strict (recommended) / Base / Cancel
  ```
  exits 1 (`NativeCommandError`) in a non-interactive shell, because Next.js's `next lint` prompts to scaffold a config on first run and none has ever been committed. `next.config.mjs` also sets `eslint: { ignoreDuringBuilds: true }`, so `next build` silently never runs it either — meaning **lint has, as far as this audit can determine, never actually executed successfully in this project's history.**
- **Severity**: **MEDIUM** — a real, verifiable gap in what "run lint" as a validation step can mean for this repo today; not a runtime bug, but a broken tooling promise (`package.json` advertises a working `lint` script that isn't one).
- **Recommended fix**: commit an ESLint config (`.eslintrc.json` with `"extends": "next/core-web-vitals"` is the standard non-interactive choice) so `npm run lint` is actually runnable in CI.

### FAIL-10 — Streamlit's API-fallback failure is invisible to the user
- **File/function**: `qknee/ui/dashboard.py`, inside `render_diagnostic_tab`
- **Exact code**:
  ```python
  if use_api:
      try:
          result = run_api_inference(raw_slice, api_url)
      except Exception as exc:  # noqa: BLE001 - a failed request degrades to in-process/mock, not a crash
          logger.warning("API inference failed (%s); falling back to in-process/mock.", exc)
  ```
- **Problem**: this is not a fabricated result (whatever the fallback produces is itself correctly labeled by its own provenance badge) — but the *transition itself* is silent to the viewer. If `$QKNEE_API_URL` is configured and passes the cheap `/health` reachability check at boot, then the specific `/predict` call times out or 500s mid-request (a realistic Render cold-start/network-blip scenario), the exception is caught and only `logger.warning`'d server-side. A viewer watching only the UI cannot distinguish "the preferred remote API path was attempted and failed just now" from "the API was never configured at all."
- **Severity**: **LOW-MEDIUM**.
- **Recommended fix**: surface a transient `st.toast`/`st.caption` on this exact except branch.

---

## Category-by-category checklist (as requested)

| # | Category | Verdict |
|---|---|---|
| 1 | Fake benchmark number | **PASS** — every number traced to `kaggle_benchmark_summary.json`, exact match. |
| 2 | Unsupported clinical claim | **WARNING** — see FAIL-6; everything else adequately hedged. |
| 3 | Fake quantum expectation | **PASS** — mock/random Pauli-Z values are always tagged `backend="mock"`/`quantum_execution="unavailable"` correctly; never presented as a real circuit's output. |
| 4 | Preset value leaking into live inference | **PASS** — traced `page.tsx`'s live branch and dashboard's cached/fast-path branches; no leak path found. |
| 5 | Mock result presented as live | **FAIL** — see FAIL-2 (badge-level, Streamlit only; the FastAPI backend's own `backend`/`provenance` gating was re-verified correct). |
| 6 | Random-weight model producing inference | **WARNING** — per-head score gating is correct (`None` when untrained), but the badge doesn't reflect it (FAIL-2), and FAIL-1's synthetic-noise path runs full inference regardless of checkpoint state. |
| 7 | MRI image confused with Grad-CAM | **PASS** — verified byte-distinct construction on every code path (live, mock, cached, preset). |
| 8 | Cosmetic viewer control | **WARNING** — every control checked is functionally wired except FAIL-5 (fabricated WL/WW labels on a real-but-mislabeled effect). |
| 9 | Unsupported 3D/volumetric claim | **FAIL** — FAIL-1 (noise-as-volume) and FAIL-3 (fake Coronal/Sagittal in the dashboard specifically). |
| 10 | Data leakage in evaluation | **PASS** (mechanics) / **UNSUPPORTED** (robustness) — see FAIL-7. |
| 11 | Stale deployment path | **FAIL** — `extras/deployment/*` itself is clean (re-verified), but FAIL-8 found `pyproject.toml` was missed entirely by the prior pass. |
| 12 | Insecure secret/default | **PASS** — no bypass found in `resolve_jwt_secret`; no committed real secret; demo account is server-side blocked from inference by role, not just UI-hidden. |
| 13 | Frontend/backend schema mismatch | **FAIL** — FAIL-4 (`model_status` missing); every other field of `PredictionResponse`, `PlaneInfo`, `Token`, `UserResponse` matches. |
| 14 | Endpoint silently converts failure to success | **PASS** (API) / **WARNING** (dashboard) — see FAIL-10; no FastAPI route or `ProxyBackend`/`CacheService` path found that masks a real failure as a 200. |
| 15 | Documentation contradiction | **FAIL** — FAIL-8 (`pyproject.toml` vs. everything else); FAIL-3 (dashboard vs. API on Coronal/Sagittal honesty). All other cross-document checks (primary-interface framing, dataset naming, benchmark numbers) passed. |

---

## Validation runs performed

| Check | Command | Result |
|---|---|---|
| Backend tests (core ML) | `pytest qknee/tests/ -q` | **390 passed**, 0 failed |
| Backend tests (API/auth/deployment) | `pytest extras/tests/ -q` (excluding 2 pre-existing collection errors in `test_auth_view.py`/`test_landing_page.py` — legacy `qknee.ui.auth_view`/`qknee.ui.landing_page` import paths, unrelated to this audit's scope) | **181 passed**, 11 failed, 24 errors — all 35 are the same pre-existing, previously-documented failures (`test_auth_and_navigation.py` Streamlit-navigation tests ×3, `test_hybrid_onnx_engine.py` ×9, `test_rsna_kaggle_submission.py` ×22); zero new failures |
| Frontend tests | `npm test` (vitest) | **41 passed**, 0 failed |
| Frontend typecheck | `npx tsc --noEmit` | **clean**, 0 errors |
| Frontend build | `npm run build` | **succeeds** |
| Frontend lint | `npm run lint` | **FAIL** — see FAIL-9 |
| Model-loading validation | `qknee.observability.model_health.full_status()` against the live config | `primary`: available; `acl`/`meniscus`/`mcl`: unavailable (see FAIL-2 for why this matters) |
| Deployment/build validation | `docker compose config --quiet` (from `extras/deployment/`); `python -c "import extras.api.server"` | Both **succeed** — compose file resolves all paths correctly, FastAPI app imports and exposes `/health`/`/predict`/`/api/v1/auth/login` cleanly with no test-only import shim active |

---

## What this audit found to be genuinely solid (not just "no news")

- Base-image/Grad-CAM-overlay separation is real and byte-distinct on every traced code path, including the two most easily-missed ones (`CachedFallbackBackend`'s empty-base-image case, and preset placeholder generation).
- The JWT secret resolution logic (`extras/api/auth.py`) has no bypass this audit could construct — both the "no secret" and "weak secret" branches require explicit, named opt-ins to become permissive, and both default to hard failure.
- `QKneeBackend.predict()`'s live/mock gating (the FastAPI side of AUDIT P1#6/#7) is implemented correctly and was re-verified by reading the current code, not assumed from prior sessions' summaries.
- The evaluation split (train/test) genuinely has no leakage in its mechanics — the one gap found (FAIL-7) is a disclosure gap, not a computed-number problem.
