# Q-Knee Repository Audit

Scope: full repository — `README.md`, `RESULTS.md`, `qknee/config`, `qknee/data`,
`qknee/models`, `qknee/xai`, `qknee/ui`, `extras/api`, `extras/deployment`,
`frontend/`, `qknee/tests`, `extras/tests`, `qknee/artifacts` metadata.

**No source code was modified. This is read-only analysis.**

## 0. Framing — read this before the findings below

This repo is in an unusual state: it contains **two generations of the
project**, only one of which the maintainers consider in-scope.

- **Current, judged scope** (per `README.md`'s own frontmatter and body):
  `ingestion → ResNet18 → PCA → 4-qubit VQC → Streamlit UI → Grad-CAM → SVM
  benchmark`. The Streamlit dashboard (`qknee/ui/dashboard.py`) opens directly
  into the diagnostic workstation, unauthenticated, backed by
  `qknee.models.pipeline.PipelineRunner`.
- **Quarantined** (moved to `extras/`, explicitly "outside the judged PRD
  scope" per `extras/README.md`): the FastAPI backend (`extras/api/`), JWT/Argon2
  clinician auth, the institutional landing page, and all Docker/Render/Vercel
  deployment config.
- **Undocumented third generation**: a Next.js/React frontend at `frontend/`
  (`app/workstation/page.tsx`, `MriViewport.tsx`, `QuantumTelemetry.tsx`, etc.)
  that is **mentioned nowhere** in `README.md`'s project structure or in
  `extras/README.md`'s inventory of quarantined code. It talks to the
  quarantined FastAPI backend (`lib/api.ts`'s `API_BASE_URL` default,
  `https://qknee.onrender.com`) via a contract (`lib/types.ts`) that only
  partially matches what that backend actually returns.

**Practical consequence**: if anyone demos via `frontend/` (the more
visually impressive of the two UIs), they are demoing code the project's own
documentation says is not the current architecture, wired to a backend the
same documentation says was intentionally removed from scope. Findings below
are grouped by which of these three surfaces they affect, since "does this
affect the demo" depends entirely on which surface is being shown.

The core pipeline itself (`qknee/models/`, `qknee/xai/`, `qknee/data/`) is
**well-engineered and unusually honest** — see §H for the positive findings
this audit would be incomplete without.

---

## A. Critical bugs

**A1. `extras/deployment/vercel.json` and `render.yaml` reference a module
path that no longer exists.**
- File: `extras/deployment/vercel.json:5-12`, `extras/deployment/render.yaml:24`
- What's wrong: `vercel.json`'s `builds[0].src` is `"qknee/api/server.py"` and
  `render.yaml`'s `startCommand` is `python -m qknee.api.server`. Per
  `extras/README.md`: *"Import paths are NOT preserved. Code that lived at
  `qknee/api/...` now lives at `extras/api/...`"* — the file at
  `qknee/api/server.py` doesn't exist; the real file is
  `extras/api/server.py`, importable only as `extras.api.server`.
- Why it matters: both configs would fail immediately on deploy
  (`ModuleNotFoundError` / build error) — they were not updated when the
  quarantine move happened.
- Fix: update to `extras/api/server.py` / `extras.api.server:app`, or delete
  these files if the deploy targets are genuinely abandoned.
- Demo impact: **No effect on the current judged demo** (Streamlit), but
  blocks anyone trying to restore the old multi-service deployment.

**A2. `render.yaml`'s `buildCommand` installs the wrong requirements file.**
- File: `extras/deployment/render.yaml:20-23`
- What's wrong: `pip install -r requirements.txt` installs the **repo-root**
  `requirements.txt`, which (per its own comment, line 71-73) explicitly
  states FastAPI/uvicorn/argon2-cffi/PyJWT/SQLAlchemy were removed from it
  when the API was quarantined — they now live in `extras/api/requirements.txt`,
  which this build command never references.
- Why it matters: even after fixing A1's path, this build would install a
  requirements file with no `fastapi`, so `extras.api.server` would fail to
  import at all.
- Fix: point `buildCommand` at `extras/api/requirements.txt`.
- Demo impact: same as A1 — dormant until someone tries to redeploy the API.

**A3. Frontend's "live" inference path silently substitutes mock data for
one entire panel.**
- File: `frontend/app/workstation/page.tsx:44-50`
```js
qubitExpectations: activeCase.qubitExpectations,  // reused from the preset, not the backend
source: "live",
```
- What's wrong: even when a real upload gets a real `risk_score` back from
  the backend, `QuantumTelemetry`'s per-qubit values are unconditionally
  taken from whatever hardcoded preset case is currently selected — never
  from the API. `lib/types.ts`'s `PredictionResponse` (the real backend
  contract) has no field for per-qubit expectations at all, so this isn't a
  wiring bug, it's structurally guaranteed: this data can never be real in
  the current frontend/backend contract.
- Why it matters: the "4-Qubit Variational Quantum Circuit" telemetry panel —
  the single most quantum-computing-specific piece of UI in the project — is
  decorative in 100% of code paths, including the one labeled `source: "live"`.
- Fix: either add a `pauli_z_expectations` field to the FastAPI
  `PredictionResponse` (the backend already computes this — see
  `qknee.ui.dashboard.get_pauli_z_expectations` for the equivalent Streamlit
  code path) and thread it through, or remove/relabel the panel so it isn't
  presented as per-image live data.
- Demo impact: **High**, if `frontend/` is what gets demoed.

---

## B. Demo-breaking bugs

**B1. `MriViewport.tsx`'s slice slider and plane selector control a text
label only — never the rendered image.**
- File: `frontend/components/workstation/MriViewport.tsx:24-28, 76-83, 99, 111-118`
- What's wrong: `slice` and `plane` state changes only feed the
  `"{plane.toUpperCase()} · SLICE {slice}/{SLICE_COUNT}"` caption; the
  `<img>` element always renders `result.heatmap` regardless of either
  value.
- Why it matters: a judge scrubbing the slice slider or switching
  Sagittal/Coronal/Axial will see the counter change and the image stay
  frozen — an easily-noticed break in a live demo.
- Fix: either wire real per-slice/per-plane data through (requires backend
  support the current `/predict` contract doesn't have), or remove the
  controls until they do something, so as not to imply multi-slice/tri-planar
  browsing that doesn't exist.
- Demo impact: **High**, `frontend/` only.

**B2. The "MRI slice" and "Grad-CAM overlay" images in `MriViewport.tsx` are
the same image asset rendered twice.**
- File: `frontend/components/workstation/MriViewport.tsx:76-88`
```jsx
<img src={result.heatmap...} alt="MRI slice" .../>
{heatmapOn && <img src={result.heatmap...} alt="Grad-CAM overlay" className="... mix-blend-screen" .../>}
```
- What's wrong: there is no separate raw-MRI-slice asset anywhere in the
  mock or live path; `result.heatmap` is used as both the base layer and the
  "overlay" blended on top of itself.
- Why it matters: opening dev tools' Network tab during a demo shows one
  image request, not two — undermines the "Grad-CAM overlay on the original
  slice" claim on inspection.
- Fix: surface the actual pre-inference slice image (the backend already has
  it — `display_slice` in `extras/api/server.py`) as a distinct field/asset.
- Demo impact: **High** if inspected, `frontend/` only.

**B3. Streamlit dashboard's MCL head has no dedicated checkpoint at all.**
- File: `qknee/ui/dashboard.py:219-222`
```python
torch.manual_seed(21)
mcl_model = VQCClassifier()
logger.warning("No MCL checkpoint configured (paths.mcl_checkpoint); using randomly initialized weights.")
```
- What's wrong: unlike ACL/meniscus (which load real checkpoints when
  present), MCL always runs on a randomly-initialized VQC — there isn't even
  a `paths.mcl_checkpoint` key in `config.yaml` to eventually fill in.
- Why it matters: the MCL risk score shown in the triage UI is not
  meaningfully different from noise, on every run, permanently — not a
  fallback-when-missing case like the other two heads.
- Fix: either add a real MCL checkpoint + config path, or remove the MCL
  gauge from the UI until one exists, rather than showing a permanently
  random number with the same visual weight as the two real heads.
- Demo impact: **Medium** — only apparent to someone who notices MCL's score
  never correlates with anything, but it's a real disclosure gap since this
  isn't flagged in the UI itself the way `backend="mock"` is.

---

## C. Scientific/ML validity problems

**C1. RESULTS.md's own honest reporting: VQC underperforms both classical
baselines on real data — this is disclosed, not hidden, but worth restating
plainly for whoever reads this audit next.**
- File: `RESULTS.md` §1, §4
- Finding: on real RSNA Knee ground truth (n=58), the hybrid VQC's
  macro-AUC (0.5632 / 0.5579 excl. Effusion) trails the plain ResNet18
  linear-probe baseline (0.6732 / 0.6574) by ~0.10–0.11, and a
  parameter-matched classical MLP (same 41 trainable params) is
  statistically indistinguishable from (arguably slightly ahead of) the VQC
  across 5 seeds.
- This is **not a bug** — it is the single most scientifically credible part
  of this repository, and should be preserved exactly as written if anyone
  is tempted to "improve" the numbers before a demo. Flagging it here only so
  whoever reads this audit doesn't miss that "quantum advantage" is
  explicitly **not** claimed anywhere in this codebase, contrary to what a
  reader might expect from a quantum-hackathon submission.

**C2. Multi-slice "volumetric" processing is slice-wise feature extraction +
pooling, not 3D modeling — accurately described in code/docs, but worth
confirming explicitly since the audit was asked to check for this.**
- File: `qknee/models/resnet_extractor.py` `forward_volume` (lines 264–301)
- Finding: a `(B, S, 3, 224, 224)` volume is folded into `(B*S, 3, 224, 224)`,
  run through the *same* 2D ResNet18 per slice, then aggregated
  (mean / attention / top-k-max) back to one `(B, 512)` embedding. There is
  no 3D convolution, no cross-slice spatial modeling — each slice is scored
  independently and pooled.
- This matches what `README.md`'s architecture table says
  ("multi-slice volumes are mean-pooled into one embedding") — **no
  overclaiming found here**. Documented as a finding only because the task
  explicitly asked to check for this pattern; no fix needed, just noting it's
  correctly disclosed.

**C3. Grad-CAM's central-slice choice for multi-slice volumes is a
reasonable heuristic, but silently mismatches whichever slice a viewer
might expect if a UI ever adds real slice browsing.**
- File: `qknee/models/pipeline.py` `PipelineRunner.run` (lines 677–686)
```python
central_slice_index = batch.shape[1] // 2
single_slice = batch[:, central_slice_index]
heatmap = self.explain(single_slice)
```
- What's wrong: the risk score is computed from *all* slices (mean-pooled),
  but the Grad-CAM heatmap is only ever computed for the anatomical
  midpoint slice — reasonable today since neither UI currently lets a user
  pick a different slice for a volume upload, but this is exactly the gap
  B1 would expose if `frontend/`'s slice slider were ever wired up: whichever
  slice a user scrubs to would still show the *central* slice's heatmap, not
  its own.
- Fix: none needed today; flagging as a design constraint to respect if slice
  browsing is ever implemented — `qknee.xai.gradcam.compute_volumetric_gradcam`
  already exists and computes a heatmap **per slice**, so the building block
  for a correct fix already exists in the codebase, just isn't wired to any
  UI yet.

**C4a. `frontend/components/landing/BenchmarksTable.tsx`'s headline numbers
are not merely unsourced (D4) — they directly contradict every real
benchmark artifact this repo actually produced.**
- File: `frontend/components/landing/BenchmarksTable.tsx:4-17` vs.
  `qknee/artifacts/benchmark_results.json`, `qknee/artifacts/kaggle_benchmark_summary.json`,
  `RESULTS.md` §1/§4.
- What's wrong: the landing page states AUC `0.884` / `0.912` under a
  **"Verified Clinical Benchmarks"** heading, with a caption naming a
  "Stanford MRNet validation cohort." No artifact in this repo contains
  either number:
  - `benchmark_results.json` is a **mock**-dataset smoke run
    (`"source": "mock"`, n_test=15) where the Hybrid VQC scores ROC-AUC
    **0.407** (worse than chance) and the classical linear baseline scores
    **0.296**.
  - `kaggle_benchmark_summary.json` — the real-data run (n=58 real RSNA
    studies, not MRNet) — has the VQC macro-AUC at **0.563**, *trailing*
    both classical baselines (0.673 ResNet18 linear-probe, 0.594 SVM).
  - `RESULTS.md` states this outcome explicitly: *"With Effusion included,
    VQC trails the ResNet18 baseline by 0.110."* No MRNet cohort is scored
    anywhere in this codebase — MRNet only appears as a synthetic-shape
    mock-data generator (`qknee/data/dataset.py`'s
    `generate_mock_mrnet_dataset`) used for pipeline smoke tests, never as
    an evaluated dataset.
- Why it matters: this is the single largest gap between a documentation/
  claim problem and a fabricated-looking demo number in the entire repo —
  a hackathon judge who reads `RESULTS.md` (honest: VQC underperforms) and
  then looks at the landing page (claims 0.884–0.912 "verified") will see
  a direct, checkable contradiction, not just an unsourced number.
- Fix: remove the hardcoded numbers and either (a) drop the benchmarks
  table until there's a real result worth publishing, or (b) show the
  actual `kaggle_benchmark_summary.json` numbers with the same honest
  framing `RESULTS.md` already uses ("directional, n=58, VQC trails
  classical baselines").
- Demo impact: **Critical if anyone cross-references RESULTS.md against
  the landing page** — this is a P0, not the P2 this was originally filed
  under in D4; see the ranked list in §I, updated accordingly.

**C4b. Silent random-weight checkpoint fallback is a general pattern, not
just the MCL-specific case in B3.**
- File: `qknee/models/pipeline.py:346-372` (`PipelineRunner.__init__`),
  `qknee/models/qknee_model.py:523-551` (`load_best_checkpoint_or_init`)
- What's wrong: on a fresh clone/deploy where the trained checkpoint
  (`qknee/artifacts/checkpoints/best_qknee_model.pt`, gitignored) is
  absent, the VQC head loads with **randomly initialized weights** and
  still returns a real-looking risk score in `[0,1]`, computed from real
  ResNet+PCA features feeding a meaningless classifier. The only signal is
  a `logger.warning` — invisible in a live demo unless someone is tailing
  server logs.
- Why it matters: unlike B3 (MCL, which is *permanently* fake because no
  checkpoint path exists at all), this is a *conditional* fallback that
  depends entirely on whether the gitignored checkpoint file happens to be
  present on whatever machine is running the demo — a classic
  "works on my machine, silently garbage on a fresh clone" failure mode.
- Fix: fail loudly (raise, or surface a hard `backend`/`degraded: true`
  flag in the UI) rather than logging a warning and proceeding, mirroring
  the good pattern already used for `_ALLOW_INMEMORY_DB` in
  `extras/api/auth.py` (see E2).
- Demo impact: **High** on any environment where the checkpoint wasn't
  carried over (e.g., a fresh clone for judging) — silently downgrades
  every score without visibly changing the UI.

**C4c. "SIMULATION MODE" badge in the Streamlit dashboard is ambiguous
between two different meanings of "simulation."**
- File: `qknee/ui/dashboard.py:634-661` (`render_quantum_status`)
- What's wrong: when the PCA artifact is missing, the sidebar shows
  "SIMULATION MODE — Quantum kernel unavailable." But the real backend
  is *also* always a simulator in the ordinary sense (PennyLane
  `default.qubit`/`lightning.qubit`, no physical QPU anywhere in this
  project) and is itself labeled "KERNEL ONLINE — NISQ Simulator Active"
  in the non-degraded case. A viewer could easily read "SIMULATION MODE"
  as "yes, this is the (expected) quantum circuit simulator" rather than
  its intended meaning, "this result is fabricated because a required
  artifact is missing." The disambiguating caption is small print in the
  sidebar, not adjacent to the risk score itself.
- Fix: rename the degraded-state badge to something unambiguous, e.g.
  "ARTIFACT MISSING — MOCK RESULT," and keep "NISQ Simulator" language
  reserved exclusively for describing the (expected, always-true) use of a
  classical quantum simulator backend.
- Demo impact: **Medium** — a real risk if a demo happens to run without
  the PCA artifact and a viewer conflates the two meanings.

**C4. Effusion label-quality issue is well-audited and excluded correctly —
positive finding, restated for completeness.**
- File: `RESULTS.md` §2, §5
- The ~43% (25/58) same-phrasing-mapped-to-opposite-labels issue is
  identified, quantified, and handled by exclusion (not imputation) in the
  primary reported comparison. `scripts/effusion_severity_rule.py`'s
  scaled-labeling exercise explicitly refuses to auto-label the coin-flip
  "mild/small" tier. No corrective action needed — flagged as a model of how
  the rest of the project's claims should be trusted, in contrast to §D/§H.

---

## D. Frontend/backend integration problems

*(See the frontend subagent's full report, folded in below; only the highest-
severity items are restated here with file/line detail — consult the
original for the complete per-file breakdown.)*

**D1. `PredictionResponse` (the real backend contract, `lib/types.ts:33-39`)
has no field for per-qubit quantum data, latency breakdown by stage, or a
distinct raw-slice image — yet the UI displays all three as if backed by
it.** See A3/B2 above.

**D2. No prominent on-screen mock/live disclosure.**
- File: `frontend/components/workstation/TriageCard.tsx:57`,
  `frontend/components/workstation/CommandBar.tsx:73-93`
- The only in-UI disclosure is one small "Backend: Live/Preset" metric tile,
  equal visual weight to "Diagnosis" and "Latency," no color/icon emphasis.
  The `CommandBar` health indicator reflects `/health` reachability, not
  whether the *currently displayed* result is mock — "API Online" and a
  stale preset result can coexist on screen with no contradiction flagged.
- Contrast with `qknee/ui/dashboard.py`, which does this well: `backend`
  ("live"/"mock"/"api"/"cache-fallback/...") is surfaced in the sidebar,
  the Grad-CAM panel caption, and the exported report text (lines 688, 777,
  787, 1198) — the Streamlit surface sets the bar the Next.js surface
  should be held to.
- Fix: add a persistent, high-contrast banner/watermark distinguishing
  preset/mock from live results, matching what the Streamlit dashboard
  already does.

**D3. `gradcam_heatmap` typed non-nullable in `lib/types.ts:36` but the
backend can return an empty/absent heatmap** (`CachedFallbackBackend.predict`
in `extras/api/server.py:757` defaults to `""` when a cached case has none).
No null-guard exists before `MriViewport.tsx` builds a `data:image/png;base64,`
src from it — a genuinely empty heatmap would render a broken image with no
graceful fallback. **P1.**

**D4. `BenchmarksTable.tsx` (lines 4-17) hardcodes AUC/parameter-count
numbers as marketing copy on the landing page**, never fetched from
`/health`'s real `latency_benchmark` field even though that field exists and
is populated by `_latency_benchmark_status()` in `extras/api/server.py`.
Low severity (landing page, not the diagnostic workstation) but presented
under a "Verified Clinical Benchmarks" heading, which reads as stronger than
"unverified constant in a `.ts` file."

**D5. No runtime schema validation anywhere in `frontend/lib/api.ts`**
(`parseJsonOrThrow` does an unchecked `as` type assertion). A backend
response shape drift (e.g. Vercel proxy returning an error body shaped
differently than `PredictionResponse`) would flow straight into rendering
code with no guard. **P2.**

---

## E. Security problems

The quarantined `extras/api/auth.py` is, on inspection, a **notably
well-built** auth implementation for a hackathon project — Argon2id password
hashing (not bcrypt/MD5/plaintext), JWT with an explicit warning when the
insecure dev-default signing key is still active, per-IP rate limiting on
`/register`/`/login` via `slowapi`, role-gated radiologist self-registration
behind an invite code, and timing-attack awareness (explicitly documented,
not silently ignored) on the login path. The findings below are gaps, not a
verdict that this code is unsafe by default.

**E1. JWT falls back to a publicly-known, committed default signing key if
the operator forgets to set an env var.**
- File: `qknee/config/config.yaml:93`, `extras/api/auth.py:253-262`
- What's wrong: `_SECRET_KEY` falls back through `$QKNEE_JWT_SECRET_KEY` →
  `$SECRET_KEY` → the literal string
  `"INSECURE-DEV-ONLY-CHANGE-ME-VIA-QKNEE_JWT_SECRET_KEY-ENV-VAR"` committed
  in `config.yaml`. A `logger.warning` fires, but **the process still
  starts and issues real, verifiable tokens** signed with a key anyone who
  reads this public repo already knows.
- Why it matters: if `render.yaml`'s `QKNEE_JWT_SECRET_KEY: sync: false` is
  ever left unset on a real deploy, any attacker can forge a valid
  `radiologist`-role token without ever registering an account.
- Fix: make this a hard startup failure (raise, don't just warn) outside of
  an explicit `QKNEE_ALLOW_INSECURE_JWT_KEY=1`-style local-dev opt-in,
  mirroring the pattern already used for `_ALLOW_INMEMORY_DB` a few lines
  below in the same file (E1 is the one place this project's otherwise-good
  "refuse to silently degrade on something security-relevant" pattern isn't
  applied consistently — see E2 for where it *is* applied well).
- Demo impact: none for the current Streamlit-only judged demo (auth isn't
  in that path at all); relevant only if `extras/api` is ever redeployed.

**E2. (Positive, for contrast with E1) In-memory user-store fallback
correctly refuses to start rather than silently discarding accounts.**
- File: `extras/api/auth.py:396-434`
- `_ALLOW_INMEMORY_DB` must be explicitly set or the process raises
  `RuntimeError` rather than quietly running on a store that forgets every
  user on restart. This is the right pattern; E1 should follow it.

**E3. `radiologist` self-registration invite code, when set, is a shared
static secret with no rotation/expiry.**
- File: `extras/api/auth.py:271-279, 464-474`
- Minor — acceptable for "lightweight stand-in for admin approval" as
  documented, but worth noting it's a single long-lived shared secret, not
  a scoped/expiring invite token. **P2.**

**E4. CORS origins list is a fixed allowlist (good) but includes a
Streamlit Community Cloud URL that anyone can technically front differently
if that subdomain is ever released** (`config.yaml:82-87`). Low risk,
standard SaaS-URL-squatting caveat, not specific to this project. **P2, informational.**

**E5. Frontend auth session trust is entirely client-side after login.**
- File: `frontend/lib/auth-context.tsx` (per frontend audit) — token stored
  in a JS-readable, non-`Secure` cookie; `user.role` read from `localStorage`
  with zero server round-trip to re-validate on mount; `Token.expires_in_minutes`
  declared but never enforced client-side.
- Why it matters: client-side role gating (`canDiagnose = user?.role ===
  "radiologist"`) only controls which UI affordances render — this is fine
  **only if** `extras/api/server.py`'s `require_role(INFERENCE_ROLES)`
  dependency is the actual enforcement point (confirmed: it is —
  `/predict`, `/explain`, `/report` all gate on it server-side, so a
  hand-edited `localStorage` role can unlock UI buttons but not bypass
  server-side authorization). Restated here so the two facts (weak client
  trust + strong server enforcement) are read together rather than the
  client-side weakness being mistaken for an actual bypass.

**E6. `users.json` at `extras/api/users.json` is a vestigial empty file**
(`{"users": {}}`) that doesn't correspond to any code path — `auth.py`'s
real user store is SQLAlchemy/SQLite (`qknee_users.db`), not a JSON file, and
no `LocalFileUserRepository` class exists anywhere in the codebase despite
`config.yaml:104`'s comment (`local_users_path: ... # LocalFileUserRepository
store, used when database_url == ""`) describing one. **Dead/aspirational
config — see F-series below.**

---

## F. Deployment problems

**F1. `config.yaml`'s `storage.local_users_path` documents a
`LocalFileUserRepository` class that does not exist in this codebase.**
- File: `qknee/config/config.yaml:104`, `qknee/config/loader.py:200-201`
```yaml
local_users_path: "qknee/artifacts/users.json"  # LocalFileUserRepository store, used when database_url == ""
```
- Reality: `extras/api/auth.py`'s actual empty-`DATABASE_URL` fallback is a
  local **SQLite file** via SQLAlchemy (`sqlite:///./qknee_users.db`), not a
  JSON file, and `local_users_path`/`StorageConfig.local_users_path` is
  parsed by `loader.py` but **never read by any consumer** — grep confirms
  zero call sites. This is a config field and a class name that describe a
  storage mechanism that was either replaced (SQLite) or never built.
- Fix: either implement the documented `LocalFileUserRepository` or remove
  the dead config field/comment so it stops describing nonexistent
  behavior.
- Demo impact: none directly, but misleads anyone auditing where user data
  actually lives.

**F2. See A1/A2** — `vercel.json`/`render.yaml` reference stale
pre-quarantine module paths and the wrong requirements file.

**F3. `docker-compose.yml`'s `mlflow` service is wired (`MLFLOW_TRACKING_URI`
env var on both `api`/`ui`) but nothing in the pipeline calls MLflow** —
acknowledged explicitly in the compose file's own comment
("nothing in the pipeline calls MLflow yet"). Not a bug, just unused
infrastructure sitting in the compose file; low-priority cleanup.

**F4. `Dockerfile`/`docker-compose.yml` were not read in this pass in
sufficient depth to confirm the base image + `qknee.api.server` command
consistency beyond what's shown in F1/A1** — recommend a follow-up pass
specifically diffing `Dockerfile`'s `COPY`/`WORKDIR` against
`extras/api/server.py`'s actual import root before ever attempting to
resurrect this deployment path.

---

## G. Testing gaps

- `qknee/tests/` (testpaths per `pytest.ini`) has substantial, real coverage
  of the in-scope pipeline: `test_pipeline_runner.py` (681 lines),
  `test_gradcam.py` (372 lines), `test_multitarget_model.py`,
  `test_volumetric_aggregation.py`, `test_quantum_simulator_mocking.py`,
  `test_multiplane_ingestion.py`, `test_precomputed_cache.py` — this is
  meaningfully deeper than a typical hackathon test suite and was not
  further scrutinized line-by-line in this pass given time constraints;
  spot-checked file sizes/names only.
- **Zero automated test coverage for `frontend/`** — no jest/vitest/
  playwright/testing-library dependency in `package.json`, no test script.
  Every finding in §D/B/A3 (mock-data leakage, dead UI controls, type
  contract mismatches) would have been caught by even a minimal
  component-level test asserting `QuantumTelemetry` receives its data from
  a live API call in the `"live"` code path.
- `extras/tests/` (auth, API server, landing page, hybrid ONNX, RSNA
  submission) totals ~2,700 lines and is excluded from the default
  `pytest` run (`pytest.ini`'s `testpaths = qknee/tests`) — reasonable given
  the quarantine, but means **CI (if any) never actually exercises the auth
  security properties described in §E** unless someone runs
  `pytest extras/tests` explicitly. Recommend documenting that command in
  `extras/README.md` if the auth code is ever meant to be kept correct going
  forward.
- No test found (in either suite) asserting the **real-vs-mock
  distinguishability contract** itself — e.g. "a `CachedFallbackBackend`
  response's `backend` field always starts with `cache-fallback/`" or "the
  frontend never renders `qubitExpectations` from a preset when
  `source === 'live'`" (which is currently **false** per A3 — a
  regression test for A3 would have caught it at introduction).

---

## H. Documentation/claim problems

**H1. `frontend/` is entirely absent from `README.md`'s project structure
and from `extras/README.md`'s "what's here and why" inventory.**
- Neither document explains whether this Next.js app is: (a) a legacy UI
  predating the PRD-scoping decision, (b) a parallel effort meant to
  eventually replace Streamlit, or (c) abandoned. Given it targets the
  quarantined FastAPI backend by default (`https://qknee.onrender.com`),
  its status is genuinely ambiguous from the docs alone.
- Fix: add a paragraph to either README (or a `frontend/README.md`)
  stating its relationship to the judged PRD scope, one way or the other.

**H2. (Positive) `RESULTS.md`'s "RETRACTED" handling is a model of honest
reporting, not a problem — restated for completeness per the audit
checklist's explicit ask to check for this.**
- `README.md`'s "Reproducing the SSL-pretrained backbone" section and
  `RESULTS.md` §3 both explicitly document that an earlier SSL-pretraining
  comparison was retracted due to a preprocessing shortcut-learning bug
  (`Resize` instead of `RandomResizedCrop` let the model key off background
  padding instead of anatomy), name the retracted files
  (`_RETRACTED_original_unseeded_run.*`), and explain the fix. This is
  disclosed, not buried — no corrective action needed.

**H3. `README.md`'s "Key metrics" framing is careful and accurate against
`RESULTS.md`/`kaggle_benchmark_summary.json`** — spot-checked the 10-
condition and 9-condition tables against `RESULTS.md`'s identical
reproduction; numbers match verbatim between the two files, and the caveat
language ("directional, not statistically robust," "n=58") is consistent in
both places. No drift found.

**H4. `README.md`'s "Appendix: synthetic pipeline sanity check" table is
clearly and repeatedly labeled as not-a-real-benchmark** ("not measured on
real patient data," "should not be quoted as model performance") — correctly
distinguished from the real §"Key metrics" table above it. No finding here;
flagged only because distinguishing synthetic-sanity-check numbers from
real-data numbers was explicitly on the audit checklist and this repo does
it correctly.

---

## I. Recommended fixes, ranked

### P0 (fix before any demo — the first two are the most reputationally
dangerous findings in this audit, because they're independently checkable
by any reviewer who reads more than one file)
1. **C4a** — `BenchmarksTable.tsx`'s "0.884/0.912, Verified Clinical
   Benchmarks, Stanford MRNet" claims directly contradict this repo's own
   real evaluation artifacts (VQC macro-AUC 0.563 trailing classical
   baselines on real n=58 data; 0.407 on the mock smoke-test run). Remove
   or replace with the real, honestly-framed numbers before anyone shows
   both the landing page and `RESULTS.md` to the same audience.
2. **A3 / D1** — Either wire real per-qubit data into `PredictionResponse`
   and `QuantumTelemetry`, or remove/clearly relabel the quantum telemetry
   panel so it isn't presented as live per-image data.
3. **B2** — Give the frontend a real distinct raw-slice image asset,
   separate from the Grad-CAM overlay.
4. **B1** — Either wire the slice/plane controls to real data, or remove
   them until they do something.

### P1 (fix before any demo, low effort / high credibility payoff)
5. **D2** — Add a prominent, persistent mock/live indicator to the Next.js
   workstation, matching what `qknee/ui/dashboard.py` already does well.
6. **D3** — Null-guard `gradcam_heatmap` before building an `<img src>`.
7. **B3 / C4b** — Add a real MCL checkpoint (or remove the MCL gauge), and
   more broadly, make the ACL/meniscus random-weight-checkpoint fallback in
   `pipeline.py`/`qknee_model.py` fail loudly or surface a hard "degraded"
   flag instead of only logging a warning — this is a silent-on-fresh-clone
   risk, not just a permanent MCL-specific gap.
8. **C4c** — Rename the Streamlit "SIMULATION MODE" badge (artifact-missing
   fallback) to something that can't be misread as "expected quantum
   simulator backend, working normally."
9. **E1** — Make the insecure default JWT key a hard startup failure
   outside an explicit opt-in env var (mirror the existing
   `_ALLOW_INMEMORY_DB` pattern in the same file).
10. **A1 / A2 / F2** — Fix or delete the stale `vercel.json`/`render.yaml`
    deploy configs (wrong module path, wrong requirements file).

### P2 (cleanup, no demo impact)
11. **F1** — Remove or implement the phantom `LocalFileUserRepository`
    config field.
12. **D5** — Add minimal runtime response validation in `lib/api.ts`.
13. **G** — Add a frontend test suite; add a regression test for the
    real-vs-mock distinguishability contract (A3's exact failure mode).
14. **H1** — Document `frontend/`'s relationship to the judged PRD scope
    in the README.
15. **E3 / E4** — Note as accepted risk or harden invite-code
    rotation/expiry if `extras/api` auth is ever taken out of quarantine
    for real use.

---

*Everything in §C and §H marked "positive"/"no finding" is included
deliberately — an audit that only lists problems would misrepresent a
codebase that, in its judged-scope core, is unusually careful about not
overclaiming.*
