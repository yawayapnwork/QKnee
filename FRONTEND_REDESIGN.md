# Q-Knee Frontend — Hostile Teardown & Redesign Mandate

**Author posture**: incoming principal frontend engineer. Assume nothing prior
deserves to survive on the grounds that it already exists. Every judgment
below is sourced to a specific file and line, verified against the current
repository and the actual backend contract (`extras/api/server.py`'s
`PredictionResponse`/`HealthResponse`), not against what the code *says* it
does in its own comments.

**No source code was touched to produce this document.**

---

## Verdict

**The data layer survives. The presentation layer does not.**

The TypeScript contract (`lib/types.ts`), the viewer math (`lib/viewer.ts`),
the provenance classification (`lib/provenance.ts`), and the API client
(`lib/api.ts`) are honest, well-typed, and mostly correct — this is real
engineering, not decoration. That is exactly why it's worth saying plainly:
**the UI built on top of that honest data layer actively undersells and
sometimes contradicts it.** The component layer was built to *look* like a
hackathon "quantum medical AI" demo — glow shadows, a permanent scanning
laser line, a `text-gradient` logo, `backdrop-blur` baked into the base
`Card` primitive — instead of an instrument a radiologist or a judge could
trust. Two components actively lie by omission (`ReportExport.tsx` ignores
the provenance system entirely; `PipelineVisualizer.tsx` advertises "3D
sagittal/coronal decoding" that the backend explicitly refuses to produce).

This is not a "polish pass." It is a systematic rebuild of the presentation
layer on top of a data layer that gets to keep its job.

---

## Part 1 — The indictment (evidence, not opinion)

### 1.1 Misleading visualizations

- **`frontend/components/landing/PipelineVisualizer.tsx:7-8`** — Step 1 of
  the public landing page's "Hybrid Inference Pipeline" reads: *"Volumetric
  MRI Ingestion — 3D sagittal / coronal DICOM & NumPy volume decoding."*
  This is false. `extras/api/server.py:809-816` documents, in its own code
  comments, that a typical single-series MRI upload cannot honestly produce
  independent Coronal/Sagittal views and marks both `available: false` for
  exactly that reason. The single most-visible claim on the marketing page
  contradicts the backend's own honesty guarantee. **This is the worst
  finding in the entire frontend.**
- **`MriViewport.tsx:18-21, 225-227`** — "Window / Level" buttons display
  fabricated `WL500/WW2000`-style DICOM windowing numbers next to a button
  that actually applies a hardcoded CSS `contrast()/brightness()` filter
  with no relationship to those numbers (`MriViewport.tsx:47`). A reader who
  knows what WL/WW mean will assume real pixel-intensity windowing is
  happening. It is not.
- **`MriViewport.tsx:136`** — a permanent, infinite, decorative
  `animate-scan-line` sweeps across the MRI image at all times, including
  when no image is loaded, when the result is a MOCK/FALLBACK, and when the
  viewer is empty. It communicates nothing about system state; it exists to
  look like a sci-fi scanner.

### 1.2 Fake / cosmetic controls

| Control | File | Verdict |
|---|---|---|
| Plane selector | `MriViewport.tsx:58-61` | Real — wired to `resolvePlane`. |
| Slice slider | `MriViewport.tsx:154-162` | Real — wired to `clampSliceIndex`. |
| Grad-CAM opacity slider | `MriViewport.tsx:199-206` | Real — wired to inline `opacity` style. |
| Grad-CAM on/off toggle | `MriViewport.tsx:176-190` | Real, but implemented as a raw `<button>` styled as a switch with **no `role="switch"`, no `aria-checked`** — invisible to a screen reader. |
| **Window/Level presets** | `MriViewport.tsx:210-231` | **Cosmetic-adjacent.** A real (if crude) filter change hiding behind fabricated clinical numbers. Worse than a no-op. |
| **"Export PDF" button** | `ReportExport.tsx:74-82` | **Fake.** Calls `window.print()`. No PDF is generated. Mislabeled. |

### 1.3 Fake-looking / duplicated / inconsistent telemetry and provenance

This is the deepest structural problem in the frontend, and it was never
caught by prior audits because it spans multiple files that each look fine
in isolation.

- **Three parallel "what kind of result is this" fields exist on one
  object simultaneously** (`lib/types.ts:211-221`, `DiagnosticResult`):
  `source: "live" | "mock"` (2-state, legacy), `backend: string` (a raw,
  free-form tag with no type safety), and `provenance: ProvenanceInfo`
  (5-state, the actual current system: LIVE / PRECOMPUTED DEMO /
  MOCK-FALLBACK / CACHED / PROXY). Three sources of truth for one question.
- **Two separate, non-overlapping provenance *type systems* exist for
  quantum telemetry specifically**: `QuantumTelemetryProvenance` (`lib/
  types.ts:201`, values `"live" | "precomputed-demo" | "unavailable"`,
  hyphenated) is a completely different enum from `Provenance` (`lib/
  types.ts:95`, values `"live" | "precomputed_demo" | "mock_fallback" |
  "cached" | "proxy"`, underscored). `QuantumTelemetry.tsx` renders its own
  badge from the first system; `TriageCard.tsx:51` renders `ProvenanceBadge`
  from the second system, directly above it. **A single screen can show two
  differently-worded, differently-styled provenance badges for the same
  result, one of which (the quantum one) cannot even express
  `mock_fallback`.**
- **`ReportExport.tsx:8-12, 28, 39`** — the exported report (the one
  artifact a judge or clinician might actually save) builds its own
  **third, independently hardcoded copy** of the quantum-provenance label
  map, and cites `result.backend`/`result.source` in its header — never
  `result.provenance`. The most authoritative, most recently-hardened
  provenance system in the codebase (`lib/provenance.ts`, built specifically
  to fix AUDIT.md P1 #5/#7) **is invisible in the one document a user takes
  away from the session.**
- **`frontend/lib/types.ts`'s `HealthResponse` is missing the backend's
  `model_status` field** (present in `extras/api/server.py:508`). The
  Next.js side has no way to show "MCL unavailable" / "ACL checkpoint
  missing" the way the Streamlit sidebar does — a real, silent asymmetry
  between the two interfaces this project claims share one honesty
  standard.

### 1.4 Excessive dashboard decoration / "sci-fi gimmicks"

- Base `Card` primitive (`components/ui/Card.tsx:7`) applies
  `backdrop-blur-sm` **unconditionally to every card in the product** —
  triage card, pipeline-visualizer steps, everything. Glassmorphism as a
  default, not a deliberate choice for one hero moment.
- `boxShadow.glow` / `glow-rose` (`tailwind.config.ts:20-23`) — a 24px teal
  glow baked into the design tokens and applied to every primary button
  (`Button.tsx:9`) and the landing CTA (`Hero.tsx:30`). This is the "more
  glowing cards" aesthetic the redesign brief explicitly asks to kill.
- **Two independently-defined, never-consolidated "scanning line"
  animations** exist: `globals.css:32-41` (`@keyframes scan-line`, 3s
  ease-in-out, actually used) and `tailwind.config.ts:24-33` (`keyframes.
  scan`, 2.4s linear, **dead — grep confirms zero usages**). Nobody
  noticed the second one was orphaned. That is itself a symptom: decoration
  was added faster than anyone could track what decoration existed.
- `text-gradient` teal→cyan wordmark (`globals.css:25-30`), gradient logo
  tile (`Navbar.tsx:19`), gradient CTA button, gradient SVG connector line
  between pipeline steps (`PipelineVisualizer.tsx:56-66`) — the entire
  visual identity of this "clinical workstation" is teal/cyan gradients and
  glow. Nothing in premium radiology software (Osirix, Horos, 3D Slicer,
  Philips IntelliSpace) looks like this. It looks like a crypto-dashboard
  template with medical copy pasted in.

### 1.5 Weak information hierarchy / redundant cards

- `TriageCard.tsx` stacks, in order: severity badge → not-a-radiologist
  warning → provenance badge → untrustworthy-result warning → risk gauge →
  a 2-cell metric grid (Diagnosis, Latency) → quantum telemetry block (with
  its *own* internal provenance badge) → report export buttons. **Eight
  visually distinct zones in one card, no clear "this is the one number
  that matters" moment** — the risk gauge (the actual clinical output) is
  buried in the middle, competing for attention with two separate
  provenance systems.
- `Metric` cells (`TriageCard.tsx:86-93`) show "Diagnosis" and "Latency" as
  equal-weight boxes. Latency is operational telemetry; Diagnosis is the
  clinical output. They should never share visual weight.

### 1.6 Oversized/generic landing page

- `Hero.tsx` is a single full-viewport-height centered block: badge → h1 →
  paragraph → two buttons → disclaimer — the exact shape of every AI SaaS
  landing page built since 2023. Nothing about its layout communicates
  "clinical research tool" versus "consumer AI product launch."
- The unhedged headline (*"Q-Knee Diagnostic Platform" / "Accelerating
  orthopedic knee MRI triage"*) sits large and bold; the actual disclaimer
  (*"Investigational research prototype... not a clinical validation"*)
  is `text-xs text-slate-600` below the CTA row — the least visually
  weighted text on the page carries the single most legally/ethically
  important sentence on the page.

### 1.7 Accessibility — not a gap, an absence

Verified directly: **zero `aria-*` attributes and zero `role` attributes
anywhere in the entire `frontend/` tree.** Every `<label>` element that
exists (`AuthModal.tsx`, `MriViewport.tsx`) is visually adjacent to its
input but has no `htmlFor`/`id` pairing, so no screen reader or
label-click-to-focus behavior works. The custom toggle switch, the custom
sample-case dropdown (`CommandBar.tsx`'s `SampleSelector`), and every
slider are unreachable-by-meaning to assistive technology. For a tool whose
own pitch is "clinical," this is disqualifying, not a nice-to-have.

### 1.8 Responsiveness — landing yes, workstation no

Confirmed by direct grep: `components/landing/*` uses `sm:`/`lg:`
meaningfully throughout. **`components/workstation/*` — the actual
product — has zero responsive classes anywhere except one grid declaration
at the page root** (`app/workstation/page.tsx:73`). `Gauge.tsx` is a
hardcoded `h-36 w-36`. On a tablet (the single most likely real-world
device for a "clinical workstation" demo) the two-column workstation layout
does not adapt gracefully below `lg:` — it just stacks two full-width,
non-optimized columns.

---

## Part 2 — Component classification

No component survives by default. Each is judged on whether it earns its
place in the rebuilt product.

| Component | File | Verdict | Why |
|---|---|---|---|
| `lib/types.ts` | data | **REBUILD (types only, not the data)** | The Python↔TS contract itself is sound; the *duplication* (`source`/`backend`/`provenance`, two provenance enums) must be collapsed to one field, one enum. |
| `lib/provenance.ts`, `lib/viewer.ts`, `lib/api.ts` | data | **KEEP** | Correct, well-tested, single-source-of-truth logic. This is the part of the frontend that deserves to survive unmodified. |
| `lib/quantum-telemetry.ts` | data | **MERGE into `lib/provenance.ts`** | A second, redundant provenance vocabulary for one sub-feature. Delete the type, fold quantum-specific fields (`expectations`, `nQubits`, `device`) into the one `ProvenanceInfo`-adjacent shape. |
| `lib/mock-data.ts` | data | **KEEP, rename** | Logic is correct (explicit `isFailedLiveFallback` flag). Rename `PresetCase` → `DemoCase` everywhere in the UI layer — "preset" reads as internal jargon; "Demo Case" is what a viewer needs to understand instantly. |
| `MriViewport.tsx` | viewer | **REBUILD** | Real control logic underneath, but: kill the scan-line animation, kill the fabricated WL/WW numbers, add real accessibility roles, add responsive behavior, and restructure so the base image and Grad-CAM overlay are unmistakably two layers (a visible layer-stack affordance, not just an opacity slider). |
| `TriageCard.tsx` | result | **REBUILD** | Correct data, wrong hierarchy. Split into a single dominant result module + a collapsed/secondary telemetry section (see Part 3). |
| `QuantumTelemetry.tsx` | telemetry | **REBUILD** | Merge its provenance display into the one shared badge system; keep the per-qubit bar visualization (it's honest and clear) but restyle to look like instrument output, not a crypto-ticker (currently cyan/rose bars with `+`/`-` signs read like a trading dashboard). |
| `ProvenanceBadge.tsx` | provenance | **KEEP, promote** | This is the correct pattern (one shared component, one shared data shape). Make it the *only* provenance-rendering component in the app — everything else (`QuantumTelemetry`, `ReportExport`) must consume it, not reimplement it. |
| `ReportExport.tsx` | report | **REBUILD** | Fix "Export PDF" (either implement it via a print-optimized stylesheet + real trigger, or rename the button to what it does — "Print / Save as PDF"). Rewrite `buildMarkdown` to source provenance from `result.provenance`, not the legacy fields. |
| `CommandBar.tsx` | nav | **SIMPLIFY** | Health indicator + case switcher + auth state in one bar is reasonable information architecture; the custom `SampleSelector` dropdown should become a native `<select>` or a proper `role="listbox"` combobox, not a hand-rolled `<div>` menu with no keyboard support. |
| `AuthModal.tsx` | auth | **SIMPLIFY** | Functionally fine; fix label/input association (`htmlFor`/`id`), fix focus trapping (none currently), move from a bespoke modal to a proper dialog primitive with `role="dialog"`/`aria-modal`. |
| `Hero.tsx` | landing | **REBUILD** | Not because it's broken — because its entire visual grammar (centered hero, gradient wordmark, glow CTA, tiny disclaimer) is the generic-SaaS template this brief explicitly rejects. See Part 3, "Landing page strategy." |
| `PipelineVisualizer.tsx` | landing | **REBUILD** | The false "3D sagittal/coronal" claim must be corrected as content, and the "four equal cards in a row" treatment should become an actual annotated pipeline diagram that shows real tensor shapes, not four generic icon cards. |
| `BenchmarksTable.tsx` | landing | **SIMPLIFY** | Content is honest and already correctly hedged (verified against `kaggle_benchmark_summary.json` — exact match). Visual treatment (a plain table in a card) is fine; just needs to sit inside the new information architecture, not be a bolt-on section. |
| `Navbar.tsx` | landing | **KEEP, minor rework** | Structurally sound; drop the gradient logo tile for a flat mark consistent with the new color system. |
| `Card.tsx`, `Badge.tsx`, `Button.tsx` (ui primitives) | design system | **REBUILD** | These are the actual source of the "SaaS dashboard" feel — `backdrop-blur` baked into `Card`, gradient+glow baked into `Button`'s primary variant. Rebuild as the new design system (Part 3) with zero unconditional blur/glow. |
| `Gauge.tsx` | ui | **KEEP, restyle** | The SVG donut math is fine and honest (real `risk_score`, real severity color). Restyle to fit the new, flatter visual language — no other change needed. |
| `app/layout.tsx`, font loading | infra | **KEEP** | Inter + JetBrains Mono via `next/font` is a correct, professional choice already. Not everything here is bad. |
| `auth-context.tsx` | state | **KEEP** | Simple, correct React Context; no need for a state library at this app's current size. Don't add Redux/Zustand to seem more serious — that would be decoration too. |

---

## Part 3 — The new frontend

### A. Information architecture

Three top-level surfaces, not four:

1. **Landing** (`/`) — what this is, what it is not, and a way in.
2. **Workstation** (`/workstation`) — the single working surface: upload,
   view, diagnose, export.
3. **Methods** (`/methods`, new) — the pipeline diagram, the benchmark
   table, and the model-provenance/checkpoint status, pulled *out* of the
   landing page. A landing page's job is orientation in 20 seconds; a
   methods page's job is letting a skeptical reviewer verify claims. These
   are different audiences and belong on different pages.

Delete the idea of the landing page as a "pitch deck" — it becomes a
one-screen orientation, not a scroll-through marketing site.

### B. New page structure

```
/                     Landing — identity, disclaimer, one CTA into /workstation, one link to /methods
/workstation          The instrument: viewer + result, nothing else
/methods              Pipeline diagram, benchmark table, model/checkpoint status, dataset provenance
```

No `/login` route needed — auth stays a modal triggered from the
workstation's own gate (see O below), not the landing page. A visitor
should never be asked to sign in before they've seen what the tool does.

### C. Navigation system

One persistent top bar across all three pages: wordmark (flat, no
gradient) · page links (Workstation / Methods) · session state (signed out:
"Sign In"; signed in: name + role + sign out). No sticky secondary nav
inside the workstation — the workstation's own command row (case selector +
provenance status) replaces it.

### D. Design system — philosophy

Flat, high-contrast, instrument-panel aesthetic. Every visual effect must
answer "what does this communicate," not "does this look advanced." Default
posture: **no gradient, no glow, no blur**, unless a specific, named
interaction earns it (see Motion, below).

### E. Typography system

Keep Inter + JetBrains Mono (this was already the right call) but assign
them *roles*, not just a sans/mono split:

- **Inter** — all prose, labels, navigation, body copy.
- **JetBrains Mono** — every number that is a measurement or an
  identifier: risk percentages, qubit expectations, latency, slice
  indices, checkpoint hashes, case IDs. Mono numerals are how instrument
  panels signal "this is data, not marketing copy" — use this
  consistently and nowhere else.

Scale (rem, four steps, not eight): `0.75` (labels/captions) / `0.875`
(body) / `1.25` (section headers) / `2` (the one number that matters per
screen — the risk gauge's percentage, and only that). No `text-4xl`/`text-6xl`
hero type anywhere in the product — a workstation does not need a billboard.

### F. Color tokens

Replace the teal/cyan gradient identity with a **restrained, purposeful**
palette. Color must mean something, not decorate:

| Token | Value (dark base) | Meaning |
|---|---|---|
| `surface-0` / `surface-1` / `surface-2` | `#0b0f14` / `#11161d` / `#171d26` | Background layering, flat, no blur |
| `text-primary` / `text-muted` / `text-faint` | `#e6e9ee` / `#8b95a3` / `#565f6c` | Prose hierarchy |
| `accent` | `#2dd4bf` (single teal, used sparingly: active state, primary action) | Interactive affordance only |
| `status-live` | `#22c55e` | Provenance: LIVE only |
| `status-demo` | `#eab308` | Provenance: PRECOMPUTED DEMO / CACHED |
| `status-fallback` | `#ef4444` | Provenance: MOCK/FALLBACK — never shares a hue with anything else |
| `status-proxy` | `#64748b` | Provenance: PROXY |
| `severity-normal/indeterminate/urgent` | `#22c55e` / `#eab308` / `#ef4444` | Clinical severity — **intentionally the same hue family as provenance status**, because both answer "how much should I trust this," and a reader should learn the mapping once |

One accent color, not a gradient pair. Status colors are never used
decoratively — if a color appears, it is because it is answering a
trust/severity question.

### G. Spacing system

8px base unit, 6 steps (`4/8/16/24/32/48`px). Cards get one consistent
internal padding (`24px`), not the current mix of `px-5 py-4`,
`px-4 py-3`, `p-6` scattered per-component. One spacing scale, applied
consistently, is itself a hierarchy signal — inconsistent spacing is why
the current UI feels like several people's work stitched together (which,
functionally, it is: layered feature-fixes over many sessions).

### H. Component hierarchy

```
DesignSystem (tokens, no logic)
 └─ Primitives: Surface, Text, StatusDot, Button, Field, Slider, Toggle
     └─ Composed: ProvenanceIndicator (the ONE provenance renderer), ResultGauge,
                   ImageLayerStack (base+overlay viewer core), MetricRow
         └─ Screens: WorkstationViewer, ResultPanel, MethodsPipelineDiagram,
                      BenchmarkTable, LandingHero
```

Rule: no screen-level component may format a provenance/status string
itself. Every status string flows through `ProvenanceIndicator`, full stop
— this is the direct fix for §1.3's three-parallel-systems problem.

### I. Motion principles

Motion must indicate a **state change**, never run on a timer.

- Allowed: gauge arc animating to its new value on data arrival (`700ms`
  ease-out — keep `Gauge.tsx`'s existing curve, it's correct); a skeleton
  shimmer only while genuinely loading; a toast sliding in for a
  transition event (e.g. FAIL-10's silent-fallback problem — surface it
  as a real, brief toast, not a permanent animation).
- Banned: any `infinite` keyframe animation. The scan-line dies. If the
  team wants a "processing" indicator, it must stop the instant processing
  stops — it is a state indicator, not wallpaper.

### J. Responsive strategy

Workstation layout: `lg:` two-column (viewer | result) exactly as now, but
below `lg:` the result panel becomes a **bottom sheet accessible via a
persistent summary bar** (risk % + provenance status, always visible),
not two full-width stacked columns. Every control in `MriViewport` gets a
minimum 44px touch target on `<md:`. `Gauge` becomes a `clamp()`-sized SVG,
not a fixed 144px box.

### K. Accessibility strategy

Non-negotiable baseline before this ships again:
- Every `<label>` gets `htmlFor` matched to its input's `id`.
- The Grad-CAM toggle becomes a real `<Switch role="switch" aria-checked>`.
- The plane selector becomes a `role="tablist"`/`role="tab"` group (it is,
  functionally, a tab set).
- The case selector becomes a native `<select>` or a `role="combobox"` +
  `role="listbox"` pair with full keyboard support.
- All icon-only buttons get `aria-label`.
- Modal (`AuthModal`) gets `role="dialog"`, `aria-modal="true"`, and a
  real focus trap.
- Color is never the only signal for provenance/severity — every status
  badge pairs color with a text label and an icon (this is already true
  for `ProvenanceBadge`; extend the rule everywhere).

### L. MRI viewer UX

Reframe as an **image-layer stack**, not "a picture with sliders bolted
on": a persistent, visible layer list (Base MRI · Grad-CAM Overlay) with
per-layer visibility + opacity, so the base/overlay separation the data
layer already guarantees becomes *visually self-evident*, not something
you infer from a toggle switch. Plane tabs show a disabled state with an
inline reason (already correct in `resolvePlane`/`PlaneInfo` — just needs
the tooltip promoted to always-visible microcopy, not a hover-only
`title=`). Delete Window/Level entirely until it does real pixel
remapping — a missing feature is honest; a fake one is not.

### M. Quantum telemetry UX

Reposition as **instrument output**, not a trading-app ticker. Per-qubit
bars stay (they're a genuinely clear encoding of signed values), but:
drop the `+`/`−` colored bars in cyan/rose (too close to "gain/loss") in
favor of a single neutral hue with a center-zero baseline, label each qubit
by index only (no icon), and route the provenance line through the one
shared `ProvenanceIndicator` instead of `QuantumTelemetry.tsx`'s own badge.

### N. Explainability UX

Grad-CAM overlay gets an explicit, always-visible caption: which slice/
plane it was computed for (already available as data —
`gradcamPlane`/`gradcamSliceIndex` — currently only surfaced as a
parenthetical warning when viewing a *different* slice). State the scope
of the explanation up front, not as a caveat you only see when you've
already scrolled away from it.

### O. Prediction/result UX

One result module, top to bottom, no competing cards:
1. Provenance status (one line, one component).
2. The number that matters (risk gauge) — the single largest visual
   element on the panel.
3. Diagnosis label, inline with the gauge, not a separate equal-weight box.
4. Latency + case metadata — demoted to a small caption row, not a
   metric card.
5. Quantum telemetry — collapsed by default behind a disclosure ("View
   circuit telemetry"), because it's supporting evidence, not the headline.
6. Grad-CAM caption/scope note.
7. Export actions.

Un-authenticated viewers still see this whole structure against demo data
— clearly labeled PRECOMPUTED DEMO — rather than the current pattern of an
amber warning banner *plus* an otherwise-unchanged card.

### P. Demo/live provenance UX

One `ProvenanceIndicator`, one enum (`Provenance` from `lib/provenance.ts`,
after `QuantumTelemetryProvenance` is deleted), rendered in exactly one
place per screen, always visible without scrolling, always paired with the
model-source and quantum-execution sub-labels already computed by the
backend. MOCK/FALLBACK gets a full-width, non-dismissable band (not just a
badge) across the top of the result module — it should be structurally
impossible to see a mock result formatted identically to a live one.

### Q. Landing page strategy

Replace the full-viewport hero with a compact, single-screen orientation:
wordmark + one sentence of what this is + one sentence of what it is not
(same visual weight, side by side, not headline-vs-footnote) + one primary
CTA ("Open Workstation") + one secondary link ("Methods & Benchmarks").
Move the pipeline diagram and benchmark table to `/methods` entirely. A
landing page that takes four full-height sections to say "this is a
research prototype, click here" is optimizing for scroll depth, which is
a marketing-site metric, not a credibility signal.

### R. Benchmark strategy

Keep the honest numbers and the disclaimers (`BenchmarksTable.tsx`'s
content is correct today). Move it to `/methods`, alongside — not
separate from — the model-provenance/checkpoint-status panel (the
`model_status` field currently missing from the TS types, per §1.3), so a
reviewer sees "here are the numbers" and "here is exactly which model
produced them, trained or not" in one place, not two.

### S. Error/loading states

- **Loading**: replace the generic centered spinner with an inline skeleton
  of the result module's actual shape (gauge outline, label placeholders)
  so the layout doesn't jump when data arrives.
- **Empty**: workstation with no case selected and no upload shows a single
  clear instruction state, not a diagnosed result on synthetic noise (this
  mirrors FINAL_AUDIT.md FAIL-1's dashboard-side finding — the frontend
  does not currently have this bug, since it requires an explicit
  upload/case-select action; keep it that way, do not add a "demo volume"
  auto-load to the Next.js side).
- **Error**: failed live inference shows a single, dismissable error
  banner **and** the fallback result is stamped MOCK/FALLBACK via the one
  shared indicator (this part is already correct in `page.tsx` — preserve
  the behavior, rebuild only the visual treatment).

---

## Part 4 — Screen-by-screen wireframes (text)

### Landing (`/`)

```
┌─────────────────────────────────────────────────────────────┐
│ ▮ Q-Knee                              Workstation   Methods │  <- flat nav, no gradient
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   Q-Knee                                                     │
│   Hybrid ResNet18 → PCA → 4-qubit VQC knee MRI triage         │
│                                                               │
│   ┌───────────────────────────┬───────────────────────────┐ │
│   │ What this is               │ What this is not           │ │  <- EQUAL weight, side by side
│   │ A research prototype for   │ Not a certified medical    │ │
│   │ ACL/meniscal risk triage,  │ device. Not validated for  │ │
│   │ real RSNA Knee eval, n=58. │ clinical use. Confirmatory │ │
│   │                             │ radiologist review required│ │
│   └───────────────────────────┴───────────────────────────┘ │
│                                                               │
│         [ Open Workstation ]      Methods & Benchmarks →      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```
One screen. No scroll required to understand what this is.

### Workstation (`/workstation`)

```
┌─────────────────────────────────────────────────────────────┐
│ ▮ Q-Knee   Case: [Sample 02 ▾]        ● API Online  Sign In  │
├───────────────────────────────┬───────────────────────────────┤
│ AXIAL │ CORONAL* │ SAGITTAL*   │ ┌─ PRECOMPUTED DEMO ────────┐│
│  *disabled, inline reason      │ └────────────────────────────┘│
│ ┌───────────────────────────┐ │                                │
│ │                           │ │        ╭───────────╮           │
│ │      [ MRI slice ]        │ │        │   8.1%    │  <- one   │
│ │                           │ │        │  gauge    │     number│
│ │ Layers: ☑ Base  ☑ Overlay │ │        ╰───────────╯           │
│ └───────────────────────────┘ │  Normal · 12ms                │
│ Slice 12/24  ────●───────      │                                │
│ Overlay opacity ──●───────      │ ▸ Circuit telemetry (collapsed)│
│                                 │ ▸ Grad-CAM scope: axial/12     │
│ [ Upload .dcm / .npy ]          │ [Export .md]  [Print / PDF]    │
└───────────────────────────────┴───────────────────────────────┘
```
Compare to today: no scan-line, no fabricated WL/WW, one provenance
indicator (top-right of result column, not duplicated inside telemetry),
result module has one dominant number instead of eight competing zones.

### Methods (`/methods`, new)

```
┌─────────────────────────────────────────────────────────────┐
│ ▮ Q-Knee                              Workstation   Methods │
├─────────────────────────────────────────────────────────────┤
│ Pipeline                                                     │
│ DICOM/.npy → ResNet18 (512d) → PCA(4) → 4-qubit VQC → Grad-CAM│
│  [annotated diagram with real tensor shapes at each arrow]    │
│                                                               │
│ Model status                                                  │
│  Primary (unified head)         ● Available   checkpoint a1b2 │
│  ACL                            ○ Unavailable  no checkpoint  │
│  Meniscus                       ○ Unavailable  no checkpoint  │
│  MCL                            ○ Unavailable  research-only  │
│                                                               │
│ Benchmark (RSNA Knee, n=58, macro-AUC, 9 conditions)          │
│  ResNet-18 linear probe          0.6574                       │
│  ResNet-18 → RBF SVM             0.5596                       │
│  Hybrid VQC                      0.5579   [Hybrid]             │
│  Hybrid VQC trails both classical baselines — no quantum       │
│  advantage demonstrated at this sample size.                  │
└─────────────────────────────────────────────────────────────┘
```
This is the page a hostile judge actually wants: claims, numbers, and
model provenance in one place, with nothing hidden behind marketing copy.

---

## Closing note

Nothing here requires new infrastructure, a charting library, or a state
manager — the fix is almost entirely *subtractive*: delete the scan-line,
delete the fake WL/WW numbers, delete two of the three provenance systems,
delete the fake PDF button's false label, delete the landing page's
false 3D claim, delete `backdrop-blur`/`glow` as defaults. What's left,
tightened into one design system with real accessibility semantics, is
already most of a credible clinical research workstation. The team did not
need more components. It needed fewer, more honest ones.
