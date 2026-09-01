"""
Standardized, professional clinical radiology PDF report generation for
Q-Knee predictions.

Compiles the analyzed MRI slice, its risk-targeted Grad-CAM overlay,
patient/study metadata, per-condition tear-risk scores, and (when supplied)
the 4-qubit VQC's raw quantum measurements into a structured, multi-section
report — the source for the Streamlit dashboards' "Download Report" button
(`qknee.ui.dashboard.render_report_download`) and for
`qknee/artifacts/demo_radiology_report.pdf`.

Five sections, laid out across up to two letter-size pages:
    1. Header       - clinic/system branding, generated timestamp, and an
                       anonymized Study ID (never the patient's name/DOB).
    2. Clinical      - overall risk score, predicted condition (Normal /
       Impression      ACL Tear / Meniscal Tear), and a confidence interval.
    3. Visual        - side-by-side original MRI slice + Grad-CAM overlay.
       Evidence
    4. Quantum       - per-qubit Pauli-Z expectation values and (when
       Feature          supplied) the trained readout layer's per-qubit
       Attribution      impact weights / weighted contributions.
    5. Disclaimer    - a standard automated NISQ-screening-AI disclosure
                       notice, plus a timestamped signature placeholder.

RESEARCH PROTOTYPE — every generated report carries the Section 5
disclaimer; this module produces a demo/prototype artifact, not a
validated diagnostic report, the "clinic branding" is a placeholder
name (no real hospital/institution is implied), and the "digital
signature" is a placeholder hash, not a cryptographic attestation.

Built directly on a `reportlab.pdfgen.canvas.Canvas` (rather than a
`SimpleDocTemplate` flowable story): every element is placed at an
explicit y-coordinate computed top-down within each page, and page breaks
happen only at the two fixed points this module controls itself (never
from flowable content silently overflowing), so the report is always
exactly two pages — deterministic, not "however many pages the content
happens to need." `Table`/`Paragraph` flowables are still used for
anything genuinely tabular/wrapping text, drawn onto the canvas via
`wrapOn`/`drawOn`.

Two entry points:
    - `generate_radiology_report(output_path, ...)` - returns raw PDF bytes
      (for `st.download_button(data=...)`), and also writes them to
      `output_path` when one is given (e.g.
      `qknee/artifacts/demo_radiology_report.pdf`) — pass `output_path=None`
      for an in-memory-only report.
    - `generate_radiology_text_snippet(...)` - a short plain-text summary
      of the same prediction, for callers that don't need a full PDF.
"""

from __future__ import annotations

import os

# Set before any matplotlib import — this module renders its charts/tables
# directly with reportlab, not matplotlib, but is imported lazily by
# `qknee.api.server`'s /report handler alongside the rest of that
# request's (matplotlib-free today, but chain-adjacent) import graph; a
# plain env-var write costs nothing and imports nothing, so setting it
# unconditionally here is free insurance against a slow cold font-cache
# build if matplotlib ever ends up imported nearby. `setdefault` so an
# operator-supplied override wins.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import io
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from qknee.config.logging_config import get_logger

logger = get_logger(__name__)

PAGE_WIDTH, PAGE_HEIGHT = LETTER
MARGIN = 0.6 * inch
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

# Demo/placeholder branding — this module implies no real hospital or
# clinical institution; override via metadata["clinic_name"] if a specific
# deployment needs its own (still placeholder, non-legal) letterhead text.
DEFAULT_CLINIC_NAME = "Q-KNEE QUANTUM DIAGNOSTICS CENTER"
DEFAULT_CLINIC_SUBTITLE = "Quantum-Assisted Musculoskeletal MRI Screening"

# Risk-tier thresholds/colors — matches qknee.ui.dashboard.render_risk_gauge's
# LOW/MODERATE/HIGH bands, so the report and the live dashboard agree.
RISK_LOW_MAX = 0.33
RISK_MODERATE_MAX = 0.66
_TIER_COLORS = {
    "LOW": colors.HexColor("#2ECC71"),
    "MODERATE": colors.HexColor("#F5A623"),
    "HIGH": colors.HexColor("#E74C3C"),
    "N/A": colors.grey,
}
_INK = colors.HexColor("#16222A")

# Heuristic confidence-interval half-width used when a caller doesn't
# supply an explicit `acl_risk_ci`/`meniscus_risk_ci`/`overall_risk_ci` —
# see `_confidence_interval`'s docstring for why this is a heuristic, not a
# statistically derived interval.
DEFAULT_CI_MARGIN = 0.08


# --------------------------------------------------------------------------- #
# Small formatting / clinical-derivation helpers
# --------------------------------------------------------------------------- #
def _risk_tier(risk: Optional[float]) -> str:
    if risk is None:
        return "N/A"
    if risk >= RISK_MODERATE_MAX:
        return "HIGH"
    if risk >= RISK_LOW_MAX:
        return "MODERATE"
    return "LOW"


def _format_percent(risk: Optional[float]) -> str:
    return "N/A" if risk is None else f"{risk * 100:.1f}%"


def _format_ms(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f} ms"


def _classification_label(risk: Optional[float]) -> str:
    """Binary classification label at the standard 0.5 decision threshold,
    independent of the LOW/MODERATE/HIGH *triage* tier above (a MODERATE
    tier score can still fall on either side of the 0.5 classification
    boundary — the two are reported side by side, not conflated)."""
    if risk is None:
        return "N/A"
    return "TEAR LIKELY" if risk >= 0.5 else "NO TEAR INDICATED"


def _get(d: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    return default if not d else d.get(key, default)


# The three demo-category conditions this report breaks risk down by —
# matches `qknee.models.vqc_multitarget.TRIAD_CONDITIONS`' primary clinical
# triad (ACL, MCL, Medial Meniscus), the three conditions a trained
# multi-target head scores with a dedicated quantum sub-circuit each.
# `(display_name, prediction_results risk key, condition-name-for-"X Tear")`.
CONDITIONS: Tuple[Tuple[str, str, str], ...] = (
    ("ACL", "acl_risk", "ACL Tear"),
    ("MCL", "mcl_risk", "MCL Tear"),
    ("Meniscus", "meniscus_risk", "Meniscal Tear"),
)


def _predicted_condition(risks: Sequence[Optional[float]]) -> str:
    """Single overall predicted condition across `CONDITIONS` — for the
    Clinical Impression section, derived from each condition's risk score
    at the standard 0.5 classification threshold. Every condition crossing
    threshold is named (real multi-site pathology is clinically possible);
    "Normal" only when every supplied risk is sub-threshold."""
    if all(risk is None for risk in risks):
        return "N/A (insufficient data)"

    positive_labels = [
        tear_label
        for (_, _, tear_label), risk in zip(CONDITIONS, risks)
        if risk is not None and risk >= 0.5
    ]
    return " & ".join(positive_labels) if positive_labels else "Normal"


def _overall_risk(risks: Sequence[Optional[float]]) -> Optional[float]:
    """Overall risk score for the Clinical Impression header: the highest
    of the supplied condition-specific risks (worst-case/triage framing —
    the report should not understate risk by averaging a clear positive
    finding against unrelated negative ones)."""
    candidates = [r for r in risks if r is not None]
    return max(candidates) if candidates else None


def _confidence_interval(
    risk: Optional[float],
    explicit_ci: Optional[Sequence[float]],
    margin: float = DEFAULT_CI_MARGIN,
) -> Optional[Tuple[float, float]]:
    """Resolves a `(low, high)` confidence interval for one risk score.

    The trained VQC (`qknee.models.vqc.VQCClassifier`) produces a single
    deterministic point estimate from an exact state-vector simulation
    (`default.qubit`, no shot noise) — there is no sampling distribution to
    derive a statistical confidence interval from without additional
    machinery (e.g. an ensemble of checkpoints, bootstrap resampling, or a
    real shot-based NISQ backend). So:

        - If the caller supplies `explicit_ci` (e.g. computed upstream
          from such an ensemble/bootstrap), it's used directly.
        - Otherwise, a symmetric `±margin` heuristic band is used instead,
          and every place this interval is rendered is labeled
          "(heuristic)" so it is never mistaken for a statistically
          derived interval.
    """
    if risk is None:
        return None
    if explicit_ci is not None:
        low, high = explicit_ci
        return max(0.0, float(low)), min(1.0, float(high))
    return max(0.0, risk - margin), min(1.0, risk + margin)


def _format_ci(ci: Optional[Tuple[float, float]]) -> str:
    return "N/A" if ci is None else f"{ci[0] * 100:.1f}%–{ci[1] * 100:.1f}%"


def _anonymized_study_id(metadata: Dict[str, Any]) -> str:
    """Resolves the header's de-identified Study ID: an explicit
    `study_id`/`patient_id` from `metadata` if present (assumed already
    de-identified by the caller, e.g. `"DEMO-001"`), else a short
    deterministic pseudonymous ID derived by hashing whatever identifying
    fields *are* present — so the header never displays a real name/DOB
    even when the fuller metadata table below (patient-facing, not
    de-identified) does."""
    explicit = _get(metadata, "study_id") or _get(metadata, "patient_id")
    if explicit:
        return str(explicit)

    identity_source = "|".join(
        str(_get(metadata, key, "")) for key in ("patient_name", "date_of_birth", "scan_date")
    )
    if not identity_source.strip("|"):
        return "N/A"
    digest = sha256(identity_source.encode("utf-8")).hexdigest()[:10].upper()
    return f"ANON-{digest}"


# --------------------------------------------------------------------------- #
# Text-snippet entry point
# --------------------------------------------------------------------------- #
def generate_radiology_text_snippet(
    prediction_results: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates a short, automated plain-text clinical-style summary from
    the same `prediction_results`/`metadata` schema `generate_radiology_report`
    accepts (see its docstring for the full recognized-key list) — for
    callers that want a text snippet (a demo cache entry, a log line, a
    Slack/API summary) rather than a full PDF page.

    Args:
        prediction_results: Recognized keys: `acl_risk`, `mcl_risk`,
            `meniscus_risk` (floats in `[0, 1]`, or `None`/absent), plus
            optional `acl_classification`/`mcl_classification`/
            `meniscus_classification` override strings.
        metadata: Recognized keys: `patient_id`, `plane`/`modality`,
            `clinical_indication` — all optional, folded into the header
            line when present.

    Returns:
        A multi-line plain-text string, always ending with the standard
        research-prototype disclaimer sentence.
    """
    risks = [_get(prediction_results, risk_key) for _, risk_key, _ in CONDITIONS]

    case_id = _get(metadata, "patient_id", "N/A")
    plane = _get(metadata, "plane") or _get(metadata, "modality", "unspecified plane")

    lines = [
        f"Q-KNEE AUTOMATED SCREENING SUMMARY — Case {case_id} ({plane})",
        f"Predicted condition: {_predicted_condition(risks)}",
    ]
    for (display_name, _, _), risk in zip(CONDITIONS, risks):
        tier = _risk_tier(risk)
        label = _get(prediction_results, f"{display_name.lower()}_classification") or _classification_label(risk)
        lines.append(f"{display_name}: {_format_percent(risk)} tear-risk probability, {tier} tier — {label}.")
    lines.append(
        "Research prototype output, not for clinical use — findings require independent review by a "
        "licensed radiologist or orthopedic clinician."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #
def _to_image_reader(array: Optional[np.ndarray], channels: str = "L") -> Optional[Tuple[ImageReader, int, int]]:
    """Converts a numpy array (grayscale `"L"` or BGR `"BGR"`) into a
    `(reportlab.lib.utils.ImageReader, width, height)` tuple via an
    in-memory PNG (no temp files touch disk). Returns None for a missing
    array, so callers can render a placeholder box instead."""
    if array is None:
        return None

    array = np.asarray(array)
    if channels == "BGR" and array.ndim == 3:
        array = array[..., ::-1]  # BGR -> RGB for PIL

    if array.dtype != np.uint8:
        norm = array.astype(np.float32)
        lo, hi = float(norm.min()), float(norm.max())
        norm = (norm - lo) / (hi - lo) if hi > lo else np.zeros_like(norm)
        array = (norm * 255).astype(np.uint8)

    mode = "L" if array.ndim == 2 else "RGB"
    pil_image = PILImage.fromarray(array, mode=mode)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageReader(buffer), pil_image.width, pil_image.height


def _fit(width: int, height: int, max_w: float, max_h: float) -> Tuple[float, float]:
    scale = min(max_w / width, max_h / height, 1.0)
    return width * scale, height * scale


def _draw_image_panel(
    c: canvas.Canvas,
    reader_info: Optional[Tuple[ImageReader, int, int]],
    x: float,
    y: float,
    box_w: float,
    box_h: float,
    caption: str,
    placeholder_text: str,
) -> None:
    """Draws one MRI image panel: a bordered box containing the (fitted,
    centered) image, labeled anatomical-orientation axis markers on all
    four edges, and a caption underneath.

    Axis convention: the analyzed slices are sagittal knee MRI, so the
    box's top/bottom edges are labeled Superior/Inferior and the
    left/right edges Anterior/Posterior — the standard radiological
    orientation markers a reviewing clinician expects on any embedded
    slice image, regardless of the slice's raw pixel orientation.
    """
    c.setStrokeColor(colors.HexColor("#AAAAAA"))
    c.setLineWidth(0.75)
    c.rect(x, y, box_w, box_h, stroke=1, fill=0)

    if reader_info is not None:
        reader, img_w, img_h = reader_info
        fit_w, fit_h = _fit(img_w, img_h, box_w - 0.28 * inch, box_h - 0.28 * inch)
        img_x = x + (box_w - fit_w) / 2
        img_y = y + (box_h - fit_h) / 2
        c.drawImage(reader, img_x, img_y, width=fit_w, height=fit_h, preserveAspectRatio=True, mask="auto")
    else:
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColor(colors.grey)
        c.drawCentredString(x + box_w / 2, y + box_h / 2, placeholder_text)

    # Anatomical orientation markers (S / I / A / P), just inside each edge.
    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawCentredString(x + box_w / 2, y + box_h - 9, "S (Superior)")
    c.drawCentredString(x + box_w / 2, y + 3, "I (Inferior)")
    c.saveState()
    c.translate(x + 9, y + box_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "A (Anterior)")
    c.restoreState()
    c.saveState()
    c.translate(x + box_w - 9, y + box_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "P (Posterior)")
    c.restoreState()

    c.setFont("Helvetica", 9)
    c.setFillColor(_INK)
    c.drawCentredString(x + box_w / 2, y - 13, caption)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _clinical_impression_table(prediction_results: Dict[str, Any]) -> Table:
    risks = [_get(prediction_results, risk_key) for _, risk_key, _ in CONDITIONS]
    overall_risk = _overall_risk(risks)
    overall_tier = _risk_tier(overall_risk)
    predicted_condition = _predicted_condition(risks)

    overall_ci = _confidence_interval(overall_risk, _get(prediction_results, "overall_risk_ci"))
    ci_source = "caller-supplied" if _get(prediction_results, "overall_risk_ci") is not None else "heuristic ±{:.0f}pp".format(DEFAULT_CI_MARGIN * 100)

    rows = [
        ["Overall Risk Score", _format_percent(overall_risk), overall_tier],
        ["Predicted Condition", predicted_condition, ""],
        [f"Confidence Interval ({ci_source})", _format_ci(overall_ci), ""],
    ]
    table = Table(rows, colWidths=[2.3 * inch, 2.2 * inch, 1.2 * inch])
    table.setStyle(TableStyle([
        ("SPAN", (1, 1), (2, 1)),
        ("SPAN", (1, 2), (2, 2)),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F0F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TEXTCOLOR", (1, 0), (1, 0), _TIER_COLORS[overall_tier]),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
    ]))
    return table


def _metadata_table(metadata: Dict[str, Any]) -> Table:
    rows = [
        ["Patient ID", str(_get(metadata, "patient_id", "N/A")), "Scan Date", str(_get(metadata, "scan_date", "N/A"))],
        ["Patient Name", str(_get(metadata, "patient_name", "N/A")), "Modality", str(_get(metadata, "modality", "MRI Knee"))],
        [
            "Date of Birth", str(_get(metadata, "date_of_birth", "N/A")),
            "Referring Physician", str(_get(metadata, "referring_physician", "N/A")),
        ],
        ["Clinical Indication", str(_get(metadata, "clinical_indication", "N/A")), "", ""],
    ]
    table = Table(rows, colWidths=[1.3 * inch, 2.15 * inch, 1.55 * inch, 1.7 * inch])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F0F0")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F0F0F0")),
        ("SPAN", (1, 3), (3, 3)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _quantum_attribution_table(prediction_results: Dict[str, Any]) -> Optional[Table]:
    """Builds the Quantum Feature Attribution table: one row per qubit's
    Pauli-Z expectation value <Z>, plus (when the caller supplies the
    trained readout layer's weights) the feature-impact weight and the
    resulting weighted contribution to the pre-sigmoid risk logit —
    `weight_i * <Z_i>`, i.e. exactly the term that qubit contributes inside
    `VQCClassifier.readout` (`Linear(n_qubits, 1)`).

    Recognized `prediction_results` keys:
        pauli_z_expectations: sequence of floats in [-1, 1], one per qubit.
        readout_weights: optional sequence of floats, same length — the
            trained `Linear(n_qubits, 1)` readout's per-qubit weights.

    Returns `None` if `pauli_z_expectations` isn't supplied, so the caller
    can skip rendering this section entirely rather than showing an
    all-"N/A" table.
    """
    pauli_z = _get(prediction_results, "pauli_z_expectations")
    if pauli_z is None:
        return None

    readout_weights = _get(prediction_results, "readout_weights")
    has_weights = readout_weights is not None and len(readout_weights) == len(pauli_z)

    header = ["Qubit", "Pauli-Z Expectation <Z>", "Feature Impact Weight", "Weighted Contribution"]
    rows = [header]
    for i, expval in enumerate(pauli_z):
        weight = readout_weights[i] if has_weights else None
        contribution = (weight * expval) if has_weights else None
        rows.append([
            f"Q{i}",
            f"{expval:+.4f}",
            f"{weight:+.4f}" if weight is not None else "N/A",
            f"{contribution:+.4f}" if contribution is not None else "N/A",
        ])

    table = Table(rows, colWidths=[0.8 * inch, 1.75 * inch, 1.75 * inch, 1.75 * inch])
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (-1, 0), _INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index, expval in enumerate(pauli_z, start=1):
        color = colors.HexColor("#E74C3C") if expval >= 0 else colors.HexColor("#2ECC71")
        style.append(("TEXTCOLOR", (1, row_index), (1, row_index), color))
    table.setStyle(TableStyle(style))
    return table


def _diagnostic_table(prediction_results: Dict[str, Any]) -> Table:
    """Per-condition Diagnostic Breakdown table: one row per `CONDITIONS`
    entry (ACL / MCL / Meniscus — the primary clinical triad
    `qknee.models.vqc_multitarget` scores with a dedicated quantum
    sub-circuit each), each with its tear-risk probability, confidence
    interval, risk tier, and 0.5-threshold classification label."""
    rows = [["Region", "Tear Risk", "95% CI", "Risk Tier", "Classification"]]
    tiers = []
    for display_name, risk_key, _ in CONDITIONS:
        risk = _get(prediction_results, risk_key)
        tier = _risk_tier(risk)
        label = _get(prediction_results, f"{display_name.lower()}_classification") or _classification_label(risk)
        ci = _confidence_interval(risk, _get(prediction_results, f"{risk_key}_ci"))
        rows.append([display_name, _format_percent(risk), _format_ci(ci), tier, label])
        tiers.append(tier)

    table = Table(rows, colWidths=[1.0 * inch, 1.0 * inch, 1.3 * inch, 1.0 * inch, 1.75 * inch])
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (-1, 0), _INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row_index, tier in enumerate(tiers, start=1):
        style.append(("TEXTCOLOR", (3, row_index), (3, row_index), _TIER_COLORS[tier]))
        style.append(("FONTNAME", (3, row_index), (3, row_index), "Helvetica-Bold"))
    table.setStyle(TableStyle(style))
    return table


def _latency_table(prediction_results: Dict[str, Any]) -> Table:
    resnet_ms = _get(prediction_results, "resnet_latency_ms")
    pca_ms = _get(prediction_results, "pca_latency_ms")
    quantum_ms = _get(prediction_results, "quantum_latency_ms")
    total_ms = _get(prediction_results, "total_latency_ms")

    rows = [
        ["Feature Extraction", "Dim. Reduction", "Quantum Circuit Inference", "Total (End-to-End)"],
        [_format_ms(resnet_ms), _format_ms(pca_ms), _format_ms(quantum_ms), _format_ms(total_ms)],
    ]
    table = Table(rows, colWidths=[1.65 * inch, 1.55 * inch, 2.05 * inch, 1.45 * inch])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _draw_table(c: canvas.Canvas, table: Table, x: float, top_y: float, max_width: float) -> float:
    """Draws a flowable `Table` onto the canvas with its top-left corner at
    `(x, top_y)`, returning the y-coordinate immediately below it."""
    width, height = table.wrapOn(c, max_width, PAGE_HEIGHT)
    table.drawOn(c, x, top_y - height)
    return top_y - height


def _section_heading(c: canvas.Canvas, text: str, y: float) -> float:
    c.setFont("Helvetica-Bold", 10.5)
    c.setFillColor(_INK)
    c.drawString(MARGIN, y, text)
    return y - 14


# --------------------------------------------------------------------------- #
# Section 1: Header (branding + timestamp + anonymized Study ID)
# --------------------------------------------------------------------------- #
def _draw_header(c: canvas.Canvas, metadata: Dict[str, Any], generated_at: datetime, page_label: str) -> float:
    clinic_name = str(_get(metadata, "clinic_name", DEFAULT_CLINIC_NAME))
    study_id = _anonymized_study_id(metadata)

    y = PAGE_HEIGHT - MARGIN

    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(_INK)
    c.drawCentredString(PAGE_WIDTH / 2, y, clinic_name)
    y -= 15

    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(PAGE_WIDTH / 2, y, f"Quantum-Assisted MRI Analysis Report {page_label}")
    y -= 12

    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColor(colors.grey)
    c.drawCentredString(
        PAGE_WIDTH / 2, y,
        f"{DEFAULT_CLINIC_SUBTITLE} — research prototype, not a certified medical device.",
    )
    y -= 13

    c.setFont("Helvetica", 8)
    c.setFillColor(_INK)
    c.drawString(MARGIN, y, f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}")
    c.drawRightString(PAGE_WIDTH - MARGIN, y, f"Study ID (De-identified): {study_id}")
    y -= 10

    c.setStrokeColor(_INK)
    c.setLineWidth(1)
    c.line(MARGIN, y, PAGE_WIDTH - MARGIN, y)
    return y - 14


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_radiology_report(
    output_path: Optional[Union[str, Path]],
    mri_slice: Optional[np.ndarray],
    gradcam_overlay: Optional[np.ndarray],
    prediction_results: Dict[str, Any],
    metadata: Dict[str, Any],
) -> bytes:
    """Builds a structured, multi-section clinical radiology PDF (Header,
    Clinical Impression, Visual Evidence, Quantum Feature Attribution,
    Disclaimer — across up to two letter-size pages) and returns it as raw
    bytes, so callers can feed the same call straight into
    `st.download_button(data=...)` without an extra disk round-trip; also
    writes the same bytes to `output_path` when one is given.

    Args:
        output_path: Destination `.pdf` path (parent directories created
            if missing), or `None` to skip writing to disk and only
            return the in-memory bytes.
        mri_slice: `(H, W)` grayscale or `(H, W, 3)` RGB display array of
            the analyzed slice, or `None` to render a placeholder panel.
        gradcam_overlay: `(H, W, 3)` BGR risk-targeted Grad-CAM overlay
            (as produced by `qknee.xai.gradcam.overlay_heatmap`), or
            `None` to render a placeholder panel.
        prediction_results: Diagnostic/inference payload. Recognized keys
            (all optional, missing ones render as "N/A"):
                acl_risk, mcl_risk, meniscus_risk - floats in [0, 1] — the
                                                     primary clinical triad
                                                     (matches
                                                     qknee.models.vqc_multitarget.
                                                     TRIAD_CONDITIONS)
                acl_risk_ci, mcl_risk_ci,
                meniscus_risk_ci, overall_risk_ci - optional (low, high)
                                                     confidence-interval
                                                     overrides; see
                                                     `_confidence_interval`
                acl_classification,
                mcl_classification,
                meniscus_classification           - override label strings;
                                                     default is a 0.5-threshold
                                                     "TEAR LIKELY"/"NO TEAR
                                                     INDICATED" derived from
                                                     the risk score
                pauli_z_expectations              - sequence of floats in
                                                     [-1, 1], one per qubit —
                                                     enables the Quantum
                                                     Feature Attribution
                                                     section when present
                readout_weights                   - optional sequence of
                                                     floats (same length as
                                                     pauli_z_expectations),
                                                     the trained readout
                                                     layer's per-qubit weights
                resnet_latency_ms, pca_latency_ms,
                quantum_latency_ms, total_latency_ms - float milliseconds
                backend                           - str, e.g. "live"/"mock"/"api"
        metadata: Patient/study identifiers. Recognized keys (all
            optional, missing ones render as "N/A"):
                patient_id, patient_name, date_of_birth, scan_date,
                modality (default "MRI Knee"), clinical_indication,
                referring_physician, clinic_name (letterhead override),
                study_id (explicit anonymized ID; derived if omitted)

    Returns:
        Raw PDF bytes.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    c.setTitle("Q-Knee Radiology Report")

    generated_at = datetime.now(timezone.utc)
    backend = _get(prediction_results, "backend", "live")

    # =================================================================== #
    # PAGE 1 — Header, Clinical Impression, Patient/Study Info, Visual Evidence
    # =================================================================== #
    y = _draw_header(c, metadata, generated_at, page_label="(Page 1 of 2)")

    y = _section_heading(c, "Clinical Impression", y)
    y = _draw_table(c, _clinical_impression_table(prediction_results), MARGIN, y - 2, CONTENT_WIDTH) - 16

    y = _section_heading(c, "Patient & Study Information", y)
    y = _draw_table(c, _metadata_table(metadata), MARGIN, y - 2, CONTENT_WIDTH) - 18

    y = _section_heading(c, "Visual Evidence — MRI Slice & Risk-Targeted Grad-CAM Overlay", y)

    panel_gap = 0.3 * inch
    panel_w = (CONTENT_WIDTH - panel_gap) / 2
    panel_h = 2.55 * inch
    caption_space = 18
    panel_top = y - caption_space
    panel_y = panel_top - panel_h

    slice_reader = _to_image_reader(mri_slice, channels="L")
    overlay_reader = _to_image_reader(gradcam_overlay, channels="BGR")

    _draw_image_panel(
        c, slice_reader, MARGIN, panel_y, panel_w, panel_h,
        caption="Original MRI Slice", placeholder_text="(slice unavailable)",
    )
    _draw_image_panel(
        c, overlay_reader, MARGIN + panel_w + panel_gap, panel_y, panel_w, panel_h,
        caption="Grad-CAM Risk-Region Overlay", placeholder_text="(Grad-CAM unavailable)",
    )

    c.setFont("Helvetica", 7)
    c.setFillColor(colors.grey)
    c.drawCentredString(PAGE_WIDTH / 2, MARGIN * 0.55, f"Page 1 of 2 — {DEFAULT_CLINIC_NAME}")

    # =================================================================== #
    # PAGE 2 — Quantum Feature Attribution, Diagnostic Breakdown, Latency,
    # Disclaimer + signature
    # =================================================================== #
    c.showPage()
    y = _draw_header(c, metadata, generated_at, page_label="(Page 2 of 2)")

    attribution_table = _quantum_attribution_table(prediction_results)
    y = _section_heading(c, "Quantum Feature Attribution — 4-Qubit VQC Measurement", y)
    if attribution_table is not None:
        y = _draw_table(c, attribution_table, MARGIN, y - 2, CONTENT_WIDTH) - 4
        c.setFont("Helvetica-Oblique", 7)
        c.setFillColor(colors.grey)
        c.drawString(
            MARGIN, y - 8,
            "Each qubit's Pauli-Z expectation <Z> (measurement basis, range [-1, 1]) and, when available, "
            "its trained readout weight and weighted contribution to the pre-sigmoid risk logit.",
        )
        y -= 22
    else:
        c.setFont("Helvetica-Oblique", 8.5)
        c.setFillColor(colors.grey)
        c.drawString(MARGIN, y - 2, "Per-qubit measurement data not available for this inference.")
        y -= 20

    y = _section_heading(c, "Diagnostic Breakdown", y)
    y = _draw_table(c, _diagnostic_table(prediction_results), MARGIN, y - 2, CONTENT_WIDTH) - 16

    y = _section_heading(c, "Quantum Circuit Inference Latency", y)
    y = _draw_table(c, _latency_table(prediction_results), MARGIN, y - 2, CONTENT_WIDTH) - 4
    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColor(colors.grey)
    c.drawString(MARGIN, y - 8, f"Inference backend: {backend}")
    y -= 24

    # --- Section 5: Legal/Clinical Disclaimer + timestamped signature placeholder ---
    footer_top = MARGIN + 0.72 * inch
    c.setStrokeColor(colors.HexColor("#CCCCCC"))
    c.setLineWidth(0.5)
    c.line(MARGIN, footer_top, PAGE_WIDTH - MARGIN, footer_top)

    disclaimer_style = ParagraphStyle(
        "Disclaimer", fontName="Helvetica-Oblique", fontSize=6.6, leading=8.2, textColor=colors.grey,
    )
    disclaimer = Paragraph(
        "AUTOMATED NISQ SCREENING AI DISCLOSURE: This report was generated by Q-Knee, an investigational "
        "clinical decision-support tool using a noisy intermediate-scale quantum (NISQ) simulated variational "
        "circuit as part of a hybrid classical/quantum inference pipeline. It is NOT a validated diagnostic "
        "device, has NOT been reviewed or cleared by the FDA or any other regulatory authority, and is not "
        "intended to diagnose, treat, cure, or prevent any disease. All risk scores, confidence intervals, and "
        "quantum feature attributions are probabilistic estimates from a research prototype and must not be "
        "used, in isolation or otherwise, as the basis for diagnosis or treatment. This report does not "
        "establish a clinician-patient relationship and creates no duty of care on the part of its developers. "
        "All findings require independent interpretation, confirmation, and clinical correlation by a "
        "qualified, licensed radiologist or orthopedic clinician before any clinical action is taken.",
        disclaimer_style,
    )
    disclaimer_width, disclaimer_height = disclaimer.wrapOn(c, CONTENT_WIDTH, 1.2 * inch)
    disclaimer.drawOn(c, MARGIN, footer_top - disclaimer_height - 4)

    signature_source = f"{_anonymized_study_id(metadata)}|{generated_at.isoformat()}|{backend}"
    signature_hash = sha256(signature_source.encode("utf-8")).hexdigest()[:16].upper()

    sig_y = footer_top - disclaimer_height - 20
    c.setFont("Helvetica", 6.8)
    c.setFillColor(colors.HexColor("#888888"))
    c.drawString(
        MARGIN, sig_y,
        f"Digitally signed (placeholder, non-cryptographic) by Q-Knee Automated Analysis System — "
        f"generated {generated_at.strftime('%Y-%m-%d %H:%M:%S %Z').strip()} — signature ref: {signature_hash}",
    )
    # No separate "Page 2 of 2" footer label here — the header already
    # states it, and this page's bottom margin is already occupied by the
    # disclaimer + signature line above.

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(pdf_bytes)
        logger.info("Saved radiology report to %s (%d bytes)", output_path, len(pdf_bytes))

    return pdf_bytes


class ReportGenerator:
    """Thin object-oriented facade over `generate_radiology_report`/
    `generate_radiology_text_snippet`, for callers (and tests) that prefer a
    reusable reporter instance — e.g. one that pins shared `metadata`
    defaults (clinic name, referring physician) across many per-case report
    builds — over calling the module-level functions directly. Delegates
    its entire implementation to those functions; no report-building logic
    lives here.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}

    def build_pdf_report(
        self,
        prediction_results: Dict[str, Any],
        mri_slice: Optional[np.ndarray] = None,
        gradcam_overlay: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> bytes:
        """Builds the same two-page clinical PDF report as
        `generate_radiology_report` (see its docstring for the full
        recognized `prediction_results`/`metadata` key list), merging
        this instance's `self.metadata` defaults with the call-site
        `metadata` (call-site keys win on conflict). Returns raw PDF
        bytes; also writes them to `output_path` when one is given.
        """
        merged_metadata = {**self.metadata, **(metadata or {})}
        return generate_radiology_report(
            output_path=output_path,
            mri_slice=mri_slice,
            gradcam_overlay=gradcam_overlay,
            prediction_results=prediction_results,
            metadata=merged_metadata,
        )

    def build_text_snippet(
        self, prediction_results: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Same merged-metadata convenience as `build_pdf_report`, for the
        short plain-text summary (`generate_radiology_text_snippet`)."""
        merged_metadata = {**self.metadata, **(metadata or {})}
        return generate_radiology_text_snippet(prediction_results, merged_metadata)


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()

    rng = np.random.default_rng(0)
    dummy_slice = rng.integers(0, 255, size=(224, 224), dtype=np.uint8)
    dummy_overlay = np.dstack([dummy_slice] * 3)  # stand-in BGR "overlay" for the smoke test

    pdf_bytes = generate_radiology_report(
        output_path="qknee/artifacts/demo_radiology_report.pdf",
        mri_slice=dummy_slice,
        gradcam_overlay=dummy_overlay,
        prediction_results={
            "acl_risk": 0.72,
            "mcl_risk": 0.44,
            "meniscus_risk": 0.31,
            "pauli_z_expectations": [0.42, -0.18, 0.63, -0.07],
            "readout_weights": [0.85, -0.40, 1.10, 0.25],
            "resnet_latency_ms": 18.4,
            "pca_latency_ms": 1.2,
            "quantum_latency_ms": 42.7,
            "total_latency_ms": 62.3,
            "backend": "live",
        },
        metadata={
            "patient_id": "DEMO-001",
            "patient_name": "Test Patient",
            "date_of_birth": "1990-01-01",
            "scan_date": datetime.now().strftime("%Y-%m-%d"),
            "modality": "MRI Knee",
            "clinical_indication": "Chronic knee pain, rule out ACL/meniscal tear",
            "referring_physician": "Dr. A. Example",
        },
    )
    assert pdf_bytes[:4] == b"%PDF"
    assert Path("qknee/artifacts/demo_radiology_report.pdf").stat().st_size > 0
    logger.info("Report generated: %d bytes, valid PDF header, written to disk.", len(pdf_bytes))

    bytes_only = generate_radiology_report(
        output_path=None,
        mri_slice=dummy_slice,
        gradcam_overlay=None,
        prediction_results={"acl_risk": 0.15, "mcl_risk": None, "meniscus_risk": None, "backend": "mock"},
        metadata={},
    )
    assert bytes_only[:4] == b"%PDF"
    logger.info("In-memory-only report generated: %d bytes (no disk write). All checks passed.", len(bytes_only))
