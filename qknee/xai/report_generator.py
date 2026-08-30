"""
Single-page clinical radiology PDF report generation for Q-Knee predictions.

Compiles the analyzed MRI slice, its risk-targeted Grad-CAM overlay,
patient/study metadata, and ACL/meniscus tear-risk scores into one
letter-size page — the source for the Streamlit dashboards' "Download
Report" button (`qknee.ui.dashboard.render_report_download`).

RESEARCH PROTOTYPE — every generated report carries a "not for clinical
use" disclaimer footer; this module produces a demo/prototype artifact,
not a validated diagnostic report, and the "digital signature" is a
placeholder hash, not a cryptographic attestation.

Built directly on a `reportlab.pdfgen.canvas.Canvas` (rather than a
`SimpleDocTemplate` flowable story) so the page is guaranteed to be
exactly one page: every element is placed at an explicit y-coordinate
computed top-down, so there is no flowing/paginating content that could
silently spill onto a second page. `Table` flowables are still used for
the tabular sections (metadata banner, diagnostic breakdown) and drawn
onto the canvas via `wrapOn`/`drawOn`, combining canvas-level layout
control with flowables for anything genuinely tabular.
"""

from __future__ import annotations

import io
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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


# --------------------------------------------------------------------------- #
# Small formatting helpers
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


def _diagnostic_table(prediction_results: Dict[str, Any]) -> Table:
    acl_risk = _get(prediction_results, "acl_risk")
    meniscus_risk = _get(prediction_results, "meniscus_risk")
    acl_tier = _risk_tier(acl_risk)
    meniscus_tier = _risk_tier(meniscus_risk)
    acl_label = _get(prediction_results, "acl_classification") or _classification_label(acl_risk)
    meniscus_label = _get(prediction_results, "meniscus_classification") or _classification_label(meniscus_risk)

    rows = [
        ["Region", "Tear Risk", "Risk Tier", "Classification"],
        ["ACL", _format_percent(acl_risk), acl_tier, acl_label],
        ["Meniscus", _format_percent(meniscus_risk), meniscus_tier, meniscus_label],
    ]
    table = Table(rows, colWidths=[1.4 * inch, 1.4 * inch, 1.4 * inch, 2.5 * inch])
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (-1, 0), _INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TEXTCOLOR", (2, 1), (2, 1), _TIER_COLORS[acl_tier]),
        ("FONTNAME", (2, 1), (2, 1), "Helvetica-Bold"),
        ("TEXTCOLOR", (2, 2), (2, 2), _TIER_COLORS[meniscus_tier]),
        ("FONTNAME", (2, 2), (2, 2), "Helvetica-Bold"),
    ]
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
    """Builds a formal, one-page clinical radiology summary PDF and returns
    it as raw bytes (also writing it to `output_path` when one is given),
    so callers can feed the same call straight into
    `st.download_button(data=...)` without an extra disk round-trip.

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
                acl_risk, meniscus_risk           - floats in [0, 1]
                acl_classification,
                meniscus_classification           - override label strings;
                                                     default is a 0.5-threshold
                                                     "TEAR LIKELY"/"NO TEAR
                                                     INDICATED" derived from
                                                     the risk score
                resnet_latency_ms, pca_latency_ms,
                quantum_latency_ms, total_latency_ms - float milliseconds
                backend                           - str, e.g. "live"/"mock"/"api"
        metadata: Patient/study identifiers. Recognized keys (all
            optional, missing ones render as "N/A"):
                patient_id, patient_name, date_of_birth, scan_date,
                modality (default "MRI Knee"), clinical_indication,
                referring_physician

    Returns:
        Raw PDF bytes.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    c.setTitle("Q-Knee Radiology Report")

    content_width = PAGE_WIDTH - 2 * MARGIN
    y = PAGE_HEIGHT - MARGIN

    # --- Header ---
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(_INK)
    c.drawCentredString(PAGE_WIDTH / 2, y, "Q-KNEE QUANTUM-ASSISTED MRI ANALYSIS REPORT")
    y -= 14

    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColor(colors.grey)
    c.drawCentredString(
        PAGE_WIDTH / 2, y,
        "Research prototype — quantum-assisted ACL / meniscal tear-risk triage. Not a certified medical device.",
    )
    y -= 16

    c.setStrokeColor(_INK)
    c.setLineWidth(1)
    c.line(MARGIN, y, PAGE_WIDTH - MARGIN, y)
    y -= 14

    # --- Patient / study metadata banner ---
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(_INK)
    c.drawString(MARGIN, y, "Patient & Study Information")
    y -= 4
    y = _draw_table(c, _metadata_table(metadata), MARGIN, y - 10, content_width) - 16

    # --- Imaging: side-by-side MRI slice + Grad-CAM overlay ---
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(_INK)
    c.drawString(MARGIN, y, "Imaging — MRI Slice & Risk-Targeted Grad-CAM Overlay")
    y -= 12

    panel_gap = 0.3 * inch
    panel_w = (content_width - panel_gap) / 2
    panel_h = 2.55 * inch
    caption_space = 18
    panel_top = y - caption_space  # leave room for the caption drawn below each panel
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
    y = panel_y - 22

    # --- Diagnostic breakdown ---
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(_INK)
    c.drawString(MARGIN, y, "Diagnostic Breakdown")
    y -= 4
    y = _draw_table(c, _diagnostic_table(prediction_results), MARGIN, y - 10, content_width) - 16

    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(_INK)
    c.drawString(MARGIN, y, "Quantum Circuit Inference Latency")
    y -= 4
    backend = _get(prediction_results, "backend", "live")
    y = _draw_table(c, _latency_table(prediction_results), MARGIN, y - 10, content_width) - 4
    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColor(colors.grey)
    c.drawString(MARGIN, y - 8, f"Inference backend: {backend}")
    y -= 20

    # --- Footer: clinical disclaimer + timestamped digital signature placeholder ---
    footer_top = MARGIN + 0.62 * inch
    c.setStrokeColor(colors.HexColor("#CCCCCC"))
    c.setLineWidth(0.5)
    c.line(MARGIN, footer_top, PAGE_WIDTH - MARGIN, footer_top)

    disclaimer_style = ParagraphStyle(
        "Disclaimer", fontName="Helvetica-Oblique", fontSize=6.8, leading=8.4, textColor=colors.grey,
    )
    disclaimer = Paragraph(
        "CLINICAL DISCLAIMER: This report is generated by Q-Knee, a research-prototype quantum-assisted "
        "decision-support tool. It is NOT a validated diagnostic device and has NOT been reviewed or cleared "
        "by any regulatory authority. Findings are probabilistic risk estimates only and must not be used, "
        "in isolation or otherwise, as the basis for diagnosis or treatment. All results require independent "
        "interpretation and confirmation by a qualified, licensed radiologist or orthopedic clinician.",
        disclaimer_style,
    )
    disclaimer_width, disclaimer_height = disclaimer.wrapOn(c, content_width, 1 * inch)
    disclaimer.drawOn(c, MARGIN, footer_top - disclaimer_height - 4)

    timestamp = datetime.now()
    signature_source = f"{_get(metadata, 'patient_id', 'N/A')}|{timestamp.isoformat()}|{backend}"
    signature_hash = sha256(signature_source.encode("utf-8")).hexdigest()[:16].upper()

    sig_y = footer_top - disclaimer_height - 20
    c.setFont("Helvetica", 6.8)
    c.setFillColor(colors.HexColor("#888888"))
    c.drawString(
        MARGIN, sig_y,
        f"Digitally signed (placeholder, non-cryptographic) by Q-Knee Automated Analysis System — "
        f"generated {timestamp.strftime('%Y-%m-%d %H:%M:%S')} — signature ref: {signature_hash}",
    )

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(pdf_bytes)
        logger.info("Saved radiology report to %s (%d bytes)", output_path, len(pdf_bytes))

    return pdf_bytes


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
            "meniscus_risk": 0.31,
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
        prediction_results={"acl_risk": 0.15, "meniscus_risk": None, "backend": "mock"},
        metadata={},
    )
    assert bytes_only[:4] == b"%PDF"
    logger.info("In-memory-only report generated: %d bytes (no disk write). All checks passed.", len(bytes_only))
