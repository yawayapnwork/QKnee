"""
PDF radiology-report generation for Q-Knee predictions.

Compiles the analyzed MRI slice, its Grad-CAM overlay, patient/study
metadata, and ACL/meniscus tear-risk scores into a clean, multi-page
ReportLab document — the source for the Streamlit dashboards' "Download
Report" button (`qknee.ui.dashboard`).

RESEARCH PROTOTYPE — every generated report carries a "not for clinical
use" disclaimer; this module produces a demo/prototype artifact, not a
validated diagnostic report.

Two entry points, sharing the same layout via `_build_story`:
    - `generate_radiology_report`       - writes a .pdf file to disk.
    - `generate_radiology_report_bytes` - builds directly into an
      in-memory buffer, for `st.download_button(data=...)`, which needs
      raw bytes rather than a file path.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from qknee.config.logging_config import get_logger

logger = get_logger(__name__)

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


@dataclass
class PatientMetadata:
    """Patient/study identifiers shown on the report header. All fields
    are free-text and optional — omitted fields render as 'N/A' rather
    than blocking report generation (this is a research-prototype report,
    not a system of record for patient identity)."""

    patient_id: str = "N/A"
    patient_name: str = "N/A"
    date_of_birth: str = "N/A"
    study_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    referring_physician: str = "N/A"
    scan_description: str = "Knee MRI"


def _risk_tier(risk: Optional[float]) -> str:
    if risk is None:
        return "N/A"
    if risk >= RISK_MODERATE_MAX:
        return "HIGH"
    if risk >= RISK_LOW_MAX:
        return "MODERATE"
    return "LOW"


def _format_risk(risk: Optional[float]) -> str:
    return "N/A" if risk is None else f"{risk * 100:.1f}%"


def _array_to_flowable_image(
    array: Optional[np.ndarray], max_width: float, max_height: float, channels: str = "L"
) -> Optional[RLImage]:
    """Converts a numpy array (grayscale `"L"` or BGR `"BGR"`) into a
    size-constrained ReportLab `Image` flowable, via an in-memory PNG (no
    temp files touch disk). Returns None for a missing array, so callers
    can conditionally show a placeholder instead."""
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
    pil_image = Image.fromarray(array, mode=mode)

    width, height = pil_image.size
    scale = min(max_width / width, max_height / height, 1.0)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return RLImage(buffer, width=width * scale, height=height * scale)


def _build_story(
    styles,
    patient_metadata: PatientMetadata,
    mri_slice: Optional[np.ndarray],
    gradcam_overlay: Optional[np.ndarray],
    acl_risk: float,
    meniscus_risk: Optional[float],
    backend: str,
    notes: Optional[str],
) -> List:
    """Builds the ReportLab flowable list shared by both entry points below."""
    story: List = []

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#16222A"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.grey, fontSize=9,
    )

    story.append(Paragraph("Q-Knee Quantum-Assisted MRI Analysis Report", title_style))
    story.append(Paragraph(
        "Research prototype — quantum-assisted ACL / meniscal tear risk triage. "
        "Not a certified medical device. Not for clinical use.",
        subtitle_style,
    ))
    story.append(Spacer(1, 0.25 * inch))

    # --- Patient / study metadata table ---
    metadata_rows = [
        ["Patient ID", patient_metadata.patient_id, "Study Date", patient_metadata.study_date],
        ["Patient Name", patient_metadata.patient_name, "Referring Physician", patient_metadata.referring_physician],
        ["Date of Birth", patient_metadata.date_of_birth, "Scan Description", patient_metadata.scan_description],
    ]
    metadata_table = Table(metadata_rows, colWidths=[1.3 * inch, 2.0 * inch, 1.6 * inch, 1.8 * inch])
    metadata_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F0F0F0")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F0F0F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 0.3 * inch))

    # --- Risk summary table ---
    story.append(Paragraph("Tear-Risk Assessment", styles["Heading2"]))
    acl_tier, meniscus_tier = _risk_tier(acl_risk), _risk_tier(meniscus_risk)
    risk_rows = [
        ["Condition", "Risk Score", "Risk Tier"],
        ["ACL", _format_risk(acl_risk), acl_tier],
        ["Meniscus", _format_risk(meniscus_risk), meniscus_tier],
    ]
    risk_table = Table(risk_rows, colWidths=[2.2 * inch, 2.2 * inch, 2.2 * inch])
    risk_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16222A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TEXTCOLOR", (2, 1), (2, 1), _TIER_COLORS[acl_tier]),
        ("FONTNAME", (2, 1), (2, 1), "Helvetica-Bold"),
        ("TEXTCOLOR", (2, 2), (2, 2), _TIER_COLORS[meniscus_tier]),
        ("FONTNAME", (2, 2), (2, 2), "Helvetica-Bold"),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"Inference backend: <b>{backend}</b>", styles["Normal"]))

    if notes:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("Notes", styles["Heading3"]))
        story.append(Paragraph(notes, styles["Normal"]))

    # --- Page 2: imaging ---
    story.append(PageBreak())
    story.append(Paragraph("MRI Slice & Grad-CAM Explainability", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))

    image_max_width, image_max_height = 3.2 * inch, 3.2 * inch
    slice_flowable = _array_to_flowable_image(mri_slice, image_max_width, image_max_height, channels="L")
    gradcam_flowable = _array_to_flowable_image(gradcam_overlay, image_max_width, image_max_height, channels="BGR")

    image_row = [
        slice_flowable or Paragraph("(slice unavailable)", styles["Normal"]),
        gradcam_flowable or Paragraph("(Grad-CAM unavailable)", styles["Normal"]),
    ]
    caption_row = [
        Paragraph("Original MRI Slice", subtitle_style),
        Paragraph("Grad-CAM Risk-Region Overlay", subtitle_style),
    ]
    image_table = Table([image_row, caption_row], colWidths=[3.4 * inch, 3.4 * inch])
    image_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
        ("TOPPADDING", (0, 1), (-1, 1), 6),
    ]))
    story.append(image_table)

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "Grad-CAM highlights the anatomical regions that most influenced the predicted risk score "
        "(backpropagated from the model's own risk output, not raw feature energy) — for "
        "explainability review, not as an independent diagnostic finding.",
        styles["Normal"],
    ))

    return story


def generate_radiology_report(
    output_path: Union[str, Path],
    mri_slice: Optional[np.ndarray],
    acl_risk: float,
    meniscus_risk: Optional[float] = None,
    gradcam_overlay: Optional[np.ndarray] = None,
    patient_metadata: Optional[PatientMetadata] = None,
    backend: str = "live",
    notes: Optional[str] = None,
) -> Path:
    """Builds a multi-page PDF radiology report and writes it to `output_path`.

    Args:
        output_path: Destination `.pdf` path (parent directories are
            created if missing).
        mri_slice: `(H, W)` or `(H, W, 3)` grayscale/RGB display array of
            the analyzed slice, or None to render a placeholder.
        acl_risk: ACL tear-risk probability in `[0, 1]`.
        meniscus_risk: Meniscus tear-risk probability in `[0, 1]`, or None
            when unavailable (e.g. the unified single-score API backend).
        gradcam_overlay: `(H, W, 3)` BGR Grad-CAM overlay (as produced by
            `qknee.xai.gradcam.overlay_heatmap`), or None to omit it.
        patient_metadata: Patient/study identifiers; defaults to all-"N/A"
            placeholders with today's date as the study date.
        backend: Which inference backend produced the scores ("live",
            "mock", or "api/..."), shown for provenance.
        notes: Optional free-text clinical notes appended to page 1.

    Returns:
        `output_path`, for chaining (e.g. into code that then reads the
        file back as bytes).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(output_path), pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title="Q-Knee Radiology Report",
    )
    story = _build_story(
        styles, patient_metadata or PatientMetadata(), mri_slice, gradcam_overlay,
        acl_risk, meniscus_risk, backend, notes,
    )
    doc.build(story)
    logger.info("Saved radiology report to %s", output_path)
    return output_path


def generate_radiology_report_bytes(
    mri_slice: Optional[np.ndarray],
    acl_risk: float,
    meniscus_risk: Optional[float] = None,
    gradcam_overlay: Optional[np.ndarray] = None,
    patient_metadata: Optional[PatientMetadata] = None,
    backend: str = "live",
    notes: Optional[str] = None,
) -> bytes:
    """In-memory variant of `generate_radiology_report`, for
    `st.download_button(data=...)`, which needs raw bytes rather than a
    file on disk. Same arguments (minus `output_path`); builds directly
    into a `BytesIO` buffer instead of writing a temp file."""
    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title="Q-Knee Radiology Report",
    )
    story = _build_story(
        styles, patient_metadata or PatientMetadata(), mri_slice, gradcam_overlay,
        acl_risk, meniscus_risk, backend, notes,
    )
    doc.build(story)
    return buffer.getvalue()


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()

    rng = np.random.default_rng(0)
    dummy_slice = rng.integers(0, 255, size=(224, 224), dtype=np.uint8)
    dummy_overlay = np.dstack([dummy_slice] * 3)  # stand-in BGR "overlay" for the smoke test

    saved_path = generate_radiology_report(
        output_path="qknee/artifacts/demo_radiology_report.pdf",
        mri_slice=dummy_slice,
        acl_risk=0.72,
        meniscus_risk=0.31,
        gradcam_overlay=dummy_overlay,
        patient_metadata=PatientMetadata(patient_id="DEMO-001", patient_name="Test Patient"),
        backend="live",
        notes="Smoke-test report generated by report_generator.py's __main__ block.",
    )
    assert saved_path.exists() and saved_path.stat().st_size > 0
    logger.info("Report written to %s (%d bytes)", saved_path, saved_path.stat().st_size)

    report_bytes = generate_radiology_report_bytes(
        mri_slice=dummy_slice, acl_risk=0.15, meniscus_risk=None, backend="mock",
    )
    assert report_bytes[:4] == b"%PDF"
    logger.info("In-memory report generated: %d bytes, valid PDF header.", len(report_bytes))
    logger.info("All checks passed.")
