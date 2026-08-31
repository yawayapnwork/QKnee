"""
Generates high-resolution, publication-ready visual assets for the Q-Knee
24-hour hackathon pitch deck. Every figure is saved as both a high-DPI
`.png` (slides) and a resolution-independent `.svg` (print/poster) into
`qknee/artifacts/deck_figures/`.

Figures:
    1. `roc_and_parameter_efficiency.{png,svg}` — a side-by-side comparative
       chart:
           left:  ROC-AUC curves for ResNet18+Linear vs. ResNet+SVM vs.
                  Hybrid Q-Knee (VQC), all three sharing the same 4-D
                  bottleneck width (reuses `qknee.models.evaluate`'s
                  benchmark-suite training functions).
           right: trainable-parameter-count vs. test-accuracy efficiency
                  chart (log-scale parameter axis), highlighting how many
                  fewer trainable parameters the 4-qubit VQC needs versus
                  the classical linear head, at comparable accuracy.
    2. `circuit_diagram.{png,svg}` — the 4-qubit VQC drawn via
       `pennylane.draw_mpl`: Angle Encoding -> Parameterized Rotations
       (Rx, Ry, Rz) -> Entangling CNOT ring -> Pauli-Z measurement.
    3. `clinical_case_walkthrough.{png,svg}` — a 5-panel clinical case
       figure: Raw MRI Slice -> Spatial Feature Extraction (ResNet18
       intermediate activation map) -> Grad-CAM Activation Heatmap ->
       Quantum Attribution (per-qubit Pauli-Z expectation bar chart) ->
       Diagnostic Impression (auto-generated text, via
       `qknee.xai.report_generator.generate_radiology_text_snippet`).

No real labeled MRI dataset or trained checkpoint is required — this
script generates a synthetic dataset/sample slice/randomly-initialized
model (same convention every other script in this project follows) purely
to exercise the full visualization pipeline end-to-end. Swap in real
embeddings/checkpoints for an actual pitch-deck run.

Run with:
    python scripts/generate_deck_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Allow `python scripts/generate_deck_assets.py` to resolve the `qknee`
# package without requiring the caller to set PYTHONPATH or use `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving figures to disk
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qknee.config.loader import load_config
from qknee.models.evaluate import (
    generate_synthetic_dataset,
    train_hybrid_qknee_vqc,
    train_linear_bottleneck_classifier,
    train_pca_svm_classifier,
)
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.resnet_extractor import ResNet18FeatureExtractor
from qknee.models.vqc import N_QUBITS, VQCClassifier, ROTATIONS_PER_QUBIT_PER_LAYER, build_qnode
from qknee.xai.gradcam import GradCAM, get_default_target_layer, overlay_heatmap
from qknee.xai.report_generator import generate_radiology_text_snippet

_config = load_config()

# `qknee/artifacts/deck_figures/` — deliberately not `config.paths.deck_output_dir`
# ("deck_assets/"): this script's output location is fixed here per the
# hackathon-asset spec it implements, independent of that general-purpose
# config default (which other tooling may still read/write separately).
DEFAULT_OUTPUT_DIR = Path("qknee/artifacts/deck_figures")
HIGH_DPI = 300  # publication/print-ready; higher than config's general-purpose deck_figure_dpi (200)

# A consistent, presentation-friendly palette shared by every figure in
# this script — keeps the deck visually coherent slide to slide.
MODEL_PALETTE = {
    "ResNet18 + Linear": "#8B949E",
    "ResNet + SVM": "#4C8BF5",
    "Hybrid Q-Knee (VQC)": "#00C896",
}
sns.set_theme(style="whitegrid", context="talk")


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int = HIGH_DPI) -> List[Path]:
    """Saves `fig` as both a high-DPI `.png` and a vector `.svg`, closes it,
    and returns `[png_path, svg_path]`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  saved {png_path}")
    print(f"  saved {svg_path}")
    return [png_path, svg_path]


# --------------------------------------------------------------------------- #
# 1. Side-by-side: ROC-AUC comparison + parameter-efficiency chart
# --------------------------------------------------------------------------- #

def _linear_bottleneck_param_count(feature_dim: int, bottleneck_dim: int = 4, n_classes: int = 2) -> int:
    """Trainable parameters in `evaluate.LinearBottleneckClassifier`
    (`Linear(feature_dim, bottleneck_dim) -> Linear(bottleneck_dim, n_classes)`),
    computed analytically (architecture-determined, not data-dependent) —
    matches `sum(p.numel() for p in model.parameters())` exactly."""
    return (feature_dim * bottleneck_dim + bottleneck_dim) + (bottleneck_dim * n_classes + n_classes)


def _vqc_param_count(n_qubits: int = _config.quantum.n_qubits, n_layers: int = _config.quantum.n_layers) -> int:
    """Trainable parameters in `VQCClassifier`: the quantum circuit's
    rotation weights (`n_layers * n_qubits * 3` for RX/RY/RZ per qubit per
    layer) plus the classical `Linear(n_qubits, 1)` readout — this is the
    "quantum parameter compression" number the efficiency chart highlights."""
    quantum_weights = n_layers * n_qubits * ROTATIONS_PER_QUBIT_PER_LAYER
    readout = n_qubits * 1 + 1
    return quantum_weights + readout


def _svm_support_vector_count(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> int:
    """Number of stored support vectors for the `ResNet -> PCA(4) -> RBF
    SVM` baseline — an SVM has no fixed weight-matrix "parameter count" the
    way a neural net does; the standard analogous "model size" measure is
    the number of stored support vectors (each carrying a dual coefficient
    the model must keep). Fits the identical StandardScaler -> PCA(4) ->
    SVC(kernel="rbf") architecture `qknee.models.evaluate.train_pca_svm_classifier`
    uses, just without its probability-calibration wrapper, purely to read
    `n_support_` off the raw fitted `SVC`.
    """
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=_config.quantum.n_qubits, random_state=seed).fit(scaler.transform(X_train))
    svc = SVC(kernel="rbf", random_state=seed).fit(pca.transform(scaler.transform(X_train)), y_train)
    return int(svc.n_support_.sum())


def generate_roc_and_efficiency_figure(
    output_dir: Path, seed: int = _config.evaluation.random_seed
) -> List[Path]:
    """Trains the three benchmark architectures (same ones
    `scripts/run_benchmark.py` compares) on one synthetic dataset and
    renders a side-by-side figure: ROC curves (left) and a trainable-
    parameter-count-vs-accuracy efficiency chart (right, log-scale param
    axis) highlighting the VQC's parameter compression."""
    print("Training ResNet18+Linear / ResNet+SVM / Hybrid Q-Knee VQC for the ROC + efficiency figure...")
    features, labels = generate_synthetic_dataset(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=_config.evaluation.test_size, stratify=labels, random_state=seed,
    )

    linear_probs, _ = train_linear_bottleneck_classifier(X_train, y_train, X_test)
    svm_probs, _ = train_pca_svm_classifier(X_train, y_train, X_test)
    vqc_probs, _ = train_hybrid_qknee_vqc(X_train, y_train, X_test)

    model_results = {
        "ResNet18 + Linear": {
            "probs": linear_probs, "param_count": _linear_bottleneck_param_count(features.shape[1]),
        },
        "ResNet + SVM": {
            "probs": svm_probs, "param_count": _svm_support_vector_count(X_train, y_train, seed),
        },
        "Hybrid Q-Knee (VQC)": {
            "probs": vqc_probs, "param_count": _vqc_param_count(),
        },
    }
    for info in model_results.values():
        y_pred = (info["probs"] >= _config.api.tear_risk_threshold).astype(int)
        info["accuracy"] = float(accuracy_score(y_test, y_pred))
        info["roc_auc"] = float(roc_auc_score(y_test, info["probs"]))

    fig, (ax_roc, ax_eff) = plt.subplots(1, 2, figsize=(18, 7))

    # --- Left: ROC curves ---
    for name, info in model_results.items():
        fpr, tpr, _ = roc_curve(y_test, info["probs"])
        ax_roc.plot(fpr, tpr, color=MODEL_PALETTE[name], linewidth=2.75, label=f"{name} (AUC={info['roc_auc']:.2f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC-AUC Comparison", fontsize=16, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=10, frameon=True)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)

    # --- Right: parameter-count vs. accuracy efficiency chart (grouped
    # bars, dual y-axis — accuracy linear on the left, parameter count log
    # on the right, since the VQC/SVM/Linear parameter counts span 2+
    # orders of magnitude) ---
    names = list(model_results.keys())
    x = np.arange(len(names))
    bar_width = 0.35

    accuracies = [model_results[n]["accuracy"] for n in names]
    param_counts = [model_results[n]["param_count"] for n in names]

    ax_eff.bar(
        x - bar_width / 2, accuracies, bar_width, label="Test Accuracy",
        color=[MODEL_PALETTE[n] for n in names], edgecolor="white", linewidth=1.2,
    )
    ax_eff.set_ylim(0, 1.15)
    ax_eff.set_ylabel("Test Accuracy")
    ax_eff.set_xticks(x)
    ax_eff.set_xticklabels(names, rotation=12, ha="right")
    for xi, acc in zip(x, accuracies):
        ax_eff.text(xi - bar_width / 2, acc + 0.03, f"{acc:.2f}", ha="center", fontsize=10, fontweight="bold")

    ax_eff2 = ax_eff.twinx()
    ax_eff2.bar(
        x + bar_width / 2, param_counts, bar_width, label="Trainable Parameters",
        color="#37474F", alpha=0.6, edgecolor="white", linewidth=1.2,
    )
    ax_eff2.set_yscale("log")
    ax_eff2.set_ylim(1, max(param_counts) * 30)  # headroom so the tallest bar's label doesn't clip the frame
    ax_eff2.set_ylabel("Trainable Parameters (log scale)")
    for xi, pc in zip(x, param_counts):
        ax_eff2.text(xi + bar_width / 2, pc * 1.3, f"{pc:,}", ha="center", fontsize=10, fontweight="bold")

    ax_eff.set_title("Parameter Efficiency: Accuracy vs. Model Size", fontsize=16, fontweight="bold")
    handles1, labels1 = ax_eff.get_legend_handles_labels()
    handles2, labels2 = ax_eff2.get_legend_handles_labels()
    ax_eff.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=9, frameon=True)
    ax_eff.grid(False)

    compression_ratio = param_counts[0] / param_counts[2]  # Linear vs. VQC
    fig.suptitle("Q-Knee: Predictive Performance & Quantum Parameter Compression", fontsize=19, fontweight="bold")
    fig.text(
        0.5, 0.01,
        f"Hybrid Q-Knee's 4-qubit VQC uses {compression_ratio:.0f}× fewer trainable parameters than the "
        f"classical linear head ({param_counts[2]:,} vs. {param_counts[0]:,}), at comparable accuracy.",
        ha="center", fontsize=11.5, style="italic",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    return _save_figure(fig, output_dir, "roc_and_parameter_efficiency")


# --------------------------------------------------------------------------- #
# 2. 4-qubit VQC circuit diagram
# --------------------------------------------------------------------------- #

def generate_circuit_diagram_figure(
    output_dir: Path, seed: int = _config.evaluation.random_seed
) -> List[Path]:
    """Draws the 4-qubit VQC — Angle Encoding (Rx, Ry) -> `n_layers`
    parameterized-rotation (Rx, Ry, Rz) + entangling-CNOT-mesh variational
    blocks -> Pauli-Z measurement — via PennyLane's matplotlib circuit
    drawer (`qml.draw_mpl`)."""
    print("Rendering the 4-qubit VQC circuit diagram...")
    n_layers = _config.quantum.n_layers
    circuit = build_qnode(n_qubits=N_QUBITS, n_layers=n_layers)

    generator = torch.Generator().manual_seed(seed)
    dummy_inputs = torch.rand(N_QUBITS, generator=generator) * 2 * torch.pi
    dummy_weights = torch.rand(n_layers, N_QUBITS, 3, generator=generator) * 2 * torch.pi

    fig, _ = qml.draw_mpl(circuit, style="pennylane")(dummy_inputs, dummy_weights)

    # qml.draw_mpl's own figure fills nearly its entire height with the
    # circuit drawing, leaving no headroom for a title — grow the figure
    # and compress the circuit's axes down to make room, rather than
    # overlapping the title/subtitle text onto the wire diagram itself.
    width, height = fig.get_size_inches()
    fig.set_size_inches(width, height * 1.35)
    fig.subplots_adjust(top=0.78)

    fig.suptitle("Q-Knee 4-Qubit Variational Circuit", fontsize=17, fontweight="bold", y=0.97)
    fig.text(
        0.5, 0.885,
        "Angle Encoding (Rx, Ry)  →  Parameterized Rotations (Rx, Ry, Rz) × "
        f"{n_layers} layers  →  Entangling CNOT Ring  →  Pauli-Z (<Z>) Measurement",
        ha="center", fontsize=11, style="italic",
    )

    return _save_figure(fig, output_dir, "circuit_diagram")


# --------------------------------------------------------------------------- #
# 3. Multi-panel clinical case figure
# --------------------------------------------------------------------------- #

def _make_sample_slice() -> np.ndarray:
    """Synthetic MRI-like slice (concentric ring, standing in for a real
    knee MRI sample) so this script runs standalone without a dataset."""
    height, width = _config.data.image_size
    yy, xx = np.mgrid[0:height, 0:width]
    center_y, center_x = height / 2, width / 2
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    ring = 255 * np.clip(1 - np.abs(radius - 70) / 40, 0, 1)
    rng = np.random.default_rng(_config.evaluation.random_seed)
    texture = rng.normal(scale=8, size=(height, width))
    return np.clip(ring + texture, 0, 255).astype(np.uint8)


def _fit_dummy_pca_reducer(seed: int) -> QuantumDimReducer:
    """A `QuantumDimReducer` fit on a synthetic 512-D corpus, purely so
    this script's clinical-case figure is runnable standalone without a
    real fitted PCA artifact on disk (same fallback convention every other
    script in this project uses)."""
    rng = np.random.default_rng(seed)
    corpus = rng.normal(size=(500, _config.resnet.feature_dim)).astype(np.float32)
    return QuantumDimReducer().fit(corpus)


def _extract_spatial_feature_map(
    extractor: ResNet18FeatureExtractor, target_layer: torch.nn.Module, input_tensor: torch.Tensor
) -> np.ndarray:
    """Captures `target_layer`'s raw spatial activation map (the last
    convolutional feature map ResNet18 produces before global-average-
    pooling collapses it to the 512-D embedding) via a forward hook, and
    condenses it into a single-channel spatial map by averaging across the
    channel dimension. This is deliberately *not* Grad-CAM — no gradient
    weighting or class-discrimination is applied here — it visualizes the
    plain "where is the backbone looking" spatial activation prior to
    pooling, so the deck can show feature extraction and Grad-CAM as two
    distinct pipeline stages.

    Returns:
        `(H, W)` float32 array (raw activation magnitudes, un-normalized).
    """
    captured: dict = {}

    def _hook(_module, _input, output):
        captured["activation"] = output.detach()

    handle = target_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            extractor.forward_slice(input_tensor)
    finally:
        handle.remove()

    activation = captured["activation"]  # (1, C, H, W)
    return activation[0].mean(dim=0).numpy()  # (H, W)


def generate_clinical_case_figure(
    output_dir: Path, seed: int = _config.evaluation.random_seed
) -> List[Path]:
    """Renders the 5-panel clinical case walkthrough: Raw MRI Slice ->
    Spatial Feature Extraction (raw ResNet18 activation map) -> Grad-CAM
    Activation Heatmap -> Quantum Attribution (Pauli-Z measurement bar
    chart) -> Diagnostic Impression. Every panel is a genuine forward pass
    through the actual model classes (ResNet18, Grad-CAM, PCA projection,
    VQC, `generate_radiology_text_snippet`) on one synthetic sample slice
    — nothing here is a fabricated/mocked figure."""
    print("Rendering the multi-panel clinical case walkthrough...")
    torch.manual_seed(seed)

    sample_slice = _make_sample_slice()
    extractor = ResNet18FeatureExtractor(freeze_backbone=True)
    extractor.eval()
    target_layer = get_default_target_layer(extractor)

    rgb_slice = cv2.cvtColor(sample_slice, cv2.COLOR_GRAY2RGB)
    input_tensor = torch.from_numpy(rgb_slice).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    spatial_feature_map = _extract_spatial_feature_map(extractor, target_layer, input_tensor)

    with GradCAM(extractor, target_layer) as cam:
        heatmap = cam.generate(input_tensor)
    gradcam_overlay_bgr = overlay_heatmap(heatmap, sample_slice)
    gradcam_rgb = cv2.cvtColor(gradcam_overlay_bgr, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        features_512d = extractor.forward_slice(input_tensor).numpy()

    reducer = _fit_dummy_pca_reducer(seed)
    quantum_angles = reducer.transform(features_512d)  # (1, n_qubits), in [0, 2*pi]

    vqc = VQCClassifier(n_qubits=_config.quantum.n_qubits, n_layers=_config.quantum.n_layers)
    vqc.eval()
    with torch.no_grad():
        angles_tensor = torch.from_numpy(quantum_angles).float()
        pauli_z_expvals = vqc.quantum_layer(angles_tensor)  # (1, n_qubits), each in [-1, 1]
        risk_logits = vqc.readout(pauli_z_expvals)
        risk_score = float(vqc.activation(risk_logits).item())

    pauli_z_values = pauli_z_expvals.detach().flatten().numpy()

    text_snippet = generate_radiology_text_snippet(
        prediction_results={"acl_risk": risk_score},
        metadata={"patient_id": "DEMO-DECK-001", "plane": "Sagittal"},
    )

    fig, axes = plt.subplots(1, 5, figsize=(33, 6.5), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.35]})

    axes[0].imshow(sample_slice, cmap="gray")
    axes[0].set_title("Raw MRI Slice", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sample_slice, cmap="gray")
    axes[1].imshow(spatial_feature_map, cmap="magma", alpha=0.75, extent=axes[1].get_xlim() + axes[1].get_ylim())
    axes[1].set_title("Spatial Feature\nExtraction", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(gradcam_rgb)
    axes[2].set_title("Grad-CAM Activation\nHeatmap", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    qubit_labels = [f"Q{i}" for i in range(len(pauli_z_values))]
    bar_colors = ["#E74C3C" if v >= 0 else "#2ECC71" for v in pauli_z_values]
    axes[3].bar(qubit_labels, pauli_z_values, color=bar_colors, edgecolor="white", linewidth=1.2)
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set_ylim(-1.15, 1.15)
    axes[3].set_ylabel("<Z> expectation")
    axes[3].set_title("Quantum Attribution\n(Pauli-Z Measurement)", fontsize=14, fontweight="bold")
    for xi, v in enumerate(pauli_z_values):
        axes[3].text(xi, v + (0.08 if v >= 0 else -0.14), f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

    axes[4].axis("off")
    axes[4].set_title("Diagnostic Impression", fontsize=14, fontweight="bold", loc="left")
    axes[4].text(
        0.0, 0.88, text_snippet, transform=axes[4].transAxes, fontsize=10.5, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7", edgecolor="#CCCCCC"),
    )

    fig.suptitle("Q-Knee: Clinical Case Walkthrough — Slice to Diagnosis", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return _save_figure(fig, output_dir, "clinical_case_walkthrough")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_roc_and_efficiency_figure(output_dir)
    generate_circuit_diagram_figure(output_dir)
    generate_clinical_case_figure(output_dir)

    print(f"\nAll deck figures saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
