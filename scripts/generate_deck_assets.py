"""
Generates presentation assets for the Q-Knee pitch deck:

    1. `performance_comparison.png` — grouped bar chart of ROC-AUC,
       Sensitivity, and Specificity for Baseline ResNet18, Baseline SVM,
       and the Q-Knee Hybrid VQC pipeline (reuses the training/eval
       functions from evaluate.py on the same synthetic dataset).
    2. A sample-slice explainability triptych:
           - original_slice.png      - the raw MRI slice
           - gradcam_overlay.png     - Grad-CAM attention overlay (gradcam.py)
           - circuit_diagram.png     - the 4-qubit VQC drawn via PennyLane's
                                       matplotlib circuit drawer
           - slice_gradcam_circuit_composite.png - all three side-by-side,
             ready to drop directly into a slide.

All outputs are written to `deck_assets/`.

Run with:
    python generate_deck_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/generate_deck_assets.py` to resolve the `qknee`
# package without requiring the caller to set PYTHONPATH or use `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving figures to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split

from qknee.config.loader import load_config
from qknee.models.evaluate import (
    compute_metrics,
    generate_synthetic_dataset,
    train_quantum_vqc,
    train_resnet_linear_baseline,
    train_svm_baseline,
)
from qknee.xai.gradcam import GradCAM, get_default_target_layer, overlay_heatmap
from qknee.models.resnet_extractor import ResNet18FeatureExtractor
from qknee.models.vqc import N_QUBITS, build_qnode

_config = load_config()
OUTPUT_DIR = _config.paths.deck_output_dir

# A consistent, presentation-friendly palette shared by every figure in
# this script — keeps the deck visually coherent slide to slide.
MODEL_PALETTE = {
    "Baseline ResNet18": "#8B949E",
    "Baseline SVM": "#4C8BF5",
    "Q-Knee Hybrid VQC": "#00C896",
}
sns.set_theme(style="whitegrid", context="talk")


# --------------------------------------------------------------------------- #
# 1. Comparative performance bar chart
# --------------------------------------------------------------------------- #

def generate_performance_comparison_figure(
    output_dir: Path, seed: int = _config.evaluation.random_seed
) -> Path:
    """Trains the three models from evaluate.py on one synthetic dataset and
    renders a grouped bar chart (metric on the x-axis, model as hue) of
    ROC-AUC / Sensitivity / Specificity."""
    print("Training models for the performance comparison chart...")
    features, labels = generate_synthetic_dataset(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=_config.evaluation.test_size, stratify=labels, random_state=seed
    )

    resnet_probs = train_resnet_linear_baseline(X_train, y_train, X_test)
    svm_probs = train_svm_baseline(X_train, y_train, X_test)
    vqc_probs = train_quantum_vqc(X_train, y_train, X_test)

    results = {
        "Baseline ResNet18": compute_metrics("Baseline ResNet18", y_test, resnet_probs),
        "Baseline SVM": compute_metrics("Baseline SVM", y_test, svm_probs),
        "Q-Knee Hybrid VQC": compute_metrics("Q-Knee Hybrid VQC", y_test, vqc_probs),
    }

    records = []
    for model_name, metrics in results.items():
        records.append({"Model": model_name, "Metric": "ROC-AUC", "Score": metrics.roc_auc})
        records.append({"Model": model_name, "Metric": "Sensitivity", "Score": metrics.sensitivity})
        records.append({"Model": model_name, "Metric": "Specificity", "Score": metrics.specificity})
    df = pd.DataFrame.from_records(records)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.barplot(
        data=df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=MODEL_PALETTE,
        hue_order=list(MODEL_PALETTE.keys()),
        ax=ax,
        edgecolor="white",
        linewidth=1.2,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=11)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Q-Knee Hybrid VQC vs. Classical Baselines", fontsize=18, fontweight="bold", pad=16)
    ax.legend(title=None, loc="lower right", frameon=True)
    sns.despine(left=True)
    fig.tight_layout()

    output_path = output_dir / "performance_comparison.png"
    fig.savefig(output_path, dpi=_config.evaluation.deck_figure_dpi)
    plt.close(fig)
    print(f"  saved {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
# 2. Sample-slice explainability triptych
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


def _render_gradcam_overlay(sample_slice: np.ndarray) -> np.ndarray:
    """Runs the sample slice through the frozen ResNet18 backbone and
    returns an RGB Grad-CAM overlay (gradcam.py)."""
    torch.manual_seed(_config.evaluation.random_seed)
    extractor = ResNet18FeatureExtractor(freeze_backbone=True)
    extractor.eval()
    target_layer = get_default_target_layer(extractor)

    rgb_slice = cv2.cvtColor(sample_slice, cv2.COLOR_GRAY2RGB)
    input_tensor = torch.from_numpy(rgb_slice).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with GradCAM(extractor, target_layer) as cam:
        heatmap = cam.generate(input_tensor)

    overlay_bgr = overlay_heatmap(heatmap, sample_slice)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def _render_circuit_diagram(output_dir: Path) -> tuple[Path, np.ndarray]:
    """Draws the 4-qubit VQC (angle encoding + 3 variational layers) using
    PennyLane's matplotlib circuit drawer and saves it as its own PNG.

    Returns the saved path and the rendered image as an RGB array (for
    embedding in the composite figure).
    """
    n_layers = _config.quantum.n_layers
    circuit = build_qnode(n_qubits=N_QUBITS, n_layers=n_layers)

    generator = torch.Generator().manual_seed(_config.evaluation.random_seed)
    dummy_inputs = torch.rand(N_QUBITS, generator=generator) * 2 * torch.pi
    dummy_weights = torch.rand(n_layers, N_QUBITS, 3, generator=generator) * 2 * torch.pi

    fig, _ = qml.draw_mpl(circuit, style="pennylane")(dummy_inputs, dummy_weights)
    fig.suptitle("Q-Knee 4-Qubit Variational Circuit", fontsize=14, fontweight="bold")

    output_path = output_dir / "circuit_diagram.png"
    fig.savefig(output_path, dpi=_config.evaluation.deck_figure_dpi, bbox_inches="tight")

    fig.canvas.draw()
    image_array = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    return output_path, image_array


def generate_slice_gradcam_circuit_assets(output_dir: Path) -> Path:
    """Produces the original slice, Grad-CAM overlay, and circuit diagram
    as individual PNGs, plus one side-by-side composite PNG."""
    print("Rendering sample slice / Grad-CAM overlay / circuit diagram...")

    sample_slice = _make_sample_slice()
    gradcam_rgb = _render_gradcam_overlay(sample_slice)
    circuit_path, circuit_rgb = _render_circuit_diagram(output_dir)

    original_path = output_dir / "original_slice.png"
    cv2.imwrite(str(original_path), sample_slice)
    print(f"  saved {original_path}")

    gradcam_path = output_dir / "gradcam_overlay.png"
    cv2.imwrite(str(gradcam_path), cv2.cvtColor(gradcam_rgb, cv2.COLOR_RGB2BGR))
    print(f"  saved {gradcam_path}")
    print(f"  saved {circuit_path}")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), gridspec_kw={"width_ratios": [1, 1, 2]})

    axes[0].imshow(sample_slice, cmap="gray")
    axes[0].set_title("Original MRI Slice", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gradcam_rgb)
    axes[1].set_title("Grad-CAM Attention Overlay", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(circuit_rgb)
    axes[2].set_title("4-Qubit VQC Circuit", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle("Q-Knee: From MRI Slice to Quantum Decision", fontsize=18, fontweight="bold")
    fig.tight_layout()

    composite_path = output_dir / "slice_gradcam_circuit_composite.png"
    fig.savefig(composite_path, dpi=_config.evaluation.deck_figure_dpi)
    plt.close(fig)
    print(f"  saved {composite_path}")

    return composite_path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_performance_comparison_figure(OUTPUT_DIR)
    generate_slice_gradcam_circuit_assets(OUTPUT_DIR)

    print(f"\nAll deck assets saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
