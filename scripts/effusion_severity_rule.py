"""
Codified severity-hedge-language rule for the Effusion label, built and
validated against the 58 ground-truth-labeled RSNA Knee reports.

Background: qknee/data audit (see RESULTS.md #2) found that identical/
near-identical severity phrasing ("small joint effusion", "mild effusion",
"some amount of effusion", etc.) is mapped to *opposite* Effusion labels in
a large fraction of the 58 labeled studies. This script makes that finding
explicit and actionable: it defines a severity-tier lookup table, fits it
against the 58 labels (majority vote per tier), and reports the ceiling
accuracy achievable by *any* pure keyword rule on this label convention —
the number that should be trusted (or not) before applying the same rule to
the 4,349 unlabeled reports.

Usage:
    # Validate the rule against the 58 ground-truth-labeled reports:
    PYTHONPATH=. python scripts/effusion_severity_rule.py \
        --input rsna_effusion_audit_input.csv

    # Apply it to the unlabeled reports in train.csv (rows with a blank
    # Effusion column), assigning a label only for the tiers validated
    # above chance on the 58-set. The MILD tier (mild/small/some amount
    # of) is a coin flip there (53%) and is always left unlabeled here,
    # regardless of what TIER_TO_LABEL says, so it can't silently degrade
    # if that dict is ever edited.
    PYTHONPATH=. python scripts/effusion_severity_rule.py \
        --apply --input train.csv \
        --output qknee/artifacts/effusion_scaled_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict

# Effusion-concept anchor, multilingual (observed in the 58-report sample:
# English, Spanish, Turkish, Croatian/Bosnian, Bulgarian, Greek, Dutch,
# German).
ANCHOR_RE = re.compile(
    r"effusion|derrame|излив|opzetting|vocht|υγρού|υγρό|joint fluid|"
    r"gewrichtsvocht|efusion|efus[aã]o|ergu[sş]|s[iı]v[iı]|ef[üu]zyon|"
    r"izljev|izliv",
    re.IGNORECASE,
)

NEGATION_RE = re.compile(
    r"\bno\b|\bwithout\b|\bsin\b|\bgeen\b|\bkein\w*\b",
    re.IGNORECASE,
)

# Tier regexes, checked in this priority order against the qualifier text
# surrounding each anchor match. First match wins.
TIER_PATTERNS = [
    ("LARGE", re.compile(
        r"large|massive|extensive|opsežan|opsezan|yaygın|yaygin|büyük|buyuk",
        re.IGNORECASE)),
    ("MODERATE", re.compile(
        r"moderate|matige|umjeren|µέτρια|μέτρια|ικανή|ikani",
        re.IGNORECASE)),
    ("TRACE", re.compile(
        r"trace|minimal|geringer?|minimalni|минимален|slight",
        re.IGNORECASE)),
    ("MILD", re.compile(
        r"mild|small|some|leve|licht|blago|hafif|manja|küçük|kucuk",
        re.IGNORECASE)),
]

# Best-fit tier -> predicted label, derived from majority vote against the
# 58 ground-truth labels (see the crosstab this script prints).
TIER_TO_LABEL = {
    "NONE": 0,
    "TRACE": 0,
    "MILD": 1,        # coin flip in the 58-set (11 pos / 11 neg) — see note
    "MODERATE": 1,
    "LARGE": 1,
    "UNQUALIFIED": 1,  # effusion mentioned, no severity qualifier at all
}


def classify_report(text: str) -> str:
    """Return the severity tier for the *last* (most conclusion-proximal)
    effusion mention in the report, or 'NONE' if no positive mention
    survives negation / no anchor is found at all."""
    matches = list(ANCHOR_RE.finditer(text))
    if not matches:
        return "NONE"

    # Radiology reports typically restate findings in an Impression/
    # Conclusion section; the last mention is the closest proxy to that
    # final read, so prefer it over the first.
    for m in reversed(matches):
        window_before = text[max(0, m.start() - 25): m.start()]
        window = text[max(0, m.start() - 40): min(len(text), m.end() + 40)]

        if NEGATION_RE.search(window_before):
            return "NONE"

        for tier, pat in TIER_PATTERNS:
            if pat.search(window):
                return tier

        return "UNQUALIFIED"

    return "NONE"


# Tiers whose validated accuracy on the 58-set was better than a coin
# flip and are therefore eligible to receive a scaled label. MILD is
# deliberately excluded here (not just left out of a "confident" filter
# downstream) so applying the rule can never silently label it even if
# TIER_TO_LABEL is edited later.
SCALABLE_TIERS = {"NONE", "TRACE", "MODERATE", "LARGE", "UNQUALIFIED"}


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate(rows: list[dict]) -> tuple[dict, int]:
    """Fit-and-evaluate the tier->label rule against ground-truth-labeled
    rows. Returns (bucket of tier -> [true labels], n correct)."""
    bucket = defaultdict(list)
    correct = 0
    for row in rows:
        label = int(float(row["Effusion"]))
        tier = classify_report(row["Report"])
        pred = TIER_TO_LABEL[tier]
        bucket[tier].append(label)
        correct += int(pred == label)
    return bucket, correct


def print_validation(rows: list[dict]) -> dict:
    bucket, correct = validate(rows)
    print(f"n reports: {len(rows)}\n")
    print(f"{'Tier':12s} {'n':>3s} {'pos':>4s} {'neg':>4s} {'pred':>5s} {'tier acc':>9s}")
    tier_accuracy = {}
    for tier in ["NONE", "TRACE", "MILD", "MODERATE", "LARGE", "UNQUALIFIED"]:
        labels = bucket.get(tier, [])
        if not labels:
            continue
        c = Counter(labels)
        pred = TIER_TO_LABEL[tier]
        tier_correct = sum(1 for l in labels if l == pred)
        acc = tier_correct / len(labels)
        tier_accuracy[tier] = acc
        print(f"{tier:12s} {len(labels):3d} {c.get(1,0):4d} {c.get(0,0):4d} "
              f"{pred:5d} {tier_correct}/{len(labels)} = {acc:.0%}")

    print(f"\nOverall rule accuracy on the 58 labeled reports: "
          f"{correct}/{len(rows)} = {correct/len(rows):.1%}")
    return tier_accuracy


def apply_to_unlabeled(
    unlabeled_rows: list[dict], tier_accuracy: dict, output_path: str
) -> None:
    fieldnames = [
        "StudyInstanceUID", "tier", "assigned_label",
        "tier_accuracy_on_58set", "needs_review",
    ]
    tier_counts = Counter()
    n_labeled = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in unlabeled_rows:
            tier = classify_report(row["Report"])
            tier_counts[tier] += 1
            confident = tier in SCALABLE_TIERS
            if confident:
                n_labeled += 1
            w.writerow({
                "StudyInstanceUID": row["StudyInstanceUID"],
                "tier": tier,
                "assigned_label": TIER_TO_LABEL[tier] if confident else "",
                "tier_accuracy_on_58set": f"{tier_accuracy.get(tier, float('nan')):.2f}",
                "needs_review": "" if confident else "1",
            })

    n_total = len(unlabeled_rows)
    n_unresolved = n_total - n_labeled
    print(f"\nApplied to {n_total} unlabeled reports (train.csv rows with blank Effusion):")
    print(f"{'Tier':12s} {'n':>5s}  scaled?")
    for tier in ["NONE", "TRACE", "MILD", "MODERATE", "LARGE", "UNQUALIFIED"]:
        n = tier_counts.get(tier, 0)
        print(f"{tier:12s} {n:5d}  {'yes' if tier in SCALABLE_TIERS else 'NO (left null)'}")
    print(f"\nConfidently labeled: {n_labeled}/{n_total} = {n_labeled/n_total:.1%}")
    print(f"Left unresolved (needs_review): {n_unresolved}/{n_total} = {n_unresolved/n_total:.1%}")
    print(f"\nWrote {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="rsna_effusion_audit_input.csv",
                     help="CSV with StudyInstanceUID, Report, Effusion columns")
    ap.add_argument("--apply", action="store_true",
                     help="Apply the rule to rows in --input with a blank "
                          "Effusion column, instead of validating against "
                          "labeled rows.")
    ap.add_argument("--reference", default="rsna_effusion_audit_input.csv",
                     help="Ground-truth-labeled CSV used to fit/validate "
                          "the tier->label rule when --apply is set.")
    ap.add_argument("--output", default="qknee/artifacts/effusion_scaled_labels.csv")
    args = ap.parse_args()

    if not args.apply:
        rows = load_rows(args.input)
        print_validation(rows)
        return

    reference_rows = load_rows(args.reference)
    tier_accuracy = print_validation(reference_rows)

    all_rows = load_rows(args.input)
    unlabeled_rows = [r for r in all_rows if r.get("Effusion", "") == ""]
    apply_to_unlabeled(unlabeled_rows, tier_accuracy, args.output)


if __name__ == "__main__":
    main()
