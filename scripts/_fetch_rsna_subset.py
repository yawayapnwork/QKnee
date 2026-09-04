"""
One-off helper (not part of the package) to pull a small REAL subset of the
RSNA Knee Abnormality Detection competition's train_series/ DICOMs into the
exact <StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm layout
RSNAKneeDataset expects.

Strategy: competition_list_files() is capped at 200 entries/page and has no
prefix filter, and "test_series/..." sorts before "train_series/..." — so we
must page through the whole test_series block first. To keep the actual
download small, once inside train_series/ we take only the first
MAX_SLICES_PER_SERIES files per series (not the full 20-45-slice volume) —
enough for DataIngestion.preprocess() to build a real (S,3,224,224) stack,
just a shorter S.

Writes:
    train_series/<UID>/<SeriesUID>/<SOP>.dcm   - the downloaded slices
    rsna_subset_manifest.json                   - which studies/series/files
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "rsna-knee-abnormality-detection"
TARGET_STUDIES = 75          # within the requested 50-100 range
MAX_SLICES_PER_SERIES = 4    # real slices, just not the full volume
REQUIRED_PLANES = {"Sagittal", "Coronal", "Axial"}
OUT_ROOT = Path(__file__).resolve().parent.parent / "train_series"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "rsna_subset_manifest.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    api = KaggleApi()
    api.authenticate()

    # study_uid -> plane -> series_uid -> [sop_uid, ...]
    collected: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(lambda: defaultdict(dict))
    complete_studies: set[str] = set()

    # train_series.csv (already downloaded) maps SeriesInstanceUID -> Anatomical_Plane
    import pandas as pd
    series_meta = pd.read_csv(Path(__file__).resolve().parent.parent / "train_series.csv")
    series_to_plane = dict(zip(series_meta["SeriesInstanceUID"], series_meta["Anatomical_Plane"]))

    page_token = None
    pages_seen = 0
    entries_seen = 0
    reached_train = False

    while len(complete_studies) < TARGET_STUDIES:
        resp = api.competition_list_files(COMPETITION, page_token=page_token, page_size=200)
        files = getattr(resp, "files", resp)
        page_token = getattr(resp, "next_page_token", None) or getattr(resp, "nextPageToken", None)
        pages_seen += 1
        entries_seen += len(files)

        for f in files:
            name = f.name
            if not name.startswith("train_series/"):
                continue
            if not reached_train:
                reached_train = True
                log(f"reached train_series/ after {pages_seen} pages / {entries_seen} entries")

            parts = name.split("/")
            if len(parts) != 4:  # train_series/<study>/<series>/<sop>.dcm
                continue
            _, study_uid, series_uid, sop_file = parts
            if study_uid in complete_studies:
                continue

            plane = series_to_plane.get(series_uid)
            if plane not in REQUIRED_PLANES:
                continue  # series not in our lookup, or plane not one of the 3

            slot = collected[study_uid][plane].setdefault(series_uid, [])
            if len(slot) < MAX_SLICES_PER_SERIES:
                slot.append(sop_file)

            planes_ready = {
                p for p, series_dict in collected[study_uid].items()
                if any(len(v) > 0 for v in series_dict.values())
            }
            if planes_ready >= REQUIRED_PLANES and study_uid not in complete_studies:
                complete_studies.add(study_uid)
                if len(complete_studies) % 5 == 0 or len(complete_studies) == TARGET_STUDIES:
                    log(f"{len(complete_studies)}/{TARGET_STUDIES} studies with all 3 planes found "
                        f"(pages={pages_seen}, entries={entries_seen})")

        if page_token is None:
            log("exhausted file listing before reaching target study count")
            break

    log(f"listing done: {len(complete_studies)} studies ready to download "
        f"(pages={pages_seen}, entries={entries_seen})")

    # --- download phase ---
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    n_downloaded = 0
    n_studies_done = 0
    for study_uid in list(complete_studies)[:TARGET_STUDIES]:
        manifest[study_uid] = {}
        for plane, series_dict in collected[study_uid].items():
            for series_uid, sop_files in series_dict.items():
                if not sop_files:
                    continue
                dest_dir = OUT_ROOT / study_uid / series_uid
                dest_dir.mkdir(parents=True, exist_ok=True)
                manifest[study_uid].setdefault(plane, []).append(series_uid)
                for sop_file in sop_files:
                    remote_path = f"train_series/{study_uid}/{series_uid}/{sop_file}"
                    dest_file = dest_dir / sop_file
                    if dest_file.exists():
                        continue
                    api.competition_download_file(COMPETITION, remote_path, path=str(dest_dir), force=True)
                    n_downloaded += 1
                    if n_downloaded % 25 == 0:
                        log(f"downloaded {n_downloaded} file(s)...")
        n_studies_done += 1
        if n_studies_done % 10 == 0:
            log(f"downloaded {n_studies_done}/{len(complete_studies)} studies' worth of files")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"DONE: {len(manifest)} studies, {n_downloaded} files downloaded, manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
