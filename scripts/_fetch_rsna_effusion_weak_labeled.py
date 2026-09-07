"""
One-off helper (not part of the package) to pull DICOM images for the 3,723
"confidently" weak-labeled Effusion studies in
qknee/artifacts/effusion_scaled_labels.csv (rows with no `needs_review`
flag) -- these UIDs come from a classifier over the UNLABELED portion of the
RSNA Knee Abnormality Detection train set (i.e. NOT the 58 officially
ground-truth-labeled studies _fetch_rsna_labeled_subset.py already pulled
into train_series/ and ../rsna-knee/train_series/).

Adapted from _fetch_rsna_labeled_subset.py (same retry/backoff/checkpoint
strategy, same train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOP>.dcm
layout, same MAX_SLICES_PER_SERIES cap) but:
    - target set is the weak-labeled UID list, not train.csv's labeled rows
    - skips any UID already fully resolved on a prior run (checkpoint) or
      already present under train_series/ (129 studies overlap with the
      earlier SSL-pretrain-pool download and don't need re-fetching)
    - own checkpoint/manifest paths so this never collides with the
      existing 58-study fetch's checkpoint/manifest

This targets nearly the entire competition file listing (3,723 of ~4,407
total studies). Observed throughput (no rate-limiting hit yet): ~1.5
min/study, i.e. the full pool would take days unattended -- so this
self-terminates after MAX_RUNTIME_SECONDS wall-clock (env var
FETCH_MAX_RUNTIME_SECONDS, default 3 hours) rather than running unbounded,
same pattern as _fetch_rsna_pretrain_pool.py. Checkpoints after every
completed study, so re-running resumes instead of re-paging from scratch.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "rsna-knee-abnormality-detection"
MAX_SLICES_PER_SERIES = 4
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "train_series"
LABELS_CSV = REPO_ROOT / "qknee" / "artifacts" / "effusion_scaled_labels.csv"
MANIFEST_PATH = REPO_ROOT / "rsna_effusion_weak_labeled_manifest.json"
CHECKPOINT_PATH = REPO_ROOT / "rsna_effusion_weak_labeled_checkpoint.json"
MAX_RUNTIME_SECONDS = float(os.environ.get("FETCH_MAX_RUNTIME_SECONDS", str(40 * 60 * 60)))
TARGET_TOTAL_STUDIES = int(os.environ.get("FETCH_TARGET_TOTAL_STUDIES", "1000"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_rate_limit_state = {"delay": 0.3}


def with_retry(fn, *args, max_wait: float = 60.0, **kwargs):
    if _rate_limit_state["delay"] > 0:
        time.sleep(_rate_limit_state["delay"])
    while True:
        try:
            result = fn(*args, **kwargs)
            _rate_limit_state["delay"] = max(0.0, _rate_limit_state["delay"] * 0.9)
            return result
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429 or (status is not None and 500 <= status < 600):
                _rate_limit_state["delay"] = min(max(_rate_limit_state["delay"] * 2, 2.0), max_wait)
                log(f"HTTP {status} — backing off {_rate_limit_state['delay']:.1f}s "
                    f"(now the standing per-request delay too) before retrying...")
                time.sleep(_rate_limit_state["delay"])
                continue
            raise
        except requests.exceptions.RequestException as exc:
            _rate_limit_state["delay"] = min(max(_rate_limit_state["delay"] * 2, 2.0), max_wait)
            log(f"{type(exc).__name__} — backing off {_rate_limit_state['delay']:.1f}s before retrying... ({exc})")
            time.sleep(_rate_limit_state["delay"])
            continue


def save_checkpoint(page_token, resolved_uids: set, downloaded_uids: set) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps({
            "page_token": page_token,
            "resolved_uids": sorted(resolved_uids),
            "downloaded_uids": sorted(downloaded_uids),
        }),
        encoding="utf-8",
    )


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return None, set(), set()
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return data.get("page_token"), set(data.get("resolved_uids", [])), set(data.get("downloaded_uids", []))


def load_weak_labeled_targets() -> set:
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    confident = [r["StudyInstanceUID"] for r in rows if not r["needs_review"].strip()]
    return set(confident)


def already_on_disk(study_uid: str, study_series_planes: dict) -> bool:
    """A study counts as already fetched if every known series directory
    exists under OUT_ROOT with >=1 file (mirrors download_study's
    per-file existence skip, but at study granularity so we don't even
    need to resolve it via the file listing)."""
    known = study_series_planes.get(study_uid, {})
    if not known:
        return False
    for series_uid in known:
        series_dir = OUT_ROOT / study_uid / series_uid
        if not series_dir.is_dir() or not any(series_dir.iterdir()):
            return False
    return True


def download_study(api: KaggleApi, study_uid: str, series_dict: dict) -> int:
    n = 0
    for series_uid, sop_files in series_dict.items():
        if not sop_files:
            continue
        dest_dir = OUT_ROOT / study_uid / series_uid
        dest_dir.mkdir(parents=True, exist_ok=True)
        for sop_file in sop_files:
            dest_file = dest_dir / sop_file
            if dest_file.exists():
                continue
            remote_path = f"train_series/{study_uid}/{series_uid}/{sop_file}"
            with_retry(api.competition_download_file, COMPETITION, remote_path, path=str(dest_dir), force=True)
            n += 1
    return n


def main() -> None:
    target_studies = load_weak_labeled_targets()
    log(f"target: {len(target_studies)} confidently weak-labeled studies")

    series_meta = pd.read_csv(REPO_ROOT / "train_series.csv")
    series_meta["StudyInstanceUID"] = series_meta["StudyInstanceUID"].astype(str)
    series_meta["SeriesInstanceUID"] = series_meta["SeriesInstanceUID"].astype(str)
    study_series_planes: dict[str, dict[str, str]] = {}
    for study_uid, group in series_meta[series_meta["StudyInstanceUID"].isin(target_studies)].groupby("StudyInstanceUID"):
        study_series_planes[study_uid] = dict(zip(group["SeriesInstanceUID"], group["Anatomical_Plane"]))
    total_known_series = sum(len(v) for v in study_series_planes.values())
    log(f"these studies have {total_known_series} known series total (across all planes)")

    api = KaggleApi()
    api.authenticate()

    page_token, resolved_uids, downloaded_uids = load_checkpoint()
    if page_token is not None or resolved_uids:
        log(f"resuming from checkpoint: {len(resolved_uids)} resolved, {len(downloaded_uids)} already downloaded, "
            f"page_token={'<set>' if page_token else None}")

    pre_skipped = 0
    for study_uid in list(target_studies):
        if study_uid in downloaded_uids:
            continue
        if already_on_disk(study_uid, study_series_planes):
            downloaded_uids.add(study_uid)
            resolved_uids.add(study_uid)
            pre_skipped += 1
    if pre_skipped:
        log(f"pre-skipped {pre_skipped} studies already fully present on disk from prior downloads")
        save_checkpoint(page_token, resolved_uids, downloaded_uids)

    collected: dict[str, dict[str, list[str]]] = {uid: {} for uid in target_studies}
    done_series: set[tuple[str, str]] = set()

    def study_is_done(study_uid: str) -> bool:
        known = study_series_planes.get(study_uid, {})
        return bool(known) and all((study_uid, s) in done_series for s in known)

    remaining = target_studies - downloaded_uids
    pages_seen = 0
    entries_seen = 0
    reached_train = page_token is not None
    n_downloaded_total = 0
    t_start = time.time()

    log(f"{len(remaining)} studies remaining to fetch; target total downloaded = "
        f"{TARGET_TOTAL_STUDIES} (currently {len(downloaded_uids)}); will self-terminate after "
        f"{MAX_RUNTIME_SECONDS/3600:.1f}h wall-clock this run regardless")

    while remaining:
        if len(downloaded_uids) >= TARGET_TOTAL_STUDIES:
            log(f"reached TARGET_TOTAL_STUDIES ({TARGET_TOTAL_STUDIES}) -- stopping "
                f"with {len(remaining)} studies still unfetched")
            break
        if time.time() - t_start > MAX_RUNTIME_SECONDS:
            log(f"MAX_RUNTIME_SECONDS ({MAX_RUNTIME_SECONDS:.0f}s) reached -- stopping this run "
                f"with {len(remaining)} studies still unresolved (resume later to continue)")
            break
        resp = with_retry(api.competition_list_files, COMPETITION, page_token=page_token, page_size=200)
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
                log(f"reached train_series/ after {pages_seen} pages this run / {entries_seen} entries this run")

            parts = name.split("/")
            if len(parts) != 4:
                continue
            _, study_uid, series_uid, sop_file = parts
            if study_uid not in remaining:
                continue
            known_series = study_series_planes.get(study_uid, {})
            if series_uid not in known_series:
                continue
            if (study_uid, series_uid) in done_series:
                continue

            slot = collected[study_uid].setdefault(series_uid, [])
            if len(slot) < MAX_SLICES_PER_SERIES:
                slot.append(sop_file)
            if len(slot) >= MAX_SLICES_PER_SERIES:
                done_series.add((study_uid, series_uid))

        newly_done = [uid for uid in list(remaining) if study_is_done(uid)]
        for study_uid in newly_done:
            n = download_study(api, study_uid, collected[study_uid])
            n_downloaded_total += n
            resolved_uids.add(study_uid)
            downloaded_uids.add(study_uid)
            remaining.discard(study_uid)
            elapsed_min = (time.time() - t_start) / 60
            log(f"{len(downloaded_uids)}/{len(target_studies)} weak-labeled studies resolved+downloaded "
                f"({n} file(s) this study; pages={pages_seen}, entries={entries_seen}, elapsed_this_run={elapsed_min:.1f}min)")
            save_checkpoint(page_token, resolved_uids, downloaded_uids)

        if pages_seen % 50 == 0:
            elapsed_min = (time.time() - t_start) / 60
            log(f"progress: pages={pages_seen}, entries={entries_seen}, "
                f"resolved={len(downloaded_uids)}/{len(target_studies)}, elapsed_this_run={elapsed_min:.1f}min")
            save_checkpoint(page_token, resolved_uids, downloaded_uids)

        if page_token is None:
            log(f"exhausted file listing with {len(remaining)} studies still unresolved: {len(remaining)} UIDs")
            break

    log(f"listing done: pages={pages_seen}, entries={entries_seen}, "
        f"resolved={len(downloaded_uids)}/{len(target_studies)}")

    manifest: dict[str, dict[str, list[str]]] = {}
    for study_uid in downloaded_uids:
        series_dict = collected.get(study_uid, {})
        manifest[study_uid] = {}
        for series_uid, sop_files in series_dict.items():
            if not sop_files:
                continue
            plane = study_series_planes[study_uid][series_uid]
            manifest[study_uid].setdefault(plane, []).append(series_uid)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"DONE: {len(manifest)}/{len(target_studies)} weak-labeled studies with >=1 usable series, "
        f"{n_downloaded_total} files downloaded this run, manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
