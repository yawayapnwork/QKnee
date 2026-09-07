"""
One-off helper (not part of the package) to grow the REAL, unlabeled RSNA
Knee Abnormality Detection train_series/ pool used for self-supervised
backbone pretraining (see qknee/models/ssl_pretrain.py).

Unlike _fetch_rsna_labeled_subset.py (which hunts down 58 SPECIFIC,
scattered StudyInstanceUIDs — slow, since competition_list_files() has no
prefix filter and must page past every non-matching entry), this script
takes WHATEVER new studies it encounters once paging reaches train_series/,
skipping any StudyInstanceUID already present on disk (in either
train_series/ or the sibling ../rsna-knee/train_series/ pull). No label
awareness is needed for self-supervised pretraining, so this is much faster
per new study than the labeled-subset fetch.

Self-terminates after MAX_RUNTIME_SECONDS wall-clock (default 40 min) OR
once TARGET_NEW_STUDIES new studies are collected, whichever comes first —
a bulk download job like this must not run unbounded. Checkpoints
(page_token + downloaded UIDs) after every completed study, so a re-run
resumes instead of re-paging from the start.

Writes into train_series/ (same root every other fetch script uses; study
UIDs don't collide across runs) plus rsna_pretrain_pool_manifest.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "rsna-knee-abnormality-detection"
MAX_SLICES_PER_SERIES = 4
TARGET_NEW_STUDIES = int(os.environ.get("FETCH_TARGET_NEW_STUDIES", "2000"))
MAX_RUNTIME_SECONDS = float(os.environ.get("FETCH_MAX_RUNTIME_SECONDS", str(40 * 60)))

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "train_series"
SIBLING_ROOT = REPO_ROOT.parent / "rsna-knee" / "train_series"
MANIFEST_PATH = REPO_ROOT / "rsna_pretrain_pool_manifest.json"
CHECKPOINT_PATH = REPO_ROOT / "rsna_pretrain_pool_checkpoint.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_rate_limit_state = {"delay": 0.3}


def with_retry(fn, *args, max_wait: float = 60.0, **kwargs):
    """Retries fn(*args, **kwargs) with shared, ratcheting backoff on
    429/5xx and connection-level failures. See _fetch_rsna_labeled_subset.py
    for the full rationale — this is the same implementation."""
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
                log(f"HTTP {status} — backing off {_rate_limit_state['delay']:.1f}s before retrying...")
                time.sleep(_rate_limit_state["delay"])
                continue
            raise
        except requests.exceptions.RequestException as exc:
            _rate_limit_state["delay"] = min(max(_rate_limit_state["delay"] * 2, 2.0), max_wait)
            log(f"{type(exc).__name__} — backing off {_rate_limit_state['delay']:.1f}s before retrying... ({exc})")
            time.sleep(_rate_limit_state["delay"])
            continue


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return None, set()
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return data.get("page_token"), set(data.get("downloaded_uids", []))


def save_checkpoint(page_token, downloaded_uids: set) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps({"page_token": page_token, "downloaded_uids": sorted(downloaded_uids)}),
        encoding="utf-8",
    )


def existing_study_uids() -> set:
    uids = set()
    if OUT_ROOT.is_dir():
        uids |= {p.name for p in OUT_ROOT.iterdir() if p.is_dir()}
    if SIBLING_ROOT.is_dir():
        uids |= {p.name for p in SIBLING_ROOT.iterdir() if p.is_dir()}
    return uids


def main() -> None:
    api = KaggleApi()
    api.authenticate()

    already_on_disk = existing_study_uids()
    log(f"{len(already_on_disk)} study UID(s) already on disk (across train_series/ + sibling) — will be skipped")

    checkpoint_page_token, checkpoint_downloaded = load_checkpoint()
    downloaded_this_run: set = set()
    skip_uids = already_on_disk | checkpoint_downloaded
    if checkpoint_downloaded:
        log(f"resuming from checkpoint: {len(checkpoint_downloaded)} already downloaded this campaign")

    collected: dict = defaultdict(lambda: defaultdict(list))  # study_uid -> series_uid -> [sop_file, ...]
    complete_studies: set = set()

    page_token = checkpoint_page_token
    pages_seen = 0
    entries_seen = 0
    reached_train = page_token is not None
    t_start = time.time()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    def flush_study(study_uid: str) -> int:
        n = 0
        for series_uid, sop_files in collected[study_uid].items():
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
        del collected[study_uid]
        return n

    import shutil

    MIN_FREE_BYTES = float(os.environ.get("FETCH_MIN_FREE_GB", "4")) * (1024 ** 3)

    n_downloaded_total = 0
    while len(downloaded_this_run) < TARGET_NEW_STUDIES:
        elapsed = time.time() - t_start
        if elapsed > MAX_RUNTIME_SECONDS:
            log(f"time budget ({MAX_RUNTIME_SECONDS:.0f}s) reached — stopping cleanly")
            break

        free_bytes = shutil.disk_usage(REPO_ROOT).free
        if free_bytes < MIN_FREE_BYTES:
            log(f"disk free space ({free_bytes / 1024**3:.1f}GB) below safety floor — stopping cleanly")
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
                log(f"reached train_series/ after {pages_seen} page(s) this run / {entries_seen} entries this run")

            parts = name.split("/")
            if len(parts) != 4:
                continue
            _, study_uid, series_uid, sop_file = parts
            if study_uid in skip_uids or study_uid in complete_studies:
                continue

            slot = collected[study_uid][series_uid]
            if len(slot) < MAX_SLICES_PER_SERIES:
                slot.append(sop_file)

            if sum(len(v) for v in collected[study_uid].values()) >= MAX_SLICES_PER_SERIES:
                complete_studies.add(study_uid)
                n = flush_study(study_uid)
                n_downloaded_total += n
                downloaded_this_run.add(study_uid)
                skip_uids.add(study_uid)
                if len(downloaded_this_run) % 10 == 0:
                    elapsed_min = (time.time() - t_start) / 60
                    log(
                        f"{len(downloaded_this_run)}/{TARGET_NEW_STUDIES} new studies downloaded "
                        f"(pages={pages_seen}, entries={entries_seen}, elapsed={elapsed_min:.1f}min)"
                    )
                    save_checkpoint(page_token, checkpoint_downloaded | downloaded_this_run)

                if len(downloaded_this_run) >= TARGET_NEW_STUDIES:
                    break

        if page_token is None:
            log("exhausted the full competition file listing")
            break

    save_checkpoint(page_token, checkpoint_downloaded | downloaded_this_run)

    manifest = {
        study_uid: {series_uid: sop_files for series_uid, sop_files in collected.get(study_uid, {}).items()}
        for study_uid in downloaded_this_run
    }
    # collected entries were deleted on flush; record just the UID list instead
    manifest_uids = sorted(downloaded_this_run)
    MANIFEST_PATH.write_text(json.dumps({"downloaded_this_run": manifest_uids}, indent=2), encoding="utf-8")

    elapsed_min = (time.time() - t_start) / 60
    log(
        f"DONE: {len(downloaded_this_run)} new studies downloaded this run "
        f"({n_downloaded_total} files, {elapsed_min:.1f}min), manifest -> {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
