"""
One-off helper (not part of the package) to pull the REAL, fully-labeled
RSNA Knee Abnormality Detection studies — only 58 of the 4,407 training
studies carry any per-condition label at all (confirmed directly from the
downloaded train.csv) — into the exact
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
layout RSNAKneeDataset expects.

v2: the first attempt crashed on a 429 (Too Many Requests) at 29/58 after
~50 minutes, and lost all of it because discovery only downloaded after
ALL 58 were found. This version:
    - retries every API call with exponential backoff on 429/5xx instead
      of crashing outright.
    - downloads a study's files as soon as THAT study is fully resolved,
      not after all 58 are — a crash only costs the study in flight.
    - checkpoints (page_token, resolved-so-far) to disk after every page,
      and resumes from the checkpoint on restart instead of re-paging
      from the start of the listing.

Writes into the SAME train_series/ root _fetch_rsna_subset.py used (study
UIDs don't collide), plus rsna_labeled_subset_manifest.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "rsna-knee-abnormality-detection"
MAX_SLICES_PER_SERIES = 4
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "train_series"
MANIFEST_PATH = REPO_ROOT / "rsna_labeled_subset_manifest.json"
CHECKPOINT_PATH = REPO_ROOT / "rsna_labeled_fetch_checkpoint.json"

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_rate_limit_state = {"delay": 0.3}  # PERSISTS across with_retry() calls — see below


def with_retry(fn, *args, max_wait: float = 60.0, **kwargs):
    """Retries `fn(*args, **kwargs)` with backoff, indefinitely, on:
        - a 429/5xx HTTPError (rate limiting / server error), or
        - any connection-level failure (`requests.exceptions.ConnectionError`,
          `Timeout`, etc. — everything `requests.exceptions.RequestException`
          covers that ISN'T a clean HTTP response with a bad status code,
          e.g. a `ConnectTimeout` from a flaky link to api.kaggle.com — the
          second real crash this script hit, after the first cut only
          caught `HTTPError` and died on exactly this).
    This script is expected to run long and unattended, so "eventually
    succeed" beats "crash and lose an hour of discovery" every time.
    Anything else (auth failure, bad request) still raises immediately.

    The backoff delay is held in `_rate_limit_state`, a MODULE-LEVEL dict,
    not a local variable — v2's first cut reset `delay` to its starting
    value on every fresh `with_retry()` call, so a sustained rate limit
    (many calls in a row each getting one 429) never actually slowed the
    request rate down; it just retried at the same pace forever. Sharing
    state across calls means a run of 429s ratchets the delay up and KEEPS
    it up for subsequent calls too — this also proactively sleeps that
    same delay *before* every call (even ones that end up succeeding), so
    once the limiter has been tripped we back off the request rate itself,
    not just the retry wait. On a clean success the delay decays by 10%,
    so a temporary rate-limit episode doesn't slow the whole rest of the
    run down permanently.
    """
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
            # Connection-level failure (timeout, DNS, reset, ...) — not an
            # HTTP status at all, so it can't be an auth/bad-request error;
            # always worth retrying.
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


def download_study(api: KaggleApi, study_uid: str, series_dict: dict, study_series_planes: dict) -> int:
    """Downloads every collected slice for one fully-resolved study.
    Returns the number of files actually downloaded (existing files skipped)."""
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
    train = pd.read_csv(REPO_ROOT / "train.csv")
    labeled = train.loc[train[TARGET_COLUMNS].notna().any(axis=1), "StudyInstanceUID"].astype(str)
    target_studies = set(labeled)
    log(f"target: {len(target_studies)} labeled studies")

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

    while remaining:
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
            n = download_study(api, study_uid, collected[study_uid], study_series_planes)
            n_downloaded_total += n
            resolved_uids.add(study_uid)
            downloaded_uids.add(study_uid)
            remaining.discard(study_uid)
            elapsed_min = (time.time() - t_start) / 60
            log(f"{len(downloaded_uids)}/{len(target_studies)} labeled studies resolved+downloaded "
                f"({n} file(s) this study; pages={pages_seen}, entries={entries_seen}, elapsed_this_run={elapsed_min:.1f}min)")
            save_checkpoint(page_token, resolved_uids, downloaded_uids)

        if pages_seen % 50 == 0:
            elapsed_min = (time.time() - t_start) / 60
            log(f"progress: pages={pages_seen}, entries={entries_seen}, "
                f"resolved={len(downloaded_uids)}/{len(target_studies)}, elapsed_this_run={elapsed_min:.1f}min")
            save_checkpoint(page_token, resolved_uids, downloaded_uids)

        if page_token is None:
            log(f"exhausted file listing with {len(remaining)} studies still unresolved: {sorted(remaining)}")
            break

    log(f"listing done: pages={pages_seen}, entries={entries_seen}, "
        f"resolved={len(downloaded_uids)}/{len(target_studies)}")

    manifest: dict[str, dict[str, list[str]]] = {}
    for study_uid, series_dict in collected.items():
        if study_uid not in downloaded_uids:
            continue
        manifest[study_uid] = {}
        for series_uid, sop_files in series_dict.items():
            if not sop_files:
                continue
            plane = study_series_planes[study_uid][series_uid]
            manifest[study_uid].setdefault(plane, []).append(series_uid)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"DONE: {len(manifest)}/{len(target_studies)} labeled studies with >=1 usable series, "
        f"{n_downloaded_total} files downloaded this run, manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
