#!/usr/bin/env python3
"""Build CodeRL+/PRIME-aligned APPS training pool for multi-turn RL (+ SFT collect).

Primary source (preferred): local CodeRL+ HF dump
  $CODERLPLUS_DATA/train_set.parquet (+ optional validation_set.parquet)
  Download:
    hf download xueniki/data_CodeRLPLUS --repo-type dataset \\
      --local-dir /mnt/z4/solariewang/CODERLPLUS/data

Fallback: PRIME-RL/Eurus-2-RL-Data via ``datasets.load_dataset`` (apps-source only).

Safety:
  - Keep ability=code rows with parseable stdin/stdout ``ground_truth``.
  - Drop rows whose normalized question text matches APPS *test* (eval holdout).
  - Dedup by normalized question text.

Writes (selfrepair_io / apps adapter compatible jsonl):
  $SWEBENCH_DATA_ROOT/codeparrot_apps_prime_extra/train.jsonl
      → extras only (not in APPS train/test); used by preset ``apps_train_coderl``
  $SWEBENCH_DATA_ROOT/codeparrot_apps_coderl_train/train.jsonl
      → APPS train ∪ extras (convenience dump for inspection / offline tools)
  manifests next to each.

Usage (train host paths OK; IDE uses /apdcephfs/...):
  source scripts/hf_mirror_env.sh  # optional
  python3 scripts/build_apps_train_coderl.py
  python3 scripts/build_apps_train_coderl.py --dry-run
  python3 scripts/build_apps_train_coderl.py --include-validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# APPS / CodeRL+ IO sometimes embeds huge integer literals; allow json.loads.
try:
    sys.set_int_max_str_digits(0)
except Exception:  # noqa: BLE001
    pass

_ROOT = Path(os.environ.get("ROOT", "/mnt/z4/solariewang"))
if not _ROOT.exists():
    _ROOT = Path("/apdcephfs/z4/solariewang")

DATA_ROOT = Path(
    os.environ.get("SWEBENCH_DATA_ROOT")
    or os.environ.get("DATA_ROOT")
    or (_ROOT / "datasets")
)
APPS_DIR = DATA_ROOT / "codeparrot_apps"
EXTRA_DIR = DATA_ROOT / "codeparrot_apps_prime_extra"
MERGED_DIR = DATA_ROOT / "codeparrot_apps_coderl_train"
EXTRA_JSONL = EXTRA_DIR / "train.jsonl"
MERGED_JSONL = MERGED_DIR / "train.jsonl"
EXTRA_MANIFEST = EXTRA_DIR / "manifest.json"
MERGED_MANIFEST = MERGED_DIR / "manifest.json"

CODERLPLUS_DATA = Path(
    os.environ.get("CODERLPLUS_DATA")
    or (_ROOT / "CODERLPLUS" / "data")
)

# Historical PRIME filter (fallback path only).
_PRIME_APPS_SOURCES = frozenset({"apps", "codeparrot/apps", "codeparrot_apps"})


def _norm_question(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _qhash(text: Any) -> str:
    return hashlib.sha256(_norm_question(text).encode("utf-8")).hexdigest()


def _io_fingerprint(io: Any) -> Optional[str]:
    """Stable hash of the first few IO cases (survives question rewording)."""
    raw = None
    if isinstance(io, str):
        raw = io
        try:
            io = json.loads(io)
        except (json.JSONDecodeError, ValueError):
            # Fallback: hash raw string prefix (handles pathological huge ints).
            return hashlib.sha256(io[:8192].encode("utf-8", errors="ignore")).hexdigest()
    if not isinstance(io, dict):
        return None
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        if raw:
            return hashlib.sha256(raw[:8192].encode("utf-8", errors="ignore")).hexdigest()
        return None
    n = min(3, len(inputs), len(outputs))
    if n <= 0:
        return None
    try:
        blob = json.dumps(
            {"i": [str(inputs[j]) for j in range(n)], "o": [str(outputs[j]) for j in range(n)]},
            ensure_ascii=False,
            sort_keys=True,
        )
    except (TypeError, ValueError):
        blob = repr((inputs[:n], outputs[:n]))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_user_prompt(prompt_field: Any) -> str:
    if isinstance(prompt_field, list):
        for msg in reversed(prompt_field):
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                return str(msg.get("content") or "")
        # fall back: last contentful message
        for msg in reversed(prompt_field):
            if isinstance(msg, dict) and msg.get("content"):
                return str(msg.get("content"))
    if isinstance(prompt_field, str):
        return prompt_field
    return ""


def _parse_io_ground_truth(gt: Any) -> Optional[Dict[str, Any]]:
    if isinstance(gt, dict):
        obj = gt
    elif isinstance(gt, str) and gt.strip():
        try:
            obj = json.loads(gt)
        except json.JSONDecodeError:
            return None
    else:
        return None
    if not isinstance(obj, dict):
        return None
    inputs = obj.get("inputs")
    outputs = obj.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    if not inputs or not outputs or len(inputs) != len(outputs):
        return None
    out: Dict[str, Any] = {
        "inputs": [str(x) for x in inputs],
        "outputs": [str(x) for x in outputs],
    }
    if obj.get("fn_name"):
        out["fn_name"] = obj["fn_name"]
    return out


def _row_to_apps(
    *,
    question: str,
    io: Dict[str, Any],
    source: str,
    seq: int,
    difficulty: str = "unknown",
    starter_code: str = "",
    url: str = "",
) -> Dict[str, Any]:
    pid = f"coderlplus_{source}_{seq}"
    return {
        "id": pid,
        "problem_id": pid,
        "question": question,
        "input_output": json.dumps(io, ensure_ascii=False),
        "solutions": "[]",
        "starter_code": starter_code or "",
        "difficulty": difficulty or "unknown",
        "url": url or "",
        "_source": source,
    }


def _coderlplus_row_to_apps(row: Dict[str, Any], seq: int) -> Optional[Dict[str, Any]]:
    ability = str(row.get("ability") or "").strip().lower()
    if ability and ability != "code":
        return None
    src = str(row.get("data_source") or "coderlplus").strip().lower() or "coderlplus"
    rm = row.get("reward_model") or {}
    if not isinstance(rm, dict):
        return None
    io = _parse_io_ground_truth(rm.get("ground_truth"))
    if io is None:
        return None
    question = _extract_user_prompt(row.get("prompt")).strip()
    if len(question) < 32:
        return None
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    # NOTE: CodeRL+ extra_info.index is NOT unique (often 0 for every row).
    # Always use the caller-provided monotonic ``seq`` for problem_id.
    return _row_to_apps(
        question=question,
        io=io,
        source=src,
        seq=seq,
        difficulty=str(extra.get("difficulty") or row.get("difficulty") or "unknown"),
    )


def _load_coderlplus_parquet(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install pyarrow") from e

    converted: List[Dict[str, Any]] = []
    seq = 0
    for path in paths:
        if not path.is_file():
            print(f"[build] skip missing {path}")
            continue
        print(f"[build] reading {path} ...")
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            cols = {name: table[name].to_pylist() for name in table.column_names}
            n = len(next(iter(cols.values())))
            for i in range(n):
                row = {k: cols[k][i] for k in cols}
                apps_row = _coderlplus_row_to_apps(row, seq)
                seq += 1
                if apps_row is not None:
                    converted.append(apps_row)
        print(f"[build]   cumulative converted={len(converted)}")
    return converted


def _prime_row_to_apps(row: Dict[str, Any], seq: int) -> Optional[Dict[str, Any]]:
    src = str(row.get("data_source") or "").strip().lower()
    if src not in _PRIME_APPS_SOURCES:
        return None
    ability = str(row.get("ability") or "").strip().lower()
    if ability and ability != "code":
        return None
    rm = row.get("reward_model") or {}
    if not isinstance(rm, dict):
        return None
    io = _parse_io_ground_truth(rm.get("ground_truth"))
    if io is None:
        return None
    question = _extract_user_prompt(row.get("prompt")).strip()
    if len(question) < 32:
        return None
    extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
    return _row_to_apps(
        question=question,
        io=io,
        source="prime_apps",
        seq=seq,
        difficulty=str(extra.get("difficulty") or "unknown"),
    )


def _load_prime_apps() -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install datasets") from e
    ds = load_dataset("PRIME-RL/Eurus-2-RL-Data", split="train")
    converted: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        apps_row = _prime_row_to_apps(dict(row), i)
        if apps_row is not None:
            converted.append(apps_row)
    return converted


def _apps_row_copy(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["_source"] = "apps_train"
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(
    dry_run: bool = False,
    include_validation: bool = True,
    use_prime_fallback: bool = True,
) -> Dict[str, Any]:
    train_path = APPS_DIR / "train.jsonl"
    test_path = APPS_DIR / "test.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Missing {train_path}. Stage APPS first:\n"
            f"  hf download codeparrot/apps --repo-type dataset --local-dir {APPS_DIR}"
        )
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing {test_path} (needed for eval holdout filter).")

    apps_train = _load_jsonl(train_path)
    apps_test = _load_jsonl(test_path)
    test_hashes: Set[str] = set()
    train_hashes: Set[str] = set()
    test_io: Set[str] = set()
    train_io: Set[str] = set()
    for row in apps_test:
        test_hashes.add(_qhash(row.get("question")))
        fp = _io_fingerprint(row.get("input_output"))
        if fp:
            test_io.add(fp)
    for row in apps_train:
        train_hashes.add(_qhash(row.get("question")))
        fp = _io_fingerprint(row.get("input_output"))
        if fp:
            train_io.add(fp)
    print(
        f"[build] APPS train={len(apps_train)} (io_fp={len(train_io)}) "
        f"test(blocklist)={len(apps_test)} (io_fp={len(test_io)})"
    )

    parquet_paths = [CODERLPLUS_DATA / "train_set.parquet"]
    if include_validation:
        parquet_paths.append(CODERLPLUS_DATA / "validation_set.parquet")

    stats: Dict[str, Any] = {
        "apps_train": len(apps_train),
        "apps_test_blocklist": len(apps_test),
        "apps_train_io_fps": len(train_io),
        "apps_test_io_fps": len(test_io),
        "coderlplus_converted": 0,
        "prime_converted": 0,
        "extras_written": 0,
        "merged_written": 0,
        "skipped_test_overlap": 0,
        "skipped_train_dup": 0,
        "skipped_dup": 0,
        "by_source_in": {},
        "by_source_extras": {},
        "coderlplus_data": str(CODERLPLUS_DATA),
        "include_validation": include_validation,
    }

    candidate_rows: List[Dict[str, Any]] = []
    have_coderl = (CODERLPLUS_DATA / "train_set.parquet").is_file()
    if have_coderl:
        candidate_rows = _load_coderlplus_parquet(parquet_paths)
        stats["coderlplus_converted"] = len(candidate_rows)
        from collections import Counter

        stats["by_source_in"] = dict(Counter(r.get("_source") for r in candidate_rows))
        print(f"[build] CodeRL+ IO rows={len(candidate_rows)} by_source={stats['by_source_in']}")
    elif use_prime_fallback:
        print("[build] CodeRL+ train_set.parquet missing; trying PRIME apps-only fallback...")
        try:
            candidate_rows = _load_prime_apps()
            stats["prime_converted"] = len(candidate_rows)
            print(f"[build] PRIME apps-source rows={len(candidate_rows)}")
        except Exception as e:  # noqa: BLE001
            print(f"[build] WARN: could not load PRIME ({e}); extras empty.")
            stats["prime_load_error"] = str(e)
    else:
        print("[build] WARN: no CodeRL+ data and --no-prime-fallback; extras empty.")

    seen_q: Set[str] = set(train_hashes)  # extras must not duplicate APPS train either
    seen_io: Set[str] = set(train_io)
    extras: List[Dict[str, Any]] = []

    def _try_add_extra(row: Dict[str, Any]) -> None:
        h = _qhash(row.get("question"))
        fp = _io_fingerprint(row.get("input_output"))
        # Prefer IO fingerprint: CodeRL+ rewrites APPS prompts so qhash often misses.
        if (fp and fp in test_io) or h in test_hashes:
            stats["skipped_test_overlap"] += 1
            return
        if (fp and fp in train_io) or h in train_hashes:
            stats["skipped_train_dup"] += 1
            return
        if h in seen_q or (fp and fp in seen_io):
            stats["skipped_dup"] += 1
            return
        seen_q.add(h)
        if fp:
            seen_io.add(fp)
        extras.append(row)

    for row in candidate_rows:
        _try_add_extra(row)

    from collections import Counter

    stats["by_source_extras"] = dict(Counter(r.get("_source") for r in extras))
    stats["extras_written"] = len(extras)

    merged = [_apps_row_copy(r) for r in apps_train] + extras
    stats["merged_written"] = len(merged)

    print(
        f"[build] extras={len(extras)} (skip_test={stats['skipped_test_overlap']}, "
        f"skip_apps_train={stats['skipped_train_dup']}, skip_dup={stats['skipped_dup']})"
    )
    print(f"[build] extras by_source={stats['by_source_extras']}")
    print(f"[build] merged total={len(merged)} (= APPS train {len(apps_train)} + extras)")

    if dry_run:
        print("[build] dry-run: not writing files")
        return stats

    _write_jsonl(EXTRA_JSONL, extras)
    EXTRA_MANIFEST.write_text(
        json.dumps(
            {
                "preset": "apps_train_coderl = apps_train + apps_prime_extra",
                "output": str(EXTRA_JSONL),
                "stats": stats,
                "note": (
                    "Extras only: CodeRL+/PRIME IO coding rows not in APPS train/test. "
                    "Compatible with apps adapter / selfrepair_io multi-turn RL & SFT collect."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(MERGED_JSONL, merged)
    MERGED_MANIFEST.write_text(
        json.dumps(
            {
                "output": str(MERGED_JSONL),
                "stats": stats,
                "note": "Convenience merge: APPS train ∪ extras (deduped, test-holdout filtered).",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[build] wrote {EXTRA_JSONL}")
    print(f"[build] wrote {MERGED_JSONL}")
    print(f"[build] manifest {EXTRA_MANIFEST}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build apps_train_coderl jsonl from CodeRL+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also ingest CodeRL+ validation_set.parquet (default: true)",
    )
    parser.add_argument(
        "--no-prime-fallback",
        action="store_true",
        help="Do not fall back to HF PRIME if CodeRL+ parquet missing",
    )
    args = parser.parse_args()
    build(
        dry_run=args.dry_run,
        include_validation=args.include_validation,
        use_prime_fallback=not args.no_prime_fallback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
