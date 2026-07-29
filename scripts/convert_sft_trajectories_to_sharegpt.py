#!/usr/bin/env python3
"""Convert API-collected trajectories (--export-full-transcript) to verl multi-turn SFT data.

Output format (per row): {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]}
Compatible with: verl.trainer.fsdp_sft_trainer (data.multiturn.enable=true, parquet with messages column)

Granularity:
  step    — one training row per turn (recommended; matches GRPO step-wise prompts).
  episode — one row per full multi-turn episode.

Usage:
  python3 scripts/convert_sft_trajectories_to_sharegpt.py \
    --inputs logs/sft_collect_apps_train.json logs/sft_collect_humaneval.json \
    --out data/sft/apps_mt8_mix_sharegpt.json \
    --granularity step
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _turn_messages(step: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    user = step.get("user_prompt")
    assistant = step.get("assistant_response")
    if not isinstance(user, str) or not user.strip():
        return None
    if not isinstance(assistant, str) or not assistant.strip():
        return None
    return [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]


def _episode_to_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    transcript = row.get("transcript") or []
    messages: List[Dict[str, str]] = []
    for step in transcript:
        pair = _turn_messages(step)
        if pair is None:
            return None
        messages.extend(pair)
    if len(messages) < 2:
        return None
    return messages


def _episode_to_step_samples(row: Dict[str, Any]) -> List[List[Dict[str, str]]]:
    out: List[List[Dict[str, str]]] = []
    for step in row.get("transcript") or []:
        pair = _turn_messages(step)
        if pair is not None:
            out.append(pair)
    return out


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("per_instance") or [])
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported JSON shape: {path}")


def _filter_episode(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    if row.get("outcome") not in args.outcomes:
        return False
    turns = int(row.get("turns") or 0)
    if turns < args.min_turns:
        return False
    if args.max_turns > 0 and turns > args.max_turns:
        return False
    transcript = row.get("transcript") or []
    if args.require_valid_actions and any(not t.get("valid_action") for t in transcript):
        return False
    if not transcript:
        return False
    if args.granularity == "episode":
        return _episode_to_messages(row) is not None
    return len(_episode_to_step_samples(row)) > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Trajectory JSON files")
    parser.add_argument("--out", required=True, help="Output .json or .jsonl path")
    parser.add_argument("--outcomes", default="won", help="Comma-separated outcomes to keep")
    parser.add_argument(
        "--granularity",
        choices=("step", "episode"),
        default="step",
        help="step=one row per turn (recommended); episode=full multi-turn conversation",
    )
    parser.add_argument(
        "--require-valid-actions",
        action="store_true",
        help="Drop episodes with any invalid_action turn",
    )
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=0, help="0 = no cap")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = keep all (episodes before step expand)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--val-out", default="", help="Optional validation JSON/JSONL")
    parser.add_argument("--parquet-out", default="", help="Optional train parquet for verl SFT")
    parser.add_argument("--val-parquet-out", default="", help="Optional val parquet for verl SFT")
    args = parser.parse_args()
    args.outcomes = {x.strip() for x in args.outcomes.split(",") if x.strip()}

    episodes: List[Dict[str, Any]] = []
    skipped = 0
    missing: List[str] = []

    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            missing.append(str(path))
            continue
        bench = path.stem.replace("sft_collect_", "")
        for row in _load_rows(path):
            if not _filter_episode(row, args):
                skipped += 1
                continue
            episodes.append({
                "row": row,
                "benchmark": bench,
            })

    if missing:
        print(f"[convert] WARN: missing inputs skipped: {missing}")

    if args.max_samples > 0 and len(episodes) > args.max_samples:
        rng = random.Random(args.seed)
        episodes = rng.sample(episodes, args.max_samples)

    val_eps: List[Dict[str, Any]] = []
    if args.val_ratio > 0 and episodes:
        rng = random.Random(args.seed)
        rng.shuffle(episodes)
        n_val = max(1, int(len(episodes) * args.val_ratio))
        if len(episodes) > n_val:
            val_eps = episodes[:n_val]
            episodes = episodes[n_val:]

    def _expand(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for item in items:
            row = item["row"]
            if args.granularity == "episode":
                messages = _episode_to_messages(row)
                if messages:
                    records.append({"messages": messages})
            else:
                for pair in _episode_to_step_samples(row):
                    records.append({"messages": pair})
        return records

    train_out = _expand(episodes)
    val_out = _expand(val_eps)
    if not val_out and train_out and (args.val_out or args.val_parquet_out):
        val_out = [train_out[0]]
        print("[convert] WARN: val split empty; using 1 train row for validation")
    out_path = Path(args.out)

    def _write_records(path: Path, records: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            with path.open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _write_parquet(path: Path, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_parquet(path, index=False)

    _write_records(out_path, train_out)

    meta_path = out_path.with_name(out_path.stem + ".meta.json")
    meta_path.write_text(json.dumps({
        "n_train": len(train_out),
        "n_val": len(val_out),
        "n_episodes_train": len(episodes),
        "n_episodes_val": len(val_eps),
        "n_skipped": skipped,
        "granularity": args.granularity,
        "inputs": [p for p in args.inputs if Path(p).exists()],
        "missing_inputs": missing,
        "outcomes": sorted(args.outcomes),
        "require_valid_actions": args.require_valid_actions,
    }, indent=2) + "\n", encoding="utf-8")

    if args.val_out and val_out:
        val_path = Path(args.val_out)
        _write_records(val_path, val_out)

    if args.parquet_out:
        pq_path = Path(args.parquet_out)
        _write_parquet(pq_path, train_out)
        print(f"[convert] wrote {pq_path}")

    if args.val_parquet_out and val_out:
        val_pq_path = Path(args.val_parquet_out)
        _write_parquet(val_pq_path, val_out)
        print(f"[convert] wrote {val_pq_path}")

    print(
        f"[convert] granularity={args.granularity} "
        f"episodes={len(episodes)} train_rows={len(train_out)} val_rows={len(val_out)} skipped={skipped}"
    )
    print(f"[convert] wrote {out_path}")
    print(f"[convert] meta {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
