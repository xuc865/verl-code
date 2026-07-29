#!/usr/bin/env python3
"""Rebuild a clean SwanLab .swanlab file from logs/ JSONL columns.

The live run-*.swanlab can become unreadable mid-file after concurrent
`swanlab sync` races. Training scalars keep landing in logs/<col>/1000.log,
but sync only reads the .swanlab datastore — so cloud curves freeze.

Usage:
  python3 scripts/swanlab_rebuild_from_logs.py \
    --src swanlog/run-20260721_012920-zbavplj0 \
    --out /tmp/swanlab_rebuild/zbavplj0fix \
    --run-id zbavplj0fix \
    --project grpo_coderl \
    --name grpo_coderl_qwen35_4b_sft_mt8_tp1_g16
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from google.protobuf.timestamp_pb2 import Timestamp

from swanlab.proto.swanlab.metric.column.v1.column_pb2 import (
    ColumnClass,
    ColumnRecord,
    ColumnType,
    SectionType,
)
from swanlab.proto.swanlab.metric.data.v1.data_pb2 import ScalarRecord, ScalarValue
from swanlab.proto.swanlab.record.v1.record_pb2 import Record
from swanlab.proto.swanlab.run.v1 import run_pb2
from swanlab.proto.swanlab.run.v1.run_pb2 import FinishRecord, StartRecord
from swanlab.sdk.internal.core_python.pkg import builder
from swanlab.sdk.internal.core_python.pkg.counter import Counter
from swanlab.sdk.internal.core_python.store import DataStoreReader, DataStoreWriter

# Column folder id -> metric key for verl GRPO console metrics.
# Order matches the first-seen custom columns in the original run.
DEFAULT_MAP = {
    1: "global_seqlen/min",
    2: "global_seqlen/max",
    3: "global_seqlen/minmax_diff",
    4: "global_seqlen/balanced_min",
    5: "global_seqlen/balanced_max",
    6: "global_seqlen/mean",
    7: "actor/entropy_loss",
    8: "training/rollout_probs_diff_max",
    9: "training/rollout_probs_diff_mean",
    10: "training/rollout_probs_diff_std",
    11: "episode/valid_action_ratio",
    12: "actor/kl_loss",
    13: "actor/kl_coef",
    14: "actor/pg_loss",
    15: "actor/pg_clipfrac",
    16: "actor/ppo_kl",
    17: "actor/pg_clipfrac_lower",
    18: "actor/grad_norm",
    19: "perf/mfu/actor",
    20: "perf/max_memory_allocated_gb",
    21: "perf/max_memory_reserved_gb",
    22: "perf/cpu_memory_used_gb",
    23: "actor/lr",
    24: "training/global_step",
    25: "training/epoch",
    26: "critic/score/mean",
    27: "critic/score/max",
    28: "critic/score/min",
    29: "critic/rewards/mean",
    30: "critic/rewards/max",
    31: "critic/rewards/min",
    32: "critic/advantages/mean",
    33: "critic/advantages/max",
    34: "critic/advantages/min",
    35: "critic/returns/mean",
    36: "critic/returns/max",
    37: "critic/returns/min",
    38: "response_length/mean",
    39: "response_length/max",
    40: "response_length/min",
    41: "response_length/clip_ratio",
    42: "prompt_length/mean",
    43: "prompt_length/max",
    44: "prompt_length/min",
    45: "prompt_length/clip_ratio",
    46: "episode/reward/mean",
    47: "episode/reward/max",
    48: "episode/reward/min",
    49: "episode/length/mean",
    50: "episode/length/max",
    51: "episode/length/min",
    52: "episode/tool_call_count/mean",
    53: "episode/success_rate",
    54: "timing_s/gen",
    55: "timing_s/reward",
    56: "timing_s/old_log_prob",
    57: "timing_s/ref",
    58: "timing_s/adv",
    59: "timing_s/update_actor",
    60: "timing_s/step",
    61: "timing_per_token_ms/gen",
    62: "timing_per_token_ms/adv",
    63: "timing_per_token_ms/update_actor",
    64: "timing_per_token_ms/ref",
    65: "perf/total_num_tokens",
    66: "perf/time_per_step",
    67: "perf/throughput",
}


def discover_col_map(src: Path) -> dict[int, str] | None:
    """Map logs/<id>/ folders to metric keys using custom columns in run-*.swanlab."""
    swan_files = sorted(src.glob("run-*.swanlab"))
    if not swan_files:
        return None
    try:
        from swanlab.sdk.internal.core_python.store import DataStoreReader
        from swanlab.proto.swanlab.record.v1.record_pb2 import Record
        from swanlab.proto.swanlab.metric.column.v1.column_pb2 import ColumnClass
    except Exception as e:
        print(f"[rebuild] WARN: cannot import swanlab for col discovery: {e}")
        return None

    reader = DataStoreReader()
    reader.open(swan_files[0])
    custom: list[str] = []
    try:
        for raw in reader:
            rec = Record()
            try:
                rec.ParseFromString(raw)
            except Exception:
                continue
            if rec.HasField("column") and rec.column.column_class == ColumnClass.COLUMN_CLASS_CUSTOM:
                custom.append(rec.column.column_key)
    finally:
        reader.close()
    if not custom:
        return None
    return {i: k for i, k in enumerate(custom, start=1)}


def load_points(src: Path, col_map: dict[int, str]) -> dict[int, dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for cid, key in col_map.items():
        path = src / "logs" / str(cid) / "1000.log"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            obj = json.loads(line)
            step = int(obj["index"])
            by_step.setdefault(step, {})[key] = float(obj["data"])
    return by_step


def write_run(
    *,
    out: Path,
    run_id: str,
    project: str,
    name: str,
    by_step: dict[int, dict[str, float]],
) -> Path:
    if out.exists():
        shutil.rmtree(out)
    for sub in ("files", "logs", "console", "media", "debug"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    swan_path = out / f"run-{run_id}.swanlab"
    keys = sorted({k for step_map in by_step.values() for k in step_map})
    steps = sorted(by_step)
    if not steps:
        raise SystemExit(f"no scalar points found under {out}")

    writer = DataStoreWriter()
    writer.open(swan_path)
    counter = Counter()
    ts = Timestamp()
    ts.GetCurrentTime()

    start = StartRecord(project=project, name=name, color="#78e89a", id=run_id, started_at=ts)
    writer.write(builder.build_start_record(start).SerializeToString())

    for key in keys:
        parts = key.split("/")
        section = parts[0] if len(parts) >= 2 else ""
        col = ColumnRecord(
            column_class=ColumnClass.COLUMN_CLASS_CUSTOM,
            column_type=ColumnType.COLUMN_TYPE_SCALAR,
            column_key=key,
            section_name=section,
            section_type=SectionType.SECTION_TYPE_PUBLIC,
        )
        writer.write(builder.build_column_record(counter, col).SerializeToString())

    for step in steps:
        for key, val in by_step[step].items():
            sr = ScalarRecord(
                key=key,
                step=step,
                type=ColumnType.COLUMN_TYPE_SCALAR,
                timestamp=ts,
                value=ScalarValue(number=val),
            )
            writer.write(builder.build_scalar_record(counter, sr).SerializeToString())

    writer.write(
        builder.build_finish_record(FinishRecord(state=run_pb2.RUN_STATE_FINISHED)).SerializeToString()
    )
    writer.close()

    # Quick self-check
    reader = DataStoreReader()
    reader.open(swan_path)
    max_step = 0
    n = 0
    for raw in reader:
        n += 1
        rec = Record()
        rec.ParseFromString(raw)
        if rec.HasField("scalar"):
            max_step = max(max_step, rec.scalar.step)
    reader.close()
    print(f"[rebuild] wrote {swan_path} records={n} steps={steps[0]}..{max_step} keys={len(keys)}")
    return swan_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project", default="grpo_coderl")
    ap.add_argument("--name", default="grpo_coderl_qwen35_4b_sft_mt8_tp1_g16")
    args = ap.parse_args()

    src = args.src.resolve()
    col_map = discover_col_map(src) or DEFAULT_MAP
    if col_map is not DEFAULT_MAP:
        print(f"[rebuild] discovered {len(col_map)} custom columns from .swanlab")
    else:
        print(f"[rebuild] using DEFAULT_MAP ({len(DEFAULT_MAP)} columns)")
    by_step = load_points(src, col_map)
    write_run(
        out=args.out.resolve(),
        run_id=args.run_id,
        project=args.project,
        name=args.name,
        by_step=by_step,
    )


if __name__ == "__main__":
    main()
