#!/usr/bin/env python3
"""Post-process SFT trajectories to add missing <think> blocks.

Collection used DISABLE_THINKING=1, so most assistant turns lack <think>,
which makes projection mark valid_action=0 even when edit/bash/finish parse.

Strategy (per assistant_response):
  1. Already has <think> / Thought: → leave unchanged.
  2. Prose before first action tag → wrap that prose as <think>...</think>.
  3. Pure action (edit/bash/finish) → prepend a short synthetic <think>.
  4. Prose only (no action tag) → wrap whole text as <think> (still noop for
     env, but keeps reasoning for SFT instead of dropping it).

Writes patched collection JSONs (won transcripts only mutated) and prints
valid_action recovery stats.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THINK_RE = re.compile(r"<think>|^\s*Thought\s*:", re.I | re.M)
_ACTION_RE = re.compile(r"<(edit|execute_bash|finish)\b", re.I)
_BASH_INNER = re.compile(r"<execute_bash>(.*?)</execute_bash>", re.I | re.S)


def _synthetic_think(resp: str, action: str) -> str:
    action = action.lower()
    if action == "execute_bash":
        m = _BASH_INNER.search(resp)
        cmd = (m.group(1).strip() if m else "")[:120]
        if "cat " in cmd:
            return "Inspect the current files before editing."
        if "python" in cmd or "echo" in cmd:
            return "Run a self-test with chosen stdin to check the program."
        return "Run a shell command to inspect or test the solution."
    if action == "edit":
        return "Revise solution.py to fix the bug based on the problem and feedback."
    if action == "finish":
        return "The solution looks ready; submit for hidden grading."
    return "Reason about the next repair step, then act."


def inject_think(resp: str) -> Tuple[str, str]:
    """Return (new_response, mode) where mode is kept|wrap_prose|synth|wrap_all."""
    if not isinstance(resp, str) or not resp.strip():
        return resp, "kept"
    if _THINK_RE.search(resp):
        return resp, "kept"

    m = _ACTION_RE.search(resp)
    if m is None:
        body = resp.strip()
        return f"<think>\n{body}\n</think>", "wrap_all"

    before = resp[: m.start()].strip()
    after = resp[m.start() :].lstrip()
    action = m.group(1)
    if before:
        return f"<think>\n{before}\n</think>\n{after}", "wrap_prose"

    think = _synthetic_think(resp, action)
    return f"<think>\n{think}\n</think>\n{after}", "synth"


def _patch_row(row: Dict[str, Any], only_won: bool) -> Dict[str, int]:
    counts = {"kept": 0, "wrap_prose": 0, "synth": 0, "wrap_all": 0, "steps": 0}
    if only_won and not (row.get("won") or row.get("outcome") == "won"):
        return counts
    tx = row.get("transcript")
    if not isinstance(tx, list):
        return counts
    for step in tx:
        if not isinstance(step, dict):
            continue
        resp = step.get("assistant_response")
        if not isinstance(resp, str):
            continue
        new_resp, mode = inject_think(resp)
        step["assistant_response"] = new_resp
        step["think_injected"] = mode != "kept"
        step["think_inject_mode"] = mode
        counts[mode] += 1
        counts["steps"] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "logs/sft_collect_apps_train.json",
            "logs/sft_collect_apps_train_topup_introductory.json",
            "logs/sft_collect_apps_train_topup_competition.json",
        ],
    )
    ap.add_argument("--out-dir", default="logs/sft_with_think")
    ap.add_argument("--only-won", action="store_true", default=True)
    ap.add_argument("--no-only-won", action="store_false", dest="only_won")
    ap.add_argument(
        "--check-valid",
        action="store_true",
        help="Re-parse with projection and report valid_action rate on won steps",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    total = {"kept": 0, "wrap_prose": 0, "synth": 0, "wrap_all": 0, "steps": 0}
    out_paths: List[Path] = []

    for rel in args.inputs:
        src = Path(rel)
        if not src.is_absolute():
            src = repo / src
        if not src.exists():
            print(f"[skip] missing {src}")
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        rows = data.get("per_instance") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise SystemExit(f"bad shape: {src}")
        for row in rows:
            c = _patch_row(row, only_won=args.only_won)
            for k, v in c.items():
                total[k] = total.get(k, 0) + v
        dst = out_dir / src.name
        dst.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
        out_paths.append(dst)
        print(f"[wrote] {dst} ({dst.stat().st_size / 1e6:.1f}MB)")

    print("[inject]", {k: total[k] for k in ["steps", "kept", "wrap_prose", "synth", "wrap_all"]})

    if args.check_valid and out_paths:
        import importlib.util

        proj_path = repo / "agent_system/environments/env_package/swebench/projection.py"
        spec = importlib.util.spec_from_file_location("swe_proj", proj_path)
        proj = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(proj)

        n = v1 = 0
        for p in out_paths:
            data = json.loads(p.read_text(encoding="utf-8"))
            for row in data.get("per_instance") or []:
                if not (row.get("won") or row.get("outcome") == "won"):
                    continue
                for step in row.get("transcript") or []:
                    resp = step.get("assistant_response") or ""
                    _a, valid = proj.parse_swebench_action(resp)
                    n += 1
                    v1 += int(bool(valid))
        print(f"[check] won steps valid_action={v1}/{n} ({100 * v1 / max(1, n):.1f}%)")

    manifest = out_dir / "inject_think.meta.json"
    manifest.write_text(
        json.dumps({"inputs": args.inputs, "out_paths": [str(p) for p in out_paths], "counts": total}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"[meta] {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
