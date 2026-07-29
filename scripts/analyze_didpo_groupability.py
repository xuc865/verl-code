#!/usr/bin/env python3
"""DIDPO groupability diagnostics — cross-rollout snippet groups + within-traj diff-gate.

DIDPO only gets non-trivial snippet advantage when, within the same prompt/uid
group of G rollouts, *changed* snippets form clusters with |G|>1 (psi(1)=0).

This script answers:
  - Within one trajectory: how often are snippets byte-identical re-emissions
    (diff-gated out)?
  - Across G rollouts of the same instance: how many snippet groups form, and
    what is the size histogram / singleton rate / mean groupability score (GS)?

Input formats
-------------
1) Multi-rollout groups JSON (preferred, for GRPO/DIDPO dumps)::

    {
      "groups": [
        {
          "uid": "apps__idx0",
          "rollouts": [
            {"responses": ["<edit>...<code>def f():\\n  return 1</code></edit>", ...]},
            {"responses": [...]}
          ]
        }
      ]
    }

2) Eval / SFT-collect JSON with ``per_instance[].transcript[].assistant_response``
   — used for *within-trajectory* diff-gate stats only (one rollout per uid).

3) ``--demo`` — synthetic G=32 groups at several duplication rates (methodology
   preview until live GRPO dumps exist).

Usage::

  python3 scripts/analyze_didpo_groupability.py --demo \\
    --out logs/didpo_groupability_report.json

  python3 scripts/analyze_didpo_groupability.py \\
    --groups-json path/to/groups.json --out logs/didpo_groupability_report.json

  python3 scripts/analyze_didpo_groupability.py \\
    --eval-json logs/backup/sft_collect_....json --out logs/didpo_groupability_within.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from didpo.core_didpo import build_snippet_groups, groupability_score
from didpo.snippet import (
    LEVEL_FUNCTION,
    LEVEL_HUNK,
    apply_diff_gate,
    extract_snippets,
)


def _responses_from_transcript(transcript: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for turn in transcript or []:
        resp = (
            turn.get("assistant_response")
            or turn.get("response")
            or turn.get("response_preview")
            or ""
        )
        if isinstance(resp, str) and resp.strip():
            out.append(resp)
    return out


def _snippets_for_rollout(responses: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract snippets across steps with traj-local diff-gating."""
    memory: Dict[str, str] = {}
    all_snips: List[Dict[str, Any]] = []
    n_steps = 0
    n_with_code = 0
    n_changed = 0
    n_gated = 0
    for resp in responses:
        n_steps += 1
        snips = extract_snippets(resp)
        if not snips:
            continue
        n_with_code += 1
        apply_diff_gate(snips, memory)
        for s in snips:
            d = s.to_dict()
            all_snips.append(d)
            if d.get("changed", True):
                n_changed += 1
            else:
                n_gated += 1
    stats = {
        "n_steps": n_steps,
        "n_steps_with_code": n_with_code,
        "n_snippets": len(all_snips),
        "n_changed": n_changed,
        "n_diff_gated": n_gated,
        "diff_gate_rate": (n_gated / max(1, n_changed + n_gated)),
    }
    return all_snips, stats


def _group_size_hist(sizes: Sequence[int]) -> Dict[str, int]:
    """Bucket sizes for display: 1, 2, 3-4, 5-8, 9-16, 17+."""
    buckets = {"1": 0, "2": 0, "3-4": 0, "5-8": 0, "9-16": 0, "17+": 0}
    for s in sizes:
        if s <= 1:
            buckets["1"] += 1
        elif s == 2:
            buckets["2"] += 1
        elif s <= 4:
            buckets["3-4"] += 1
        elif s <= 8:
            buckets["5-8"] += 1
        elif s <= 16:
            buckets["9-16"] += 1
        else:
            buckets["17+"] += 1
    return buckets


def analyze_cross_rollout_group(
    rollouts: Sequence[Sequence[str]],
    *,
    uid: str = "uid0",
    sim_thresh: float = 0.8,
    phi_s0: float = 8.0,
    psi_count_ref: float = 8.0,
) -> Dict[str, Any]:
    """One SWE-bench instance with G rollouts → DIDPO grouping stats."""
    rows_snips: List[List[Dict[str, Any]]] = []
    within_stats: List[Dict[str, Any]] = []
    for responses in rollouts:
        snips, st = _snippets_for_rollout(responses)
        rows_snips.append(snips)
        within_stats.append(st)

    index = np.zeros(len(rows_snips), dtype=int)
    level_reports: Dict[str, Any] = {}
    for level, name in ((LEVEL_FUNCTION, "function"), (LEVEL_HUNK, "hunk")):
        gmap = build_snippet_groups(rows_snips, index, level, sim_thresh=sim_thresh)
        size_by_gid: Dict[str, int] = Counter(gmap.values())
        sizes = list(size_by_gid.values())
        # GS for each grouped snippet member
        gs_vals: List[float] = []
        for (row, sidx), gid in gmap.items():
            snip = rows_snips[row][sidx]
            gs_vals.append(
                groupability_score(
                    float(snip.get("size", 1.0)),
                    int(size_by_gid[gid]),
                    s0=phi_s0,
                    count_ref=psi_count_ref,
                )
            )
        n_changed = sum(1 for snips in rows_snips for s in snips if s.get("level") == level and s.get("changed", True))
        n_groups = len(sizes)
        n_singleton = sum(1 for s in sizes if s == 1)
        n_multi = n_groups - n_singleton
        members_in_multi = sum(s for s in sizes if s > 1)
        level_reports[name] = {
            "n_changed_snippets": n_changed,
            "n_groups": n_groups,
            "n_singleton_groups": n_singleton,
            "n_multi_groups": n_multi,
            "singleton_group_rate": n_singleton / max(1, n_groups),
            "share_members_in_multi_groups": members_in_multi / max(1, sum(sizes)),
            "group_size_mean": float(np.mean(sizes)) if sizes else 0.0,
            "group_size_max": int(max(sizes)) if sizes else 0,
            "group_size_hist": _group_size_hist(sizes),
            "gs_mean": float(np.mean(gs_vals)) if gs_vals else 0.0,
            "gs_frac_positive": float(np.mean([g > 0 for g in gs_vals])) if gs_vals else 0.0,
            # DIDPO usable if a non-trivial fraction of changed snippets land in |G|>1
            "didpo_usable_hint": (members_in_multi / max(1, sum(sizes))) >= 0.25 and n_multi >= 1,
        }

    return {
        "uid": uid,
        "n_rollouts": len(rollouts),
        "within_traj_diff_gate_rate_mean": float(
            np.mean([s["diff_gate_rate"] for s in within_stats])
        )
        if within_stats
        else 0.0,
        "levels": level_reports,
    }


def analyze_groups_file(
    groups: Sequence[Dict[str, Any]],
    *,
    sim_thresh: float,
    phi_s0: float,
    psi_count_ref: float,
) -> Dict[str, Any]:
    per_uid = []
    for g in groups:
        uid = str(g.get("uid", f"uid{len(per_uid)}"))
        rollouts_raw = g.get("rollouts") or []
        rollouts: List[List[str]] = []
        for ro in rollouts_raw:
            if isinstance(ro, dict):
                rollouts.append(list(ro.get("responses") or []))
            elif isinstance(ro, list):
                rollouts.append([str(x) for x in ro])
        if len(rollouts) < 2:
            continue
        per_uid.append(
            analyze_cross_rollout_group(
                rollouts,
                uid=uid,
                sim_thresh=sim_thresh,
                phi_s0=phi_s0,
                psi_count_ref=psi_count_ref,
            )
        )
    return _aggregate(per_uid, sim_thresh=sim_thresh)


def analyze_within_from_eval(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("per_instance") or []
    within = []
    n_code_traj = 0
    for row in rows:
        responses = _responses_from_transcript(row.get("transcript") or [])
        snips, st = _snippets_for_rollout(responses)
        if st["n_steps_with_code"] > 0:
            n_code_traj += 1
        within.append(st)
    if not within:
        return {"source": str(path), "n_instances": 0}
    return {
        "source": str(path),
        "mode": "within_trajectory_diff_gate_only",
        "n_instances": len(within),
        "n_with_code_steps": n_code_traj,
        "diff_gate_rate_mean": float(np.mean([w["diff_gate_rate"] for w in within])),
        "diff_gate_rate_p50": float(np.median([w["diff_gate_rate"] for w in within])),
        "snippets_per_traj_mean": float(np.mean([w["n_snippets"] for w in within])),
        "changed_per_traj_mean": float(np.mean([w["n_changed"] for w in within])),
        "note": (
            "Single rollout per instance — cannot estimate cross-rollout DIDPO groups. "
            "Need G>=2 rollouts per uid (GRPO group dump or multi-rollout probe)."
        ),
    }


def _aggregate(per_uid: List[Dict[str, Any]], *, sim_thresh: float) -> Dict[str, Any]:
    if not per_uid:
        return {"n_uids": 0, "levels": {}}
    out: Dict[str, Any] = {
        "n_uids": len(per_uid),
        "n_rollouts_mean": float(np.mean([u["n_rollouts"] for u in per_uid])),
        "sim_thresh": sim_thresh,
        "within_traj_diff_gate_rate_mean": float(
            np.mean([u["within_traj_diff_gate_rate_mean"] for u in per_uid])
        ),
        "levels": {},
        "per_uid_sample": per_uid[:8],
    }
    for level in ("function", "hunk"):
        usable = [u["levels"][level] for u in per_uid if level in u["levels"]]
        if not usable:
            continue
        # merge hist
        hist = Counter()
        for u in usable:
            hist.update(u["group_size_hist"])
        out["levels"][level] = {
            "n_groups_mean": float(np.mean([u["n_groups"] for u in usable])),
            "n_multi_groups_mean": float(np.mean([u["n_multi_groups"] for u in usable])),
            "singleton_group_rate_mean": float(np.mean([u["singleton_group_rate"] for u in usable])),
            "share_members_in_multi_groups_mean": float(
                np.mean([u["share_members_in_multi_groups"] for u in usable])
            ),
            "group_size_mean_mean": float(np.mean([u["group_size_mean"] for u in usable])),
            "group_size_max_max": int(max(u["group_size_max"] for u in usable)),
            "gs_mean_mean": float(np.mean([u["gs_mean"] for u in usable])),
            "gs_frac_positive_mean": float(np.mean([u["gs_frac_positive"] for u in usable])),
            "didpo_usable_uid_rate": float(np.mean([1.0 if u["didpo_usable_hint"] else 0.0 for u in usable])),
            "group_size_hist_total": dict(hist),
        }
    return out


def _edit_resp(body: str, fname: str = "add") -> str:
    return (
        f"<think>edit</think>\n"
        f'<edit path="solution.py"><code>\n'
        f"def {fname}(a, b):\n"
        f"    {body}\n"
        f"</code></edit>"
    )


def build_demo_groups(group_size: int = 32) -> List[Dict[str, Any]]:
    """Synthetic regimes: high / mid / low cross-rollout duplication.

    Uses *different function names* across clusters so structural signatures
    separate buckets; within a bucket, bodies are byte-identical (exact match).
    """
    regimes = []

    # high: 24 identical ``solve`` + 8 unique ``u0..u7``
    rollouts = [{"responses": [_edit_resp("return a + b", "solve")]} for _ in range(24)]
    for i in range(group_size - 24):
        rollouts.append({"responses": [_edit_resp(f"return a + b + {i}", f"u{i}")]})
    regimes.append({"uid": "demo_high_dup", "rollouts": rollouts})

    # mid: 4 clusters of 8 (distinct signatures, identical bodies within cluster)
    rollouts = []
    for k in range(4):
        for _ in range(group_size // 4):
            rollouts.append({"responses": [_edit_resp(f"return a * {k} + b", f"cluster{k}")]})
    regimes.append({"uid": "demo_mid_clusters", "rollouts": rollouts[:group_size]})

    # low: all unique signatures
    rollouts = [
        {"responses": [_edit_resp(f"return a - {i}", f"solo{i}")]} for i in range(group_size)
    ]
    regimes.append({"uid": "demo_low_dup", "rollouts": rollouts})

    # within-traj re-emission (diff-gate): step1 write, step2 identical, step3 edit
    regimes.append(
        {
            "uid": "demo_diff_gate",
            "rollouts": [
                {
                    "responses": [
                        _edit_resp("return a + b", "solve"),
                        _edit_resp("return a + b", "solve"),  # gated
                        _edit_resp("return a * b", "solve"),
                    ]
                }
                for _ in range(group_size)
            ],
        }
    )
    return regimes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups-json", default="", help="Multi-rollout groups JSON")
    ap.add_argument("--eval-json", default="", help="Eval/SFT JSON for within-traj diff-gate only")
    ap.add_argument("--demo", action="store_true", help="Run synthetic G=32 regimes")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--sim-thresh", type=float, default=0.8)
    ap.add_argument("--phi-s0", type=float, default=8.0)
    ap.add_argument("--psi-count-ref", type=float, default=8.0)
    ap.add_argument("--out", default=str(REPO / "logs" / "didpo_groupability_report.json"))
    args = ap.parse_args()

    report: Dict[str, Any] = {
        "metric": "didpo_groupability",
        "definition": {
            "cross_rollout_group": (
                "Within one uid (same prompt), bucket changed snippets by structural "
                "signature, refine by source similarity >= sim_thresh. |G|=1 => GS=0 "
                "(no snippet advantage; falls back to episode/GRPO)."
            ),
            "diff_gate": (
                "Within one trajectory, byte-identical re-emission of the same signature "
                "is changed=False and excluded from grouping."
            ),
            "didpo_usable_hint": (
                "UID marked usable if >=25% of grouped snippet members sit in multi-member "
                "groups and there is at least one multi group."
            ),
        },
        "config": {
            "sim_thresh": args.sim_thresh,
            "phi_s0": args.phi_s0,
            "psi_count_ref": args.psi_count_ref,
            "group_size": args.group_size,
        },
        "sections": {},
    }

    if args.demo:
        groups = build_demo_groups(args.group_size)
        report["sections"]["demo_regimes"] = analyze_groups_file(
            groups,
            sim_thresh=args.sim_thresh,
            phi_s0=args.phi_s0,
            psi_count_ref=args.psi_count_ref,
        )
        report["sections"]["demo_regimes"]["label"] = (
            f"Synthetic preview at G={args.group_size} (not live GRPO rollouts)"
        )

    if args.groups_json:
        raw = json.loads(Path(args.groups_json).read_text(encoding="utf-8"))
        groups = raw.get("groups") if isinstance(raw, dict) else raw
        report["sections"]["live_groups"] = analyze_groups_file(
            groups,
            sim_thresh=args.sim_thresh,
            phi_s0=args.phi_s0,
            psi_count_ref=args.psi_count_ref,
        )
        report["sections"]["live_groups"]["source"] = args.groups_json

    if args.eval_json:
        report["sections"]["within_traj"] = analyze_within_from_eval(Path(args.eval_json))

    if not report["sections"]:
        print("Nothing to analyze. Pass --demo and/or --groups-json / --eval-json.", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report["sections"].get(k, {}).get("n_uids") or report["sections"].get(k, {}).get("n_instances") for k in report["sections"]}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
