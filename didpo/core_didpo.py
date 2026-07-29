# Copyright 2026 The DIDPO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DIDPO core advantage estimation (trainer side) — paper-aligned + engineered.

Pipeline:

  1. Episode-level GRPO-style advantage A^E over task groups.
  2. Dynamic sub-diff anchors (engineered):
       - exact-hash full roots / blocks
       - pairwise difflib opcodes (budgeted) → equal / high-sim replace blocks
       - rep-centric fuzzy merge (no transitive union-find paste)
       - greedy facility-location selection of C* by GS
       - per-root non-overlapping interval cut
  3. Uncovered regions fall back to AST structural proposals, then grouped
     by signature + similarity.
  4. Diff-level group-relative advantage A^D from step return-to-go.
  5. Token fill: Â_{i,t} = A^E_i + λ · A^D(s_{i,t}).

Also returns a diagnostics dict for SwanLab / fixed-prompt tracking
(including ``align_time_ms``).
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from didpo.snippet import (
    LEVEL_FUNCTION,
    LEVEL_HUNK,
    NUM_LEVELS,
    ORIGIN_FALLBACK,
    ORIGIN_ROOT,
    source_similarity,
)


def __getattr__(name):  # noqa: D401 - module-level lazy attribute (PEP 562)
    if name == "compute_step_discounted_returns":
        from gigpo.core_gigpo import compute_step_discounted_returns as _f
        return _f
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# Groupability score (paper Eq. 9; default saturating × saturating)           #
# --------------------------------------------------------------------------- #
def phi_size(size: float, s0: float = 8.0, form: str = "saturating") -> float:
    size = max(0.0, float(size))
    if form == "saturating":
        return 1.0 - math.exp(-size / max(1e-6, s0))
    if form == "log":
        return min(1.0, math.log1p(size) / math.log1p(s0))
    if form == "linear":
        return min(1.0, size / max(1e-6, s0))
    raise ValueError(f"Unknown phi form: {form}")


def psi_count(count: int, count_ref: float = 8.0, form: str = "saturating") -> float:
    """ψ(n): 0 for singleton; paper uses 1 - exp(-(n-1)/g0)."""
    count = int(count)
    if count <= 1:
        return 0.0
    if form == "saturating":
        return 1.0 - math.exp(-(count - 1) / max(1e-6, count_ref))
    if form == "log":
        return min(1.0, math.log(count) / math.log(max(2.0, count_ref)))
    if form == "linear":
        return min(1.0, (count - 1) / max(1e-6, (count_ref - 1)))
    raise ValueError(f"Unknown psi form: {form}")


def groupability_score(size: float, count: int, *,
                       s0: float = 8.0, count_ref: float = 8.0,
                       phi_form: str = "saturating", psi_form: str = "saturating") -> float:
    return phi_size(size, s0, phi_form) * psi_count(count, count_ref, psi_form)


# --------------------------------------------------------------------------- #
# Engineered alignment: exact-hash → pairwise LCS opcodes → greedy C*         #
# --------------------------------------------------------------------------- #
class _Occ:
    __slots__ = ("row", "root_idx", "span", "text", "size", "edit_type",
                 "token_start", "token_end", "char_start", "char_end", "uid")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Anchor:
    __slots__ = ("occ", "gs", "mean_size", "n", "preview", "uid")

    def __init__(self, occ: List[_Occ], gs: float, mean_size: float, n: int, preview: str, uid: Any):
        self.occ = occ
        self.gs = gs
        self.mean_size = mean_size
        self.n = n
        self.preview = preview
        self.uid = uid


class _RootView:
    __slots__ = ("row", "root_idx", "lines", "source", "edit_type",
                 "char_start", "char_end", "token_start", "token_end", "line_starts")

    def __init__(self, row, root_idx, snip: Dict[str, Any]):
        self.row = row
        self.root_idx = root_idx
        self.lines = list(snip.get("lines") or [
            ln for ln in str(snip.get("source", "")).splitlines() if ln.strip()
        ])
        self.source = str(snip.get("source", ""))
        self.edit_type = str(snip.get("edit_type", "none"))
        self.char_start = int(snip.get("char_start", 0))
        self.char_end = int(snip.get("char_end", 0))
        self.token_start = int(snip.get("token_start", 0))
        self.token_end = int(snip.get("token_end", 0))
        # Cumulative char offsets of normalized lines inside ``source``.
        self.line_starts: List[int] = []
        cursor = 0
        for i, ln in enumerate(self.lines):
            idx = self.source.find(ln, cursor)
            if idx < 0:
                idx = cursor
            self.line_starts.append(idx)
            cursor = idx + len(ln)


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _occ_from_root_span(root: _RootView, span: Tuple[int, int], uid: Any) -> Optional[_Occ]:
    p, q = span
    if q <= p or p < 0 or q > len(root.lines):
        return None
    text = "\n".join(root.lines[p:q])
    if not text.strip():
        return None
    # Char span from line starts; end = start of next line or end of last line.
    local_cs = root.line_starts[p] if root.line_starts else 0
    if q < len(root.lines) and root.line_starts:
        local_ce = root.line_starts[q]
    else:
        local_ce = local_cs + len(text)
    cs = root.char_start + local_cs
    ce = root.char_start + local_ce
    root_chars = max(1, root.char_end - root.char_start)
    root_toks = max(1, root.token_end - root.token_start)
    rel_s = local_cs / root_chars
    rel_e = local_ce / root_chars
    ts = root.token_start + int(rel_s * root_toks)
    te = root.token_start + max(int(rel_e * root_toks), int(rel_s * root_toks) + 1)
    te = min(root.token_end, max(te, ts + 1))
    return _Occ(
        row=root.row, root_idx=root.root_idx, span=(p, q), text=text,
        size=float(q - p), edit_type=root.edit_type,
        token_start=ts, token_end=te, char_start=cs, char_end=ce, uid=uid,
    )


def _pair_budget_indices(n: int, max_pairs: int) -> List[Tuple[int, int]]:
    """All pairs if small; otherwise star + stride sampling under budget.

    ``max_pairs <= 0`` means *no* pairwise LCS (exact-hash only) — near-zero
    alignment overhead on top of GRPO.
    """
    if n < 2 or max_pairs == 0:
        return []
    if max_pairs < 0:
        # negative => unlimited
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    total = n * (n - 1) // 2
    if total <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs: List[Tuple[int, int]] = []
    # Star: compare everyone to the first few hubs.
    n_hubs = max(1, min(n - 1, int(math.ceil(math.sqrt(max_pairs)))))
    for h in range(n_hubs):
        for j in range(h + 1, n):
            pairs.append((h, j))
            if len(pairs) >= max_pairs:
                return pairs
    # Fill with strided pairs.
    step = max(1, n // 4)
    for i in range(n):
        for j in range(i + step, n, step):
            pairs.append((i, j))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _opcode_matched_blocks(
    a: _RootView,
    b: _RootView,
    *,
    min_block_lines: int,
    sim_thresh: float,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
    """Return matched (span_a, span_b, canonical_text) from difflib opcodes."""
    if not a.lines or not b.lines:
        return []
    sm = SequenceMatcher(None, a.lines, b.lines, autojunk=False)
    out: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            if (i2 - i1) < min_block_lines:
                continue
            text = "\n".join(a.lines[i1:i2])
            out.append(((i1, i2), (j1, j2), text))
        elif tag == "replace":
            # High-similarity replacements still form useful anchors (Fig.2 style).
            if (i2 - i1) < min_block_lines and (j2 - j1) < min_block_lines:
                continue
            ta = "\n".join(a.lines[i1:i2])
            tb = "\n".join(b.lines[j1:j2])
            if not ta.strip() or not tb.strip():
                continue
            if ta == tb or source_similarity(ta, tb) >= sim_thresh:
                # Canonicalize to the longer text for hashing stability.
                canon = ta if len(ta) >= len(tb) else tb
                out.append(((i1, i2), (j1, j2), canon))
    return out


def _dedupe_occs_by_row(occs: List[_Occ]) -> List[_Occ]:
    by_row: Dict[int, _Occ] = {}
    for o in occs:
        prev = by_row.get(o.row)
        if prev is None or o.size > prev.size:
            by_row[o.row] = o
    return list(by_row.values())


def _rep_centric_merge(
    buckets: Dict[str, List[_Occ]],
    sim_thresh: float,
) -> Dict[str, List[_Occ]]:
    """Merge fuzzy near-duplicate buckets against a representative (no long chains).

    Exact-hash buckets are already separate; this only merges buckets whose
    representative texts are pairwise-similar to a chosen survivor rep.
    One-shot (non-transitive): each bucket attaches to the first compatible rep.
    """
    if sim_thresh >= 1.0 or len(buckets) <= 1:
        return buckets
    items = [(k, v, v[0].text if v else "") for k, v in buckets.items() if v]
    items.sort(key=lambda x: -len(x[1]))  # larger support first
    survivors: List[Tuple[str, List[_Occ], str]] = []
    for key, occs, rep in items:
        attached = False
        for i, (sk, soccs, srep) in enumerate(survivors):
            if source_similarity(rep, srep) >= sim_thresh:
                soccs.extend(occs)
                survivors[i] = (sk, soccs, srep)
                attached = True
                break
        if not attached:
            survivors.append((key, list(occs), rep))
    return {k: _dedupe_occs_by_row(occs) for k, occs, _ in survivors}


def _greedy_select_anchors(
    candidates: List[_Anchor],
    max_anchors: int,
) -> List[_Anchor]:
    """Facility-location greedy on occurrence keys (row, root_idx, span)."""
    if not candidates:
        return []
    K = max(1, int(max_anchors)) if max_anchors and max_anchors > 0 else len(candidates)
    covered_best: Dict[Tuple[int, int, Tuple[int, int]], float] = {}
    for anc in candidates:
        for o in anc.occ:
            covered_best[(o.row, o.root_idx, o.span)] = 0.0

    selected: List[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < K:
        best_a = None
        best_gain = 0.0
        for aidx in remaining:
            anc = candidates[aidx]
            if anc.gs <= 0.0:
                continue
            gain = 0.0
            for o in anc.occ:
                key = (o.row, o.root_idx, o.span)
                gain += max(covered_best.get(key, 0.0), anc.gs) - covered_best.get(key, 0.0)
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_a = aidx
        if best_a is None or best_gain <= 1e-12:
            break
        selected.append(best_a)
        remaining.remove(best_a)
        for o in candidates[best_a].occ:
            key = (o.row, o.root_idx, o.span)
            covered_best[key] = max(covered_best.get(key, 0.0), candidates[best_a].gs)
    return [candidates[i] for i in selected]


def _nonoverlap_cut(anchors: List[_Anchor]) -> List[_Anchor]:
    """Per (row, root_idx): keep non-overlapping spans, preferring higher GS."""
    # Collect (gs, anc_idx, occ) then schedule per root.
    per_root: Dict[Tuple[int, int], List[Tuple[float, int, _Occ]]] = defaultdict(list)
    for aidx, anc in enumerate(anchors):
        for o in anc.occ:
            per_root[(o.row, o.root_idx)].append((anc.gs, aidx, o))

    keep_occ: Dict[int, List[_Occ]] = defaultdict(list)  # aidx -> kept occs
    for _, items in per_root.items():
        items.sort(key=lambda t: (-t[0], -(t[2].span[1] - t[2].span[0])))
        taken: List[Tuple[int, int]] = []
        for gs, aidx, o in items:
            p, q = o.span
            if any(not (q <= tp or p >= tq) for tp, tq in taken):
                continue
            taken.append((p, q))
            keep_occ[aidx].append(o)

    out: List[_Anchor] = []
    for aidx, anc in enumerate(anchors):
        occs = _dedupe_occs_by_row(keep_occ.get(aidx, []))
        if len(occs) <= 1:
            continue  # singleton after cut → useless for baseline
        mean_size = float(np.mean([o.size for o in occs]))
        # Recompute n/gs after cut (support may drop).
        n = len(occs)
        gs = groupability_score(mean_size, n, s0=8.0, count_ref=8.0)  # overwritten by caller params below
        out.append(_Anchor(occs, gs, mean_size, n, anc.preview, anc.uid))
    return out


def _rebuild_anchor_gs(
    anchors: List[_Anchor],
    *,
    s0: float,
    count_ref: float,
    phi_form: str,
    psi_form: str,
) -> List[_Anchor]:
    rebuilt: List[_Anchor] = []
    for anc in anchors:
        n = len(anc.occ)
        if n <= 1:
            continue
        mean_size = float(np.mean([o.size for o in anc.occ]))
        gs = groupability_score(mean_size, n, s0=s0, count_ref=count_ref,
                                phi_form=phi_form, psi_form=psi_form)
        if gs <= 0.0:
            continue
        rebuilt.append(_Anchor(anc.occ, gs, mean_size, n, anc.preview, anc.uid))
    return rebuilt


def _build_anchors_for_uid(
    roots: List[Tuple[int, int, Dict[str, Any]]],
    uid: Any,
    *,
    sim_thresh: float,
    max_span: int,  # retained for API compat; unused in LCS path
    s0: float,
    count_ref: float,
    phi_form: str,
    psi_form: str,
    max_anchors: int,
    min_block_lines: int = 1,
    max_pairs_per_uid: int = 64,
) -> List[_Anchor]:
    """LCS-opcode alignment + exact-hash + greedy C* + non-overlap cut.

    Engineering path (preferred over sliding-window × union-find):
      1) exact full-root / block hashing
      2) pairwise difflib opcodes → matched blocks (Fig.2-style)
      3) rep-centric fuzzy merge (no transitive paste)
      4) greedy facility-location on GS
      5) per-root non-overlapping interval scheduling
    """
    del max_span  # API compat with older call sites / config
    views: List[_RootView] = []
    for row, root_idx, snip in roots:
        if not snip.get("changed", True):
            continue
        origin = snip.get("origin", ORIGIN_ROOT)
        if origin == ORIGIN_FALLBACK:
            continue
        v = _RootView(row, root_idx, snip)
        if not v.lines:
            continue
        views.append(v)
    if len(views) < 2:
        return []

    # ---- bucket keyed by (edit_type, text_hash) -> occs ---- #
    buckets: Dict[str, List[_Occ]] = defaultdict(list)

    # (1) Exact full-root matches
    for v in views:
        full = "\n".join(v.lines)
        key = f"{v.edit_type}::full::{_text_hash(full)}"
        occ = _occ_from_root_span(v, (0, len(v.lines)), uid)
        if occ is not None:
            buckets[key].append(occ)

    # (2) Pairwise LCS / opcodes → matched blocks
    # Group views by edit_type so add/del/none never cross-match.
    by_etype: Dict[str, List[_RootView]] = defaultdict(list)
    for v in views:
        by_etype[v.edit_type].append(v)

    for etype, group in by_etype.items():
        pairs = _pair_budget_indices(len(group), max_pairs_per_uid)
        for i, j in pairs:
            a, b = group[i], group[j]
            for span_a, span_b, canon in _opcode_matched_blocks(
                a, b, min_block_lines=min_block_lines, sim_thresh=sim_thresh
            ):
                key = f"{etype}::blk::{_text_hash(canon)}"
                oa = _occ_from_root_span(a, span_a, uid)
                ob = _occ_from_root_span(b, span_b, uid)
                if oa is not None:
                    # Prefer storing the actual local text; hash key uses canon.
                    buckets[key].append(oa)
                if ob is not None:
                    buckets[key].append(ob)

    # Dedupe inside buckets, then optional fuzzy merge across near-duplicate keys.
    buckets = {k: _dedupe_occs_by_row(v) for k, v in buckets.items() if v}
    buckets = _rep_centric_merge(buckets, sim_thresh)

    candidates: List[_Anchor] = []
    for occs in buckets.values():
        occs = _dedupe_occs_by_row(occs)
        if len(occs) <= 1:
            continue
        mean_size = float(np.mean([o.size for o in occs]))
        n = len(occs)
        gs = groupability_score(mean_size, n, s0=s0, count_ref=count_ref,
                                phi_form=phi_form, psi_form=psi_form)
        if gs <= 0.0:
            continue
        preview = occs[0].text[:120].replace("\n", "\\n")
        candidates.append(_Anchor(occs, gs, mean_size, n, preview, uid))

    selected = _greedy_select_anchors(candidates, max_anchors)
    # (5) Non-overlapping cut per root, then refresh GS with true support.
    cut = _nonoverlap_cut(selected)
    return _rebuild_anchor_gs(cut, s0=s0, count_ref=count_ref,
                              phi_form=phi_form, psi_form=psi_form)


# --------------------------------------------------------------------------- #
# Structural-fallback grouping (legacy signature bucket → refine)             #
# --------------------------------------------------------------------------- #
def _refine_bucket(items: List[Tuple[int, int, str]],
                   sim_thresh: float) -> List[List[Tuple[int, int]]]:
    clusters: List[Dict[str, Any]] = []
    for row, sidx, src in items:
        placed = False
        for cl in clusters:
            if sim_thresh >= 1.0:
                ok = (src == cl["rep"])
            else:
                ok = source_similarity(src, cl["rep"]) >= sim_thresh
            if ok:
                cl["members"].append((row, sidx))
                placed = True
                break
        if not placed:
            clusters.append({"rep": src, "members": [(row, sidx)]})
    return [cl["members"] for cl in clusters]


def build_snippet_groups(snippets_per_row: Sequence[List[Dict[str, Any]]],
                         index: np.ndarray,
                         level: int,
                         sim_thresh: float = 0.8) -> Dict[Tuple[int, int], str]:
    """Legacy / fallback grouping by (uid, signature) then similarity refine."""
    buckets: Dict[Tuple[Any, str], List[Tuple[int, int, str]]] = defaultdict(list)
    for row, snips in enumerate(snippets_per_row):
        uid = index[row]
        for sidx, s in enumerate(snips):
            if s.get("level", -1) != level or not s.get("changed", True):
                continue
            # Prefer fallback units here; roots are handled by alignment.
            if s.get("origin") == ORIGIN_ROOT:
                continue
            sig = str(s.get("signature", ""))
            buckets[(uid, sig)].append((row, sidx, s.get("source", "")))

    mapping: Dict[Tuple[int, int], str] = {}
    for _, items in buckets.items():
        for members in _refine_bucket(items, sim_thresh):
            gid = str(uuid.uuid4())
            for (row, sidx) in members:
                mapping[(row, sidx)] = gid
    return mapping


# --------------------------------------------------------------------------- #
# Episode / group baselines                                                   #
# --------------------------------------------------------------------------- #
def episode_norm_reward(token_level_rewards,  # torch.Tensor
                        response_mask,
                        index: np.ndarray,
                        traj_index: np.ndarray,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True):
    """Episode-level GRPO/GiGPO-style advantage (aligned with ``gigpo.core_gigpo``).

    With the default ``compute_mean_std_cross_steps=True`` (same as GRPO/GiGPO),
    every flattened multi-turn row contributes to the group mean/std. Set False
    to count each ``(uid, traj_uid)`` once (true episode-level baseline).
    """
    import torch

    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score: Dict[Any, List] = defaultdict(list)
    id2mean: Dict[Any, Any] = {}
    id2std: Dict[Any, Any] = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                # Match GiGPO/GRPO nested-list std construction.
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    return episode_advantages


def _group_baselined_scores(step_rewards,
                            group_of: Dict[Tuple[int, int], str],
                            epsilon: float,
                            remove_std: bool) -> Dict[Tuple[int, int], float]:
    grp2vals: Dict[str, List[float]] = defaultdict(list)
    for (row, _sidx), gid in group_of.items():
        val = float(step_rewards[row]) if not hasattr(step_rewards[row], "item") else float(step_rewards[row].item())
        grp2vals[gid].append(val)

    grp2mean: Dict[str, float] = {}
    grp2std: Dict[str, float] = {}
    for gid, vals in grp2vals.items():
        arr = np.asarray(vals, dtype=np.float64)
        grp2mean[gid] = float(arr.mean())
        grp2std[gid] = float(arr.std()) if len(arr) > 1 else 1.0

    adv: Dict[Tuple[int, int], float] = {}
    for (row, sidx), gid in group_of.items():
        val = float(step_rewards[row]) if not hasattr(step_rewards[row], "item") else float(step_rewards[row].item())
        a = val - grp2mean[gid]
        if not remove_std:
            a = a / (grp2std[gid] + epsilon)
        adv[(row, sidx)] = a
    return adv


# --------------------------------------------------------------------------- #
# Diagnostics helpers                                                         #
# --------------------------------------------------------------------------- #
def _empty_diagnostics() -> Dict[str, Any]:
    return {
        "n_groups": 0,
        "mean_group_size": 0.0,
        "median_group_size": 0.0,
        "singleton_rate": 0.0,
        "n_anchors_selected": 0,
        "alignment_group_frac": 0.0,
        "fallback_group_frac": 0.0,
        "changed_snippet_ratio": 0.0,
        "snippet_token_frac": 0.0,
        "align_time_ms": 0.0,
        "group_size_hist": {},
        "per_uid": {},
    }


def resolve_overhead_mode(mode: str) -> Dict[str, Any]:
    """Map overhead_mode → concrete alignment / fallback knobs.

    Relative to GRPO, DiDPO's *intrinsic* extra work is CPU-only. These modes
    keep that cost negligible vs. vLLM rollout + FSDP update:

      - ``exact_only``   : exact-hash anchors only (no LCS pairs, no AST fallback)
      - ``lightweight``  : exact-hash + few LCS pairs; AST fallback only if no align groups
      - ``full``         : full LCS budget + structural fallback always on
    """
    m = (mode or "lightweight").lower()
    if m in ("exact_only", "exact", "hash_only"):
        return {
            "overhead_mode": "exact_only",
            "max_pairs_per_uid": 0,
            "use_structural_fallback": False,
            "fallback_if_no_align": False,
            "collector_mode": "exact_only",
        }
    if m in ("lightweight", "light", "roots_only"):
        return {
            "overhead_mode": "lightweight",
            "max_pairs_per_uid": 8,
            "use_structural_fallback": False,
            "fallback_if_no_align": True,
            "collector_mode": "lightweight",
        }
    return {
        "overhead_mode": "full",
        "max_pairs_per_uid": None,  # keep caller default
        "use_structural_fallback": True,
        "fallback_if_no_align": False,
        "collector_mode": "full",
    }


def summarize_group_sizes(sizes: List[int]) -> Dict[str, float]:
    if not sizes:
        return {"n_groups": 0, "mean_group_size": 0.0, "median_group_size": 0.0,
                "singleton_rate": 0.0}
    arr = np.asarray(sizes, dtype=np.float64)
    return {
        "n_groups": int(len(sizes)),
        "mean_group_size": float(arr.mean()),
        "median_group_size": float(np.median(arr)),
        "singleton_rate": float(np.mean(arr <= 1.0)),
    }


# --------------------------------------------------------------------------- #
# Main entry                                                                  #
# --------------------------------------------------------------------------- #
def compute_didpo_outcome_advantage(token_level_rewards,
                                    step_rewards,
                                    response_mask,
                                    snippets: np.ndarray,
                                    index: np.ndarray,
                                    traj_index: np.ndarray,
                                    epsilon: float = 1e-6,
                                    snippet_advantage_w: float = 1.0,
                                    mode: str = "mean_norm",
                                    sim_thresh: float = 0.8,
                                    phi_s0: float = 8.0,
                                    psi_count_ref: float = 8.0,
                                    phi_form: str = "saturating",
                                    psi_form: str = "saturating",
                                    level_select: str = "argmax_gs",
                                    gs_min: float = 0.0,
                                    max_span_lines: int = 32,
                                    max_anchors_per_uid: int = 64,
                                    min_block_lines: int = 1,
                                    max_pairs_per_uid: int = 64,
                                    use_alignment: bool = True,
                                    use_structural_fallback: bool = True,
                                    overhead_mode: str = "lightweight",
                                    instance_ids: Optional[np.ndarray] = None,
                                    return_diagnostics: bool = True):
    """Compute DiDPO advantages; optionally return diagnostics for logging."""
    import torch

    ov = resolve_overhead_mode(overhead_mode)
    if ov["max_pairs_per_uid"] is not None:
        max_pairs_per_uid = int(ov["max_pairs_per_uid"])
    # Structural fallback: full always; lightweight only if alignment found nothing.
    fallback_if_no_align = bool(ov.get("fallback_if_no_align", False))
    if ov["overhead_mode"] == "full":
        use_structural_fallback = True if use_structural_fallback is None else use_structural_fallback
    elif ov["overhead_mode"] == "exact_only":
        use_structural_fallback = False
    else:
        # lightweight: defer — may enable after alignment if empty
        use_structural_fallback = False

    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    device = token_level_rewards.device
    bsz, response_length = token_level_rewards.shape

    episode_advantages = episode_norm_reward(
        token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)

    snippets_per_row: List[List[Dict[str, Any]]] = [
        list(snippets[i]) if snippets[i] is not None else [] for i in range(bsz)
    ]

    # Synthetic units produced by alignment (appended per row for token fill).
    aligned_units: List[List[Dict[str, Any]]] = [[] for _ in range(bsz)]
    # group_of maps (row, unit_key) where unit_key is ("align", local_idx) or ("fb", sidx)
    group_of: Dict[Tuple[Any, ...], str] = {}
    group_meta: Dict[str, Dict[str, Any]] = {}  # gid -> {size, origin, uid, preview, gs}

    # -------------------- alignment path -------------------- #
    n_align_groups = 0
    align_t0 = time.perf_counter()
    if use_alignment:
        uid_to_roots: Dict[Any, List[Tuple[int, int, Dict[str, Any]]]] = defaultdict(list)
        for row, snips in enumerate(snippets_per_row):
            for sidx, s in enumerate(snips):
                origin = s.get("origin", ORIGIN_FALLBACK)
                if origin == ORIGIN_ROOT or (
                    origin != ORIGIN_FALLBACK and s.get("level") == LEVEL_FUNCTION and "root::" in str(s.get("signature", ""))
                ):
                    uid_to_roots[index[row]].append((row, sidx, s))
                elif origin not in (ORIGIN_ROOT, ORIGIN_FALLBACK) and s.get("lines"):
                    # Older dicts without origin: treat function-level as root-like.
                    if s.get("level") == LEVEL_FUNCTION:
                        uid_to_roots[index[row]].append((row, sidx, {**s, "origin": ORIGIN_ROOT}))

        for uid, roots in uid_to_roots.items():
            anchors = _build_anchors_for_uid(
                roots, uid,
                sim_thresh=sim_thresh,
                max_span=max_span_lines,
                s0=phi_s0,
                count_ref=psi_count_ref,
                phi_form=phi_form,
                psi_form=psi_form,
                max_anchors=max_anchors_per_uid,
                min_block_lines=min_block_lines,
                max_pairs_per_uid=max_pairs_per_uid,
            )
            for anc in anchors:
                if anc.gs <= gs_min or anc.n <= 1:
                    # Singletons cannot support a baseline; skip (paper ψ(1)=0).
                    continue
                gid = str(uuid.uuid4())
                n_align_groups += 1
                group_meta[gid] = {
                    "size": anc.n,
                    "origin": "alignment",
                    "uid": uid,
                    "preview": anc.preview,
                    "gs": float(anc.gs),
                    "mean_size": float(anc.mean_size),
                }
                for o in anc.occ:
                    local_idx = len(aligned_units[o.row])
                    unit = {
                        "level": LEVEL_FUNCTION,
                        "signature": f"align::{gid[:8]}",
                        "source": o.text,
                        "size": o.size,
                        "token_start": o.token_start,
                        "token_end": o.token_end,
                        "char_start": o.char_start,
                        "char_end": o.char_end,
                        "changed": True,
                        "edit_type": o.edit_type,
                        "origin": "alignment",
                        "gs": float(anc.gs),
                        "group_size": anc.n,
                    }
                    aligned_units[o.row].append(unit)
                    group_of[("align", o.row, local_idx)] = gid
    align_time_ms = (time.perf_counter() - align_t0) * 1000.0

    # Lightweight: if alignment produced nothing, optionally fall back to AST groups.
    if fallback_if_no_align and n_align_groups == 0:
        use_structural_fallback = True

    # -------------------- structural fallback -------------------- #
    n_fb_groups = 0
    if use_structural_fallback:
        for lvl in range(NUM_LEVELS):
            gmap = build_snippet_groups(snippets_per_row, index, lvl, sim_thresh)
            # sizes
            gcount: Dict[str, int] = defaultdict(int)
            for k, gid in gmap.items():
                gcount[gid] += 1
            for (row, sidx), gid in gmap.items():
                sz = gcount[gid]
                if sz <= 1:
                    continue  # singleton → no local credit
                if gid not in group_meta:
                    n_fb_groups += 1
                    sn = snippets_per_row[row][sidx]
                    group_meta[gid] = {
                        "size": sz,
                        "origin": "fallback",
                        "uid": index[row],
                        "preview": str(sn.get("source", ""))[:120].replace("\n", "\\n"),
                        "gs": groupability_score(float(sn.get("size", 1.0)), sz,
                                                 s0=phi_s0, count_ref=psi_count_ref,
                                                 phi_form=phi_form, psi_form=psi_form),
                        "mean_size": float(sn.get("size", 1.0)),
                    }
                group_of[("fb", row, sidx)] = gid

    # Build combined (row -> units for token fill) with GS for selection
    fill_units: List[List[Dict[str, Any]]] = [[] for _ in range(bsz)]
    key_to_unit_pos: Dict[Tuple[Any, ...], Tuple[int, int]] = {}

    for row in range(bsz):
        for local_idx, u in enumerate(aligned_units[row]):
            fill_units[row].append(u)
            key_to_unit_pos[("align", row, local_idx)] = (row, len(fill_units[row]) - 1)
        if use_structural_fallback:
            for sidx, s in enumerate(snippets_per_row[row]):
                key = ("fb", row, sidx)
                if key not in group_of:
                    continue
                gid = group_of[key]
                meta = group_meta.get(gid, {})
                if meta.get("size", 0) <= 1:
                    continue
                unit = {
                    **s,
                    "gs": float(meta.get("gs", 0.0)),
                    "group_size": int(meta.get("size", 1)),
                    "origin": "fallback",
                }
                fill_units[row].append(unit)
                key_to_unit_pos[key] = (row, len(fill_units[row]) - 1)

    # Group-baselined A^D for every grouped unit key
    # Map fill positions back for scoring: use a flat group_of on (row, fill_idx)
    flat_group_of: Dict[Tuple[int, int], str] = {}
    for key, gid in group_of.items():
        if key not in key_to_unit_pos:
            continue
        row, fill_idx = key_to_unit_pos[key]
        flat_group_of[(row, fill_idx)] = gid

    level_adv = _group_baselined_scores(step_rewards, flat_group_of, epsilon, remove_std)

    # Token fill with argmax-GS among covering units
    snippet_advantages = torch.zeros((bsz, response_length), dtype=episode_advantages.dtype, device=device)
    selected_level_counter = defaultdict(int)
    grouped_token_count = 0
    align_token_count = 0
    fallback_token_count = 0

    for row in range(bsz):
        units = fill_units[row]
        if not units:
            continue
        best_gs = np.full(response_length, -1.0, dtype=np.float64)
        best_adv = np.zeros(response_length, dtype=np.float64)
        best_origin = np.full(response_length, -1, dtype=np.int64)  # 0 align, 1 fb
        best_lvl = np.full(response_length, -1, dtype=np.int64)

        for fill_idx, u in enumerate(units):
            key = (row, fill_idx)
            if key not in flat_group_of:
                continue
            count = int(u.get("group_size", 1))
            size = float(u.get("size", 1.0))
            gs = float(u.get("gs", groupability_score(
                size, count, s0=phi_s0, count_ref=psi_count_ref,
                phi_form=phi_form, psi_form=psi_form)))
            if level_select == "bpe_greedy" and int(u.get("level", 0)) == LEVEL_FUNCTION and gs > gs_min:
                gs = gs + 1.0
            if gs <= gs_min:
                continue
            a = level_adv.get(key, 0.0)
            ts, te = int(u.get("token_start", -1)), int(u.get("token_end", -1))
            if ts < 0 or te <= ts:
                continue
            ts = max(0, ts)
            te = min(response_length, te)
            origin_id = 0 if u.get("origin") == "alignment" else 1
            lvl = int(u.get("level", 0))
            for t in range(ts, te):
                if gs > best_gs[t]:
                    best_gs[t] = gs
                    best_adv[t] = a
                    best_origin[t] = origin_id
                    best_lvl[t] = lvl

        sel = best_lvl >= 0
        if sel.any():
            snippet_advantages[row] = torch.tensor(best_adv, dtype=snippet_advantages.dtype, device=device)
            grouped_token_count += int(sel.sum())
            align_token_count += int(((best_origin == 0) & sel).sum())
            fallback_token_count += int(((best_origin == 1) & sel).sum())
            for lvl in range(NUM_LEVELS):
                selected_level_counter[lvl] += int((best_lvl == lvl).sum())

    snippet_advantages = snippet_advantages * response_mask
    scores = episode_advantages + snippet_advantage_w * snippet_advantages

    # -------------------- diagnostics -------------------- #
    diagnostics = _empty_diagnostics()
    sizes = [int(m["size"]) for m in group_meta.values()]
    diagnostics.update(summarize_group_sizes(sizes))
    diagnostics["n_anchors_selected"] = int(n_align_groups)
    diagnostics["align_time_ms"] = float(align_time_ms)
    total_g = max(1, n_align_groups + n_fb_groups)
    diagnostics["alignment_group_frac"] = float(n_align_groups) / float(total_g)
    diagnostics["fallback_group_frac"] = float(n_fb_groups) / float(total_g)

    n_changed = 0
    n_total = 0
    for snips in snippets_per_row:
        for s in snips:
            n_total += 1
            if s.get("changed", True):
                n_changed += 1
    diagnostics["changed_snippet_ratio"] = float(n_changed) / float(max(1, n_total))
    valid_tokens = float(response_mask.sum().item()) if hasattr(response_mask, "sum") else float(np.sum(response_mask))
    diagnostics["snippet_token_frac"] = float(grouped_token_count) / max(1.0, valid_tokens)
    hist: Dict[str, int] = defaultdict(int)
    for sz in sizes:
        bucket = str(sz) if sz <= 8 else ("9-16" if sz <= 16 else "17+")
        hist[bucket] += 1
    diagnostics["group_size_hist"] = dict(hist)

    # Per-uid detail (rollout group uuid) + per-instance (stable task id)
    per_uid: Dict[str, Dict[str, Any]] = {}
    uid_groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for gid, meta in group_meta.items():
        uid_groups[meta["uid"]].append({
            "group_id": gid[:8],
            "size": int(meta["size"]),
            "gs": float(meta.get("gs", 0.0)),
            "origin": meta.get("origin"),
            "preview": meta.get("preview", ""),
            "mean_size": float(meta.get("mean_size", 0.0)),
        })
    for uid, groups in uid_groups.items():
        gsizes = [g["size"] for g in groups]
        per_uid[str(uid)] = {
            **summarize_group_sizes(gsizes),
            "groups": sorted(groups, key=lambda g: (-g["size"], -g["gs"])),
        }
    diagnostics["per_uid"] = per_uid

    # Remap uid → instance_id when the collector attached instance_id per row.
    per_instance: Dict[str, Dict[str, Any]] = {}
    if instance_ids is not None and len(instance_ids) == bsz:
        uid_to_iid: Dict[str, str] = {}
        all_iids: List[str] = []
        for row in range(bsz):
            iid = str(instance_ids[row])
            all_iids.append(iid)
            u = str(index[row])
            if u not in uid_to_iid:
                uid_to_iid[u] = iid
        # Ensure every instance that appeared is represented (even with 0 groups),
        # so the tracker can mark present=1 / n_groups=0 correctly.
        for iid in dict.fromkeys(all_iids):
            if iid:
                per_instance[iid] = {
                    **summarize_group_sizes([]),
                    "groups": [],
                }
        # Merge group lists if multiple uids somehow share an instance_id.
        iid_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for uid, groups in uid_groups.items():
            iid = uid_to_iid.get(str(uid), str(uid))
            iid_groups[iid].extend(groups)
        for iid, groups in iid_groups.items():
            # Deduplicate by group_id (same group listed once).
            seen_g: Set[str] = set()
            uniq: List[Dict[str, Any]] = []
            for g in groups:
                gid = g.get("group_id")
                if gid in seen_g:
                    continue
                seen_g.add(gid)
                uniq.append(g)
            gsizes = [g["size"] for g in uniq]
            per_instance[iid] = {
                **summarize_group_sizes(gsizes),
                "groups": sorted(uniq, key=lambda g: (-g["size"], -g["gs"])),
            }
    diagnostics["per_instance"] = per_instance

    diagnostics["align_token_count"] = int(align_token_count)
    diagnostics["fallback_token_count"] = int(fallback_token_count)
    diagnostics["selected_level_token_counts"] = {str(k): int(v) for k, v in selected_level_counter.items()}

    _print_diagnostics(diagnostics)

    if return_diagnostics:
        return scores, scores, diagnostics
    return scores, scores


def _print_diagnostics(diagnostics: Dict[str, Any]) -> None:
    print(
        "[DIDPO] groups=", diagnostics.get("n_groups"),
        "mean_size=", round(float(diagnostics.get("mean_group_size", 0.0)), 3),
        "singleton_rate=", round(float(diagnostics.get("singleton_rate", 0.0)), 3),
        "anchors=", diagnostics.get("n_anchors_selected"),
        "align_frac=", round(float(diagnostics.get("alignment_group_frac", 0.0)), 3),
        "snippet_tok_frac=", round(float(diagnostics.get("snippet_token_frac", 0.0)), 3),
        "align_ms=", round(float(diagnostics.get("align_time_ms", 0.0)), 1),
    )


def diagnostics_to_metrics(diagnostics: Dict[str, Any], prefix: str = "didpo") -> Dict[str, float]:
    """Flatten diagnostics into scalar metrics for SwanLab / wandb."""
    if not diagnostics:
        return {}
    out = {
        f"{prefix}/n_groups": float(diagnostics.get("n_groups", 0)),
        f"{prefix}/mean_group_size": float(diagnostics.get("mean_group_size", 0.0)),
        f"{prefix}/median_group_size": float(diagnostics.get("median_group_size", 0.0)),
        f"{prefix}/singleton_rate": float(diagnostics.get("singleton_rate", 0.0)),
        f"{prefix}/n_anchors_selected": float(diagnostics.get("n_anchors_selected", 0)),
        f"{prefix}/alignment_group_frac": float(diagnostics.get("alignment_group_frac", 0.0)),
        f"{prefix}/fallback_group_frac": float(diagnostics.get("fallback_group_frac", 0.0)),
        f"{prefix}/changed_snippet_ratio": float(diagnostics.get("changed_snippet_ratio", 0.0)),
        f"{prefix}/snippet_token_frac": float(diagnostics.get("snippet_token_frac", 0.0)),
        f"{prefix}/align_time_ms": float(diagnostics.get("align_time_ms", 0.0)),
    }
    for bucket, cnt in (diagnostics.get("group_size_hist") or {}).items():
        out[f"{prefix}/group_size_hist/{bucket}"] = float(cnt)
    return out
