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
Deterministic verification harness for the DIDPO algorithm itself.

This exercises the *real* production functions (no re-implementation) on
hand-constructed rollouts with known structure, and checks that:

  1. ``phi`` / ``psi`` / ``GS`` have the designed shape (boilerplate suppressed,
     singleton groups score 0, GS rises with snippet mass at fixed group size).
  2. **diff-gating** marks a byte-identical re-emission of the same signature as
     ``changed=False`` (no credit), and a real edit as ``changed=True``.
  3. snippet extraction + **token-span mapping** land on the right substring of
     the *full response* (verified on real tokens via a 1-1 char tokenizer).
  4. **cross-rollout functional grouping** clusters identical functions across
     rollouts of the same instance and keeps a divergent one separate.
  5. the **group-baselined snippet advantage** equals reward - group-mean.
  6. (torch-gated) the full ``compute_didpo_outcome_advantage`` assembles
     A = A_episode + lambda * A_snippet per token, picking the function level
     via argmax-GS, with hand-computed expected values.

Run with plain Python (no pytest needed):

    cd verl-code && python3 didpo/tests/test_didpo_algorithm.py

The torch-free parts (1-5) run anywhere; part (6) auto-skips if torch is not
installed and runs the real tensor path where it is.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter

# Make the repo root importable when run as a standalone script.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from didpo.snippet import (
    LEVEL_FUNCTION,
    LEVEL_HUNK,
    apply_diff_gate,
    extract_snippets,
    parse_response_to_snippets,
)
from didpo.core_didpo import (
    _group_baselined_scores,
    build_snippet_groups,
    groupability_score,
    phi_size,
    psi_count,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
class CharTokenizer:
    """A 1-to-1 character tokenizer: token id == ord(char).

    ``decode([id]) == chr(id)``, so the reconstructed text equals the original
    response and every token spans exactly one character. This makes token
    spans trivially hand-verifiable (token_start == char_start).
    """

    def decode(self, ids, skip_special_tokens=False):  # noqa: D401
        return "".join(chr(int(i)) for i in ids)


def encode(text: str):
    return [ord(c) for c in text]


def _fn_response(body: str) -> str:
    """An agent response that edits one file with a single ``add`` function."""
    return (
        "<think>fix the bug</think>"
        '<edit path="src/calc.py"><code>\n'
        "def add(a, b):\n"
        f"    {body}\n"
        "</code></edit>"
    )


def _check(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# --------------------------------------------------------------------------- #
# 1. Groupability score shape                                                 #
# --------------------------------------------------------------------------- #
def test_groupability_score():
    print("[1] groupability score (phi / psi / GS) ...")
    # phi: increasing in size, bounded in (0, 1) for saturating form.
    _check(phi_size(1) < phi_size(10) < phi_size(100) < 1.0, "phi increases with size and < 1")
    _check(phi_size(1000, form="saturating") > 0.99, "phi saturates toward 1 for big snippets")

    # psi: zero for a singleton, increasing, saturates to ~1 at count_ref.
    _check(psi_count(1) == 0.0, "psi(1) == 0 (a singleton cannot support a baseline)")
    _check(psi_count(2) < psi_count(4) < psi_count(8), "psi increases with #siblings")
    _check(math.isclose(psi_count(8, count_ref=8, form="log"), 1.0, abs_tol=1e-9),
           "psi saturates to 1 at count == count_ref")

    # GS = phi * psi. Singleton -> 0 regardless of mass. Bigger mass -> bigger GS
    # at fixed group size (the 'function level wins over boilerplate' intuition).
    _check(groupability_score(size=100, count=1) == 0.0, "GS == 0 for any singleton snippet")
    gs_boiler = groupability_score(size=2, count=6)
    gs_func = groupability_score(size=40, count=6)
    _check(gs_func > gs_boiler, "GS(big snippet) > GS(boilerplate) at equal group size")


# --------------------------------------------------------------------------- #
# 2. Diff-gating                                                              #
# --------------------------------------------------------------------------- #
def test_diff_gating():
    print("[2] diff-gating across steps of one trajectory ...")
    prev: dict = {}

    step1 = extract_snippets(_fn_response("return a + b"))
    apply_diff_gate(step1, prev)
    _check(all(s.changed for s in step1), "first emission of every snippet is changed=True")

    # Re-emit the byte-identical code at the next step -> no-op -> gated out.
    step2 = extract_snippets(_fn_response("return a + b"))
    apply_diff_gate(step2, prev)
    _check(all(not s.changed for s in step2), "byte-identical re-emission is changed=False (no credit)")

    # A real edit (same signature, different body) -> changed again.
    step3 = extract_snippets(_fn_response("return a * b"))
    apply_diff_gate(step3, prev)
    fn3 = [s for s in step3 if s.level == LEVEL_FUNCTION]
    _check(fn3 and all(s.changed for s in fn3), "a genuine edit flips the snippet back to changed=True")


# --------------------------------------------------------------------------- #
# 3. Snippet extraction + token-span mapping on real tokens                   #
# --------------------------------------------------------------------------- #
def test_token_span_mapping():
    print("[3] snippet extraction + token-span mapping on real tokens ...")
    resp = _fn_response("return a + b")
    tok = CharTokenizer()
    ids = encode(resp)
    snips = parse_response_to_snippets(ids, tok, {})

    levels = {s["level"] for s in snips}
    _check(LEVEL_FUNCTION in levels, "a function-level snippet is recovered")
    _check(LEVEL_HUNK in levels, "a hunk-level (statement) snippet is recovered")

    for s in snips:
        _check(0 <= s["token_start"] < s["token_end"] <= len(ids),
               f"valid token span for level-{s['level']} snippet")
        # 1-1 char tokenizer => token span must equal char span.
        _check(s["token_start"] == s["char_start"] and s["token_end"] == s["char_end"],
               "token span == char span under the 1-1 char tokenizer")

    fn = [s for s in snips if s["level"] == LEVEL_FUNCTION][0]
    fn_text = resp[fn["char_start"]:fn["char_end"]]
    _check("def add" in fn_text, "function snippet span covers the 'def add' definition")


# --------------------------------------------------------------------------- #
# 4. Cross-rollout functional grouping                                        #
# --------------------------------------------------------------------------- #
def _rows_for_grouping():
    """4 rollouts emit an identical ``add``; a 5th emits a divergent body."""
    bodies = ["return a + b"] * 4 + ["return a - b"]
    rows = []
    for body in bodies:
        snips = extract_snippets(_fn_response(body))
        apply_diff_gate(snips, {})  # fresh memory per rollout -> all changed
        rows.append([s.to_dict() for s in snips])
    return rows


def test_cross_rollout_grouping():
    print("[4] cross-rollout functional grouping ...")
    rows = _rows_for_grouping()
    index = np.zeros(len(rows), dtype=int)  # all the same coding instance

    # Exact (sim_thresh=1.0) clustering for a deterministic assertion.
    gmap = build_snippet_groups(rows, index, LEVEL_FUNCTION, sim_thresh=1.0)
    gid_sizes = sorted(Counter(gmap.values()).values())
    _check(gid_sizes == [1, 4], "4 identical functions cluster together, the divergent one stays alone")

    # diff-gated (changed=False) snippets must be excluded from grouping.
    rows2 = _rows_for_grouping()
    for s in rows2[0]:
        if s["level"] == LEVEL_FUNCTION:
            s["changed"] = False  # pretend row 0's function was a no-op
    gmap2 = build_snippet_groups(rows2, index, LEVEL_FUNCTION, sim_thresh=1.0)
    _check(all(row != 0 for (row, _sidx) in gmap2), "diff-gated (no-op) snippet is excluded from grouping")

    # Different instances must not be grouped together.
    index_split = np.array([0, 0, 1, 1, 1])
    gmap3 = build_snippet_groups(_rows_for_grouping(), index_split, LEVEL_FUNCTION, sim_thresh=1.0)
    # Within uid 0: rows 0,1 identical -> size 2. Within uid 1: rows 2,3 identical (size2) + row4 alone.
    _check(sorted(Counter(gmap3.values()).values()) == [1, 2, 2],
           "grouping is restricted to rollouts of the same instance (uid)")


# --------------------------------------------------------------------------- #
# 5. Group-baselined snippet advantage                                        #
# --------------------------------------------------------------------------- #
def test_group_baselined_advantage():
    print("[5] group-baselined snippet advantage (reward - group mean) ...")
    # rows 0-3 in group A, row 4 alone in group B.
    gmap = {(0, 0): "A", (1, 0): "A", (2, 0): "A", (3, 0): "A", (4, 0): "B"}
    step_rewards = [1.0, 1.0, 0.0, 0.0, 0.7]
    adv = _group_baselined_scores(step_rewards, gmap, epsilon=1e-6, remove_std=True)
    # group A mean == 0.5 -> +/-0.5; singleton B -> baseline is its own value -> 0.
    for row, expected in [(0, 0.5), (1, 0.5), (2, -0.5), (3, -0.5), (4, 0.0)]:
        _check(math.isclose(adv[(row, 0)], expected, abs_tol=1e-9),
               f"A^snip(row {row}) == {expected}")


# --------------------------------------------------------------------------- #
# 6. End-to-end per-token advantage assembly (torch-gated)                    #
# --------------------------------------------------------------------------- #
def test_end_to_end_advantage():
    print("[6] end-to-end compute_didpo_outcome_advantage (real tensor path) ...")
    try:
        import torch
    except Exception:  # noqa: BLE001
        print("  SKIP: torch not installed -- run this on the training box "
              "(or `pip install torch`) to verify the per-token tensor assembly.")
        return

    from didpo.core_didpo import compute_didpo_outcome_advantage

    tok = CharTokenizer()
    resp = _fn_response("return a + b")
    ids = encode(resp)
    bsz, L = 4, len(ids)

    # 4 rollouts of the SAME instance, all emit the identical add() function.
    snippets = np.empty(bsz, dtype=object)
    for i in range(bsz):
        snippets[i] = parse_response_to_snippets(ids, tok, {})

    index = np.zeros(bsz, dtype=int)       # one instance group
    traj_index = np.arange(bsz)            # distinct trajectories
    final_reward = [1.0, 1.0, 0.0, 0.0]    # two pass, two fail

    token_level_rewards = torch.zeros(bsz, L)
    for i in range(bsz):
        token_level_rewards[i, -1] = final_reward[i]
    response_mask = torch.ones(bsz, L)
    # single step => discounted return-to-go == final reward.
    step_rewards = torch.tensor(final_reward)

    result = compute_didpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        step_rewards=step_rewards,
        response_mask=response_mask,
        snippets=snippets,
        index=index,
        traj_index=traj_index,
        snippet_advantage_w=1.0,
        sim_thresh=1.0,
        use_alignment=True,
        use_structural_fallback=True,
        overhead_mode="full",
        return_diagnostics=True,
    )
    adv, ret, diag = result

    _check(tuple(adv.shape) == (bsz, L), "advantage tensor has shape (bsz, L)")
    _check(torch.equal(adv, ret), "returns == advantages (critic-free)")
    _check(int(diag.get("n_groups", 0)) >= 1, "diagnostics report at least one non-singleton group")

    # episode advantage: scores == final reward; group mean 0.5 -> [.5,.5,-.5,-.5].
    # diff advantage on grouped code tokens: group mean 0.5 -> [.5,.5,-.5,-.5].
    # => grouped token total = +1.0 (pass) / -1.0 (fail); ungrouped token = episode only.
    fn = [s for s in snippets[0] if s["level"] == LEVEL_FUNCTION][0]
    t_code = fn["token_start"]  # a token inside the function / root diff
    _check(math.isclose(adv[0, t_code].item(), 1.0, abs_tol=1e-5),
           "pass rollout: grouped code token == A_episode + A_diff == +1.0")
    _check(math.isclose(adv[2, t_code].item(), -1.0, abs_tol=1e-5),
           "fail rollout: grouped code token == -1.0")

    # Token 0 is '<' of <think> -- outside any code snippet -> episode advantage only.
    _check(math.isclose(adv[0, 0].item(), 0.5, abs_tol=1e-5),
           "non-code token carries the episode advantage only (+0.5)")
    _check(math.isclose(adv[2, 0].item(), -0.5, abs_tol=1e-5),
           "non-code token of a fail rollout carries -0.5")


def test_alignment_anchors():
    """Paper core: identical sub-diffs across rollouts form an alignment group."""
    print("[7] dynamic sub-diff alignment anchors ...")
    from didpo.core_didpo import compute_didpo_outcome_advantage
    try:
        import torch
    except Exception:
        print("  SKIP: torch not installed")
        return

    tok = CharTokenizer()
    # Partially-overlapping diffs (paper Fig.2 style): shared helper + divergent body.
    shared = "def helper(x):\n    return x + 1\n"
    bodies = [
        shared + "def solve(a):\n    return helper(a)\n",
        shared + "def solve(a):\n    return helper(a)\n",
        shared + "def solve(a):\n    return helper(a) * 2\n",
        shared + "def solve(a):\n    return helper(a) * 2\n",
    ]
    bsz = len(bodies)
    snippets = np.empty(bsz, dtype=object)
    max_L = 0
    rows_ids = []
    for i, body in enumerate(bodies):
        resp = (
            "<think>fix</think>"
            f'<edit path="solution.py"><code>\n{body}</code></edit>'
        )
        ids = encode(resp)
        rows_ids.append(ids)
        max_L = max(max_L, len(ids))
        snippets[i] = parse_response_to_snippets(ids, tok, {})

    # Pad responses to common L for the tensor path.
    L = max_L
    token_level_rewards = torch.zeros(bsz, L)
    response_mask = torch.zeros(bsz, L)
    for i, ids in enumerate(rows_ids):
        response_mask[i, :len(ids)] = 1.0
        token_level_rewards[i, len(ids) - 1] = 1.0 if i < 2 else 0.0
    step_rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    index = np.zeros(bsz, dtype=int)
    traj_index = np.arange(bsz)

    _, _, diag = compute_didpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        step_rewards=step_rewards,
        response_mask=response_mask,
        snippets=snippets,
        index=index,
        traj_index=traj_index,
        sim_thresh=0.9,
        use_alignment=True,
        use_structural_fallback=False,
        overhead_mode="lightweight",
        return_diagnostics=True,
    )
    _check(int(diag.get("n_anchors_selected", 0)) >= 1,
           "alignment selects at least one multi-rollout anchor")
    _check(float(diag.get("mean_group_size", 0.0)) >= 2.0,
           "mean group size >= 2 under identical/shared sub-diffs")
    _check(float(diag.get("align_time_ms", -1.0)) >= 0.0,
           "align_time_ms is recorded")


def test_lcs_nonoverlap_and_exact_hash():
    """Engineering path: shared block groups; no transitive paste; non-overlap."""
    print("[8] LCS exact-hash + non-overlap cut ...")
    from didpo.core_didpo import _build_anchors_for_uid

    shared = ["def helper(x):", "    return x + 1"]
    roots = []
    bodies = [
        shared + ["def solve(a):", "    return helper(a)"],
        shared + ["def solve(a):", "    return helper(a)"],
        shared + ["def solve(a):", "    return helper(a) * 2"],
        shared + ["def solve(a):", "    return helper(a) * 2"],
    ]
    for i, lines in enumerate(bodies):
        src = "\n".join(lines)
        roots.append((i, 0, {
            "changed": True,
            "origin": "root",
            "lines": lines,
            "source": src,
            "edit_type": "none",
            "char_start": 0,
            "char_end": len(src),
            "token_start": 0,
            "token_end": len(src),
        }))
    anchors = _build_anchors_for_uid(
        roots, uid=0,
        sim_thresh=0.9,
        max_span=32,
        s0=8.0,
        count_ref=8.0,
        phi_form="saturating",
        psi_form="saturating",
        max_anchors=16,
        min_block_lines=1,
        max_pairs_per_uid=64,
    )
    _check(len(anchors) >= 1, "at least one multi-member LCS anchor")
    # Helper should form a group of 4.
    sizes = sorted(a.n for a in anchors)
    _check(max(sizes) >= 2, "largest anchor covers >=2 rollouts")
    # Within one root, selected spans must not overlap.
    for row in range(4):
        spans = []
        for a in anchors:
            for o in a.occ:
                if o.row == row:
                    spans.append(o.span)
        spans.sort()
        for (p1, q1), (p2, q2) in zip(spans, spans[1:]):
            _check(q1 <= p2, f"non-overlapping spans on row {row}: {(p1,q1)} vs {(p2,q2)}")


# --------------------------------------------------------------------------- #
def main():
    tests = [
        test_groupability_score,
        test_diff_gating,
        test_token_span_mapping,
        test_cross_rollout_grouping,
        test_group_baselined_advantage,
        test_end_to_end_advantage,
        test_alignment_anchors,
        test_lcs_nonoverlap_and_exact_hash,
    ]
    for t in tests:
        t()
    print("\nALL DIDPO ALGORITHM CHECKS PASSED")


if __name__ == "__main__":
    main()
