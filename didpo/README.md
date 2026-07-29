# DIDPO — Diff-in-Diff Policy Optimization

Paper-aligned implementation of **Diff-in-Diff Policy Optimization** for coding
agents (critic-free RLVR with dynamic sub-diff credit).

---

## 1. Method (paper)

DiDPO keeps a GRPO-style episode advantage \(A^E\), then looks *inside* each
code-producing action:

1. **Root diffs as states** with metadata \((u, v, c, q)\): normalized source,
   token span, edit type \(c \in \{\mathrm{add},\mathrm{del},\mathrm{none}\}\),
   task uid.
2. **Dynamic sub-diff anchors**: enumerate multi-scale contiguous segments of
   each changed root diff; match segments across rollouts of the same task with
   the same edit type when \(\mathrm{sim} \ge \eta\); form candidate anchors.
3. **Groupability** \(\mathrm{GS}(c)=(1-e^{-\bar L/s_0})(1-e^{-(n-1)/g_0})\),
   then **greedy facility-location** selection of a cardinality-bounded set
   \(C^\star\) (each occurrence contributes through at most one anchor).
4. Selected anchors cut diffs into grouped sub-diffs; uncovered regions use a
   **structural (AST) fallback**. Unchanged re-emissions are **diff-gated** out.
5. Diff-level group-relative advantage \(A^D\) from step return-to-go; tokens get
   \(\hat A_{i,t}=A^E_i+\lambda\cdot A^D(s_{i,t})\).

---

## 2. Code map

| Paper piece | Code |
|---|---|
| Root diffs + edit type + gating + token spans | `didpo/snippet.py` |
| LCS/exact-hash anchors, greedy \(C^\star\), non-overlap cut, \(A^D\) | `didpo/core_didpo.py` |
| SwanLab scalars + fixed-prompt JSONL evolution | `didpo/group_tracker.py` + `ray_trainer.py` |
| Structural fallback | `extract_structural_fallback` / `build_snippet_groups` |

### Engineered alignment (not sliding-window × union-find)

1. **Exact hash** of full roots / matched blocks  
2. **Pairwise `difflib` opcodes** (budgeted) → equal / high-sim replace blocks  
3. **Rep-centric fuzzy merge** (no transitive paste)  
4. **Greedy facility-location** on GS  
5. **Per-root non-overlapping** interval cut  
6. Timing → `didpo/align_time_ms`

### Overhead vs GRPO

DiDPO adds **CPU-only** work (no extra rollouts / no critic). Default
`overhead_mode=lightweight` keeps that negligible vs vLLM+FSDP:

| mode | collector | alignment | typical extra |
|---|---|---|---|
| `exact_only` | roots only, skip non-edit | hash only | ~0 (ms-level) |
| `lightweight` (default) | roots only | hash + ≤8 LCS pairs | ≪1% of a train step |
| `full` | roots + AST | full LCS budget + AST | still usually ≪ rollout time |

If alignment finds no multi-member groups, advantage **degenerates to GRPO**
(\(A^E\) only).

---

## 3. Logging

Every training step emits (via SwanLab / wandb / console):

- `didpo/n_groups`, `didpo/mean_group_size`, `didpo/median_group_size`
- `didpo/singleton_rate`, `didpo/n_anchors_selected`
- `didpo/alignment_group_frac`, `didpo/fallback_group_frac`
- `didpo/changed_snippet_ratio`, `didpo/snippet_token_frac`
- `didpo/group_size_hist/*`

Fixed prompts (first `track_prompt_n` uids, or `track_prompt_uids`):

- scalars `didpo/prompt/<id>/n_groups`, `.../mean_group_size`, …
- JSONL trail at `{group_dump_dir}/didpo_prompt_groups.jsonl`
- per-uid latest snapshot `prompt_<id>_latest.json`

---

## 4. Config (`algorithm.didpo`)

See `verl/trainer/config/ppo_trainer.yaml`. Important knobs:

| key | default | meaning |
|---|---|---|
| `snippet_advantage_w` | `1.0` | \(\lambda\) |
| `sim_thresh` | `0.8` | \(\eta\) |
| `phi_s0` / `psi_count_ref` | `8.0` | \(s_0\), \(g_0\) |
| `use_alignment` | `True` | paper core path |
| `use_structural_fallback` | `True` | AST fallback |
| `track_prompt_n` | `3` | auto-track N prompts |
| `group_dump_dir` | `…/didpo_groups` | JSONL dump root |

---

## 5. Run / test

```bash
bash examples/didpo_trainer/run_swebench.sh
python3 didpo/tests/test_didpo_algorithm.py
```
