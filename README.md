<h1 align="center">code-swe</h1>

<h3 align="center">
<b>Reinforcement Learning for SWE-bench Coding Agents</b>
<br>
<b>with DIDPO — Diff-in-Diff Policy Optimization</b>
</h3>

<p align="center">
  <i>A SWE-bench–focused RL training stack built on top of
  <a href="https://github.com/langfengQ/verl-agent">verl-agent</a> /
  <a href="https://github.com/volcengine/verl">veRL</a>.</i>
</p>

---

`code-swe` is a reinforcement-learning training stack for **software-engineering
agents** — LLMs that read a real repository, run shell commands, edit files, and
submit a patch that has to make a hidden test set go from red to green
(the SWE-bench task formulation).

It is built on top of [verl-agent](https://github.com/langfengQ/verl-agent)
(itself an extension of [veRL](https://github.com/volcengine/verl)), but it is
**not** the general multi-environment agent playground that verl-agent ships as.
We have stripped the project down to a single domain — **SWE-bench coding** — and
added a new credit-assignment algorithm, **DIDPO**, designed specifically for the
structure of code-editing trajectories.

> If you are looking for the upstream multi-environment framework
> (ALFWorld / WebShop / Sokoban / GiGPO, etc.), see the original
> [verl-agent](https://github.com/langfengQ/verl-agent) repository. Those
> environments are **deprecated and removed** here — see
> [Scope](#scope) below.

---

# Table of Contents

- [What this project is](#what-this-project-is)
- [Scope](#scope)
- [DIDPO in one paragraph](#didpo-in-one-paragraph)
- [RL algorithms](#rl-algorithms)
- [The SWE-bench environment](#the-swe-bench-environment)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Testing](#testing)
- [Running on remote GPUs](#running-on-remote-gpus)
- [Status & roadmap](#status--roadmap)
- [Acknowledgement](#acknowledgement)

---

# What this project is

The goal is to train a code-editing agent with RL on SWE-bench-style tasks and to
study **finer-grained credit assignment** than episode-level GRPO can offer.

A single agent action in this setting emits a **chunk of code** (a file edit). A
plain GRPO/GiGPO update assigns one scalar advantage to that whole step. The core
idea of this project — **DIDPO** — is to split that chunk into functional
*snippets*, group functionally-equivalent snippets across the rollouts of the
same SWE-bench instance, and hand each snippet a group-baselined advantage. The
result is a critic-free, no-extra-rollout algorithm that does sub-step credit
assignment *inside a single generation*.

The end-to-end pipeline is:

```
SWE-bench instance ──► agent rollout (bash / edit / finish)
                              │  N rollouts per instance (group)
                              ▼
                       sparse ORM reward (tests pass ⇒ 1.0)
                              │
                              ▼
        advantage estimation: GRPO | GSPO | DAPO | DIDPO
                              │
                              ▼
                    FSDP policy update (vLLM rollout)
```

---

# Scope

This fork is intentionally narrowed. To keep the reframing honest, here is exactly
what is in and out of scope **for the current round of work**:

| Item | Status |
|---|---|
| **SWE-bench coding environment** | ✅ The only supported environment |
| Legacy verl-agent environments (ALFWorld / WebShop / Sokoban / Gym Cards / AppWorld / Search) | ❌ Deprecated & removed (not wired in `env_manager.py`) |
| **DIDPO** (snippet-level grouped advantage) | ✅ Implemented — this project's core contribution |
| **GRPO / GSPO / DAPO** baselines on SWE-bench | ✅ Provided as `run_swebench.sh` scripts |
| **GRPO++** | ⏸️ Out of scope this round. The script (`examples/grpo_trainer/run_swebench_pp.sh`) is **kept in the repo** but not part of the current effort. |
| **DAPO dynamic sampling** (`filter_groups`) | ⏸️ Declared in the YAML but **not wired** into the training loop in this fork — currently a no-op. Not being connected this round. |

---

# DIDPO in one paragraph

DIDPO (Diff-in-Diff Policy Optimization) treats code diffs as divisible states.
It retains a GRPO-style episode advantage \(A^E\), then aligns *sub-diffs* across
rollouts of the same task (same edit type, similarity \(\ge\eta\)) to form
dynamic anchors. A groupability score
\(\mathrm{GS}=(1-e^{-\bar L/s_0})(1-e^{-(n-1)/g_0})\) drives greedy
facility-location selection of anchors; selected anchors cut diffs into grouped
segments (with AST structural fallback when alignment is sparse). Unchanged
re-emissions are diff-gated out. Each grouped sub-diff receives a local
return-to-go baseline \(A^D\), and tokens are supervised with
\(\hat A_{i,t}=A^E_i+\lambda\cdot A^D(s_{i,t})\).

Full theory ↔ code mapping lives in **[`didpo/README.md`](./didpo/README.md)**.

---

# RL algorithms

All algorithms share the same SWE-bench rollout collector and reward; they differ
only in the advantage estimator (`algorithm.adv_estimator`):

| Algorithm | `adv_estimator` | Notes |
|---|---|---|
| **DIDPO** | `didpo` | This project's algorithm. Snippet-level grouped advantage; `λ=0` degenerates to GRPO. Trainer side: `didpo/core_didpo.py`; collector side: `didpo/snippet.py`. |
| **GRPO**  | `grpo`  | Critic-free, episode-level group baseline. |
| **GSPO**  | `gspo`  | Sequence-level importance-weighted variant. |
| **DAPO**  | `dapo`  | GRPO + clip-higher (dynamic sampling **not** wired — see [Scope](#scope)). |

---

# The SWE-bench environment

The only supported environment lives in
`agent_system/environments/env_package/swebench/` and is wired through
`SWEBenchEnvironmentManager` in `agent_system/environments/env_manager.py`. See
[`agent_system/environments/README.md`](./agent_system/environments/README.md)
for the full reference.

**Action space** (one tool call per step, tag-wrapped):

- `<think>...</think>` — reasoning (required for an action to count as valid).
- `<execute_bash>cmd</execute_bash>` — run a shell command to inspect the repo.
- `<edit path="...">``<code>...</code>``</edit>` — overwrite a file.
- `<finish>...</finish>` — submit the patch for grading.

**Reward**: a sparse ORM — `1.0` iff the resolved test set passes, else `0.0`.

**Execution backends** (`env.swebench.backend`):

- `local_stub` *(default)* — in-memory, **Docker-free** backend with transparent
  grading. Lets you exercise the entire RL loop (including DIDPO advantages)
  with **no dataset download and no Docker**. This is what the example scripts
  use out of the box.
- `docker` / `r2e_gym` — real SWE-bench harness integration. `r2e_gym` is the
  path to use for real training; it expects the R2E-Gym image-backed datasets
  (e.g. `R2E-Gym/SWE-Bench-Lite`), **not** the `princeton-nlp` originals
  (those rows lack the `docker_image` metadata needed to spin up containers).

---

# Repository layout

```
code-swe/
├── didpo/                       # DIDPO algorithm (this project's core)
│   ├── core_didpo.py            #   trainer-side: grouping, GS, advantage fill
│   ├── snippet.py               #   collector-side: response → snippets
│   ├── README.md                #   theory ↔ code map
│   └── tests/                   #   offline unit tests + real-env smoke test
├── agent_system/
│   ├── environments/
│   │   ├── env_package/swebench/   # gym-style, Ray-vectorized SWE-bench env
│   │   ├── env_manager.py          # SWEBenchEnvironmentManager
│   │   └── prompts/swebench.py     # prompt templates
│   ├── multi_turn_rollout/      # rollout loop (attaches didpo_snippets)
│   └── memory/                  # history/memory module
├── examples/
│   ├── didpo_trainer/run_swebench.sh
│   ├── grpo_trainer/run_swebench.sh   (+ run_swebench_pp.sh, GRPO++, parked)
│   ├── gspo_trainer/run_swebench.sh
│   └── dapo_trainer/run_swebench.sh
├── verl/                        # vendored veRL framework (trainer, FSDP, vLLM)
└── REMOTE_SETUP_CHECKLIST.md    # detailed remote-GPU bring-up checklist
```

---

# Installation

```bash
conda create -n code-swe python==3.12 -y
conda activate code-swe

pip install -e .
pip install -r requirements.txt

# Two deps that requirements.txt does not pin but the real backend needs:
pip install vllm==0.8.4        # (commented out in requirements; rollout engine)
pip install r2egym             # only imported by the real (r2e_gym) backend
```

Default model: **`Qwen/Qwen2.5-Coder-7B-Instruct`**.

> The `local_stub` backend needs **none** of the SWE-bench data/Docker machinery,
> so you can validate the pipeline before installing any of the heavy pieces.

---

# Quick start

Run DIDPO on the Docker-free synthetic task set — no dataset, no Docker required:

```bash
bash examples/didpo_trainer/run_swebench.sh
```

The baselines run the same way:

```bash
bash examples/grpo_trainer/run_swebench.sh
bash examples/gspo_trainer/run_swebench.sh
bash examples/dapo_trainer/run_swebench.sh
```

Switch to a real backend via the SWE-bench config, e.g.:

```bash
bash examples/didpo_trainer/run_swebench.sh -- env.swebench.benchmark=swe_bench_lite
```

---

# Testing

Tests live in `didpo/tests/`. They are layered from "no dependencies" to "needs
Docker", forming a small testing pyramid:

| Level | What | Dependency |
|---|---|---|
| **L1** offline unit tests | `test_didpo_algorithm.py` (DIDPO math), `test_selfrepair_env.py` (local_stub reward), `test_r2egym_contract.py` (backend API contract via fake module), `test_phase3_benchmark.py` (benchmark preset resolution) | none (CPU) |
| **L2** local smoke training | `bash examples/didpo_trainer/run_swebench.sh` — full train loop, zero download | GPU |
| **L3** real-instance smoke | `DIDPO_R2EGYM_SMOKE=1 python3 didpo/tests/smoke_r2egym_real.py` | GPU + Docker + r2egym |
| **L4** full training | the `run_swebench.sh` scripts on `r2e_gym_subset` | GPU + Docker |

Run the L1 suite (CPU, seconds):

```bash
for t in test_didpo_algorithm test_phase3_benchmark test_r2egym_contract test_selfrepair_env; do
  python3 didpo/tests/$t.py
done
```

---

# Running on remote GPUs

A detailed, source-verified bring-up checklist for a fresh GPU box (hardware,
deps, dataset/backend mapping, Docker/r2egym wiring, validation ladder, and the
info to report back) lives in
[`REMOTE_SETUP_CHECKLIST.md`](./REMOTE_SETUP_CHECKLIST.md).

---

# Status & roadmap

**Done**

- DIDPO algorithm (collector + trainer sides) wired into the verl-agent pipeline.
- SWE-bench environment with `local_stub` (Docker-free) backend.
- GRPO / GSPO / DAPO baselines as runnable SWE-bench scripts.
- L1 offline test suite (algorithm, reward signal, backend contract, presets).

**To do / open**

- Real-backend training: finish `docker` / `r2e_gym` end-to-end on a GPU box.
- L3/L4 validation on `swe_bench_lite` → `r2e_gym_subset`.
- (Parked, see [Scope](#scope)) GRPO++; DAPO dynamic sampling (`filter_groups`).
- DIDPO limitations tracked in [`didpo/README.md`](./didpo/README.md) §7
  (reward confounding, function-level smearing, syntactic-similarity grouping).

---

# Acknowledgement

`code-swe` is built on [verl-agent](https://github.com/langfengQ/verl-agent) and
[veRL](https://github.com/volcengine/verl). The SWE-bench task formulation and the
R2E-Gym image-backed datasets come from the SWE-bench and
[R2E-Gym](https://github.com/R2E-Gym) projects. We thank the authors and
contributors of those projects for their work.
