# Environment Setup

This repo trains **multi-turn coding agents**. Legacy verl-agent environments
(ALFWorld / WebShop / Sokoban / Gym Cards / AppWorld / Search) have been
**deprecated and removed**. The supported path is the coding environment under
`env_package/swebench/` (module name is historical; the default workloads are
CodeRL / APPS-style coding tasks).

## Coding environment

Wired through `SWEBenchEnvironmentManager` in
`agent_system/environments/env_manager.py`.

### Action space

Each step the policy emits one tool call wrapped in tags:

- `<think>...</think>` — reasoning (required for an action to be marked valid).
- `<execute_bash>cmd</execute_bash>` — run a shell command (inspect the repo).
- `<edit path="...">` `<code>...</code>` `</edit>` — overwrite a file.
- `<finish>...</finish>` — submit for grading.

See `agent_system/environments/prompts/swebench.py` for the full prompt
templates and `env_package/swebench/projection.py` for parsing.

### Execution backends

`env_package/swebench/envs.py` exposes a pluggable backend via `make_backend(cfg)`:

- `local_stub` (default) — in-memory, Docker-free backend with transparent
  grading. Smoke-tests the full RL pipeline without external deps.
- `docker` / `r2egym` — optional harness backends for repo-level coding suites.

### Configuration

Configured under `env.swebench.*` in `verl/trainer/config/ppo_trainer.yaml`
(`dataset_name`, `split`, `subset_size`, `max_turns`, `backend`,
`image_prefix`, `timeout`, `reward_success`, `reward_fail`).

### Quick start

```bash
bash examples/didpo_trainer/run_swebench.sh
# also: examples/grpo_trainer/run_swebench.sh
#       examples/gigpo_trainer/run_swebench.sh
```

Default `benchmark=local` needs no Docker / dataset download.
