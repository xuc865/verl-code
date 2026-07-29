# Environment Setup

DIDPO targets **coding benchmarks only**. The legacy verl-agent environments
(ALFWorld / WebShop / Sokoban / Gym Cards / AppWorld / Search) have been
**deprecated and removed**. The only supported environment is the SWE-bench
coding environment.

## SWE-bench

The SWE-bench environment lives in
`agent_system/environments/env_package/swebench/` and is wired through
`SWEBenchEnvironmentManager` in `agent_system/environments/env_manager.py`.

### Action space

Each step the policy emits one tool call wrapped in tags:

- `<think>...</think>` — reasoning (required for an action to be marked valid).
- `<execute_bash>cmd</execute_bash>` — run a shell command (inspect the repo).
- `<edit path="...">` `<code>...</code>` `</edit>` — overwrite a file.
- `<finish>...</finish>` — submit the patch for grading.

See `agent_system/environments/prompts/swebench.py` for the full prompt
templates and `env_package/swebench/projection.py` for parsing.

### Execution backends

`env_package/swebench/envs.py` exposes a pluggable backend via `make_backend(cfg)`:

- `local_stub` (default) — in-memory, Docker-free backend with transparent
  grading. Lets you smoke-test the full RL pipeline without any external
  dependency. Used by `examples/didpo_trainer/run_swebench.sh`.
- `docker` / `r2egym` — stubs for real SWE-bench harness integration
  (`DockerBackend` / `R2EGymBackend`), not yet implemented.

### Configuration

Configured under `env.swebench.*` in `verl/trainer/config/ppo_trainer.yaml`
(`dataset_name`, `split`, `subset_size`, `max_turns`, `backend`,
`image_prefix`, `timeout`, `reward_success`, `reward_fail`).

### Quick start

```bash
bash examples/didpo_trainer/run_swebench.sh
```

This runs DIDPO (`algorithm.adv_estimator=didpo`) on the synthetic
`local_stub` task set — no Docker / dataset download required.
