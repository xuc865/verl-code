#!/usr/bin/env python3
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

"""Minimal REAL-instance smoke test for the R2E-Gym backend.

Run this on a machine that has **r2egym installed and Docker running**. It does
NOT require GPUs or model weights -- it only exercises the executor plumbing our
``R2EGymBackend`` depends on, end to end, on one real instance:

    1. load a dataset row,
    2. open the Docker-backed RepoEnv via our SWEBenchSingleEnv,
    3. run a real bash command in the container (`ls`, `python --version`),
    4. fetch the task instruction,
    5. call compute_reward (the unit-test grader) on the *unmodified* repo,
    6. tear the container down.

A healthy run prints a real directory listing, a non-empty task instruction,
and a numeric reward (expected ~0 on the unmodified buggy repo). Any Docker /
image-pull / API-mismatch problem surfaces here, before wiring into training.

Usage
-----
    # defaults: R2E-Gym/R2E-Gym-Lite, split=train, index=0
    python didpo/tests/smoke_r2egym_real.py

    # pick a dataset / split / index
    R2E_DATASET=R2E-Gym/R2E-Gym-Subset R2E_SPLIT=train R2E_INDEX=5 \
        python didpo/tests/smoke_r2egym_real.py

Datasets: R2E-Gym/R2E-Gym-Lite | R2E-Gym/R2E-Gym-Subset | R2E-Gym/R2E-Gym-Full |
          R2E-Gym/SWE-Bench-Verified | R2E-Gym/SWE-Bench-Lite
"""

import importlib.util
import os
import sys

REPO = os.environ.get("DIDPO_REPO", "/Users/wangxucong/Documents/code-swe")
DATASET = os.environ.get("R2E_DATASET", "R2E-Gym/R2E-Gym-Lite")
SPLIT = os.environ.get("R2E_SPLIT", "train")
INDEX = int(os.environ.get("R2E_INDEX", "0"))


def _load_envs():
    spec = importlib.util.spec_from_file_location(
        "swebench_envs_smoke",
        os.path.join(REPO, "agent_system/environments/env_package/swebench/envs.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swebench_envs_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    try:
        import r2egym  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] r2egym not importable: {e}\n"
              "Install r2egym and ensure Docker is running, then re-run.")
        return 2

    from datasets import load_dataset

    envs = _load_envs()

    print(f"[1/6] loading {DATASET} [{SPLIT}] ...")
    ds = load_dataset(DATASET)
    row = dict(ds[SPLIT][INDEX])
    inst = envs._normalize_instance(row)
    print(f"      instance_id = {inst['instance_id']}")

    print("[2/6] opening Docker-backed RepoEnv (may pull an image; can take a while) ...")
    env = envs.SWEBenchSingleEnv({"backend": "r2e_gym", "verbose": True, "max_turns": 5})
    obs, info = env.reset(inst)
    print("      reset OK")

    print("[3/6] task instruction (first 400 chars):")
    instr = info.get("problem_statement", "")
    assert instr.strip(), "empty task instruction -- get_task_instruction() failed"
    print("      " + instr.strip()[:400].replace("\n", "\n      "))

    print("[4/6] running real bash in the container ...")
    o1, r1, d1, _ = env.step({"type": "bash", "cmd": "ls -la"})
    print("      $ ls -la ->\n      " + o1.strip()[:600].replace("\n", "\n      "))
    o2, _, _, _ = env.step({"type": "bash", "cmd": "python --version 2>&1 || python3 --version"})
    print("      python version -> " + o2.strip()[:120])

    print("[5/6] compute_reward on the UNMODIFIED repo (expected ~0 / unresolved) ...")
    won = bool(env.backend.evaluate(inst, env.files))
    print(f"      evaluate(unmodified) -> won={won}  (False/0 is expected here)")

    print("[6/6] closing container ...")
    env.close()
    print("\nR2E-GYM REAL SMOKE TEST OK -- executor plumbing works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
