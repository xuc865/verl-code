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

"""Offline *contract* test for R2EGymBackend (no Docker / no real r2egym).

We inject a fake ``r2egym.agenthub.environment.env`` module exposing ``EnvArgs``
and ``RepoEnv`` with the **same surface** as the real one (verified against
r2egym/agenthub/environment/env.py):

    EnvArgs(ds=<row dict>)
    env = RepoEnv(args, backend=..., verbose=..., step_timeout=..., reward_timeout=...)
    env.reset()
    out, code = env.runtime.run(bash_cmd, timeout=...)
    reward = env.compute_reward(timeout=...)
    instr = env.get_task_instruction()
    env.close()

The fake records every call so we can assert our backend:
  * passes the *raw* dataset row to ``EnvArgs(ds=...)``,
  * pushes edits into the container via a base64 round-trip (content-safe),
  * delegates bash to ``runtime.run`` and reports non-zero exit codes,
  * grades via ``compute_reward`` mapped to a bool,
  * tears the container down on re-setup / close.

This gives a real local signal that the wiring is correct; the actual Docker
execution is exercised separately by ``smoke_r2egym_real.py`` on a machine that
has r2egym + Docker.
"""

import base64
import importlib.util
import os
import sys
import types

REPO = "/Users/wangxucong/Documents/code-swe"


# --------------------------------------------------------------------------- #
# Fake r2egym executor (records calls; mimics the real API surface)           #
# --------------------------------------------------------------------------- #
class _FakeRuntime:
    def __init__(self, ds):
        self.ds = ds
        self.calls = []           # list of (cmd, timeout)
        self._files = {}          # path -> content (decoded), the "container fs"
        self.closed = False

    def run(self, cmd, timeout=None):
        self.calls.append((cmd, timeout))
        # Emulate just enough: a base64 write `... | base64 -d > path`
        if "base64 -d > " in cmd:
            # extract the b64 payload (printf %s '<b64>') and the target path
            payload = cmd.split("printf %s ", 1)[1].split(" | base64 -d", 1)[0]
            payload = payload.strip().strip("'")
            path = cmd.rsplit("base64 -d > ", 1)[1].strip().strip('"').strip("'")
            self._files[path] = base64.b64decode(payload).decode("utf-8")
            return "", 0
        if cmd.startswith("cat "):
            p = cmd[4:].strip()
            return (self._files.get(p, f"cat: {p}: No such file"),
                    0 if p in self._files else 1)
        if cmd.startswith("FAILCMD"):
            return "boom", 2
        return f"$ {cmd}", 0


class _FakeRepoEnv:
    instances = []  # all envs created (to inspect lifecycle)

    def __init__(self, args, backend="docker", verbose=True,
                 step_timeout=90, reward_timeout=300):
        self.args = args
        self.backend = backend
        self.verbose = verbose
        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.runtime = _FakeRuntime(args.ds)
        self.reset_count = 0
        self.closed = False
        self._reward = float(args.ds.get("_fake_reward", 0.0))
        _FakeRepoEnv.instances.append(self)

    def reset(self):
        self.reset_count += 1
        return "Environment reset"

    def compute_reward(self, timeout=None):
        return self._reward

    def get_task_instruction(self):
        return self.args.ds.get("problem_statement", "(no instruction)")

    def close(self):
        self.closed = True
        self.runtime.closed = True


def _install_fake_r2egym():
    pkg = types.ModuleType("r2egym")
    sub1 = types.ModuleType("r2egym.agenthub")
    sub2 = types.ModuleType("r2egym.agenthub.environment")
    envmod = types.ModuleType("r2egym.agenthub.environment.env")

    class EnvArgs:  # mirrors @dataclass(frozen=True) EnvArgs(ds, repo_path, docker_image)
        def __init__(self, ds, repo_path=None, docker_image=None):
            self.ds = ds
            self.repo_path = repo_path
            self.docker_image = docker_image

    envmod.EnvArgs = EnvArgs
    envmod.RepoEnv = _FakeRepoEnv
    sys.modules["r2egym"] = pkg
    sys.modules["r2egym.agenthub"] = sub1
    sys.modules["r2egym.agenthub.environment"] = sub2
    sys.modules["r2egym.agenthub.environment.env"] = envmod


def _load_envs():
    spec = importlib.util.spec_from_file_location(
        "swebench_envs_r2e",
        os.path.join(REPO, "agent_system/environments/env_package/swebench/envs.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swebench_envs_r2e"] = mod
    spec.loader.exec_module(mod)
    return mod


_install_fake_r2egym()
envs = _load_envs()


TRICKY = "x = \"it's a 'quote' and <<EOF\"\n$(rm -rf /)\nprint('ok')\n"


def _instance(reward=0.0):
    raw = {"instance_id": "r2e__demo-1", "problem_statement": "Fix the bug in foo.",
           "docker_image": "namanjain12/demo:latest", "_fake_reward": reward}
    # mimic what load_instances/_normalize_instance produce
    return envs._normalize_instance(raw)


def part1_make_backend_and_envargs():
    inst = _instance()
    be = envs.make_backend({"backend": "r2e_gym", "verbose": False, "step_timeout": 42})
    assert isinstance(be, envs.R2EGymBackend)
    files = be.setup(inst)
    assert files == {}, "container is the source of truth; map starts empty"
    env = _FakeRepoEnv.instances[-1]
    # EnvArgs got the RAW dataset row, not the normalized wrapper
    assert env.args.ds is inst["_raw"], "must pass the raw dataset row to EnvArgs(ds=...)"
    assert env.args.ds["docker_image"] == "namanjain12/demo:latest"
    assert env.backend == "docker" and env.verbose is False and env.step_timeout == 42
    assert env.reset_count == 1
    assert be.task_instruction(inst) == "Fix the bug in foo."
    print("[part1] setup -> EnvArgs(ds=raw row), RepoEnv(reset), task instruction OK")
    be.close()


def part2_edit_pushes_content_safely():
    inst = _instance()
    be = envs.make_backend({"backend": "r2e_gym"})
    files = be.setup(inst)
    env = _FakeRepoEnv.instances[-1]

    be.apply_edit(inst, files, "src/foo.py", TRICKY)
    # repo-view map kept in sync
    assert files["src/foo.py"] == TRICKY
    # the FAKE container decoded the base64 back to the exact bytes -> no shell
    # injection, no EOF/quote corruption
    assert env.runtime._files["src/foo.py"] == TRICKY, "content must survive verbatim"
    # and reading it back through bash returns the same content
    out, code = env.runtime.run("cat src/foo.py")
    assert code == 0 and out == TRICKY
    print("[part2] apply_edit -> base64 round-trip into container, content verbatim OK")
    be.close()


def part3_run_bash_delegates_and_reports_exit():
    inst = _instance()
    be = envs.make_backend({"backend": "r2e_gym"})
    files = be.setup(inst)
    ok = be.run_bash(inst, files, "ls -la")
    assert ok == "$ ls -la"
    bad = be.run_bash(inst, files, "FAILCMD now")
    assert "boom" in bad and "[exit code 2]" in bad, bad
    print("[part3] run_bash -> runtime.run delegation + non-zero exit surfaced OK")
    be.close()


def part4_evaluate_maps_reward_to_bool():
    be_fail = envs.make_backend({"backend": "r2e_gym"})
    inst_fail = _instance(reward=0.0)
    be_fail.setup(inst_fail)
    assert be_fail.evaluate(inst_fail, {}) is False

    be_pass = envs.make_backend({"backend": "r2e_gym"})
    inst_pass = _instance(reward=1.0)
    be_pass.setup(inst_pass)
    assert be_pass.evaluate(inst_pass, {}) is True
    print("[part4] evaluate -> compute_reward mapped to bool (0->False, 1->True) OK")
    be_fail.close(); be_pass.close()


def part5_lifecycle_close_on_resetup():
    be = envs.make_backend({"backend": "r2e_gym"})
    be.setup(_instance())
    first = _FakeRepoEnv.instances[-1]
    be.setup(_instance())  # re-setup must tear down the previous container
    second = _FakeRepoEnv.instances[-1]
    assert first is not second
    assert first.closed is True, "previous container must be closed on re-setup"
    be.close()
    assert second.closed is True, "close() must tear down the live container"
    print("[part5] lifecycle -> previous container closed on re-setup and on close() OK")


def part6_full_loop_through_single_env():
    """End-to-end through SWEBenchSingleEnv with the fake executor."""
    inst = _instance(reward=1.0)
    env = envs.SWEBenchSingleEnv({"backend": "r2e_gym", "max_turns": 10})
    obs, info = env.reset(inst)
    assert "Fix the bug in foo." in obs, obs   # task instruction came from runtime
    assert info["problem_statement"] == "Fix the bug in foo."

    # bash inspect
    obs, r, done, info = env.step({"type": "bash", "cmd": "ls"})
    assert done is False and "$ ls" in obs
    # edit (goes into the container via apply_edit)
    obs, r, done, info = env.step({"type": "edit", "path": "src/foo.py", "content": TRICKY})
    assert done is False and "Edited src/foo.py" in obs
    # finish -> grade via compute_reward (reward=1.0 -> won)
    obs, r, done, info = env.step({"type": "finish"})
    assert done is True and r == 1.0 and info["won"] is True
    env.close()
    print("[part6] SWEBenchSingleEnv full loop over fake R2E-Gym (reset/bash/edit/finish) OK")


def main():
    part1_make_backend_and_envargs()
    part2_edit_pushes_content_safely()
    part3_run_bash_delegates_and_reports_exit()
    part4_evaluate_maps_reward_to_bool()
    part5_lifecycle_close_on_resetup()
    part6_full_loop_through_single_env()
    print("\nALL R2E-GYM CONTRACT CHECKS PASSED (offline, no Docker)")


if __name__ == "__main__":
    main()
