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

"""Deterministic verification of the LocalStub *self-repair* test bed.

This exercises the full single-turn-to-multi-turn loop with the real,
torch-free production code (``envs.py`` + ``projection.py``):

    run pytest (see real failing traceback) -> edit source -> run pytest
    (all green -> auto done) -> sparse outcome reward == 1.0

It proves the local backend now *executes* tests rather than keyword-matching,
which is the prerequisite for training DIDPO on a genuine repair signal.
"""

import importlib.util
import os
import sys

REPO = "/Users/wangxucong/Documents/code-swe"


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


envs = _load("swebench_envs", "agent_system/environments/env_package/swebench/envs.py")
proj = _load("swebench_proj", "agent_system/environments/env_package/swebench/projection.py")


CALC_FIX = (
    "def add(a, b):\n"
    "    return a + b\n\n"
    "def mul(a, b):\n"
    "    return a * b\n"
)


def _calc_instance():
    for inst in envs._synthetic_instances():
        if inst["instance_id"].startswith("synthetic__calc-"):
            return inst
    raise AssertionError("no calc synthetic instance found")


def part1_instances_carry_real_tests():
    inst = _calc_instance()
    files = inst["initial_files"]
    assert "calc.py" in files and "test_calc.py" in files, files.keys()
    assert "def test_add" in files["test_calc.py"]
    assert envs._parse_test_names(inst["FAIL_TO_PASS"]) == ["test_add", "test_mul"]
    print("[part1] synthetic instance ships runnable test_calc.py + FAIL_TO_PASS OK")


def part2_buggy_code_really_fails():
    backend = envs.LocalStubBackend()
    inst = _calc_instance()
    files = backend.setup(inst)

    passed, report = backend.run_tests(inst, files)
    assert passed is False, report
    assert "FAILED test_add" in report and "FAILED test_mul" in report, report
    assert "AssertionError" in report, "traceback must surface the assertion for repair"
    assert backend.evaluate(inst, files) is False
    print("[part2] buggy code -> real test failures + traceback (evaluate=False) OK")
    print("        report tail:\n          " + report.strip().splitlines()[-1])


def part3_fixed_code_really_passes():
    backend = envs.LocalStubBackend()
    inst = _calc_instance()
    files = backend.setup(inst)
    files["calc.py"] = CALC_FIX

    passed, report = backend.run_tests(inst, files)
    assert passed is True, report
    assert "FAILED" not in report, report
    assert backend.evaluate(inst, files) is True
    print("[part3] fixed code -> all tests pass (evaluate=True) OK")


def part4_full_self_repair_loop_via_projection():
    """Drive the env with *text* actions, parsed by the real projection."""
    inst = _calc_instance()
    env = envs.SWEBenchSingleEnv({"backend": "local_stub", "max_turns": 10})
    obs, info = env.reset(inst)
    assert "calc.py" in obs and "Problem statement" in obs

    # Turn 1: run the suite and observe the failure.
    txt = "<think>let me run the tests first</think><execute_bash>pytest</execute_bash>"
    actions, valids = proj.swebench_projection([txt])
    assert valids == [1] and actions[0]["type"] == "bash"
    obs, reward, done, info = env.step(actions[0])
    assert done is False and reward == 0.0
    assert "FAILED test_add" in obs, obs
    print("[part4.1] pytest turn -> model observes FAILED tests, not done OK")

    # Turn 2: edit the source with the fix.
    edit_txt = (
        "<think>add should sum, mul should multiply</think>"
        '<edit path="calc.py"><code>\n' + CALC_FIX + "</code></edit>"
    )
    actions, valids = proj.swebench_projection([edit_txt])
    assert valids == [1] and actions[0]["type"] == "edit" and actions[0]["path"] == "calc.py"
    obs, reward, done, info = env.step(actions[0])
    assert done is False and "Edited calc.py" in obs
    print("[part4.2] edit turn -> source patched, not done OK")

    # Turn 3: re-run -> all green -> auto done with outcome reward 1.0.
    actions, _ = proj.swebench_projection(["<think>re-run</think><execute_bash>pytest</execute_bash>"])
    obs, reward, done, info = env.step(actions[0])
    assert done is True, "all-pass should auto-finish"
    assert reward == 1.0, reward
    assert info["won"] is True
    assert "0 failed" in obs, obs
    print("[part4.3] re-run turn -> all pass -> auto done, reward=1.0, won=True OK")


def part5_fresh_reimport_across_edits():
    """A second edit in the same process must reflect the *latest* file map,
    proving the runner re-imports fresh (no stale module cache leakage)."""
    backend = envs.LocalStubBackend()
    inst = _calc_instance()
    files = backend.setup(inst)
    assert backend.run_tests(inst, files)[0] is False
    files["calc.py"] = CALC_FIX
    assert backend.run_tests(inst, files)[0] is True
    # break it again -> must fail again (no cached pass)
    files["calc.py"] = "def add(a,b):\n    return 0\ndef mul(a,b):\n    return 0\n"
    assert backend.run_tests(inst, files)[0] is False
    print("[part5] re-running after edits re-imports fresh sources OK")


def main():
    part1_instances_carry_real_tests()
    part2_buggy_code_really_fails()
    part3_fixed_code_really_passes()
    part4_full_self_repair_loop_via_projection()
    part5_fresh_reimport_across_edits()
    print("\nALL SELF-REPAIR ENV CHECKS PASSED")


if __name__ == "__main__":
    main()
