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

"""Offline test for Phase 3: benchmark presets (one-line dataset switching).

Verifies, with no Docker / no HuggingFace download, that:
  * each preset resolves to the right dataset_name/split/backend,
  * a preset is authoritative over the raw fields (the whole point of switching),
  * 'custom' / absent leaves raw fields untouched (backward compatible),
  * unknown benchmark names fail loudly,
  * resolve_benchmark never mutates its input,
  * make_backend dispatches to the correct backend class per preset,
  * load_instances(benchmark=local) yields the synthetic self-repair set,
  * the shipped ppo_trainer.yaml default (benchmark=local) stays consistent.
"""

import importlib.util
import os
import re
import sys

REPO = "/Users/wangxucong/Documents/code-swe"


def _load_envs():
    spec = importlib.util.spec_from_file_location(
        "swebench_envs_p3",
        os.path.join(REPO, "agent_system/environments/env_package/swebench/envs.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swebench_envs_p3"] = mod
    spec.loader.exec_module(mod)
    return mod


envs = _load_envs()


_EXPECTED = {
    "local":              ("",                          "test",  "local_stub"),
    "swe_bench_verified": ("R2E-Gym/SWE-Bench-Verified", "test",  "r2e_gym"),
    "swe_bench_lite":     ("R2E-Gym/SWE-Bench-Lite",     "test",  "r2e_gym"),
    "r2e_gym_subset":     ("R2E-Gym/R2E-Gym-Subset",     "train", "r2e_gym"),
    "r2e_gym_lite":       ("R2E-Gym/R2E-Gym-Lite",       "train", "r2e_gym"),
}


def part1_presets_resolve():
    for key, (name, split, backend) in _EXPECTED.items():
        out = envs.resolve_benchmark({"benchmark": key})
        assert out["dataset_name"] == name, (key, out)
        assert out["split"] == split, (key, out)
        assert out["backend"] == backend, (key, out)
    # case / whitespace insensitive
    out = envs.resolve_benchmark({"benchmark": "  SWE_Bench_Verified "})
    assert out["dataset_name"] == "R2E-Gym/SWE-Bench-Verified"
    print(f"[part1] all {len(_EXPECTED)} presets resolve to correct dataset/split/backend OK")


def part2_preset_overrides_raw_fields():
    # User left stale raw fields (e.g. yaml defaults) but picked a preset:
    # the preset must win for dataset_name/split/backend.
    cfg = {"benchmark": "swe_bench_verified",
           "dataset_name": "princeton-nlp/SWE-bench_Verified",  # WRONG for r2e_gym
           "split": "validation", "backend": "local_stub",
           "subset_size": 5, "reward_timeout": 123}
    out = envs.resolve_benchmark(cfg)
    assert out["dataset_name"] == "R2E-Gym/SWE-Bench-Verified"
    assert out["split"] == "test"
    assert out["backend"] == "r2e_gym"
    # non-preset keys are preserved untouched
    assert out["subset_size"] == 5 and out["reward_timeout"] == 123
    print("[part2] preset is authoritative over stale raw fields; other keys preserved OK")


def part3_custom_and_absent_passthrough():
    raw = {"dataset_name": "my/dataset", "split": "dev", "backend": "docker"}
    # absent benchmark -> verbatim
    assert envs.resolve_benchmark(dict(raw)) == raw
    # explicit custom -> verbatim
    cfg = dict(raw); cfg["benchmark"] = "custom"
    out = envs.resolve_benchmark(cfg)
    assert out["dataset_name"] == "my/dataset" and out["backend"] == "docker"
    # "" / None also treated as custom
    for sentinel in ("", None):
        cfg = dict(raw); cfg["benchmark"] = sentinel
        assert envs.resolve_benchmark(cfg)["dataset_name"] == "my/dataset"
    print("[part3] custom / absent benchmark passes raw fields through unchanged OK")


def part4_unknown_benchmark_raises():
    try:
        envs.resolve_benchmark({"benchmark": "totally_made_up"})
    except ValueError as e:
        assert "Unknown benchmark" in str(e) and "swe_bench_verified" in str(e)
        print("[part4] unknown benchmark raises a helpful ValueError OK")
        return
    raise AssertionError("expected ValueError for unknown benchmark")


def part5_no_mutation_of_input():
    src = {"benchmark": "r2e_gym_lite", "subset_size": 7}
    snapshot = dict(src)
    _ = envs.resolve_benchmark(src)
    assert src == snapshot, "resolve_benchmark must not mutate its argument"
    print("[part5] resolve_benchmark does not mutate the input dict OK")


def part6_make_backend_dispatch():
    assert isinstance(envs.make_backend({"benchmark": "local"}), envs.LocalStubBackend)
    be = envs.make_backend({"benchmark": "swe_bench_verified"})
    assert isinstance(be, envs.R2EGymBackend)
    # R2EGymBackend must construct without importing r2egym (import is lazy in setup)
    assert be._backend == "docker"
    # raw custom path still works
    assert isinstance(envs.make_backend({"backend": "local_stub"}), envs.LocalStubBackend)
    print("[part6] make_backend dispatches local->LocalStub, swe_bench_verified->R2EGym OK")


def part7_load_instances_local_synthetic():
    insts = envs.load_instances({"benchmark": "local"})
    assert len(insts) > 0
    # synthetic self-repair instances carry runnable test files + FAIL_TO_PASS
    inst = insts[0]
    assert any(p.startswith("test_") for p in inst["initial_files"]), inst["initial_files"].keys()
    assert inst["FAIL_TO_PASS"]
    # subset_size still applies after preset resolution
    assert len(envs.load_instances({"benchmark": "local", "subset_size": 3})) == 3
    print(f"[part7] load_instances(benchmark=local) -> {len(insts)} synthetic instances OK")


def part8_yaml_default_consistent():
    yaml_path = os.path.join(REPO, "verl/trainer/config/ppo_trainer.yaml")
    with open(yaml_path) as fh:
        text = fh.read()
    m = re.search(r"^\s*benchmark:\s*\"?([\w]+)\"?", text, re.MULTILINE)
    assert m, "no benchmark key found in ppo_trainer.yaml swebench block"
    default = m.group(1)
    assert default == "local", f"expected default benchmark=local, got {default}"
    # the default must be a real preset
    assert default in _EXPECTED
    print(f"[part8] ppo_trainer.yaml default benchmark={default} is a valid preset OK")


def main():
    part1_presets_resolve()
    part2_preset_overrides_raw_fields()
    part3_custom_and_absent_passthrough()
    part4_unknown_benchmark_raises()
    part5_no_mutation_of_input()
    part6_make_backend_dispatch()
    part7_load_instances_local_synthetic()
    part8_yaml_default_consistent()
    print("\nALL PHASE 3 BENCHMARK-PRESET CHECKS PASSED (offline, no Docker)")


if __name__ == "__main__":
    main()
