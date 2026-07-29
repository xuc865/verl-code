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

"""Dataset adapters: turn external coding benchmarks into **multi-turn**
self-repair instances for the CPU bed.

Each adapter exposes ``to_selfrepair_instances(rows) -> List[dict]``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def make_assert_instance(
    instance_id: str,
    problem_statement: str,
    solution_stub: str,
    test_code: str,
    fail_to_pass: List[str],
    repo: Optional[str] = None,
    raw: Optional[Dict[str, Any]] = None,
    task_mode: str = "repair",
) -> Dict[str, Any]:
    """Build an assert-style multi-turn instance (graded by ``local_pytest``)."""
    return {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "repo": repo or instance_id.split("__", 1)[0],
        "base_commit": "0000000000000000000000000000000000000000",
        "test_patch": "",
        "patch": "",
        "FAIL_TO_PASS": json.dumps(list(fail_to_pass)),
        "PASS_TO_PASS": "[]",
        "initial_files": {
            "solution.py": solution_stub,
            "test_solution.py": test_code,
        },
        "target_file": "solution.py",
        "solution_keywords": [],
        "task_mode": task_mode,
        "_raw": dict(raw or {}),
    }


def make_io_instance(
    instance_id: str,
    problem_statement: str,
    solution_stub: str,
    io_tests: Dict[str, Any],
    repo: Optional[str] = None,
    raw: Optional[Dict[str, Any]] = None,
    task_mode: str = "repair",
) -> Dict[str, Any]:
    """Build a stdin/stdout multi-turn instance (graded by ``selfrepair_io``)."""
    return {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "repo": repo or instance_id.split("__", 1)[0],
        "base_commit": "0000000000000000000000000000000000000000",
        "test_patch": "",
        "patch": "",
        "FAIL_TO_PASS": json.dumps(["io"]),
        "PASS_TO_PASS": "[]",
        "initial_files": {"solution.py": solution_stub},
        "target_file": "solution.py",
        "solution_keywords": [],
        "io_tests": dict(io_tests or {}),
        "task_mode": task_mode,
        "_raw": dict(raw or {}),
    }


def get_adapter(name: str):
    name = (name or "").strip().lower()
    if name in ("humaneval", "humaneval_plus", "humanevalplus"):
        from .humaneval import to_selfrepair_instances
        return to_selfrepair_instances
    if name in ("mbpp", "mbpp_plus", "mbppplus"):
        from .mbpp import to_selfrepair_instances
        return to_selfrepair_instances
    if name == "apps":
        from .apps import to_selfrepair_instances
        return to_selfrepair_instances
    if name in ("livecodebench", "lcb"):
        from .livecodebench import to_selfrepair_instances
        return to_selfrepair_instances
    if name == "usaco":
        from .usaco import to_selfrepair_instances
        return to_selfrepair_instances
    if name == "ojbench":
        from .ojbench import to_selfrepair_instances
        return to_selfrepair_instances
    if name in ("icpc", "icpc_eval"):
        from .icpc import to_selfrepair_instances
        return to_selfrepair_instances
    if name in ("leetcode", "leetcodedataset", "lcd"):
        from .leetcode import to_selfrepair_instances
        return to_selfrepair_instances
    raise ValueError(
        f"Unknown adapter '{name}'. Known: humaneval, mbpp, apps, livecodebench, "
        "usaco, ojbench, icpc, leetcode.")


KNOWN_ADAPTERS = (
    "humaneval", "mbpp", "apps", "livecodebench", "usaco", "ojbench", "icpc", "leetcode",
)
