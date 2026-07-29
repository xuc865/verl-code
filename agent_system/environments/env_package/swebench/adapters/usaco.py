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

"""USACO benchmark -> self-repair instances (stdin/stdout, ``selfrepair_io``).

Dataset: ``dapumptu/usaco_benchmark`` (307 problems with analyses).

Each row carries:
  * ``problem_id``     -- unique id
  * ``problem_level``  -- bronze / silver / gold / platinum
  * ``description``    -- full problem statement
  * ``input`` / ``output`` -- dicts of numbered hidden test cases
  * ``runtime_limit`` / ``memory_limit`` -- metadata (not enforced in CPU bed)
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import make_io_instance
from .io_utils import competition_problem_text, dict_cases_to_io_lists, stdin_stub


def to_selfrepair_instances(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        inputs, outputs = dict_cases_to_io_lists(row.get("input"), row.get("output"))
        if not inputs or not outputs or len(inputs) != len(outputs):
            continue
        pid = str(row.get("problem_id", f"idx{len(out)}")).replace("/", "_").replace(" ", "_")
        level = str(row.get("problem_level", "unknown")).strip().lower()
        raw = dict(row)
        raw.setdefault("difficulty", level)
        out.append(make_io_instance(
            instance_id=f"usaco__{pid}",
            problem_statement=competition_problem_text(row.get("description", "")),
            solution_stub=stdin_stub(),
            io_tests={"inputs": inputs, "outputs": outputs},
            repo="usaco",
            raw=raw,
        ))
    return out
