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

"""ICPC-Eval -> self-repair instances (stdin/stdout, ``selfrepair_io``).

Dataset: ``RUC-AIBOX/ICPC-Eval`` (~118 problems).

We only keep problems with plain input/output pairs. Rows that require special
judges (``spj`` / ``special_judge`` flags) are skipped for the CPU IO bed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from . import make_io_instance
from .io_utils import (
    competition_problem_text,
    dict_cases_to_io_lists,
    list_cases_to_io_lists,
    stdin_stub,
)


def _is_spj(row: Dict[str, Any]) -> bool:
    spj_code = row.get("spj_code")
    if isinstance(spj_code, str) and spj_code.strip():
        return True
    for key in ("spj", "special_judge", "has_spj", "use_spj"):
        val = row.get(key)
        if isinstance(val, bool) and val:
            return True
        if isinstance(val, str) and val.strip().lower() in ("1", "true", "yes", "spj"):
            return True
    checker = str(row.get("checker", "") or row.get("judge_type", "")).lower()
    return checker not in ("", "standard", "ncmp", "token", "identical", "exact")


def _extract_io(row: Dict[str, Any]) -> tuple[List[str], List[str]]:
    examples = row.get("examples")
    if isinstance(examples, list):
        inputs, outputs = [], []
        for ex in examples:
            if isinstance(ex, (list, tuple)) and len(ex) >= 2:
                inputs.append(str(ex[0]))
                outputs.append(str(ex[1]))
        if inputs and len(inputs) == len(outputs):
            return inputs, outputs
    if isinstance(row.get("input"), dict) and isinstance(row.get("output"), dict):
        return dict_cases_to_io_lists(row.get("input"), row.get("output"))
    for key in ("test_cases", "tests", "samples", "public_tests"):
        inputs, outputs = list_cases_to_io_lists(row.get(key))
        if inputs and outputs:
            return inputs, outputs
    raw_cases = row.get("test_case")
    if isinstance(raw_cases, str) and raw_cases.strip():
        try:
            parsed = json.loads(raw_cases)
            if isinstance(parsed, list):
                return list_cases_to_io_lists(parsed)
        except Exception:  # noqa: BLE001
            pass
    inputs = row.get("inputs")
    outputs = row.get("outputs")
    if isinstance(inputs, list) and isinstance(outputs, list):
        return [str(x) for x in inputs], [str(x) for x in outputs]
    return [], []


def to_selfrepair_instances(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if _is_spj(row):
            continue
        inputs, outputs = _extract_io(row)
        if not inputs or not outputs or len(inputs) != len(outputs):
            continue
        pid = str(
            row.get("problem_id")
            or row.get("id")
            or row.get("name")
            or f"idx{len(out)}"
        ).replace("/", "_").replace(" ", "_")
        statement = (
            row.get("problem_statement")
            or row.get("description")
            or row.get("statement")
            or row.get("prompt")
            or row.get("title")
            or ""
        )
        contest = str(row.get("contest", row.get("source", "icpc"))).strip().lower()
        diff = str(row.get("difficulty", row.get("level", row.get("type", "unknown")))).strip().lower()
        raw = dict(row)
        raw.setdefault("difficulty", diff)
        raw.setdefault("platform", contest)
        out.append(make_io_instance(
            instance_id=f"icpc__{pid}",
            problem_statement=competition_problem_text(str(statement)),
            solution_stub=stdin_stub(),
            io_tests={"inputs": inputs, "outputs": outputs},
            repo="icpc",
            raw=raw,
        ))
    return out
