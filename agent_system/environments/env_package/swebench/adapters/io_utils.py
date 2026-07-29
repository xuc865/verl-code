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

"""Shared helpers for stdin/stdout benchmark adapters."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

_STDIN_STUB = (
    "import sys\n\n"
    "def main():\n"
    "    data = sys.stdin.read()\n"
    "    # BUG: placeholder reads input but prints a wrong constant.\n"
    "    #      Parse `data`, compute the answer, and print it instead.\n"
    "    print(0)\n\n\n"
    'if __name__ == "__main__":\n'
    "    main()\n"
)

_COMPETITION_PROBLEM_TMPL = (
    "Solve the following competitive-programming problem in `solution.py`, "
    "reading input from STDIN and printing to STDOUT. The current stub is "
    "incorrect. Edit `solution.py`, run it against the tests, read the per-case "
    "diffs, and iterate until all cases pass.\n\n"
    "----- problem -----\n{question}"
)


def stdin_stub() -> str:
    return _STDIN_STUB


def competition_problem_text(question: str) -> str:
    return _COMPETITION_PROBLEM_TMPL.format(question=question or "")


def _sort_case_keys(keys: List[Any]) -> List[Any]:
    def _key(k: Any):
        s = str(k)
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted(keys, key=_key)


def dict_cases_to_io_lists(
    inp: Any,
    out: Any,
) -> Tuple[List[str], List[str]]:
    """Convert numbered dict test cases (USACO-style) to parallel IO lists."""
    if not isinstance(inp, dict) or not isinstance(out, dict):
        return [], []
    inputs: List[str] = []
    outputs: List[str] = []
    for key in _sort_case_keys(list(inp.keys())):
        if key not in out:
            continue
        i_val = inp.get(key)
        o_val = out.get(key)
        if i_val is None or o_val is None:
            continue
        inputs.append(str(i_val))
        outputs.append(str(o_val))
    return inputs, outputs


def list_cases_to_io_lists(cases: Any) -> Tuple[List[str], List[str]]:
    """Convert a list of ``{"input","output"}`` dicts to parallel IO lists."""
    if not isinstance(cases, list):
        return [], []
    inputs: List[str] = []
    outputs: List[str] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        if "input" not in case or "output" not in case:
            continue
        inputs.append(str(case.get("input", "")))
        outputs.append(str(case.get("output", "")))
    return inputs, outputs


def normalize_io_tests(io_tests: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy with stringified IO pairs."""
    inputs = [str(x) for x in (io_tests or {}).get("inputs") or []]
    outputs = [str(x) for x in (io_tests or {}).get("outputs") or []]
    out = {"inputs": inputs, "outputs": outputs}
    if io_tests and io_tests.get("fn_name"):
        out["fn_name"] = io_tests["fn_name"]
    return out
