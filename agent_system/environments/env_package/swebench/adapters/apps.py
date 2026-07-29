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

"""APPS -> self-repair instances (stdin/stdout, graded by ``selfrepair_io``).

Call-based (``fn_name``) rows are converted to the same stdin/stdout protocol used
by APPS test: JSON-lines args on stdin, JSON result on stdout, with a small
``__main__`` harness appended to the ``Solution`` stub.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from . import make_io_instance

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

_PROBLEM_TMPL = (
    "Solve the following programming problem in `solution.py`. Your program must "
    "read input from STDIN and print the answer to STDOUT (unless a function name "
    "is specified). The current stub is a buggy placeholder and does not produce "
    "correct output. Edit `solution.py`, run it against the tests (e.g. "
    "`python solution.py`), inspect the per-case diffs, and iterate until all "
    "cases pass.\n\n"
    "----- problem -----\n{question}"
)

_CALL_PROBLEM_TMPL = (
    "Solve the following programming problem in `solution.py`.\n"
    "This is a call-based APPS problem rewritten to the same stdin/stdout protocol "
    "used by APPS test / exec-mode self-repair:\n"
    "  * Implement `Solution.{fn_name}` (keep the class / method name).\n"
    "  * Do NOT remove the `if __name__ == \"__main__\"` harness.\n"
    "  * The harness reads one JSON value per argument from STDIN (one per line), "
    "calls `Solution().{fn_name}(*args)`, and prints `json.dumps(result)`.\n"
    "  * Self-test like: "
    "`printf '%s\\n' '<json-arg1>' '<json-arg2>' | python solution.py`\n"
    "The current stub is a buggy placeholder. Edit the method body, self-test, "
    "then `<finish>` when ready.\n\n"
    "----- problem -----\n{question}"
)

_CONVERT_FN_NAME = os.environ.get("APPS_CONVERT_FN_NAME_TO_STDIN", "1").lower() in (
    "1", "true", "yes",
)


def _parse_io(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            v = json.loads(raw)
            if isinstance(v, dict):
                return v
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _encode_call_stdin(args: Any) -> str:
    if isinstance(args, str):
        text = args if args.endswith("\n") else args + "\n"
        try:
            for ln in text.splitlines():
                if ln.strip():
                    json.loads(ln)
            return text
        except Exception:  # noqa: BLE001
            return json.dumps(args) + "\n"
    if not isinstance(args, (list, tuple)):
        return json.dumps(args) + "\n"
    return "".join(json.dumps(a) + "\n" for a in args)


def _encode_call_stdout(result: Any) -> str:
    return json.dumps(result) + "\n"


def _normalize_call_starter(starter: str, fn_name: str) -> str:
    code = (starter or "").strip("\n")
    if not code:
        return (
            "from typing import *\n\n"
            f"class Solution:\n"
            f"    def {fn_name}(self, *args):\n"
            f"        return None\n"
        )
    needs_typing = any(
        tok in code for tok in ("List[", "Dict[", "Optional[", "Tuple[", "Set[", "Any")
    )
    if needs_typing and "typing" not in code:
        code = "from typing import *\n" + code
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass
    if code.rstrip().endswith(":"):
        code = code.rstrip() + "\n        return None\n"
    else:
        code = code + "\n        return None\n"
    try:
        ast.parse(code)
    except SyntaxError:
        pass
    return code


def _append_call_harness(solution_code: str, fn_name: str) -> str:
    harness = (
        "\n\n"
        'if __name__ == "__main__":\n'
        "    import sys as _sys, json as _json\n"
        "    _lines = [ln for ln in _sys.stdin.read().splitlines() if ln.strip() != \"\"]\n"
        "    _args = [_json.loads(ln) for ln in _lines]\n"
        f"    _result = getattr(Solution(), {fn_name!r})(*_args)\n"
        "    print(_json.dumps(_result))\n"
    )
    if 'getattr(Solution(),' in solution_code and "__main__" in solution_code:
        return solution_code
    return solution_code.rstrip() + harness


def convert_fn_name_to_stdin(
    fn_name: str,
    starter: str,
    inputs: List[Any],
    outputs: List[Any],
) -> Tuple[str, Dict[str, Any], str]:
    stub = _append_call_harness(_normalize_call_starter(starter, fn_name), fn_name)
    n = min(len(inputs), len(outputs))
    io_tests = {
        "inputs": [_encode_call_stdin(inputs[i]) for i in range(n)],
        "outputs": [_encode_call_stdout(outputs[i]) for i in range(n)],
        "_original_fn_name": fn_name,
    }
    return stub, io_tests, _CALL_PROBLEM_TMPL


def to_selfrepair_instances(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        io = _parse_io(row.get("input_output"))
        inputs = io.get("inputs") or []
        outputs = io.get("outputs") or []
        if not inputs or not outputs:
            continue
        pid = str(row.get("problem_id", f"idx{len(out)}"))
        starter = (row.get("starter_code") or "").strip()
        fn_name: Optional[str] = io.get("fn_name")
        raw = dict(row)

        if fn_name and _CONVERT_FN_NAME:
            stub, io_tests, tmpl = convert_fn_name_to_stdin(
                str(fn_name), starter, list(inputs), list(outputs),
            )
            problem = tmpl.format(question=row.get("question", ""), fn_name=fn_name)
            raw["_converted_fn_name_to_stdin"] = True
            raw["_original_fn_name"] = fn_name
        else:
            stub = (starter + "\n") if starter else _STDIN_STUB
            io_tests = {"inputs": inputs, "outputs": outputs}
            if fn_name:
                io_tests["fn_name"] = fn_name
            problem = _PROBLEM_TMPL.format(question=row.get("question", ""))

        out.append(make_io_instance(
            instance_id=f"apps__{pid}",
            problem_statement=problem,
            solution_stub=stub,
            io_tests=io_tests,
            repo="apps",
            raw=raw,
        ))
    return out
