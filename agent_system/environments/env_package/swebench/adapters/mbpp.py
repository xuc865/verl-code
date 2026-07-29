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

"""MBPP(+) (sanitized) -> self-repair instances (assert-style, ``local_stub``).

A sanitized MBPP row carries:
  * ``prompt``       -- natural-language task description
  * ``code``         -- reference solution (one or more ``def``s)
  * ``test_imports`` -- list of import lines the tests need
  * ``test_list``    -- list of ``assert <call> == <expected>`` strings

Self-repair rewrite:
  * ``solution.py``      = the reference's function **signatures** with a
                           deliberately-wrong body (``return None``), so it
                           imports but is incorrect. (Stub-by-signature keeps the
                           public names the asserts call.)
  * ``test_solution.py`` = ``test_imports`` + ``from solution import *`` + a
                           single ``def test_mbpp()`` wrapping every assert.
  * ``FAIL_TO_PASS``     = ``["test_mbpp"]``
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from . import make_assert_instance

_FAIL = "test_mbpp"
_DEF_RE = re.compile(r"^def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*:", re.MULTILINE)

_PROBLEM_TMPL = (
    "Implement the following in `solution.py` so the tests pass. The current "
    "bodies are buggy placeholders that return wrong values. Edit `solution.py`, "
    "run `pytest`, read the failing assertions, and iterate until green.\n\n"
    "----- task -----\n{prompt}"
)


def _stub_from_code(code: str) -> str:
    """Keep each top-level ``def ...:`` signature, replace its body with a
    deliberately-wrong placeholder (``return None``) -- an injected bug, so the
    module imports cleanly but the asserts fail at turn 0.

    Falls back to a single failing placeholder if no signature is found, which
    still yields a (trivially failing) task rather than a crash.
    """
    sigs = _DEF_RE.findall(code or "")
    if not sigs:
        return ("def solve(*args, **kwargs):\n"
                "    return None  # BUG: placeholder returns a wrong value\n")
    parts = []
    for sig in sigs:
        parts.append(sig.rstrip())
        parts.append("    return None  # BUG: placeholder returns a wrong value")
        parts.append("")
    return "\n".join(parts) + "\n"


def _test_module(test_imports: List[str], test_list: List[str]) -> str:
    lines: List[str] = []
    for imp in (test_imports or []):
        lines.append(str(imp).rstrip())
    lines.append("from solution import *")
    lines.append("")
    lines.append(f"def {_FAIL}():")
    for assertion in (test_list or []):
        lines.append("    " + str(assertion).strip())
    if not test_list:
        lines.append("    assert False, 'no test_list in row'")
    return "\n".join(lines) + "\n"


def to_selfrepair_instances(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        code = row.get("code") or ""
        test_list = row.get("test_list") or []
        if not code or not test_list:
            continue
        task_id = str(row.get("task_id", f"idx{len(out)}"))
        out.append(make_assert_instance(
            instance_id=f"mbpp__{task_id}",
            problem_statement=_PROBLEM_TMPL.format(prompt=row.get("prompt", "")),
            solution_stub=_stub_from_code(code),
            test_code=_test_module(row.get("test_imports", []), test_list),
            fail_to_pass=[_FAIL],
            repo="mbpp",
            raw=row,
        ))
    return out
