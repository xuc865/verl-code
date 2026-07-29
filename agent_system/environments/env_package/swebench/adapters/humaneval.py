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

"""HumanEval(+) -> self-repair instances (assert-style, graded by ``local_stub``).

A HumanEval row carries:
  * ``prompt``            -- the function signature + docstring (no body)
  * ``entry_point``       -- the function name under test
  * ``test``              -- a ``def check(candidate): assert ...`` block
  * ``canonical_solution``-- the reference body (unused; we want it to fail first)

Self-repair rewrite (one row -> one instance):
  * ``solution.py``      = ``prompt`` + an **injected-bug** body (``return None``,
                           a deliberately wrong placeholder)
  * ``test_solution.py`` = ``from solution import <entry_point>`` + the row's
                           ``check`` block + a ``def test_humaneval(): check(<entry_point>)``
  * ``FAIL_TO_PASS``     = ``["test_humaneval"]``

The stub returns a wrong value, so the test fails; once the agent implements the
function the test passes -- a genuine multi-turn, goal-directed episode.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import make_assert_instance

_FAIL = "test_humaneval"

_PROBLEM_TMPL = (
    "Implement the function `{entry}` in `solution.py` so that it satisfies its "
    "docstring/specification below. The current body is a buggy placeholder that "
    "returns a wrong value, so the tests fail. Edit `solution.py`, then run "
    "`pytest` to see the failures and iterate until the tests pass.\n\n"
    "----- specification (solution.py) -----\n{prompt}"
)


def _stub_body(prompt: str) -> str:
    """``prompt`` ends after the signature/docstring with no body; append a
    deliberately-wrong placeholder body so the module imports cleanly but the
    function is incorrect (an injected bug, not a crash) -> tests fail at turn 0."""
    body = prompt
    if not body.endswith("\n"):
        body += "\n"
    body += "    return None  # BUG: placeholder returns a wrong value; implement the real logic\n"
    return body


def _test_module(entry: str, check_block: str) -> str:
    return (
        f"from solution import {entry}\n\n"
        f"{check_block.rstrip()}\n\n\n"
        f"def {_FAIL}():\n"
        f"    check({entry})\n"
    )


def to_selfrepair_instances(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        entry = (row.get("entry_point") or "candidate").strip()
        prompt = row.get("prompt") or ""
        check_block = row.get("test") or ""
        task_id = str(row.get("task_id", f"idx{len(out)}"))
        if not prompt or not check_block:
            # row is missing the pieces we need to build a self-repair task
            continue
        slug = task_id.replace("/", "_")
        out.append(make_assert_instance(
            instance_id=f"humaneval__{slug}",
            problem_statement=_PROBLEM_TMPL.format(entry=entry, prompt=prompt),
            solution_stub=_stub_body(prompt),
            test_code=_test_module(entry, check_block),
            fail_to_pass=[_FAIL],
            repo="humaneval",
            raw=row,
        ))
    return out
