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

"""newfacade/LeetCodeDataset -> self-repair instances (assert-style, ``local_stub``).

Each row carries:
  * ``problem_description`` -- full LeetCode statement
  * ``starter_code``        -- ``class Solution: def method(...)``
  * ``prompt``              -- import prefix (typing, collections, ...)
  * ``entry_point``         -- e.g. ``Solution().twoSum`` (bound method passed to check)
  * ``test``                -- ``def check(candidate): assert ...``
  * ``difficulty``          -- Easy / Medium / Hard
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from . import make_assert_instance
from .leetcode_helpers import helpers_source

_FAIL = "test_leetcode"

_PROBLEM_TMPL = (
    "Solve the LeetCode problem in `solution.py` (difficulty: {difficulty}). "
    "The current method bodies are buggy placeholders that return wrong values. "
    "Edit `solution.py`, inspect with `cat solution.py` if needed, and submit "
    "with `<finish>` for hidden grading.\n\n"
    "----- problem -----\n{description}"
)

_DEF_TAIL_RE = re.compile(r"^(\s*)def\s+\w+\s*\([^)]*\)(?:\s*->\s*[^:]+)?\s*:\s*$")


def _normalize_difficulty(raw: str) -> str:
    d = str(raw or "unknown").strip().lower()
    if d in ("easy", "medium", "hard"):
        return d
    return d or "unknown"


def _import_prefix(prompt: str) -> str:
    """Keep only the import prefix from a LeetCodeDataset ``prompt`` field."""
    lines: List[str] = []
    started = False
    for line in (prompt or "").splitlines():
        s = line.strip()
        if not s:
            if started:
                continue
            continue
        is_import = (
            s.startswith("import ")
            or s.startswith("from ")
            or re.match(r"^inf\s*=", s) is not None
        )
        if is_import:
            lines.append(line.rstrip())
            started = True
        elif started:
            break
    return "\n".join(lines).strip()


def _inject_bug(starter_code: str) -> str:
    """Turn an empty LeetCode method stub into a deliberately-wrong body."""
    lines = starter_code.rstrip("\n").split("\n")
    out: List[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        m = _DEF_TAIL_RE.match(line.rstrip())
        if not m:
            continue
        base_indent = len(m.group(1))
        child_indent = base_indent + 4
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        nxt_indent = len(nxt) - len(nxt.lstrip()) if nxt.strip() else -1
        if not nxt.strip() or nxt_indent <= base_indent:
            out.append(" " * child_indent + "return None  # BUG: placeholder")
    return "\n".join(out) + "\n"


def _solution_stub(row: Dict[str, Any]) -> str:
    imports = _import_prefix(row.get("prompt") or "")
    body = _inject_bug(row.get("starter_code") or "")
    parts = [p for p in (imports, body) if p.strip()]
    return "\n\n".join(parts) + "\n"


def _needs_helpers(check_block: str, starter_code: str) -> bool:
    blob = f"{check_block}\n{starter_code}"
    tokens = ("list_node", "tree_node", "is_same_list", "ListNode", "TreeNode")
    return any(t in blob for t in tokens)


def _solution_import(entry_point: str) -> str:
    m = re.match(r"(\w+)\s*\(", str(entry_point or "").strip())
    if m:
        return f"from solution import {m.group(1)}\n"
    return ""


def _test_module(entry_point: str, check_block: str, starter_code: str) -> str:
    parts: List[str] = []
    sol_import = _solution_import(entry_point)
    if sol_import:
        parts.append(sol_import.rstrip())
    if _needs_helpers(check_block, starter_code):
        parts.append(helpers_source().rstrip())
    parts.append(check_block.rstrip())
    parts.append("")
    parts.append(f"def {_FAIL}():")
    parts.append(f"    check({entry_point})")
    return "\n".join(parts) + "\n"


def to_selfrepair_instances(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        task_id = str(row.get("task_id") or f"idx{len(out)}").strip()
        entry_point = str(row.get("entry_point") or "").strip()
        check_block = row.get("test") or ""
        starter_code = row.get("starter_code") or ""
        description = row.get("problem_description") or row.get("query") or ""
        if not task_id or not entry_point or not check_block or not starter_code:
            continue
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
        diff = _normalize_difficulty(row.get("difficulty", "unknown"))
        raw = dict(row)
        raw["difficulty"] = diff
        out.append(make_assert_instance(
            instance_id=f"leetcode__{slug}",
            problem_statement=_PROBLEM_TMPL.format(
                difficulty=diff,
                description=description.strip(),
            ),
            solution_stub=_solution_stub(row),
            test_code=_test_module(entry_point, check_block, starter_code),
            fail_to_pass=[_FAIL],
            repo="leetcode",
            raw=raw,
        ))
    return out
