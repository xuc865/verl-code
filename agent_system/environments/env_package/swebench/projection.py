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

"""
Action projection for the multi-turn coding environment.

The agent's textual response is parsed into a *structured action*. We support a
small, explicit command set so the scaffold stays simple and the generated
code is easy to localize for DIDPO snippet extraction:

    <think> ... </think>                          (required reasoning block)
    <execute_bash> shell command </execute_bash>   run ls/cat/python (not grep)
    <edit path="relative/file.py"><search>old lines</search><replace>new lines</replace></edit>
                                                   patch a unique region (preferred for small fixes)
    <edit path="relative/file.py"><code> ...new file content... </code></edit>
                                                   overwrite (or create) a whole file
    <edit path="relative/file.py"><insert>snippet</insert></edit>
                                                   append a code fragment to the file
    <finish></finish>                              submit the current patch

``grep`` / ``rg`` / ``ag`` are not part of the action space and are rejected by
the environment if requested via ``<execute_bash>``.

The code the agent writes lives inside <code> ... </code>, which is exactly the
block DIDPO's snippet extractor looks for, so functional snippets are recovered
verbatim from the response.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_BASH_RE = re.compile(r"<execute_bash>(.*?)</execute_bash>", re.DOTALL)
_EDIT_RE = re.compile(r"<edit\s+path=[\"'](.*?)[\"']\s*>(.*?)</edit>", re.DOTALL)
_CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_REPLACE_RE = re.compile(r"<replace>(.*?)</replace>", re.DOTALL)
_OLD_RE = re.compile(r"<old>(.*?)</old>", re.DOTALL)
_NEW_RE = re.compile(r"<new>(.*?)</new>", re.DOTALL)
_INSERT_RE = re.compile(r"<insert>(.*?)</insert>", re.DOTALL)
_FINISH_RE = re.compile(r"<finish\s*/?>", re.DOTALL)
_THOUGHT_LINE_RE = re.compile(r"(?im)^\s*Thought\s*:")
_BASH_FENCE_RE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)
_PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_ACTION_LINE_RE = re.compile(r"(?im)^\s*Action\s*:\s*(.+)$")


def _has_think(text: str) -> bool:
    if _THINK_RE.search(text) is not None:
        return True
    return _THOUGHT_LINE_RE.search(text) is not None


def _fallback_bash(text: str) -> Optional[str]:
    m = _BASH_FENCE_RE.search(text)
    if m:
        cmd = m.group(1).strip()
        if cmd:
            return cmd.split("\n")[0].strip() if cmd.count("\n") == 0 else cmd.strip()
    m = _ACTION_LINE_RE.search(text)
    if m:
        cmd = m.group(1).strip().strip("`")
        if cmd:
            return cmd
    m = _PY_FENCE_RE.search(text)
    if m:
        body = m.group(1).strip()
        first = body.split("\n", 1)[0].strip()
        if first.startswith("python") or "solution.py" in first:
            return first
    return None


def _fallback_edit(text: str) -> Optional[Tuple[str, str]]:
    m = _EDIT_RE.search(text)
    if m is None:
        path_m = re.search(
            r"(?im)(?:edit|overwrite|write)\s+(?:file\s+)?[`\"']?([^\s`\"']+\.py)[`\"']?",
            text,
        )
        code_m = _PY_FENCE_RE.search(text)
        if path_m and code_m:
            return path_m.group(1).strip(), code_m.group(1)
        return None
    path = m.group(1).strip()
    body = m.group(2)
    code_m = _CODE_RE.search(body)
    content = code_m.group(1) if code_m else body
    if path and content.strip():
        return path, content
    return None


def _parse_single(text: str) -> Tuple[Dict[str, Any], int]:
    """Parse one response into (action_dict, valid)."""
    has_think = _has_think(text)

    finish = _FINISH_RE.search(text)
    edit = _EDIT_RE.search(text)
    bash = _BASH_RE.search(text)

    if edit is not None:
        path = edit.group(1).strip()
        body = edit.group(2)
        search_m = _SEARCH_RE.search(body) or _OLD_RE.search(body)
        replace_m = _REPLACE_RE.search(body) or _NEW_RE.search(body)
        insert_m = _INSERT_RE.search(body)
        code_m = _CODE_RE.search(body)

        if search_m is not None and replace_m is not None:
            search = search_m.group(1)
            replace = replace_m.group(1)
            action = {
                "type": "edit",
                "path": path,
                "mode": "patch",
                "search": search,
                "replace": replace,
            }
            valid = 1 if (has_think and path and search.strip()) else 0
            return action, valid

        if insert_m is not None:
            insert = insert_m.group(1)
            action = {"type": "edit", "path": path, "mode": "insert", "insert": insert}
            valid = 1 if (has_think and path and insert.strip()) else 0
            return action, valid

        content = code_m.group(1) if code_m else body
        action = {"type": "edit", "path": path, "mode": "overwrite", "content": content}
        valid = 1 if (has_think and path and content.strip()) else 0
        return action, valid

    if bash is not None:
        cmd = bash.group(1).strip()
        action = {"type": "bash", "cmd": cmd}
        valid = 1 if (has_think and cmd) else 0
        return action, valid

    if finish is not None:
        return {"type": "finish"}, (1 if has_think else 0)

    fb_edit = _fallback_edit(text)
    if fb_edit is not None:
        path, content = fb_edit
        action = {"type": "edit", "path": path, "mode": "overwrite", "content": content}
        return action, (1 if (has_think and path and content.strip()) else 0)

    fb_bash = _fallback_bash(text)
    if fb_bash is not None:
        action = {"type": "bash", "cmd": fb_bash}
        return action, (1 if (has_think and fb_bash) else 0)

    if re.search(r"<finish\b", text, re.IGNORECASE) and has_think:
        return {"type": "finish"}, 1

    # No recognizable action -> no-op (invalid)
    return {"type": "noop"}, 0


def parse_swebench_action(text: str) -> Tuple[Dict[str, Any], int]:
    """Public entry point shared by the env and DIDPO snippet extraction."""
    return _parse_single(text if isinstance(text, str) else str(text))


def swebench_projection(actions: List[str]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Vectorized projection: list[str] -> (list[action_dict], list[valid])."""
    parsed: List[Dict[str, Any]] = []
    valids: List[int] = []
    for text in actions:
        a, v = _parse_single(text if isinstance(text, str) else str(text))
        parsed.append(a)
        valids.append(v)
    return parsed, valids
