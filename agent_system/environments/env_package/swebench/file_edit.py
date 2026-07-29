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

"""Merge agent file edits: overwrite, search/replace patch, or append."""

from __future__ import annotations

from typing import Any, Dict


class EditApplyError(ValueError):
    """Raised when a patch/insert cannot be applied to the current file."""


def apply_edit_action(files: Dict[str, str], path: str, action: Dict[str, Any]) -> str:
    """Apply an edit action to ``files[path]`` and return the merged content.

    Supported ``action`` shapes (``type`` must be ``edit``):

    - ``mode=overwrite`` (default): ``content`` replaces the whole file.
    - ``mode=patch``: replace one unique ``search`` occurrence with ``replace``.
    - ``mode=insert``: append ``insert`` (or ``content``) to the file.
    """
    mode = str(action.get("mode", "overwrite")).lower()
    current = files.get(path, "")

    if mode == "patch":
        search = action.get("search", "")
        replace = action.get("replace", "")
        if not search:
            raise EditApplyError("patch edit requires non-empty <search> text")
        count = current.count(search)
        if count == 0:
            raise EditApplyError(
                f"search text not found in {path}; use <execute_bash>cat {path}</execute_bash> "
                "to inspect the current file"
            )
        if count > 1:
            raise EditApplyError(
                f"search text matches {count} times in {path}; include more surrounding context "
                "so the match is unique"
            )
        result = current.replace(search, replace, 1)
    elif mode == "insert":
        insert = action.get("insert", action.get("content", ""))
        if not insert.strip():
            raise EditApplyError("insert edit requires non-empty <insert> text")
        if not current:
            result = insert
        elif current.endswith("\n"):
            result = current + insert
        else:
            result = current + "\n" + insert
    else:
        result = action.get("content", "")
        if not result and not current:
            raise EditApplyError("overwrite edit requires non-empty <code> content")

    files[path] = result
    return result
