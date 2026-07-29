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

"""Serialization helpers for LeetCodeDataset call-based grading (linked lists / trees).

Mirrors the utilities bundled with `newfacade/LeetCodeDataset` evaluation so
`test_solution.py` can execute rows' ``check(candidate)`` blocks verbatim.
"""

from __future__ import annotations

_LCD_HELPERS = '''
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: "Optional[ListNode]" = None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val: int = 0, left: "Optional[TreeNode]" = None,
                 right: "Optional[TreeNode]" = None):
        self.val = val
        self.left = left
        self.right = right


def list_node(values: List) -> Optional[ListNode]:
    if not values:
        return None
    head = ListNode(values[0])
    cur = head
    for v in values[1:]:
        cur.next = ListNode(v)
        cur = cur.next
    return head


def linked_list_to_list(head: Optional[ListNode]) -> List:
    out: List = []
    while head is not None:
        out.append(head.val)
        head = head.next
    return out


def is_same_list(p: Optional[ListNode], q: Optional[ListNode]) -> bool:
    return linked_list_to_list(p) == linked_list_to_list(q)


def tree_node(values: List) -> Optional[TreeNode]:
    if not values:
        return None
    nodes = [TreeNode(v) if v is not None else None for v in values]
    kids = [n for n in nodes if n is not None]
    kid_i = 0
    for j in range(1, len(nodes)):
        if nodes[j] is None:
            continue
        parent = kids[kid_i]
        if j % 2 == 1:
            parent.left = nodes[j]
        else:
            parent.right = nodes[j]
            kid_i += 1
    return nodes[0]


def tree_node_to_list(root: Optional[TreeNode]) -> List:
    if root is None:
        return []
    from collections import deque
    q = deque([root])
    out: List = []
    while q:
        node = q.popleft()
        if node is None:
            out.append(None)
            continue
        out.append(node.val)
        q.append(node.left)
        q.append(node.right)
    while out and out[-1] is None:
        out.pop()
    return out
'''


def helpers_source() -> str:
    return _LCD_HELPERS.strip() + "\n"
