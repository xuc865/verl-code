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
DIDPO diff / snippet extraction (tokenizer side).

Paper-aligned pipeline:

  1. Parse each agent response into *root diffs* with metadata
     ``(u, v, c, q)`` — normalized source, token span, edit type
     ``c ∈ {add, del, none}``, and (later) task uid ``q``.
  2. Provide a *structural fallback* splitter (function / hunk AST units)
     used when cross-rollout alignment yields no usable boundary.
  3. Diff-gate unchanged re-emissions so copied code receives no local credit.
  4. Map char spans → response-token spans for advantage projection.

Collector entry point: ``parse_response_to_snippets``.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Structural fallback levels (coarse -> fine). Kept for ablations / fallback.
LEVEL_FUNCTION = 0
LEVEL_HUNK = 1
NUM_LEVELS = 2

EDIT_ADD = "add"
EDIT_DEL = "del"
EDIT_NONE = "none"

# Origin tags on emitted units.
ORIGIN_ROOT = "root"
ORIGIN_FALLBACK = "fallback"


@dataclass
class Snippet:
    """A root diff or a structural-fallback sub-diff candidate."""

    level: int
    signature: str
    source: str
    size: float
    char_start: int
    char_end: int
    token_start: int = -1
    token_end: int = -1
    changed: bool = True
    edit_type: str = EDIT_NONE
    origin: str = ORIGIN_FALLBACK
    path: str = ""
    lines: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("lines") is None:
            d["lines"] = [ln for ln in self.source.splitlines() if ln.strip()]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Snippet":
        keys = {
            "level", "signature", "source", "size", "char_start", "char_end",
            "token_start", "token_end", "changed", "edit_type", "origin", "path", "lines",
        }
        return Snippet(**{k: d[k] for k in keys if k in d})


_CODE_BLOCK_PATTERNS = [
    re.compile(r"<code>(.*?)</code>", re.DOTALL),
    re.compile(r"<action>(.*?)</action>", re.DOTALL),
    re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL),
]


def extract_action_code(response_text: str) -> Tuple[str, int]:
    for pat in _CODE_BLOCK_PATTERNS:
        m = pat.search(response_text)
        if m:
            return m.group(1), m.start(1)
    return response_text, 0


def normalize_source(src: str) -> str:
    """Whitespace/comment-insensitive normalization used for similarity & gating."""
    lines = []
    for line in src.splitlines():
        stripped = re.sub(r"\s+#.*$", "", line).rstrip()
        if stripped.strip():
            lines.append(stripped)
    return "\n".join(lines)


def normalize_lines(src: str) -> List[str]:
    norm = normalize_source(src)
    return norm.splitlines() if norm else []


def source_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------------------------------- #
# Edit-type / root-diff extraction                                            #
# --------------------------------------------------------------------------- #
_EDIT_RE = re.compile(r"<edit\s+path=[\"'](.*?)[\"']\s*>(.*?)</edit>", re.DOTALL)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_REPLACE_RE = re.compile(r"<replace>(.*?)</replace>", re.DOTALL | re.IGNORECASE)
_OLD_RE = re.compile(r"<old>(.*?)</old>", re.DOTALL | re.IGNORECASE)
_NEW_RE = re.compile(r"<new>(.*?)</new>", re.DOTALL | re.IGNORECASE)
_INSERT_RE = re.compile(r"<insert>(.*?)</insert>", re.DOTALL | re.IGNORECASE)
_CODE_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL)


def _root_units_from_response(response_text: str) -> List[Tuple[str, str, int, int, str]]:
    """Return list of (edit_type, payload_text, char_start, char_end, path)."""
    m = _EDIT_RE.search(response_text)
    if m is None:
        # Best-effort: treat fenced / <code> payload as overwrite (c=none).
        code, off = extract_action_code(response_text)
        if normalize_source(code):
            return [(EDIT_NONE, code, off, off + len(code), "")]
        return []

    path = m.group(1).strip()
    body = m.group(2)
    search_m = _SEARCH_RE.search(body) or _OLD_RE.search(body)
    replace_m = _REPLACE_RE.search(body) or _NEW_RE.search(body)
    insert_m = _INSERT_RE.search(body)
    code_m = _CODE_RE.search(body)
    units: List[Tuple[str, str, int, int, str]] = []

    if search_m is not None and replace_m is not None:
        # Patch: del(search) + add(replace), spans relative to full response.
        body_off = m.start(2)
        if search_m.group(1).strip():
            units.append((
                EDIT_DEL, search_m.group(1),
                body_off + search_m.start(1), body_off + search_m.end(1), path,
            ))
        if replace_m.group(1).strip():
            units.append((
                EDIT_ADD, replace_m.group(1),
                body_off + replace_m.start(1), body_off + replace_m.end(1), path,
            ))
        return units

    if insert_m is not None and insert_m.group(1).strip():
        body_off = m.start(2)
        return [(
            EDIT_ADD, insert_m.group(1),
            body_off + insert_m.start(1), body_off + insert_m.end(1), path,
        )]

    if code_m is not None:
        body_off = m.start(2)
        return [(
            EDIT_NONE, code_m.group(1),
            body_off + code_m.start(1), body_off + code_m.end(1), path,
        )]

    # Raw edit body as overwrite.
    return [(EDIT_NONE, body, m.start(2), m.end(2), path)]


def extract_root_diffs(response_text: str) -> List[Snippet]:
    """Extract root diffs with edit-type metadata (paper Metadata(s))."""
    out: List[Snippet] = []
    for etype, payload, cs, ce, path in _root_units_from_response(response_text):
        norm = normalize_source(payload)
        if not norm:
            continue
        lines = normalize_lines(payload)
        sig = f"root::{etype}::{path}::{hashlib.md5(norm.encode('utf-8')).hexdigest()[:10]}"
        out.append(Snippet(
            level=LEVEL_FUNCTION,
            signature=sig,
            source=norm,
            size=float(max(1, len(lines))),
            char_start=int(cs),
            char_end=int(ce),
            edit_type=etype,
            origin=ORIGIN_ROOT,
            path=path,
            lines=lines,
        ))
    return out


# --------------------------------------------------------------------------- #
# Structural fallback (rule-based splitter)                                   #
# --------------------------------------------------------------------------- #
def _func_signature(node: ast.AST) -> str:
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    n_pos = len(node.args.args)
    n_kw = len(node.args.kwonlyargs)
    has_vararg = node.args.vararg is not None
    has_kwarg = node.args.kwarg is not None
    has_return = any(isinstance(d, ast.Return) and d.value is not None for d in ast.walk(node))
    return (
        f"fn::{node.name}::pos{n_pos}::kw{n_kw}::va{int(has_vararg)}"
        f"::kwa{int(has_kwarg)}::ret{int(has_return)}"
    )


def _stmt_signature(node: ast.stmt) -> str:
    kind = type(node).__name__
    if isinstance(node, ast.Assign):
        targets = "+".join(_target_name(t) for t in node.targets)
        return f"stmt::Assign::{targets}"
    if isinstance(node, ast.AugAssign):
        return f"stmt::AugAssign::{_target_name(node.target)}"
    if isinstance(node, ast.AnnAssign):
        return f"stmt::AnnAssign::{_target_name(node.target)}"
    if isinstance(node, (ast.If, ast.While)):
        return f"stmt::{kind}::{_expr_shape(node.test)}"
    if isinstance(node, ast.For):
        return f"stmt::For::{_target_name(node.target)}"
    if isinstance(node, ast.Return):
        return "stmt::Return"
    if isinstance(node, ast.Expr):
        return f"stmt::Expr::{_expr_shape(node.value)}"
    if isinstance(node, (ast.Raise, ast.Assert, ast.Pass, ast.Break, ast.Continue)):
        return f"stmt::{kind}"
    return f"stmt::{kind}"


def _target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return "subscript"
    if isinstance(node, (ast.Tuple, ast.List)):
        return "tuple"
    return type(node).__name__


def _expr_shape(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return f"call:{_target_name(node.func)}"
    if isinstance(node, ast.Compare):
        return "compare"
    if isinstance(node, ast.BoolOp):
        return "boolop"
    if isinstance(node, ast.BinOp):
        return "binop"
    return type(node).__name__


def _node_size(node: ast.AST) -> int:
    return sum(1 for _ in ast.walk(node))


def _linecol_to_offset(source: str, lineno: int, col: int) -> int:
    offset = 0
    cur_line = 1
    i = 0
    n = len(source)
    while cur_line < lineno and i < n:
        if source[i] == "\n":
            cur_line += 1
        i += 1
    return i + col


def _src_segment(source: str, node: ast.AST) -> Optional[Tuple[str, int, int]]:
    seg = ast.get_source_segment(source, node, padded=False)
    if seg is None:
        return None
    try:
        start = _linecol_to_offset(source, node.lineno, node.col_offset)
        end = _linecol_to_offset(source, node.end_lineno, node.end_col_offset)
    except Exception:
        return None
    return seg, start, end


def _maybe_add_stmt(snippets: List[Snippet], code: str, stmt: ast.stmt, base_off: int,
                    edit_type: str, path: str) -> None:
    seg = _src_segment(code, stmt)
    if seg is None:
        return
    text, cs, ce = seg
    norm = normalize_source(text)
    if not norm:
        return
    snippets.append(Snippet(
        level=LEVEL_HUNK,
        signature=_stmt_signature(stmt),
        source=norm,
        size=float(_node_size(stmt)),
        char_start=base_off + cs,
        char_end=base_off + ce,
        edit_type=edit_type,
        origin=ORIGIN_FALLBACK,
        path=path,
        lines=normalize_lines(text),
    ))


def extract_structural_fallback(response_text: str, edit_type: str = EDIT_NONE,
                                path: str = "", code: Optional[str] = None,
                                base_off: int = 0) -> List[Snippet]:
    """Rule-based structural proposals (paper: fallback when alignment fails)."""
    if code is None:
        code, base_off = extract_action_code(response_text)
    snippets: List[Snippet] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        norm = normalize_source(code)
        if not norm:
            return []
        sig = "opaque::" + hashlib.md5(norm.encode("utf-8")).hexdigest()[:8]
        return [Snippet(
            level=LEVEL_FUNCTION,
            signature=sig,
            source=norm,
            size=float(max(1, len(norm.split()))),
            char_start=base_off,
            char_end=base_off + len(code),
            edit_type=edit_type,
            origin=ORIGIN_FALLBACK,
            path=path,
            lines=normalize_lines(code),
        )]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = _src_segment(code, node)
            if seg is None:
                continue
            text, cs, ce = seg
            snippets.append(Snippet(
                level=LEVEL_FUNCTION,
                signature=_func_signature(node),
                source=normalize_source(text),
                size=float(_node_size(node)),
                char_start=base_off + cs,
                char_end=base_off + ce,
                edit_type=edit_type,
                origin=ORIGIN_FALLBACK,
                path=path,
                lines=normalize_lines(text),
            ))
            for stmt in node.body:
                _maybe_add_stmt(snippets, code, stmt, base_off, edit_type, path)

    for stmt in tree.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _maybe_add_stmt(snippets, code, stmt, base_off, edit_type, path)
        elif isinstance(stmt, ast.ClassDef):
            for inner in stmt.body:
                if not isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _maybe_add_stmt(snippets, code, inner, base_off, edit_type, path)
    return snippets


def extract_snippets(response_text: str) -> List[Snippet]:
    """Backward-compatible: root diffs + structural fallback proposals."""
    roots = extract_root_diffs(response_text)
    out: List[Snippet] = list(roots)
    if roots:
        for r in roots:
            # Map char span back into raw payload for AST parse.
            payload = response_text[r.char_start:r.char_end]
            out.extend(extract_structural_fallback(
                response_text, edit_type=r.edit_type, path=r.path,
                code=payload, base_off=r.char_start,
            ))
    else:
        out.extend(extract_structural_fallback(response_text))
    return out


# --------------------------------------------------------------------------- #
# Diff-gating                                                                 #
# --------------------------------------------------------------------------- #
def apply_diff_gate(snippets: Sequence[Snippet], prev_sources: Dict[str, str]) -> None:
    """Mark ``changed`` iff source differs from the previous same-signature emission."""
    for s in snippets:
        key = f"{s.edit_type}::{s.signature}"
        prev = prev_sources.get(key)
        if prev is not None and prev == s.source:
            s.changed = False
        else:
            s.changed = True
        prev_sources[key] = s.source


# --------------------------------------------------------------------------- #
# Token-span mapping                                                          #
# --------------------------------------------------------------------------- #
def build_token_char_offsets(token_ids: Sequence[int], tokenizer) -> Tuple[str, List[Tuple[int, int]]]:
    offsets: List[Tuple[int, int]] = []
    pieces: List[str] = []
    cursor = 0
    for tid in token_ids:
        piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
        start = cursor
        end = cursor + len(piece)
        offsets.append((start, end))
        pieces.append(piece)
        cursor = end
    return "".join(pieces), offsets


def char_span_to_token_span(offsets: Sequence[Tuple[int, int]],
                            char_start: int,
                            char_end: int) -> Tuple[int, int]:
    tok_start = -1
    tok_end = -1
    for k, (cs, ce) in enumerate(offsets):
        if ce <= char_start:
            continue
        if cs >= char_end:
            break
        if tok_start == -1:
            tok_start = k
        tok_end = k + 1
    return tok_start, tok_end


def map_snippets_to_token_spans(snippets: Sequence[Snippet],
                                offsets: Sequence[Tuple[int, int]]) -> None:
    for s in snippets:
        s.token_start, s.token_end = char_span_to_token_span(
            offsets, s.char_start, s.char_end)


def map_snippets_to_token_spans_proportional(snippets: Sequence[Snippet],
                                             text_len: int,
                                             n_tokens: int) -> None:
    """Cheap span map: char fraction → token index (no per-token decode)."""
    text_len = max(1, int(text_len))
    n_tokens = max(1, int(n_tokens))
    for s in snippets:
        ts = int(s.char_start / text_len * n_tokens)
        te = int(s.char_end / text_len * n_tokens)
        te = max(ts + 1, min(n_tokens, te))
        s.token_start, s.token_end = max(0, ts), te


_EDIT_HINT_RE = re.compile(r"<edit\b|<code>|```", re.IGNORECASE)


def parse_response_to_snippets(token_ids: Sequence[int],
                               tokenizer,
                               prev_sources: Dict[str, str],
                               *,
                               mode: str = "full",
                               predecoded_text: Optional[str] = None) -> List[Dict[str, Any]]:
    """Collector helper: root diffs (+ optional structural fallback), gated, token-mapped.

    Parameters
    ----------
    mode :
      - ``full``: root diffs + AST fallback; precise per-token char offsets.
      - ``roots_only`` / ``lightweight``: root diffs only; proportional token spans.
        Skips non-edit responses quickly. Near-zero overhead vs GRPO collect path.
    predecoded_text :
      If the rollout loop already ``batch_decode``d the response, pass it here to
      avoid a second decode / per-token decode in lightweight mode.
    """
    if token_ids is None or len(token_ids) == 0:
        return []

    mode = (mode or "full").lower()
    lightweight = mode in ("roots_only", "lightweight", "exact_only")

    if lightweight:
        text = predecoded_text
        if text is None:
            text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
        if not _EDIT_HINT_RE.search(text):
            return []
        snippets = extract_root_diffs(text)
        apply_diff_gate(snippets, prev_sources)
        map_snippets_to_token_spans_proportional(snippets, len(text), len(token_ids))
        snippets = [s for s in snippets if s.token_start >= 0 and s.token_end > s.token_start]
        return [s.to_dict() for s in snippets]

    text, offsets = build_token_char_offsets(token_ids, tokenizer)
    if not _EDIT_HINT_RE.search(text):
        return []
    snippets = extract_snippets(text)
    apply_diff_gate(snippets, prev_sources)
    map_snippets_to_token_spans(snippets, offsets)
    snippets = [s for s in snippets if s.token_start >= 0 and s.token_end > s.token_start]
    return [s.to_dict() for s in snippets]
