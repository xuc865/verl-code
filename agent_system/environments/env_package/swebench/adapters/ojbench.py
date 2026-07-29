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

"""OJBench -> self-repair instances (stdin/stdout, ``selfrepair_io``).

Primary local layout (``hf download He-Ren/OJBench_testdata``):
  OJBench_testdata/prompts/full.jsonl
  OJBench_testdata/{ICPC,NOI}/<problem_id>/*.in + *.out

Each jsonl row:
  * ``id``         -- problem id
  * ``prompt``     -- problem statement
  * ``dataset``    -- ICPC or NOI
  * ``language``   -- python or cpp (we keep python only)
  * ``difficulty`` -- easy / medium / hard

HF hub rows (if present) are accepted when they already expose IO lists or
numbered dicts like USACO.
"""

from __future__ import annotations

import glob
import json
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from . import make_io_instance
from .io_utils import (
    competition_problem_text,
    dict_cases_to_io_lists,
    list_cases_to_io_lists,
    stdin_stub,
)


def _load_prompt_jsonl(root: str) -> List[Dict[str, Any]]:
    candidates = [
        os.path.join(root, "prompts", "full.jsonl"),
        os.path.join(root, "full.jsonl"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        rows: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    return []


def _pair_in_out_files(problem_dir: str) -> Tuple[List[str], List[str]]:
    """Pair ``*.in`` with ``*.out`` (or ``*.ans``) under a problem directory."""
    ins = sorted(glob.glob(os.path.join(problem_dir, "*.in")))
    if not ins:
        ins = sorted(glob.glob(os.path.join(problem_dir, "input*.txt")))
    inputs: List[str] = []
    outputs: List[str] = []
    for in_path in ins:
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_candidates = [
            os.path.join(problem_dir, f"{base}.out"),
            os.path.join(problem_dir, f"{base}.ans"),
            os.path.join(problem_dir, base.replace("input", "output") + ".txt"),
        ]
        out_path = next((p for p in out_candidates if os.path.isfile(p)), None)
        if not out_path:
            # common pattern: 1.in -> 1.out
            m = re.match(r"(\d+)", base)
            if m:
                out_path = os.path.join(problem_dir, f"{m.group(1)}.out")
                if not os.path.isfile(out_path):
                    out_path = os.path.join(problem_dir, f"{m.group(1)}.ans")
        if not out_path or not os.path.isfile(out_path):
            continue
        with open(in_path, encoding="utf-8", errors="replace") as fh:
            inputs.append(fh.read())
        with open(out_path, encoding="utf-8", errors="replace") as fh:
            outputs.append(fh.read())
    return inputs, outputs


def _ojbench_roots(data_root: str) -> List[str]:
    roots: List[str] = []
    for sub in ("OJBench_testdata", "He-Ren_OJBench_testdata"):
        path = os.path.join(data_root, sub)
        if os.path.isdir(path):
            roots.append(path)
    if os.path.isfile(os.path.join(data_root, "prompts", "full.jsonl")):
        roots.append(data_root)
    return roots


def _problem_dir(data_root: str, row: Dict[str, Any]) -> Optional[str]:
    """Map a jsonl row to a staged problem directory (NOI/ICPC)."""
    pid = row.get("id", "")
    dataset = str(row.get("dataset", "ICPC")).strip()
    ds_key = "ICPC" if dataset.lower() == "icpc" else dataset
    slug = str(pid)
    if ds_key == "NOI" and str(pid).isdigit():
        slug = f"loj-{pid}"
    for root in _ojbench_roots(data_root):
        for candidate in (
            os.path.join(root, ds_key, slug),
            os.path.join(root, dataset, slug),
            os.path.join(root, ds_key, str(pid)),
            os.path.join(root, dataset, str(pid)),
        ):
            if os.path.isdir(candidate):
                return candidate
    return None


def _load_io_from_zip_dir(problem_dir: str) -> Tuple[List[str], List[str]]:
    """Read paired cases from ``init.yml`` + ``tests.zip`` (OJBench layout)."""
    init_path = os.path.join(problem_dir, "init.yml")
    zip_path = os.path.join(problem_dir, "tests.zip")
    if not os.path.isfile(init_path) or not os.path.isfile(zip_path):
        return [], []
    if yaml is None:
        return [], []
    with open(init_path, encoding="utf-8") as fh:
        meta = yaml.safe_load(fh) or {}
    cases = meta.get("test_cases") or []
    if not isinstance(cases, list):
        return [], []
    inputs: List[str] = []
    outputs: List[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for case in cases:
            if not isinstance(case, dict):
                continue
            in_name = case.get("in")
            out_name = case.get("out")
            if not in_name or not out_name:
                continue
            try:
                inputs.append(zf.read(str(in_name)).decode("utf-8", errors="replace"))
                outputs.append(zf.read(str(out_name)).decode("utf-8", errors="replace"))
            except KeyError:
                continue
    return inputs, outputs


def _load_problem_io(data_root: Optional[str], row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    if not data_root:
        return [], []
    problem_dir = _problem_dir(data_root, row)
    if problem_dir:
        inputs, outputs = _load_io_from_zip_dir(problem_dir)
        if inputs and outputs:
            return inputs, outputs
        return _pair_in_out_files(problem_dir)
    pid = str(row.get("id", ""))
    dataset = str(row.get("dataset", "ICPC")).strip()
    for sub in (
        os.path.join(data_root, "OJBench_testdata", dataset, pid),
        os.path.join(data_root, "He-Ren_OJBench_testdata", dataset, pid),
        os.path.join(data_root, dataset, pid),
    ):
        if os.path.isdir(sub):
            return _pair_in_out_files(sub)
    return [], []


def _row_builtin_io(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    if isinstance(row.get("input"), dict) and isinstance(row.get("output"), dict):
        return dict_cases_to_io_lists(row.get("input"), row.get("output"))
    for key in ("test_cases", "tests", "public_test_cases"):
        inputs, outputs = list_cases_to_io_lists(row.get(key))
        if inputs and outputs:
            return inputs, outputs
    inputs = row.get("inputs") or row.get("input_list")
    outputs = row.get("outputs") or row.get("output_list")
    if isinstance(inputs, list) and isinstance(outputs, list) and inputs and outputs:
        return [str(x) for x in inputs], [str(x) for x in outputs]
    return [], []


def to_selfrepair_instances(
    rows: List[Dict[str, Any]],
    data_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # When HF returns a single metadata row, expand from staged jsonl + test dirs.
    if len(rows) <= 1 and data_root:
        expanded = _load_prompt_jsonl(data_root)
        if not expanded:
            for sub in ("OJBench_testdata", "He-Ren_OJBench_testdata"):
                expanded = _load_prompt_jsonl(os.path.join(data_root, sub))
                if expanded:
                    break
        if expanded:
            rows = expanded

    out: List[Dict[str, Any]] = []
    for row in rows:
        lang = str(row.get("language", "python")).strip().lower()
        if lang and lang not in ("python", "py"):
            continue
        inputs, outputs = _row_builtin_io(row)
        if not inputs:
            inputs, outputs = _load_problem_io(data_root, row)
        if not inputs or not outputs or len(inputs) != len(outputs):
            continue
        pid = str(row.get("id", row.get("problem_id", f"idx{len(out)}")))
        diff = str(row.get("difficulty", "unknown")).strip().lower()
        ds = str(row.get("dataset", "ojbench")).strip().lower()
        raw = dict(row)
        raw.setdefault("difficulty", diff)
        raw.setdefault("platform", ds)
        prompt = row.get("prompt") or row.get("description") or row.get("question", "")
        out.append(make_io_instance(
            instance_id=f"ojbench__{ds}_{pid}".replace("/", "_").replace(" ", "_"),
            problem_statement=competition_problem_text(str(prompt)),
            solution_stub=stdin_stub(),
            io_tests={"inputs": inputs, "outputs": outputs},
            repo="ojbench",
            raw=raw,
        ))
    return out
