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

"""LiveCodeBench -> self-repair instances (stdin/stdout, ``selfrepair_io``).

LiveCodeBench (``livecodebench/code_generation_lite``) rows carry:
  * ``question_content``   -- problem statement
  * ``starter_code``       -- optional skeleton (call-based when present)
  * ``public_test_cases``  -- JSON list of ``{"input","output","testtype"}``
  * ``private_test_cases`` -- same shape but often zlib+base64+pickle-encoded
  * ``contest_date``       -- used for time-window contamination control

> Contamination control: LCB is versioned by time window. Filter rows by
>   ``contest_date`` to keep only problems released AFTER the model's training
>   cutoff (pass ``min_date``). Field names/encodings vary across LCB releases,
>   so re-verify ``public_test_cases`` shape for the version you pin.

We parse the **public** test cases into APPS-style ``{"inputs","outputs"}``
(robust, plain JSON). Private cases are decoded best-effort and skipped if the
encoding is not plain JSON, so the adapter never crashes on a format change.
"""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any, Dict, List, Optional

from . import make_io_instance

_STDIN_STUB = 'import sys\n\ndef main():\n    data = sys.stdin.read()\n    # BUG: placeholder reads input but prints a wrong constant.\n    #      Parse `data`, compute the answer, and print it instead.\n    print(0)\n\n\nif __name__ == "__main__":\n    main()\n'

_PROBLEM_TMPL = (
'Solve the following competitive-programming problem in `solution.py`, reading input from STDIN and printing to STDOUT. The current stub is incorrect. Edit `solution.py`, run it against the tests, read the per-case diffs, and iterate until all cases pass.\n\n----- problem -----\n{question}'
)


def _normalize_difficulty(raw: Any) -> str:
    diff = str(raw or "").strip().lower()
    if diff in ("easy", "medium", "hard"):
        return diff
    return diff or "unknown"


def _decode_cases(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict)]
    if not (isinstance(raw, str) and raw.strip()):
        return []
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return [c for c in v if isinstance(c, dict)]
    except Exception:
        pass
    try:
        blob = zlib.decompress(base64.b64decode(raw))
        v = json.loads(blob)
        if isinstance(v, list):
            return [c for c in v if isinstance(c, dict)]
    except Exception:
        pass
    return []


def to_selfrepair_instances(
    rows: List[Dict[str, Any]],
    min_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if min_date and str(row.get("contest_date", "")) < str(min_date):
            continue
        cases = _decode_cases(row.get("public_test_cases"))
        cases += _decode_cases(row.get("private_test_cases"))
        inputs = [c.get("input", "") for c in cases if "input" in c]
        outputs = [c.get("output", "") for c in cases if "input" in c]
        if not inputs or not outputs or len(inputs) != len(outputs):
            continue
        qid = str(row.get("question_id", row.get("question_title", f"idx{len(out)}")))
        starter = (row.get("starter_code") or "").strip()
        stub = (starter + "\n") if starter else _STDIN_STUB
        raw = dict(row)
        raw["difficulty"] = _normalize_difficulty(row.get("difficulty"))
        out.append(make_io_instance(
            instance_id=("livecodebench__" + qid).replace("/", "_").replace(" ", "_"),
            problem_statement=_PROBLEM_TMPL.format(question=row.get("question_content", "")),
            solution_stub=stub,
            io_tests={"inputs": inputs, "outputs": outputs},
            repo="livecodebench",
            raw=raw,
        ))
    return out
