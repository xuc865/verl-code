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
Multi-turn coding environment (gym-style, Ray-vectorized).

Each sub-environment hosts a single coding instance (CodeRL / APPS-style by
default; optional repo-level suites via docker/r2e backends). The agent issues
``edit`` / ``execute_bash`` / ``finish`` actions; the environment applies them
to a working copy of the repository, optionally runs the instance's tests, and
returns a sparse outcome reward (1.0 iff the resolved test set passes).
``grep`` / ripgrep-style search is **not** an allowed action (rejected by the
backends); inspect files with ``cat`` / ``ls`` instead.

Execution backends
------------------
The heavy, infrastructure-specific part (checking out the repo, running tests in
the right container) is isolated behind :class:`SWEBenchBackend`:

- ``LocalStubBackend``  : a dependency-free, in-memory backend that *really*
                          runs toy ``pytest`` tasks (a CPU self-repair bed) so
                          the full RL loop can be exercised without Docker.
- ``DockerBackend``     : optional Docker evaluation harness (stub: wire to yours).
- ``R2EGymBackend``     : optional integration with the R2E-Gym executor
                          (``r2egym`` + Docker); the container is the source of
                          truth and grading uses ``compute_reward``.

Pick the stack with a single ``env.swebench.benchmark`` preset (``local`` /
``apps_train_coderl`` / ``swe_bench_verified`` / ``swe_bench_lite`` /
``r2e_gym_subset`` /
``r2e_gym_lite``), which resolves the dataset + backend together (see
:func:`resolve_benchmark`). Use ``benchmark=custom`` to drive the raw
``dataset_name`` / ``split`` / ``backend`` fields directly.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gym
except Exception:  # pragma: no cover - gym is a hard dep at runtime
    gym = None

try:
    import ray
except Exception:  # pragma: no cover
    ray = None


def _load_file_edit_module():
    """Load sibling ``file_edit`` without importing ``agent_system`` (torch-free tests)."""
    import pathlib
    path = pathlib.Path(__file__).with_name("file_edit.py")
    spec = importlib.util.spec_from_file_location("_swebench_file_edit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_adapter_fn(name: str):
    """Load adapter callables without importing ``agent_system`` package root."""
    import pathlib
    path = pathlib.Path(__file__).parent / "adapters" / "__init__.py"
    spec = importlib.util.spec_from_file_location("_swebench_adapters", path)
    mod = importlib.util.module_from_spec(spec)
    # Ensure sibling adapter modules (apps.py, ...) resolve as package children.
    pkg_name = "_swebench_adapters"
    mod.__package__ = pkg_name
    mod.__path__ = [str(path.parent)]  # type: ignore[attr-defined]
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod.get_adapter(name)


_file_edit = _load_file_edit_module()
EditApplyError = _file_edit.EditApplyError
apply_edit_action = _file_edit.apply_edit_action


# --------------------------------------------------------------------------- #
# Action-space policy                                                         #
# --------------------------------------------------------------------------- #
# Allowed agent actions: edit (add/del/overwrite) / execute_bash (inspect+run) /
# finish. ``grep`` is intentionally *not* an allowed action type for DiDPO —
# credit units are code diffs, and search-in-file via grep is out of scope.
_DISALLOWED_GREP_RE = re.compile(
    r"(?:^|[;&|]\s*|\n\s*)(?:grep|egrep|fgrep|rg|ag)\b",
    re.IGNORECASE,
)

_GREP_REJECT_MSG = (
    "Action not allowed: grep/search-in-file is outside the supported action "
    "types. Allowed actions are: <edit> (overwrite/patch/insert), "
    "<execute_bash> for ls/cat/python test runs, and <finish>. "
    "Use <execute_bash>cat <file></execute_bash> to inspect a file."
)


def is_disallowed_grep_command(cmd: str) -> bool:
    """True if ``cmd`` is primarily a grep/ripgrep-style search."""
    s = (cmd or "").strip()
    if not s:
        return False
    return _DISALLOWED_GREP_RE.search(s) is not None


# --------------------------------------------------------------------------- #
# Execution backends                                                          #
# --------------------------------------------------------------------------- #
class SWEBenchBackend:
    """Interface for setting up a repo and grading a candidate patch."""

    def setup(self, instance: Dict[str, Any]) -> Dict[str, str]:
        """Return the initial (virtual) file map {path: content} for an instance."""
        raise NotImplementedError

    def run_bash(self, instance: Dict[str, Any], files: Dict[str, str], cmd: str) -> str:
        """Run a shell command and return its (truncated) stdout/stderr."""
        raise NotImplementedError

    def evaluate(self, instance: Dict[str, Any], files: Dict[str, str]) -> bool:
        """Run the instance's resolved test set; return True iff it passes."""
        raise NotImplementedError

    # -- self-repair hooks (overridable; default = no native test runner) -- #
    def is_test_command(self, cmd: str) -> bool:
        """Whether ``cmd`` should be routed to :meth:`run_tests`.

        Default ``False`` so backends without a native runner keep treating
        every command as a plain shell command (their existing behaviour).
        """
        return False

    def is_hidden_suite_command(self, cmd: str) -> bool:
        """Commands that would run the hidden grading suite (pytest, etc.)."""
        return self.is_test_command(cmd)

    def is_program_run_command(self, cmd: str) -> bool:
        """Commands that run the agent's program with agent-chosen stdin only."""
        return False

    def run_program_exec(self, instance: Dict[str, Any], files: Dict[str, str],
                         cmd: str) -> str:
        """Run ``solution.py`` with agent-provided input; no hidden test cases."""
        raise NotImplementedError

    def score_program_exec_match(
        self,
        instance: Dict[str, Any],
        files: Dict[str, str],
        cmd: str,
    ) -> float:
        """Fractional credit when an exec-mode self-test matches a hidden IO case."""
        return 0.0

    def run_tests(self, instance: Dict[str, Any], files: Dict[str, str],
                  only: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Run the test suite, returning ``(all_passed, human_readable_report)``.

        ``only`` optionally restricts execution to the named test functions
        (e.g. the instance's ``FAIL_TO_PASS`` set). Backends that support a
        native runner override this; the default raises.
        """
        raise NotImplementedError

    # -- editing & lifecycle hooks ------------------------------------- #
    def apply_edit(self, instance: Dict[str, Any], files: Dict[str, str],
                   path: str, content: str) -> None:
        """Apply a file overwrite/create.

        Default (in-memory backends): just record it in the Python ``files``
        map. Stateful container backends (R2E-Gym/Docker) override this to also
        push the new content into the live workspace, since the container -- not
        the Python dict -- is the source of truth for ``run_bash``/``evaluate``.
        """
        files[path] = content

    def task_instruction(self, instance: Dict[str, Any]) -> Optional[str]:
        """Optional backend-provided task/problem text (R2E-Gym fetches it from
        the runtime). ``None`` means: use the instance's ``problem_statement``."""
        return None

    def close(self) -> None:
        """Release any per-instance resources (e.g. tear down a container)."""
        return None


# --------------------------------------------------------------------------- #
# stdin/stdout self-repair helpers + SelfRepairIOBackend                      #
# --------------------------------------------------------------------------- #
def _normalize_output(text: str) -> str:
    return (text or "").replace("\r\n", "\n").strip()


def _io_value_to_text(val: Any) -> str:
    """Coerce APPS/LCB stdin/stdout field to a single text blob for the judge."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (list, tuple)):
        return "\n".join(_io_value_to_text(x) for x in val)
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True)
    return str(val)


def _memory_limit_preexec(max_bytes: Optional[int]):
    """Return a ``preexec_fn`` that caps address space, or ``None`` if disabled.

    Without this, a pathological ``solution.py`` can allocate terabytes within the
    wall-clock timeout and trigger Ray host-RAM OOM kills on the training workers.
    """
    if max_bytes is None or int(max_bytes) <= 0:
        return None

    limit = int(max_bytes)

    def _preexec() -> None:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # Never raise an existing hard cap; only tighten.
        new_hard = hard if hard != resource.RLIM_INFINITY else limit
        new_hard = min(new_hard, limit)
        new_soft = min(limit, new_hard)
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (new_soft, new_hard))
        except (ValueError, OSError):
            pass

    return _preexec


def _parse_memory_limit_bytes(cfg_or_mb: Any, default_mb: int = 2048) -> Optional[int]:
    """Resolve ``io_memory_limit_mb`` (0/None/negative → disabled)."""
    if isinstance(cfg_or_mb, dict):
        raw = cfg_or_mb.get("io_memory_limit_mb", default_mb)
    else:
        raw = cfg_or_mb if cfg_or_mb is not None else default_mb
    try:
        mb = int(raw)
    except (TypeError, ValueError):
        mb = default_mb
    if mb <= 0:
        return None
    return mb * 1024 * 1024


def _run_io_program(
    code: str,
    inputs: List[Any],
    outputs: List[Any],
    *,
    timeout: int = 8,
    max_cases: int = 64,
    case_indices: Optional[List[int]] = None,
    memory_limit_bytes: Optional[int] = None,
) -> Tuple[bool, int, int, str]:
    """Run ``code`` against stdin/stdout pairs in an isolated subprocess."""
    import subprocess

    if not inputs or not outputs:
        return False, 0, 0, "no io_tests on this instance"
    n = min(len(inputs), len(outputs), max(1, int(max_cases)))
    indices = list(case_indices) if case_indices is not None else list(range(n))
    indices = [i for i in indices if 0 <= i < n]
    if not indices:
        return False, 0, 0, "no io cases selected"

    preexec = _memory_limit_preexec(memory_limit_bytes)
    tmp = tempfile.mkdtemp(prefix="didpo_io_")
    sol = os.path.join(tmp, "solution.py")
    try:
        with open(sol, "w", encoding="utf-8") as fh:
            fh.write(code or "")
        results: List[Tuple[int, bool, str]] = []
        first_fail = ""
        for i in indices:
            stdin = _io_value_to_text(inputs[i])
            expected = _io_value_to_text(outputs[i])
            try:
                proc = subprocess.run(
                    [sys.executable, sol],
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmp,
                    preexec_fn=preexec,
                )
                got = proc.stdout or ""
                ok = proc.returncode == 0 and _normalize_output(got) == _normalize_output(expected)
                if not ok and not first_fail:
                    oom_hint = ""
                    if proc.returncode in (-9, 137) or "MemoryError" in (proc.stderr or ""):
                        oom_hint = " (likely hit io_memory_limit)"
                    first_fail = (
                        f"case[{i}] exit={proc.returncode}{oom_hint}\n"
                        f"  input:    {stdin!r}\n"
                        f"  expected: {_normalize_output(expected)!r}\n"
                        f"  got:      {_normalize_output(got)!r}\n"
                        f"  stderr:   {(proc.stderr or '').strip()[:500]!r}"
                    )
                results.append((i, ok, got))
            except subprocess.TimeoutExpired:
                if not first_fail:
                    first_fail = f"case[{i}] TIMEOUT after {timeout}s\n  input: {stdin!r}"
                results.append((i, False, ""))
            except Exception as e:  # noqa: BLE001
                if not first_fail:
                    first_fail = f"case[{i}] ERROR: {e!r}"
                results.append((i, False, ""))
        n_pass = sum(1 for _, ok, _ in results if ok)
        n_fail = len(results) - n_pass
        lines = [f"{'PASSED' if ok else 'FAILED'} case[{i}]" for i, ok, _ in results]
        report = "\n".join(lines) if lines else "collected 0 items"
        if first_fail:
            report += "\n\n----- first failure -----\n" + first_fail
        report += f"\n\n===== {n_pass} passed, {n_fail} failed ====="
        return (n_fail == 0 and n_pass > 0), n_pass, n_fail, report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _grade_io_tests(
    code: str,
    io: Dict[str, Any],
    *,
    timeout: int = 8,
    max_cases: int = 64,
    continuous: bool = False,
    case_indices: Optional[List[int]] = None,
    memory_limit_bytes: Optional[int] = None,
) -> Tuple[bool, int, int, str]:
    """Grade ``code`` against APPS/LCB ``io_tests`` (stdin or call-based)."""
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    if not inputs or not outputs:
        return False, 0, 0, "no io_tests on this instance"

    fn_name = io.get("fn_name")
    if fn_name:
        try:
            from verl.utils.reward_score.prime_code import compute_score

            ok, _meta = compute_score(code, dict(io), continuous=continuous)
            n = min(len(inputs), len(outputs), max_cases)
            if continuous and isinstance(ok, float):
                n_pass = max(0, min(n, int(round(float(ok) * n))))
                n_fail = n - n_pass
                all_passed = float(ok) >= 1.0
                report = (
                    f"prime_code grader (fn_name={fn_name!r}): "
                    f"{float(ok):.3f} pass rate over {n} case(s)"
                )
                return all_passed, n_pass, n_fail, report
            passed = (ok is True) or (isinstance(ok, (int, float)) and float(ok) >= 1.0)
            n_pass, n_fail = (n, 0) if passed else (0, n)
            report = f"prime_code grader (fn_name={fn_name!r}): {'PASS' if passed else 'FAIL'}"
            return passed, n_pass, n_fail, report
        except Exception as e:  # noqa: BLE001
            return False, 0, min(len(inputs), max_cases), f"prime_code grader failed: {e!r}"

    return _run_io_program(
        code, inputs, outputs, timeout=timeout, max_cases=max_cases,
        case_indices=case_indices, memory_limit_bytes=memory_limit_bytes,
    )


def _parse_agent_stdin_from_cmd(cmd: str) -> str:
    """Extract stdin piped into ``python solution.py`` from a one-line shell command."""
    cmd = cmd.strip()
    patterns = (
        r"echo\s+-n\s+(['\"])(.+?)\1\s*\|",
        r"echo\s+(['\"])(.+?)\1\s*\|",
        r"printf\s+%s\s+(['\"])(.+?)\1\s*\|",
        r"printf\s+(['\"])(.+?)\1\s*\|",
    )
    for pat in patterns:
        m = re.search(pat, cmd, re.DOTALL)
        if m:
            return m.group(2)
    # Multi-line printf '%s\n' 'a' 'b' | python solution.py
    m = re.search(
        r"printf\s+(['\"])%s\\n\1\s+((?:(['\"])(?:.*?)\3\s*)+)\|",
        cmd,
        re.DOTALL,
    )
    if m:
        parts = re.findall(r"(['\"])(.*?)\1", m.group(2), re.DOTALL)
        if parts:
            return "\n".join(p[1] for p in parts) + "\n"
    return ""


def _format_program_exec_report(cmd: str, exit_code: int,
                                stdout: str, stderr: str) -> str:
    """Human-readable exec feedback without hidden expected outputs."""
    out = (stdout or "").strip()
    err = (stderr or "").strip()
    lines = [
        "----- program execution (your input only; hidden tests NOT run) -----",
        f"$ {cmd.strip()}",
        f"exit_code: {exit_code}",
        "stdout:",
        out if out else "(empty)",
        "stderr:",
        err if err else "(empty)",
        "----- end execution (no pass/fail vs hidden suite; fix crashes, then <finish>) -----",
    ]
    return "\n".join(lines)


def _run_agent_program_workspace(
    files: Dict[str, str],
    target: str,
    stdin: str,
    *,
    timeout: int,
    memory_limit_bytes: Optional[int] = None,
) -> Tuple[int, str, str]:
    """Run ``target`` in an isolated temp workspace; return (exit_code, stdout, stderr)."""
    import subprocess

    preexec = _memory_limit_preexec(memory_limit_bytes)
    tmp = tempfile.mkdtemp(prefix="didpo_exec_")
    try:
        for rel_path, content in files.items():
            full = os.path.join(tmp, rel_path)
            os.makedirs(os.path.dirname(full) or tmp, exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(content)
        sol = os.path.join(tmp, target)
        if not os.path.isfile(sol):
            return 127, "", f"error: {target!r} not found in workspace"
        proc = subprocess.run(
            [sys.executable, sol],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmp,
            preexec_fn=preexec,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"TIMEOUT after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return 1, "", f"ERROR launching program: {e!r}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class SelfRepairIOBackend(SWEBenchBackend):
    """In-memory stdin/stdout grader for APPS / LiveCodeBench / OJ-style tasks."""

    _HIDDEN_SUITE_TOKENS = ("pytest", "py.test")
    _PROGRAM_RUN_MARKERS = (
        "python solution.py",
        "python3 solution.py",
        "python ./solution.py",
        "python3 ./solution.py",
    )

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self._grader = cfg.get("grader", "subprocess")
        self._timeout = int(cfg.get("io_timeout", 8))
        self._max_cases = int(cfg.get("io_max_cases", 64))
        self._exec_timeout = int(cfg.get("exec_timeout", self._timeout))
        # Cap per-case solution.py RSS (default 2GiB). 0 disables. Prevents host OOM.
        self._memory_limit_bytes = _parse_memory_limit_bytes(cfg)

    @staticmethod
    def _io(instance: Dict[str, Any]) -> Dict[str, Any]:
        return dict(instance.get("io_tests") or {})

    @staticmethod
    def _solution_code(instance: Dict[str, Any], files: Dict[str, str]) -> str:
        target = instance.get("target_file", "solution.py")
        return files.get(target, "") or ""

    def setup(self, instance: Dict[str, Any]) -> Dict[str, str]:
        return dict(instance.get("initial_files", {}))

    def is_hidden_suite_command(self, cmd: str) -> bool:
        c = cmd.strip().lower()
        if any(tok in c for tok in self._HIDDEN_SUITE_TOKENS):
            return True
        if "python -m" in c and any(x in c for x in ("pytest", "unittest", "test")):
            return True
        return False

    def is_program_run_command(self, cmd: str) -> bool:
        c = cmd.strip().lower()
        return any(m in c for m in self._PROGRAM_RUN_MARKERS)

    def is_test_command(self, cmd: str) -> bool:
        return self.is_hidden_suite_command(cmd) or self.is_program_run_command(cmd)

    def run_tests(self, instance: Dict[str, Any], files: Dict[str, str],
                  only: Optional[List[str]] = None) -> Tuple[bool, str]:
        del only  # IO suites are not pytest-filtered by name
        io = self._io(instance)
        code = self._solution_code(instance, files)
        all_passed, _, _, report = _grade_io_tests(
            code, io, timeout=self._timeout, max_cases=self._max_cases,
            case_indices=instance.get("_io_case_indices"),
            memory_limit_bytes=self._memory_limit_bytes,
        )
        return all_passed, report

    def evaluate(self, instance: Dict[str, Any], files: Dict[str, str]) -> bool:
        passed, _ = self.run_tests(instance, files)
        return bool(passed)

    def score(self, instance: Dict[str, Any], files: Dict[str, str]) -> float:
        io = self._io(instance)
        if not (io.get("inputs") or []):
            return 0.0
        code = self._solution_code(instance, files)
        if io.get("fn_name"):
            try:
                from verl.utils.reward_score.prime_code import compute_score
                ok, _meta = compute_score(code, dict(io), continuous=True)
                if isinstance(ok, float):
                    return max(0.0, min(1.0, float(ok)))
                return 1.0 if ok else 0.0
            except Exception:  # noqa: BLE001
                return 0.0
        _, n_pass, n_fail, _ = _grade_io_tests(
            code, io, timeout=self._timeout, max_cases=self._max_cases,
            case_indices=instance.get("_io_case_indices"),
            memory_limit_bytes=self._memory_limit_bytes,
        )
        total = n_pass + n_fail
        return (n_pass / total) if total else 0.0

    def run_program_exec(self, instance: Dict[str, Any], files: Dict[str, str],
                         cmd: str) -> str:
        target = instance.get("target_file", "solution.py")
        stdin = _parse_agent_stdin_from_cmd(cmd)
        code = self._solution_code(instance, files)
        if not code.strip():
            return _format_program_exec_report(cmd, 127, "", f"{target} is empty")
        try:
            import ast
            ast.parse(code, filename=target)
        except SyntaxError as e:
            return _format_program_exec_report(cmd, 1, "", f"SyntaxError: {e}")
        exit_code, stdout, stderr = _run_agent_program_workspace(
            files, target, stdin, timeout=self._exec_timeout,
            memory_limit_bytes=self._memory_limit_bytes,
        )
        return _format_program_exec_report(cmd, exit_code, stdout, stderr)

    def score_program_exec_match(
        self,
        instance: Dict[str, Any],
        files: Dict[str, str],
        cmd: str,
    ) -> float:
        """Reward agent-chosen stdin runs that match a hidden IO pair (exec mode)."""
        io = self._io(instance)
        if io.get("fn_name"):
            return 0.0
        agent_stdin = _parse_agent_stdin_from_cmd(cmd)
        if not str(agent_stdin or "").strip():
            return 0.0
        target = instance.get("target_file", "solution.py")
        code = self._solution_code(instance, files)
        if not code.strip():
            return 0.0
        agent_in = _io_value_to_text(agent_stdin).strip()
        inputs = io.get("inputs") or []
        outputs = io.get("outputs") or []
        for inp, out in zip(inputs, outputs):
            if _io_value_to_text(inp).strip() != agent_in:
                continue
            exit_code, stdout, _stderr = _run_agent_program_workspace(
                files, target, agent_stdin, timeout=self._exec_timeout,
                memory_limit_bytes=self._memory_limit_bytes,
            )
            if exit_code != 0:
                return 0.0
            if _normalize_output(stdout) == _normalize_output(_io_value_to_text(out)):
                return 1.0
            return 0.0
        exit_code, _stdout, _stderr = _run_agent_program_workspace(
            files, target, agent_stdin, timeout=self._exec_timeout,
            memory_limit_bytes=self._memory_limit_bytes,
        )
        return 0.1 if exit_code == 0 else 0.0

    def run_bash(self, instance: Dict[str, Any], files: Dict[str, str], cmd: str) -> str:
        cmd = cmd.strip()
        if is_disallowed_grep_command(cmd):
            return _GREP_REJECT_MSG
        if self.is_hidden_suite_command(cmd):
            _, report = self.run_tests(instance, files)
            return report
        if self.is_program_run_command(cmd):
            return (
                "Hidden grading suite is not run during repair. "
                "Use test_feedback_mode=exec to run your program with your own stdin "
                "(stdout/stderr only), or <finish> for full hidden grading."
            )
        if cmd.startswith("cat "):
            path = cmd[4:].strip()
            return files.get(path, f"cat: {path}: No such file")
        if cmd.startswith("ls"):
            return "\n".join(sorted(files.keys()))
        return f"$ {cmd}\n(selfrepair_io: command executed, no real shell)"


class MixedBackend(SWEBenchBackend):
    """Route each instance to the correct sub-backend by instance shape.

    Used by the ``light_eval`` preset (APPS IO + HumanEval/MBPP assert) and any
    merged coding benchmark pool that mixes IO and pytest instances.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg or {})
        self._io = SelfRepairIOBackend(self.cfg)
        self._stub = LocalStubBackend()
        self._r2e: Optional[R2EGymBackend] = None

    def _pick(self, instance: Dict[str, Any]) -> SWEBenchBackend:
        if instance.get("io_tests"):
            return self._io
        raw = instance.get("_raw") or {}
        if raw.get("docker_image") or raw.get("image_name"):
            if self._r2e is None:
                self._r2e = R2EGymBackend(self.cfg)
            return self._r2e
        return self._stub

    def setup(self, instance: Dict[str, Any]) -> Dict[str, str]:
        return self._pick(instance).setup(instance)

    def run_bash(self, instance: Dict[str, Any], files: Dict[str, str], cmd: str) -> str:
        return self._pick(instance).run_bash(instance, files, cmd)

    def evaluate(self, instance: Dict[str, Any], files: Dict[str, str]) -> bool:
        return self._pick(instance).evaluate(instance, files)

    def score(self, instance: Dict[str, Any], files: Dict[str, str]) -> float:
        return self._pick(instance).score(instance, files)

    def is_test_command(self, cmd: str) -> bool:
        return self.is_hidden_suite_command(cmd) or self.is_program_run_command(cmd)

    def is_hidden_suite_command(self, cmd: str) -> bool:
        return self._io.is_hidden_suite_command(cmd) or self._stub.is_hidden_suite_command(cmd)

    def is_program_run_command(self, cmd: str) -> bool:
        return self._io.is_program_run_command(cmd)

    def run_program_exec(self, instance: Dict[str, Any], files: Dict[str, str],
                         cmd: str) -> str:
        return self._pick(instance).run_program_exec(instance, files, cmd)

    def score_program_exec_match(
        self,
        instance: Dict[str, Any],
        files: Dict[str, str],
        cmd: str,
    ) -> float:
        return self._pick(instance).score_program_exec_match(instance, files, cmd)

    def run_tests(self, instance: Dict[str, Any], files: Dict[str, str],
                  only: Optional[List[str]] = None) -> Tuple[bool, str]:
        return self._pick(instance).run_tests(instance, files, only=only)

    def apply_edit(self, instance: Dict[str, Any], files: Dict[str, str],
                   path: str, content: str) -> None:
        self._pick(instance).apply_edit(instance, files, path, content)

    def task_instruction(self, instance: Dict[str, Any]) -> Optional[str]:
        return self._pick(instance).task_instruction(instance)

    def close(self) -> None:
        self._stub.close()
        self._io.close()
        if self._r2e is not None:
            self._r2e.close()


def _parse_test_names(raw: Any) -> List[str]:
    """Normalize a SWE-bench ``FAIL_TO_PASS`` field into a list of test names.

    Accepts a JSON-encoded list (``'["test_a", "test_b"]'``), a real list, or a
    comma-separated string. Real SWE-bench names may be node ids such as
    ``tests/test_x.py::test_a``; we keep them verbatim and match leniently.
    """
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            v = json.loads(raw)
            if isinstance(v, list):
                return [str(x) for x in v]
        except Exception:  # noqa: BLE001
            return [s.strip() for s in raw.split(",") if s.strip()]
    return []


def _name_selected(fn_name: str, only: Optional[List[str]]) -> bool:
    """True if ``fn_name`` is selected by the ``only`` filter (None = all)."""
    if only is None:
        return True
    return any(o == fn_name or o.endswith("::" + fn_name) for o in only)


def _run_pytest_like(files: Dict[str, str],
                     only: Optional[List[str]] = None) -> Tuple[bool, int, int, str]:
    """Really execute the ``test_*.py`` files in ``files`` in a throwaway dir.

    The file map is materialized to a fresh temp directory, which is put on
    ``sys.path`` so the test modules can ``import`` the (edited) sources. Each
    ``test_*`` function is run; assertion errors are captured as failures. The
    temp dir and any modules it created are purged afterwards, so every call is
    a clean re-import that reflects the agent's *current* edits.

    Returns ``(all_passed, n_pass, n_fail, report)``.
    """
    test_files = sorted(
        p for p in files
        if os.path.basename(p).startswith("test_") and p.endswith(".py"))
    if not test_files:
        return False, 0, 0, "collected 0 items (no test_*.py files found)"

    tmp = tempfile.mkdtemp(prefix="didpo_selfrepair_")
    for path, content in files.items():
        abspath = os.path.join(tmp, path)
        os.makedirs(os.path.dirname(abspath) or tmp, exist_ok=True)
        with open(abspath, "w") as fh:
            fh.write(content)

    def _purge_tmp_modules() -> None:
        for mod_name in list(sys.modules):
            mod = sys.modules.get(mod_name)
            mod_file = getattr(mod, "__file__", None)
            if mod_file and mod_file.startswith(tmp):
                del sys.modules[mod_name]

    results: List[Tuple[str, bool, str]] = []  # (name, passed, traceback)
    sys.path.insert(0, tmp)
    try:
        importlib.invalidate_caches()
        for tf in test_files:
            modname = os.path.splitext(os.path.basename(tf))[0]
            _purge_tmp_modules()
            sys.modules.pop(modname, None)
            try:
                mod = importlib.import_module(modname)
            except Exception:  # noqa: BLE001 - collection error
                results.append((f"{modname} (import)", False, traceback.format_exc()))
                continue
            fns = sorted(
                (n for n in dir(mod)
                 if n.startswith("test_") and callable(getattr(mod, n))))
            for n in fns:
                if not _name_selected(n, only):
                    continue
                try:
                    getattr(mod, n)()
                    results.append((n, True, ""))
                except Exception:  # noqa: BLE001 - test failure
                    results.append((n, False, traceback.format_exc()))
    finally:
        try:
            sys.path.remove(tmp)
        except ValueError:
            pass
        _purge_tmp_modules()
        shutil.rmtree(tmp, ignore_errors=True)

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    lines = []
    for n, ok, _ in results:
        lines.append(f"{'PASSED' if ok else 'FAILED'} {n}")
    report = "\n".join(lines) if lines else "collected 0 items (no matching tests)"
    first_fail = next((tb for _, ok, tb in results if not ok and tb), "")
    if first_fail:
        report += "\n\n----- first failure traceback -----\n" + first_fail.strip()
    report += f"\n\n===== {n_pass} passed, {n_fail} failed ====="
    all_passed = n_fail == 0 and n_pass > 0
    return all_passed, n_pass, n_fail, report


class LocalStubBackend(SWEBenchBackend):
    """In-memory, Docker-free backend used as a real *self-repair* test bed.

    Unlike a transparent keyword grader, this backend **actually executes** the
    instance's toy tests against the agent's edited file map (see
    :func:`_run_pytest_like`). The agent can therefore run ``pytest``, observe a
    real failing traceback, edit the source, and re-run -- the full self-repair
    loop -- entirely on CPU and with no external infrastructure. Toy sources are
    trusted, so in-process ``exec`` carries negligible sandbox risk.
    """

    _TEST_TOKENS = ("pytest", "py.test", "python -m unittest", "unittest")

    def setup(self, instance: Dict[str, Any]) -> Dict[str, str]:
        return dict(instance.get("initial_files", {}))

    def is_hidden_suite_command(self, cmd: str) -> bool:
        c = cmd.strip()
        return any(tok in c for tok in self._TEST_TOKENS)

    def is_test_command(self, cmd: str) -> bool:
        return self.is_hidden_suite_command(cmd)

    def run_program_exec(self, instance: Dict[str, Any], files: Dict[str, str],
                         cmd: str) -> str:
        return "program execution is not supported on local_stub instances"

    def run_tests(self, instance: Dict[str, Any], files: Dict[str, str],
                  only: Optional[List[str]] = None) -> Tuple[bool, str]:
        all_passed, _, _, report = _run_pytest_like(files, only=only)
        return all_passed, report

    def run_bash(self, instance: Dict[str, Any], files: Dict[str, str], cmd: str) -> str:
        # Emulate a few read-only commands so the agent can "inspect" the repo.
        # grep is intentionally rejected — not part of the DiDPO action space.
        cmd = cmd.strip()
        if is_disallowed_grep_command(cmd):
            return _GREP_REJECT_MSG
        if self.is_test_command(cmd):
            # Run the full suite so the agent sees every failure to repair.
            _, report = self.run_tests(instance, files, only=None)
            return report
        if cmd.startswith("cat "):
            path = cmd[4:].strip()
            return files.get(path, f"cat: {path}: No such file")
        if cmd.startswith("ls"):
            return "\n".join(sorted(files.keys()))
        return f"$ {cmd}\n(local_stub: command executed, no real shell)"

    def evaluate(self, instance: Dict[str, Any], files: Dict[str, str]) -> bool:
        # Grade by really running the resolved (FAIL_TO_PASS) test set.
        only = _parse_test_names(instance.get("FAIL_TO_PASS", "")) or None
        all_passed, _ = self.run_tests(instance, files, only=only)
        return all_passed

    def score(self, instance: Dict[str, Any], files: Dict[str, str]) -> float:
        return 1.0 if self.evaluate(instance, files) else 0.0


class DockerBackend(SWEBenchBackend):  # pragma: no cover - infra dependent
    """Run instances in official SWE-bench images. Stub to be wired to a harness."""

    def __init__(self, image_prefix: str, timeout: int = 600):
        self.image_prefix = image_prefix
        self.timeout = timeout

    def setup(self, instance):
        raise NotImplementedError(
            "DockerBackend.setup: wire this to your SWE-bench evaluation harness "
            "(checkout base_commit, apply test_patch, mount working dir).")

    def run_bash(self, instance, files, cmd):
        raise NotImplementedError("DockerBackend.run_bash: exec inside the container.")

    def evaluate(self, instance, files):
        raise NotImplementedError(
            "DockerBackend.evaluate: apply the candidate patch and run "
            "FAIL_TO_PASS / PASS_TO_PASS test sets in the image.")


class R2EGymBackend(SWEBenchBackend):  # pragma: no cover - infra dependent
    """Real integration with the R2E-Gym executor (``r2egym`` + Docker).

    Maps this env's file-map abstraction onto R2E-Gym's *stateful container*
    model. One :class:`r2egym...RepoEnv` (hence one Docker container) is held per
    instance; the container is the source of truth, so edits are pushed into it
    and grading delegates to the executor's unit-test-based reward.

    Real R2E-Gym API used (see r2egym/agenthub/environment/env.py):

        from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
        env = RepoEnv(EnvArgs(ds=<dataset row dict>))
        env.reset()
        out, code = env.runtime.run(bash_cmd, timeout=...)
        reward = env.compute_reward()           # runtime._calculate_reward
        instr = env.get_task_instruction()
        env.close()

    Config keys (``env.swebench``): ``r2e_backend`` (default ``"docker"``),
    ``step_timeout`` (90), ``reward_timeout`` (300), ``verbose`` (False).
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self._backend = cfg.get("r2e_backend", "docker")
        self._step_timeout = int(cfg.get("step_timeout", 90))
        self._reward_timeout = int(cfg.get("reward_timeout", 300))
        self._verbose = bool(cfg.get("verbose", False))
        self._env = None  # live RepoEnv (one Docker container)
        self._instruction: Optional[str] = None

    # -- helpers --------------------------------------------------------- #
    @staticmethod
    def _entry(instance: Dict[str, Any]) -> Dict[str, Any]:
        """The raw dataset row R2E-Gym expects as ``EnvArgs(ds=...)``."""
        raw = instance.get("_raw", instance)
        if not isinstance(raw, dict):
            raw = dict(instance)
        else:
            raw = dict(raw)
        if not raw.get("docker_image") and not raw.get("image_name"):
            iid = raw.get("instance_id") or instance.get("instance_id")
            if iid and "__" in str(iid):
                raw["image_name"] = get_swebench_docker_image_name(raw)
        if raw.get("docker_image"):
            raw["docker_image"] = normalize_docker_image_ref(str(raw["docker_image"]))
        if raw.get("image_name"):
            raw["image_name"] = normalize_docker_image_ref(str(raw["image_name"]))
        return raw

    def _run(self, cmd: str, timeout: Optional[int] = None) -> Tuple[str, int]:
        out, code = self._env.runtime.run(cmd, timeout=timeout or self._step_timeout)
        return out, code

    # -- lifecycle ------------------------------------------------------- #
    def setup(self, instance: Dict[str, Any]) -> Dict[str, str]:
        from r2egym.agenthub.environment.env import EnvArgs, RepoEnv

        self.close()  # tear down any previous container first
        env_args = EnvArgs(ds=self._entry(instance))
        self._env = RepoEnv(env_args, backend=self._backend, verbose=self._verbose,
                            step_timeout=self._step_timeout,
                            reward_timeout=self._reward_timeout)
        self._env.reset()
        try:
            self._instruction = self._env.get_task_instruction()
        except Exception:  # noqa: BLE001
            self._instruction = instance.get("problem_statement") or None
        # Container holds the real files; the Python map starts empty and grows
        # as the agent edits (used only for the repo-view display).
        return {}

    def task_instruction(self, instance: Dict[str, Any]) -> Optional[str]:
        return self._instruction

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
            self._env = None

    # -- editing & execution -------------------------------------------- #
    def apply_edit(self, instance: Dict[str, Any], files: Dict[str, str],
                   path: str, content: str) -> None:
        import base64
        import shlex

        # base64 round-trip writes arbitrary content safely (any quotes/EOF/etc).
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        qpath = shlex.quote(path)
        self._run(f"mkdir -p \"$(dirname {qpath})\" 2>/dev/null; "
                  f"printf %s {shlex.quote(b64)} | base64 -d > {qpath}")
        files[path] = content  # keep repo-view in sync

    def run_bash(self, instance: Dict[str, Any], files: Dict[str, str], cmd: str) -> str:
        if is_disallowed_grep_command(cmd):
            return _GREP_REJECT_MSG
        out, code = self._run(cmd)
        return out if code == 0 else f"{out}\n[exit code {code}]"

    def evaluate(self, instance: Dict[str, Any], files: Dict[str, str]) -> bool:
        try:
            reward = self._env.compute_reward(timeout=self._reward_timeout)
        except Exception:  # noqa: BLE001
            return False
        try:
            return float(reward) > 0.0
        except (TypeError, ValueError):
            return bool(reward)


# --------------------------------------------------------------------------- #
# Benchmark presets (one-line dataset switching)                              #
# --------------------------------------------------------------------------- #
# Each preset maps a friendly key to the dataset id + split + the execution
# backend that can actually grade it. This is the single knob that flips the
# whole stack between the local CPU self-repair bed and a real Docker benchmark.
#
# IMPORTANT: SWE-bench Verified/Lite here point at the **R2E-Gym mirrors**
# (``R2E-Gym/SWE-Bench-*``), whose rows carry the ``docker_image`` + test
# metadata that the ``r2e_gym`` executor needs. The official
# ``princeton-nlp/SWE-bench_Verified`` rows do NOT carry that and are not
# runnable by :class:`R2EGymBackend` -- so we deliberately route through the
# R2E-Gym datasets to keep "switch benchmark" a safe, single edit.
_BENCHMARK_PRESETS: Dict[str, Dict[str, Any]] = {
    # CPU-only, Docker-free self-repair bed (synthetic toy bugs).
    "local":              {"dataset_name": "",                          "split": "test",  "backend": "local_stub"},
    # Real eval benchmarks (R2E-Gym executor + Docker).
    "swe_bench_verified": {"dataset_name": "R2E-Gym/SWE-Bench-Verified", "split": "test",  "backend": "r2e_gym"},
    "swe_bench_lite":     {"dataset_name": "R2E-Gym/SWE-Bench-Lite",     "split": "test",  "backend": "r2e_gym"},
    # Real training benchmarks (R2E-Gym executor + Docker).
    "r2e_gym_subset":     {"dataset_name": "R2E-Gym/R2E-Gym-Subset",     "split": "train", "backend": "r2e_gym"},
    "r2e_gym_lite":       {"dataset_name": "R2E-Gym/R2E-Gym-Lite",       "split": "train", "backend": "r2e_gym"},
    # stdin/stdout coding (CPU selfrepair_io). Local jsonl under data_root preferred.
    "apps":               {"dataset_name": "codeparrot/apps",            "split": "test",  "backend": "selfrepair_io", "adapter": "apps", "dataset_config": "all"},
    "apps_train":         {"dataset_name": "codeparrot/apps",            "split": "train", "backend": "selfrepair_io", "adapter": "apps", "dataset_config": "all"},
    # APPS train + optional PRIME apps-only extras (mt8_v4 / CodeRL+).
    # Build extras: scripts/build_apps_train_coderl.py -> codeparrot_apps_prime_extra/
    "apps_train_coderl": {
        "backend": "mixed",
        "merge_benchmarks": [
            {"benchmark": "apps_train"},
            {"benchmark": "apps_prime_extra"},
        ],
    },
    "apps_prime_extra": {
        "dataset_name": "codeparrot/apps_prime_extra",
        "split": "train",
        "backend": "selfrepair_io",
        "adapter": "apps",
        "optional_empty": True,
    },
}


def resolve_benchmark(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a ``benchmark`` preset into concrete dataset/split/backend.

    When ``benchmark`` is set (and not ``"custom"``), the matching preset is
    authoritative for ``dataset_name`` / ``split`` / ``backend`` -- one knob
    flips the whole stack from the local self-repair bed to a real benchmark.
    When it is absent or ``"custom"``, the raw fields are used verbatim (fully
    backward compatible). Returns a new dict; the input is never mutated.
    """
    cfg = dict(cfg or {})
    bench = cfg.get("benchmark")
    if not bench or str(bench).strip().lower() == "custom":
        return cfg
    key = str(bench).strip().lower()
    if key not in _BENCHMARK_PRESETS:
        raise ValueError(
            f"Unknown benchmark '{bench}'. Known presets: "
            f"{sorted(_BENCHMARK_PRESETS)} (or 'custom' to use raw fields).")
    cfg.update(_BENCHMARK_PRESETS[key])
    return cfg


def make_backend(cfg: Dict[str, Any]) -> SWEBenchBackend:
    cfg = resolve_benchmark(cfg)
    backend = (cfg or {}).get("backend", "local_stub")
    if backend == "local_stub":
        return LocalStubBackend()
    if backend == "selfrepair_io":
        return SelfRepairIOBackend(cfg)
    if backend == "mixed":
        return MixedBackend(cfg)
    if backend == "docker":
        return DockerBackend(cfg.get("image_prefix", ""), cfg.get("timeout", 600))
    if backend == "r2e_gym":
        return R2EGymBackend(cfg)
    raise ValueError(f"Unknown swebench backend: {backend}")


# --------------------------------------------------------------------------- #
# Dataset loading                                                             #
# --------------------------------------------------------------------------- #
def _hf_datasets_cache_dir(cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve HuggingFace datasets cache (mirrors mini-swe-agent swebench.py)."""
    cfg = cfg or {}
    for key in ("hf_datasets_cache", "data_root"):
        val = cfg.get(key) or os.environ.get(
            "HF_DATASETS_CACHE" if key == "hf_datasets_cache" else "SWEBENCH_DATA_ROOT")
        if val:
            return str(val)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "datasets")
    return None


def _local_dataset_dir(name: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Map HF dataset id to a staged local directory under data_root.

    ``codeparrot/apps`` -> ``$data_root/codeparrot_apps`` (mt8_v4 layout).
    """
    name = (name or "").strip()
    if not name:
        return None
    cfg = cfg or {}
    root = cfg.get("data_root") or os.environ.get("SWEBENCH_DATA_ROOT") or ""
    root = str(root).strip()
    if not root:
        return None
    dirname = name.replace("/", "_")
    for cand in (
        os.path.join(root, dirname),
        os.path.join(root, name.split("/")[-1]),
    ):
        if os.path.isdir(cand):
            return cand
    return None


def _load_rows_local(
    local_dir: str,
    split: str,
    dataset_config: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Load staged jsonl/parquet; return None if directory has no usable split files."""
    del dataset_config  # reserved for HF config parity
    import glob

    jsonl = os.path.join(local_dir, f"{split}.jsonl")
    if os.path.isfile(jsonl):
        rows: List[Dict[str, Any]] = []
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    patterns = [
        os.path.join(local_dir, f"{split}-*.parquet"),
        os.path.join(local_dir, f"{split}.parquet"),
        os.path.join(local_dir, "data", f"{split}-*.parquet"),
        os.path.join(local_dir, "**", f"{split}-*.parquet"),
    ]
    files: List[str] = []
    for pat in patterns:
        files = sorted(glob.glob(pat, recursive=True))
        if files:
            break
    if not files:
        return None
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=files, split="train")
    return list(ds)


def _load_single_benchmark_instances(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load instances for one benchmark preset (no merge, no top-level selection)."""
    cfg = resolve_benchmark(cfg)
    name = (cfg or {}).get("dataset_name", "")
    split = (cfg or {}).get("split", "test")
    subset = (cfg or {}).get("subset_size", None)
    adapter = (cfg or {}).get("adapter") or ""
    dataset_config = (cfg or {}).get("dataset_config") or None

    instances: List[Dict[str, Any]] = []
    if not name:
        # No dataset id (e.g. benchmark=local) -> CPU self-repair bed.
        instances = _synthetic_instances()
    else:
        rows: Optional[List[Dict[str, Any]]] = None
        # (1) Prefer staged local data (offline train nodes).
        local_dir = _local_dataset_dir(name, cfg)
        if local_dir:
            try:
                rows = _load_rows_local(local_dir, split, dataset_config)
                if rows is not None:
                    print(f"[swebench] loaded {len(rows)} rows for '{name}' "
                          f"from local dir '{local_dir}'.")
            except Exception as e:  # noqa: BLE001
                print(f"[swebench] local load for '{name}' at '{local_dir}' "
                      f"failed ({e}); falling back to the HuggingFace hub.")
                rows = None
            if rows is None and cfg.get("optional_empty"):
                print(f"[swebench] optional local dataset '{name}' empty at "
                      f"'{local_dir}' — skip (no hub fetch).")
                rows = []
        elif cfg.get("optional_empty"):
            print(f"[swebench] optional local dataset '{name}' not staged — skip.")
            rows = []
        # (2) Fall back to the HuggingFace hub / cache.
        if rows is None:
            try:
                from datasets import load_dataset
                cache_dir = _hf_datasets_cache_dir(cfg)
                load_kwargs: Dict[str, Any] = {}
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                    load_kwargs["cache_dir"] = cache_dir
                if dataset_config:
                    ds = load_dataset(name, dataset_config, split=split, **load_kwargs)
                else:
                    ds = load_dataset(name, split=split, **load_kwargs)
                rows = list(ds)
            except Exception as e:  # noqa: BLE001
                if cfg.get("optional_empty"):
                    print(f"[swebench] optional dataset '{name}' hub load failed "
                          f"({e}) — skip.")
                    rows = []
                else:
                    print(f"[swebench] could not load dataset '{name}' ({e}); "
                          f"using synthetic stub set.")
                    rows = None

        if rows is None:
            instances = _synthetic_instances()
        elif not rows:
            bench = (cfg.get("benchmark") or "").strip().lower()
            if cfg.get("optional_empty"):
                print(f"[swebench] optional benchmark '{bench}' has 0 instances — skip.")
                return []
            if bench not in ("", "local") and name:
                raise RuntimeError(
                    f"Dataset '{name}' split '{split}' produced zero instances for "
                    f"benchmark '{bench}'.")
            print(f"[swebench] dataset '{name}' split '{split}' was empty; "
                  f"using synthetic stub set.")
            instances = _synthetic_instances()
        elif adapter:
            instances = _get_adapter_fn(str(adapter))(rows)
        else:
            instances = [_normalize_instance(ex) for ex in rows]

    if cfg.get("skip_difficulty_filter"):
        select_cfg = dict(cfg)
        select_cfg["difficulty_filter"] = ""
        instances = apply_instance_selection(instances, select_cfg)
    else:
        instances = apply_instance_selection(instances, cfg)

    # subset_size is also applied inside apply_instance_selection when present;
    # keep a final clamp for callers that only set subset on the raw list.
    if subset is not None and str(subset).strip() != "" and not cfg.get("instance_slice"):
        n = int(subset)
        if n >= 0:
            instances = instances[:n]
    return instances


def normalize_docker_image_ref(image: str) -> str:
    """Fully qualify short refs (``user/repo:tag``) so podman does not prompt for registry."""
    image = (image or "").strip()
    if not image:
        return image
    head = image.split("/", 1)[0]
    if head in {"localhost"} or "." in head or ":" in head:
        return image
    return f"docker.io/{image}"


def get_swebench_docker_image_name(instance: Dict[str, Any]) -> str:
    """Official SWE-bench eval image name from an instance row (mini-swe-agent).

    Used as fallback when a row has ``instance_id`` but no ``docker_image`` /
    ``image_name`` (e.g. Princeton HF rows). R2E-Gym rows should already carry
    ``docker_image`` and never need this.
    """
    image_name = instance.get("image_name") or instance.get("docker_image")
    if image_name:
        return normalize_docker_image_ref(str(image_name))
    iid = instance.get("instance_id") or instance.get("id") or "unknown"
    id_docker_compatible = str(iid).replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()


# APPS / similar rows store ``difficulty`` on the original HF row under ``_raw``.
_DIFFICULTY_RANK = {
    "introductory": 0,
    "interview": 1,
    "competition": 2,
}


def instance_difficulty(inst: Dict[str, Any]) -> str:
    raw = inst.get("_raw") or {}
    for key in ("difficulty", "problem_level", "level"):
        val = str(raw.get(key, "")).strip().lower()
        if val and val not in ("unknown", "none", ""):
            return val
    return "unknown"


def parse_difficulty_allowlist(spec: Any) -> Optional[set]:
    """Parse ``introductory`` or ``introductory,interview`` into a lowercase set."""
    if spec is None:
        return None
    text = str(spec).strip()
    if not text or text.lower() in ("all", "*"):
        return None
    return {p.strip().lower() for p in text.split(",") if p.strip()}


def parse_curriculum_schedule(spec: Any) -> Optional[List[Tuple[int, Optional[set]]]]:
    """Parse ``25:introductory,55:introductory+interview,80:all`` (epoch upper bounds)."""
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    stages: List[Tuple[int, Optional[set]]] = []
    for part in text.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        until_s, diff_part = part.split(":", 1)
        until = int(until_s.strip())
        diff_part = diff_part.strip().lower()
        if diff_part in ("all", "*"):
            allowed = None
        elif "+" in diff_part:
            allowed = {p.strip() for p in diff_part.split("+") if p.strip()}
        else:
            allowed = {diff_part}
        stages.append((until, allowed))
    if not stages:
        return None
    stages.sort(key=lambda x: x[0])
    return stages


def difficulties_for_epoch(
    epoch: int,
    schedule: Optional[List[Tuple[int, Optional[set]]]],
    static_allowlist: Optional[set] = None,
) -> Optional[set]:
    """Return allowed difficulty tags for a training epoch (``None`` = all)."""
    if schedule:
        for until, allowed in schedule:
            if epoch < until:
                return allowed
        return schedule[-1][1]
    return static_allowlist


def filter_instances_by_difficulty(
    instances: Sequence[Dict[str, Any]],
    allowed: Optional[set],
) -> List[Dict[str, Any]]:
    if allowed is None:
        return list(instances)
    out = [inst for inst in instances if instance_difficulty(inst) in allowed]
    return out


def sort_instances_by_difficulty(
    instances: List[Dict[str, Any]],
    order_spec: Any = None,
) -> List[Dict[str, Any]]:
    """Stable sort easy -> hard. ``order_spec`` defaults to introductory,interview,competition."""
    if order_spec is None or str(order_spec).strip() == "":
        order = list(_DIFFICULTY_RANK.keys())
    else:
        order = [p.strip().lower() for p in str(order_spec).split(",") if p.strip()]
    rank = {name: i for i, name in enumerate(order)}
    default_rank = len(order)

    def _key(inst: Dict[str, Any]):
        diff = instance_difficulty(inst)
        return (rank.get(diff, default_rank), str(inst.get("instance_id", "")))

    return sorted(instances, key=_key)


def apply_instance_selection(
    instances: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Filter / shuffle / slice instances (borrowed from mini-swe-agent swebench).

    Config keys (all optional):
      - ``difficulty_filter``: comma allowlist, e.g. ``introductory`` or ``introductory,interview``
      - ``difficulty_order``: sort easy->hard before shuffle (default: introductory,interview,competition)
      - ``instance_filter``: regex matched against ``instance_id`` (empty = no filter)
      - ``instance_slice``: Python slice spec, e.g. ``0:5``, ``10:``, ``:3``
      - ``shuffle_seed``: int; stable shuffle before filter/slice (None = keep order)
      - ``subset_size``: keep first N after the above (legacy shortcut)
    """
    cfg = cfg or {}
    out = list(instances)

    allowed = parse_difficulty_allowlist(cfg.get("difficulty_filter"))
    if allowed is not None:
        before = len(out)
        out = filter_instances_by_difficulty(out, allowed)
        if len(out) != before:
            print(f"[swebench] difficulty_filter {sorted(allowed)!r}: {before} -> {len(out)}")

    if str(cfg.get("difficulty_order", "")).strip().lower() not in ("", "none", "0", "false"):
        out = sort_instances_by_difficulty(out, cfg.get("difficulty_order"))

    shuffle_seed = cfg.get("shuffle_seed", None)
    if shuffle_seed is not None:
        out = sorted(out, key=lambda x: str(x.get("instance_id", "")))
        rng = random.Random(int(shuffle_seed))
        rng.shuffle(out)

    filter_spec = (cfg.get("instance_filter") or "").strip()
    if filter_spec:
        before = len(out)
        out = [inst for inst in out if re.match(filter_spec, str(inst.get("instance_id", "")))]
        if len(out) != before:
            print(f"[swebench] instance_filter {filter_spec!r}: {before} -> {len(out)}")

    slice_spec = (cfg.get("instance_slice") or "").strip()
    if slice_spec:
        before = len(out)
        parts = [int(x) if x.strip() else None for x in slice_spec.split(":")]
        if len(parts) > 3:
            raise ValueError(f"invalid instance_slice {slice_spec!r}")
        while len(parts) < 3:
            parts.append(None)
        out = out[slice(*parts)]
        if len(out) != before:
            print(f"[swebench] instance_slice {slice_spec!r}: {before} -> {len(out)}")

    subset = cfg.get("subset_size", None)
    if subset is not None and str(subset).strip() != "":
        out = out[: int(subset)]
    return out


def load_instances(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load SWE-bench-style instances.

    A ``benchmark`` preset (if set) first resolves the dataset id/split/backend
    (see :func:`resolve_benchmark`). Presets with ``merge_benchmarks`` (e.g.
    ``light_eval``) concatenate several benchmarks for in-training validation.
    """
    cfg = resolve_benchmark(cfg)
    merge_specs = cfg.get("merge_benchmarks")
    if merge_specs:
        merged: List[Dict[str, Any]] = []
        base = dict(cfg)
        base.pop("merge_benchmarks", None)
        for sub in merge_specs:
            sub_cfg = {**base, **sub}
            sub_cfg.pop("merge_benchmarks", None)
            sub_cfg["benchmark"] = sub["benchmark"]
            merged.extend(_load_single_benchmark_instances(sub_cfg))
        return merged
    return _load_single_benchmark_instances(cfg)


def _normalize_instance(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Map a raw coding-dataset / R2E-Gym row to the fields this env needs.

    The full original row is preserved under ``_raw`` so container backends
    (e.g. R2E-Gym, which does ``EnvArgs(ds=row)``) get every field they need.
    """
    return {
        "instance_id": ex.get("instance_id", ex.get("id", "unknown")),
        "problem_statement": ex.get("problem_statement", ex.get("problem", "")),
        "repo": ex.get("repo", ""),
        "base_commit": ex.get("base_commit", ""),
        "test_patch": ex.get("test_patch", ""),
        "patch": ex.get("patch", ""),
        "FAIL_TO_PASS": ex.get("FAIL_TO_PASS", ""),
        "PASS_TO_PASS": ex.get("PASS_TO_PASS", ""),
        # full original row for container backends (R2E-Gym EnvArgs(ds=...))
        "_raw": dict(ex),
        # local-stub-only fields (absent for real instances; backend ignores them)
        "initial_files": ex.get("initial_files", {}),
        "target_file": ex.get("target_file", None),
        "solution_keywords": ex.get("solution_keywords", []),
    }


def _synthetic_instances() -> List[Dict[str, Any]]:
    """Toy bug-fix tasks whose tests *really run* under the local stub backend.

    Each instance ships a buggy source module plus a ``test_*.py`` file that
    imports it (flat layout, no package needed). The tests genuinely fail on the
    buggy code and pass once the agent fixes it, so the local backend exercises a
    real self-repair loop (run pytest -> read traceback -> edit -> re-run).
    """
    specs = [
        {
            "slug": "calc",
            "src_path": "calc.py",
            "src": (
                "def add(a, b):\n"
                "    return a - b  # BUG: should be a + b\n\n"
                "def mul(a, b):\n"
                "    return a + b  # BUG: should be a * b\n"
            ),
            "test_path": "test_calc.py",
            "test": (
                "from calc import add, mul\n\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
                "    assert add(-1, 1) == 0\n\n"
                "def test_mul():\n"
                "    assert mul(2, 3) == 6\n"
                "    assert mul(4, 0) == 0\n"
            ),
            "fail_to_pass": ["test_add", "test_mul"],
            "problem": (
                "The functions in calc.py are buggy: `add` must return the sum "
                "and `mul` must return the product. Run `pytest` to see the "
                "failures, then fix calc.py until the tests pass."),
        },
        {
            "slug": "strutil",
            "src_path": "strutil.py",
            "src": (
                "def reverse_words(s):\n"
                "    # BUG: returns the original string instead of reversing word order\n"
                "    return s\n\n"
                "def is_palindrome(s):\n"
                "    # BUG: ignores case/spaces incorrectly\n"
                "    return s == s[::-1]\n"
            ),
            "test_path": "test_strutil.py",
            "test": (
                "from strutil import reverse_words, is_palindrome\n\n"
                "def test_reverse_words():\n"
                "    assert reverse_words('a b c') == 'c b a'\n"
                "    assert reverse_words('hello world') == 'world hello'\n\n"
                "def test_is_palindrome():\n"
                "    assert is_palindrome('A man a plan a canal Panama')\n"
                "    assert not is_palindrome('hello')\n"
            ),
            "fail_to_pass": ["test_reverse_words", "test_is_palindrome"],
            "problem": (
                "strutil.py is buggy: `reverse_words` should reverse the order of "
                "whitespace-separated words, and `is_palindrome` should ignore "
                "case and spaces. Run `pytest`, read the traceback, and fix "
                "strutil.py until the tests pass."),
        },
    ]

    insts: List[Dict[str, Any]] = []
    # Replicate each spec so a reset group (group_n identical rollouts) has data,
    # while keeping a couple of distinct bug shapes for cross-instance variety.
    for rep in range(4):
        for spec in specs:
            insts.append({
                "instance_id": f"synthetic__{spec['slug']}-{rep}",
                "problem_statement": spec["problem"],
                "repo": f"synthetic/{spec['slug']}",
                "base_commit": "0" * 40,
                "test_patch": "",
                "patch": "",
                "FAIL_TO_PASS": json.dumps(spec["fail_to_pass"]),
                "PASS_TO_PASS": "[]",
                "initial_files": {
                    spec["src_path"]: spec["src"],
                    spec["test_path"]: spec["test"],
                },
                "target_file": spec["src_path"],
                "solution_keywords": [],
            })
    return insts


# --------------------------------------------------------------------------- #
# Single-instance environment                                                 #
# --------------------------------------------------------------------------- #
MAX_OBS_CHARS = 4000

# Shown when ``test_feedback_mode=blind``: no hidden-suite results during repair.
BLIND_TEST_FEEDBACK_MSG = (
    "Hidden unit tests are not run during repair. Inspect code with cat/ls, "
    "revise with <edit>, and submit with <finish> when ready. "
    "Grading uses the full test suite only at submission."
)

# Shown when ``test_feedback_mode=exec`` and the agent tries pytest / hidden suite.
EXEC_HIDDEN_SUITE_MSG = (
    "The hidden grading suite is not run during repair (including pytest). "
    "You may run your program with input you choose, e.g. "
    "<execute_bash>echo '1 2' | python solution.py</execute_bash> "
    "— you will see stdout/stderr only, not expected answers. "
    "Submit with <finish> for full hidden grading."
)


class SWEBenchSingleEnv:
    """Hosts one coding instance with a working-copy file map."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.backend = make_backend(self.cfg)
        self.max_turns = int(self.cfg.get("max_turns", 30))
        self.min_turns_before_finish = int(self.cfg.get("min_turns_before_finish", 0))
        self.reward_success = float(self.cfg.get("reward_success", 1.0))
        self.reward_fail = float(self.cfg.get("reward_fail", 0.0))
        self.reward_mode = str(self.cfg.get("reward_mode", "graded")).lower()
        mode = str(self.cfg.get("test_feedback_mode", "blind")).lower()
        if mode == "oracle":
            mode = "interactive"
        if mode not in ("blind", "exec", "interactive"):
            mode = "blind"
        self.test_feedback_mode = mode
        interactive = mode == "interactive"
        self.step_reward_coef = float(
            self.cfg.get("step_reward_coef", 0.2 if interactive else 0.0))
        self.auto_test_after_edit = bool(
            self.cfg.get("auto_test_after_edit", interactive))
        if not interactive:
            self.auto_test_after_edit = False
        self.instance: Optional[Dict[str, Any]] = None
        self.files: Dict[str, str] = {}
        self.turn = 0
        self.last_output = ""
        self.problem_statement = ""

    def reset(self, instance: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.instance = instance
        self.files = self.backend.setup(instance)
        self.turn = 0
        self._episode_done = False
        self._last_bash_cmd = ""
        self.last_output = "(no command run yet)"
        # Container backends (R2E-Gym) supply the task text from the runtime;
        # fall back to the instance's own problem statement otherwise.
        ps = self.backend.task_instruction(instance) or instance.get("problem_statement", "")
        self.problem_statement = ps
        info = {"instance_id": instance["instance_id"], "won": False,
                "problem_statement": ps,
                "available_actions": ["edit", "execute_bash", "finish"],
                "difficulty": instance.get("difficulty", "unknown"),
                "anchor": self._build_anchor_state()}
        return self._build_repo_view(), info

    def _graded_partial_reward(self, info: Dict[str, Any]) -> float:
        """Dense step reward from current pass fraction (graded + interactive only)."""
        if self.test_feedback_mode != "interactive" or self.reward_mode != "graded":
            return 0.0
        try:
            frac = float(self.backend.score(self.instance, self.files))
            frac = max(0.0, min(1.0, frac))
            info["score"] = frac
            span = self.reward_success - self.reward_fail
            return self.step_reward_coef * span * frac
        except Exception:  # noqa: BLE001
            return 0.0

    def _auto_test_after_edit(self, info: Dict[str, Any]) -> Tuple[float, bool]:
        """Run the test suite immediately after an edit (hard execution feedback).

        Returns (step_reward, early_done). Mirrors the bash test-command path so
        Murphy/CodeRL-style RL always sees execution feedback after a code change.
        """
        if self.test_feedback_mode != "interactive" or not self.auto_test_after_edit:
            return 0.0, False
        passed, report = self.backend.run_tests(self.instance, self.files)
        header = "----- tests after edit -----"
        body = report[:MAX_OBS_CHARS]
        self.last_output = f"{self.last_output}\n\n{header}\n{body}"
        reward = 0.0
        if not passed:
            reward = self._graded_partial_reward(info)
        return reward, passed

    def step(self, action: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.turn += 1
        atype = action.get("type", "noop")
        done = False
        reward = 0.0
        info: Dict[str, Any] = {"instance_id": self.instance["instance_id"], "won": False,
                                "problem_statement": self.problem_statement,
                                "available_actions": ["edit", "execute_bash", "finish"],
                                "difficulty": self.instance.get("difficulty", "unknown")}

        if atype == "edit":
            path = str(action.get("path") or "").strip()
            if not path:
                self.last_output = "Edit failed: missing path."
            else:
                try:
                    # Supports overwrite / patch (search-replace) / insert.
                    # Projection emits patch/insert without a top-level ``content``.
                    new_content = apply_edit_action(self.files, path, action)
                    self.backend.apply_edit(self.instance, self.files, path, new_content)
                    mode = str(action.get("mode", "overwrite")).lower()
                    self.last_output = (
                        f"Edited {path} via {mode} ({len(new_content)} chars)."
                    )
                    edit_reward, early_done = self._auto_test_after_edit(info)
                    reward += edit_reward
                    if early_done:
                        done = True
                except EditApplyError as exc:
                    self.last_output = f"Edit failed: {exc}"
        elif atype == "bash":
            cmd = action["cmd"]
            self._last_bash_cmd = cmd
            if self.backend.is_hidden_suite_command(cmd):
                if self.test_feedback_mode == "interactive":
                    passed, report = self.backend.run_tests(self.instance, self.files)
                    self.last_output = report[:MAX_OBS_CHARS]
                    if passed:
                        done = True
                    else:
                        reward += self._graded_partial_reward(info)
                elif self.test_feedback_mode == "exec":
                    self.last_output = EXEC_HIDDEN_SUITE_MSG
                else:
                    self.last_output = BLIND_TEST_FEEDBACK_MSG
            elif (self.test_feedback_mode == "exec"
                  and self.backend.is_program_run_command(cmd)):
                self.last_output = self.backend.run_program_exec(
                    self.instance, self.files, cmd)[:MAX_OBS_CHARS]
                if self.step_reward_coef > 0:
                    frac = float(self.backend.score_program_exec_match(
                        self.instance, self.files, cmd))
                    if frac > 0:
                        span = self.reward_success - self.reward_fail
                        reward += self.step_reward_coef * span * frac
                        info["exec_selftest_frac"] = frac
            else:
                self.last_output = self.backend.run_bash(
                    self.instance, self.files, cmd)[:MAX_OBS_CHARS]
        elif atype == "finish":
            if (self.min_turns_before_finish > 0
                    and self.turn < self.min_turns_before_finish):
                self.last_output = (
                    f"Cannot submit yet — at least {self.min_turns_before_finish} "
                    f"interaction turn(s) required (currently turn {self.turn}). "
                    "Continue inspecting or editing before <finish>."
                )
            else:
                done = True
        else:  # noop / invalid
            self.last_output = (
                "No valid Action recognized. Use Thought "
                "(<think>...</think>) then one of: "
                "<edit>, <execute_bash>, or <finish>."
            )

        # Force termination at the turn budget (and grade whatever we have).
        if self.turn >= self.max_turns:
            done = True

        if done:
            won = bool(self.backend.evaluate(self.instance, self.files))
            terminal = self.reward_success if won else self.reward_fail
            reward = (reward + terminal) if self.step_reward_coef > 0 else terminal
            info["won"] = won

        info["anchor"] = self._build_anchor_state()
        return self._build_obs(), reward, done, info

    # ------------------------------------------------------------------ #
    def _build_repo_view(self) -> str:
        listing = "\n".join(sorted(self.files.keys())) or "(use bash, e.g. `ls`, to inspect the repo)"
        return (f"## Repository files\n{listing}\n\n"
                f"## Problem statement\n{self.problem_statement}\n")

    def _build_obs(self) -> str:
        return f"Observation: {self.last_output}\n"

    def _build_anchor_state(self) -> str:
        """Workspace file map as GiGPO step-state (no history / prompt wrapper)."""
        if not self.files:
            return ""
        parts = [f"===== {path} =====\n{self.files[path]}" for path in sorted(self.files)]
        return "\n".join(parts)

    def close(self) -> None:
        self.backend.close()


# --------------------------------------------------------------------------- #
# Ray-vectorized environment                                                  #
# --------------------------------------------------------------------------- #
if ray is not None:

    @ray.remote
    class _SWEBenchWorker:
        def __init__(self, cfg):
            self.env = SWEBenchSingleEnv(cfg)

        def reset(self, instance):
            return self.env.reset(instance)

        def step(self, action):
            return self.env.step(action)

        def close(self):
            self.env.close()
            return True


class SWEBenchMultiProcessEnv:
    """Vectorized, Ray-based wrapper hosting ``env_num * group_n`` instances.

    Group semantics mirror the rest of verl-agent: the ``group_n`` sub-envs in a
    group are reset to the *same* coding instance, which is the set of rollouts
    GRPO / GiGPO / DiDPO group across.
    """

    def __init__(self, cfg: Dict[str, Any], env_num: int, group_n: int,
                 resources_per_worker: dict, is_train: bool = True, seed: int = 0):
        if ray is not None and not ray.is_initialized():
            ray.init()
        self.cfg = dict(cfg)
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self._rng = np.random.RandomState(seed)
        if not is_train:
            assert group_n == 1

        self._curriculum_schedule = parse_curriculum_schedule(self.cfg.get("difficulty_curriculum"))
        self._static_difficulty_filter = parse_difficulty_allowlist(self.cfg.get("difficulty_filter"))
        self._curriculum_epoch = -1
        if self._curriculum_schedule:
            # Load full gradable pool; subset/slice apply per-stage in _rebuild_instance_pool.
            load_cfg = {
                **self.cfg,
                "skip_difficulty_filter": True,
                "subset_size": None,
                "instance_slice": "",
            }
            self._all_instances = load_instances(load_cfg)
            self._rebuild_instance_pool(0)
        else:
            self.instances = load_instances(self.cfg)
            self._all_instances = self.instances
        assert len(self.instances) > 0, "No coding instances loaded."

        rpw = resources_per_worker or {}
        worker_cls = _SWEBenchWorker.options(**rpw) if ray is not None else None
        self._workers = [worker_cls.remote(cfg) for _ in range(self.num_processes)]
        self._cur_instances: List[Optional[Dict[str, Any]]] = [None] * self.num_processes

    def reset(self):
        n_groups = self.env_num
        # Optionally pin tracked instance_ids so SwanLab can observe group evolution.
        pinned = self._resolve_pinned_instance_ids(n_groups)
        pick: List[int] = []
        if pinned:
            id_to_idx = {
                str(inst.get("instance_id", "")): i for i, inst in enumerate(self.instances)
            }
            missing = []
            for iid in pinned:
                if iid in id_to_idx:
                    pick.append(id_to_idx[iid])
                else:
                    missing.append(iid)
            if missing:
                print(
                    f"[SWEBenchMultiProcessEnv] WARN: pinned instance_id(s) not in "
                    f"loaded benchmark pool (skip): {missing[:5]}"
                    + ("..." if len(missing) > 5 else "")
                )
            # Fill remaining slots randomly (without forcing duplicates of pinned).
            remain = n_groups - len(pick)
            if remain > 0:
                pool = [i for i in range(len(self.instances)) if i not in set(pick)]
                if not pool:
                    pool = list(range(len(self.instances)))
                extra = self._rng.choice(
                    pool, size=remain, replace=len(pool) < remain
                ).tolist()
                pick.extend(extra)
        else:
            pick = self._rng.choice(
                len(self.instances), size=n_groups, replace=len(self.instances) < n_groups
            ).tolist()
        pick = np.repeat(pick, self.group_n).tolist()
        futures = []
        for w, idx in zip(self._workers, pick):
            inst = self.instances[idx]
            self._cur_instances[self._workers.index(w)] = inst
            futures.append(w.reset.remote(inst))
        results = ray.get(futures)
        obs = [o for o, _ in results]
        infos = [i for _, i in results]
        return obs, infos

    def _resolve_pinned_instance_ids(self, n_groups: int) -> List[str]:
        """Return up to ``n_groups`` stable instance_ids to force into this batch.

        Train only — never pin on val envs (would bias evaluation).
        """
        if not getattr(self, "is_train", True):
            return []
        pinned: List[str] = []
        cfg_list = self.cfg.get("pinned_instance_ids") or []
        if cfg_list:
            pinned.extend(str(x) for x in cfg_list)
        path = self.cfg.get("tracked_instance_ids_file") or ""
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ids = data.get("instance_ids") or data.get("uids") or []
                for x in ids:
                    sx = str(x).strip()
                    # Skip empty / ephemeral UUID uids left by the old tracker.
                    if not sx or sx in pinned:
                        continue
                    if len(sx) == 36 and sx.count("-") == 4:
                        # Heuristic UUID shape — not a stable instance_id.
                        continue
                    pinned.append(sx)
            except Exception:  # noqa: BLE001
                pass
        return pinned[: max(0, int(n_groups))]

    def step(self, actions: List[Dict[str, Any]]):
        if len(actions) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} actions, got {len(actions)}")
        futures = [w.step.remote(a) for w, a in zip(self._workers, actions)]
        results = ray.get(futures)
        obs = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def close(self):
        if getattr(self, "_closed", False):
            return
        ray.get([w.close.remote() for w in self._workers])
        for w in self._workers:
            ray.kill(w)
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build_swebench_envs(env_config, env_num: int, group_n: int,
                        resources_per_worker: dict, is_train: bool = True, seed: int = 0,
                        benchmark_override: str | None = None):
    """Factory mirroring ``build_webshop_envs`` so the manager can swap seamlessly."""
    cfg = dict(env_config.swebench) if hasattr(env_config, "swebench") else dict(env_config.get("swebench", {}))
    if benchmark_override:
        cfg = resolve_benchmark({**cfg, "benchmark": benchmark_override})
        val_slice = (cfg.get("val_instance_slice") or "").strip()
        if not is_train and val_slice:
            cfg = {**cfg, "instance_slice": val_slice}
    return SWEBenchMultiProcessEnv(
        cfg=cfg, env_num=env_num, group_n=group_n,
        resources_per_worker=resources_per_worker, is_train=is_train, seed=seed)
