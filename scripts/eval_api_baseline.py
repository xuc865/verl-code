#!/usr/bin/env python3
"""Multi-turn self-repair eval against an OpenAI-compatible API (no Ray/GPU).

Usage:
  python3 scripts/eval_api_baseline.py \\
    --api-base http://29.163.228.8:8080/v1 \\
    --model Kimi-K2.7-Code \\
    --benchmark light_eval \\
    --data-root /mnt/z4/solariewang/datasets \\
    --out logs/eval_api_kimi_light.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import random
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_prompts = _import_module(
    "swebench_prompts_api_eval",
    REPO / "agent_system/environments/prompts/swebench.py",
)
SWEBENCH_TEMPLATE = _prompts.SWEBENCH_TEMPLATE
SWEBENCH_TEMPLATE_NO_HIS = _prompts.SWEBENCH_TEMPLATE_NO_HIS
_ACTION_SPEC = _prompts._ACTION_SPEC
action_spec_for_mode = _prompts.action_spec_for_mode
_SINGLE_TURN_NOTE = (
    "\n## Protocol (single-turn)\n"
    "You have exactly ONE step. Produce the complete solution in a single "
    "<edit path=\"solution.py\"><code>...</code></edit> block "
    "(or <finish> after editing). Hidden tests run once at the end; "
    "no further turns.\n"
)
parse_swebench_action = _import_module(
    "swebench_projection_api_eval",
    REPO / "agent_system/environments/env_package/swebench/projection.py",
).parse_swebench_action

_PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_FREEFORM_DIRECT_PROMPT = """You are an expert Python programmer.

## Problem
{problem_statement}

## Current workspace observation
{repo_view}

Write a complete solution in `solution.py`.
Put the final code in a single ```python``` fenced block.
No special XML / ReAct action format is required.
"""

_FREEFORM_COT_PROMPT = """You are an expert Python programmer.

## Problem
{problem_statement}

## Current workspace observation
{repo_view}

Think step by step about the algorithm and edge cases, then write a complete
solution in `solution.py`.
Put the final code in a single ```python``` fenced block.
No special XML / ReAct action format is required.
"""

_FREEFORM_REPAIR_PROMPT = """You are an expert Python programmer revising a solution.

## Problem
{problem_statement}

## Prior attempts (most recent {history_length} of {step_count})
{action_history}

## Latest observation
{current_observation}

Continue reasoning as needed, then provide a revised complete `solution.py`.
Put the final code in a single ```python``` fenced block.
No special XML / ReAct action format is required.
"""

# Self-planning baseline: (1) emit a numbered intent plan, (2) implement from the plan.
# Plan prompt follows the common Self-Planning intent→plan template.
_SELF_PLAN_EXAMPLES = """\
Example 1:
Problem: Given a list of integers, return True if any two numbers are closer than a given threshold.
Plan:
1. Sort the numbers in ascending order.
2. Scan adjacent pairs and check their absolute difference against the threshold.
3. Return True on the first close pair; otherwise return False.

Example 2:
Problem: Write a function that returns the longest common prefix string among an array of strings.
Plan:
1. Handle the empty-array edge case.
2. Take the first string as the initial prefix candidate.
3. Shrink the candidate until every string starts with it.
4. Return the remaining prefix.

Example 3:
Problem: Given n non-negative integers representing an elevation map, compute how much water it can trap after raining.
Plan:
1. Precompute the tallest bar to the left of each index.
2. Precompute the tallest bar to the right of each index.
3. For each index, add max(0, min(left_max, right_max) - height).
4. Return the total trapped water.

Example 4:
Problem: Determine whether a string of brackets is valid (correctly matched and nested).
Plan:
1. Use a stack to store opening brackets.
2. Map each closing bracket to its matching opener.
3. On a closing bracket, pop and verify the match; reject on mismatch or empty stack.
4. Accept only if the stack is empty at the end.

Example 5:
Problem: Given an array of integers, return indices of the two numbers that add up to a target.
Plan:
1. Build a value-to-index map while scanning once.
2. For each number, look up target - number in the map.
3. Skip using the same index twice.
4. Return the pair of indices when found.

Example 6:
Problem: Merge two sorted linked lists and return it as a sorted list.
Plan:
1. Create a dummy head for the merged list.
2. Walk both lists, always attaching the smaller current node.
3. Append any remaining tail from either list.
4. Return dummy.next.

Example 7:
Problem: Given a binary tree, return the level-order traversal of its nodes' values.
Plan:
1. Return an empty list for a null root.
2. BFS with a queue starting from the root.
3. For each level, collect all node values then enqueue children.
4. Append each level list to the result.

Example 8:
Problem: Find the length of the longest substring without repeating characters.
Plan:
1. Maintain a sliding window with left and right pointers.
2. Track the last index of each character inside the window.
3. When a repeat appears, move left past the previous occurrence.
4. Update the maximum window length as right advances.
"""

_SELF_PLANNING_PLAN_PROMPT = """You are given a programming problem. Decompose it into a concise numbered plan.

Rules:
1. Each step should be a single implementable subtask.
2. Use imperative sentences.
3. Keep steps high-level and concise.
4. Do not write code.
5. Include conditionals or loops only when necessary.

Examples:
{examples}

Problem:
{problem}

Plan:
"""

_SELF_PLANNING_CODE_PROMPT = """You are an expert Python programmer.

## Problem
{problem_statement}

## Implementation plan
{plan}

## Current workspace observation
{repo_view}

Follow the plan and write a complete solution in `solution.py`.
Put the final code in a single ```python``` fenced block.
No special XML / ReAct action format is required.
"""

_SELF_PLANNING_REPAIR_PROMPT = """You are an expert Python programmer revising a solution.

## Problem
{problem_statement}

## Implementation plan
{plan}

## Prior attempts (most recent {history_length} of {step_count})
{action_history}

## Latest observation
{current_observation}

Stay consistent with the plan unless the observation shows the plan itself is wrong.
Provide a revised complete `solution.py` in a single ```python``` fenced block.
No special XML / ReAct action format is required.
"""

_ADAPTERS_DIR = REPO / "agent_system/environments/env_package/swebench/adapters"
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

_BENCHMARK_SPECS: Dict[str, Dict[str, Any]] = {
    "humaneval": {
        "parquet_url": f"{_HF_MIRROR}/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet",
        "adapter": "humaneval",
        "backend": "local_stub",
    },
    "mbpp": {
        "parquet_url": f"{_HF_MIRROR}/datasets/google-research-datasets/mbpp/resolve/main/sanitized/test-00000-of-00001.parquet",
        "adapter": "mbpp",
        "backend": "local_stub",
    },
    "apps": {
        "jsonl": "codeparrot_apps/test.jsonl",
        "adapter": "apps",
        "backend": "selfrepair_io",
    },
    "apps_train": {
        "jsonl": "codeparrot_apps/train.jsonl",
        "adapter": "apps",
        "backend": "selfrepair_io",
    },
    "livecodebench": {
        "jsonl_glob": "livecodebench_code_generation_lite/test*.jsonl",
        "adapter": "livecodebench",
        "backend": "selfrepair_io",
    },
    "usaco": {
        "parquet_urls": [
            f"{_HF_MIRROR}/datasets/dapumptu/usaco_benchmark/resolve/main/data/train-{i:05d}-of-00003.parquet"
            for i in range(3)
        ],
        "adapter": "usaco",
        "backend": "selfrepair_io",
    },
    "ojbench": {
        "jsonl": "OJBench_testdata/prompts/full.jsonl",
        "adapter": "ojbench",
        "backend": "selfrepair_io",
        "needs_data_root": True,
    },
    "icpc": {
        # Staged via scripts/preload_icpc_eval.sh -> RUC-AIBOX_ICPC-Eval/data/test-*.parquet
        "local_parquet_glob": "RUC-AIBOX_ICPC-Eval/data/test-*.parquet",
        "parquet_urls": [
            f"{_HF_MIRROR}/datasets/RUC-AIBOX/ICPC-Eval/resolve/main/data/test-{i:05d}-of-00012.parquet"
            for i in range(12)
        ],
        "adapter": "icpc",
        "backend": "selfrepair_io",
    },
    "leetcode": {
        "jsonl": "newfacade_LeetCodeDataset/LeetCodeDataset-test.jsonl",
        "adapter": "leetcode",
        "backend": "local_stub",
    },
    "light_eval": {
        "backend": "mixed",
        "merge": [
            {"benchmark": "apps", "instance_slice": "0:64"},
            {"benchmark": "humaneval", "instance_slice": "0:32"},
            {"benchmark": "mbpp", "instance_slice": "0:32"},
        ],
    },
}


def _load_adapters():
    spec = importlib.util.spec_from_file_location(
        "swebench_adapters",
        _ADAPTERS_DIR / "__init__.py",
        submodule_search_locations=[str(_ADAPTERS_DIR)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["swebench_adapters"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


def _apply_slice(instances: List[Dict[str, Any]], slice_spec: str) -> List[Dict[str, Any]]:
    slice_spec = (slice_spec or "").strip()
    if not slice_spec:
        return instances
    parts = slice_spec.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid instance_slice {slice_spec!r}")
    while len(parts) < 3:
        parts.append("")
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if parts[2] else None
    return instances[slice(start, stop, step)]


def _apply_random_sample(
    instances: List[Dict[str, Any]], sample_size: int, sample_seed: int
) -> List[Dict[str, Any]]:
    if sample_size <= 0:
        return instances
    pool = len(instances)
    if pool == 0:
        return instances
    if sample_size >= pool:
        print(
            f"[sample] requested n={sample_size} >= pool={pool}, using full pool",
            flush=True,
        )
        return instances
    rng = random.Random(sample_seed)
    order = list(range(pool))
    rng.shuffle(order)
    picked = [instances[i] for i in order[:sample_size]]
    print(
        f"[sample] random n={sample_size} seed={sample_seed} from pool={pool}",
        flush=True,
    )
    return picked


def _read_parquet_rows(url: str, cache_dir: Path) -> List[Dict[str, Any]]:
    import hashlib

    cache_dir.mkdir(parents=True, exist_ok=True)
    base = url.split("/")[-1]
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    local = cache_dir / f"{digest}_{base}"
    if not local.is_file():
        print(f"[data] downloading {url}", flush=True)
        urllib.request.urlretrieve(url, local)
    return _read_parquet_file(local)


def _read_parquet_file(path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def _read_staged_parquet_glob(data_root: Path, pattern: str) -> List[Dict[str, Any]]:
    files = sorted(data_root.glob(pattern))
    if not files:
        return []
    rows: List[Dict[str, Any]] = []
    for fp in files:
        rows.extend(_read_parquet_file(fp))
    print(f"[data] loaded {len(rows)} rows from {len(files)} staged parquet(s) "
          f"({pattern})", flush=True)
    return rows


def _read_parquet_url_list(urls: List[str], cache_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for url in urls:
        try:
            rows.extend(_read_parquet_rows(url, cache_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"[data] skip parquet {url}: {exc}", flush=True)
    return rows


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_benchmark_instances(
    benchmark: str,
    data_root: Path,
    *,
    slice_spec: str = "",
    min_date: str = "",
) -> Tuple[List[Dict[str, Any]], str]:
    spec = _BENCHMARK_SPECS[benchmark]
    adapters = _load_adapters()
    cache_dir = data_root / "hf_cache" / "api_eval_parquet"

    if benchmark == "light_eval":
        merged: List[Dict[str, Any]] = []
        backend = spec["backend"]
        for sub in spec["merge"]:
            sub_rows, _ = _load_benchmark_instances(
                sub["benchmark"], data_root, slice_spec=sub.get("instance_slice", ""), min_date=min_date,
            )
            merged.extend(sub_rows)
        return merged, backend

    adapter_name = spec["adapter"]
    adapter_fn = adapters.get_adapter(adapter_name)
    rows: List[Dict[str, Any]]
    if "local_parquet_glob" in spec:
        rows = _read_staged_parquet_glob(data_root, spec["local_parquet_glob"])
        if not rows and "parquet_urls" in spec:
            rows = _read_parquet_url_list(spec["parquet_urls"], cache_dir)
        if not rows:
            raise RuntimeError(f"no rows loaded for benchmark {benchmark}")
    elif "parquet_urls" in spec:
        rows = _read_parquet_url_list(spec["parquet_urls"], cache_dir)
        if not rows:
            if spec.get("optional"):
                print(f"[data] optional benchmark {benchmark}: no rows, skipping load", flush=True)
                return [], spec["backend"]
            raise RuntimeError(f"no rows loaded for benchmark {benchmark}")
    elif "parquet_url" in spec:
        rows = _read_parquet_rows(spec["parquet_url"], cache_dir)
    elif "jsonl" in spec:
        rows = _read_jsonl_rows(data_root / spec["jsonl"])
    elif "jsonl_glob" in spec:
        rows = []
        for p in sorted(data_root.glob(spec["jsonl_glob"])):
            rows.extend(_read_jsonl_rows(p))
    else:
        raise ValueError(f"no loader for benchmark {benchmark}")

    if adapter_name == "ojbench":
        instances = adapter_fn(rows if rows else [{}], data_root=str(data_root))
    elif adapter_name == "livecodebench" and min_date:
        instances = adapter_fn(rows, min_date=min_date)
    else:
        instances = adapter_fn(rows)
    instances = _apply_slice(instances, slice_spec)
    return instances, spec["backend"]


_ENV_MOD = None


def _load_envs():
    global _ENV_MOD
    if _ENV_MOD is not None:
        return _ENV_MOD
    spec = importlib.util.spec_from_file_location(
        "swebench_envs_api_eval",
        REPO / "agent_system/environments/env_package/swebench/envs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swebench_envs_api_eval"] = mod
    spec.loader.exec_module(mod)
    _ENV_MOD = mod
    return mod


def _make_env(backend: str, data_root: str, max_turns: int, test_feedback_mode: str = "blind"):
    if test_feedback_mode == "oracle":
        test_feedback_mode = "interactive"
    envs = _load_envs()
    interactive = test_feedback_mode == "interactive"
    return envs.SWEBenchSingleEnv({
        "backend": backend,
        "max_turns": max_turns,
        "test_feedback_mode": test_feedback_mode,
        "auto_test_after_edit": interactive,
        "reward_mode": "graded",
        "step_reward_coef": 0.2 if interactive else 0.0,
        "data_root": data_root,
    })


def _message_text(msg: Dict[str, Any], *, include_reasoning: bool = True) -> str:
    # OpenAI-compat servers disagree on the thinking field name:
    # Qwen/vLLM often use ``reasoning``; GLM uses ``reasoning_content``.
    keys = ("content",) if not include_reasoning else (
        "content",
        "reasoning",
        "reasoning_content",
    )
    parts = []
    for key in keys:
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return "\n".join(parts).strip()


class OpenAICompatClient:
    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: int = 1800,
        retries: int = 3,
        disable_thinking: bool = False,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.disable_thinking = disable_thinking

    @staticmethod
    def _is_timeout_error(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return True
        if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, socket.timeout):
            return True
        return "timed out" in str(exc).lower()

    def chat(self, messages: List[Dict[str, str]], *, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"thinking": False}
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "verl-swe-eval/1.0 (OpenAI-compatible)",
            },
            method="POST",
        )
        last_timeout: Optional[BaseException] = None
        # Internal vLLM / IDC endpoints must not go through http_proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for attempt in range(self.retries):
            try:
                with opener.open(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"API HTTP {exc.code}: {detail[:500]}") from exc
            except Exception as exc:  # noqa: BLE001 - classify timeout vs hard fail
                if self._is_timeout_error(exc):
                    last_timeout = exc
                    if attempt + 1 < self.retries:
                        time.sleep(min(60, 10 * (attempt + 1)))
                        continue
                    raise TimeoutError(
                        f"API timed out after {self.retries} attempt(s) "
                        f"(timeout={self.timeout}s per call)"
                    ) from exc
                raise
        else:
            raise TimeoutError(
                f"API timed out after {self.retries} attempt(s) "
                f"(timeout={self.timeout}s per call)"
            ) from last_timeout
        choice = body["choices"][0]["message"]
        text = _message_text(choice, include_reasoning=not self.disable_thinking)
        if not text and self.disable_thinking:
            # Qwen/vLLM may still emit chain-of-thought in ``reasoning`` while
            # ``content`` is null when thinking=false is requested.
            text = _message_text(choice, include_reasoning=True)
        if not text:
            raise RuntimeError(f"empty model response: {json.dumps(choice)[:300]}")
        return text


def _render_action(action: Dict[str, Any]) -> str:
    t = action.get("type", "noop")
    if t == "edit":
        return f"edit {action.get('path', '')}"
    if t == "bash":
        return f"bash: {str(action.get('cmd', ''))[:80]}"
    if t == "finish":
        return "finish"
    return t


def _extract_freeform_code(text: str) -> Optional[str]:
    """Pull a Python solution from free-form model output."""
    if not text:
        return None
    fences = list(_PY_FENCE_RE.finditer(text))
    if fences:
        body = max((m.group(1) for m in fences), key=lambda s: len(s.strip())).strip()
        if body:
            return body
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:python|py)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    if not cleaned:
        return None
    # Heuristic: looks like Python source rather than pure prose.
    py_hints = ("def ", "class ", "import ", "from ", "return ", "if __name__")
    if any(h in cleaned for h in py_hints):
        return cleaned
    return None


def _normalize_plan_text(text: str) -> str:
    """Keep plan as prose; strip thinking tags and accidental code fences."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Drop fenced blocks (model sometimes emits code despite instructions).
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL).strip()
    return cleaned.strip()


def _is_code_prompt_mode(prompt_mode: str) -> bool:
    return prompt_mode in ("freeform", "self_planning")


def _freeform_repo_view(env: Any, problem_statement: str, fallback_obs: str = "") -> str:
    """Build freeform workspace text including editable source stubs.

    Default env repo view only lists filenames. Freeform agents cannot ``cat``
    files, so MBPP/HumanEval-style tasks need the ``solution.py`` stub (function
    signatures) in the prompt. Hidden ``test_*.py`` contents are omitted to
    avoid leaking assert expected values.
    """
    files = dict(getattr(env, "files", None) or {})
    target = "solution.py"
    if getattr(env, "instance", None):
        target = str(env.instance.get("target_file") or target)

    show_paths: List[str] = []
    if target in files:
        show_paths.append(target)
    for path in sorted(files):
        base = os.path.basename(path)
        if path in show_paths:
            continue
        if base.startswith("test_") and base.endswith(".py"):
            continue
        show_paths.append(path)

    if not show_paths:
        if fallback_obs and "Problem statement" in fallback_obs:
            return fallback_obs
        return (
            f"## Repository files\n(no editable sources visible)\n\n"
            f"## Problem statement\n{problem_statement}\n"
        )

    parts = ["## Repository files"]
    parts.append(
        "(Editable sources below. Implement/fix these; do not rename the "
        "public functions the tests import.)"
    )
    for path in show_paths:
        content = str(files.get(path) or "").rstrip()
        if len(content) > 12000:
            content = content[:12000] + "\n# ... truncated ..."
        fence = "python" if path.endswith(".py") else ""
        parts.append(f"\n### `{path}`\n```{fence}\n{content}\n```")
    parts.append(f"\n## Problem statement\n{problem_statement}\n")
    return "\n".join(parts)


def _build_prompt(
    *,
    problem_statement: str,
    repo_view: str,
    history: List[Tuple[str, str]],
    current_obs: str,
    history_length: int,
    max_turns: int = 12,
    test_feedback_mode: str = "blind",
    prompt_mode: str = "react",
    encourage_cot: bool = False,
    plan: str = "",
) -> str:
    if prompt_mode == "self_planning":
        if not history:
            return _SELF_PLANNING_CODE_PROMPT.format(
                problem_statement=problem_statement,
                plan=plan or "(no plan generated)",
                repo_view=repo_view,
            )
        action_history = "\n".join(
            f"Attempt {i + 1}: {act}\nObservation {i + 1}: {obs}"
            for i, (act, obs) in enumerate(history[-history_length:])
        )
        return _SELF_PLANNING_REPAIR_PROMPT.format(
            problem_statement=problem_statement,
            plan=plan or "(no plan generated)",
            step_count=len(history),
            history_length=min(history_length, len(history)),
            action_history=action_history,
            current_observation=current_obs,
        )

    if prompt_mode == "freeform":
        if not history:
            tmpl = _FREEFORM_COT_PROMPT if encourage_cot else _FREEFORM_DIRECT_PROMPT
            return tmpl.format(problem_statement=problem_statement, repo_view=repo_view)
        action_history = "\n".join(
            f"Attempt {i + 1}: {act}\nObservation {i + 1}: {obs}"
            for i, (act, obs) in enumerate(history[-history_length:])
        )
        return _FREEFORM_REPAIR_PROMPT.format(
            problem_statement=problem_statement,
            step_count=len(history),
            history_length=min(history_length, len(history)),
            action_history=action_history,
            current_observation=current_obs,
        )

    action_spec = action_spec_for_mode(test_feedback_mode)
    if max_turns <= 1:
        action_spec = action_spec + _SINGLE_TURN_NOTE
    if not history:
        return SWEBENCH_TEMPLATE_NO_HIS.format(repo_view=repo_view, action_spec=action_spec)
    action_history = "\n".join(
        f"Action {i + 1}: {act}\nObservation {i + 1}: {obs}"
        for i, (act, obs) in enumerate(history[-history_length:])
    )
    return SWEBENCH_TEMPLATE.format(
        problem_statement=problem_statement,
        step_count=len(history),
        history_length=min(history_length, len(history)),
        action_history=action_history,
        current_step=len(history) + 1,
        current_observation=current_obs,
        action_spec=action_spec,
    )


def run_episode(
    client: OpenAICompatClient,
    env,
    instance: Dict[str, Any],
    *,
    max_turns: int,
    history_length: int,
    max_tokens: int,
    temperature: float,
    test_feedback_mode: str = "blind",
    export_full_transcript: bool = False,
    prompt_mode: str = "react",
    encourage_cot: bool = False,
) -> Dict[str, Any]:
    obs, info = env.reset(instance)
    problem_statement = info.get("problem_statement", instance.get("problem_statement", ""))
    if _is_code_prompt_mode(prompt_mode):
        repo_view = _freeform_repo_view(env, problem_statement, fallback_obs=obs)
    else:
        repo_view = (
            obs if "Problem statement" in obs
            else f"## Repository files\n\n## Problem statement\n{problem_statement}\n"
        )
    history: List[Tuple[str, str]] = []
    transcript: List[Dict[str, Any]] = []
    plan = ""

    # Self-planning: one dedicated plan call before any code attempt.
    if prompt_mode == "self_planning":
        plan_prompt = _SELF_PLANNING_PLAN_PROMPT.format(
            examples=_SELF_PLAN_EXAMPLES,
            problem=problem_statement,
        )
        t0 = time.perf_counter()
        plan_raw = client.chat(
            [{"role": "user", "content": plan_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        plan_latency = time.perf_counter() - t0
        plan = _normalize_plan_text(plan_raw) or plan_raw.strip()
        plan_rec: Dict[str, Any] = {
            "turn": 0,
            "phase": "plan",
            "valid_action": True,
            "action_type": "plan",
            "latency_s": round(plan_latency, 3),
            "won": False,
            "prompt_mode": prompt_mode,
            "plan_preview": plan[:500],
        }
        if export_full_transcript:
            plan_rec.update({
                "user_prompt": plan_prompt,
                "assistant_response": plan_raw,
                "plan": plan,
            })
        transcript.append(plan_rec)

    for turn in range(max_turns):
        user_prompt = _build_prompt(
            problem_statement=problem_statement,
            repo_view=repo_view if turn == 0 else repo_view,
            history=history,
            current_obs=obs,
            history_length=history_length,
            max_turns=max_turns,
            test_feedback_mode=test_feedback_mode,
            prompt_mode=prompt_mode,
            encourage_cot=encourage_cot,
            plan=plan,
        )
        messages = [{"role": "user", "content": user_prompt}]
        t0 = time.perf_counter()
        response = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        latency = time.perf_counter() - t0

        if _is_code_prompt_mode(prompt_mode):
            code = _extract_freeform_code(response)
            if code:
                action = {
                    "type": "edit",
                    "path": "solution.py",
                    "mode": "overwrite",
                    "content": code,
                }
                valid = True
            else:
                action, valid = {"type": "noop"}, False
        else:
            action, valid = parse_swebench_action(response)

        next_obs, reward, done, step_info = env.step(action)
        history.append((_render_action(action), obs))
        obs = next_obs
        turn_rec: Dict[str, Any] = {
            "turn": turn + 1,
            "valid_action": bool(valid),
            "action_type": action.get("type"),
            "latency_s": round(latency, 3),
            "won": bool(step_info.get("won")),
            "prompt_mode": prompt_mode,
        }
        if prompt_mode == "self_planning":
            turn_rec["phase"] = "code"
        if export_full_transcript:
            turn_rec.update({
                "user_prompt": user_prompt,
                "assistant_response": response,
                "observation": history[-1][1] if history else "",
                "rendered_action": _render_action(action),
            })
            if plan:
                turn_rec["plan"] = plan
        else:
            turn_rec["response_preview"] = response[:500]
        transcript.append(turn_rec)

        # Freeform / self-planning: always submit after an edit so hidden tests run.
        # If lost and turns remain, reset and continue (multi-attempt repair).
        if _is_code_prompt_mode(prompt_mode) and action.get("type") == "edit" and not done:
            next_obs, reward, done, step_info = env.step({"type": "finish"})
            obs = next_obs
            finish_rec: Dict[str, Any] = {
                "turn": turn + 1,
                "valid_action": True,
                "action_type": "finish",
                "latency_s": 0.0,
                "won": bool(step_info.get("won")),
                "prompt_mode": prompt_mode,
                "auto_finish": True,
            }
            if prompt_mode == "self_planning":
                finish_rec["phase"] = "code"
            transcript.append(finish_rec)
            history.append(("finish", obs))
            if bool(step_info.get("won")):
                out = {
                    "instance_id": instance.get("instance_id"),
                    "outcome": "won",
                    "won": True,
                    "turns": turn + 1,
                    "reward": float(reward),
                    "difficulty": step_info.get("difficulty", "unknown"),
                    "transcript": transcript,
                }
                if plan:
                    out["plan"] = plan
                return out
            if (turn + 1) < max_turns:
                obs, info = env.reset(instance)
                problem_statement = info.get(
                    "problem_statement", instance.get("problem_statement", "")
                )
                if _is_code_prompt_mode(prompt_mode):
                    repo_view = _freeform_repo_view(
                        env, problem_statement, fallback_obs=obs
                    )
                else:
                    repo_view = (
                        obs if "Problem statement" in obs
                        else f"## Repository files\n\n## Problem statement\n{problem_statement}\n"
                    )
                continue
            out = {
                "instance_id": instance.get("instance_id"),
                "outcome": "lost",
                "won": False,
                "turns": turn + 1,
                "reward": float(reward),
                "difficulty": step_info.get("difficulty", "unknown"),
                "transcript": transcript,
            }
            if plan:
                out["plan"] = plan
            return out

        if done:
            out = {
                "instance_id": instance.get("instance_id"),
                "outcome": "won" if bool(step_info.get("won")) else "lost",
                "won": bool(step_info.get("won")),
                "turns": turn + 1,
                "reward": float(reward),
                "difficulty": step_info.get("difficulty", "unknown"),
                "transcript": transcript,
            }
            if plan:
                out["plan"] = plan
            return out

    out = {
        "instance_id": instance.get("instance_id"),
        "outcome": "lost",
        "won": False,
        "turns": max_turns,
        "reward": 0.0,
        "difficulty": "unknown",
        "transcript": transcript,
    }
    if plan:
        out["plan"] = plan
    return out


def _instance_difficulty(inst: Dict[str, Any]) -> str:
    raw = inst.get("_raw") or {}
    for key in ("difficulty", "problem_level", "level"):
        val = str(raw.get(key, "")).strip().lower()
        if val and val not in ("unknown", "none", ""):
            return val
    return "unknown"


def _protocol_metric_docs(feedback_mode: str) -> Dict[str, str]:
    if feedback_mode == "interactive":
        return {
            "primary_metric": "avg_turns_won",
            "primary_direction": "lower_is_better",
            "secondary_metric": "success_rate",
            "description": (
                "interactive: test feedback available during repair; compare efficiency "
                "by turns-to-solve on won episodes (fewer turns = better). "
                "Pass rate is secondary."
            ),
        }
    if feedback_mode == "exec":
        return {
            "primary_metric": "success_rate",
            "primary_direction": "higher_is_better",
            "secondary_metric": "avg_turns_won",
            "description": (
                "exec: agent may run solution.py with self-chosen stdin during repair; "
                "observations show stdout/stderr/traceback only (no expected answers). "
                "Full hidden grading at <finish>; compare final pass@1."
            ),
        }
    return {
        "primary_metric": "success_rate",
        "primary_direction": "higher_is_better",
        "secondary_metric": "avg_turns_won",
        "description": (
            "blind: no test results during repair; compare final pass@1 at submission "
            "(single-turn for closed API baselines; multi-turn for local RL agents)."
        ),
    }


def _classify_eval_error(exc: BaseException) -> str:
    if OpenAICompatClient._is_timeout_error(exc):
        return "api_timeout"
    if isinstance(exc, RuntimeError) and str(exc).startswith("API HTTP"):
        return "api_http_error"
    return "eval_error"


def _instance_timeout_row(inst: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    iid = str(inst.get("instance_id"))
    return {
        "instance_id": iid,
        "outcome": "error",
        "won": None,
        "error": f"instance timeout after {int(timeout_s)}s — skipped",
        "error_type": "instance_timeout",
        "difficulty": _instance_difficulty(inst),
        "turns": None,
    }


def _mp_evaluate_instance(payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Picklable worker for per-instance process timeout (rebuilds API client)."""
    client = OpenAICompatClient(
        payload["api_base"],
        payload["model"],
        api_key=payload.get("api_key", "EMPTY"),
        timeout=int(payload.get("api_timeout", 1800)),
        retries=int(payload.get("api_retries", 3)),
        disable_thinking=bool(payload.get("disable_thinking", True)),
    )
    return _evaluate_instance(
        idx=int(payload["idx"]),
        inst=payload["inst"],
        backend=payload["backend"],
        data_root=payload["data_root"],
        max_turns=int(payload["max_turns"]),
        feedback_mode=payload["feedback_mode"],
        client=client,
        history_length=int(payload["history_length"]),
        max_tokens=int(payload["max_tokens"]),
        temperature=float(payload["temperature"]),
        export_full_transcript=bool(payload.get("export_full_transcript", False)),
        max_rollouts_per_instance=int(payload.get("max_rollouts_per_instance", 1)),
        prior=payload.get("prior"),
        on_partial=None,  # cross-process partial checkpoints not supported
        prompt_mode=payload.get("prompt_mode", "react"),
        encourage_cot=bool(payload.get("encourage_cot", False)),
    )


def _kill_process_tree(proc: "mp.Process", grace_s: float = 5.0) -> None:
    """Terminate a hung instance worker (and best-effort its children)."""
    if proc.pid is None:
        return
    if not proc.is_alive():
        proc.join(timeout=1)
        return
    proc.terminate()
    proc.join(timeout=grace_s)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=grace_s)


def _mp_queue_worker(out_q: "mp.Queue", payload: Dict[str, Any]) -> None:
    """Module-level entry so spawn/pickle works under Process."""
    try:
        out_q.put(("ok", _mp_evaluate_instance(payload)))
    except Exception as exc:  # noqa: BLE001
        iid = str(payload["inst"].get("instance_id"))
        out_q.put((
            "err",
            int(payload["idx"]),
            {
                "instance_id": iid,
                "outcome": "error",
                "won": None,
                "error": str(exc),
                "error_type": _classify_eval_error(exc),
                "difficulty": _instance_difficulty(payload["inst"]),
            },
        ))


def _run_pending_with_instance_timeout(
    *,
    pending: List[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
    workers: int,
    instance_timeout: float,
    base_payload: Dict[str, Any],
    on_done: Callable[[int, Dict[str, Any]], None],
) -> None:
    """Run pending instances in at most ``workers`` processes; skip after timeout.

    Threads cannot be killed when a solution subprocess hangs forever; each
    instance runs in its own process so we can terminate and free the slot.
    """
    if not pending:
        return
    ctx = mp.get_context("spawn")
    pending_iter = iter(pending)
    # job_id -> (Process, Queue, idx, inst, start_t)
    in_flight: Dict[int, Tuple[Any, Any, int, Dict[str, Any], float]] = {}
    next_job_id = 0

    def _submit_one() -> bool:
        nonlocal next_job_id
        try:
            idx, inst, prior = next(pending_iter)
        except StopIteration:
            return False
        payload = dict(base_payload)
        payload.update({"idx": idx, "inst": inst, "prior": prior})
        q: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_mp_queue_worker, args=(q, payload), daemon=True)
        proc.start()
        in_flight[next_job_id] = (proc, q, idx, inst, time.monotonic())
        next_job_id += 1
        return True

    for _ in range(min(workers, len(pending))):
        if not _submit_one():
            break

    while in_flight:
        done_ids: List[int] = []
        now = time.monotonic()
        for job_id, (proc, q, idx, inst, start_t) in list(in_flight.items()):
            elapsed = now - start_t
            finished = not proc.is_alive()
            timed_out = instance_timeout > 0 and elapsed >= instance_timeout

            if not finished and not timed_out:
                continue

            if timed_out and not finished:
                iid = str(inst.get("instance_id"))
                print(
                    f"  !! instance timeout {iid} after {int(elapsed)}s "
                    f"(limit={int(instance_timeout)}s) — killing worker & skipping",
                    flush=True,
                )
                _kill_process_tree(proc)
                row = _instance_timeout_row(inst, instance_timeout)
                on_done(idx, row)
                done_ids.append(job_id)
                continue

            # Process exited — collect result (brief wait for queue put)
            try:
                status, *rest = q.get(timeout=30)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "instance_id": str(inst.get("instance_id")),
                    "outcome": "error",
                    "won": None,
                    "error": f"worker exited without result: {exc}",
                    "error_type": "eval_error",
                    "difficulty": _instance_difficulty(inst),
                }
                on_done(idx, row)
                done_ids.append(job_id)
                proc.join(timeout=1)
                continue

            if status == "ok":
                out_idx, row = rest[0]
                on_done(out_idx, row)
            else:
                # ("err", idx, row_dict)
                out_idx = int(rest[0])
                row = rest[1]
                on_done(out_idx, row)
            proc.join(timeout=1)
            done_ids.append(job_id)

        for job_id in done_ids:
            in_flight.pop(job_id, None)
            _submit_one()

        if not done_ids:
            time.sleep(1.0)


def _summarize_outcomes(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = [r for r in results if r.get("outcome") == "error"]
    graded = [r for r in results if r.get("outcome") in ("won", "lost")]
    wins = sum(1 for r in graded if r.get("outcome") == "won")
    n = len(results)
    n_graded = len(graded)
    n_errors = len(errors)
    by_err = Counter(str(r.get("error_type", "eval_error")) for r in errors)
    return {
        "n_instances": n,
        "n_graded": n_graded,
        "n_errors": n_errors,
        "n_won": wins,
        "n_lost": n_graded - wins,
        "success_rate": wins / n_graded if n_graded else 0.0,
        "success_rate_including_errors_as_fail": wins / n if n else 0.0,
        "error_breakdown": dict(sorted(by_err.items())),
    }


def _summarize_turns(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_turns = [int(r["turns"]) for r in results if isinstance(r.get("turns"), int)]
    won_turns = [
        int(r["turns"]) for r in results
        if r.get("outcome") == "won" and isinstance(r.get("turns"), int)
    ]
    out: Dict[str, Any] = {
        "n_solved": len(won_turns),
        "avg_turns_all": (sum(all_turns) / len(all_turns)) if all_turns else None,
        "avg_turns_won": (sum(won_turns) / len(won_turns)) if won_turns else None,
    }
    if won_turns:
        won_sorted = sorted(won_turns)
        mid = len(won_sorted) // 2
        out["median_turns_won"] = (
            won_sorted[mid] if len(won_sorted) % 2 else (won_sorted[mid - 1] + won_sorted[mid]) / 2
        )
    else:
        out["median_turns_won"] = None
    by_diff: Dict[str, List[int]] = defaultdict(list)
    for r in results:
        if r.get("outcome") == "won" and isinstance(r.get("turns"), int):
            by_diff[str(r.get("difficulty", "unknown"))].append(int(r["turns"]))
    if by_diff:
        out["avg_turns_won_by_difficulty"] = {
            k: sum(v) / len(v) for k, v in sorted(by_diff.items())
        }
    return out


def _load_resume_map(out_path: Path) -> Dict[str, Dict[str, Any]]:
    if not out_path.is_file():
        return {}
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = data.get("per_instance") or []
    return {str(r.get("instance_id")): r for r in rows if r.get("instance_id")}


def _parse_path_list(raw: str) -> List[Path]:
  out: List[Path] = []
  for part in (raw or "").split(","):
    part = part.strip()
    if part:
      out.append(Path(part))
  return out


def _load_exclude_ids(paths: List[Path], mode: str) -> set[str]:
    """Collect instance_ids to drop before sampling.

    mode=won: only rows with outcome=won (SFT top-up — never re-sample solved problems).
    mode=all: every instance_id present in the JSON files.
    """
    exclude: set[str] = set()
    for path in paths:
        rows = _load_resume_map(path)
        for iid, row in rows.items():
            if mode == "all" or row.get("outcome") == "won":
                exclude.add(iid)
    return exclude


def _build_summary(
    *,
    args: argparse.Namespace,
    feedback_mode: str,
    results: List[Dict[str, Any]],
    duration_s: float,
) -> Dict[str, Any]:
    outcome_stats = _summarize_outcomes(results)
    by_diff = defaultdict(list)
    for r in results:
        if r.get("outcome") not in ("won", "lost"):
            continue
        by_diff[str(r.get("difficulty", "unknown"))].append(1.0 if r.get("outcome") == "won" else 0.0)

    metric_docs = _protocol_metric_docs(feedback_mode)
    turn_stats = _summarize_turns(results)
    summary: Dict[str, Any] = {
        "method": args.method_name or args.model,
        "api_base": args.api_base,
        "model": args.model,
        "benchmark": args.benchmark,
        "slice": args.slice or None,
        "sample_size": int(args.sample_size) if args.sample_size > 0 else None,
        "sample_seed": int(args.sample_seed) if args.sample_size > 0 else None,
        "exclude_from": [str(p) for p in getattr(args, "exclude_paths", [])],
        "exclude_from_mode": getattr(args, "exclude_from_mode", None),
        "protocol": {
            "test_feedback_mode": feedback_mode,
            "max_turns": args.max_turns,
            "eval_profile": (
                "closed_api_single_turn" if args.max_turns <= 1 else "local_multi_turn_repair"
            ),
            "history_length": args.history_length,
            "api_timeout_s": args.api_timeout,
            "api_retries": args.api_retries,
            "disable_thinking": bool(args.disable_thinking),
            "prompt_mode": getattr(args, "prompt_mode", "react"),
            "encourage_cot": bool(getattr(args, "encourage_cot", False)),
            "workers": int(getattr(args, "workers", 1)),
            "instance_timeout_s": int(getattr(args, "instance_timeout", 0)),
            "export_full_transcript": bool(getattr(args, "export_full_transcript", False)),
            "max_rollouts_per_instance": int(getattr(args, "max_rollouts_per_instance", 1)),
            **metric_docs,
        },
        "n_instances": outcome_stats["n_instances"],
        "success_rate": outcome_stats["success_rate"],
        "outcome_stats": outcome_stats,
        "turn_stats": turn_stats,
        "duration_s": round(duration_s, 2),
        "difficulty_breakdown": {
            k: (sum(v) / len(v) if v else 0.0) for k, v in sorted(by_diff.items())
        },
        "per_instance": results,
    }
    if args.benchmark == "livecodebench":
        lcb = {d: summary["difficulty_breakdown"].get(d) for d in ("easy", "medium", "hard")
               if d in summary["difficulty_breakdown"]}
        if lcb:
            summary["lcb_difficulty_breakdown"] = lcb
    return summary


def _instance_needs_rollout(row: Optional[Dict[str, Any]], max_rollouts: int) -> bool:
    """Whether an instance still needs collection/eval work."""
    if max_rollouts <= 1:
        return row is None
    if row is None:
        return True
    if row.get("outcome") == "won":
        return False
    attempted = int(row.get("rollouts_attempted") or 0)
    return attempted < max_rollouts


def _evaluate_instance(
    *,
    idx: int,
    inst: Dict[str, Any],
    backend: str,
    data_root: str,
    max_turns: int,
    feedback_mode: str,
    client: OpenAICompatClient,
    history_length: int,
    max_tokens: int,
    temperature: float,
    export_full_transcript: bool = False,
    max_rollouts_per_instance: int = 1,
    prior: Optional[Dict[str, Any]] = None,
    on_partial: Optional[Callable[[Dict[str, Any]], None]] = None,
    prompt_mode: str = "react",
    encourage_cot: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    iid = str(inst.get("instance_id"))
    max_rollouts = max(1, int(max_rollouts_per_instance))
    start_attempt = 1
    rollout_summaries: List[Dict[str, Any]] = []
    if prior:
        rollout_summaries = list(prior.get("rollout_summaries") or [])
        if prior.get("outcome") == "won":
            return idx, prior
        start_attempt = int(prior.get("rollouts_attempted") or 0) + 1

    last_row: Optional[Dict[str, Any]] = None
    for attempt in range(start_attempt, max_rollouts + 1):
        print(
            f"  -> start {iid} rollout {attempt}/{max_rollouts}",
            flush=True,
        )
        env = _make_env(backend, data_root, max_turns, feedback_mode)
        try:
            row = run_episode(
                client, env, inst,
                max_turns=max_turns,
                history_length=history_length,
                max_tokens=max_tokens,
                temperature=temperature,
                test_feedback_mode=feedback_mode,
                export_full_transcript=export_full_transcript,
                prompt_mode=prompt_mode,
                encourage_cot=encourage_cot,
            )
        except Exception as exc:  # noqa: BLE001
            err_type = _classify_eval_error(exc)
            row = {
                "instance_id": iid,
                "outcome": "error",
                "won": None,
                "error": str(exc),
                "error_type": err_type,
                "difficulty": _instance_difficulty(inst),
            }
        row["difficulty"] = row.get("difficulty") or _instance_difficulty(inst)
        row["rollout_attempt"] = attempt
        row["rollouts_attempted"] = attempt
        row["rollout_summaries"] = rollout_summaries + [{
            "attempt": attempt,
            "outcome": row.get("outcome"),
            "turns": row.get("turns"),
            "won": row.get("won"),
        }]
        last_row = row

        if row.get("outcome") == "won":
            return idx, row

        if attempt < max_rollouts and on_partial is not None:
            partial = dict(row)
            partial.pop("transcript", None)
            on_partial(partial)

    assert last_row is not None
    final = dict(last_row)
    # Keep API/eval failures as outcome=error so --retry-errors can pick them up.
    if final.get("outcome") != "error":
        final["outcome"] = "lost"
        final["won"] = False
    final["rollouts_attempted"] = max_rollouts
    final.pop("transcript", None)
    return idx, final


def main() -> int:
    parser = argparse.ArgumentParser(description="API baseline eval (multi-turn self-repair)")
    parser.add_argument("--api-base", required=True)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")),
        help="Bearer token (default: API_KEY or OPENAI_API_KEY env, else EMPTY)",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="light_eval")
    parser.add_argument("--data-root", default=os.environ.get("SWEBENCH_DATA_ROOT", "/mnt/z4/solariewang/datasets"))
    parser.add_argument("--slice", default="", help="Optional instance_slice, e.g. 0:8")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=int(os.environ.get("EVAL_SAMPLE_SIZE", "0")),
        help="Random sample N instances after load/filter (0=all). Use with --sample-seed.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=int(os.environ.get("EVAL_SAMPLE_SEED", "42")),
        help="RNG seed for --sample-size (default 42)",
    )
    parser.add_argument(
        "--instance-filter",
        default=os.environ.get("EVAL_INSTANCE_FILTER", ""),
        help="Regex on instance_id (e.g. 'platinum' for USACO platinum only)",
    )
    parser.add_argument(
        "--difficulty-filter",
        default=os.environ.get("EVAL_DIFFICULTY_FILTER", ""),
        help="Comma-separated difficulty tiers (e.g. 'hard' for OJBench hard only)",
    )
    parser.add_argument(
        "--exclude-from",
        action="append",
        default=[],
        help="JSON file(s) whose instance_ids should be excluded before sampling "
        "(repeat flag or comma-separated paths via EVAL_EXCLUDE_FROM)",
    )
    parser.add_argument(
        "--exclude-from-mode",
        choices=("won", "all"),
        default=os.environ.get("EVAL_EXCLUDE_FROM_MODE", "won"),
        help="won=drop only outcome=won rows; all=drop every instance_id in --exclude-from",
    )
    parser.add_argument("--min-date", default=os.environ.get("LCB_MIN_DATE", "2025-02-01"))
    parser.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("EVAL_MAX_TURNS", "12")),
        help="1 = single-turn pass@1 (closed API baselines); 12+ = multi-turn self-repair (local RL)",
    )
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Cap number of instances (0=all)")
    parser.add_argument("--out", default="")
    parser.add_argument("--method-name", default="")
    parser.add_argument(
        "--test-feedback-mode",
        choices=("blind", "exec", "interactive", "oracle"),
        default=os.environ.get("TEST_FEEDBACK_MODE", "exec"),
        help="blind | exec (default) | interactive | oracle; "
        "exec: run program with agent stdin (stdout/stderr only); "
        "interactive: test feedback during repair — compare avg_turns_won (lower=better)",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=int(os.environ.get("API_TIMEOUT", "1800")),
        help="Per-request HTTP timeout seconds (default 1800; Kimi reasoning can be slow)",
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=int(os.environ.get("API_RETRIES", "3")),
        help="Retries on API timeout only (default 3)",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("react", "freeform", "self_planning"),
        default=os.environ.get("PROMPT_MODE", "react"),
        help=(
            "react=ReAct Thought/Action XML (CodeAct); "
            "freeform=unrestricted code generation; "
            "self_planning=generate a numbered plan then code from the plan"
        ),
    )
    parser.add_argument(
        "--encourage-cot",
        action="store_true",
        default=os.environ.get("ENCOURAGE_COT", "0").lower() in ("1", "true", "yes"),
        help="With --prompt-mode freeform: ask the model to think step-by-step before coding",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        help="Kimi/vLLM: chat_template_kwargs.thinking=false; parse content only",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="Keep model reasoning/thinking enabled",
    )
    parser.set_defaults(
        disable_thinking=os.environ.get("DISABLE_THINKING", "1").lower() in ("1", "true", "yes"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing --out JSON (skip finished instance_ids)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing --out JSON and re-run all instances",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        default=False,
        help="With --resume: re-run rows whose outcome is error (keeps won/lost)",
    )
    parser.add_argument(
        "--retry-error-types",
        default=os.environ.get("EVAL_RETRY_ERROR_TYPES", "api_http_error,eval_error"),
        help="Comma-separated error_type values to retry when --retry-errors is set",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(os.environ.get("EVAL_CHECKPOINT_EVERY", "10")),
        help="Write partial JSON every N newly finished instances (0=only at end)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("EVAL_WORKERS", "1")),
        help="Concurrent API eval workers (default 1; use 5 for Kimi APPS)",
    )
    parser.add_argument(
        "--instance-timeout",
        type=int,
        default=int(os.environ.get("EVAL_INSTANCE_TIMEOUT", "3600")),
        help="Per-instance wall-clock timeout seconds (default 3600). "
        "On timeout: kill worker process, record outcome=error/"
        "error_type=instance_timeout, continue. 0 disables.",
    )
    parser.add_argument(
        "--export-full-transcript",
        action="store_true",
        default=os.environ.get("EXPORT_FULL_TRANSCRIPT", "0").lower() in ("1", "true", "yes"),
        help="Save full user_prompt + assistant_response per turn (for SFT collection)",
    )
    parser.add_argument(
        "--max-rollouts-per-instance",
        type=int,
        default=int(os.environ.get("SFT_MAX_ROLLOUTS", "1")),
        help="Up to N rollouts per instance; keep first won transcript (SFT: 5)",
    )
    args = parser.parse_args()
    if args.no_resume:
        args.resume = False
    elif not args.resume and os.environ.get("EVAL_RESUME", "0").lower() in ("1", "true", "yes"):
        args.resume = True
    if os.environ.get("EVAL_RETRY_ERRORS", "0").lower() in ("1", "true", "yes"):
        args.retry_errors = True
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.instance_timeout < 0:
        raise SystemExit("--instance-timeout must be >= 0")
    if args.max_rollouts_per_instance < 1:
        raise SystemExit("--max-rollouts-per-instance must be >= 1")

    data_root = Path(args.data_root)
    if args.benchmark not in _BENCHMARK_SPECS:
        raise SystemExit(f"unsupported benchmark for API eval: {args.benchmark}")

    instances, backend = _load_benchmark_instances(
        args.benchmark,
        data_root,
        slice_spec=args.slice,
        min_date=args.min_date if args.benchmark == "livecodebench" else "",
    )
    if args.limit > 0 and args.sample_size <= 0:
        instances = instances[: args.limit]
    filter_spec = (args.instance_filter or "").strip()
    difficulty_filter = (args.difficulty_filter or "").strip()
    difficulty_tiers = {
        t.strip().lower() for t in difficulty_filter.split(",") if t.strip()
    }
    if filter_spec:
        before = len(instances)
        instances = [
            inst for inst in instances
            if re.search(filter_spec, str(inst.get("instance_id", "")))
        ]
        print(f"[instance-filter] {filter_spec!r}: {before} -> {len(instances)}", flush=True)
        if not instances:
            raise SystemExit(f"--instance-filter {filter_spec!r} matched 0 instances")
    if difficulty_tiers:
        before = len(instances)
        instances = [
            inst for inst in instances
            if _instance_difficulty(inst) in difficulty_tiers
        ]
        print(
            f"[difficulty-filter] {sorted(difficulty_tiers)}: {before} -> {len(instances)}",
            flush=True,
        )
        if not instances:
            raise SystemExit(
                f"--difficulty-filter {difficulty_filter!r} matched 0 instances"
            )
    exclude_paths: List[Path] = []
    for item in args.exclude_from:
        exclude_paths.extend(_parse_path_list(item))
    exclude_paths.extend(_parse_path_list(os.environ.get("EVAL_EXCLUDE_FROM", "")))
    if exclude_paths:
        exclude_ids = _load_exclude_ids(exclude_paths, args.exclude_from_mode)
        before = len(instances)
        instances = [
            inst for inst in instances
            if str(inst.get("instance_id")) not in exclude_ids
        ]
        print(
            f"[exclude-from] mode={args.exclude_from_mode} files={len(exclude_paths)} "
            f"excluded={before - len(instances)} remaining={len(instances)}",
            flush=True,
        )
        if not instances:
            raise SystemExit("--exclude-from removed all instances; nothing left to run")
    args.exclude_paths = exclude_paths
    if args.sample_size > 0:
        instances = _apply_random_sample(instances, args.sample_size, args.sample_seed)
    partial_subset = bool(filter_spec or difficulty_tiers or args.sample_size > 0)
    filtered_ids = {str(i["instance_id"]) for i in instances}

    feedback_mode = args.test_feedback_mode
    if feedback_mode == "oracle":
        feedback_mode = "interactive"
    client = OpenAICompatClient(
        args.api_base,
        args.model,
        api_key=args.api_key,
        timeout=args.api_timeout,
        retries=args.api_retries,
        disable_thinking=args.disable_thinking,
    )

    out_path = Path(args.out) if args.out else REPO / "logs" / f"eval_api_{args.model.replace('/', '_')}_{args.benchmark}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    original_map = _load_resume_map(out_path) if args.resume else {}
    results_by_id: Dict[str, Dict[str, Any]] = dict(original_map)
    resume_map: Dict[str, Dict[str, Any]] = dict(original_map)
    if args.retry_errors:
        if not args.resume:
            raise SystemExit("--retry-errors requires --resume")
        retry_types = {t.strip() for t in args.retry_error_types.split(",") if t.strip()}
        to_retry = {
            iid for iid, row in original_map.items()
            if row.get("outcome") == "error"
            and str(row.get("error_type", "eval_error")) in retry_types
        }
        if partial_subset:
            to_retry &= filtered_ids
        for iid in to_retry:
            results_by_id.pop(iid, None)
            resume_map.pop(iid, None)
        print(
            f"[retry-errors] re-run {len(to_retry)} instances (types={sorted(retry_types)}); "
            f"keep {len(resume_map)} finished rows",
            flush=True,
        )
    if resume_map:
        print(f"[resume] loaded {len(original_map)} rows from {out_path}", flush=True)

    def _ordered_results() -> List[Dict[str, Any]]:
        if partial_subset and original_map:
            out: List[Dict[str, Any]] = []
            for iid, row in original_map.items():
                if iid in filtered_ids:
                    if iid in results_by_id:
                        out.append(results_by_id[iid])
                else:
                    out.append(results_by_id.get(iid, row))
            return out
        out: List[Dict[str, Any]] = []
        for inst in instances:
            iid = str(inst.get("instance_id"))
            if iid in results_by_id:
                out.append(results_by_id[iid])
        return out

    pending: List[Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]] = []
    for idx, inst in enumerate(instances):
        iid = str(inst.get("instance_id"))
        prior = resume_map.get(iid)
        if _instance_needs_rollout(prior, args.max_rollouts_per_instance):
            pending.append((idx, inst, prior))
    total = len(instances)
    print(
        f"[eval] benchmark={args.benchmark} pending={len(pending)}/{total} "
        f"workers={args.workers} rollouts_per_instance={args.max_rollouts_per_instance} "
        f"instance_timeout={args.instance_timeout}s "
        f"thinking={'off' if args.disable_thinking else 'on'} "
        f"prompt_mode={args.prompt_mode}",
        flush=True,
    )

    t0 = time.perf_counter()
    new_since_ckpt = 0
    lock = threading.Lock()

    def _on_done(idx: int, row: Dict[str, Any]) -> None:
        nonlocal new_since_ckpt
        iid = str(row.get("instance_id"))
        with lock:
            results_by_id[iid] = row
            new_since_ckpt += 1
            done_n = len(results_by_id)
            if row.get("outcome") == "error":
                print(
                    f"[{done_n}/{total}] {iid} ERROR ({row.get('error_type')}) "
                    f"— not counted as model failure",
                    flush=True,
                )
            else:
                attempt = row.get("rollout_attempt")
                extra = ""
                if attempt is not None and args.max_rollouts_per_instance > 1:
                    extra = f" rollout={attempt}/{args.max_rollouts_per_instance}"
                print(
                    f"[{done_n}/{total}] {iid} outcome={row.get('outcome')} "
                    f"turns={row.get('turns', '?')}{extra}",
                    flush=True,
                )
            if args.checkpoint_every > 0 and new_since_ckpt >= args.checkpoint_every:
                results = _ordered_results()
                summary = _build_summary(
                    args=args, feedback_mode=feedback_mode, results=results,
                    duration_s=time.perf_counter() - t0,
                )
                out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                new_since_ckpt = 0
                print(f"  [checkpoint] wrote {len(results)}/{total} -> {out_path}", flush=True)

    def _checkpoint_partial(row: Dict[str, Any]) -> None:
        iid = str(row.get("instance_id"))
        with lock:
            results_by_id[iid] = row

    if args.instance_timeout > 0:
        base_payload = {
            "api_base": args.api_base,
            "model": args.model,
            "api_key": args.api_key,
            "api_timeout": args.api_timeout,
            "api_retries": args.api_retries,
            "disable_thinking": args.disable_thinking,
            "backend": backend,
            "data_root": str(data_root),
            "max_turns": args.max_turns,
            "feedback_mode": feedback_mode,
            "history_length": args.history_length,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "export_full_transcript": args.export_full_transcript,
            "max_rollouts_per_instance": args.max_rollouts_per_instance,
            "prompt_mode": args.prompt_mode,
            "encourage_cot": args.encourage_cot,
        }
        _run_pending_with_instance_timeout(
            pending=pending,
            workers=args.workers,
            instance_timeout=float(args.instance_timeout),
            base_payload=base_payload,
            on_done=_on_done,
        )
    elif args.workers == 1:
        for idx, inst, prior in pending:
            _, row = _evaluate_instance(
                idx=idx, inst=inst, backend=backend, data_root=str(data_root),
                max_turns=args.max_turns, feedback_mode=feedback_mode, client=client,
                history_length=args.history_length, max_tokens=args.max_tokens,
                temperature=args.temperature,
                export_full_transcript=args.export_full_transcript,
                max_rollouts_per_instance=args.max_rollouts_per_instance,
                prior=prior,
                on_partial=_checkpoint_partial if args.max_rollouts_per_instance > 1 else None,
                prompt_mode=args.prompt_mode,
                encourage_cot=args.encourage_cot,
            )
            _on_done(idx, row)
    else:
        pending_iter = iter(pending)
        in_flight: Dict[Any, Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]] = {}

        def _submit(pool: ThreadPoolExecutor) -> None:
            try:
                idx, inst, prior = next(pending_iter)
            except StopIteration:
                return
            fut = pool.submit(
                _evaluate_instance,
                idx=idx, inst=inst, backend=backend, data_root=str(data_root),
                max_turns=args.max_turns, feedback_mode=feedback_mode, client=client,
                history_length=args.history_length, max_tokens=args.max_tokens,
                temperature=args.temperature,
                export_full_transcript=args.export_full_transcript,
                max_rollouts_per_instance=args.max_rollouts_per_instance,
                prior=prior,
                on_partial=_checkpoint_partial if args.max_rollouts_per_instance > 1 else None,
                prompt_mode=args.prompt_mode,
                encourage_cot=args.encourage_cot,
            )
            in_flight[fut] = (idx, inst, prior)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in range(min(args.workers, len(pending))):
                _submit(pool)
            while in_flight:
                for fut in as_completed(list(in_flight.keys())):
                    in_flight.pop(fut)
                    idx, row = fut.result()
                    _on_done(idx, row)
                    _submit(pool)
                    break

    results = _ordered_results()
    summary = _build_summary(
        args=args, feedback_mode=feedback_mode, results=results,
        duration_s=time.perf_counter() - t0,
    )
    outcome_stats = summary["outcome_stats"]
    n = outcome_stats["n_instances"]
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print_payload: Dict[str, Any] = {
        "out": str(out_path),
        "n": n,
        "protocol": feedback_mode,
        "disable_thinking": args.disable_thinking,
        "primary_metric": summary["protocol"]["primary_metric"],
        "success_rate": summary["success_rate"],
        "outcome_stats": outcome_stats,
        "turn_stats": summary["turn_stats"],
        "difficulty_breakdown": summary["difficulty_breakdown"],
    }
    print(json.dumps(print_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
