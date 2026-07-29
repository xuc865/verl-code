#!/usr/bin/env python3
"""Preflight imports for CodeRL GRPO (no-LoRA, text CausalLM) before Ray starts.

Fails fast on the same class of ImportError that burned recent launches:
  - transformers without AutoModelForVision2Seq (text path must still load CausalLM)
  - vllm_utils / vllm_rollout_spmd LoRA hard-imports
  - apps_train_coderl preset resolution
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    repo = os.environ.get("REPO", "/mnt/z4/solariewang/verl-swe")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    print("[preflight] python", sys.version.split()[0])

    import transformers
    import vllm

    print("[preflight] transformers", getattr(transformers, "__version__", "?"))
    print("[preflight] vllm", getattr(vllm, "__version__", "?"))

    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: F401

    try:
        from transformers import AutoModelForVision2Seq  # noqa: F401

        print("[preflight] AutoModelForVision2Seq: ok")
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText  # noqa: F401

            print("[preflight] AutoModelForVision2Seq: missing; ImageTextToText ok")
        except ImportError:
            print("[preflight] AutoModelForVision2Seq/ImageTextToText: missing (ok for text GRPO)")

    from verl.utils.vllm_utils import is_version_ge, _VLLM_LORA_AVAILABLE

    print("[preflight] vllm_utils.is_version_ge ok; lora_available=", _VLLM_LORA_AVAILABLE)
    _ = is_version_ge(pkg="vllm", minver="0.7.3")

    # SwanLab: local mode must skip login; Ray workers must receive SWANLAB_MODE.
    os.environ.setdefault("SWANLAB_MODE", "local")
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    re = get_ppo_ray_runtime_env()
    ev = re.get("env_vars") or {}
    assert ev.get("SWANLAB_MODE") == "local" or os.environ.get("SWANLAB_MODE") == "local"
    if "SWANLAB_MODE" not in ev:
        print("[preflight] WARN: SWANLAB_MODE not in ray runtime_env yet — export before ray.init")
    else:
        print("[preflight] ray runtime_env SWANLAB_MODE=", ev.get("SWANLAB_MODE"))

    mode = os.environ.get("SWANLAB_MODE", "cloud")
    assert mode in ("local", "offline", "disabled"), f"expected local swanlab mode, got {mode}"
    print("[preflight] swanlab mode=", mode, "(login skipped for local)")

    from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout  # noqa: F401

    print("[preflight] vllm_rollout mode=", vllm_mode)

    from agent_system.environments.env_package.swebench.envs import (
        _BENCHMARK_PRESETS,
        resolve_benchmark,
    )

    assert "apps_train_coderl" in _BENCHMARK_PRESETS, "apps_train_coderl preset missing"
    cfg = resolve_benchmark(
        {
            "benchmark": "apps_train_coderl",
            "data_root": os.environ.get(
                "SWEBENCH_DATA_ROOT", "/mnt/z4/solariewang/datasets"
            ),
        }
    )
    assert cfg.get("backend") == "mixed", cfg
    print("[preflight] apps_train_coderl preset ok")

    # Smoke the exact fsdp_workers import block used at worker init.
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
        except ImportError:
            AutoModelForVision2Seq = None
    assert AutoModelForCausalLM is not None
    print("[preflight] fsdp_workers-style HF imports ok; Vision2Seq=", AutoModelForVision2Seq is not None)
    print("[preflight] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
