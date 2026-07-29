# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # symmetric memory allreduce not work properly in spmd mode
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}

# Forwarded into Ray workers when set on the driver (mt8_v4 / one.sh train-node pattern).
# Without this, TaskRunner may miss SWANLAB_MODE=local and fall back to cloud login.
_PPO_RAY_FORWARDED_ENV = (
    "SWANLAB_MODE",
    "SWANLAB_LOG_DIR",
    "SWANLAB_API_KEY",
    "SWANLAB_API_KEY_FILE",
    "SWANLAB_RUN_ID",
    "SWANLAB_ID",
    "SWANLAB_RESUME",
    "NO_PROXY",
    "no_proxy",
    "HF_ENDPOINT",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "SWEBENCH_DATA_ROOT",
    "SWEBENCH_TIMING_LOG",
    "VLLM_ATTENTION_BACKEND",
    "PYTORCH_CUDA_ALLOC_CONF",
)


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)

    # Explicitly inject train-node logging / data env into workers.
    for key in _PPO_RAY_FORWARDED_ENV:
        val = os.environ.get(key)
        if val is not None and val != "":
            runtime_env["env_vars"][key] = val

    return runtime_env
