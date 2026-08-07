# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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
Environment managers.

Multi-turn coding agents (GRPO / GiGPO / DiDPO). Legacy verl-agent environments
(ALFWorld / WebShop / Sokoban / Gym Cards / AppWorld / Search) have been
deprecated and removed from the main path.
"""

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
from functools import partial
import os

import numpy as np

from agent_system.environments.prompts import (
    SWEBENCH_TEMPLATE,
    SWEBENCH_TEMPLATE_NO_HIS,
    _ACTION_SPEC,
)
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from omegaconf import OmegaConf


class SWEBenchEnvironmentManager(EnvironmentManagerBase):
    """Environment manager for multi-turn coding agents.

    Produces text observations that interleave the task, a short interaction
    history (managed by :class:`SimpleMemory`), and the latest command output.

    GiGPO uses ``anchor`` = post-step workspace code state (no history).
    DiDPO ignores ``anchor`` and groups on snippets instead.
    """

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.problem_statements: List[str] = []
        super().__init__(envs, projection_f, config)

    @staticmethod
    def _anchors_from(obs: List[str], infos: List[Dict]) -> List[Any]:
        """Prefer env-provided workspace state; fall back to raw obs."""
        out: List[Any] = []
        for i, info in enumerate(infos):
            a = info.get("anchor")
            out.append(a if a is not None else obs[i])
        return out

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset()
        self.problem_statements = [info.get("problem_statement", "") for info in infos]
        self.memory.reset(batch_size=len(infos))
        observations = {
            "text": self._build_init_obs(obs),
            "image": None,
            "anchor": self._anchors_from(obs, infos),
        }
        for i, info in enumerate(infos):
            info["is_action_valid"] = np.array(True)
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # store a compact rendering of the action for the history
        rendered_actions = [self._render_action(a) for a in actions]
        self.memory.store({"text_obs": next_obs, "action": rendered_actions})

        observations = {
            "text": self.build_text_obs(next_obs, infos),
            "image": None,
            "anchor": self._anchors_from(next_obs, infos),
        }
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            info["tool_calling"] = 1.0 if actions[i].get("type") in ("edit", "bash") else 0.0

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_action(action: Dict[str, Any]) -> str:
        t = action.get("type", "noop")
        if t == "edit":
            return f"edit {action.get('path','')}"
        if t == "bash":
            return f"bash: {action.get('cmd','')[:80]}"
        return t

    def _build_init_obs(self, repo_views: List[str]) -> List[str]:
        return [
            SWEBENCH_TEMPLATE_NO_HIS.format(repo_view=rv, action_spec=_ACTION_SPEC)
            for rv in repo_views
        ]

    def build_text_obs(self, text_obs: List[str], infos: List[Dict], init: bool = False) -> List[str]:
        out: List[str] = []
        history_length = self.config.env.history_length
        if not init and history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                history_length, obs_key="text_obs", action_key="action")
        for i in range(len(text_obs)):
            if init or history_length <= 0:
                out.append(SWEBENCH_TEMPLATE_NO_HIS.format(
                    repo_view=text_obs[i], action_spec=_ACTION_SPEC))
            else:
                out.append(SWEBENCH_TEMPLATE.format(
                    problem_statement=self.problem_statements[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    action_spec=_ACTION_SPEC,
                ))
        return out

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                success["success_rate"].append(float(info["won"]))
                return


def make_envs(config):
    """Create train/val coding environments (GRPO / GiGPO / DiDPO)."""
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "swebench" in config.env.env_name.lower():
        # DiDPO: wire tracked_instance_ids_file before env construction so
        # reset() can pin stable instance_ids into every batch.
        try:
            if str(getattr(config.algorithm, "adv_estimator", "")).lower() == "didpo":
                import os
                from omegaconf import open_dict
                didpo_cfg = config.algorithm.get("didpo", {}) or {}
                dump = didpo_cfg.get("group_dump_dir") or "logs/didpo_groups"
                if not os.path.isabs(str(dump)):
                    dump = os.path.join(os.getcwd(), str(dump))
                os.makedirs(dump, exist_ok=True)
                track_file = os.path.join(dump, "tracked_instance_ids.json")
                with open_dict(config.env.swebench):
                    cur = config.env.swebench.get("tracked_instance_ids_file", None)
                    if cur in (None, ""):
                        config.env.swebench.tracked_instance_ids_file = track_file
                    elif not os.path.isabs(str(cur)):
                        config.env.swebench.tracked_instance_ids_file = os.path.join(
                            os.getcwd(), str(cur)
                        )
                print(f"[make_envs] DiDPO tracked_instance_ids_file="
                      f"{config.env.swebench.tracked_instance_ids_file}")
        except Exception as exc:  # noqa: BLE001
            print(f"[make_envs] WARN: could not wire DiDPO track file: {exc}")

        from agent_system.environments.env_package.swebench import (
            build_swebench_envs, swebench_projection)
        _envs = build_swebench_envs(
            env_config=config.env, env_num=config.data.train_batch_size,
            group_n=group_n, resources_per_worker=resources_per_worker,
            is_train=True, seed=config.env.seed)
        _val_envs = build_swebench_envs(
            env_config=config.env, env_num=config.data.val_batch_size,
            group_n=1, resources_per_worker=resources_per_worker,
            is_train=False, seed=config.env.seed + 1000)
        projection_f = partial(swebench_projection)
        envs = SWEBenchEnvironmentManager(_envs, projection_f, config)
        val_envs = SWEBenchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs

    raise ValueError(
        f"Unsupported environment: {config.env.env_name}. "
        "Only the coding env (env_name containing 'swebench') is supported; "
        "legacy environments were deprecated.")
