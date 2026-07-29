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

"""Fixed-``instance_id`` DIDPO group evolution tracker.

Locks a small set of **stable** SWE-bench / APPS ``instance_id``s (not ephemeral
rollout ``uid`` UUIDs) and, at every training step those instances appear,
records DiDPO group stats (count, sizes, previews) for SwanLab + JSONL.

Also writes ``tracked_instance_ids.json`` so the env can pin those instances
into subsequent batches (see ``env.swebench.tracked_instance_ids_file``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# Ephemeral rollout uid UUIDs must never be locked as "instance_ids".
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_stable_instance_id(value: Any) -> bool:
    s = str(value).strip()
    if not s:
        return False
    if _UUID_RE.match(s):
        return False
    return True


class DidpoGroupTracker:
    def __init__(
        self,
        dump_dir: Optional[str] = None,
        track_n: int = 3,
        track_instance_ids: Optional[Sequence[str]] = None,
        track_uids: Optional[Sequence[str]] = None,  # legacy alias
        preview_groups: int = 8,
    ):
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.track_n = max(0, int(track_n))
        self.preview_groups = max(1, int(preview_groups))
        # Prefer explicit instance ids; fall back to legacy track_uids name.
        seed = track_instance_ids if track_instance_ids is not None else track_uids
        self._fixed: List[str] = [
            str(u) for u in (seed or []) if _is_stable_instance_id(u)
        ]
        self._locked = bool(self._fixed)
        if self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl = self.dump_dir / "didpo_prompt_groups.jsonl"
            self._tracked_file = self.dump_dir / "tracked_instance_ids.json"
            # Resume: reload previously locked instance ids if any.
            if not self._fixed and self._tracked_file.exists():
                try:
                    data = json.loads(self._tracked_file.read_text(encoding="utf-8"))
                    ids = data.get("instance_ids") or data.get("uids") or []
                    cleaned = [
                        str(x) for x in ids if _is_stable_instance_id(x)
                    ][: self.track_n]
                    if cleaned:
                        self._fixed = cleaned
                        self._locked = True
                    else:
                        # Stale UUID-only file from the old tracker — ignore.
                        print(
                            "[DidpoGroupTracker] WARN: tracked_instance_ids.json "
                            "had no stable instance_ids (UUID-only?); re-locking "
                            "from the next batch."
                        )
                except Exception:  # noqa: BLE001
                    pass
            if self._fixed:
                self._persist_tracked()
        else:
            self._jsonl = None
            self._tracked_file = None

    @property
    def tracked_instance_ids(self) -> List[str]:
        return list(self._fixed)

    # Back-compat alias used by older call sites / docs.
    @property
    def tracked_uids(self) -> List[str]:
        return self.tracked_instance_ids

    def observe_batch_instance_ids(self, instance_ids: Iterable[Any]) -> None:
        """Auto-register the first ``track_n`` unique instance_ids if unlocked."""
        if self._locked or self.track_n <= 0:
            return
        seen: Set[str] = set(self._fixed)
        grew = False
        for u in instance_ids:
            su = str(u)
            if not _is_stable_instance_id(su) or su in seen:
                continue
            self._fixed.append(su)
            seen.add(su)
            grew = True
            if len(self._fixed) >= self.track_n:
                self._locked = True
                break
        if grew:
            self._persist_tracked()

    # Back-compat name
    def observe_batch_uids(self, uids: Iterable[Any]) -> None:
        self.observe_batch_instance_ids(uids)

    def _persist_tracked(self) -> None:
        if self._tracked_file is None:
            return
        payload = {
            "instance_ids": list(self._fixed),
            "locked": bool(self._locked),
        }
        self._tracked_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def step(
        self,
        global_step: int,
        diagnostics: Dict[str, Any],
    ) -> Dict[str, float]:
        """Record tracked instances for this step; return SwanLab scalars."""
        metrics: Dict[str, float] = {}
        # Prefer stable instance-keyed stats.
        # IMPORTANT: empty dict must NOT fall through to per_uid (UUID keys).
        if "per_instance" in diagnostics:
            per = diagnostics.get("per_instance") or {}
        else:
            per = diagnostics.get("per_uid") or {}
        if not self._fixed:
            self.observe_batch_instance_ids(per.keys())
        if not self._fixed:
            return metrics

        records = []
        for iid in self._fixed:
            info = per.get(iid) or per.get(str(iid))
            short = _short(iid)
            if not info:
                metrics[f"didpo/instance/{short}/present"] = 0.0
                # Keep legacy metric name so existing SwanLab panels still update.
                metrics[f"didpo/prompt/{short}/present"] = 0.0
                continue
            metrics[f"didpo/instance/{short}/present"] = 1.0
            metrics[f"didpo/instance/{short}/n_groups"] = float(info.get("n_groups", 0))
            metrics[f"didpo/instance/{short}/mean_group_size"] = float(
                info.get("mean_group_size", 0.0)
            )
            metrics[f"didpo/instance/{short}/median_group_size"] = float(
                info.get("median_group_size", 0.0)
            )
            metrics[f"didpo/instance/{short}/singleton_rate"] = float(
                info.get("singleton_rate", 0.0)
            )
            # Legacy aliases
            metrics[f"didpo/prompt/{short}/present"] = 1.0
            metrics[f"didpo/prompt/{short}/n_groups"] = float(info.get("n_groups", 0))
            metrics[f"didpo/prompt/{short}/mean_group_size"] = float(
                info.get("mean_group_size", 0.0)
            )
            metrics[f"didpo/prompt/{short}/median_group_size"] = float(
                info.get("median_group_size", 0.0)
            )
            metrics[f"didpo/prompt/{short}/singleton_rate"] = float(
                info.get("singleton_rate", 0.0)
            )
            groups = list(info.get("groups") or [])[: self.preview_groups]
            records.append({
                "step": int(global_step),
                "instance_id": iid,
                "uid": iid,  # back-compat for old readers
                "n_groups": int(info.get("n_groups", 0)),
                "mean_group_size": float(info.get("mean_group_size", 0.0)),
                "median_group_size": float(info.get("median_group_size", 0.0)),
                "singleton_rate": float(info.get("singleton_rate", 0.0)),
                "groups": groups,
            })

        if self.dump_dir is not None and records:
            with open(self._jsonl, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for rec in records:
                snap = self.dump_dir / f"instance_{_safe(rec['instance_id'])}_latest.json"
                with open(snap, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)

        return metrics


def _safe(uid: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(uid))[:64]


def _short(uid: str) -> str:
    s = str(uid)
    return s if len(s) <= 24 else s[:24]
