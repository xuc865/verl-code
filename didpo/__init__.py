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
DIDPO -- Diff-in-Diff Policy Optimization.

Paper-aligned critic-free RLVR for coding agents:
  - dynamic sub-diff anchors via cross-rollout similarity matching
  - greedy GS facility-location selection of anchors
  - structural (AST) fallback when alignment is sparse
  - episode GRPO advantage + λ · diff-level group advantage

Public surfaces:
- ``didpo.snippet``       : collector-side root-diff extraction + fallback + gating
- ``didpo.core_didpo``    : trainer-side alignment, GS, advantage fill, diagnostics
- ``didpo.group_tracker`` : fixed-prompt group evolution dumps for SwanLab
"""

__all__ = ["snippet", "core_didpo", "group_tracker"]


def __getattr__(name):  # PEP 562 lazy submodule import
    if name in __all__:
        import importlib
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
