## Memory Manager

`code-swe` lets each rollout step flexibly choose what interaction history to
include — e.g. recent steps, key events, summaries, or external knowledge —
rather than always concatenating the full trajectory. In the SWE-bench setting
this keeps the per-step context bounded even when an episode runs many
bash/edit turns.

We provide a simple memory implementation as a starting point (see
`memory.py`). It is invoked from
`agent_system/environments/env_manager.py` (`build_text_obs()`) to construct the
observation at each step. Developers are encouraged to extend this module with
custom strategies such as dynamic summarization, selective retention, or
external-knowledge integration to improve handling of long coding trajectories.

> Note: this memory module is the visibility premise behind DIDPO's
> "same observation, different history" case — see `didpo/README.md` §7. If the
> discriminative context is truncated out of the window by aggressive
> compression here, snippet grouping quality can degrade.
