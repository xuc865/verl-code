# DiDPO — Diff-in-Diff Policy Optimization

DiDPO is a critic-free RL algorithm for **multi-turn coding agents**. It keeps
the episode-level advantage of GRPO and adds a finer-grained local advantage on
aligned code snippets inside a response.

In short:

- **GRPO** assigns credit to the whole response.
- **DiDPO** further assigns credit to the edited parts inside that response.

---

## Quick Summary

| Item | Description |
|---|---|
| Algorithm type | Critic-free RL |
| Main idea | Snippet-level credit assignment for code edits |
| Training cost | No extra rollouts; mainly CPU-side alignment overhead |
| Main benchmark preset | `apps_train_coderl` |
| Main launcher | `scripts/launch_didpo_coderl_sft_mt8.sh` |

---

## Code Map

| Component | File |
|---|---|
| snippet extraction | `didpo/snippet.py` |
| grouping / local advantage | `didpo/core_didpo.py` |
| tracked dumps | `didpo/group_tracker.py` |
| tests | `didpo/tests/` |
| plots | `didpo/plots/` |

---

## Main Workflow

```text
trajectory collection
    -> SFT data build
    -> SFT
    -> DiDPO RL
```

### Build SFT data

```bash
cd verl-code
bash scripts/build_apps_mt8_sft_dataset.sh
```

### Run SFT

```bash
bash examples/sft/apps_mt8/run_apps_mt8_sft.sh
```

or

```bash
bash examples/sft/apps_mt8/run_apps_mt8_sft_qwen35_4b.sh
```

### Run DiDPO

```bash
bash scripts/launch_didpo_coderl_sft_mt8.sh
```

### Run Qwen3.5-4B DiDPO

```bash
MODEL_PATH=checkpoints/apps_mt8_sft_qwen35_4b_think/global_step_162 \
EXP_NAME=didpo_coderl_qwen35_4b_sft_mt8 \
PROJECT_NAME=didpo_coderl \
bash scripts/launch_didpo_coderl_sft_mt8.sh
```

---

## Dataset

- **SFT dataset (HF):** [xuc865/DiDPO-SFT-Data](https://huggingface.co/datasets/xuc865/DiDPO-SFT-Data)
- **RL dataset (HF):** [PRIME-RL/Eurus-2-SFT-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-SFT-Data)

---

## More Details

- Hyperparameters: [`didpo/HYPERPARAMETERS.md`](./HYPERPARAMETERS.md)
- Project card: [`didpo/PROJECT_CARD.md`](./PROJECT_CARD.md)
- Metadata: [`didpo/PROJECT_METADATA.yaml`](./PROJECT_METADATA.yaml)
