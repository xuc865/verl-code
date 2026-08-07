# DiDPO / CodeRL 项目展示卡

## 项目名称

**DiDPO for Multi-turn CodeRL**

副标题：

**Diff-in-Diff Policy Optimization for code-editing agents**

---

## 一句话简介

这是一个面向多轮代码修复智能体的训练项目：先用多轮轨迹做 SFT，再在
`apps_train_coderl` 等任务上做 RL，并用 **DiDPO** 给代码编辑响应内部的
不同片段分配更细粒度的 credit。

---

## 项目亮点

- 面向 **多轮 coding agent**，而不是单轮 code generation
- 支持 **SFT -> RL** 的完整训练链路
- DiDPO 在 GRPO 的 episode advantage 之上，进一步做 **snippet-level**
  diff credit assignment
- 兼容现有的 CodeRL / APPS-style self-repair 环境
- 提供实际可运行的：
  - 数据构建脚本
  - SFT 启动脚本
  - GRPO / DiDPO 启动脚本
  - 评测脚本

---

## 当前主实验路径

```text
多轮轨迹采集
  -> 构建 SFT parquet
  -> 注入 <think>
  -> SFT
  -> DiDPO / GRPO RL
```

当前最常用配置：

- benchmark: `apps_train_coderl`
- max turns: `8`
- group size: `32`
- RL algorithm: `didpo`
- 常见模型：
  - Qwen2.5-Coder-7B
  - Qwen3.5-4B

---

## 数据信息

下面这些链接请你后续手动补：

- **SFT 数据 HF 链接：** [xuc865/DiDPO-SFT-Data](https://huggingface.co/datasets/xuc865/DiDPO-SFT-Data)
- **RL 数据 / benchmark HF 链接：** [PRIME-RL/Eurus-2-SFT-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-SFT-Data)

也可以补内部路径：

- **SFT 内部路径：** `TODO_FILL_ME_INTERNAL_SFT_DATA_PATH`
- **RL 内部路径：** `TODO_FILL_ME_INTERNAL_RL_DATA_PATH`

---

## 关键入口

### 数据准备

```bash
bash scripts/build_apps_mt8_sft_dataset.sh
```

### Qwen2.5-Coder-7B SFT

```bash
bash examples/sft/apps_mt8/run_apps_mt8_sft.sh
```

### Qwen3.5-4B SFT

```bash
bash examples/sft/apps_mt8/run_apps_mt8_sft_qwen35_4b.sh
```

### DiDPO RL

```bash
bash scripts/launch_didpo_coderl_sft_mt8.sh
```

### Qwen3.5-4B DiDPO RL

```bash
MODEL_PATH=/mnt/z4/solariewang/verl-swe/checkpoints/apps_mt8_sft_qwen35_4b_think/global_step_162 \
EXP_NAME=didpo_coderl_qwen35_4b_sft_mt8 \
PROJECT_NAME=didpo_coderl \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
bash /mnt/z4/solariewang/verl-swe/scripts/launch_didpo_coderl_sft_mt8.sh
```

---

## 产物位置

常见输出目录：

- 日志：`logs/`
- SFT ckpt：`checkpoints/apps_mt8_sft_*`
- RL ckpt：`checkpoints/<project>/<exp>/global_step_*`
- DiDPO group dump：`logs/didpo_groups/`
- SwanLab 本地日志：`swanlog/`

---

## 路径映射

训练机和开发机共享 Ceph，但路径前缀不同：

| 机器 | 路径前缀 |
|---|---|
| 训练机 | `/mnt/z4/solariewang` |
| 开发机 | `/apdcephfs/z4/solariewang` |

例如：

```text
/mnt/z4/solariewang/verl-swe
/apdcephfs/z4/solariewang/verl-swe
```

是同一份仓库内容。

---

## 对外展示可直接使用的简介文案

> We build a practical SFT-to-RL training stack for multi-turn coding agents.
> On top of GRPO-style episode-level reinforcement learning, we introduce
> DiDPO, a diff-aware credit assignment method that aligns and groups edited code
> snippets across rollouts and assigns finer-grained learning signals within a
> single response.

---

## 相关文档

- 总 README：`README.md`
- 算法说明：`didpo/README.md`
- 结构化 metadata：`didpo/PROJECT_METADATA.yaml`
- 超参表：`didpo/HYPERPARAMETERS.md`

