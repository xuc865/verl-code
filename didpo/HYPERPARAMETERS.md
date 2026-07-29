# DiDPO Hyperparameters (CodeRL 主实验)

整理自：

- `scripts/launch_didpo_coderl_sft_mt8.sh` / `launch_didpo_coderl_sft_mt8_resume20.sh`
- `examples/didpo_trainer/run_swebench.sh`
- `verl/trainer/config/ppo_trainer.yaml` → `algorithm.didpo`
- `didpo/core_didpo.py`（含 `overhead_mode` 生效覆盖）

**本实验**指 Qwen2.5-Coder-7B SFT→DiDPO、`apps_train_coderl`、mt8、`GROUP_SIZE=32`、6×H20 那条跑法。

列为：**超参 | 配置键 | 本实验值 | 代码/YAML 默认 | 说明**。

---

## 1. DiDPO 核心（`algorithm.didpo.*`）

| 超参 | 配置键 | 本实验 | 默认 | 说明 |
|------|--------|--------|------|------|
| Diff 优势权重 \(\lambda\) | `snippet_advantage_w` | **1.0** | 1.0 | \(\hat A = A^E + \lambda A^D\) |
| 组内 baseline 模式 | `mode` | **mean_norm** | mean_norm | `mean_norm` / `mean_std_norm` |
| 子 diff 相似度阈 \(\eta\) | `sim_thresh` | **0.8** | 0.8 | exact / LCS 合并门槛 |
| GS 长度尺度 \(s_0\) | `phi_s0` | **8.0** | 8.0 | \(\phi = 1 - e^{-\bar L / s_0}\) |
| GS 人数尺度 \(g_0\) | `psi_count_ref` | **8.0** | 8.0 | \(\psi = 1 - e^{-(n-1)/g_0}\)，\(\psi(1)=0\) |
| \(\phi\) 形态 | `phi_form` | **saturating** | saturating | saturating / log / linear |
| \(\psi\) 形态 | `psi_form` | **saturating** | saturating | 同上 |
| Token 覆盖选择 | `level_select` | **argmax_gs** | argmax_gs | argmax_gs / bpe_greedy |
| GS 下限 | `gs_min` | **0.0** | 0.0 | 低于则该覆盖不参与 \(A^D\) |
| 每 uid 最大 anchor 数 \(K\) | `max_anchors_per_uid` | **64** | 64 | greedy facility-location |
| 最短 opcode block（行） | `min_block_lines` | **1** | 1 | 短于则丢弃 |
| LCS pair 预算 | `max_pairs_per_uid` | **8（被 overhead 覆盖）** | 64 | lightweight→8；exact_only→0 |
| 最大 span（兼容） | `max_span_lines` | **32** | 32 | LCS 路径基本不用滑窗 |
| 开对齐 | `use_alignment` | **True** | True | cross-rollout hash / LCS |
| AST fallback 开关 | `use_structural_fallback` | yaml True；**lightweight 运行时 False**† | True | †见 overhead |
| 开销模式 | `overhead_mode` | **lightweight** | lightweight | exact_only / lightweight / full |
| 自动追踪题数 | `track_prompt_n` | **3** | 3 | 固定 instance 组演化 |
| 显式追踪 id 列表 | `track_instance_ids` | 经 `tracked_instance_ids.json` 锁定 | null | 覆盖 auto |
| 旧别名 | `track_prompt_uids` | — | null | legacy = `track_instance_ids` |
| dump 每题 top groups | `track_preview_groups` | **8** | 8 | JSONL / latest dump |
| 组 dump 目录 | `group_dump_dir` | `logs/didpo_groups` | null→ckpt 下 | |

† `overhead_mode=lightweight` 生效后：`max_pairs_per_uid=8`，平时关 AST；**仅当对齐产出 0 组时**才 `fallback_if_no_align=True`。

### `overhead_mode` 展开

| `overhead_mode` | LCS pairs | AST fallback | collector |
|-----------------|-----------|--------------|-----------|
| `exact_only` | 0 | 关 | roots / hash only |
| **`lightweight`（本实验）** | **≤8** | **仅对齐为空时** | roots |
| `full` | 用配置默认 (64) | 始终可开 | roots + AST |

---

## 2. Episode / 共享 RL（\(A^E\) 侧）

| 超参 | 配置键 | 本实验 | 默认 | 说明 |
|------|--------|--------|------|------|
| Advantage 估计器 | `algorithm.adv_estimator` | **didpo** | — | 启用 DiDPO |
| 折扣 \(\gamma\) | `algorithm.gamma` | **0.95** | （见 yaml） | step RTG（GiGPO 同款） |
| Reward 内 KL | `algorithm.use_kl_in_reward` | **False** | False | |
| DAPO filter_groups | `filter_groups.enable` | **True** | False | 过滤同质组 |
| filter 最大补采样批 | `filter_groups.max_num_gen_batches` | **20** | 10 | Keep generating |
| 每题 rollout 数 \(G\) | `env.rollout.n` ← `GROUP_SIZE` | **32** | 8（run 脚本） | uid 组大小 |
| train batch（题数） | `data.train_batch_size` | **6**（= n_gpus） | — | 每步约 \(6\times 32 = 192\) traj |

---

## 3. 环境 / CodeRL 任务

| 超参 | 配置键 | 本实验 | 说明 |
|------|--------|--------|------|
| Benchmark | `env.swebench.benchmark` | **apps_train_coderl** | |
| 反馈模式 | `env.swebench.test_feedback_mode` | **interactive** | |
| 编辑后自动测 | `env.swebench.auto_test_after_edit` | **True** | |
| 步奖励系数 | `env.swebench.step_reward_coef` | **0.15** | |
| IO case 上限 | `env.swebench.io_max_cases` | **64** | |
| IO 内存 | `env.swebench.io_memory_limit_mb` | **2048** | |
| 最大步 | `env.max_steps` | **8** | mt8 |
| 最大轮 | `env.swebench.max_turns` | **8** | mt8 |
| finish 前最少轮 | `env.swebench.min_turns_before_finish` | **3** | |
| 历史长度 | `env.history_length` | **5** | |
| env CPU / worker | `env.resources_per_worker.num_cpus` | **0.5** | |
| 数据根 | `env.swebench.data_root` | `$ROOT/datasets` | |
| 固定追踪题文件 | `env.swebench.tracked_instance_ids_file` | `logs/didpo_groups/tracked_instance_ids.json` | `apps__idx3836` / `417` / `46` |

---

## 4. 模型 / Actor / PPO / Rollout

| 超参 | 配置键 | 本实验 | 默认 / 备注 |
|------|--------|--------|-------------|
| 初始化 | `actor_rollout_ref.model.path` | **SFT `global_step_242`** | Qwen2.5-Coder-7B-Instruct 系 |
| trust_remote_code | `model` / `data` | **True** | |
| 学习率 | `actor_rollout_ref.actor.optim.lr` | **1e-6** | |
| PPO clip | `clip_ratio` / `clip_ratio_low` / `high` | **0.2** | yaml |
| Dual-clip \(c\) | `clip_ratio_c` | **3.0** | |
| Entropy | `entropy_coeff` | **0.001** | |
| Grad clip | `grad_clip` | **1.0** | |
| PPO epochs | `ppo_epochs` | **1** | |
| Loss 聚合 | `loss_agg_mode` | **token-mean** | 对齐 GRPO |
| KL loss | `use_kl_loss` | **True** | |
| KL 系数 | `kl_loss_coef` | **0.01** | |
| KL 类型 | `kl_loss_type` | **low_var_kl** | |
| 非法 action 惩罚 | `use_invalid_action_penalty` | **True** | |
| 非法 action 系数 | `invalid_action_penalty_coef` | **0.01** | run 默认 0.1，launch 覆盖 |
| Mini-batch | `ppo_mini_batch_size` | **48**（\(8 \times 6\) GPU） | |
| Micro / GPU | `ppo_micro_batch_size_per_gpu` | **4** | |
| log_prob micro（rollout/ref） | `*_log_prob_micro_batch_size_per_gpu` | **4** | |
| 采样温度（train） | `rollout.temperature` | **1.0** | yaml |
| top_p | `rollout.top_p` | **1** | |
| Val 温度 | `rollout.val_kwargs.temperature` | **0.6** | run_swebench |
| Val do_sample | `rollout.val_kwargs.do_sample` | **True** | |
| Engine | `rollout.name` | **vllm** | |
| TP | `tensor_model_parallel_size` | **2** | |
| GPU util | `gpu_memory_utilization` | **0.70** | |
| max batched tokens | `max_num_batched_tokens` | **12288** | |
| chunked prefill | `enable_chunked_prefill` | **True** | |
| enforce_eager | `enforce_eager` | **False** | CUDA graph |
| free_cache_engine | `free_cache_engine` | **False** | |
| Ref param offload | `ref.fsdp_config.param_offload` | **True** | |
| Actor param offload | `actor.fsdp_config.param_offload` | **False** | |
| Actor optim offload | `actor.fsdp_config.optimizer_offload` | **False** | |
| remove_padding | `model.use_remove_padding` | **True** | |
| gradient checkpointing | `model.enable_gradient_checkpointing` | **True** | |

---

## 5. 数据 / 序列长度

| 超参 | 配置键 | 本实验 | 说明 |
|------|--------|--------|------|
| max prompt | `data.max_prompt_length` | **8192** | launch 覆盖（run 默认 4096） |
| max response | `data.max_response_length` | **4096** | launch 覆盖（run 默认 1024） |
| truncation | `data.truncation` | **left** | |
| filter overlong | `data.filter_overlong_prompts` | True（run） | |
| return_raw_chat | `data.return_raw_chat` | True | agent 多轮 |

---

## 6. Trainer / 资源 / 日志

| 超参 | 配置键 | 本实验 | 说明 |
|------|--------|--------|------|
| GPU 列表 | `CUDA_VISIBLE_DEVICES` | **2,3,4,5,6,7** | |
| n_gpus | `trainer.n_gpus_per_node` | **6** | |
| nnodes | `trainer.nnodes` | **1** | |
| 总步数 | `trainer.total_training_steps` | **100** | 建议早停 ~60 |
| total_epochs | `trainer.total_epochs` | **9999** | 以 step 为准 |
| save_freq | `trainer.save_freq` | **20** | ckpt 20/40/60/80 |
| max_actor_ckpt_to_keep | `trainer.max_actor_ckpt_to_keep` | **3** | |
| test_freq | `trainer.test_freq` | **-1** | 训中不周期性 test |
| val_before_train | `trainer.val_before_train` | **False** | |
| skip_val_envs | `trainer.skip_val_envs` | **True** | |
| critic_warmup | `trainer.critic_warmup` | **0** | critic-free |
| logger | `trainer.logger` | `['console','swanlab']` | |
| SwanLab mode | `SWANLAB_MODE` | **local** | 离线写 `swanlog/` |
| project / exp | `trainer.project_name` / `experiment_name` | `didpo_coderl` / `didpo_coderl_qwen25_7b_sft_mt8` | |
| resume（续跑） | ckpt + `SWANLAB_RUN_ID` | `global_step_20`；run `c08qn4a8` | resume20 脚本 |

---

## 7. 分析用（非训练，分层图）

| 名称 | 值 | 说明 |
|------|-----|------|
| block vs fragment | \(\bar L \ge 8\) → block，否则 fragment | 仅可视化宏类，不进 loss；见 `didpo/scripts/plot_group_composition_strata.py` |
| stub | preview 含 `return None` 等 | |
| harness | `__main__` / stdin / typing 等脚手架 | |

---

## 8. 公式速查（本实验默认）

\[
\mathrm{GS} = \bigl(1 - e^{-\bar L / 8}\bigr)\bigl(1 - e^{-(n-1)/8}\bigr),
\qquad
\hat A_{i,t} = A^E_i + 1.0 \cdot A^D(s_{i,t})
\]

- 组内 \(A^D\)：`mean_norm`（减组均值，**不**除 std）
- 对齐：`sim ≥ 0.8` + `overhead_mode=lightweight`（hash + ≤8 LCS pairs）
- singleton：\(\psi(1)=0\)，不进 \(C^\star\) / \(A^D\)

---

## 9. 关键路径速查

| 用途 | 路径（IDE） | 路径（训练机） |
|------|-------------|----------------|
| 仓库 | `/apdcephfs/z4/solariewang/verl-swe` | `/mnt/z4/solariewang/verl-swe` |
| 启动 | `scripts/launch_didpo_coderl_sft_mt8.sh` | 同左 |
| 续跑 | `scripts/launch_didpo_coderl_sft_mt8_resume20.sh` | 同左 |
| 组 dump | `logs/didpo_groups/` | 同左 |
| ckpt | `checkpoints/didpo_coderl/didpo_coderl_qwen25_7b_sft_mt8/` | 同左 |

---

*Generated for the DiDPO CodeRL writeup / ablation checklist.*
