# 远端 GPU 机器准备清单（verl-code / coding agents + DIDPO）

> 目的：在一台带 GPU（+ 可选 Docker）的机器上，把这个仓库从零跑通——先用零下载的
> `local` 自修复床冒烟，再切到 CodeRL / APPS 等 coding 训练集做完整 RL。
> 本清单中的每个事实都已对照仓库源码核实，括号里标了出处文件。

---

## 0. 一句话背景

仓库是 **verl-agent 的 fork**，面向 **multi-turn coding agents**，并加入 **DIDPO**
（snippet 级分组优势）等算法。环境为 gym 风格、Ray 向量化的 coding 交互。
训练入口是 `verl.trainer.main_ppo`，算法/数据/环境全部用 hydra 命令行覆盖。
已准备好多个算法入口（GRPO / GRPO++ / GSPO / DAPO / DIDPO / GiGPO），默认
`benchmark=local`。

---

## 1. 硬件要求

| 项 | 要求 | 依据 |
|---|---|---|
| GPU 数量 | **≥ 2 张**（脚本默认 `trainer.n_gpus_per_node=2` + `tensor_model_parallel_size=2`） | `examples/didpo_trainer/run_swebench.sh` L53/L87 |
| 单卡显存 | 跑 `Qwen2.5-Coder-7B-Instruct` + vLLM，建议 **≥ 40GB/卡**（A100 40G / A800 / H800 等）；`gpu_memory_utilization=0.6` 留了余量 | 同上 L41/L55 |
| 内存 | ≥ 64GB（Ray + datasets + 多 env worker） | — |
| 磁盘 | **≥ 200GB 空闲**（若走 docker/r2e 真集）：HF 数据集 + 模型权重(~15GB) + 每实例 GB 级镜像；纯 CodeRL/local 可少很多 | `_BENCHMARK_PRESETS`（envs.py） |
| Docker | **必须**有可用 Docker daemon（真集走 `r2e_gym` backend，容器是评测唯一真相源） | envs.py L304-353 |

> ⚠️ 本机（开发用 Mac）没有 Docker，所以真集只能在远端跑。`local` 路径不需要 Docker/GPU 也能跑逻辑，但完整训练 run 必须 GPU。

---

## 2. 拉取代码

```bash
# 把整个 verl-code 目录同步到远端（git 或 rsync 均可）
# 关键子目录：verl/ didpo/ gigpo/ agent_system/ examples/ requirements.txt setup.py pyproject.toml
cd <远端路径>/verl-code
```

需要确保以下文件存在（清单核心交付物，已写好）：
- `examples/grpo_trainer/run_swebench.sh`      （GRPO）
- `examples/grpo_trainer/run_swebench_pp.sh`   （GRPO++：Dr.GRPO 无偏 + clip-higher + token-mean）
- `examples/gspo_trainer/run_swebench.sh`       （GSPO：`policy_loss.loss_mode=gspo` + 序列级聚合）
- `examples/dapo_trainer/run_swebench.sh`       （DAPO：clip-higher + token-mean）
- `examples/didpo_trainer/run_swebench.sh`      （DIDPO，原有）

---

## 3. Python 环境

建议 **Python 3.10**，新建独立 conda/venv。

### 3.1 核心依赖（`requirements.txt`）
```bash
pip install -r requirements.txt
```
内含：`datasets` `transformers==4.51.1` `tensordict<=0.6.2` `ray[default]` `flash-attn`
`accelerate` `peft` `liger-kernel` `hydra-core` `wandb` `pyarrow>=19.0.0` 等。

### 3.2 必须单独装的两个关键依赖（requirements.txt 里被注释/未列）

1. **vLLM**（推理引擎，脚本默认 `rollout.name=vllm`）——requirements 里是注释行 `# vllm==0.8.4`，
   需手动装，**版本要与 transformers==4.51.1 兼容**：
   ```bash
   pip install vllm==0.8.4
   ```
2. **r2egym**（真集 executor，仅当跑真 benchmark 时需要）——代码里 `from r2egym.agenthub.environment.env import EnvArgs, RepoEnv`：
   ```bash
   pip install r2egym        # 若 PyPI 无对应版本，按 R2E-Gym 官方仓库源码安装
   ```
   > `local` 路径**不需要** r2egym（envs.py 里只有切到 `r2e_gym` backend 才 import）。

### 3.3 flash-attn 注意
`flash-attn` 经常需要匹配 CUDA/torch 版本，装失败时用 `pip install flash-attn --no-build-isolation`，
或临时在脚本里关闭依赖它的特性。

### 3.4 环境变量
```bash
export VLLM_ATTENTION_BACKEND=XFORMERS   # 脚本已 export，确保 xformers 已装
export WANDB_API_KEY=<你的key>           # 脚本 logger=['console','wandb']；不想用 wandb 见 §6 改 console
export HF_TOKEN=<huggingface token>      # 拉模型/数据集用
```

---

## 4. 模型权重

脚本默认：`actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-7B-Instruct`
```bash
# 联网可直接由 transformers/vLLM 自动拉；或预先下载到本地再把 model.path 指向本地目录
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct
```

---

## 5. 数据准备（分两层，都要做）

### 5.1 占位 parquet（所有 benchmark 都要先跑）
训练器读 `data.train_files/val_files` 指向的 parquet，由预处理脚本生成（只编码 modality + 数据量，真实样本由 env 在线加载）：
```bash
python3 -m examples.data_preprocess.prepare --mode 'text' --train_data_size 8 --val_data_size 8
# 产物：$HOME/data/verl-agent/text/train.parquet 和 test.parquet
```

### 5.2 真实 benchmark 数据集（仅切真集时）
`benchmark` 预设 → HF 数据集映射（`envs.py` `_BENCHMARK_PRESETS` L415-424）：

| `env.swebench.benchmark` | HF 数据集 | split | backend | 需要 Docker |
|---|---|---|---|---|
| `local` | （内置合成集，**零下载**） | test | `local_stub` | 否 |
| `swe_bench_lite` | `R2E-Gym/SWE-Bench-Lite` | test | `r2e_gym` | **是** |
| `swe_bench_verified` | `R2E-Gym/SWE-Bench-Verified` | test | `r2e_gym` | **是** |
| `r2e_gym_subset` | `R2E-Gym/R2E-Gym-Subset` | train | `r2e_gym` | **是** |
| `r2e_gym_lite` | `R2E-Gym/R2E-Gym-Lite` | train | `r2e_gym` | **是** |

> ⚠️ **必须用 R2E-Gym 镜像版数据集**（`R2E-Gym/SWE-Bench-*`），其行里带 `docker_image` + 测试元数据，
> r2egym executor 才能评测。官方 `princeton-nlp/SWE-bench_Verified` 的行**不带**这些，跑不了（envs.py L409-414 明确说明）。

```bash
# 预热数据集缓存（以 Lite 为例，最小的真集）
python3 -c "from datasets import load_dataset; load_dataset('R2E-Gym/SWE-Bench-Lite', split='test')"
```

真集首次 step 时，r2egym 会按实例拉对应 Docker 镜像（GB 级，第一次很慢）。可提前预拉镜像以加速。

---

## 6. 关键陷阱（务必读，避免踩坑）

1. **DockerBackend 是 stub，不能直接用**：`benchmark` 预设里**没有任何一个走 DockerBackend**，真集统一走 `r2e_gym`。
   `DockerBackend.setup/run_bash/evaluate` 全是 `NotImplementedError`（envs.py L283-301）。**不要**手动把 backend 设成 `docker`。
2. **DAPO 的动态采样 `filter_groups` 在本 fork 未接线**：`clip-higher` + `token-mean` 是真生效的；
   但 `algorithm.filter_groups.*` 全库只在 yaml 声明、训练主循环（`ray_trainer.py`）**没有消费**，
   即 `enable=True` 当前是**空操作**。DAPO 脚本头已标注。若需要真动态采样，要在 `ray_trainer.py`
   主循环加 group 重采样逻辑（这必须在 GPU 机上单独验证）——**先别动，跑通其它算法为先**。
3. **GRPO++ 非外部标准定义**：本库按现有旋钮组合实现（Dr.GRPO 无偏 `norm_adv_by_std_in_grpo=False`
   + clip-higher + token-mean），脚本头逐条写明。若你们心里的 GRPO++ 是某篇具体论文定义，请反馈。
4. **wandb**：脚本默认 `trainer.logger=['console','wandb']`。无 wandb 账号就改成 `trainer.logger=['console']`
   或设 `WANDB_MODE=offline`，否则会卡在登录。
5. **GPU 数 / TP 一致性**：`n_gpus_per_node` 要 ≥ `tensor_model_parallel_size`。若只有 1 张卡，
   需把脚本里 `trainer.n_gpus_per_node=1` 且 `tensor_model_parallel_size=1`（7B 单卡 40G 勉强，注意 OOM）。

---

## 7. 验证步骤（按顺序，逐级放大）

### 阶段 A：离线单元测试（无需 GPU / Docker，先确认代码完整）
```bash
cd verl-code
for t in test_phase3_benchmark test_r2egym_contract test_selfrepair_env test_didpo_algorithm; do
  echo "=== $t ==="; python3 didpo/tests/$t.py && echo PASS || echo FAIL
done
```
预期：4 套全 PASS（本机已验证）。

### 阶段 B：`local` 冒烟（用 GPU，零下载，验证训练闭环）
```bash
bash examples/didpo_trainer/run_swebench.sh        # DIDPO
# 或任一算法：
bash examples/grpo_trainer/run_swebench.sh
```
预期：能进入 verl 训练循环、产生 rollout、算出 advantage、走完几个 step（不依赖 Docker）。
这一步过了说明 GPU/vLLM/FSDP/Ray 全链路 OK。

### 阶段 C：最小真集（Lite + Docker + r2egym）
```bash
# 确认 docker 可用 + r2egym 已装 + 数据集已缓存后：
bash examples/didpo_trainer/run_swebench.sh -- env.swebench.benchmark=swe_bench_lite
```
预期：首次会拉镜像（慢），之后每实例在容器里真跑测试、产出稀疏 ORM reward。

### 阶段 D：完整训练
切 `r2e_gym_subset`（train split）做正式 RL，按显存调 `train_data_size/group_size/ppo_*_batch_size`。

---

## 8. 五个算法启动命令（默认 local，加 `env.swebench.benchmark=...` 切真集）

```bash
# GRPO
bash examples/grpo_trainer/run_swebench.sh
# GRPO++（Dr.GRPO 无偏 + clip-higher + token-mean）
bash examples/grpo_trainer/run_swebench_pp.sh
# GSPO（序列级 importance ratio）
bash examples/gspo_trainer/run_swebench.sh
# DAPO（clip-higher + token-mean；filter_groups 未接线，见 §6.2）
bash examples/dapo_trainer/run_swebench.sh
# DIDPO（snippet 级分组优势）
bash examples/didpo_trainer/run_swebench.sh

# 切真集示例（任意脚本通用）：
bash examples/grpo_trainer/run_swebench.sh -- env.swebench.benchmark=swe_bench_lite
```

> 所有脚本结尾带 `$@`，可直接在命令行追加任意 hydra 覆盖（如 GPU 数、batch、benchmark）。

---

## 9. 需要远端反馈的信息（跑前/跑后回传）

1. `nvidia-smi`：GPU 型号、数量、单卡显存。
2. `docker --version` + `docker ps`：Docker 是否可用。
3. `pip show vllm transformers r2egym datasets`：实际装到的版本。
4. 阶段 A 测试结果（4 套是否全 PASS）。
5. 阶段 B 是否成功进入训练循环（贴前 ~20 行日志 + 是否出现 advantage 计算）。
6. 切真集时：第一个实例镜像拉取是否成功、`evaluate` 是否返回了 reward。
7. 是否需要：① 把 DAPO `filter_groups` 真正接线；② 调整 GRPO++ 定义。

---

## 10. TL;DR 给远端 agent 的最短路径

```bash
# 1) 环境
conda create -n codeswe python=3.10 -y && conda activate codeswe
pip install -r requirements.txt && pip install vllm==0.8.4
export VLLM_ATTENTION_BACKEND=XFORMERS WANDB_MODE=offline
# 2) 离线自测
for t in test_phase3_benchmark test_r2egym_contract test_selfrepair_env test_didpo_algorithm; do python3 didpo/tests/$t.py; done
# 3) local 冒烟（GPU）
python3 -m examples.data_preprocess.prepare --mode text --train_data_size 8 --val_data_size 8
bash examples/didpo_trainer/run_swebench.sh
# 4) 真集（需 docker + r2egym）
pip install r2egym
python3 -c "from datasets import load_dataset; load_dataset('R2E-Gym/SWE-Bench-Lite', split='test')"
bash examples/didpo_trainer/run_swebench.sh -- env.swebench.benchmark=swe_bench_lite
```
