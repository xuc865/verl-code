# Benchmark 接入说明（交接给执行 agent）

> 目标读者：负责实现的另一个 agent。本文件把「现在支持什么 / 还要接什么 / 每个 benchmark 该怎么接 / 接在哪几个文件 / 怎么算验收」一次说清。
> 仓库：`code-swe`（SWE-bench RL 训练栈，核心算法 DIDPO，底座 vendored veRL / verl-agent）。
> 所有路径相对仓库根 `code-swe/`。

## ⚠️ 硬性约束（需求方拍板）

**只接 multi-turn / self-repair 范式的任务。绝不引入 single-turn（一次生成、一次判题）的环境或数据形态。**
任何 benchmark 接进来，都必须表现为：agent 多轮 `edit` → `execute_bash`（跑测试）→ 看失败 → 再 edit → … → `finish`，由 resolved 测试集给稀疏 outcome reward。
→ 因此**不新增 single-turn 环境**。LiveCodeBench / HumanEval / MBPP / APPS 等竞赛/函数级题目，**必须先被改写成 self-repair 实例**，再骑现有的多轮环境。

---

## 0. 现状（已支持）

环境只有一类：**SWE-bench 式 agentic 编码环境**（多轮 `edit` / `execute_bash` / `finish`，稀疏 outcome reward，1.0 当且仅当 resolved 测试集通过）。这本身就是 multi-turn / self-repair 范式。

`env.swebench.benchmark` 预设（见 `resolve_benchmark()`）：

| 预设 | dataset | split | backend | Docker |
|---|---|---|---|---|
| `local`（默认） | 合成自修复集 | test | `local_stub` | 否 |
| `swe_bench_verified` | `R2E-Gym/SWE-Bench-Verified` | test | `r2e_gym` | 是 |
| `swe_bench_lite` | `R2E-Gym/SWE-Bench-Lite` | test | `r2e_gym` | 是 |
| `r2e_gym_subset` | `R2E-Gym/R2E-Gym-Subset` | train | `r2e_gym` | 是 |
| `r2e_gym_lite` | `R2E-Gym/R2E-Gym-Lite` | train | `r2e_gym` | 是 |
| `custom` / 空 | 透传 `dataset_name`/`split`/`backend` | — | — | 取决于 backend |

**关键既有能力 —— `LocalStubBackend` 是一个真正会跑 pytest 的 self-repair 床**（`envs.py` 的 `_run_pytest_like`）：它把实例的 `initial_files`（含 `test_*.py`）落地到临时目录、import 并逐个跑 `test_*` 函数、按 `FAIL_TO_PASS` 过滤、返回真实 traceback。agent 可以 `pytest` 看失败 → 改源码 → 再跑，全程 CPU、无 Docker。

> 这正是把外部 coding benchmark「self-repair 化」后**直接复用**的落点：只要把一道题转成 `{problem_statement, initial_files(含 stub + 生成的 test_*.py), FAIL_TO_PASS}`，现有环境/projection/manager/prompt **一行不用改**就能多轮自修复。

---

## 1. 架构接缝（实现前必读）

| 接缝 | 文件 | 作用 |
|---|---|---|
| 预设表 | `agent_system/environments/env_package/swebench/envs.py` → `_BENCHMARK_PRESETS` | benchmark key → dataset/split/backend |
| benchmark 解析 | 同上 → `resolve_benchmark()` | 预设展开（对 dataset/split/backend 最高优先级） |
| backend 工厂 | 同上 → `make_backend()` | backend 名 → 实例 |
| backend 接口 | 同上 → `SWEBenchBackend` | `setup/run_bash/evaluate/is_test_command/run_tests/apply_edit/...` |
| **self-repair 执行床** | 同上 → `LocalStubBackend` + `_run_pytest_like` | 真跑 `test_*` 函数，按 `FAIL_TO_PASS` 过滤 |
| 数据加载 | 同上 → `load_instances()` + `_normalize_instance()` + `_synthetic_instances()` | HF 加载 / 合成集；实例 schema 的来源 |
| 实例 schema | `_normalize_instance` 返回字段 | `instance_id/problem_statement/FAIL_TO_PASS/initial_files/...`，原始行存 `_raw` |
| 单实例环境 | 同上 → `SWEBenchSingleEnv` | 多轮 edit/bash/finish 循环 + reward（**复用，不改**） |
| 动作解析 | `.../swebench/projection.py` | `<think>`/`<edit>`/`<execute_bash>`/`<finish>`（**复用，不改**） |
| 管理器 + 分发 | `agent_system/environments/env_manager.py` | `env_name` 含 `"swebench"` 才走本栈（**复用**） |
| prompt 模板 | `agent_system/environments/prompts.py` | 观测渲染（**复用**） |
| 预设契约测试 | `didpo/tests/test_phase3_benchmark.py` → `_EXPECTED` | 新增预设必须同步 |
| 可复用判题器 | `verl/utils/reward_score/prime_code/__init__.py` → `compute_score(completion, test_cases)` | stdin/stdout（APPS 式 in/outs）判题，stdin/stdout 类 benchmark 复用 |

---

## 2. 两条接入路线（都满足 multi-turn 硬约束）

### Track A —— agentic / 真实仓库（天然多轮，最优先）

和 SWE-bench 完全同范式。多数情况**只加一个预设**，零新代码。

前提：`R2EGymBackend` 只能判**带 R2E-Gym schema（行里有 `docker_image` + 可 `compute_reward`）的镜像版数据集**。没镜像的要么找镜像版，要么实现 `DockerBackend`（当前是 stub）。

| 优先级 | benchmark | 建议 dataset（待核对） | split | 接法 |
|---|---|---|---|---|
| P0 | SWE-bench Full | `R2E-Gym/SWE-Bench-Full`（确认存在性） | test | 加预设；无镜像则需 DockerBackend |
| P1 | R2E-Gym 全量/V1 | `R2E-Gym/R2E-Gym-V1`（核对仓库名） | train | 加预设 |
| P2 | SWE-Gym | `SWE-Gym/SWE-Gym`（无 R2E 镜像） | train | 预设 + DockerBackend |
| P3 | Multi-SWE-bench（多语言） | `Multi-SWE-bench/...`（核对） | test | 预设 + 多语言执行 backend（最重） |

### Track B —— coding benchmark 的「self-repair 化」（重点，含 LiveCodeBench）

不新增环境。**写一个 dataset adapter，把每道题转成 self-repair 实例**，骑现有多轮环境。一道题的自修复闭环 = 「给题面 + 失败/缺失实现的 stub + 一组测试 → agent 反复 edit & 跑测试直到全绿」。

判题分两种，决定 adapter 怎么生成测试与用哪个 backend：

| benchmark | 建议 dataset（待核对） | 题目判题形态 | self-repair 化方式 | backend |
|---|---|---|---|---|
| **HumanEval(+)** | `openai_humaneval` / `evalplus/humanevalplus` | assert 式单测 | stub=函数签名+docstring（body 留 `raise`/错误实现）；把 `check`/`test_list` 包成 `def test_*()` | **复用 `local_stub`**（`_run_pytest_like` 直接跑） |
| **MBPP(+)** | `mbpp`(`sanitized`) / `evalplus/mbppplus` | assert 式单测 | 同上，`test_list` 的 assert 直接拼进 `test_*` 函数 | **复用 `local_stub`** |
| **BigCodeBench** | `bigcode/bigcodebench`(`complete`) | 库调用 + 单测 | 同上（依赖较重，注意第三方库可用性） | `local_stub` 或 sandbox |
| **LiveCodeBench** | `livecodebench/code_generation_lite`（按 `version_tag` 选时间窗防污染） | stdin/stdout + functional | stub=空 `solution`/`Solution`；生成 stdin/stdout **判题 harness 测试** | **新增 `SelfRepairIOBackend`**（见 3.2），内部复用 `prime_code.compute_score` |
| **APPS** | `codeparrot/apps`(`introductory/interview/competition`) | stdin/stdout in/outs | 同 LiveCodeBench | `SelfRepairIOBackend` |
| **CodeContests** | `deepmind/code_contests` | stdin/stdout（多语言） | 同上（先只做 Python） | `SelfRepairIOBackend` |

> **建议路径**：先做 **HumanEval**——它是 assert 式，能**直接复用 `local_stub`、零新 backend**，是把「benchmark→self-repair」管线打通的最小验证。打通后再做 **LiveCodeBench**（需要 stdin/stdout backend）。

---

## 3. 实现指引

### 3.1 Track A：加一个 agentic 预设

1. `envs.py` → `_BENCHMARK_PRESETS` 加一行，如
   `"swe_bench_full": {"dataset_name": "R2E-Gym/SWE-Bench-Full", "split": "test", "backend": "r2e_gym"}`。
2. `didpo/tests/test_phase3_benchmark.py` → `_EXPECTED` 同步加期望。
3. 拉一行核对 `_normalize_instance` 取得到字段，且 `_raw` 有 `docker_image`。
4. 无 R2E 镜像 → 必须实现 `DockerBackend`（见 3.3）。

### 3.2 Track B：coding benchmark → self-repair 实例（核心工作）

**总思路：只加「数据 adapter + 必要时一个 IO 判题 backend」，环境/projection/manager/prompt 全复用。**

**(a) 新增 adapter 模块**：`agent_system/environments/env_package/swebench/adapters/`（或 `.../codegen_selfrepair.py`），每个 benchmark 一个 `to_selfrepair_instances(rows) -> List[instance]`，产出严格对齐 `_normalize_instance` 的 schema：
```python
{
  "instance_id":        f"humaneval__{task_id}",
  "problem_statement":  <题面/docstring，明确告知"跑 pytest 看失败再改">,
  "initial_files": {
      "solution.py":      <带签名/缺陷的 stub，保证初始测试一定失败>,
      "test_solution.py": <生成的 test_* 函数，import solution 并断言>,
  },
  "FAIL_TO_PASS":       json.dumps(["test_<...>", ...]),  # 必须与生成的函数名一致
  "PASS_TO_PASS":       "[]",
  "target_file":        "solution.py",
}
```
要点：
- **stub 必须让初始测试失败**（空实现 `raise NotImplementedError` 或错误实现），否则没有 self-repair 信号。
- 生成的测试函数名要与 `FAIL_TO_PASS` 一一对应（`local_stub.evaluate` 用它过滤）。
- assert 式（HumanEval/MBPP）：把原 `check(candidate)` / `test_list` 包进 `def test_xxx(): ...`，import 自 `solution`。`local_stub` 的 `_run_pytest_like` 能直接跑，**无需新 backend**。

**(b) 接入数据加载**：在 `load_instances()` 里按 benchmark 分流（或在 `_BENCHMARK_PRESETS` 增加形如
`"humaneval": {"dataset_name": "openai_humaneval", "split": "test", "backend": "local_stub", "adapter": "humaneval"}` 的预设，并在 `load_instances` 读到 `adapter` 时调用对应 adapter 而非 `_normalize_instance`）。同步 `test_phase3_benchmark.py` 的 `_EXPECTED`。

**(c) stdin/stdout 类（LiveCodeBench/APPS/CodeContests）需要一个新 backend**：`SelfRepairIOBackend(SWEBenchBackend)`，在 `make_backend` 注册（如 `backend="selfrepair_io"`）：
- `setup`：落地 `initial_files`（含 agent 要编辑的 `solution.py`）。
- `is_test_command` / `run_bash`：识别 `pytest`/`python solution.py` 等，跑 in/out 对照并返回**真实 diff 报告**（哪个用例 expected vs got），给 agent 自修复信号。
- `run_tests` / `evaluate`：复用 `from verl.utils.reward_score.prime_code import compute_score`（它吃 `{"inputs":[...],"outputs":[...]}`，从代码块抽解），全过 → reward 1.0。
- **安全**：stdin/stdout 程序要在子进程 + 超时 + 资源限制下跑，禁止裸 `exec` 不可信代码。可优先复用 `verl/utils/reward_score/sandbox_fusion`。

**(d) DIDPO 衔接（重要）**：DIDPO 的 snippet 提取依赖代码出现在可识别块里。self-repair 化后 agent 仍走现有 `<edit path="..."><code>...</code></edit>`，所以**天然兼容**——这正是「不另起 single-turn 环境」的额外好处。adapter 不要改动 projection 的标记约定。

### 3.3 可选：实现 `DockerBackend`

`envs.py` 的 `DockerBackend` 是 stub。要支持官方 `princeton-nlp/SWE-bench_*` 或 SWE-Gym：实现 `setup`（checkout base_commit + apply test_patch + 挂载工作目录）、`run_bash`（容器内 exec）、`evaluate`（跑 FAIL_TO_PASS / PASS_TO_PASS）。工作量大，仅在确需非 R2E 数据集时做。

---

## 4. 验收标准

- [ ] 所有新接任务都是 multi-turn / self-repair（无任何 single-turn 环境或 one-shot 判题路径）。
- [ ] Track B 的 self-repair 实例：**初始测试必失败、修复后必通过**（adapter 单测各验证一例）。
- [ ] `resolve_benchmark` / `make_backend` 对新 key 正确；未知 key 仍抛 `ValueError`。
- [ ] `didpo/tests/test_phase3_benchmark.py` 的 `_EXPECTED` 已同步，测试通过。
- [ ] 新增 adapter / IO backend 有不依赖网络/GPU 的 L1 单测。
- [ ] HumanEval self-repair 在本机 CPU（`local_stub`）端到端跑通（拉数据 → adapter → reset → edit → pytest → evaluate 拿 reward）。
- [ ] 真集（Track A / LiveCodeBench）在远端 Docker/GPU 端到端跑通 1 batch（L3/L4，见 `REMOTE_SETUP_CHECKLIST.md`）。
- [ ] 文档同步：根 `README.md` benchmark 表 + `didpo/README.md`。
- [ ] 不破坏 `local` 默认路径（CPU、无 Docker 仍一键跑通）。

---

## 5. 需确认的开放问题

1. **优先级**：建议 `HumanEval(self-repair，复用 local_stub 打通) → LiveCodeBench(self-repair + IO backend) → SWE-bench Full(agentic 预设)`。
2. **stub 初始态**：self-repair 化时 stub 用「空实现 raise」还是「注入一个典型 bug」？前者更简单稳定，后者更接近真实修复分布——选哪个？
3. **IO 判题沙箱**：`sandbox_fusion`（需服务）还是本地子进程受控执行？
4. **多语言**（CodeContests / Multi-SWE-bench）本轮是否纳入？建议延后（非 Python 执行栈最贵）。
5. **数据集 HF id / split / config** 逐个核对（见第 2 节「待核对」；LiveCodeBench 注意 `version_tag` 时间窗防污染）。

---

### 一句话总结给执行 agent

> 硬约束：**只做 multi-turn / self-repair，不做 single-turn**。
> 加 SWE-bench 同范式真集 = 在 `_BENCHMARK_PRESETS` 加一行 + 同步契约测试（无镜像则写 `DockerBackend`）。
> 加 LiveCodeBench/HumanEval 等 = **写 adapter 把每题转成 self-repair 实例**（stub + 生成 `test_*` + `FAIL_TO_PASS`），assert 式直接复用 `local_stub`，stdin/stdout 加一个 `SelfRepairIOBackend`（内部复用 `prime_code`）；**环境/projection/manager/prompt 全部复用，绝不新建 single-turn 环境**。
