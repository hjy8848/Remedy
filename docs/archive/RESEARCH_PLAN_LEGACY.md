# Remedy：自主实验驱动的 RSI 实施计划书

> 历史方案：本文保留早期 Designer/Editor、P/K 与 GRPO/OPSD 的设计记录。当前研究方向和运行方式以 [新版研究计划](../RESEARCH_PLAN.md) 为准。

**对应研究方向：Learning to Experiment for Recursive Agent Improvement**

版本：v1.0；日期：2026 年 10 月 25 日
文档性质：实施计划，供讨论和后续开发使用。文中的新增目录、接口、命令和阶段产物均是拟实现规格，不表示已完成开发。


## 训练实现更新（2026-10-26）

GRPO 与 OPSD 已统一到 VeRL V1；下文关于独立训练损失或 MPS 参数更新的旧规格由本段替代。Host 保留任务合成、环境执行、独立评价和轨迹记录职责；VeRL 处理优势、反向传播、优化器、调度器与 checkpoint。多次模型调用按同一研究 trajectory 回放，原生 GRPO 先在组内比较 trajectory 末尾奖励，再将优势广播到所有调用。Remedy 的 trainer 扩展保留每条 trajectory 内总和为 1 的调用权重。OPSD 只接收 edit 样本，并使每个候选的特权反馈独立。服务器环境为 `.venv-verl`，Python 3.11；旧 Python 3.10 环境不能运行该固定提交。

## 当前原型进度（2026-10-25）

本文件保留完整研究目标和 P0–P6 验收规格。当前实现是原型；代码路径接通不等于阶段验收通过，真实运行与训练证据单独记录。

已实现的代码路径包括 ALFWorld/TextWorld PDDL 状态读取、派生 PDDL 目标文件、目标见证重放、目标谓词评分、带证据/去证据修改分支、独立选择与奖励任务、实验角色 GRPO、修改角色 GRPO/OPSD、多周期编排，以及由 Docker worker 隔离执行的候选 P 源码。P 通过 Host 请求固定基础模型，K 作为 JSON 经验传入。

P2 的最小端到端验收已通过：从 4 个不同 ALFWorld 场景得到 4 个单物体放置子目标，见证在派生目标环境中重放 4/4 成功，3B Solver 在这些任务上成功 4/4。Qwen2.5-0.5B 也已在 FloorPlan304、FloorPlan313、FloorPlan316 的固定场景上生成可验证目标。这些结果只说明小规模链路可运行；尚未建立统计显著性或跨场景泛化结论。ALFWorld reset/step、模型生成及 MPS LoRA 更新已完成 smoke。

P3 的完整研究组和 φ 的真实 GRPO 更新尚未完成：3B 采样在 M4 的 16 GB 统一内存上达到 18.7 GB 峰值，0.5B 随机采样组没有生成诊断任务。VeRL PR #6833 的固定提交及 OPSD 运行依赖已安装，Replay AgentLoop、特权上下文模块、Hydra 配置和依赖检查均通过；但本机没有 NVIDIA CUDA，原生 VeRL FSDP/vLLM OPSD 权重更新尚未运行。低频 M 自改、研究策略迁移命令和正式统计分析尚未实现。

合成任务从 Proposer 探索前的 ALFWorld 初始状态开始。Host 读取真实 TextWorld PDDL 状态中新出现的放置谓词，生成自定义目标公式，并把它写入源 `.tw-pddl` 的派生副本；探索轨迹必须在派生目标环境中重放成功后才会注册。Solver 从同一场景的派生任务初始状态作答，原任务的成功标志不替代新目标判定。目标完成进度根据真实目标谓词及持有状态计算，完整成功率单独用于候选接受和最终测试。反馈蒸馏由 VeRL PR #6833 原生 trainer 执行：Replay AgentLoop 回放 Host 采样的 response，冻结教师在特权反馈上下文中计算 top-k 分布，FSDP 更新学生 LoRA。无证据分支只用于估计实验价值，不进入修改策略训练。

当前机器已安装 Python 3.11.17、PyTorch 2.14.1、Transformers、PEFT、ALFWorld/TextWorld，以及 VeRL PR #6833 固定提交 `9268aad322df93cc61e9c168d66d395256dbe87a` 和其 native trainer 运行依赖；MPS 可用，`python:3.11-slim` 镜像已拉取。Replay AgentLoop 与 PR 的 privileged-context 模块可导入，VeRL Hydra 配置组合和 `pip check` 已通过。Qwen2.5-0.5B-Instruct 权重已按 revision 固定并缓存；ALFWorld 文本环境数据从 Hugging Face `zhangdw/alfworld` revision `8f37edb5e460ef985ec3416043a177e42be860fd` 下载，归档 SHA-256 为 `a15b6c4eef2f86ccc39d082ecd2d99a9e022f92289f71f63679069099a208c59`。本机没有 NVIDIA CUDA，因此原生 VeRL FSDP/vLLM 权重更新仍未验收；完整自主合成目标的见证重放也尚未验收。

## 1. 实施目标

从零建设一个独立研究仓库，实现以下完整循环：

**研究模型在 ALFWorld 中自主探索并生成可执行实验 → 当前 Agent 执行实验 → 研究模型利用轨迹修改 Agent 程序与经验 → Host 独立评价 → 使用 RL 或反馈自蒸馏更新研究模型 → 新版本继续研究。**

第一阶段的目标是用一条真实链路证明系统能够执行、评价、更新参数并重新运行。后续阶段逐步增加自主目标合成、实验价值归因、联合学习及研究程序自改。

本计划采用全新仓库设计，不依赖此前 RSI Core 的模块、训练导出器、协议或实现状态。保留的是研究思想和清楚的任务职责。

最终完整版本应当提供：

1. 真实 ALFWorld 文本环境执行与轨迹重放。
2. 基于探索见证的自主任务及实验合成。
3. 固定基础模型 B 上的 P/K 程序改进。
4. 独立选择、训练奖励和最终测试。
5. 实验价值奖励与出题策略 GRPO 训练。
6. 修改策略的 GRPO 和反馈自蒸馏两种训练方式。
7. 训练后 checkpoint 加载及跨周期继续研究。
8. 冻结研究模型后的研究策略迁移实验。
9. 低频研究程序 M 自改。

## 2. 实施原则

已按 stop-that-shit、avoid-overdefensive-programming 和 avoid-overkill 三个技能确定以下做法。

### 2.1 以真实运行推进

每个阶段结束时，都必须能执行一条实际命令，得到该阶段要求的真实产物。

完成数据格式、生成配置文件或通过模拟测试，不代替真实环境运行和参数训练。

早期就运行一次真实 LoRA 更新，避免直到系统末尾才发现模型接口、显存或梯度路径不成立。

### 2.2 每次增加一个有用途的能力

先完成单环境、单机、单卡、顺序执行。模块数量由当前调用关系决定。

首版采用普通 Python 函数、少量 dataclass、JSON/JSONL 和直接的训练循环，不建设插件系统、通用后端注册中心、分布式调度平台或多层审查流程。

### 2.3 错误应当在发生处可见

区分两种结果：

- **候选行为失败：**程序无法完成任务、生成了无效动作或无效修改。这是算法需要学习的结果。
- **基础设施失败：**环境启动失败、模型未加载、文件写入失败、GPU OOM 或训练异常。命令失败退出，保留错误，不能当作低分任务继续统计。

仅在明确负责执行候选的边界捕获候选错误。资源释放使用正常的上下文管理或 finally。不能用全局 catch 将所有错误变成零分。

### 2.4 保留直接必要的边界

输入模型输出时检查结构；执行候选程序时隔离运行；任务成功由 Host 判断；评价数据不暴露给候选。这些边界直接关系到研究结果是否成立，需要实现。

不增加与当前结果无关的扫描、重复认证、兼容格式、自动修复链或无限重试。

### 2.5 计划先于实现

本文只规定开发内容与验收方式。正式实现从阶段 P0 开始，完成一个阶段后再进入下一个阶段。

无需在开始阶段预先生成全部模块和大量空类。目录随可运行能力增加。

## 3. 方法与范围

### 3.1 改进对象

\[
S=(B,P,K),\qquad D=(M,\phi).
\]

| 对象 | 含义 | 更新方式 |
|---|---|---|
| B | Service 的基础语言模型 | 冻结 |
| P | 上下文组织、状态管理、计划与动作程序 | 生成候选，通过实际评价选择 |
| K | 经验和策略知识 | 随候选更新 |
| M | 安排实验、修改和研究预算的程序 | 后期低频程序自改 |
| φ_E | 研究模型的实验角色 | 按实验价值做 GRPO |
| φ_U | 研究模型的修改角色 | GRPO 或反馈自蒸馏 |
| Host | 环境调用、候选执行、任务评分与训练编排 | 作为研究基础设施固定 |

### 3.2 首版模型布置

使用一个可训练的开源小模型底座：

- 禁用研究 LoRA 时，作为冻结的 B。
- 加载实验 LoRA 时，作为 φ_E。
- 加载修改 LoRA 时，作为 φ_U。

三种角色可以顺序使用同一个模型底座，减少显存占用。切换角色不能修改基础权重。

每批采样使用固定的适配器快照，采样完成后才更新参数。教师适配器在蒸馏批次内冻结。

首个模型从 1.5B–3B 量级开始，正式训练机器使用 Linux 和 NVIDIA GPU。模型名称、revision、tokenizer 和实际依赖版本在 P0 固定，并写入运行配置。若指定模型不适合，不自动改用其他模型；修改配置后重新运行。

### 3.3 首版执行范围

首先只支持：

- ALFWorld 文本环境。
- 一个本地模型运行方式。
- 单 GPU 训练。
- 顺序候选执行。
- 完整替换的短 P 源码和一个 JSON 对象形式的 K。
- 固定长度的研究周期。

M 自改在基础闭环和研究模型训练成立之后加入。视觉环境、跨机器调度和多模型供应商不属于首版。

## 4. 整体流程

~~~mermaid
flowchart TD
    A[父版本 B/P/K 与研究历史] --> B[φ_E 自主探索并提出实验]
    B --> C[Host 复现见证并执行当前 Service]
    C --> D[诊断报告 F_diag]
    D --> E[φ_U 生成 P/K 修改]
    E --> F[Host 执行独立选择与奖励任务]
    F --> G[接受新 Service]
    F --> H[执行反馈 F_post]
    F --> I[标量研究收益]
    H --> J[修改角色自蒸馏]
    I --> K[实验角色 GRPO]
    I --> L[修改角色 GRPO 对照]
    J --> M[保存并重载适配器]
    K --> M
    L --> M
    M --> A
    G --> A
~~~

需要区分两种轨迹：

- F_diag：修改之前已经获得的诊断轨迹，正常研究者可以使用。
- F_post：某个修改实际执行后才得到的反馈，用于训练时的事后自教师。

“实验价值对照”和“自蒸馏”是两项不同操作。前者比较研究结果，后者计算模型训练损失。

## 5. 拟建仓库

仓库名：Remedy。计划中的路径均相对此仓库。

~~~text
Remedy/
  pyproject.toml
  README.md
  configs/
    smoke.yaml
    pilot.yaml
  src/experiment_rsi/
    cli.py
    config.py
    records.py
    model.py
    policy.py
    worker.py
    alfworld.py
    synthesis.py
    research.py
    evaluation.py
    verl_training.py
    verl_runtime.py
    verl_agent_loop.py
    train.py
  seeds/
    service.py
    designer.py
  tests/
    test_environment.py
    test_synthesis.py
    test_research.py
    test_training.py
  runs/                         # 运行产物，不提交模型与大轨迹
~~~

文件在相应阶段实际需要时创建，不一次性铺空架构。

| 模块 | 具体职责 |
|---|---|
| cli.py | 将命令参数交给对应函数，不放算法 |
| config.py | 读取一种固定配置格式 |
| records.py | 定义任务、实验、轨迹和候选记录 |
| model.py | 生成、token 计分、适配器切换、保存与加载 |
| policy.py | Service 的 Host 侧执行循环 |
| worker.py | 在受限执行进程中调用候选 P/M |
| alfworld.py | 环境创建、重置、动作执行和状态读取 |
| synthesis.py | 探索、目标构造、见证重放和合成任务 |
| research.py | 单组实验、双分支修改和周期编排 |
| evaluation.py | 配对评分与候选选择 |
| verl_training.py / verl_runtime.py / verl_agent_loop.py | 统一导出轨迹、接入 VeRL V1、回放和轨迹权重 |
| train.py | 数据批次、反向传播和 checkpoint |

不另建通用 ModelProvider、VerifierRegistry 或 AgentManager。实际需要的函数先放在承担该职责的模块中。

## 6. 运行方式与依赖

### 6.1 环境

首个训练目标为一台 Linux/NVIDIA 机器，使用一个固定 Python 环境。使用 Python 3.11+，并在 P0 实测 ALFWorld/TextWorld 与训练依赖能否在该环境共同运行。

主要依赖：

- ALFWorld/TextWorld：真实环境。
- PyTorch：模型计算与优化。
- Transformers：加载模型、tokenizer 和生成。
- PEFT：LoRA。
- PyYAML：配置。
- pytest：少量行为测试。

依赖安装成功后固定实际版本。无需在计划阶段编造已经验证的版本组合。

### 6.2 训练实现

首版使用 PyTorch、Transformers 和 PEFT 的直接实现：

- 使用官方模型与生成接口。
- 使用官方 LoRA 保存、加载和切换接口。
- 使用标准优化器。
- GRPO 与 OPSD 均调用固定提交的 VeRL V1/FSDP trainer；项目只实现 Host 数据与轨迹权重的适配。
- 不自写模型结构、注意力实现或优化器框架。

这样可以直接表示多次独立研究调用、不同条件的教师输入和自定义实验奖励。首版不同时接入多个训练框架。

### 6.3 模型调用

统一返回一条生成记录：

~~~text
prompt_ids
generated_ids
generated_text
behavior_logprobs
model_revision
adapter_version
sampling_parameters
~~~

token、行为概率和文本来自同一次真实生成。模型输出解析失败时，保留原始生成记录，不用第二次生成覆盖第一次轨迹。

### 6.4 配置

只保留运行需要的参数：

~~~yaml
model:
  path: /models/chosen-model
  revision: pinned-revision
alfworld:
  data_root: /data/alfworld
  config: /data/alfworld/base_config.yaml
run:
  seed: 7
  max_steps: 40
research:
  experiments_per_group: 4
  tasks_per_experiment: 2
  edits_per_branch: 2
  max_exploration_steps: 30
training:
  edit_objective: opsd
  learning_rate: 0.00001
  max_new_tokens: 1024
evaluation:
  selection_tasks: 8
  reward_tasks: 8
~~~

这些是起始配置，不是已调优参数。真实模型路径和数据位置必须显式提供。

## 7. 阶段 P0：先确认真实环境与训练能力

### 目标

尽早确认三件事：环境可以执行，小模型能产生动作，LoRA 能发生真实参数更新并重新加载。

### 需要实现

1. 建立新仓库与最小安装配置。
2. 加载官方 ALFWorld 文本环境。
3. 加载一个指定模型与 tokenizer。
4. 完成一次正常生成，记录真实 token。
5. 创建修改 LoRA。
6. 在极小的真实生成样本上执行一次训练能力检查。
7. 保存适配器，重新加载并再次生成。

P0 的训练能力检查只确认梯度与保存加载链路，不登记为算法收益。

### 拟提供命令

~~~bash
python -m experiment_rsi env-smoke --config configs/smoke.yaml --out runs/p0-env
python -m experiment_rsi model-smoke --config configs/smoke.yaml --out runs/p0-model
python -m experiment_rsi train-smoke --config configs/smoke.yaml --out runs/p0-train
~~~

### 验收

- 至少一个真实 ALFWorld 实例完成 reset、动作执行和重放。
- 使用一个可完成的短任务检查终局成功，动作由模型或明确标注的环境检查脚本产生。
- 真实生成的 token 长度和行为概率对齐。
- LoRA 至少一个训练参数发生有限、非零变化。
- B 的基础权重不变。
- 保存的适配器能够重新加载，并完成真实推理。
- 不以输出一个 checkpoint 目录作为参数更新成功的唯一依据。

### 产物

环境轨迹、生成记录、训练损失、参数变化统计、适配器和重载结果。

## 8. 阶段 P1：跑通最小研究闭环

### 目标

先使用现有 ALFWorld 任务，完成真实的程序修改和一次研究学习。此阶段明确属于算法接线阶段，尚未实现自主目标合成。

### P 的接口

候选 P 提供：

~~~python
def step(view, state):
    # 返回一个直接动作，或一次对 B 的模型请求
    ...
~~~

输入 view 包含当前公开观察、动作历史、K 和预算余量；state 是 P 自己维护的 JSON 状态。

返回格式只支持两种：

~~~json
{"kind": "action", "action": "实际环境命令", "state": {}}
~~~

~~~json
{"kind": "model", "messages": [], "state": {}}
~~~

Host 调用 B 后解析动作并送入环境。下一步历史包含实际动作和环境结果，P 可以据此更新状态。

P/M 在候选执行进程中运行，Host 持有模型、环境和评价数据。首个实现使用一种固定的受限容器运行方式：无网络、只读、仅挂载候选源码与执行入口。每个 episode 复用同一 worker，避免每个动作新建容器。

这层只解决候选代码执行与 Host 数据隔离，不扩展为一套代码审查平台。缺少执行依赖时明确失败。

### K 的接口

K 是一个 JSON 对象，先采用短文本经验列表。它通过正常模型输入影响行为。

首版不建设向量数据库、复杂检索或经验评分系统。

### 修改流程

1. 当前 Service 执行任务，产生真实报告。
2. φ_U 接收当前 P/K 与报告，输出完整的新 P 源码和 K。
3. Host 检查输出结构，并在候选进程实际加载程序。
4. 在独立任务上运行父版本与候选。
5. 根据选择集决定是否接受。
6. 用奖励集产生修改奖励。
7. 以 GRPO 更新一批真实修改轨迹，保存适配器。
8. 用更新后的适配器运行下一组研究。

### 拟提供命令

~~~bash
python -m experiment_rsi research-once --config configs/smoke.yaml --out runs/p1-research
python -m experiment_rsi train --role edit --objective grpo --run runs/p1-research --out runs/p1-update
python -m experiment_rsi research-once --config configs/smoke.yaml --edit-adapter runs/p1-update/adapter --out runs/p1-next
~~~

### 验收

- P/K 候选来自真实模型生成。
- 候选确实在真实环境中执行。
- 奖励来自父版本与候选的实际配对结果。
- 至少一组有不同奖励的修改样本完成真实 GRPO 更新。
- 下一组研究明确使用新适配器。
- 没有提高分数时如实保留父版本，不能为了展示改进写入固定补丁。

## 9. 阶段 P2：真正实现自主目标合成

### 目标

将“选择官方任务”升级为“探索后提出并验证新目标”。这是整个方法的必需阶段，不能停留在任务类型采样。

### 首个合成范围

先实现两类明确目标：

1. 将一个指定类型物体放入一个可达容器。
2. 将两个不同实例的同类物体放入指定容器。

首版代码覆盖单物体放入容器，以及两个不同实例的同类型物体放入同一容器。冷却、加热和清洗留待后续扩展。Host 从真实状态构造谓词，探索模型通过合法动作选择并完成目标对象与容器。

### 合成流程

1. 实验策略进入训练环境执行探索动作。
2. 记录实际见到的物体、容器及合法状态变化。
3. 实际完成一个候选目标。
4. 提交结构化目标、目标对应自然语言描述、检查点与见证动作。
5. Host 重新开始并重放见证。
6. Host 根据状态谓词确认最终目标。
7. 将该目标注册为可再次执行的诊断任务。

模型不能仅声称目标可达；见证必须来自实际环境交互。

### 自定义目标与终止逻辑

原 ALFWorld 任务的成功标志不能直接评价新目标。

实现方式使用官方 `.tw-pddl` JSON 中的 `pddl_problem` 和 ALFWorld TextWorld `PddlEnv`：保留源域、语法和初始对象状态，只替换平衡括号范围内的 `:goal`，并请求真实 PDDL facts。目标文件写入独立派生目录，不覆盖 ALFWorld 数据。见证重放使用派生任务的 `won` 和目标 facts 双重确认；Service 评价加载同一个派生游戏文件。

新目标、评分与终止条件必须一致。原目标达成不能错误终止新任务。

代码已经按上述方式接通；ALFWorld 验证集环境 reset/step smoke 已运行，模型生成与一次 MPS LoRA 更新也已运行。P2 仍未完成验收，因为尚未通过一次完整的模型探索、目标派生和见证重放来认证合成任务。

### 必须完成的区分性验证

构造一个测试：原任务目标与新目标不同，Agent 满足其中一个而不满足另一个。检查新目标评分确实跟随新谓词，而非原任务成功标志。

### 检查点

首版记录：

- 已观察到目标对象。
- 已持有目标对象。
- 已放置目标对象。
- 已完成的目标数量。

程序轨迹同时记录 P 的 state、发给 B 的 messages 以及实际返回动作。无需额外的学习型判官。

### 拟提供命令

~~~bash
python -m experiment_rsi synthesize --config configs/smoke.yaml --count 4 --out runs/p2-synthesis
python -m experiment_rsi replay --run runs/p2-synthesis
python -m experiment_rsi evaluate-synthesized --config configs/smoke.yaml --tasks runs/p2-synthesis/tasks.jsonl --out runs/p2-service
~~~

### 验收

- 至少生成四条实际可执行的新目标记录。
- 每条任务具有真实见证和重放结果。
- 至少一条新目标与源游戏目标不同。
- Service 得到正常观察与新目标描述，没有获得见证答案。
- 所有新目标使用自己的评分和终止逻辑。
- 保存源场景与派生任务的关联，后续划分不让同场景派生任务跨诊断与独立评价集合。

## 10. 阶段 P3：实验包与实验价值奖励

### 目标

让任务合成服务于程序改进，并区分“修改本身有效”与“实验报告确实帮助修改”。

### 实验包

一份实验包含：

~~~text
hypothesis
tasks                 # 两到三个已认证任务
checkpoints
witnesses
~~~

例如，“单物体搬运、双物体搬运、较长的单物体搬运”组成一份实验，辅助分析数量管理与长程执行问题。

实验假设不单独打分；实际报告及后续改进决定实验价值。

### 两个修改分支

对同一实验 E：

- 有证据分支：看到当前 P/K、相同历史、实验设计和当前报告 F_diag。
- 无证据分支：看到同样的信息，但当前 F_diag 明确未提供。

两个分支从同一个父版本开始，各采样 L 个修改，冻结后共同评价。

这是实验价值测量，不是训练时的教师与学生。

### 奖励

对奖励包 R：

\[
J_R(S)=\frac{1}{|R|}\sum_{x\in R}s(S,x).
\]

修改奖励：

\[
r^U_{ij}=J_R(S^+_{ij})-J_R(S).
\]

两个分支的平均收益：

\[
\bar g_i^\pm=\frac{1}{L}\sum_j[J_R(S^\pm_{ij})-J_R(S)].
\]

实验奖励：

\[
r^E_i=[\bar g_i^+]_+-[\bar g_i^-]_+.
\]

先取分支均值，再取正部。使用相同奖励包和种子，父版本评价可复用。

### 实验策略更新

同组 G 个实验从相同父版本与历史出发，在组内计算实验优势，通过 GRPO 更新 φ_E。

探索可能包含多次独立模型调用。每次调用保留自己的 prompt、response 和行为概率，共享该实验的优势。不能将不同条件的调用拼成一条虚构的连续对话。

### 拟提供命令

~~~bash
python -m experiment_rsi research-group --config configs/pilot.yaml --out runs/p3-group
python -m experiment_rsi train --role experiment --objective grpo --run runs/p3-group --out runs/p3-update
~~~

### 验收

- 报告内容差异就是两个分支之间的目标信息差。
- 独立候选、分支成绩和实验奖励均可由保存记录重算。
- 无证据分支不参与真实版本发布或修改策略训练。
- 实验奖励由每个实验自己的分支差产生，不能减一个全组共同常数充当贡献奖励。
- 至少一批真实实验完成 φ_E 的参数更新。

## 11. 阶段 P4：修改策略的反馈条件 OPSD

### 目标

利用程序修改执行后的验证反馈，为修改策略提供 token 级监督。此阶段采用 On-Policy Self-Distillation（OPSD）的信息结构：教师只比学生多看候选执行后的反馈，学生生成的响应前缀保持不变。这里的特权信息是本研究任务的环境反馈，并非数学题标准答案。

### 学生和教师分别看什么

学生正常研究上下文：

\[
c=(P,K,h,E,F_{\mathrm{diag}}).
\]

学生采样一个程序修改：

\[
y\sim\pi_\phi(\cdot\mid c).
\]

Host 执行该候选，产生事后反馈 F_post，包括候选错误、实际行为、改善与回归案例。教师看到：

\[
(c,F_{\mathrm{post}},y_{<t}).
\]

学生在相同的自身生成前缀上看到：

\[
(c,y_{<t}).
\]

学生正常需要的诊断信息保留；教师额外获得的是本次修改完成后才出现的信息。不能为了制造“看／不看”结构，把部署时必需的环境事实全部从学生输入中删除。

### 损失

训练通过 VeRL PR #6833 的原生 PPO trainer 和 FSDP actor 更新 LoRA。Host 先完成环境执行、独立任务评价与反馈记录；Replay AgentLoop 把同一条已采样的 prompt/response token 序列交给 VeRL，不重新生成候选。VeRL teacher 读取该候选的真实事后反馈，按 PR 的 chat-turn 构造器建立特权输入，再在学生原始 response token 上计算 teacher top-k 分布。训练配置关闭 task-reward policy gradient，只用 PR/VeRL 实现的 top-k forward-KL：

\[
\mathcal L_{\mathrm{OPSD}}
=\frac1{|y|}\sum_t\max\!\left(0,
\sum_{v\in\mathrm{TopK}_t}p_T(v)
\left[\log p_T(v)-\log p_S(v)\right]\right),
\quad p_T=\operatorname{sg}\pi_{\bar\phi}(\cdot\mid c,F_{\mathrm{post}},y_{<t}),\quad
p_S=\pi_\phi(\cdot\mid c,y_{<t}).
\]

教师使用本轮冻结适配器，停止梯度。VeRL 的 teacher 配置接收模型路径而非 PEFT adapter 路径，所以已有 adapter 会在训练前合并为冻结 teacher 快照；基础模型不变，学生仍以 LoRA 训练。教师每个生成 token 只返回 top-k 对数概率和 token id。教师模型单独占一张 GPU，actor/rollout 占一张 GPU；报告记录模型、adapter、VeRL commit 和 top-k 值。

### 事后反馈来源

训练反馈可以来自独立奖励任务中的候选执行报告。它只在候选冻结并执行后进入教师上下文。

这些任务属于训练数据流，不称为最终测试集。训练器只回放本轮已经由学生采样并经 Host 验证的轨迹；最终测试反馈不进入教师。

### 与实验价值对照的关系

实验价值对照：

~~~text
有 F_diag 的修改者 vs 无 F_diag 的修改者
比较两者实际成绩
~~~

反馈自蒸馏：

~~~text
正常上下文的学生
vs
额外看到 F_post 的自教师
匹配学生前缀上的 token 分布
~~~

这两条信息关系应分别记录，不能复用一个含糊的 teacher/student 字段。

### 拟提供命令

~~~bash
python -m experiment_rsi train --role edit --objective opsd --run runs/p3-group --out runs/p4-update
python -m experiment_rsi research-once --config configs/smoke.yaml --edit-adapter runs/p4-update/adapter --out runs/p4-next
~~~

### 验收

- 教师和学生使用完全相同的 response token 前缀。
- 教师只有额外事后反馈，且没有梯度。
- 损失只作用于真实生成 token。
- 教师适配器和 B 不变，学生适配器实际变化。
- checkpoint 重载后，能够继续运行真实研究。
- 不把教师生成新补丁后的普通 SFT 写成 OPSD。

## 12. 阶段 P5：连续共同演化与算法比较

### 目标

在相同预算下比较训练方式，并让系统完成多个真实研究周期。

### 三种训练配置

| 配置 | 实验策略 | 修改策略 |
|---|---|---|
| A：RL | 实验收益 GRPO | 修改收益 GRPO |
| B：蒸馏对照 | 固定实验策略 | 反馈自蒸馏 |
| C：职责组合 | 实验收益 GRPO | 反馈自蒸馏 |

同时保留固定实验策略＋修改 GRPO 的基础对照，以区分实验学习与训练目标本身的影响。

首轮不增加混合损失系数。先分别运行清楚的目标；若结果支持互补，再研究同一修改角色的 RL＋蒸馏。

### 周期规则

1. 固定本轮 Service 与两个适配器快照。
2. 采样实验，执行并生成候选。
3. 冻结候选，运行选择与奖励评价。
4. 按选择结果更新 P/K。
5. 更新本轮指定的适配器。
6. 保存新版本。
7. 下一轮明确加载新版本。

实验策略更新与修改策略更新交替进行，避免两者在同一批采样中同时漂移。

### 拟提供命令

~~~bash
python -m experiment_rsi run --config configs/pilot.yaml --method rl --periods 3 --out runs/p5-rl
python -m experiment_rsi run --config configs/pilot.yaml --method distill --periods 3 --out runs/p5-distill
python -m experiment_rsi run --config configs/pilot.yaml --method combined --periods 3 --out runs/p5-combined
~~~

### 验收

- 连续完成至少三个真实周期。
- 每次参数更新都有新 checkpoint，并被下一轮使用。
- 每次 P/K 接受或保留父版本都有实际选择成绩。
- 所有探索、两条分支和训练成本进入统计。
- 一条命令即可从配置运行到结果，不依赖手工编辑中间 JSON 接续流程。

## 13. 阶段 P6：研究策略迁移与 M 自改

### 研究策略迁移

冻结训练后的 φ，在未参与训练的 Service 起点运行相同预算研究。比较训练前后的研究收益。

使用多个正常初始执行程序；人为植入的问题仅用于流程检查，不承担主结果。

### M 的接口

M 提供一个研究决策函数：

~~~python
def next_step(view, state):
    ...
~~~

首版允许的动作包括探索、提交实验、请求修改和结束研究。M 可以改变这些步骤的顺序、信息组织及停止时机。

M 只能调用 Host 提供的固定操作，不改变评分、模型基础权重或预算统计。

### 自改评价

冻结 φ，父程序 M 与候选 M' 从相同 Service 起点和预算出发，各完成一个固定长度研究周期：

\[
r_{\mathrm{self}}
=\mathbb E_S\left[
J_R(\operatorname{Research}(S;M',\phi,b))
-
J_R(\operatorname{Research}(S;M,\phi,b))
\right].
\]

M 的选择、奖励和最终验证使用不同的研究起点。

### 拟提供命令

~~~bash
python -m experiment_rsi transfer --config configs/pilot.yaml --adapters runs/p5-combined/adapters --out runs/p6-transfer
python -m experiment_rsi self-edit --config configs/pilot.yaml --adapters runs/p5-combined/adapters --out runs/p6-self-edit
~~~

### 验收

- 自改后的 M 确实主持后续研究。
- 收益来自新起点上的实际研究结果。
- 一次评价内部只运行一个固定长度周期，不递归启动无限层自改。
- 基础机制尚未跑通时，不提前实现这一阶段。

## 14. 数据划分与选择规则

### 14.1 四类数据

| 数据 | 用途 |
|---|---|
| D_diag | 自主合成，提供诊断证据 |
| D_select | 冻结候选后选择版本 |
| D_reward | 计算训练奖励，提供训练用事后反馈 |
| D_test | 最终效果及泛化评价 |

按源场景划分，保证同一源场景的派生目标不会跨越诊断与独立评价集合。最终测试保留未见场景。

### 14.2 候选选择

父版本与候选在相同选择任务、种子和部署预算下配对运行。

起始规则：选择集平均成功率严格提高时接受，否则保留父版本。并列保留父版本。先不增加多层综合评分或独立的学习型选择器。

正式实验根据开发数据固定选择规则和评价规模，然后冻结。训练奖励不参与接受决定。

### 14.3 原始数据与统计

保存完整原始轨迹及每道题成绩；结果脚本从这些记录计算均值、配对差和成本。

同一父版本在完全相同任务和种子上的成绩可以缓存。缓存键直接由父版本标识、任务标识和种子组成，不建立额外缓存服务。

## 15. 错误处理的具体约定

| 情况 | 处理 |
|---|---|
| 任务预算内未完成 | 正常失败，分数 0 |
| 生成动作无效 | 将真实动作送给环境，记录原生反馈 |
| P/K 输出格式错误、候选程序加载失败 | 候选失败，保留原始输出和错误 |
| 实验目标未被实际见证 | 实验无效，记录固定负奖励 |
| 模型加载、环境初始化、磁盘写入失败 | 当前命令失败退出 |
| GPU OOM 或训练 NaN | 当前训练失败，不发布新适配器 |
| 训练组奖励全部相同 | 优势为 0，记录这一结果，不制造分差 |
| 参数未变化 | 不记录为成功更新，查明训练路径 |

无效实验的起始奖励设为 −1，有效实验奖励范围为 [−1,1]。候选程序失败在任务评分中按失败处理；环境或模型基础设施错误不计入候选能力评分。

不设置自动换模型、自动换数据、JSON 猜测修复、多层重试和失败后切换运行后端。真实错误修复后由用户或执行入口明确重跑。

## 16. 最小运行产物

~~~text
runs/run-name/
  config.yaml
  versions.json
  tasks.jsonl
  episodes.jsonl
  research.jsonl
  scores.jsonl
  metrics.json
  candidates/
  adapters/
~~~

| 文件 | 内容 |
|---|---|
| config.yaml | 本次实际配置 |
| versions.json | 代码提交、模型 revision、适配器和依赖版本 |
| tasks.jsonl | 任务目标、源场景、见证和检查点 |
| episodes.jsonl | 动作、观察、程序 state 与模型调用 |
| research.jsonl | 研究输入输出、token、分支、奖励和训练组 |
| scores.jsonl | 逐任务父版本与候选成绩 |
| metrics.json | 汇总成绩、成本与实际更新次数 |
| candidates/ | P/K/M 源文件 |
| adapters/ | 实际保存的适配器 |

原始记录是结果来源，汇总文件可重算。首版不另建数据库、网页或证据审批目录。

## 17. 必须运行的检查

### 行为检查

仅围绕真实接口和算法主张检查：

1. 环境重置、动作执行和轨迹重放。
2. 合成目标与源目标能够被区分。
3. P/K 修改实际改变执行行为。
4. 两分支输入满足证据移除定义。
5. 奖励可以从逐题成绩复算。
6. GRPO 分组不混合不同父版本或报告条件。
7. 蒸馏前缀对齐，教师无梯度。
8. 适配器更新、保存和重载。
9. 连续周期使用正确的新版本。

### 不用模拟结果替代的检查

环境、模型生成和参数更新必须有真实运行。局部数学公式可以用小张量测试；候选执行错误可以用一个明确的坏程序测试。无需为每个内部函数制作大量 mock。

### 阶段失败后怎样推进

定位并修复阻塞当前命令的最早错误，重跑受影响的检查。没有新修改或新失败时，不重复展开一轮全面审查。

## 18. 研究实验安排

### 18.1 最早的决定性实验

固定修改策略，比较：

- 随机有效实验。
- 失败驱动实验。
- 直接共享修改收益的实验学习。
- 使用证据贡献奖励的实验学习。

如果第四项在独立收益或预算效率上没有改善，就先分析实验奖励与证据利用，不能通过增加更多模块掩盖这一结果。

### 18.2 RL 与自蒸馏

在相同研究数据来源和预算下比较：

- 修改策略 GRPO。
- 修改策略反馈自蒸馏。
- 出题 GRPO＋修改自蒸馏。

报告完整任务成功率、达到同等表现所需研究成本，以及研究模型冻结后的迁移效果。

### 18.3 关键消融

按主结果需要开展：

- 证据贡献奖励改为共享终局收益。
- 移除中间检查点，只给最终成败。
- 将报告替换为同类型其他报告。
- 冻结实验策略或修改策略。
- 冻结 M。

### 18.4 正式报告

使用至少三个研究随机种子，并将自主探索、认证、双分支、候选评价和训练全部计入成本。

主结果分别报告诊断任务修复、相关未见任务迁移和整体测试净收益。

## 19. 资源预算与开发顺序

### 19.1 起始研究规模

G=4、L=2、每份实验 2 个任务时，一组产生：

- 4 份实验。
- 8 个诊断任务。
- 8 个有证据修改。
- 8 个无证据修改。

若奖励包为 8 道任务，则候选奖励评价为 16×8=128 次 rollout，另加父版本、选择、探索与认证成本。

先实测一组耗时，再决定总组数。不能根据参数规模直接承诺完成天数。

### 19.2 实施顺序

| 顺序 | 交付 | 开始下一阶段的依据 |
|---|---|---|
| P0 | 真实环境、模型和参数更新能力 | 能运行并重载适配器 |
| P1 | 最小程序研究闭环 | 真实修改、评价和训练接通 |
| P2 | 探索后的自主新目标 | 合成目标真实有效 |
| P3 | 实验包和贡献奖励 | φ_E 获得真实训练 |
| P4 | 反馈自蒸馏 | 密集监督真实更新并可继续研究 |
| P5 | 连续共进化与训练比较 | 多周期及等成本主结果 |
| P6 | 迁移与 M 自改 | 新起点研究能力得到验证 |

### 19.3 执行每一阶段时的交付说明

交付只需要说明：

- 改了哪些文件、实现了什么。
- 执行的真实命令。
- 输出目录。
- 阶段验收是否满足。
- 尚未运行的项目，以及阻塞它的具体原因。

不把待实现功能写成已完成，不用审查数量或代码行数证明阶段成功。

## 20. 计划完成后应得到的系统

一个独立、可从命令行运行的研究仓库。它能够在真实 ALFWorld 环境中自主形成任务和执行证据，通过程序及经验修改改善固定模型 Agent，并用实际研究收益或反馈自蒸馏更新研究模型。每个阶段都有可执行入口；正式结果能够从原始轨迹重算。

下一步是在 Linux/NVIDIA 机器上完成 P3 的真实研究组与实验角色 GRPO 更新，再验证 OPSD 更新及跨周期训练；本机已有的 MPS smoke 不代替这些验收。

## 参考资料

- [ALFWorld 官方环境与安装说明](https://github.com/alfworld/alfworld)
- [PEFT 模型、适配器加载与切换接口](https://huggingface.co/docs/peft/main/en/package_reference/peft_model)
- [Self-Distilled Reasoner：OPSD](https://arxiv.org/html/2601.18734v1)
- [Reinforcement Learning via Self-Distillation：SDPO](https://arxiv.org/html/2601.20802v2)
- [CoEvolve](https://arxiv.org/html/2604.15840v1)
- [Explorer-Zero](https://arxiv.org/html/2610.19868v1)

技能约束来源：

- stop-that-shit
- avoid-overdefensive-programming
- avoid-overkill
