# Remedy

自主实验驱动的递归 Agent 改进研究原型（Recursive Self-Improvement, RSI）。

从零实现“自主实验驱动的递归 Agent 改进”。原型包含 ALFWorld/TextWorld 交互、自定义 PDDL 目标合成与见证重放、实验价值对照、Service 配对评价，以及实验/修改策略的训练路径。

研究问题、方法定义、实现对应关系和预注册式实验计划见 [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)。

## 当前验证状态

- 自主任务合成的最小端到端验收通过：从 4 个不同 ALFWorld 场景生成 4 个单物体子目标，见证在派生目标环境中重放 4/4 成功，3B Solver 在这 4 个任务上成功 4/4。
- Qwen2.5-0.5B 在 FloorPlan304、FloorPlan313、FloorPlan316 的固定场景上也生成过可验证任务。
- 这组小规模结果验证了“环境探索 → 新目标 → 见证认证 → 独立求解”链路，不代表跨场景泛化或显著性结论。P3 的完整实验组和研究策略参数更新尚未跑通；3B 采样在 M4 的 16 GB 统一内存上达到 18.7 GB 峰值，0.5B 随机 P3 组未生成诊断任务。
- VeRL PR #6833 的 OPSD trainer 已固定并完成导入及配置检查，但原生 CUDA/FSDP/vLLM 训练尚未运行。本机无 NVIDIA GPU。低频 M 自改、策略迁移和完整多周期训练也尚未验收。

## 安装

需要 Python 3.11+、Docker。Linux/NVIDIA GPU 用于正式 VeRL/vLLM 训练；Apple Silicon 可以安装依赖并用 MPS 做采样与环境流程调试；训练由 Linux/NVIDIA 上的 VeRL 执行。

实验策略 GRPO、编辑策略 GRPO 和编辑策略 OPSD 统一使用 VeRL PR [#6833](https://github.com/verl-project/verl/pull/6833) 的固定提交及 V1 trainer。Host 负责环境执行与奖励采集；`verl_training.py` 导出记录并调用原生 trainer，`verl_agent_loop.py` 原样回放 prompt/response token 和行为 log-probability。GRPO 每个数据行对应一个研究组，每个 session 对应一条完整研究轨迹，多次模型调用作为同一 session 的多个输出；VeRL 原生计算轨迹级组内优势。`verl_runtime.py` 保留每条轨迹内各次调用的权重，梯度、优化器、调度器和 checkpoint 都由 VeRL/FSDP 管理。项目不再维护独立的 GRPO 损失或优化器循环。

OPSD 使用同一入口，通过 PR 的 chat-turn 特权反馈与 top-k forward-KL 更新编辑 LoRA；每个候选单独携带自己的 Host 反馈。已有 LoRA 会先合并为冻结 teacher 快照。`pilot` 默认用一张 NVIDIA GPU 做 GRPO；`full_remedy.yaml` 将 Designer GRPO 配为两卡 VeRL/FSDP，Editor OPSD 则用一张 actor/rollout 卡和一张 teacher 卡。本次正式训练使用 zju-12 的物理 GPU 1、2。

`configs/smoke.yaml` 固定使用 Qwen2.5-0.5B-Instruct，适合在 Apple Silicon/MPS 或普通 CPU 上采样；训练 smoke 仍需要 NVIDIA GPU 和 VeRL。`configs/pilot.yaml` 是 Linux/NVIDIA 起始配置：固定到 Qwen2.5-1.5B-Instruct revision，使用 FP16、rank-4 LoRA、梯度检查点、每组 2 个实验和每个分支 2 个修改；正式比较时增加独立任务和重复种子。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# Apple Silicon/CPU sampling and config inspection
python -m pip install -e '.[verl]'
# Linux/NVIDIA: use the CUDA 12.8 PyTorch wheel, then install the pinned trainer stack
# python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
#   --index-url https://download.pytorch.org/whl/cu128
# python -m pip install -e '.[verl-gpu]'
export ALFWORLD_DATA="$HOME/.cache/alfworld"
alfworld-download --data-dir "$ALFWORLD_DATA"
docker pull python:3.11-slim
```

`.[verl]` 固定安装 PR #6833 对应的 VeRL commit，并带上该 trainer 实际导入的缓存、TransferQueue 与服务依赖。`.[verl-gpu]` 固定 PyTorch 2.8.0+cu128、vLLM 0.11.0、torchvision 0.23.0+cu128、torchaudio 2.8.0+cu128、xFormers 0.0.32.post1，以及 vLLM 0.11.0 测试锁定的 Transformers 4.55.2、Tokenizers 0.21.1、Hugging Face Hub 0.34.3。VeRL 使用同一提交的 V1 trainer。vLLM 0.11 已移除 V0 engine，因此与 `trainer.use_v1=True` 对齐，不需要运行时兼容改写。当前 Apple Silicon 环境可生成研究样本，但 MPS 不支持 VeRL 的 CUDA/FSDP/vLLM trainer。

ALFWorld 下载文本游戏和 PDDL 数据后，设置 `ALFWORLD_DATA` 指向数据根目录；`configs/alfworld_text.yaml` 使用官方数据目录结构。smoke 配置中的模型 revision 已固定，pilot 配置需要填写可访问的模型路径。配置不自动替换缺失的模型、数据或运行设备。

## P0 命令

```bash
remedy env-smoke --config configs/smoke.yaml --out runs/p0-env
remedy model-smoke --config configs/smoke.yaml --out runs/p0-model
remedy train-smoke --config configs/smoke.yaml --out runs/p0-train
```

`train-smoke` 采集两条真实生成，用长度奖励运行一次原生 VeRL GRPO，导出 LoRA 并从基础模型重新加载生成。长度奖励只用于验证训练链路；研究实验使用 ALFWorld 的真实评价奖励。

## 当前研究闭环

```bash
remedy synthesize --config configs/smoke.yaml --split eval_in_distribution --count 2 --out runs/selection
remedy synthesize --config configs/smoke.yaml --split eval_out_of_distribution --count 2 --out runs/reward
remedy research-group --config configs/smoke.yaml \
  --selection-tasks runs/selection/tasks.jsonl \
  --reward-tasks runs/reward/tasks.jsonl \
  --out runs/group-0
remedy train --config configs/smoke.yaml --role experiment --objective grpo \
  --run runs/group-0 --out runs/experiment-update
remedy train --config configs/smoke.yaml --role edit --objective grpo \
  --run runs/group-0 --out runs/edit-update
remedy train --config configs/smoke.yaml --role edit --objective opsd \
  --run runs/group-0 --out runs/edit-opsd-update
remedy research-group --config configs/smoke.yaml \
  --parent runs/group-0/service.json \
  --experiment-adapter runs/experiment-update/adapter_export/lora_adapter \
  --edit-adapter runs/edit-opsd-update/adapter_export/lora_adapter \
  --selection-tasks runs/selection/tasks.jsonl \
  --reward-tasks runs/reward/tasks.jsonl \
  --out runs/group-1
```

`research-group` 在训练 split 上采样实验；每份实验来自模型提出的假设和后续真实 ALFWorld 探索。Host 读取 TextWorld 当前 PDDL 状态中新出现的 `inReceptacle` 谓词，合成单物体放置目标；若两个不同实例同类型物体被放入同一容器，也合成双物体目标。Host 从源 `.tw-pddl` 复制任务文件，只替换 `:goal`，再从初始状态重放探索见证；只有新 PDDL 目标确实使环境返回 `won` 且谓词成立时，任务才会注册。Solver 从派生任务文件重新开始，成功和进展都由目标谓词及 `holds` 状态计算，原始任务目标与动作反馈句不参与新目标判定。相同实验分别进入“有诊断报告”和“无诊断报告”两个修改分支，二者在同一奖励任务上评分。实验策略收到两分支正向进展收益之差作为 GRPO 奖励；修改策略只使用有证据分支候选，并按候选在奖励集上的目标谓词进展差训练。无证据候选只用于归因对照。Service 接受仍依据选择集上的完整任务成功率，最终测试也报告完整任务成功率。场景交叉会直接报错。

`P` 是候选 Python 策略，`K` 是 JSON 经验列表。P 通过 `step(view, state)` 返回直接动作或模型请求；Host 持有模型、环境和评分逻辑，并仅把公开观察、动作历史、合法命令、K 与剩余预算传给 P。候选在无网络、只读文件系统、有限 CPU/内存的 Docker 容器内运行，每个 episode 复用一个 worker。目标由真实轨迹中新出现的 PDDL 放置谓词构成，源任务文件保留场景、对象、域和语法，派生文件只替换目标公式。每条任务保存完整成功见证轨迹；Host 在派生目标环境中重放见证并用真实 PDDL `won` 和谓词状态认证，Solver 也从这个派生任务的初始状态开始。探索生成、见证重放和求解都使用 ALFWorld 的合法文本动作。

P 的返回值采用两个明确结构：直接动作返回 `{"kind":"action","action":"...","state":{...}}`；请求 Host 调用基础模型返回 `{"kind":"model","messages":[...],"state":{...}}`。worker 只运行 Python 标准库代码，不能直接访问模型权重、环境对象或评价任务。

编辑策略可用 `--objective opsd` 训练：样本中的 response 是当前策略在真实研究 prompt 上生成的原始 token；自定义 VeRL AgentLoop 原样回放它们，不重新采样。teacher 额外看到 Host 执行候选后记录的验证反馈，并在同一 response 前缀上计算 top-k 分布；VeRL 的 forward-KL 更新学生 LoRA。训练只接收有诊断证据分支，最终测试反馈不会进入 teacher。直接 GRPO 是对照方法。

多周期可由 `run` 自动编排。selection/reward pool 按源场景切成不重叠周期块；最终测试池单独传入，并只在末尾评价：

```bash
remedy run --config configs/pilot.yaml \
  --selection-pool runs/selection-pool/tasks.jsonl \
  --reward-pool runs/reward-pool/tasks.jsonl \
  --periods 3 --method combined \
  --test-tasks runs/test/tasks.jsonl \
  --out runs/combined
```

`--method rl` 同时用 GRPO 更新实验和修改策略；`distill` 冻结实验策略、用 OPSD 更新修改策略；`combined` 用实验价值 GRPO 和修改反馈 OPSD。多个周期的自动演化已接通，M 自改尚未实现。

正式 combined 主实验使用 `configs/full_remedy.yaml`，每周期包含多个在线批次：每批用当前 Designer 采集新的 8 条实验轨迹、生成两个证据分支的修改候选，然后分别执行一次 VeRL GRPO 和一次 VeRL OPSD 更新。一个周期内 Service 父版本固定；批内用该周期的任务块筛选候选，周期结束时再在完整 Selection 池上配对复核，确认下一周期的 Service。Selection、Reward、ID Test 和 OOD Test 按源场景分区，测试任务只用于首尾评价。任务池少于周期数时，只在自身角色池内轮转复用，绝不跨角色共享。先生成固定任务池，再启动训练：

```bash
python scripts/prepare_full_pools.py --config configs/full_remedy.yaml \
  --out runs/full-pools
python scripts/run_with_swanlab_netrc.py run --config configs/full_remedy.yaml \
  --selection-pool runs/full-pools/selection/tasks.jsonl \
  --reward-pool runs/full-pools/reward/tasks.jsonl \
  --test-tasks runs/full-pools/id_test/tasks.jsonl \
  --ood-test-tasks runs/full-pools/ood_test/tasks.jsonl \
  --periods 10 --batches-per-period 20 --method combined \
  --out runs/full-combined
```

启动脚本从服务器已有的 `~/.swanlab/.netrc` 读取凭据，并只把 `SWANLAB_API_KEY` 注入训练进程环境，不写入仓库。VeRL 将每次 Designer/Editor 更新分别记入 SwanLab；研究主运行还记录任务合成率、修改收益、Service 选择集成绩、环境交互量和测试集首尾差值。`progress.json` 与各批次的训练日志用于核对真实完成的 optimizer updates；中断或报错不会被记作完整实验。

## 设计边界

- ALFWorld 负责真实状态转移和原生任务反馈；研究代码不伪造环境成绩。
- 每条生成记录保留同次采样的 prompt token、completion token 和行为 log-probability。
- 模型权重、环境数据和 GPU 设备由配置明确指定。
- 环境/模型/磁盘/训练错误会让命令失败；任务未完成才是算法分数 0。

数据来自 Hugging Face 上固定 revision 的 ALFWorld 原始游戏归档镜像，归档 SHA-256 已核对。本机已完成 ALFWorld reset/step、模型生成和一次 MPS LoRA smoke；硬件与更完整训练的限制见上方状态。

## 实验室服务器训练环境

服务器训练环境使用 `/home/hejingyu/projects/Remedy/.venv-verl-cu128/bin/python`（Python 3.11），独立安装 PyTorch 2.8.0+cu128 与 vLLM 0.11.0，并安装固定 VeRL commit 和 Remedy。它不继承或改写 ROSA 共享环境的 PyTorch/vLLM 包。旧 `.venv-verl` 继承 PyTorch 2.7.1+cu128、vLLM 0.10.0，只保留用于复现此前失败日志；旧 `.venv` 是 Python 3.10，无法导入当前 VeRL 的 `StrEnum` 路径。

每次更新保存 `samples.parquet`、完整 `trainer.log`、VeRL checkpoint、`adapter_export/lora_adapter` 和 `train.json`。`train.json` 仅在 trainer、checkpoint 和 adapter 导出均成功后生成；失败查看同目录日志。新 adapter 的首步参数变化使用初始为零的 LoRA B 权重核对，继续训练则与传入 adapter 比较。
