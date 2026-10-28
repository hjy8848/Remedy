# Remedy

Remedy 研究一个问题：**语言模型能否根据 Solver 的真实失败，主动设计有用的训练任务，让 Solver 通过反复学习变得更强？**

当前原型面向 Text ALFWorld。Designer 根据失败反馈构造任务，Planner 给出可验证的解题轨迹，Solver 用这些轨迹进行 LoRA 监督微调。每一轮用更新后的 Solver 重新诊断，形成下一轮任务。研究假设、实验对照与实现边界见 [研究计划](docs/RESEARCH_PLAN.md)。

```text
Solver 在训练场景中做题 → 汇总失败反馈 → Designer 构造任务
       ↑                                  ↓
       └── 更新 Solver LoRA ← SFT 数据 ← Planner 求解 + 环境重放
```

## 当前方法

1. **诊断**：Solver 在 ALFWorld `train` 场景中执行原始任务。程序记录目标、动作、观察、是否成功和失败行为。
2. **出题**：冻结的 Designer 接收诊断摘要和 ALFWorld 类型兼容表，提出假设并指定物体、容器、初始位置和放置目标。当前构造器从 ALFWorld 的 PDDL 规则组装新的文本场景，不需要先选择原始 FloorPlan。
3. **求解与验证**：TextWorld 调用 Fast Downward 求解；Host 检查目标在初始状态未满足，并在 ALFWorld 中从初始状态重放动作。仅使用重放成功且目标达成的轨迹。Planner 超时表示可解性未知。
4. **训练**：把每一步的任务、观察、历史和合法动作整理为输入，以 Planner 动作为标签，用 Transformers Trainer 和 PEFT 训练 Solver LoRA。训练和实际执行使用一致的输入结构。
5. **循环与评测**：下一轮加载更新后的 Solver，重新诊断、生成数据并训练。独立开发集用于跟踪进度，最终比较原始与更新后 Solver 在 ID/OOD 测试集上的成功率。

Designer 和 Solver 可以共用 Qwen3‑4B 基础权重：Designer 推理时不使用 Solver adapter，Solver 使用当前轮的 LoRA。当前主实验不训练 Designer，也不调用 Editor、Skills、P/K 策略选择、GRPO 或 OPSD；仓库保留的旧实现和[历史方案](docs/archive/RESEARCH_PLAN_LEGACY.md)用于追溯。

当前场景构造支持 TextWorld/PDDL 的物体放置任务：容器每种类型一个实例，物体可有多个实例和目的地。生成的是符合 ALFWorld 符号规则的文本场景，不是新建的 3D FloorPlan。任务需要通过规划与环境重放才能用于训练。

## 运行

需要 Python 3.11、ALFWorld/TextWorld 数据、TextWorld 可用的 Fast Downward 求解器，以及支持 Qwen3 的 Transformers、PEFT 和可用的模型权重。安装项目依赖并将 `ALFWORLD_DATA` 指向包含 `logic` 的 ALFWorld 数据目录：

```bash
python -m pip install -e .
export ALFWORLD_DATA=/path/to/alfworld
export PYTHONPATH=src
```

启动一轮从零开始的合成与 Solver SFT：

```bash
python scripts/run_skills_experiment.py \
  --config configs/synthetic_sft.yaml \
  --model /path/to/Qwen3-4B \
  --synthetic-logic-dir "$ALFWORLD_DATA/logic" \
  --periods 1 --tasks 20 --diagnostics 10 \
  --id-test 8 --ood-test 4 --attempts 4 --gpus 1 \
  --out runs/solver-sft
```

该入口要求指定数量的任务均通过验证才开始训练。默认每条轨迹最多 20 步，LoRA rank 为 4。生成尝试、诊断、训练样本、adapter 和评测结果保存在输出目录。该入口在每轮结束时评估 ID/OOD 测试集；多轮研究如需将最终测试留到末尾，应使用下面的续跑入口及独立开发集。

已有完整种子轮（包含 `manifest.json`、`initial_test.json`、`period-01/tasks.jsonl`、`solver_sft.jsonl` 和 `solver_adapter`）时，可继续运行：

```bash
python scripts/run_synthetic_sft_cycles.py \
  --config configs/synthetic_sft.yaml --model /path/to/Qwen3-4B \
  --seed-run /path/to/seed-run --out runs/solver-sft-continuation \
  --logic-dir "$ALFWORLD_DATA/logic" \
  --cycles 2 --tasks 20 --minimum-tasks 8 --diagnostics 10 \
  --id-dev 8 --attempts 4 --gpus 1 --train-device cuda:0
```

这里的 `--cycles 2` 表示在种子轮之后增加两轮。每轮累积种子轮和后续轮次的已验证样本训练一次，开发集每轮评估，ID/OOD 测试在末尾评估。已有输出目录因中断而续跑时，使用相同参数并加 `--resume`；已完成的诊断、任务和轮次会被读取，未完成的训练会从该轮起点重新执行。

## 实验状态

原型已完成过一轮 Qwen3‑4B Solver LoRA SFT，并开展后续多轮实验。正在运行的续跑中，1 个开发场景与种子轮诊断场景重复，第 3 轮诊断也复用了 1 个种子轮场景；后续新运行的分配会排除种子轮场景。单轮结果和正在运行的续跑不构成方法有效性的结论；需要在相同 Solver、规划器和训练预算下，与随机出题等对照比较。

旧的 VeRL、Editor 和 P/K 相关代码仍在仓库中，当前 Solver SFT 流程不调用这些路径。
