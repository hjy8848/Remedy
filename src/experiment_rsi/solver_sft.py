"""Create planner-supervised Solver data and train its LoRA adapter."""

import json
from pathlib import Path

from .alfworld import AlfworldEpisode
from .evaluation import direct_action_messages, replace_native_objective
from .synthesis import goal_reached


def collect_solver_sft_examples(config, tasks):
    examples = []
    step_budget = config["run"]["max_steps"]
    for task in tasks:
        history = []
        with AlfworldEpisode(
            config["alfworld"]["config_path"],
            game_file=task.task_game_file,
            split=task.source_split,
        ) as env:
            view = env.reset()
            if goal_reached(task.goal_atoms, view["facts"]):
                raise ValueError("SFT task goal is already satisfied at reset: {}".format(task.task_id))
            for step_index, action in enumerate(task.witness_actions):
                if step_index >= step_budget:
                    raise ValueError("planner trajectory exceeds the Solver step budget: {}".format(task.task_id))
                if action not in view["admissible_commands"]:
                    raise ValueError(
                        "planner action is not admissible while collecting SFT data: {} / {!r}".format(
                            task.task_id, action
                        )
                    )
                observation = replace_native_objective(view["observation"], task.instruction)
                messages = direct_action_messages(
                    task.instruction,
                    observation,
                    history,
                    view["admissible_commands"],
                    step_budget - step_index,
                )
                examples.append({
                    "task_id": task.task_id,
                    "step": step_index,
                    "messages": messages,
                    "target_action": action,
                })
                result = env.step(action)
                history.append({"action": action, "observation": result["observation"]})
                view = result
                if result["done"] and step_index + 1 < len(task.witness_actions):
                    raise ValueError("planner trajectory continues after environment termination: {}".format(task.task_id))
            if not view["won"] or not goal_reached(task.goal_atoms, view["facts"]):
                raise ValueError("planner trajectory did not complete its SFT task: {}".format(task.task_id))
    if not examples:
        raise ValueError("no planner actions were available for Solver SFT")
    return examples


def _read_examples(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def train_solver_sft(config, dataset_path, output_dir, parent_adapter=None, device="cuda:0"):
    import torch
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments, set_seed

    from .model import ModelSession

    examples = _read_examples(dataset_path)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    session = ModelSession(
        config,
        adapter_path=parent_adapter,
        trainable=True,
        device=device,
    )
    tokenizer = session.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Solver SFT requires a tokenizer pad or EOS token")

    class PlannerDataset(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            example = examples[index]
            prompt_ids = tokenizer.apply_chat_template(
                example["messages"], tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )
            full_ids = tokenizer.apply_chat_template(
                example["messages"] + [{"role": "assistant", "content": example["target_action"]}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("chat template changed the inference prompt before its target action")
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            return {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    def collate(rows):
        width = max(row["input_ids"].numel() for row in rows)
        input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        for index, row in enumerate(rows):
            length = row["input_ids"].numel()
            input_ids[index, :length] = row["input_ids"]
            attention_mask[index, :length] = row["attention_mask"]
            labels[index, :length] = row["labels"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(config["run"]["seed"])
    dtype = config["model"]["dtype"]
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=dtype == "bfloat16",
        fp16=dtype == "float16",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=session.model,
        args=arguments,
        train_dataset=PlannerDataset(),
        data_collator=collate,
        tokenizer=tokenizer,
    )
    trained = trainer.train()
    session.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "train_metrics": trained.metrics,
        "log_history": trainer.state.log_history,
        "training_examples": len(examples),
        "parent_adapter": str(parent_adapter) if parent_adapter else None,
        "adapter": str(output_dir.resolve()),
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
