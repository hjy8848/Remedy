import json
from pathlib import Path

import torch

from .config import load_config
from .losses import grpo_sample_loss, group_advantages
from .model import ModelSession


def read_samples(run_dir, role):
    path = Path(run_dir) / "research.jsonl"
    samples = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            if sample["role"] == role and sample["training_eligible"]:
                samples.append(sample)
    if not samples:
        raise ValueError("run has no training samples for role {}".format(role))
    return samples


def train_grpo(config, samples, out_dir, adapter_path=None):
    torch.manual_seed(config["run"]["seed"])
    adapter_version = str(adapter_path) if adapter_path else "base"
    if {sample["adapter_version"] for sample in samples} != {adapter_version}:
        raise ValueError("training adapter must match the adapter used to collect these samples")
    model_revision = str(config["model"].get("revision", "unspecified"))
    if {sample["model_revision"] for sample in samples} != {model_revision}:
        raise ValueError("training model revision must match the revision used to collect these samples")
    session = ModelSession(config, adapter_path=adapter_path, trainable=True)
    trainable = [(name, parameter) for name, parameter in session.model.named_parameters() if parameter.requires_grad]
    before = [(name, parameter.detach().clone()) for name, parameter in trainable]
    group_ids = [sample["group_id"] for sample in samples]
    trajectory_ids = [sample["task_metrics"].get("trajectory_id", str(index)) for index, sample in enumerate(samples)]
    advantages = group_advantages(
        [sample["reward"] for sample in samples], group_ids, trajectory_ids
    )
    for sample in samples:
        if sample["sampling_parameters"]["temperature"] <= 0:
            raise ValueError("GRPO samples must come from stochastic generation")
    trajectory_count = len(set(zip(group_ids, trajectory_ids)))
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=config["training"]["learning_rate"],
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    sample_losses = []
    for index, sample in enumerate(samples):
        current = session.response_logprobs(
            sample["prompt_ids"],
            sample["generated_ids"],
            sample["sampling_parameters"],
        )
        behavior = torch.tensor(sample["behavior_logprobs"], device=session.device)
        loss = grpo_sample_loss(current, behavior, advantages[index].to(session.device))
        weight = sample["task_metrics"].get("sample_weight", 1.0)
        (loss * weight / trajectory_count).backward()
        sample_losses.append(float(loss.detach().cpu().item()))
    optimizer.step()
    changed = sum(
        float((parameter.detach() - initial).abs().sum().item())
        for (_, parameter), (_, initial) in zip(trainable, before)
    )
    out_dir = Path(out_dir)
    adapter_dir = out_dir / "adapter"
    session.save_adapter(adapter_dir)
    summary = {
        "role": samples[0]["role"],
        "objective": "grpo",
        "sample_count": len(samples),
        "group_count": len(set(group_ids)),
        "mean_loss": sum(sample_losses) / len(sample_losses),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "trainable_parameter_l1_change": changed,
        "parameter_update": changed > 0,
        "adapter": str(adapter_dir.resolve()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_opsd(config, samples, out_dir, adapter_path=None):
    from .verl_opsd import train_opsd_with_verl

    return train_opsd_with_verl(config, samples, out_dir, adapter_path)


def train_run(config_path, run_dir, out_dir, role, objective, adapter_path=None):
    config = load_config(config_path)
    samples = read_samples(run_dir, role)
    if objective == "grpo":
        return train_grpo(config, samples, out_dir, adapter_path)
    if objective == "opsd":
        if role != "edit":
            raise ValueError("OPSD is defined for the edit role")
        return train_opsd(config, samples, out_dir, adapter_path)
    raise ValueError("unknown training objective: {}".format(objective))
