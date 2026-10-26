import json
from pathlib import Path

from .config import load_config


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
    from .verl_training import train_with_verl

    return train_with_verl(config, samples, out_dir, "grpo", adapter_path)


def train_opsd(config, samples, out_dir, adapter_path=None):
    from .verl_training import train_with_verl

    return train_with_verl(config, samples, out_dir, "opsd", adapter_path)


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
