import json
import statistics
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

    groups = {}
    for sample in samples:
        metrics = sample["task_metrics"]
        group_id = sample["group_id"]
        trajectory_id = metrics.get("trajectory_id", str(len(groups)))
        trajectory_rewards = groups.setdefault(group_id, {})
        reward = float(sample["reward"])
        if trajectory_id in trajectory_rewards and trajectory_rewards[trajectory_id] != reward:
            raise ValueError("one GRPO trajectory has inconsistent rewards")
        trajectory_rewards[trajectory_id] = reward
    group_variances = [
        statistics.pvariance(trajectory_rewards.values())
        for trajectory_rewards in groups.values()
    ]
    rewards = [reward for group in groups.values() for reward in group.values()]
    signal = {
        "grpo_reward_mean": statistics.mean(rewards),
        "grpo_reward_variance": statistics.pvariance(rewards),
        "grpo_group_count": len(groups),
        "grpo_nonzero_advantage_group_count": sum(variance > 0 for variance in group_variances),
        "grpo_nonzero_advantage_group_fraction": (
            sum(variance > 0 for variance in group_variances) / len(groups)
        ),
    }
    result = train_with_verl(config, samples, out_dir, "grpo", adapter_path)
    result.update(signal)
    result["effective_update"] = bool(
        result["trainer_step"] > 0
        and result["parameter_update"]
        and signal["grpo_nonzero_advantage_group_count"] > 0
    )
    return result


def train_opsd(config, samples, out_dir, adapter_path=None):
    from .verl_training import train_with_verl

    result = train_with_verl(config, samples, out_dir, "opsd", adapter_path)
    result["effective_update"] = bool(
        result["trainer_step"] > 0 and result["parameter_update"]
    )
    return result


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
