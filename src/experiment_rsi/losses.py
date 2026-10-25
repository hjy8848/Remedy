import torch


def group_advantages(rewards, group_ids, trajectory_ids=None):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    advantages = torch.zeros_like(rewards)
    trajectory_ids = trajectory_ids or [str(index) for index in range(len(rewards))]
    group_units = {}
    for index, group_id in enumerate(group_ids):
        group_units.setdefault(group_id, {}).setdefault(trajectory_ids[index], index)
    for group_id, units in group_units.items():
        unit_indices = list(units.values())
        group = rewards[unit_indices]
        scale = group.std(unbiased=False)
        if scale > 0:
            normalized = (group - group.mean()) / scale
            unit_advantage = dict(zip(units.keys(), normalized))
            for index, value in enumerate(group_ids):
                if value == group_id:
                    advantages[index] = unit_advantage[trajectory_ids[index]]
    return advantages


def grpo_sample_loss(current_logprobs, behavior_logprobs, advantage, clip_range=0.2):
    if current_logprobs.shape != behavior_logprobs.shape:
        raise ValueError("current and behavior token log-probabilities must align")
    ratio = torch.exp((current_logprobs - behavior_logprobs).clamp(-20.0, 20.0))
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    return -torch.minimum(ratio * advantage, clipped * advantage).mean()
