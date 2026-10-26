import asyncio
import json
import unittest

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories

from experiment_rsi.verl_agent_loop import ResearchReplayAgentLoop
from experiment_rsi.verl_runtime import trajectory_loss_scales
from experiment_rsi.verl_training import dataset_rows


def sample(trajectory, reward, weight):
    return {
        "group_id": "group", "prompt_ids": [1, 2], "generated_ids": [3, 4],
        "behavior_logprobs": [-0.3, -0.4], "reward": reward,
        "task_metrics": {
            "trajectory_id": trajectory, "sample_weight": weight,
            "prompt_messages": [{"role": "user", "content": "task"}],
            "post_feedback": {"result": trajectory},
        },
    }


class ReplayTrainingTests(unittest.TestCase):
    def test_grouped_replay_preserves_distinct_calls_and_rewards(self):
        samples = [sample("a", 0, 0.5), sample("a", 0, 0.5), sample("b", 2, 1)]
        rows = dataset_rows(samples, "grpo", None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["__rollout_n__"], 2)
        loop = ResearchReplayAgentLoop.__new__(ResearchReplayAgentLoop)
        calls = asyncio.run(loop.run({}, session_id=0, replay_trajectories=rows[0]["replay_trajectories"]))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].prompt_ids, samples[0]["prompt_ids"])
        self.assertEqual(calls[0].response_logprobs, samples[0]["behavior_logprobs"])
        self.assertEqual(calls[0].reward_score, 0)

    def test_native_grpo_normalizes_trajectories_before_weighting_calls(self):
        keys = ["group_0_0", "group_0_1", "group_1_0"]
        data = DataProto(
            batch=TensorDict({
                "token_level_rewards": torch.tensor([[0.0], [0.0], [2.0]]),
                "response_mask": torch.ones(3, 1),
            }, batch_size=3),
            non_tensor_batch={"uid": np.array(["group"] * 3, dtype=object)},
        )
        data = compute_advantage_for_multi_trajectories(data, keys, "grpo")
        advantages = data.batch["advantages"].flatten()
        self.assertAlmostEqual(advantages[0].item(), advantages[1].item())
        self.assertAlmostEqual(advantages[0].item(), -advantages[2].item())
        scales = trajectory_loss_scales(keys, [
            {"remedy_sample_weight": 0.5}, {"remedy_sample_weight": 0.5},
            {"remedy_sample_weight": 1.0},
        ])
        contributions = advantages * scales / 3
        self.assertAlmostEqual(contributions[:2].sum().item(), -contributions[2].item())

    def test_opsd_keeps_each_candidates_feedback_separate(self):
        rows = dataset_rows([sample("a", -1, 1), sample("b", 0, 1)], "opsd", None)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["__rollout_n__"] for row in rows], [1, 1])
        self.assertEqual(
            [json.loads(row["reward_model"]["ground_truth"])["result"] for row in rows], ["a", "b"]
        )


if __name__ == "__main__":
    unittest.main()
