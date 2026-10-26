import json

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput


class ResearchReplayAgentLoop(AgentLoopBase):
    """Replay every model call in one Host-collected research trajectory."""

    async def run(self, sampling_params, priority=0, **kwargs):
        trajectories = json.loads(kwargs["replay_trajectories"])
        trajectory = trajectories[int(kwargs["session_id"])]
        return [
            AgentLoopOutput(
                prompt_ids=call["prompt_ids"],
                response_ids=call["response_ids"],
                response_mask=[1] * len(call["response_ids"]),
                response_logprobs=call["response_logprobs"],
                reward_score=call["reward"],
                num_turns=1,
                metrics=AgentLoopMetrics(),
                extra_fields={
                    "turn_scores": [], "tool_rewards": [], "reward_extra_info": {},
                    "min_global_steps": 0, "max_global_steps": 0,
                    "remedy_sample_weight": call["sample_weight"],
                },
            )
            for call in trajectory
        ]
