from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput


class ResearchReplayAgentLoop(AgentLoopBase):
    """Replay one Host-collected on-policy candidate for VeRL's OPSD update."""

    async def run(self, sampling_params, priority=0, **kwargs):
        prompt_ids = [int(token) for token in kwargs["replay_prompt_ids"]]
        response_ids = [int(token) for token in kwargs["replay_response_ids"]]
        response_logprobs = [float(value) for value in kwargs["replay_response_logprobs"]]
        if len(response_ids) != len(response_logprobs):
            raise ValueError("replayed response tokens and behavior log-probabilities must have equal lengths")

        reward = kwargs["rsi_reward"]
        if hasattr(reward, "item"):
            reward = reward.item()

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            reward_score=float(reward),
            num_turns=1,
            metrics=AgentLoopMetrics(),
            extra_fields={"turn_scores": [], "tool_rewards": []},
        )
