import json
from pathlib import Path


def load_service(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("service file must contain a JSON object")
    if not isinstance(value["policy_source"], str) or not isinstance(value["knowledge"], list):
        raise ValueError("service must contain policy_source text and a knowledge list")
    if not all(isinstance(item, str) for item in value["knowledge"]):
        raise ValueError("every knowledge entry must be text")
    return value


def make_policy_view(observation, objective, history, commands, knowledge, remaining_steps):
    return {
        "observation": observation,
        "objective": objective,
        "history": history,
        "admissible_commands": commands,
        "knowledge": knowledge,
        "remaining_steps": remaining_steps,
    }
