import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class GenerationRecord:
    prompt_ids: List[int]
    generated_ids: List[int]
    generated_text: str
    behavior_logprobs: List[float]
    model_revision: str
    adapter_version: str
    sampling_parameters: Dict[str, Any]


@dataclass
class EpisodeStep:
    action: str
    observation: str
    admissible_commands: List[str]
    done: bool
    won: bool


@dataclass
class EpisodeRecord:
    game_file: str
    initial_observation: str
    steps: List[EpisodeStep] = field(default_factory=list)

    @property
    def won(self):
        return bool(self.steps and self.steps[-1].won)


@dataclass
class SyntheticTask:
    task_id: str
    source_game_file: str
    task_game_file: str
    source_scene_id: str
    instruction: str
    goal_kind: str
    goal_atoms: List[Dict[str, Any]]
    source_goal_text: str
    custom_goal_text: str
    witness_actions: List[str]
    source_split: str
    experiment_id: str = ""
    hypothesis_id: str = ""
    hypothesis: str = ""
    feedback_refs: List[str] = field(default_factory=list)
    diagnostic_target: Dict[str, Any] = field(default_factory=dict)
    generation_mode: str = "exploration"
    experiment_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchSample:
    group_id: str
    role: str
    prompt_ids: List[int]
    generated_ids: List[int]
    generated_text: str
    behavior_logprobs: List[float]
    reward: float
    task_metrics: Dict[str, Any]
    model_revision: str
    adapter_version: str
    sampling_parameters: Dict[str, Any]
    training_eligible: bool


def write_jsonl(path, records):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for record in records:
            value = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
