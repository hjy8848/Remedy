"""Designer-specified ALFWorld scenes and witnessed placement tasks."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .alfworld import AlfworldEpisode
from .model import seed_everything
from .records import SyntheticTask
from .scene_builder import build_scene, inspect_scene
from .synthesis import _generate_across_models, goal_reached, pddl_goal_expression, replay_witness, source_scene_id


def _parse_spec(text, feedback):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    spec = json.loads(text)
    for field in ("hypothesis", "diagnostic_behavior"):
        if not isinstance(spec[field], str) or not spec[field].strip():
            raise ValueError("{} must be nonempty text".format(field))
    refs = spec["feedback_refs"]
    if not isinstance(refs, list) or not all(isinstance(item, str) for item in refs):
        raise ValueError("feedback_refs must be a list of strings")
    available = set(feedback.get("reference_ids", []))
    if set(refs) - available:
        raise ValueError("feedback_refs contains references absent from the supplied feedback")
    if available and not set(refs).intersection(feedback.get("required_reference_ids", []) or available):
        raise ValueError("the experiment must cite a supplied diagnostic or research finding")
    return spec


def _spec_messages(game_file, parent, feedback):
    game = json.loads(Path(game_file).read_text(encoding="utf-8"))
    scene = inspect_scene(game["pddl_problem"])
    containments = sorted({(a[0].removesuffix("type"), a[1].removesuffix("type"))
                           for p, a in scene["facts"] if p == "cancontain" and len(a) == 2
                           and a[0].removesuffix("type") in scene["receptacles"]
                           and a[1].removesuffix("type") in scene["objects"]})
    return [
        {"role": "system", "content": (
            "Design one ALFWorld placement experiment to investigate the Solver feedback. "
            "Return a JSON object with hypothesis, diagnostic_behavior, feedback_refs, base_scene, "
            "objects, and goal. Use exactly the provided base_scene. objects is a list with one "
            "object: type, count (1 to 3), initial_receptacle. goal has type 'place_all', "
            "object_type, count, destination_type. Choose a pickupable object and available "
            "receptacle types that can contain it. If the types match, two receptacles are needed. Match object type and count in the "
            "goal. Cite at least one supplied feedback reference when available. Do not write PDDL.")},
        {"role": "user", "content": json.dumps({
            "base_scene": source_scene_id(game_file), "current_service": parent,
            "research_feedback": feedback, "pickupable_object_types": sorted(scene["objects"]),
            "receptacle_counts": {key: len(value) for key, value in scene["receptacles"].items()},
            "can_contain": [{"receptacle": r, "object": o} for r, o in containments],
        }, ensure_ascii=False)},
    ]


def construct_experiments(config, game_files, models, source_split, max_steps, task_dir,
                          parent_service, research_feedback, experiment_ids):
    seed_everything(config["run"]["seed"], models[0].torch)
    states = []
    with ThreadPoolExecutor(max_workers=max(1, len(models))) as executor:
        prompts = [_spec_messages(game, parent_service, research_feedback) for game in game_files]
        generations = _generate_across_models(
            models, executor, prompts, None, max(512, config["model"]["max_new_tokens"]),
            config["model"]["temperature"], config["model"]["top_p"])
        try:
            for game_file, experiment_id, generation in zip(game_files, experiment_ids, generations):
                state = {"tasks": [], "generations": [generation], "hypothesis": generation["generated_text"],
                         "hypothesis_record": None, "hypothesis_error": None, "hypothesis_id": None,
                         "research_feedback": research_feedback, "task_candidate_records": [],
                         "diagnostic_target": None, "actions": [], "trajectory": [],
                         "task_candidate_count": 0, "verified_task_count": 0,
                         "witness_replay_steps": 0, "experiment_spec": None, "active": False}
                states.append(state)
                try:
                    spec = _parse_spec(generation["generated_text"], research_feedback)
                    scene = build_scene(game_file, spec, task_dir)
                except (json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
                    state["hypothesis_error"] = str(error)
                    continue
                state.update({"scene": scene, "experiment_spec": spec, "hypothesis": spec["hypothesis"],
                              "hypothesis_record": spec, "hypothesis_id": experiment_id + "/hypothesis",
                              "diagnostic_target": {"object": spec["goal"]["object_type"],
                                                    "receptacle": spec["goal"]["destination_type"]},
                              "task_candidate_count": 1, "active": True})
                state["env"] = AlfworldEpisode(config["alfworld"]["config_path"],
                                                game_file=scene["game_file"], split=source_split)
                state["view"] = state["env"].reset()
                if goal_reached(scene["goal_atoms"], state["view"]["facts"]):
                    raise ValueError("constructed task is already satisfied at environment reset")

            for _ in range(max_steps):
                active = [state for state in states if state["active"]]
                if not active:
                    break
                prompts, choices = [], []
                for state in active:
                    view = state["view"]
                    commands = view["admissible_commands"]
                    observation = re.sub(r"your task is to[^\n]*", "Your task is to: " + state["scene"]["instruction"],
                                         view["observation"], flags=re.IGNORECASE)
                    prompts.append([
                        {"role": "system", "content": "Complete the task in ALFWorld. Copy one allowed command exactly and output only that command."},
                        {"role": "user", "content": json.dumps({
                            "task": state["scene"]["instruction"], "observation": observation,
                            "recent_actions": state["trajectory"][-5:], "allowed_commands": commands,
                        }, ensure_ascii=False)},
                    ])
                    choices.append(commands)
                outputs = _generate_across_models(models, executor, prompts, choices, 32,
                                                  config["model"]["temperature"], config["model"]["top_p"])
                for state, generation in zip(active, outputs):
                    state["generations"].append(generation)
                    action = generation["generated_text"].strip()
                    if action not in state["view"]["admissible_commands"]:
                        raise ValueError("constrained proposer returned an invalid action: {!r}".format(action))
                    result = state["env"].step(action)
                    state["actions"].append(action)
                    state["trajectory"].append({"action": action, "observation": result["observation"]})
                    state["view"] = result
                    if result["won"] and goal_reached(state["scene"]["goal_atoms"], result["facts"]):
                        state["success"] = True
                        state["active"] = False
                    elif result["done"]:
                        state["active"] = False
        finally:
            for state in states:
                if "env" in state:
                    state["env"].close()

    for state in states:
        if not state.get("scene"):
            continue
        scene, spec = state["scene"], state["experiment_spec"]
        record = {"candidate_id": state["hypothesis_id"] + "/task", "goal_atoms": scene["goal_atoms"],
                  "hypothesis_id": state["hypothesis_id"], "feedback_refs": spec["feedback_refs"],
                  "task_id": scene["task_id"], "rejection_reason": None}
        state["task_candidate_records"].append(record)
        if not state.get("success"):
            record["rejection_reason"] = "proposer_did_not_complete_goal"
            continue
        task = SyntheticTask(
            task_id=scene["task_id"], source_game_file=scene["source_game_file"],
            task_game_file=scene["game_file"], source_scene_id=scene["source_scene_id"],
            instruction=scene["instruction"], goal_kind="place_all", goal_atoms=scene["goal_atoms"],
            source_goal_text=scene["source_goal_text"], custom_goal_text=pddl_goal_expression(scene["goal_atoms"]),
            witness_actions=state["actions"], source_split=source_split,
            experiment_id=state["hypothesis_id"].split("/", 1)[0], hypothesis_id=state["hypothesis_id"],
            hypothesis=spec["hypothesis"], feedback_refs=spec["feedback_refs"],
            diagnostic_target=state["diagnostic_target"], generation_mode="constructed", experiment_spec=spec)
        replay = replay_witness(config, task)
        state["witness_replay_steps"] = len(replay["actions"])
        record["witness_replay"] = replay
        if replay["valid"]:
            state["tasks"].append(task)
            state["verified_task_count"] = 1
        else:
            record["rejection_reason"] = "witness_replay_failed"
    return states
