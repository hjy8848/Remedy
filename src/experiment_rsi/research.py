import json
import copy
import difflib
import hashlib
import random
import re
from dataclasses import asdict
from pathlib import Path

from .alfworld import list_game_files
from .evaluation import (
    choose_candidate,
    evaluate_tasks,
    evidence_removal_reward,
    mean_score,
    mean_task_reward,
    run_episode,
)
from .model import ModelSession, seed_everything
from .policy import load_service
from .records import ResearchSample
from .synthesis import (
    explore_games,
    read_tasks,
    source_scene_id,
    supported_placement_game,
    synthesize,
    write_tasks,
)


def candidate_messages(parent, diagnostic_report):
    return [
        {
            "role": "system",
            "content": (
                "You improve an ALFWorld agent's execution policy. Use the supplied evidence to "
                "return one complete JSON object with exactly two keys: policy_source (Python "
                "source defining step(view, state)) and knowledge (array of short strings). "
                "The step function must return either {kind: action, action: text, state: object} "
                "or {kind: model, messages: chat messages, state: object}. state must be a JSON object. "
                "For an action directive, "
                "action must exactly match one string in view['admissible_commands']; never return "
                "a number or index. For a model directive that requests a numbered choice, include "
                "the non-empty ordered command strings in state['action_choices']; use a fallback "
                "such as task_specific_choices or view['admissible_commands'] when filtering finds no "
                "commands. Never "
                "ask the model to choose from an empty list or resolve its number against a different "
                "list. view has objective, "
                "observation, history, admissible_commands, knowledge, and remaining_steps. "
                "The worker has only the Python standard library and no network. Do not include markdown."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"current_service": parent, "diagnostic_report": diagnostic_report},
                ensure_ascii=False,
            ),
        },
    ]


def branch_messages(parent, experiment, diagnostic_report):
    context = {
        "current_service": parent,
        "experiment_hypothesis": experiment["hypothesis"],
        "task_bundle": [
            {
                "task_id": task.task_id,
                "instruction": task.instruction,
            }
            for task in experiment["tasks"]
        ],
        "diagnostic_report": diagnostic_report,
    }
    return [
        {
            "role": "system",
            "content": (
                "Improve the ALFWorld execution policy using this experiment context. "
                "Return one complete JSON object with exactly two keys: policy_source (Python "
                "source defining step(view, state)) and knowledge (array of short strings). "
                "The step function must return either {kind: action, action: text, state: object} "
                "or {kind: model, messages: chat messages, state: object}. state must be a JSON object. "
                "For an action directive, "
                "action must exactly match one string in view['admissible_commands']; never return "
                "a number or index. For a model directive that requests a numbered choice, include "
                "the non-empty ordered command strings in state['action_choices']; use a fallback "
                "such as task_specific_choices or view['admissible_commands'] when filtering finds no "
                "commands. Never "
                "ask the model to choose from an empty list or resolve its number against a different "
                "list. view has objective, observation, history, admissible_commands, "
                "knowledge, and remaining_steps. The worker has only the Python standard library "
                "and no network. Do not include markdown."
            ),
        },
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]


def parse_candidate(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[len("```json"): -len("```")].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("candidate must be a JSON object")
    if set(value) != {"policy_source", "knowledge"}:
        raise ValueError("candidate JSON must contain exactly policy_source and knowledge")
    if not isinstance(value["policy_source"], str) or not isinstance(value["knowledge"], list):
        raise ValueError("candidate policy_source must be text and knowledge must be a list")
    if not all(isinstance(item, str) for item in value["knowledge"]):
        raise ValueError("every knowledge entry must be text")
    return value


def score_report(tasks, results):
    by_id = {result["task_id"]: result for result in results}
    report = []
    for task in tasks:
        result = by_id[task.task_id]
        report.append(
            {
                "task_id": task.task_id,
                "objective": task.instruction,
                "goal_kind": task.goal_kind,
                "success": result["success"],
                "candidate_error": result["candidate_error"],
                "completed_goal_atoms": result["completed_goal_atoms"],
                "goal_atom_count": result["goal_atom_count"],
                "goal_progress": result["task_reward"],
                "task_reward": result["task_reward"],
                "steps": result["steps"],
                "trajectory": [
                    {"action": step["action"], "observation": step["observation"]}
                    for step in result["trace"]
                ],
            }
        )
    return report


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


MAX_FEEDBACK_HISTORY = 2
MAX_FEEDBACK_FAILURES = 4
MAX_FEEDBACK_EFFECTS = 3
MAX_FEEDBACK_EXPERIMENTS = 3
FEEDBACK_TEXT_CHARS = 180


def service_version(service):
    payload = json.dumps(service, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _short(value, limit=FEEDBACK_TEXT_CHARS):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _diagnostic_case(result, index, diagnostic_id):
    trace = result["trace"]
    task_line = re.search(
        r"Your task is to:[^\n]*", result.get("initial_observation", ""), re.IGNORECASE
    )
    seen_actions = set()
    repeated = 0
    for step in trace:
        action = step["action"]
        repeated += int(action in seen_actions and step.get("stalled", False))
        seen_actions.add(action)
    stalled = sum(bool(step.get("stalled")) for step in trace)
    navigation_stalls = sum(
        step["action"].lower().startswith("go to ") and step.get("stalled", False)
        for step in trace
    )
    categories = []
    error = result.get("candidate_error")
    if error:
        categories.append("invalid_action" if "not an admissible" in error else "policy_error")
    if repeated:
        categories.append("repeated_action")
    if navigation_stalls:
        categories.append("stalled_navigation")
    if not result["success"]:
        categories.append("task_incomplete")
    tail = [
        {
            "action": _short(step["action"], 80),
            "observation": _short(step["observation"], 120),
            "goal_progress": step.get("goal_progress"),
            "stalled": bool(step.get("stalled")),
        }
        for step in trace[-3:]
    ]
    return {
        "ref": "{}/episode-{:03d}".format(diagnostic_id, index),
        "scene_id": result["source_scene_id"],
        "objective": _short(
            result.get("objective") or (task_line.group(0) if task_line else result["initial_observation"]),
            140,
        ),
        "success": bool(result["success"]),
        "goal_progress": result.get("goal_progress"),
        "steps": result["steps"],
        "categories": categories,
        "candidate_error": _short(error, 120),
        "repeated_action_count": repeated,
        "stalled_navigation_count": navigation_stalls,
        "trace_tail": tail,
    }


def summarize_diagnostic_results(results, diagnostic_id, service_path):
    cases = [_diagnostic_case(result, index, diagnostic_id) for index, result in enumerate(results)]
    failures = [case for case in cases if not case["success"]]
    category_counts = {}
    for case in cases:
        for category in case["categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1
    reference_cases = (failures or cases)[:MAX_FEEDBACK_FAILURES]
    reference_ids = [case["ref"] for case in reference_cases]
    if not reference_ids:
        reference_ids = [diagnostic_id + "/aggregate"]
    return {
        "source": "ALFWorld train cold-start diagnosis",
        "diagnostic_id": diagnostic_id,
        "service_path": str(service_path),
        "service_version": service_version(load_service(service_path)),
        "task_count": len(cases),
        "success_count": sum(case["success"] for case in cases),
        "failure_count": len(failures),
        "mean_goal_progress": (
            sum(case["goal_progress"] or 0.0 for case in cases) / len(cases) if cases else None
        ),
        "category_counts": category_counts,
        "cases": (failures or cases)[:MAX_FEEDBACK_FAILURES],
        "reference_ids": reference_ids,
    }


def run_solver_diagnostics(config, service_path, game_files, out_dir, diagnostic_id):
    diagnostic_config = copy.deepcopy(config)
    diagnostic_config["alfworld"]["split"] = config["research"].get("diagnostic_split", "train")
    model = ModelSession(diagnostic_config)
    service = load_service(service_path)
    results = []
    for index, game_file in enumerate(game_files):
        seed = config["run"]["seed"] + 50_000 + index
        seed_everything(seed, model.torch)
        result = run_episode(
            diagnostic_config,
            service,
            model,
            game_file,
            max_steps=config["run"]["max_steps"],
            seed=seed,
        )
        result.update({
            "episode_id": "{}-episode-{:03d}".format(diagnostic_id, index),
            "source_scene_id": source_scene_id(game_file),
            "initial_observation": result["initial_observation"],
            "service_version": service_version(service),
        })
        results.append(result)
    del model
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "diagnostic_episodes.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = summarize_diagnostic_results(results, diagnostic_id, service_path)
    write_json(out_dir / "diagnostic_summary.json", summary)
    return summary


def summarize_research_batch(batch_dir, label=None):
    batch_dir = Path(batch_dir)
    experiments = json.loads((batch_dir / "experiments.json").read_text(encoding="utf-8"))
    scores = read_jsonl(batch_dir / "scores.jsonl")
    episodes = read_jsonl(batch_dir / "episodes.jsonl")
    parent = json.loads((batch_dir / "parent.json").read_text(encoding="utf-8"))
    accepted = json.loads((batch_dir / "service.json").read_text(encoding="utf-8"))
    metrics = json.loads((batch_dir / "metrics.json").read_text(encoding="utf-8"))
    parent_version = service_version(parent)
    accepted_version = service_version(accepted)
    source_id = label or batch_dir.name
    accepted_candidate = metrics.get("accepted_candidate")

    candidate_rows = {}
    regressions = []
    for row in scores:
        candidate_id = row["candidate_id"]
        if candidate_id == "parent":
            continue
        item = candidate_rows.setdefault(candidate_id, {
            "branch": row["branch"],
            "sets": {},
            "service_version": row.get("candidate_service_version"),
        })
        stats = item["sets"].setdefault(row["set"], {"deltas": [], "regressions": 0})
        stats["deltas"].append(row["candidate_task_reward"] - row["parent_task_reward"])
        if row["parent_success"] and not row["candidate_success"]:
            stats["regressions"] += 1
            regressions.append({
                "ref": "{}/regression/{}/{}".format(source_id, candidate_id, row["task_id"]),
                "candidate_id": candidate_id,
                "task_id": row["task_id"],
                "set": row["set"],
                "branch": row["branch"],
            })

    effects = []
    candidates_dir = batch_dir / "candidates"
    for candidate_id, value in candidate_rows.items():
        parent_reward = value["sets"].get("reward", {}).get("deltas", [])
        reward_delta = sum(parent_reward) / len(parent_reward) if parent_reward else None
        selection = value["sets"].get("selection", {}).get("deltas", [])
        selection_delta = sum(selection) / len(selection) if selection else None
        candidate_path = candidates_dir / (candidate_id + ".json")
        change_summary = (
            candidate_change_summary(parent, json.loads(candidate_path.read_text(encoding="utf-8")))
            if candidate_path.exists() else None
        )
        effects.append({
            "ref": "{}/candidate/{}".format(source_id, candidate_id),
            "candidate_id": candidate_id,
            "branch": value["branch"],
            "service_version": value["service_version"],
            "reward_delta": reward_delta,
            "selection_delta": selection_delta,
            "reward_regressions": value["sets"].get("reward", {}).get("regressions", 0),
            "accepted": candidate_id == accepted_candidate,
            "change_summary": change_summary,
        })
    effects.sort(key=lambda item: item["reward_delta"] if item["reward_delta"] is not None else -1.0,
                 reverse=True)
    selected_effects = []
    for item in ([effects[0]] if effects else []) + ([effects[-1]] if effects else []):
        if all(existing["candidate_id"] != item["candidate_id"] for existing in selected_effects):
            selected_effects.append(item)
    accepted_effect = next((item for item in effects if item["accepted"]), None)
    if accepted_effect and all(item["candidate_id"] != accepted_effect["candidate_id"] for item in selected_effects):
        selected_effects.append(accepted_effect)

    unresolved = []
    for episode in episodes:
        is_accepted = (
            (accepted_candidate and episode.get("candidate_id") == accepted_candidate)
            or (not accepted_candidate and episode.get("candidate_id") == "parent")
        )
        if episode.get("success") or not is_accepted:
            continue
        trace = episode.get("trace", [])
        seen_actions = set()
        repeated_actions = 0
        stalled_navigation = 0
        for step in trace:
            action = step.get("action", "")
            repeated_actions += int(action in seen_actions and step.get("stalled", False))
            stalled_navigation += int(
                action.lower().startswith("go to ") and step.get("stalled", False)
            )
            seen_actions.add(action)
        issues = []
        if repeated_actions:
            issues.append("repeated_action")
        if stalled_navigation:
            issues.append("stalled_navigation")
        if episode.get("candidate_error"):
            issues.append("policy_error")
        unresolved.append({
            "ref": "{}/failure/{}/{}".format(
                source_id, accepted_candidate or "parent", episode.get("task_id") or "unknown"
            ),
            "task_id": episode.get("task_id"),
            "objective": _short(episode.get("objective"), 120),
            "failure": _short(episode.get("candidate_error") or "task not completed", 90),
            "issues": issues,
            "goal_progress": episode.get("goal_progress"),
            "trace_tail": [
                {
                    "action": _short(step.get("action"), 65),
                    "stalled": bool(step.get("stalled", False)),
                    "goal_progress": step.get("goal_progress"),
                }
                for step in trace[-2:]
            ],
        })
    if not unresolved:
        unresolved = [
            {
                "ref": "{}/failure/{}/{}".format(
                    source_id, accepted_candidate or "parent", row.get("task_id") or "unknown"
                ),
                "task_id": row.get("task_id"),
                "objective": "",
                "failure": "accepted Service regressed on selection/reward score",
                "trace_tail": [],
            }
            for row in regressions[:MAX_FEEDBACK_FAILURES]
            if row["candidate_id"] == accepted_candidate
        ]

    prior_experiments = [
        {
            "ref": "{}/{}".format(source_id, item["experiment_id"]),
            "experiment_id": item["experiment_id"],
            "hypothesis": _short(item.get("hypothesis"), 140),
            "utility": item.get("experiment_utility"),
            "with_evidence_gain": item.get("with_evidence_gain"),
            "without_evidence_gain": item.get("without_evidence_gain"),
            "task_ids": item.get("task_ids", [])[:3],
        }
        for item in experiments
    ]
    return {
        "source": source_id,
        "record_files": [
            "experiments.json", "scores.jsonl", "episodes.jsonl",
            "parent.json", "service.json", "candidates/<candidate_id>.json",
        ],
        "service_versions": {
            "parent": parent_version,
            "accepted": accepted_version,
            "accepted_candidate": accepted_candidate,
        },
        "unresolved_failures": unresolved[:MAX_FEEDBACK_FAILURES],
        "candidate_effects": selected_effects[:MAX_FEEDBACK_EFFECTS],
        "regressions": regressions[:MAX_FEEDBACK_FAILURES],
        "prior_experiment_values": prior_experiments[:MAX_FEEDBACK_EXPERIMENTS],
    }


def make_research_feedback(diagnostic, history_dirs):
    history = [
        summarize_research_batch(path, label)
        for path, label in history_dirs[-MAX_FEEDBACK_HISTORY:]
    ]
    reference_ids = list(diagnostic.get("reference_ids", []))
    required_refs = list(reference_ids)
    for summary in history:
        for key in ("unresolved_failures", "candidate_effects", "regressions", "prior_experiment_values"):
            refs = [item["ref"] for item in summary[key]]
            reference_ids.extend(refs)
            if key in ("unresolved_failures", "regressions"):
                required_refs.extend(refs)
    if not required_refs:
        required_refs = list(reference_ids)
    return {
        "current_solver_diagnostics": {
            key: diagnostic[key]
            for key in (
                "diagnostic_id", "service_version", "task_count", "success_count",
                "failure_count", "mean_goal_progress", "category_counts", "cases",
            )
        },
        "prior_research": history,
        "reference_ids": list(dict.fromkeys(reference_ids)),
        "required_reference_ids": list(dict.fromkeys(required_refs)),
    }


def swanlab_research_record(feedback, experiments_path, swanlab_module):
    experiments = json.loads(Path(experiments_path).read_text(encoding="utf-8"))
    detail = {
        "feedback_used": feedback,
        "experiments": [
            {
                key: item.get(key)
                for key in (
                    "experiment_id", "hypothesis_id", "hypothesis", "hypothesis_record",
                    "service_version", "generation_mode", "experiment_spec",
                    "research_feedback", "diagnostic_target", "task_ids", "task_candidates",
                    "experiment_utility", "with_evidence_gain", "without_evidence_gain",
                    "candidate_results",
                )
            }
            for item in experiments
        ],
    }
    return swanlab_module.Text(json.dumps(detail, ensure_ascii=False, indent=2))


def candidate_change_summary(parent, candidate):
    parent_knowledge = set(parent["knowledge"])
    candidate_knowledge = set(candidate["knowledge"])
    diff = list(difflib.unified_diff(
        parent["policy_source"].splitlines(),
        candidate["policy_source"].splitlines(),
        fromfile="parent",
        tofile="candidate",
        lineterm="",
    ))
    return {
        "knowledge_added": [_short(item, 100) for item in sorted(candidate_knowledge - parent_knowledge)[:3]],
        "knowledge_removed": [_short(item, 100) for item in sorted(parent_knowledge - candidate_knowledge)[:3]],
        "policy_source_changed": parent["policy_source"] != candidate["policy_source"],
        "policy_source_diff": _short("\n".join(diff[2:]), 280) if diff else "",
    }


def post_execution_feedback(results, parse_error=None):
    return {
        "parse_error": parse_error,
        "task_outcomes": [
            {
                "task_id": result["task_id"],
                "objective": result["objective"],
                "success": result["success"],
                "candidate_error": result["candidate_error"],
                "completed_goal_atoms": result["completed_goal_atoms"],
                "goal_atom_count": result["goal_atom_count"],
                "goal_progress": result["task_reward"],
                "task_reward": result["task_reward"],
                "trajectory": [
                    {"action": step["action"], "observation": step["observation"]}
                    for step in result["trace"]
                ],
            }
            for result in results
        ],
    }


def synthesize_run(config, out_dir, count, adapter_path=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ModelSession(config, adapter_path=adapter_path)
    tasks = synthesize(config, model, count, out_dir / "task_games")
    write_tasks(out_dir / "tasks.jsonl", tasks)
    write_json(out_dir / "summary.json", {"requested": count, "verified_tasks": len(tasks)})
    return tasks


def choose_task_files(config, split, count, excluded_scenes=(), available_files=None):
    files = [
        path
        for path in (
            available_files
            if available_files is not None
            else list_game_files(config["alfworld"]["config_path"], split)
        )
        if supported_placement_game(path)
    ]
    random.Random(config["run"]["seed"]).shuffle(files)
    excluded = set(excluded_scenes)
    selected = []
    scenes = set()
    for path in files:
        scene = source_scene_id(path)
        if scene in excluded or scene in scenes:
            continue
        scenes.add(scene)
        selected.append(path)
        if len(selected) == count:
            break
    return selected


def choose_diagnostic_batches(config, periods, excluded_scenes):
    diagnostic_split = config["research"].get("diagnostic_split", "train")
    diagnostic_count = config["research"].get("diagnostic_games_per_period", 10)
    train_files = list_game_files(config["alfworld"]["config_path"], diagnostic_split)
    reserved = set(excluded_scenes)
    batches = []
    for period in range(periods):
        period_config = copy.deepcopy(config)
        period_config["run"]["seed"] = config["run"]["seed"] + 1000 * period
        files = choose_task_files(
            period_config,
            diagnostic_split,
            diagnostic_count,
            excluded_scenes=reserved,
            available_files=train_files,
        )
        if len(files) != diagnostic_count:
            raise ValueError(
                "requested {} distinct ALFWorld {} diagnostic scenes for period {}, found {}".format(
                    diagnostic_count, diagnostic_split, period + 1, len(files)
                )
            )
        batches.append(files)
        reserved.update(source_scene_id(path) for path in files)
    return batches, reserved - set(excluded_scenes)


def assert_disjoint_scenes(experiments, selection_tasks, reward_tasks):
    experiment_scenes = set()
    for item in experiments:
        if "game_file" in item:
            experiment_scenes.add(source_scene_id(item["game_file"]))
        experiment_scenes.update(task.source_scene_id for task in item["tasks"])
    selection_scenes = {task.source_scene_id for task in selection_tasks}
    reward_scenes = {task.source_scene_id for task in reward_tasks}
    if experiment_scenes & selection_scenes:
        raise ValueError("diagnostic and selection tasks share ALFRED source scenes")
    if experiment_scenes & reward_scenes:
        raise ValueError("diagnostic and reward tasks share ALFRED source scenes")
    if selection_scenes & reward_scenes:
        raise ValueError("selection and reward tasks share ALFRED source scenes")


def experiment_group(config, selection_path, reward_path, out_dir, parent_path,
                     experiment_adapter=None, edit_adapter=None, excluded_scenes=(),
                     experiment_game_files=None, research_feedback=None):
    selection_tasks = read_tasks(selection_path)
    reward_tasks = read_tasks(reward_path)
    parent = load_service(parent_path)
    research_feedback = research_feedback or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "research_feedback.json", research_feedback)

    experiment_split = config["research"]["experiment_split"]
    experiment_count = config["research"]["experiments_per_group"]
    edits_per_branch = config["research"]["edits_per_branch"]
    if experiment_count < 2:
        raise ValueError("experiments_per_group must be at least 2 for GRPO")
    if edits_per_branch < 2:
        raise ValueError("edits_per_branch must be at least 2 for GRPO")
    game_files = choose_task_files(
        config,
        experiment_split,
        experiment_count,
        excluded_scenes,
        available_files=experiment_game_files,
    )
    if len(game_files) != experiment_count:
        raise ValueError(
            "requested {} distinct experiment scenes, found {}".format(experiment_count, len(game_files))
        )

    sampling_gpu_count = config["research"].get("exploration_gpus_per_node", 1)
    if sampling_gpu_count > 1:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < sampling_gpu_count:
            raise RuntimeError(
                "exploration requests {} visible GPUs, but only {} are available".format(
                    sampling_gpu_count, torch.cuda.device_count()
                )
            )
        devices = ["cuda:{}".format(index) for index in range(sampling_gpu_count)]
    else:
        devices = [None]
    experiment_models = [
        ModelSession(config, adapter_path=experiment_adapter, device=device)
        for device in devices
    ]
    generation_mode = config["research"].get("generation_mode", "exploration")
    if generation_mode == "constructed":
        from .constructed import construct_experiments
        generate_experiments = construct_experiments
    elif generation_mode == "exploration":
        generate_experiments = explore_games
    else:
        raise ValueError("unknown experiment generation mode: {}".format(generation_mode))
    explored_rows = generate_experiments(
        config,
        game_files,
        experiment_models,
        experiment_split,
        config["run"]["max_steps"],
        out_dir / "diagnostic_games",
        parent_service=parent,
        research_feedback=research_feedback,
        experiment_ids=["experiment-{}".format(index) for index in range(len(game_files))],
    )
    experiments = []
    for index, (game_file, explored) in enumerate(zip(game_files, explored_rows)):
        experiments.append(
            {
                "experiment_id": "experiment-{}".format(index),
                "hypothesis": explored["hypothesis"],
                "hypothesis_id": explored.get("hypothesis_id"),
                "hypothesis_record": explored.get("hypothesis_record"),
                "hypothesis_error": explored.get("hypothesis_error"),
                "generation_mode": generation_mode,
                "experiment_spec": explored.get("experiment_spec"),
                "research_feedback": explored["research_feedback"],
                "diagnostic_target": explored.get("diagnostic_target"),
                "task_candidates": explored["task_candidate_records"],
                "game_file": game_file,
                "trajectory": explored["trajectory"],
                "tasks": explored["tasks"],
                "generations": explored["generations"],
                "exploration_steps": len(explored["actions"]),
                "exploration_model_calls": len(explored["generations"]),
                "task_candidate_count": explored["task_candidate_count"],
                "verified_task_count": explored["verified_task_count"],
                "witness_replay_steps": explored["witness_replay_steps"],
            }
        )
    del experiment_models

    assert_disjoint_scenes(experiments, selection_tasks, reward_tasks)
    service_model = ModelSession(config)
    parent_version = service_version(parent)
    for experiment in experiments:
        experiment["diagnostic_results"] = evaluate_tasks(
            config, parent, service_model, experiment["tasks"]
        ) if experiment["tasks"] else []
        for result in experiment["diagnostic_results"]:
            result.update({
                "candidate_id": "parent",
                "experiment_id": experiment["experiment_id"],
                "score_set": "experiment_diagnostic",
                "service_version": parent_version,
            })
        experiment["diagnostic_report"] = score_report(
            experiment["tasks"], experiment["diagnostic_results"]
        )
    parent_selection = evaluate_tasks(config, parent, service_model, selection_tasks)
    parent_reward = evaluate_tasks(config, parent, service_model, reward_tasks)
    for result in parent_selection:
        result.update({"candidate_id": "parent", "score_set": "selection", "service_version": parent_version})
    for result in parent_reward:
        result.update({"candidate_id": "parent", "score_set": "reward", "service_version": parent_version})
    parent_selection_score = mean_score(parent_selection)
    parent_reward_score = mean_score(parent_reward)
    parent_reward_task_score = mean_task_reward(parent_reward)
    del service_model

    edit_model = ModelSession(config, adapter_path=edit_adapter)
    edit_candidates = []
    for experiment in experiments:
        for branch, report in (
            ("with_evidence", experiment["diagnostic_report"]),
            ("without_evidence", None),
        ):
            prompt = branch_messages(parent, experiment, report)
            for index in range(edits_per_branch):
                experiment_index = int(experiment["experiment_id"].rsplit("-", 1)[1])
                seed_everything(
                    config["run"]["seed"] + 1000 + experiment_index * edits_per_branch + index,
                    edit_model.torch,
                )
                generation = edit_model.generate(
                    prompt,
                    max_new_tokens=config["model"].get(
                        "editor_max_new_tokens", config["model"]["max_new_tokens"]
                    ),
                    temperature=config["model"]["temperature"],
                    top_p=config["model"]["top_p"],
                )
                try:
                    candidate = parse_candidate(generation["generated_text"])
                    error = None
                except (json.JSONDecodeError, ValueError) as candidate_error:
                    candidate = None
                    error = str(candidate_error)
                edit_candidates.append(
                    {
                        "candidate_id": "{}-{}-{}".format(experiment["experiment_id"], branch, index),
                        "experiment_id": experiment["experiment_id"],
                        "branch": branch,
                        "prompt_messages": prompt,
                        "generation": generation,
                        "candidate": candidate,
                        "parse_error": error,
                    }
                )
    del edit_model

    service_model = ModelSession(config)
    episode_records = []
    score_rows = []
    visible_gains = {item["experiment_id"]: [] for item in experiments}
    masked_gains = {item["experiment_id"]: [] for item in experiments}
    visible_selection = []
    research_samples = []
    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    for item in edit_candidates:
        candidate = item["candidate"]
        candidate_id = item["candidate_id"]
        if candidate is None:
            selection_results = []
            reward_results = []
            selection_score = 0.0
            reward_score = 0.0
            reward_task_score = 0.0
            candidate_executable = False
        else:
            write_json(candidates_dir / (candidate_id + ".json"), candidate)
            selection_results = evaluate_tasks(config, candidate, service_model, selection_tasks)
            reward_results = evaluate_tasks(config, candidate, service_model, reward_tasks)
            selection_score = mean_score(selection_results)
            reward_score = mean_score(reward_results)
            reward_task_score = mean_task_reward(reward_results)
            candidate_executable = not any(
                result["candidate_error"] for result in selection_results + reward_results
            )
        item["candidate_executable"] = candidate_executable
        reward_delta = reward_task_score - parent_reward_task_score
        item["candidate_version"] = service_version(candidate) if candidate is not None else None
        item["selection_score"] = selection_score
        item["reward_score"] = reward_score
        item["reward_task_score"] = reward_task_score
        item["reward_delta"] = reward_delta
        item["selection_delta"] = selection_score - parent_selection_score
        if candidate is not None:
            item["change_summary"] = candidate_change_summary(parent, candidate)
        gains = visible_gains if item["branch"] == "with_evidence" else masked_gains
        gains[item["experiment_id"]].append(reward_delta)
        if item["branch"] == "with_evidence":
            visible_selection.append(
                (selection_score, reward_score, candidate_id, candidate, candidate_executable)
            )
        for set_name, tasks, results, baseline in (
            ("selection", selection_tasks, selection_results, parent_selection),
            ("reward", reward_tasks, reward_results, parent_reward),
        ):
            for result in results:
                result.update({
                    "candidate_id": candidate_id,
                    "experiment_id": item["experiment_id"],
                    "branch": item["branch"],
                    "score_set": set_name,
                    "service_version": item["candidate_version"],
                })
            for index, task in enumerate(tasks):
                score_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "experiment_id": item["experiment_id"],
                        "branch": item["branch"],
                        "set": set_name,
                        "task_id": task.task_id,
                        "parent_success": baseline[index]["success"],
                        "candidate_success": results[index]["success"] if results else False,
                        "parent_task_reward": baseline[index]["task_reward"],
                        "candidate_task_reward": results[index]["task_reward"] if results else 0.0,
                        "parent_service_version": parent_version,
                        "candidate_service_version": item["candidate_version"],
                    }
                )
        episode_records.extend(selection_results)
        episode_records.extend(reward_results)
        generation = item["generation"]
        research_samples.append(asdict(ResearchSample(
            group_id="{}-{}".format(item["experiment_id"], item["branch"]),
            role="edit",
            prompt_ids=generation["prompt_ids"],
            generated_ids=generation["generated_ids"],
            generated_text=generation["generated_text"],
            behavior_logprobs=generation["behavior_logprobs"],
            reward=reward_delta,
            task_metrics={
                "candidate_id": candidate_id,
                "trajectory_id": candidate_id,
                "sample_weight": 1.0,
                "prompt_messages": item["prompt_messages"],
                "post_feedback": post_execution_feedback(reward_results, item["parse_error"]),
                "branch": item["branch"],
                "valid_candidate": candidate is not None,
                "parse_error": item["parse_error"],
                "selection_score": selection_score,
                "reward_score": reward_score,
                "reward_task_score": reward_task_score,
                "candidate_executable": candidate_executable,
            },
            model_revision=generation["model_revision"],
            adapter_version=generation["adapter_version"],
            sampling_parameters=generation["sampling_parameters"],
            training_eligible=item["branch"] == "with_evidence",
        )))

    experiment_metrics = []
    experiment_samples = []
    for experiment in experiments:
        experiment_id = experiment["experiment_id"]
        if experiment["tasks"]:
            utility = evidence_removal_reward(
                visible_gains[experiment_id], masked_gains[experiment_id]
            )
        else:
            utility = -1.0
        calls = experiment["generations"]
        for index, generation in enumerate(calls):
            experiment_samples.append(asdict(ResearchSample(
                group_id="experiment-group-0",
                role="experiment",
                prompt_ids=generation["prompt_ids"],
                generated_ids=generation["generated_ids"],
                generated_text=generation["generated_text"],
                behavior_logprobs=generation["behavior_logprobs"],
                reward=utility,
                task_metrics={
                    "experiment_id": experiment_id,
                    "trajectory_id": experiment_id,
                    "sample_weight": 1.0 / len(calls),
                    "call_index": index,
                    "research_feedback": experiment["research_feedback"],
                    "hypothesis": experiment["hypothesis"],
                    "hypothesis_record": experiment.get("hypothesis_record"),
                    "generation_mode": experiment["generation_mode"],
                    "experiment_spec": experiment["experiment_spec"],
                    "task_ids": [task.task_id for task in experiment["tasks"]],
                },
                model_revision=generation["model_revision"],
                adapter_version=generation["adapter_version"],
                sampling_parameters=generation["sampling_parameters"],
                training_eligible=True,
            )))
        experiment_metrics.append({
            "experiment_id": experiment_id,
            "service_version": parent_version,
            "hypothesis_id": experiment.get("hypothesis_id"),
            "hypothesis": experiment["hypothesis"],
            "hypothesis_record": experiment.get("hypothesis_record"),
            "hypothesis_error": experiment.get("hypothesis_error"),
            "generation_mode": experiment["generation_mode"],
            "experiment_spec": experiment["experiment_spec"],
            "research_feedback": experiment["research_feedback"],
            "diagnostic_target": experiment.get("diagnostic_target"),
            "task_candidates": experiment["task_candidates"],
            "game_file": experiment["game_file"],
            "trajectory": experiment["trajectory"],
            "task_ids": [task.task_id for task in experiment["tasks"]],
            "diagnostic_report": experiment["diagnostic_report"],
            "valid": bool(experiment["tasks"]),
            "with_evidence_gain": sum(visible_gains[experiment_id]) / len(visible_gains[experiment_id]) if visible_gains[experiment_id] else None,
            "without_evidence_gain": sum(masked_gains[experiment_id]) / len(masked_gains[experiment_id]) if masked_gains[experiment_id] else None,
            "experiment_utility": utility,
            "candidate_results": [
                {
                    "candidate_id": item["candidate_id"],
                    "branch": item["branch"],
                    "valid_candidate": item["candidate"] is not None,
                    "candidate_executable": item["candidate_executable"],
                    "selection_delta": item["selection_delta"],
                    "reward_delta": item["reward_delta"],
                    "candidate_service_version": item["candidate_version"],
                    "change_summary": item.get("change_summary"),
                }
                for item in edit_candidates if item["experiment_id"] == experiment_id
            ],
        })
        episode_records.extend(experiment["diagnostic_results"])

    accepted = parent
    selected_candidate_id = None
    accepted_selection_score = parent_selection_score
    accepted_reward_score = parent_reward_score
    eligible = [item for item in visible_selection if item[3] is not None and item[4]]
    if eligible:
        selected_index = choose_candidate(parent_selection_score, [item[0] for item in eligible])
        if selected_index is not None:
            accepted_selection_score, accepted_reward_score, candidate_id, candidate, _ = eligible[selected_index]
            accepted = candidate
            selected_candidate_id = candidate_id

    for experiment_record in experiment_metrics:
        for candidate_record in experiment_record["candidate_results"]:
            candidate_record["accepted"] = candidate_record["candidate_id"] == selected_candidate_id

    write_tasks(out_dir / "tasks.jsonl", [task for experiment in experiments for task in experiment["tasks"]])
    write_json(out_dir / "parent.json", parent)
    write_json(out_dir / "service.json", accepted)
    write_json(out_dir / "experiments.json", experiment_metrics)
    experiment_utility_values = [item["experiment_utility"] for item in experiment_metrics]
    with_evidence_gains = [
        item["with_evidence_gain"] for item in experiment_metrics
        if item["with_evidence_gain"] is not None
    ]
    without_evidence_gains = [
        item["without_evidence_gain"] for item in experiment_metrics
        if item["without_evidence_gain"] is not None
    ]
    task_candidate_count = sum(item["task_candidate_count"] for item in experiments)
    verified_task_count = sum(item["verified_task_count"] for item in experiments)
    candidate_count = len(edit_candidates)
    parse_valid_count = sum(item["candidate"] is not None for item in edit_candidates)
    executable_count = sum(
        item["candidate"] is not None and item.get("candidate_executable", False)
        for item in edit_candidates
    )
    evaluation_results = parent_selection + parent_reward + episode_records
    evaluation_steps = sum(len(result["trace"]) for result in evaluation_results)
    evaluation_model_calls = sum(
        step["policy_kind"] == "model"
        for result in evaluation_results for step in result["trace"]
    )
    model_generated_tokens = (
        sum(
            len(generation["generated_ids"])
            for experiment in experiments for generation in experiment["generations"]
        )
        + sum(len(item["generation"]["generated_ids"]) for item in edit_candidates)
        + sum(
            len(step["generation"]["generated_ids"])
            for result in evaluation_results for step in result["trace"]
            if step["generation"] is not None
        )
    )
    exploration_steps = sum(item["exploration_steps"] for item in experiments)
    witness_replay_steps = sum(item["witness_replay_steps"] for item in experiments)
    metrics = {
        "parent_selection_score": parent_selection_score,
        "parent_reward_score": parent_reward_score,
        "parent_reward_task_score": parent_reward_task_score,
        "accepted_selection_score": accepted_selection_score,
        "accepted_reward_score": accepted_reward_score,
        "experiment_utilities": experiment_utility_values,
        "mean_experiment_utility": sum(experiment_utility_values) / len(experiment_utility_values),
        "mean_with_evidence_gain": (
            sum(with_evidence_gains) / len(with_evidence_gains) if with_evidence_gains else None
        ),
        "mean_without_evidence_gain": (
            sum(without_evidence_gains) / len(without_evidence_gains) if without_evidence_gains else None
        ),
        "task_candidate_count": task_candidate_count,
        "verified_task_count": verified_task_count,
        "valid_task_generation_rate": (
            verified_task_count / task_candidate_count if task_candidate_count else 0.0
        ),
        "edit_candidate_count": candidate_count,
        "parse_valid_edit_count": parse_valid_count,
        "executable_edit_count": executable_count,
        "valid_edit_rate": executable_count / candidate_count,
        "parse_valid_edit_rate": parse_valid_count / candidate_count,
        "exploration_steps": exploration_steps,
        "witness_replay_steps": witness_replay_steps,
        "evaluation_steps": evaluation_steps,
        "environment_interaction_steps": exploration_steps + witness_replay_steps + evaluation_steps,
        "model_generation_calls": (
            sum(item["exploration_model_calls"] for item in experiments)
            + candidate_count + evaluation_model_calls
        ),
        "model_generated_tokens": model_generated_tokens,
        "accepted_candidate": selected_candidate_id,
        "parent_service_version": parent_version,
        "accepted_service_version": service_version(accepted),
        "research_feedback_reference_count": len(research_feedback.get("reference_ids", [])),
        "hypothesis_rejection_count": sum(bool(item.get("hypothesis_error")) for item in experiments),
        "task_rejections": {
            reason: sum(
                candidate.get("rejection_reason") == reason
                for item in experiments for candidate in item["task_candidates"]
            )
            for reason in sorted({
                candidate.get("rejection_reason")
                for item in experiments for candidate in item["task_candidates"]
                if candidate.get("rejection_reason")
            })
        },
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "config.json", config)
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as stream:
        for row in score_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "research.jsonl").open("w", encoding="utf-8") as stream:
        for sample in experiment_samples + research_samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for result in parent_selection + parent_reward + episode_records:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    return json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))


def split_task_pool(tasks, periods, name):
    if periods < 1 or len(tasks) < periods:
        raise ValueError("{} pool must contain at least one task per period".format(name))
    base, remainder = divmod(len(tasks), periods)
    chunks = []
    offset = 0
    for index in range(periods):
        size = base + (1 if index < remainder else 0)
        chunks.append(tasks[offset:offset + size])
        offset += size
    return chunks


def split_pool_across_periods(tasks, periods, name):
    if periods < 1 or not tasks:
        raise ValueError("{} pool must contain tasks and periods must be positive".format(name))
    chunks = [[] for _ in range(periods)]
    for index, task in enumerate(tasks):
        chunks[index % periods].append(task)
    for index, chunk in enumerate(chunks):
        if not chunk:
            chunk.append(tasks[index % len(tasks)])
    return chunks


def _write_results(path, results):
    with Path(path).open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")


def _evaluate_final_split(config, service_path, tasks, output_path):
    model = ModelSession(config)
    results = evaluate_tasks(config, load_service(service_path), model, tasks)
    del model
    _write_results(output_path, results)
    return {
        "success_rate": mean_score(results),
        "mean_task_reward": mean_task_reward(results),
        "task_count": len(results),
        "scene_count": len({task.source_scene_id for task in tasks}),
        "results_path": str(output_path),
    }


def _period_metrics(batch_rows):
    task_candidates = sum(row["task_candidate_count"] for row in batch_rows)
    edit_candidates = sum(row["edit_candidate_count"] for row in batch_rows)
    with_evidence = [row["mean_with_evidence_gain"] for row in batch_rows
                     if row["mean_with_evidence_gain"] is not None]
    without_evidence = [row["mean_without_evidence_gain"] for row in batch_rows
                        if row["mean_without_evidence_gain"] is not None]
    return {
        "experiment_value": sum(row["mean_experiment_utility"] for row in batch_rows) / len(batch_rows),
        "valid_task_generation_rate": (
            sum(row["verified_task_count"] for row in batch_rows) / task_candidates
            if task_candidates else 0.0
        ),
        "valid_edit_rate": sum(row["executable_edit_count"] for row in batch_rows) / edit_candidates,
        "parse_valid_edit_rate": sum(row["parse_valid_edit_count"] for row in batch_rows) / edit_candidates,
        "with_evidence_gain": sum(with_evidence) / len(with_evidence) if with_evidence else None,
        "without_evidence_gain": sum(without_evidence) / len(without_evidence) if without_evidence else None,
        "parent_selection_success_rate": sum(row["parent_selection_score"] for row in batch_rows) / len(batch_rows),
        "accepted_selection_success_rate": sum(row["accepted_selection_score"] for row in batch_rows) / len(batch_rows),
        "parent_reward_success_rate": sum(row["parent_reward_score"] for row in batch_rows) / len(batch_rows),
        "accepted_reward_success_rate": sum(row["accepted_reward_score"] for row in batch_rows) / len(batch_rows),
        "interaction_steps": sum(row["environment_interaction_steps"] for row in batch_rows),
        "model_generation_calls": sum(row["model_generation_calls"] for row in batch_rows),
        "model_generated_tokens": sum(row["model_generated_tokens"] for row in batch_rows),
    }


def run_full_combined_cycles(config, selection_pool_path, reward_pool_path, periods,
                             batches_per_period, out_dir, parent_path,
                             id_test_path, ood_test_path,
                             experiment_adapter=None, edit_adapter=None):
    """Run repeated on-policy Designer GRPO and Editor OPSD updates."""
    import os

    from .alfworld import list_game_files
    from .train import read_samples, train_grpo, train_opsd

    if periods < 1 or batches_per_period < 1:
        raise ValueError("periods and batches_per_period must be positive")
    if not id_test_path or not ood_test_path:
        raise ValueError("full combined runs require ID and OOD test pools")
    if "swanlab" not in config.get("tracking", {}).get("logger", []):
        raise ValueError("full combined runs require tracking.logger to include swanlab")
    api_key = os.environ.get("SWANLAB_API_KEY")
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY must be set for the full combined run")

    selection_pool = read_tasks(selection_pool_path)
    reward_pool = read_tasks(reward_pool_path)
    id_test_tasks = read_tasks(id_test_path)
    ood_test_tasks = read_tasks(ood_test_path)
    if not id_test_tasks or not ood_test_tasks:
        raise ValueError("ID and OOD test pools must both contain tasks")
    selection_chunks = split_pool_across_periods(selection_pool, periods, "selection")
    reward_chunks = split_pool_across_periods(reward_pool, periods, "reward")
    roles = {
        "selection": selection_pool,
        "reward": reward_pool,
        "id_test": id_test_tasks,
        "ood_test": ood_test_tasks,
    }
    role_scenes = {
        role: {task.source_scene_id for task in tasks}
        for role, tasks in roles.items()
    }
    for left, right in (
        ("selection", "reward"),
        ("selection", "id_test"),
        ("selection", "ood_test"),
        ("reward", "id_test"),
        ("reward", "ood_test"),
        ("id_test", "ood_test"),
    ):
        overlap = role_scenes[left] & role_scenes[right]
        if overlap:
            raise ValueError("{} and {} share source scenes: {}".format(
                left, right, sorted(overlap)
            ))

    experiment_count = config["research"]["experiments_per_group"]
    dev_scenes = set().union(*role_scenes.values())
    diagnostic_batches, diagnostic_scenes = choose_diagnostic_batches(
        config, periods, dev_scenes
    )
    experiment_files = [
        game_file
        for game_file in list_game_files(config["alfworld"]["config_path"],
                                         config["research"]["experiment_split"])
        if supported_placement_game(game_file)
        and source_scene_id(game_file) not in (dev_scenes | diagnostic_scenes)
    ]
    experiment_scenes = {source_scene_id(path) for path in experiment_files}
    if len(experiment_scenes) < experiment_count:
        raise ValueError("only {} training scenes remain after source-scene isolation; {} are required".format(
            len(experiment_scenes), experiment_count
        ))

    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError("run output already exists: {}".format(out_dir))
    out_dir.mkdir(parents=True)
    scene_manifest = {
        "selection_scenes": sorted(role_scenes["selection"]),
        "reward_scenes": sorted(role_scenes["reward"]),
        "id_test_scenes": sorted(role_scenes["id_test"]),
        "ood_test_scenes": sorted(role_scenes["ood_test"]),
        "diagnostic_scenes": sorted(diagnostic_scenes),
        "experiment_training_allowed_scenes": sorted(experiment_scenes),
        "experiment_scenes": [],
    }
    write_json(out_dir / "source_scene_manifest.json", scene_manifest)
    tracking = config["tracking"]
    run_name = "{}-{}".format(tracking["experiment_name"], out_dir.name)
    import swanlab

    swanlab.login(api_key)
    swanlab.init(
        project=tracking["project_name"],
        experiment_name=run_name,
        config={
            "method": "combined",
            "run_id": run_name,
            "periods": periods,
            "batches_per_period": batches_per_period,
            "planned_designer_updates": periods * batches_per_period,
            "planned_editor_updates": periods * batches_per_period,
            **config,
        },
        logdir=str(out_dir / "swanlab" / "research"),
        mode=os.environ.get("SWANLAB_MODE", "cloud"),
    )

    base_seed = config["run"]["seed"]
    test_config = copy.deepcopy(config)
    test_config["run"]["evaluation_seed"] = base_seed + 900_000
    initial_id = _evaluate_final_split(
        test_config, parent_path, id_test_tasks, out_dir / "initial_id_test_scores.jsonl"
    )
    initial_ood = _evaluate_final_split(
        test_config, parent_path, ood_test_tasks, out_dir / "initial_ood_test_scores.jsonl"
    )
    swanlab.log({
        "test/initial_id_success_rate": initial_id["success_rate"],
        "test/initial_ood_success_rate": initial_ood["success_rate"],
    }, step=0)

    current_parent = str(parent_path)
    current_experiment_adapter = experiment_adapter
    current_edit_adapter = edit_adapter
    summaries = []
    feedback_history_dirs = []
    completed_batches = 0
    designer_parameter_updates = 0
    editor_parameter_updates = 0
    progress_path = out_dir / "progress.json"
    progress = {
        "status": "running",
        "run_id": run_name,
        "planned_periods": periods,
        "planned_batches_per_period": batches_per_period,
        "planned_designer_updates": periods * batches_per_period,
        "planned_editor_updates": periods * batches_per_period,
        "completed_batches": 0,
        "designer_optimizer_steps": 0,
        "editor_optimizer_steps": 0,
    }
    write_json(progress_path, progress)
    try:
        for period_index in range(periods):
            period_number = period_index + 1
            period_dir = out_dir / "period-{:02d}".format(period_number)
            period_dir.mkdir()
            selection_tasks = selection_chunks[period_index]
            reward_tasks = reward_chunks[period_index]
            selection_path = period_dir / "selection_tasks.jsonl"
            reward_path = period_dir / "reward_tasks.jsonl"
            write_tasks(selection_path, selection_tasks)
            write_tasks(reward_path, reward_tasks)

            # Each batch evaluates against the same parent and paired task seeds.
            period_parent = current_parent
            period_seed = base_seed + period_number * 10_000
            diagnostic_config = copy.deepcopy(config)
            diagnostic_config["run"]["seed"] = base_seed + 1000 * period_index
            diagnostic_summary = run_solver_diagnostics(
                diagnostic_config,
                period_parent,
                diagnostic_batches[period_index],
                period_dir,
                "period-{:02d}-diagnostic".format(period_number),
            )
            # Selection baselines are recorded by each batch; the first one supplies
            # the common score used to choose a period winner.
            best_period_service = load_service(period_parent)
            best_period_candidate = None
            batch_summaries = []
            period_candidate_paths = []
            period_metrics = None

            for batch_index in range(batches_per_period):
                batch_number = batch_index + 1
                global_step = period_index * batches_per_period + batch_number
                batch_dir = period_dir / "batch-{:02d}".format(batch_number)
                batch_dir.mkdir()
                batch_config = copy.deepcopy(config)
                batch_config["run"]["seed"] = period_seed + batch_number
                batch_config["run"]["evaluation_seed"] = period_seed + 50_000
                batch_config.setdefault("tracking", {})
                batch_config["tracking"].update({
                    "run_id": run_name,
                    "period": period_number,
                    "batch": batch_number,
                    "trainer_experiment_name": "{}-p{:02d}-b{:02d}-designer-grpo".format(
                        run_name, period_number, batch_number
                    ),
                    "swanlab_log_dir": str(batch_dir / "designer-update" / "swanlab"),
                })
                research_feedback = make_research_feedback(
                    diagnostic_summary, feedback_history_dirs
                )
                batch_metrics = experiment_group(
                    batch_config,
                    selection_path,
                    reward_path,
                    batch_dir,
                    period_parent,
                    current_experiment_adapter,
                    current_edit_adapter,
                    excluded_scenes=dev_scenes | diagnostic_scenes,
                    experiment_game_files=experiment_files,
                    research_feedback=research_feedback,
                )
                feedback_history_dirs.append((
                    batch_dir,
                    "period-{:02d}/batch-{:02d}".format(period_number, batch_number),
                ))
                experiment_rows = json.loads(
                    (batch_dir / "experiments.json").read_text(encoding="utf-8")
                )
                observed_experiment_scenes = {
                    source_scene_id(row["game_file"]) for row in experiment_rows
                }
                if observed_experiment_scenes & (dev_scenes | diagnostic_scenes):
                    raise ValueError("an experiment batch overlaps a reserved source scene")
                scene_manifest["experiment_scenes"] = sorted(
                    set(scene_manifest["experiment_scenes"]) | observed_experiment_scenes
                )
                write_json(out_dir / "source_scene_manifest.json", scene_manifest)
                designer_adapter_before = current_experiment_adapter
                editor_adapter_before = current_edit_adapter
                designer_train_dir = batch_dir / "designer-update"
                designer_samples = read_samples(batch_dir, "experiment")
                designer_training = train_grpo(
                    batch_config,
                    designer_samples,
                    designer_train_dir,
                    current_experiment_adapter,
                )
                current_experiment_adapter = designer_training["adapter"]
                progress["designer_optimizer_steps"] += int(designer_training["trainer_step"] > 0)
                designer_parameter_updates += int(designer_training["parameter_update"])
                progress.update({
                    "current_period": period_number,
                    "current_batch": batch_number,
                    "designer_parameter_updates": designer_parameter_updates,
                })
                write_json(progress_path, progress)

                batch_config["tracking"]["trainer_experiment_name"] = (
                    "{}-p{:02d}-b{:02d}-editor-opsd".format(
                        run_name, period_number, batch_number
                    )
                )
                batch_config["tracking"]["swanlab_log_dir"] = str(
                    batch_dir / "editor-update" / "swanlab"
                )
                editor_training = train_opsd(
                    batch_config,
                    read_samples(batch_dir, "edit"),
                    batch_dir / "editor-update",
                    current_edit_adapter,
                )
                current_edit_adapter = editor_training["adapter"]
                progress["editor_optimizer_steps"] += int(editor_training["trainer_step"] > 0)
                editor_parameter_updates += int(editor_training["parameter_update"])
                progress.update({"editor_parameter_updates": editor_parameter_updates})
                write_json(progress_path, progress)

                batch_metrics.update({
                    "run_id": run_name,
                    "period": period_number,
                    "batch": batch_number,
                    "method": "combined",
                    "designer_training": designer_training,
                    "editor_training": editor_training,
                    "designer_adapter_before": designer_adapter_before,
                    "designer_adapter": current_experiment_adapter,
                    "editor_adapter_before": editor_adapter_before,
                    "editor_adapter": current_edit_adapter,
                    "selection_task_count": len(selection_tasks),
                    "reward_task_count": len(reward_tasks),
                })
                write_json(batch_dir / "metrics.json", batch_metrics)

                parent_score = batch_metrics["parent_selection_score"]
                accepted_score = batch_metrics["accepted_selection_score"]
                if batch_metrics["accepted_candidate"] is not None:
                    period_candidate_paths.append({
                        "candidate_id": batch_metrics["accepted_candidate"],
                        "batch": batch_number,
                        "selection_screen_score": accepted_score,
                        "reward_score": batch_metrics["accepted_reward_score"],
                        "service_path": str(batch_dir / "service.json"),
                    })

                batch_summaries.append(batch_metrics)
                completed_batches += 1
                progress.update({
                    "completed_batches": completed_batches,
                    "current_period": period_number,
                    "current_batch": batch_number,
                })
                write_json(progress_path, progress)

                batch_tracking_metrics = {
                    "research/experiment_value": batch_metrics["mean_experiment_utility"],
                    "tasks/valid_generation_rate": batch_metrics["valid_task_generation_rate"],
                    "edits/valid_rate": batch_metrics["valid_edit_rate"],
                    "edits/parse_valid_rate": batch_metrics["parse_valid_edit_rate"],
                    "edits/with_evidence_gain": batch_metrics["mean_with_evidence_gain"],
                    "edits/without_evidence_gain": batch_metrics["mean_without_evidence_gain"],
                    "service/parent_selection_success_rate": parent_score,
                    "service/accepted_selection_success_rate": accepted_score,
                    "service/parent_reward_success_rate": batch_metrics["parent_reward_score"],
                    "service/accepted_reward_success_rate": batch_metrics["accepted_reward_score"],
                    "environment/interaction_steps": batch_metrics["environment_interaction_steps"],
                    "environment/model_generation_calls": batch_metrics["model_generation_calls"],
                    "environment/model_generated_tokens": batch_metrics["model_generated_tokens"],
                    "diagnostic/success_count": diagnostic_summary["success_count"],
                    "diagnostic/failure_count": diagnostic_summary["failure_count"],
                    "diagnostic/mean_goal_progress": diagnostic_summary["mean_goal_progress"],
                    "feedback/reference_count": len(research_feedback["reference_ids"]),
                    "feedback/task_rejected_as_irrelevant": batch_metrics["task_rejections"].get(
                        "not_diagnostic_target", 0
                    ),
                    "feedback/task_goal_already_satisfied": batch_metrics["task_rejections"].get(
                        "goal_already_satisfied_initially", 0
                    ),
                    "feedback/task_rejected_as_source_goal": batch_metrics["task_rejections"].get(
                        "duplicates_source_task_goal", 0
                    ),
                    "feedback/task_witness_replay_failed": batch_metrics["task_rejections"].get(
                        "witness_replay_failed", 0
                    ),
                    "feedback/hypothesis_rejected": batch_metrics["hypothesis_rejection_count"],
                    "research/feedback_and_experiments": swanlab_research_record(
                        research_feedback, batch_dir / "experiments.json", swanlab
                    ),
                    "training/designer_reward_mean": designer_training["grpo_reward_mean"],
                    "training/designer_reward_variance": designer_training["grpo_reward_variance"],
                    "training/designer_nonzero_advantage_fraction": (
                        designer_training["grpo_nonzero_advantage_group_fraction"]
                    ),
                    "training/designer_effective_update": int(
                        designer_training["effective_update"]
                    ),
                    "training/designer_parameter_update": int(
                        designer_training["parameter_update"]
                    ),
                    "training/editor_effective_update": int(
                        editor_training["effective_update"]
                    ),
                    "training/editor_parameter_update": int(
                        editor_training["parameter_update"]
                    ),
                    "training/designer_optimizer_step": designer_training["trainer_step"],
                    "training/editor_optimizer_step": editor_training["trainer_step"],
                }
                if batch_number == batches_per_period:
                    period_metrics = _period_metrics(batch_summaries)
                    batch_tracking_metrics.update({
                        "period/experiment_value": period_metrics["experiment_value"],
                        "period/valid_generation_rate": period_metrics["valid_task_generation_rate"],
                        "period/valid_edit_rate": period_metrics["valid_edit_rate"],
                        "period/parse_valid_edit_rate": period_metrics["parse_valid_edit_rate"],
                        "period/with_evidence_gain": period_metrics["with_evidence_gain"],
                        "period/without_evidence_gain": period_metrics["without_evidence_gain"],
                        "period/parent_selection_success_rate": period_metrics["parent_selection_success_rate"],
                        "period/accepted_selection_success_rate": period_metrics["accepted_selection_success_rate"],
                        "period/interaction_steps": period_metrics["interaction_steps"],
                        "period/model_generation_calls": period_metrics["model_generation_calls"],
                        "period/model_generated_tokens": period_metrics["model_generated_tokens"],
                    })
                swanlab.log(
                    {key: value for key, value in batch_tracking_metrics.items()
                     if value is not None},
                    step=global_step,
                )

            period_validation_config = copy.deepcopy(config)
            period_validation_config["run"]["evaluation_seed"] = base_seed + 800_000
            validation_model = ModelSession(period_validation_config)
            parent_validation_results = evaluate_tasks(
                period_validation_config,
                best_period_service,
                validation_model,
                selection_pool,
            )
            parent_validation_score = mean_score(parent_validation_results)
            validation_rows = [
                {"candidate_id": "period-parent", "set": "selection", **result}
                for result in parent_validation_results
            ]
            candidate_validation = []
            for item in period_candidate_paths:
                candidate_service = load_service(item["service_path"])
                results = evaluate_tasks(
                    period_validation_config,
                    candidate_service,
                    validation_model,
                    selection_pool,
                )
                score = mean_score(results)
                executable = not any(result["candidate_error"] for result in results)
                candidate_validation.append({
                    **item,
                    "selection_validation_score": score,
                    "executable_on_full_selection": executable,
                })
                validation_rows.extend(
                    {"candidate_id": item["candidate_id"], "set": "selection", **result}
                    for result in results
                )
            del validation_model
            eligible_candidates = [item for item in candidate_validation
                                   if item["executable_on_full_selection"]]
            selected_validation_index = choose_candidate(
                parent_validation_score,
                [item["selection_validation_score"] for item in eligible_candidates],
            ) if eligible_candidates else None
            if selected_validation_index is not None:
                best_period_candidate = eligible_candidates[selected_validation_index]
                best_period_service = load_service(best_period_candidate["service_path"])
            _write_results(period_dir / "period_selection_validation_scores.jsonl", validation_rows)
            period_validation_summary = {
                "parent_selection_score": parent_validation_score,
                "selected_candidate": best_period_candidate,
                "candidate_count": len(candidate_validation),
                "candidate_results": candidate_validation,
                "selection_task_count": len(selection_pool),
                "selection_scene_count": len(role_scenes["selection"]),
                "evaluation_steps": sum(len(row["trace"]) for row in validation_rows),
            }
            write_json(period_dir / "period_selection_validation.json", period_validation_summary)
            write_json(period_dir / "service.json", best_period_service)
            period_summary = {
                "period": period_number,
                "diagnostic_summary": diagnostic_summary,
                "selection_task_ids": [task.task_id for task in selection_tasks],
                "reward_task_ids": [task.task_id for task in reward_tasks],
                "selection_scene_count": len({task.source_scene_id for task in selection_tasks}),
                "reward_scene_count": len({task.source_scene_id for task in reward_tasks}),
                "selected_service_candidate": best_period_candidate,
                "selection_validation": period_validation_summary,
                "metrics": period_metrics,
                "service": str(period_dir / "service.json"),
                "batches": batch_summaries,
                "designer_optimizer_steps": sum(
                    row["designer_training"]["trainer_step"] > 0 for row in batch_summaries
                ),
                "editor_optimizer_steps": sum(
                    row["editor_training"]["trainer_step"] > 0 for row in batch_summaries
                ),
            }
            write_json(period_dir / "period.json", period_summary)
            summaries.append(period_summary)
            current_parent = str(period_dir / "service.json")

        final_id = _evaluate_final_split(
            test_config, current_parent, id_test_tasks, out_dir / "final_id_test_scores.jsonl"
        )
        final_ood = _evaluate_final_split(
            test_config, current_parent, ood_test_tasks, out_dir / "final_ood_test_scores.jsonl"
        )
        id_delta = final_id["success_rate"] - initial_id["success_rate"]
        ood_delta = final_ood["success_rate"] - initial_ood["success_rate"]
        final = {
            "method": "combined",
            "run_id": run_name,
            "swanlab_project": tracking["project_name"],
            "periods": summaries,
            "final_service": current_parent,
            "final_experiment_adapter": current_experiment_adapter,
            "final_edit_adapter": current_edit_adapter,
            "initial_id_test": initial_id,
            "final_id_test": final_id,
            "id_test_success_rate_delta": id_delta,
            "initial_ood_test": initial_ood,
            "final_ood_test": final_ood,
            "ood_test_success_rate_delta": ood_delta,
            "planned_designer_updates": periods * batches_per_period,
            "planned_editor_updates": periods * batches_per_period,
            "designer_optimizer_steps": progress["designer_optimizer_steps"],
            "editor_optimizer_steps": progress["editor_optimizer_steps"],
            "designer_parameter_updates": designer_parameter_updates,
            "editor_parameter_updates": editor_parameter_updates,
            "designer_effective_updates": sum(
                bool(row["designer_training"]["effective_update"])
                for period in summaries for row in period["batches"]
            ),
            "editor_effective_updates": sum(
                bool(row["editor_training"]["effective_update"])
                for period in summaries for row in period["batches"]
            ),
        }
        write_json(out_dir / "run.json", final)
        progress.update({"status": "completed", **{key: value for key, value in final.items()
                                                    if key.endswith("updates") or key.endswith("steps")}})
        write_json(progress_path, progress)
        swanlab.log({
            "test/final_id_success_rate": final_id["success_rate"],
            "test/id_success_rate_delta": id_delta,
            "test/final_ood_success_rate": final_ood["success_rate"],
            "test/ood_success_rate_delta": ood_delta,
        }, step=periods * batches_per_period + 1)
        swanlab.finish()
        return final
    except Exception as error:
        progress.update({"status": "failed", "error": repr(error),
                         "completed_batches": completed_batches})
        write_json(progress_path, progress)
        swanlab.finish()
        raise


def run_cycles(config, selection_pool_path, reward_pool_path, periods, method, out_dir,
               parent_path, test_path=None, experiment_adapter=None, edit_adapter=None,
               batches_per_period=None, ood_test_path=None):
    from .train import read_samples, train_grpo, train_opsd

    if method not in ("rl", "distill", "combined", "designer-only"):
        raise ValueError("method must be rl, distill, combined, or designer-only")
    if method == "combined" and batches_per_period and batches_per_period > 1:
        return run_full_combined_cycles(
            config,
            selection_pool_path,
            reward_pool_path,
            periods,
            batches_per_period,
            out_dir,
            parent_path,
            test_path,
            ood_test_path,
            experiment_adapter,
            edit_adapter,
        )
    if method == "designer-only" and not test_path:
        raise ValueError("designer-only requires an independent final test pool")
    selection_pool = read_tasks(selection_pool_path)
    reward_pool = read_tasks(reward_pool_path)
    selection_chunks = split_task_pool(selection_pool, periods, "selection")
    reward_chunks = split_task_pool(reward_pool, periods, "reward")
    test_tasks = read_tasks(test_path) if test_path else []

    selection_scenes = {task.source_scene_id for task in selection_pool}
    reward_scenes = {task.source_scene_id for task in reward_pool}
    test_scenes = {task.source_scene_id for task in test_tasks}
    if selection_scenes & reward_scenes:
        raise ValueError("selection and reward pools share ALFRED source scenes")
    if (selection_scenes | reward_scenes) & test_scenes:
        raise ValueError("final test tasks share source scenes with development pools")
    if len(selection_scenes) != len(selection_pool) or len(reward_scenes) != len(reward_pool):
        raise ValueError("each pool must contain at most one task per ALFRED source scene")
    if len(test_scenes) != len(test_tasks):
        raise ValueError("final test pool must contain at most one task per ALFRED source scene")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    current_parent = str(parent_path)
    current_experiment_adapter = experiment_adapter
    current_edit_adapter = edit_adapter
    cycle_summaries = []
    base_seed = config["run"]["seed"]
    development_scenes = selection_scenes | reward_scenes | test_scenes
    diagnostic_batches, diagnostic_scenes = choose_diagnostic_batches(
        config, periods, development_scenes
    )
    used_experiment_scenes = set()
    feedback_history_dirs = []
    tracking_config = config.get("tracking", {})
    swanlab = None
    swanlab_run_name = None

    if method == "designer-only":
        import os
        import swanlab as swanlab_module

        api_key = os.environ.get("SWANLAB_API_KEY")
        if not api_key:
            raise RuntimeError("SWANLAB_API_KEY must be set for the designer-only run")
        if "swanlab" not in tracking_config["logger"]:
            raise ValueError("designer-only tracking.logger must include swanlab")
        project_name = tracking_config["project_name"]
        swanlab_run_name = "{}-{}".format(tracking_config["experiment_name"], out_dir.name)
        swanlab_module.login(api_key)
        swanlab_module.init(
            project=project_name,
            experiment_name=swanlab_run_name,
            config={"method": method, "run_id": swanlab_run_name, **config},
            logdir=str(out_dir / "swanlab" / "research"),
            mode=os.environ.get("SWANLAB_MODE", "cloud"),
        )
        swanlab = swanlab_module

    initial_test_results = []
    if method == "designer-only":
        initial_model = ModelSession(config)
        initial_test_results = evaluate_tasks(
            config, load_service(parent_path), initial_model, test_tasks
        )
        del initial_model
        with (out_dir / "initial_test_scores.jsonl").open("w", encoding="utf-8") as stream:
            for result in initial_test_results:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        initial_test_success_rate = mean_score(initial_test_results)
        swanlab.log(
            {"test/initial_success_rate": initial_test_success_rate,
             "test/initial_task_count": len(initial_test_results)},
            step=0,
        )

    scene_manifest = {
        "experiment_scenes": [],
        "diagnostic_scenes": sorted(diagnostic_scenes),
        "selection_scenes": sorted(selection_scenes),
        "reward_scenes": sorted(reward_scenes),
        "test_scenes": sorted(test_scenes),
    }
    write_json(out_dir / "source_scene_manifest.json", scene_manifest)

    for period in range(periods):
        period_config = copy.deepcopy(config)
        period_config["run"]["seed"] = base_seed + 1000 * period
        if method == "designer-only":
            period_config.setdefault("tracking", {})
            period_config["tracking"]["run_id"] = swanlab_run_name
            period_config["tracking"]["period"] = period + 1
            period_config["tracking"]["trainer_experiment_name"] = (
                "{}-period-{:02d}-designer-grpo".format(swanlab_run_name, period + 1)
            )
            period_config["tracking"]["swanlab_log_dir"] = str(
                out_dir / "period-{}".format(period) / "swanlab"
            )
        period_dir = out_dir / "period-{}".format(period)
        period_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_summary = run_solver_diagnostics(
            period_config,
            current_parent,
            diagnostic_batches[period],
            period_dir,
            "period-{:02d}-diagnostic".format(period + 1),
        )
        research_feedback = make_research_feedback(
            diagnostic_summary, feedback_history_dirs
        )
        write_json(period_dir / "research_feedback.json", research_feedback)
        selection_path = period_dir / "selection_tasks.jsonl"
        reward_path = period_dir / "reward_tasks.jsonl"
        write_tasks(selection_path, selection_chunks[period])
        write_tasks(reward_path, reward_chunks[period])
        metrics = experiment_group(
            period_config,
            selection_path,
            reward_path,
            period_dir,
            current_parent,
            current_experiment_adapter,
            current_edit_adapter,
            excluded_scenes=development_scenes | diagnostic_scenes | used_experiment_scenes,
            research_feedback=research_feedback,
        )

        experiment_records = json.loads((period_dir / "experiments.json").read_text(encoding="utf-8"))
        period_experiment_scenes = {
            source_scene_id(item["game_file"]) for item in experiment_records
        }
        if period_experiment_scenes & (
            development_scenes | diagnostic_scenes | used_experiment_scenes
        ):
            raise ValueError("experiment scenes overlap another evaluation or research period")
        used_experiment_scenes.update(period_experiment_scenes)
        scene_manifest["experiment_scenes"] = sorted(used_experiment_scenes)
        write_json(out_dir / "source_scene_manifest.json", scene_manifest)

        metrics.update({
            "run_id": swanlab_run_name or out_dir.name,
            "period": period + 1,
            "method": method,
        })
        write_json(period_dir / "metrics.json", metrics)
        if swanlab is not None:
            tracking_values = {
                    "research/experiment_value": metrics["mean_experiment_utility"],
                    "tasks/valid_generation_rate": metrics["valid_task_generation_rate"],
                    "edits/valid_rate": metrics["valid_edit_rate"],
                    "edits/parse_valid_rate": metrics["parse_valid_edit_rate"],
                    "edits/with_evidence_gain": metrics["mean_with_evidence_gain"],
                    "edits/without_evidence_gain": metrics["mean_without_evidence_gain"],
                    "service/parent_selection_success_rate": metrics["parent_selection_score"],
                    "service/accepted_selection_success_rate": metrics["accepted_selection_score"],
                    "service/parent_reward_success_rate": metrics["parent_reward_score"],
                    "service/accepted_reward_success_rate": metrics["accepted_reward_score"],
                    "environment/interaction_steps": metrics["environment_interaction_steps"],
                    "environment/model_generation_calls": metrics["model_generation_calls"],
                    "diagnostic/success_count": diagnostic_summary["success_count"],
                    "diagnostic/failure_count": diagnostic_summary["failure_count"],
                    "diagnostic/mean_goal_progress": diagnostic_summary["mean_goal_progress"],
                    "feedback/reference_count": len(research_feedback["reference_ids"]),
                    "feedback/task_rejected_as_irrelevant": metrics["task_rejections"].get(
                        "not_diagnostic_target", 0
                    ),
                    "feedback/task_goal_already_satisfied": metrics["task_rejections"].get(
                        "goal_already_satisfied_initially", 0
                    ),
                    "feedback/task_rejected_as_source_goal": metrics["task_rejections"].get(
                        "duplicates_source_task_goal", 0
                    ),
                    "feedback/task_witness_replay_failed": metrics["task_rejections"].get(
                        "witness_replay_failed", 0
                    ),
                    "feedback/hypothesis_rejected": metrics["hypothesis_rejection_count"],
                    "research/feedback_and_experiments": swanlab_research_record(
                        research_feedback, period_dir / "experiments.json", swanlab
                    ),
                }
            swanlab.log(
                {key: value for key, value in tracking_values.items() if value is not None},
                step=period + 1,
            )

        experiment_adapter_before = current_experiment_adapter
        if method in ("rl", "combined", "designer-only"):
            experiment_training = train_grpo(
                period_config,
                read_samples(period_dir, "experiment"),
                period_dir / "experiment-update",
                current_experiment_adapter,
            )
            if method == "designer-only" and not experiment_training["parameter_update"]:
                raise RuntimeError("Designer GRPO completed without changing its LoRA parameters")
            current_experiment_adapter = experiment_training["adapter"]
        else:
            experiment_training = None

        if method == "designer-only":
            edit_training = None
        else:
            edit_objective = "grpo" if method == "rl" else "opsd"
            edit_samples = read_samples(period_dir, "edit")
            if edit_objective == "grpo":
                edit_training = train_grpo(
                    period_config,
                    edit_samples,
                    period_dir / "edit-update",
                    current_edit_adapter,
                )
            else:
                edit_training = train_opsd(
                    period_config,
                    edit_samples,
                    period_dir / "edit-update",
                    current_edit_adapter,
                )
            current_edit_adapter = edit_training["adapter"]
        current_parent = str(period_dir / "service.json")
        cycle_summaries.append(
            {
                "period": period,
                "output_dir": str(period_dir),
                "research": metrics,
                "experiment_training": experiment_training,
                "edit_training": edit_training,
                "service": current_parent,
                "diagnostic_summary": diagnostic_summary,
                "research_feedback": research_feedback,
                "experiment_adapter_before": experiment_adapter_before,
                "experiment_adapter": current_experiment_adapter,
                "edit_adapter": current_edit_adapter,
            }
        )
        feedback_history_dirs.append((period_dir, "period-{:02d}".format(period + 1)))

    test_results = []
    if test_tasks:
        final_service = load_service(current_parent)
        final_model = ModelSession(config)
        test_results = evaluate_tasks(config, final_service, final_model, test_tasks)
        del final_model
        with (out_dir / "test_scores.jsonl").open("w", encoding="utf-8") as stream:
            for result in test_results:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    final_test_success_rate = mean_score(test_results) if test_results else None
    initial_test_success_rate = mean_score(initial_test_results) if initial_test_results else None
    test_success_rate_delta = (
        final_test_success_rate - initial_test_success_rate
        if final_test_success_rate is not None and initial_test_success_rate is not None else None
    )
    test_success_rate_relative_change = (
        test_success_rate_delta / initial_test_success_rate
        if test_success_rate_delta is not None and initial_test_success_rate else None
    )
    if swanlab is not None:
        final_test_metrics = {
            "test/final_success_rate": final_test_success_rate,
            "test/success_rate_delta": test_success_rate_delta,
            "test/task_count": len(test_results),
        }
        if test_success_rate_relative_change is not None:
            final_test_metrics["test/success_rate_relative_change"] = test_success_rate_relative_change
        swanlab.log(final_test_metrics, step=periods + 1)
        swanlab.finish()

    result = {
        "method": method,
        "swanlab_project": tracking_config.get("project_name") if swanlab is not None else None,
        "swanlab_experiment_name": swanlab_run_name,
        "periods": cycle_summaries,
        "final_service": current_parent,
        "final_experiment_adapter": current_experiment_adapter,
        "final_edit_adapter": current_edit_adapter,
        "initial_test_success_rate": initial_test_success_rate,
        "final_test_success_rate": final_test_success_rate,
        "test_success_rate": final_test_success_rate,
        "test_success_rate_delta": test_success_rate_delta,
        "test_success_rate_relative_change": test_success_rate_relative_change,
        "test_mean_task_reward": mean_task_reward(test_results) if test_results else None,
        "test_task_count": len(test_results),
    }
    write_json(out_dir / "run.json", result)
    return result


def research_once(config, diagnostic_path, selection_path, reward_path, out_dir, parent_path, edit_adapter=None):
    diagnostic_tasks = read_tasks(diagnostic_path)
    selection_tasks = read_tasks(selection_path)
    reward_tasks = read_tasks(reward_path)
    assert_disjoint_scenes([{"tasks": diagnostic_tasks}], selection_tasks, reward_tasks)
    parent = load_service(parent_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    service_model = ModelSession(config)
    diagnostic_results = evaluate_tasks(config, parent, service_model, diagnostic_tasks)
    diagnostic_report = score_report(diagnostic_tasks, diagnostic_results)
    parent_selection = evaluate_tasks(config, parent, service_model, selection_tasks)
    parent_reward = evaluate_tasks(config, parent, service_model, reward_tasks)
    parent_selection_score = mean_score(parent_selection)
    parent_reward_score = mean_score(parent_reward)
    parent_reward_task_score = mean_task_reward(parent_reward)
    del service_model

    edit_model = ModelSession(config, adapter_path=edit_adapter)
    generated = []
    prompt = candidate_messages(parent, diagnostic_report)
    for index in range(config["research"]["candidates"]):
        seed_everything(config["run"]["seed"] + index, edit_model.torch)
        generation = edit_model.generate(
            prompt,
            max_new_tokens=config["model"]["max_new_tokens"],
            temperature=config["model"]["temperature"],
            top_p=config["model"]["top_p"],
        )
        try:
            candidate = parse_candidate(generation["generated_text"])
            error = None
        except (json.JSONDecodeError, ValueError) as candidate_error:
            candidate = None
            error = str(candidate_error)
        generated.append((generation, candidate, error, "candidate-{}".format(index), prompt))
    del edit_model
    service_model = ModelSession(config)

    candidate_selection_scores = []
    candidate_reward_scores = []
    candidate_reward_task_scores = []
    score_rows = []
    candidate_episodes = []
    research_samples = []
    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    for generation, candidate, error, candidate_id, prompt_messages in generated:
        if candidate is None:
            candidate_selection_scores.append(0.0)
            candidate_reward_scores.append(0.0)
            candidate_reward_task_scores.append(0.0)
            selection_results = []
            reward_results = []
            reward = -parent_reward_task_score
        else:
            write_json(candidates_dir / (candidate_id + ".json"), candidate)
            selection_results = evaluate_tasks(config, candidate, service_model, selection_tasks)
            reward_results = evaluate_tasks(config, candidate, service_model, reward_tasks)
            candidate_selection_scores.append(mean_score(selection_results))
            candidate_reward_scores.append(mean_score(reward_results))
            candidate_reward_task_scores.append(mean_task_reward(reward_results))
            reward = candidate_reward_task_scores[-1] - parent_reward_task_score
        for set_name, tasks, results, parent_results in (
            ("selection", selection_tasks, selection_results, parent_selection),
            ("reward", reward_tasks, reward_results, parent_reward),
        ):
            for index, task in enumerate(tasks):
                score_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "set": set_name,
                        "task_id": task.task_id,
                        "parent_success": parent_results[index]["success"],
                        "candidate_success": results[index]["success"] if results else False,
                        "parent_task_reward": parent_results[index]["task_reward"],
                        "candidate_task_reward": results[index]["task_reward"] if results else 0.0,
                    }
                )
        candidate_episodes.extend(selection_results)
        candidate_episodes.extend(reward_results)
        research_samples.append(
            asdict(
                ResearchSample(
                    group_id="edit-group-0",
                    role="edit",
                    prompt_ids=generation["prompt_ids"],
                    generated_ids=generation["generated_ids"],
                    generated_text=generation["generated_text"],
                    behavior_logprobs=generation["behavior_logprobs"],
                    reward=reward,
                    task_metrics={
                        "candidate_id": candidate_id,
                        "trajectory_id": candidate_id,
                        "sample_weight": 1.0,
                        "prompt_messages": prompt_messages,
                        "post_feedback": post_execution_feedback(reward_results, error),
                        "valid_candidate": candidate is not None,
                        "parse_error": error,
                        "selection_score": candidate_selection_scores[-1],
                        "reward_score": candidate_reward_scores[-1],
                        "reward_task_score": candidate_reward_task_scores[-1],
                    },
                    model_revision=generation["model_revision"],
                    adapter_version=generation["adapter_version"],
                    sampling_parameters=generation["sampling_parameters"],
                    training_eligible=True,
                )
            )
        )

    selected_index = (
        choose_candidate(parent_selection_score, candidate_selection_scores)
        if candidate_selection_scores else None
    )
    accepted = generated[selected_index][1] if selected_index is not None else parent
    write_json(out_dir / "service.json", accepted)
    write_json(out_dir / "selection.json", {
        "parent_score": parent_selection_score,
        "candidate_scores": candidate_selection_scores,
        "accepted_candidate": generated[selected_index][3] if selected_index is not None else None,
    })
    write_json(out_dir / "parent.json", parent)
    write_json(out_dir / "diagnostic_report.json", diagnostic_report)
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "metrics.json", {
        "parent_selection_score": parent_selection_score,
        "parent_reward_score": parent_reward_score,
        "parent_reward_task_score": parent_reward_task_score,
        "candidate_selection_scores": candidate_selection_scores,
        "candidate_reward_scores": candidate_reward_scores,
        "candidate_reward_task_scores": candidate_reward_task_scores,
        "accepted_candidate": generated[selected_index][3] if selected_index is not None else None,
    })
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as stream:
        for row in score_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for result in diagnostic_results + parent_selection + parent_reward + candidate_episodes:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (out_dir / "research.jsonl").open("w", encoding="utf-8") as stream:
        for sample in research_samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
