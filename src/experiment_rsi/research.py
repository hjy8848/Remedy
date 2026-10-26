import json
import copy
import random
from dataclasses import asdict
from pathlib import Path

from .alfworld import list_game_files
from .evaluation import (
    choose_candidate,
    evaluate_tasks,
    evidence_removal_reward,
    mean_score,
    mean_task_reward,
)
from .model import ModelSession, seed_everything
from .policy import load_service
from .records import ResearchSample
from .synthesis import (
    explore_game,
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
                "The step function must return either {kind: action, action: text, state: JSON} "
                "or {kind: model, messages: chat messages, state: JSON}. view has objective, "
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
                "The step function must return either an action directive or a model directive "
                "with chat messages. view has objective, observation, history, admissible_commands, "
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


def choose_task_files(config, split, count, excluded_scenes=()):
    files = [
        path
        for path in list_game_files(config["alfworld"]["config_path"], split)
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
                     experiment_adapter=None, edit_adapter=None, excluded_scenes=()):
    selection_tasks = read_tasks(selection_path)
    reward_tasks = read_tasks(reward_path)
    parent = load_service(parent_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_split = config["research"]["experiment_split"]
    experiment_count = config["research"]["experiments_per_group"]
    edits_per_branch = config["research"]["edits_per_branch"]
    if experiment_count < 2:
        raise ValueError("experiments_per_group must be at least 2 for GRPO")
    if edits_per_branch < 2:
        raise ValueError("edits_per_branch must be at least 2 for GRPO")
    game_files = choose_task_files(config, experiment_split, experiment_count, excluded_scenes)
    if len(game_files) != experiment_count:
        raise ValueError(
            "requested {} distinct experiment scenes, found {}".format(experiment_count, len(game_files))
        )

    experiment_model = ModelSession(config, adapter_path=experiment_adapter)
    experiments = []
    for index, game_file in enumerate(game_files):
        seed_everything(config["run"]["seed"] + index, experiment_model.torch)
        explored = explore_game(
            config,
            game_file,
            experiment_model,
            experiment_split,
            config["run"]["max_steps"],
            out_dir / "diagnostic_games",
            parent_service=parent,
        )
        experiments.append(
            {
                "experiment_id": "experiment-{}".format(index),
                "hypothesis": explored["hypothesis"],
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
    del experiment_model
    if not any(experiment["tasks"] for experiment in experiments):
        raise ValueError("exploration produced no witness-verified tasks for this research group")

    assert_disjoint_scenes(experiments, selection_tasks, reward_tasks)
    service_model = ModelSession(config)
    for experiment in experiments:
        experiment["diagnostic_results"] = evaluate_tasks(
            config, parent, service_model, experiment["tasks"]
        ) if experiment["tasks"] else []
        experiment["diagnostic_report"] = score_report(
            experiment["tasks"], experiment["diagnostic_results"]
        )
    parent_selection = evaluate_tasks(config, parent, service_model, selection_tasks)
    parent_reward = evaluate_tasks(config, parent, service_model, reward_tasks)
    parent_selection_score = mean_score(parent_selection)
    parent_reward_score = mean_score(parent_reward)
    parent_reward_task_score = mean_task_reward(parent_reward)
    del service_model

    edit_model = ModelSession(config, adapter_path=edit_adapter)
    edit_candidates = []
    for experiment in experiments:
        if not experiment["tasks"]:
            continue
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
        reward_delta = reward_task_score - parent_reward_task_score
        gains = visible_gains if item["branch"] == "with_evidence" else masked_gains
        gains[item["experiment_id"]].append(reward_delta)
        if item["branch"] == "with_evidence":
            visible_selection.append((selection_score, reward_score, candidate_id, candidate))
        for set_name, tasks, results, baseline in (
            ("selection", selection_tasks, selection_results, parent_selection),
            ("reward", reward_tasks, reward_results, parent_reward),
        ):
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
                },
                model_revision=generation["model_revision"],
                adapter_version=generation["adapter_version"],
                sampling_parameters=generation["sampling_parameters"],
                training_eligible=True,
            )))
        experiment_metrics.append({
            "experiment_id": experiment_id,
            "hypothesis": experiment["hypothesis"],
            "game_file": experiment["game_file"],
            "task_ids": [task.task_id for task in experiment["tasks"]],
            "diagnostic_report": experiment["diagnostic_report"],
            "valid": bool(experiment["tasks"]),
            "with_evidence_gain": sum(visible_gains[experiment_id]) / len(visible_gains[experiment_id]) if visible_gains[experiment_id] else None,
            "without_evidence_gain": sum(masked_gains[experiment_id]) / len(masked_gains[experiment_id]) if masked_gains[experiment_id] else None,
            "experiment_utility": utility,
        })
        episode_records.extend(experiment["diagnostic_results"])

    accepted = parent
    selected_candidate_id = None
    accepted_selection_score = parent_selection_score
    accepted_reward_score = parent_reward_score
    eligible = [item for item in visible_selection if item[3] is not None]
    if eligible:
        selected_index = choose_candidate(parent_selection_score, [item[0] for item in eligible])
        if selected_index is not None:
            accepted_selection_score, accepted_reward_score, candidate_id, candidate = eligible[selected_index]
            accepted = candidate
            selected_candidate_id = candidate_id

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
        "valid_task_generation_rate": verified_task_count / task_candidate_count,
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
        "accepted_candidate": selected_candidate_id,
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


def run_cycles(config, selection_pool_path, reward_pool_path, periods, method, out_dir,
               parent_path, test_path=None, experiment_adapter=None, edit_adapter=None):
    from .train import read_samples, train_grpo, train_opsd

    if method not in ("rl", "distill", "combined", "designer-only"):
        raise ValueError("method must be rl, distill, combined, or designer-only")
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
    used_experiment_scenes = set()
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
            excluded_scenes=development_scenes | used_experiment_scenes,
        )

        experiment_records = json.loads((period_dir / "experiments.json").read_text(encoding="utf-8"))
        period_experiment_scenes = {
            source_scene_id(item["game_file"]) for item in experiment_records
        }
        if period_experiment_scenes & (development_scenes | used_experiment_scenes):
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
            swanlab.log(
                {
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
                },
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
                "experiment_adapter_before": experiment_adapter_before,
                "experiment_adapter": current_experiment_adapter,
                "edit_adapter": current_edit_adapter,
            }
        )

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
