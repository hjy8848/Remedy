"""Train the Solver on Designer-generated, planner-verified ALFWorld tasks."""

import argparse
import json
import multiprocessing as mp
import random
import re
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from experiment_rsi.alfworld import AlfworldEpisode, list_game_files
from experiment_rsi.config import load_config
from experiment_rsi.constructed import _parse_spec
from experiment_rsi.evaluation import run_episode
from experiment_rsi.model import ModelSession, seed_everything
from experiment_rsi.records import SyntheticTask
from experiment_rsi.scene_builder import build_scene, inspect_scene
from experiment_rsi.solver_sft import collect_solver_sft_examples, train_solver_sft
from experiment_rsi.synthesis import (
    goal_reached, pddl_goal_expression, replay_witness, source_scene_id,
    supported_placement_game, write_tasks,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def log(message):
    print(message, flush=True)


def grouped_games(config, split):
    groups = {}
    for path in list_game_files(config["alfworld"]["config_path"], split):
        if supported_placement_game(path):
            groups.setdefault(source_scene_id(path), []).append(path)
    return {scene: sorted(paths)[0] for scene, paths in groups.items()}


def make_manifest(config, periods, diagnostics, task_count, id_count, ood_count):
    if min(periods, diagnostics, task_count, id_count, ood_count) < 1:
        raise ValueError("period, diagnostic, task, ID-test, and OOD-test counts must be positive")
    rng = random.Random(config["run"]["seed"])
    seen = grouped_games(config, "eval_in_distribution")
    unseen = grouped_games(config, "eval_out_of_distribution")
    train = grouped_games(config, "train")
    unseen_scenes = sorted(unseen)
    rng.shuffle(unseen_scenes)
    ood = unseen_scenes[:ood_count]
    seen_scenes = sorted(set(seen) - set(ood))
    rng.shuffle(seen_scenes)
    if len(ood) < ood_count or len(seen_scenes) < id_count:
        raise ValueError("not enough distinct supported scenes for the requested ID/OOD test pools")
    id_test = seen_scenes[:id_count]
    reserved = set(id_test + ood)
    train_scenes = sorted(set(train) - reserved)
    rng.shuffle(train_scenes)
    scenes_per_period = diagnostics + task_count
    needed = periods * scenes_per_period
    if len(train_scenes) < needed:
        raise ValueError("need {} separate training scenes, found {}".format(needed, len(train_scenes)))
    rounds = []
    for period in range(periods):
        start = period * scenes_per_period
        diagnostic_scenes = train_scenes[start:start + diagnostics]
        task_scenes = train_scenes[start + diagnostics:start + scenes_per_period]
        rounds.append({
            "diagnostic": [{"scene": scene, "game": train[scene]} for scene in diagnostic_scenes],
            "tasks": [{"scene": scene, "game": train[scene]} for scene in task_scenes],
        })
    return {
        "id_test": [{"scene": scene, "game": seen[scene], "split": "eval_in_distribution"} for scene in id_test],
        "ood_test": [{"scene": scene, "game": unseen[scene], "split": "eval_out_of_distribution"} for scene in ood],
        "periods": rounds,
    }


_MODEL = None
_CONFIG = None


def init_worker(config, devices, adapter_path):
    global _MODEL, _CONFIG
    device = devices.get()
    _CONFIG = config
    _MODEL = ModelSession(config, adapter_path=adapter_path, device="cuda:{}".format(device))
    log("model ready on cuda:{} adapter={}".format(device, adapter_path or "base"))


@contextmanager
def worker_pool(config, gpus, adapter_path):
    context = mp.get_context("spawn")
    devices = context.Queue()
    for device in range(gpus):
        devices.put(device)
    try:
        with ProcessPoolExecutor(
            max_workers=gpus,
            mp_context=context,
            initializer=init_worker,
            initargs=(config, devices, str(adapter_path) if adapter_path else None),
        ) as executor:
            yield executor
    finally:
        devices.close()
        devices.join_thread()


def worker_generate(job):
    seed_everything(job["seed"], _MODEL.torch)
    result = _MODEL.generate(
        job["messages"],
        max_new_tokens=job["max_new_tokens"],
        temperature=_CONFIG["model"]["temperature"],
        top_p=_CONFIG["model"]["top_p"],
        use_adapter=False,
    )
    return result["generated_text"]


def worker_episode(job):
    config = dict(_CONFIG)
    config["alfworld"] = dict(_CONFIG["alfworld"])
    config["alfworld"]["split"] = job["split"]
    seed_everything(job["seed"], _MODEL.torch)
    result = run_episode(
        config,
        None,
        _MODEL,
        job["game"],
        task=job.get("task"),
        max_steps=config["run"]["max_steps"],
        seed=job["seed"],
        action_mode="direct",
    )
    result["source_scene_id"] = job["scene"]
    return result


def evaluate(executor, entries, seed):
    jobs = [{**entry, "seed": seed + index} for index, entry in enumerate(entries)]
    results = []
    for result in executor.map(worker_episode, jobs):
        results.append(result)
        log("evaluation {}/{}: {} success={} steps={} error={}".format(
            len(results), len(jobs), result["source_scene_id"], result["success"],
            result["steps"], result["candidate_error"]))
    success = sum(result["success"] for result in results)
    return {
        "success_count": success,
        "task_count": len(results),
        "success_rate": success / len(results),
        "results": results,
    }


def designer_task_messages(game_file, feedback):
    game = json.loads(Path(game_file).read_text(encoding="utf-8"))
    scene = inspect_scene(game["pddl_problem"])
    containments = sorted({(args[0].removesuffix("type"), args[1].removesuffix("type"))
                           for predicate, args in scene["facts"] if predicate == "cancontain"
                           and len(args) == 2 and args[0].removesuffix("type") in scene["receptacles"]
                           and args[1].removesuffix("type") in scene["objects"]})
    return [
        {"role": "system", "content": (
            "Design an ALFWorld placement training task that targets a concrete weakness in the "
            "supplied Solver diagnosis. Return JSON with hypothesis, diagnostic_behavior, "
            "feedback_refs, base_scene, objects, and goal. Use the provided base_scene. "
            "objects is a list of one to three distinct object types; each entry has type, count "
            "(1 to 3), and initial_receptacle. goal has type 'place_all' and a placements list "
            "with exactly one entry per object type; each entry has object_type, count, and "
            "destination_type. Counts must match objects. Cite at least one supplied feedback "
            "reference and make the task exercise the cited behavior. Use only listed object, "
            "receptacle, and containment types. Do not make the task artificially easy; use multiple "
            "objects, destinations, or container interactions when they fit the diagnosed weakness. "
            "Do not write PDDL. Fast Downward and ALFWorld replay will verify the proposal.")},
        {"role": "user", "content": json.dumps({
            "base_scene": source_scene_id(game_file),
            "solver_feedback": feedback,
            "pickupable_object_types": sorted(scene["objects"]),
            "receptacle_counts": {key: len(value) for key, value in scene["receptacles"].items()},
            "can_contain": [{"receptacle": receptacle, "object": object_type}
                            for receptacle, object_type in containments],
        }, ensure_ascii=False)},
    ]


def _planner_worker(config, scene, result_queue):
    with AlfworldEpisode(config["alfworld"]["config_path"], game_file=scene["game_file"], split="train") as env:
        initial = env.reset()
        if goal_reached(scene["goal_atoms"], initial["facts"]):
            result_queue.put(("initial_goal_satisfied", []))
            return
        actions = env.env._pddl_state.replan(env.env._entity_infos)
        result_queue.put(("complete", actions))


def plan_task(config, scene, spec, experiment_id):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_planner_worker, args=(config, scene, result_queue))
    process.start()
    process.join(20)
    if process.is_alive():
        process.terminate()
        process.join()
        raise ValueError("planner exceeded the 20-second search budget; solvability is unknown")
    if process.exitcode:
        raise RuntimeError("planner process failed with exit code {}".format(process.exitcode))
    status, actions = result_queue.get(timeout=2)
    result_queue.close()
    if status == "initial_goal_satisfied":
        raise ValueError("goal already satisfied at reset")
    if not actions:
        raise ValueError("planner did not find a plan")
    if len(actions) > config["run"]["max_steps"]:
        raise ValueError("planner solution exceeds the Solver step budget")
    task = SyntheticTask(
        task_id=scene["task_id"], source_game_file=scene["source_game_file"],
        task_game_file=scene["game_file"], source_scene_id=scene["source_scene_id"],
        instruction=scene["instruction"], goal_kind="place_all", goal_atoms=scene["goal_atoms"],
        source_goal_text=scene["source_goal_text"], custom_goal_text=pddl_goal_expression(scene["goal_atoms"]),
        witness_actions=actions, source_split="train", experiment_id=experiment_id,
        hypothesis_id=experiment_id + "/hypothesis", hypothesis=spec["hypothesis"],
        feedback_refs=spec["feedback_refs"], diagnostic_target={
            "placements": [
                {"object": item["object_type"], "count": item["count"],
                 "receptacle": item["destination_type"]}
                for item in scene["placements"]
            ]
        },
        generation_mode="constructed", experiment_spec=spec,
    )
    replay = replay_witness(config, task)
    if not replay["valid"]:
        raise ValueError("planner solution failed real ALFWorld replay")
    return task, replay


def design_tasks(executor, config, feedback, entries, output, period, attempts):
    pending = list(range(len(entries)))
    accepted = {}
    revision_feedback = {}
    records = []
    for attempt in range(attempts):
        if not pending:
            break
        jobs = []
        for index in pending:
            messages = designer_task_messages(entries[index]["game"], feedback)
            if index in revision_feedback:
                messages.append({"role": "user", "content": revision_feedback[index]})
            jobs.append({
                "messages": messages,
                "seed": config["run"]["seed"] + period * 1000 + attempt * 100 + index,
                "max_new_tokens": 768,
            })
        outputs = list(executor.map(worker_generate, jobs))
        remaining = []
        attempt_records = []
        for index, job, output_text in zip(pending, jobs, outputs):
            experiment_id = "period-{:02d}/task-{:02d}".format(period, index + 1)
            record = {
                "experiment_id": experiment_id,
                "source_scene_id": entries[index]["scene"],
                "attempt": attempt + 1,
                "prompt": job["messages"],
                "output": output_text,
            }
            try:
                spec = _parse_spec(output_text, feedback)
                scene = build_scene(entries[index]["game"], spec, output / "task_games")
                task, replay = plan_task(config, scene, spec, experiment_id)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
                revision_feedback[index] = (
                    "The previous proposal was rejected: {}. Correct the task specification and "
                    "keep it tied to a cited Solver failure."
                ).format(error)
                record["rejection_reason"] = str(error)
                remaining.append(index)
            else:
                record.update({
                    "spec": spec,
                    "task": asdict(task),
                    "witness_replay": replay,
                    "witness_source": "host_fast_downward",
                })
                record["rejection_reason"] = None
                accepted[index] = task
            attempt_records.append(record)
        records.extend(attempt_records)
        for record in attempt_records:
            log("period {} {} attempt {}: {}".format(
                period, record["experiment_id"], record["attempt"],
                record.get("rejection_reason") or "planner and replay verified"))
        write_jsonl(output / "task_attempts.jsonl", records)
        pending = remaining
    tasks = [accepted[index] for index in sorted(accepted)]
    write_tasks(output / "tasks.jsonl", tasks)
    if len(tasks) != len(entries):
        raise RuntimeError(
            "Designer produced {} planner-verified tasks out of {}; see task_attempts.jsonl".format(
                len(tasks), len(entries)
            )
        )
    return tasks


def _short(value, limit):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def summarize_diagnostics(results, diagnostic_id, adapter_path):
    cases = []
    for index, result in enumerate(results):
        trace = result["trace"]
        seen = set()
        repeated = 0
        for step in trace:
            if step["action"] in seen and step.get("stalled"):
                repeated += 1
            seen.add(step["action"])
        navigation_stalls = sum(
            step["action"].lower().startswith("go to ") and step.get("stalled") for step in trace
        )
        categories = []
        if result["candidate_error"]:
            categories.append("invalid_or_failed_action")
        if repeated:
            categories.append("repeated_action")
        if navigation_stalls:
            categories.append("stalled_navigation")
        if not result["success"]:
            categories.append("task_incomplete")
        cases.append({
            "ref": "{}/episode-{:03d}".format(diagnostic_id, index),
            "scene_id": result["source_scene_id"],
            "objective": _short(result.get("initial_observation"), 180),
            "success": bool(result["success"]),
            "goal_progress": result.get("goal_progress"),
            "steps": result["steps"],
            "categories": categories,
            "candidate_error": _short(result.get("candidate_error"), 160),
            "repeated_action_count": repeated,
            "stalled_navigation_count": navigation_stalls,
            "trace_tail": [{
                "action": _short(step["action"], 100),
                "observation": _short(step["observation"], 180),
                "goal_progress": step.get("goal_progress"),
                "stalled": bool(step.get("stalled")),
            } for step in trace[-4:]],
        })
    failures = [case for case in cases if not case["success"]]
    references = (failures or cases)[:4]
    reference_ids = [case["ref"] for case in references] or [diagnostic_id + "/aggregate"]
    category_counts = {}
    for case in cases:
        for category in case["categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "diagnostic_id": diagnostic_id,
        "solver_adapter": str(adapter_path) if adapter_path else "base",
        "task_count": len(cases),
        "success_count": len(cases) - len(failures),
        "failure_count": len(failures),
        "success_rate": (len(cases) - len(failures)) / len(cases),
        "category_counts": category_counts,
        "cases": references,
        "reference_ids": reference_ids,
        "required_reference_ids": reference_ids,
    }


def prepare_period(executor, config, adapter_path, allocation, output, period, attempts):
    period_dir = output / "period-{:02d}".format(period)
    period_dir.mkdir()
    write_json(period_dir / "solver_parent.json", {"adapter": str(adapter_path) if adapter_path else "base"})
    diagnostic_entries = [{**entry, "split": "train"} for entry in allocation["diagnostic"]]
    diagnostic = evaluate(executor, diagnostic_entries, 50000 + period * 1000)
    write_jsonl(period_dir / "diagnostic_episodes.jsonl", diagnostic["results"])
    feedback = summarize_diagnostics(
        diagnostic["results"], "period-{:02d}-diagnostic".format(period), adapter_path
    )
    write_json(period_dir / "diagnostic_feedback.json", feedback)
    tasks = design_tasks(
        executor, config, feedback, allocation["tasks"], period_dir, period, attempts
    )
    examples = collect_solver_sft_examples(config, tasks)
    write_jsonl(period_dir / "solver_sft.jsonl", examples)
    write_json(period_dir / "period_input.json", {
        "solver_adapter": str(adapter_path) if adapter_path else "base",
        "diagnostics": feedback,
        "verified_task_count": len(tasks),
        "sft_action_count": len(examples),
        "task_ids": [task.task_id for task in tasks],
    })
    return period_dir, tasks, examples, diagnostic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--periods", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--diagnostics", type=int, default=10)
    parser.add_argument("--id-test", type=int, default=8)
    parser.add_argument("--ood-test", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--manifest")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.attempts < 1 or args.gpus < 1:
        raise ValueError("attempts and gpus must be positive")
    config = load_config(args.config)
    config["model"].update({"path": args.model, "revision": "local"})
    manifest = (json.loads(Path(args.manifest).read_text(encoding="utf-8")) if args.manifest else
                make_manifest(config, args.periods, args.diagnostics, args.tasks,
                              args.id_test, args.ood_test))
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    write_json(output / "contract.json", {
        "arguments": vars(args),
        "config": config,
        "method": "failure_conditioned_designer_planner_supervised_solver_sft",
        "designer": {"weights_updated": False, "uses_solver_adapter": False},
        "solver": {"base_model": args.model, "training": "LoRA SFT from planner-replayed actions"},
        "planner": "TextWorld Fast Downward plus ALFWorld replay",
        "selection_pool": False,
        "editor_skills_policy": False,
        "sft_prompt_schema": ["task", "observation", "recent_actions", "allowed_commands", "remaining_steps"],
    })
    if args.prepare_only:
        log("scene allocation complete: {}".format(output))
        return

    with worker_pool(config, args.gpus, None) as executor:
        log("evaluating initial Solver on fixed ID/OOD test sets")
        initial_id = evaluate(executor, manifest["id_test"], 910000)
        initial_ood = evaluate(executor, manifest["ood_test"], 920000)
        write_json(output / "initial_test.json", {"id": initial_id, "ood": initial_ood})
        first_dir, tasks, examples, diagnostic = prepare_period(
            executor, config, None, manifest["periods"][0], output, 1, args.attempts
        )

    period_summaries = []
    current_adapter = None
    parent_id, parent_ood = initial_id, initial_ood
    prepared = (first_dir, tasks, examples, diagnostic)
    for period_index, allocation in enumerate(manifest["periods"]):
        period = period_index + 1
        period_dir, tasks, examples, diagnostic = prepared
        adapter_dir = period_dir / "solver_adapter"
        log("period {}: training Solver LoRA on {} planner actions from {} tasks".format(
            period, len(examples), len(tasks)))
        train_solver_sft(
            config,
            period_dir / "solver_sft.jsonl",
            adapter_dir,
            parent_adapter=current_adapter,
            device=args.train_device,
        )
        current_adapter = adapter_dir
        with worker_pool(config, args.gpus, current_adapter) as executor:
            id_test = evaluate(executor, manifest["id_test"], 910000)
            ood_test = evaluate(executor, manifest["ood_test"], 920000)
            write_json(period_dir / "test_results.json", {"id": id_test, "ood": ood_test})
            summary = {
                "period": period,
                "diagnostic_success_rate": diagnostic["success_rate"],
                "verified_tasks": len(tasks),
                "sft_action_count": len(examples),
                "solver_adapter": str(current_adapter),
                "id_success_rate": id_test["success_rate"],
                "ood_success_rate": ood_test["success_rate"],
                "id_delta_from_parent": id_test["success_rate"] - parent_id["success_rate"],
                "ood_delta_from_parent": ood_test["success_rate"] - parent_ood["success_rate"],
                "id_delta_from_initial": id_test["success_rate"] - initial_id["success_rate"],
                "ood_delta_from_initial": ood_test["success_rate"] - initial_ood["success_rate"],
            }
            parent_id, parent_ood = id_test, ood_test
            period_summaries.append(summary)
            write_json(period_dir / "summary.json", summary)
            log("period {} complete: ID {:.3f} ({:+.3f}), OOD {:.3f} ({:+.3f})".format(
                period, summary["id_success_rate"], summary["id_delta_from_initial"],
                summary["ood_success_rate"], summary["ood_delta_from_initial"]))
            if period < len(manifest["periods"]):
                prepared = prepare_period(
                    executor, config, current_adapter, manifest["periods"][period],
                    output, period + 1, args.attempts,
                )

    final_id = json.loads((output / "period-{:02d}".format(len(period_summaries)) /
                           "test_results.json").read_text(encoding="utf-8"))["id"]
    final_ood = json.loads((output / "period-{:02d}".format(len(period_summaries)) /
                            "test_results.json").read_text(encoding="utf-8"))["ood"]
    result = {
        "initial_id": initial_id,
        "final_id": final_id,
        "initial_ood": initial_ood,
        "final_ood": final_ood,
        "id_success_rate_delta": final_id["success_rate"] - initial_id["success_rate"],
        "ood_success_rate_delta": final_ood["success_rate"] - initial_ood["success_rate"],
        "periods": period_summaries,
        "final_solver_adapter": str(current_adapter),
    }
    write_json(output / "result.json", result)
    log("completed: ID {:.3f} -> {:.3f}; OOD {:.3f} -> {:.3f}".format(
        initial_id["success_rate"], final_id["success_rate"],
        initial_ood["success_rate"], final_ood["success_rate"]))


if __name__ == "__main__":
    main()
