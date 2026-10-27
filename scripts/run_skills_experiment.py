"""Run the frozen-model Remedy skills experiment on ALFWorld."""

import argparse
import copy
import json
import multiprocessing as mp
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from experiment_rsi.alfworld import AlfworldEpisode, list_game_files
from experiment_rsi.config import load_config
from experiment_rsi.constructed import _parse_spec
from experiment_rsi.evaluation import run_episode
from experiment_rsi.model import ModelSession, seed_everything
from experiment_rsi.records import SyntheticTask
from experiment_rsi.research import summarize_diagnostic_results
from experiment_rsi.scene_builder import build_scene, inspect_scene
from experiment_rsi.synthesis import (
    goal_reached, pddl_goal_expression, read_tasks, replay_witness, source_scene_id,
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


def make_manifest(config, periods, diagnostics, experiments, selection_count, id_count, ood_count):
    rng = random.Random(config["run"]["seed"])
    seen = grouped_games(config, "eval_in_distribution")
    unseen = grouped_games(config, "eval_out_of_distribution")
    train = grouped_games(config, "train")
    unseen_scenes = sorted(unseen)
    rng.shuffle(unseen_scenes)
    ood = unseen_scenes[:ood_count]
    seen_scenes = sorted(set(seen) - set(ood))
    rng.shuffle(seen_scenes)
    if len(ood) < ood_count or len(seen_scenes) < selection_count + id_count:
        raise ValueError("not enough distinct supported scenes for the requested evaluation pools")
    selection = seen_scenes[:selection_count]
    id_test = seen_scenes[selection_count:selection_count + id_count]
    reserved = set(selection + id_test + ood)
    train_scenes = sorted(set(train) - reserved)
    rng.shuffle(train_scenes)
    needed = periods * (diagnostics + experiments)
    if len(train_scenes) < needed:
        raise ValueError("need {} separate training scenes, found {}".format(needed, len(train_scenes)))
    rounds = []
    for period in range(periods):
        start = period * (diagnostics + experiments)
        diagnostic_scenes = train_scenes[start:start + diagnostics]
        experiment_scenes = train_scenes[start + diagnostics:start + diagnostics + experiments]
        rounds.append({
            "diagnostic": [{"scene": scene, "game": train[scene]} for scene in diagnostic_scenes],
            "experiment": [{"scene": scene, "game": train[scene]} for scene in experiment_scenes],
        })
    return {
        "selection": [{"scene": scene, "game": seen[scene], "split": "eval_in_distribution"} for scene in selection],
        "id_test": [{"scene": scene, "game": seen[scene], "split": "eval_in_distribution"} for scene in id_test],
        "ood_test": [{"scene": scene, "game": unseen[scene], "split": "eval_out_of_distribution"} for scene in ood],
        "periods": rounds,
    }


_MODEL = None
_CONFIG = None


def init_worker(config, devices):
    global _MODEL, _CONFIG
    device = devices.get()
    _CONFIG = config
    _MODEL = ModelSession(config, device="cuda:{}".format(device))
    log("model ready on cuda:{}".format(device))


def worker_generate(job):
    seed_everything(job["seed"], _MODEL.torch)
    result = _MODEL.generate(
        job["messages"], max_new_tokens=job["max_new_tokens"],
        temperature=_CONFIG["model"]["temperature"], top_p=_CONFIG["model"]["top_p"],
    )
    return result["generated_text"]


def worker_episode(job):
    config = copy.deepcopy(_CONFIG)
    config["alfworld"]["split"] = job["split"]
    seed_everything(job["seed"], _MODEL.torch)
    result = run_episode(
        config, job["service"], _MODEL, job["game"], task=job.get("task"),
        max_steps=config["run"]["max_steps"], seed=job["seed"],
        action_mode=job["action_mode"],
    )
    result["source_scene_id"] = job["scene"]
    return result


def evaluate(executor, service, entries, seed):
    jobs = [{**entry, "service": service, "seed": seed + i, "action_mode": "direct"}
            for i, entry in enumerate(entries)]
    results = []
    for result in executor.map(worker_episode, jobs):
        results.append(result)
        log("evaluation {}/{}: {} success={} steps={} error={}".format(
            len(results), len(jobs), result["source_scene_id"], result["success"],
            result["steps"], result["candidate_error"]))
    success = sum(result["success"] for result in results)
    return {"success_count": success, "task_count": len(results),
            "success_rate": success / len(results),
            "model_action_calls": sum(step["policy_kind"] == "model" for result in results for step in result["trace"]),
            "results": results}


def skill_task_messages(game_file, service, feedback):
    game = json.loads(Path(game_file).read_text(encoding="utf-8"))
    scene = inspect_scene(game["pddl_problem"])
    containments = sorted({(args[0].removesuffix("type"), args[1].removesuffix("type"))
                           for predicate, args in scene["facts"] if predicate == "cancontain"
                           and len(args) == 2 and args[0].removesuffix("type") in scene["receptacles"]
                           and args[1].removesuffix("type") in scene["objects"]})
    return [
        {"role": "system", "content": (
            "Design an ALFWorld placement experiment that directly challenges a failure in the "
            "supplied Solver feedback. Return JSON with hypothesis, diagnostic_behavior, "
            "feedback_refs, base_scene, objects, and goal. Use the provided base_scene. "
            "objects is a list of one to three distinct object types; each entry has type, count "
            "(1 to 3), and initial_receptacle. goal has type 'place_all' and a placements list "
            "with exactly one entry per object type; each entry has object_type, count, and "
            "destination_type. Counts must match objects. Cite at least one supplied feedback "
            "reference. Use only listed object, receptacle, "
            "and containment types. Make the task challenging enough to expose the cited Solver "
            "weakness; do not favor one-object, shortest-path, or otherwise easy tasks. Use multiple "
            "objects, goals, or container interactions when they test that weakness. Do not write PDDL; "
            "the host planner and ALFWorld replay will validate the task.")},
        {"role": "user", "content": json.dumps({
            "base_scene": source_scene_id(game_file),
            "current_knowledge": service["knowledge"],
            "research_feedback": feedback,
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


def _compact_solver_result(result):
    return {
        "success": result["success"],
        "candidate_error": result["candidate_error"],
        "steps": result["steps"],
        "goal_progress": result["goal_progress"],
        "trace": [{"action": step["action"], "observation": step["observation"],
                   "goal_progress": step["goal_progress"]} for step in result["trace"]],
    }


def design_tasks(executor, config, service, feedback, entries, output, period, attempts):
    pending = list(range(len(entries)))
    accepted = {}
    accepted_results = {}
    records = []
    revision_feedback = {}
    for attempt in range(attempts):
        if not pending:
            break
        jobs = []
        for index in pending:
            messages = skill_task_messages(entries[index]["game"], service, feedback)
            if index in revision_feedback:
                messages.append({"role": "user", "content": revision_feedback[index]})
            jobs.append({"messages": messages, "seed": config["run"]["seed"] + period * 1000 + attempt * 100 + index,
                         "max_new_tokens": 768})
        outputs = list(executor.map(worker_generate, jobs))
        remaining = []
        proposed = []
        attempt_records = []
        for index, job, output_text in zip(pending, jobs, outputs):
            experiment_id = "period-{:02d}/experiment-{:02d}".format(period, index + 1)
            record = {"experiment_id": experiment_id, "source_scene_id": entries[index]["scene"],
                      "attempt": attempt + 1, "prompt": job["messages"], "output": output_text}
            try:
                spec = _parse_spec(output_text, feedback)
                scene = build_scene(entries[index]["game"], spec, output / "task_games")
                task, replay = plan_task(config, scene, spec, experiment_id)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
                revision_feedback[index] = (
                    "The previous proposal was rejected: {}. Correct the task specification, "
                    "while keeping it challenging and tied to the diagnostic behavior."
                ).format(error)
                record["rejection_reason"] = str(error)
                remaining.append(index)
                attempt_records.append(record)
            else:
                record.update({"spec": spec, "task": asdict(task), "witness_replay": replay,
                               "witness_source": "host_fast_downward"})
                proposed.append((index, task, record))
        if proposed:
            attempt_entries = [{"scene": task.source_scene_id, "game": task.task_game_file,
                                "split": task.source_split, "task": task}
                               for _, task, _ in proposed]
            attempt_results = evaluate(
                executor, service, attempt_entries,
                60000 + period * 10000 + attempt * 1000,
            )
            for (index, task, record), result in zip(proposed, attempt_results["results"]):
                record["solver_attempt"] = _compact_solver_result(result)
                if result["candidate_error"]:
                    record["rejection_reason"] = "solver_execution_error"
                    attempt_records.append(record)
                    records.extend(attempt_records)
                    write_jsonl(output / "task_attempts.jsonl", records)
                    raise RuntimeError(
                        "Solver execution failed while validating a proposed challenge: {}".format(
                            result["candidate_error"]
                        )
                    )
                if result["success"]:
                    record["rejection_reason"] = "baseline_solver_solved_task"
                    revision_feedback[index] = (
                        "The unchanged Solver solved your previous planner-verified task in {} "
                        "steps. Propose a materially more challenging task that still tests the "
                        "same diagnostic behavior; do not merely rename objects or locations."
                    ).format(result["steps"])
                    remaining.append(index)
                else:
                    record["rejection_reason"] = None
                    accepted[index] = task
                    accepted_results[index] = result
                attempt_records.append(record)
        records.extend(attempt_records)
        for record in attempt_records:
            log("period {} task {} attempt {}: {}".format(
                period, record["experiment_id"], attempt + 1,
                record.get("rejection_reason", "verified failure challenge")))
        write_jsonl(output / "task_attempts.jsonl", records)
        pending = remaining
    for index in pending:
        experiment_id = "period-{:02d}/experiment-{:02d}".format(period, index + 1)
        for record in reversed(records):
            if record["experiment_id"] == experiment_id:
                record["final_status"] = "no_solver_failure_found_after_attempts"
                break
    write_jsonl(output / "task_attempts.jsonl", records)
    tasks = [accepted[index] for index in sorted(accepted)]
    results = [accepted_results[index] for index in sorted(accepted)]
    success_count = sum(result["success"] for result in results)
    return tasks, {
        "success_count": success_count,
        "task_count": len(results),
        "success_rate": success_count / len(results) if results else None,
        "model_action_calls": sum(
            step["policy_kind"] == "model" for result in results for step in result["trace"]
        ),
        "results": results,
    }


def edit_messages(service, task, result, feedback):
    trace = [{"action": step["action"], "observation": step["observation"],
              "goal_progress": step["goal_progress"]} for step in result["trace"]]
    return [
        {"role": "system", "content": (
            "The frozen Solver failed this verified ALFWorld task. Reflect on the specific error by "
            "comparing its failed trajectory with the planner's valid solution. Return exactly one "
            "JSON object with one key, skill, whose value is one concise, reusable English instruction "
            "that would help prevent this failure. Ground it in the observed failure; do not give a "
            "generic task plan, scene-specific object names, numbered locations, or a list of rules.")},
        {"role": "user", "content": json.dumps({
            "current_skills": service["knowledge"],
            "diagnostic_feedback": feedback, "experiment_hypothesis": task.hypothesis,
            "task": task.instruction, "solver_success": result["success"],
            "solver_error": result["candidate_error"], "solver_trace": trace,
            "verified_plan": task.witness_actions,
        }, ensure_ascii=False)},
    ]


def parse_skill(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    data = json.loads(text)
    if set(data) != {"skill"} or not isinstance(data["skill"], str) or not data["skill"].strip():
        raise ValueError("Editor must return one nonempty skill string")
    return data["skill"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--experiments", type=int, default=4)
    parser.add_argument("--diagnostics", type=int, default=10)
    parser.add_argument("--selection", type=int, default=8)
    parser.add_argument("--id-test", type=int, default=8)
    parser.add_argument("--ood-test", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--manifest")
    parser.add_argument("--resume-from")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config["model"].update({"path": args.model, "revision": "local"})
    manifest = (json.loads(Path(args.manifest).read_text(encoding="utf-8")) if args.manifest else
                make_manifest(config, args.periods, args.diagnostics, args.experiments,
                              args.selection, args.id_test, args.ood_test))
    previous = Path(args.resume_from).resolve() if args.resume_from else None
    if previous:
        previous_contract = json.loads((previous / "contract.json").read_text(encoding="utf-8"))
        if (previous_contract.get("solver_action_mode") != "direct_full_admissible_commands"
                or previous_contract.get("task_protocol") != "failure_gated_multi_goal_v1"):
            raise ValueError("cannot resume a run with a different Solver or task-generation protocol")
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.resume_from:
        for name in ("initial_service.json", "initial_test.json", "initial_selection.json"):
            shutil.copy2(previous / name, output / name)
        old_period = previous / "period-01"
        resumed_period = output / "period-01"
        shutil.copytree(old_period, resumed_period)
        prior_candidates = resumed_period / "editor_candidates.json"
        if prior_candidates.exists():
            prior_candidates.rename(resumed_period / "editor_candidates-first-attempt.json")
        resumed_tasks = read_tasks(resumed_period / "tasks.jsonl")
        for task in resumed_tasks:
            task.task_game_file = str(resumed_period / Path(task.task_game_file).relative_to(old_period))
        write_tasks(resumed_period / "tasks.jsonl", resumed_tasks)
    write_json(output / "manifest.json", manifest)
    write_json(output / "contract.json", {"arguments": vars(args), "config": config,
        "method": "frozen_models_skills", "selection_policy": "strictly higher paired success rate",
        "planner": "TextWorld Fast Downward plus ALFWorld replay", "weights_updated": False,
        "solver_action_mode": "direct_full_admissible_commands",
        "task_protocol": "failure_gated_multi_goal_v1",
        "skill_trigger": "reflect only after a Solver task failure"})
    if args.prepare_only:
        log("scene allocation complete: {}".format(output))
        return
    if args.resume_from:
        service = json.loads((output / "initial_service.json").read_text(encoding="utf-8"))
    else:
        seed_service = json.loads(Path("seeds/service.json").read_text(encoding="utf-8"))
        service = {"knowledge": seed_service["knowledge"]}
    initial_service = copy.deepcopy(service)
    write_json(output / "initial_service.json", initial_service)
    context = mp.get_context("spawn")
    devices = context.Queue()
    for device in range(args.gpus):
        devices.put(device)
    with ProcessPoolExecutor(max_workers=args.gpus, mp_context=context,
                             initializer=init_worker, initargs=(config, devices)) as executor:
        if args.resume_from:
            initial_test = json.loads((output / "initial_test.json").read_text(encoding="utf-8"))
            initial_id, initial_ood = initial_test["id"], initial_test["ood"]
            parent_selection = json.loads((output / "initial_selection.json").read_text(encoding="utf-8"))
            log("reusing completed initial tests, diagnosis, and verified first-period tasks")
        else:
            log("evaluating initial ID/OOD tests")
            initial_id = evaluate(executor, service, manifest["id_test"], 910000)
            initial_ood = evaluate(executor, service, manifest["ood_test"], 920000)
            write_json(output / "initial_test.json", {"id": initial_id, "ood": initial_ood})
            parent_selection = evaluate(executor, service, manifest["selection"], 900000)
            write_json(output / "initial_selection.json", parent_selection)
        history = []
        period_summaries = []
        for index, allocation in enumerate(manifest["periods"]):
            period = index + 1
            period_dir = output / "period-{:02d}".format(period)
            period_dir.mkdir(exist_ok=bool(args.resume_from and period == 1))
            service_path = period_dir / "parent.json"
            write_json(service_path, service)
            if args.resume_from and period == 1:
                diagnostic_summary = json.loads((period_dir / "diagnostic_summary.json").read_text(encoding="utf-8"))
                diagnostic = {"success_rate": diagnostic_summary["success_count"] / diagnostic_summary["task_count"]}
                feedback = json.loads((period_dir / "research_feedback.json").read_text(encoding="utf-8"))
                tasks = read_tasks(period_dir / "tasks.jsonl")
                task_results = json.loads((period_dir / "solver_experiments.json").read_text(encoding="utf-8"))
                if len(task_results["results"]) != len(tasks):
                    raise ValueError("the resumed period has mismatched tasks and Solver results")
            else:
                log("period {}: diagnosing {} scenes".format(period, len(allocation["diagnostic"])))
                diagnostic_entries = [{**row, "split": "train"} for row in allocation["diagnostic"]]
                diagnostic = evaluate(executor, service, diagnostic_entries, 50000 + period * 1000)
                diagnostic_id = "period-{:02d}-diagnostic".format(period)
                for i, result in enumerate(diagnostic["results"]):
                    result["episode_id"] = "{}-episode-{:03d}".format(diagnostic_id, i)
                write_jsonl(period_dir / "diagnostic_episodes.jsonl", diagnostic["results"])
                diagnostic_summary = summarize_diagnostic_results(diagnostic["results"], diagnostic_id, service_path)
                write_json(period_dir / "diagnostic_summary.json", diagnostic_summary)
                feedback = {"current_solver_diagnostics": diagnostic_summary, "prior_research": history[-2:],
                            "reference_ids": diagnostic_summary["reference_ids"],
                            "required_reference_ids": diagnostic_summary["reference_ids"]}
                write_json(period_dir / "research_feedback.json", feedback)
                tasks, task_results = design_tasks(
                    executor, config, service, feedback, allocation["experiment"],
                    period_dir, period, args.attempts,
                )
                write_tasks(period_dir / "tasks.jsonl", tasks)
                write_json(period_dir / "solver_experiments.json", task_results)
            candidates = []
            jobs = []
            for i, (task, result) in enumerate(zip(tasks, task_results["results"])):
                if result["success"]:
                    candidates.append({"experiment_id": task.experiment_id,
                                       "reflection_status": "skipped_solver_succeeded"})
                    continue
                if result["candidate_error"]:
                    candidates.append({"experiment_id": task.experiment_id,
                                       "reflection_status": "skipped_execution_error",
                                       "error": result["candidate_error"]})
                    continue
                job = {"messages": edit_messages(service, task, result, feedback),
                       "seed": 80000 + period * 1000 + i, "max_new_tokens": 768,
                       "task_index": i}
                jobs.append(job)
                candidates.append({"experiment_id": task.experiment_id,
                                   "task_id": task.task_id, "reflection_status": "pending"})
            outputs = list(executor.map(worker_generate, jobs))
            for job, text in zip(jobs, outputs):
                i = job["task_index"]
                row = candidates[i]
                row.update({"prompt": job["messages"], "output": text})
                try:
                    skill = parse_skill(text)
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    row["error"] = str(error)
                else:
                    row["skill"] = skill
                    if skill in service["knowledge"]:
                        row["reflection_status"] = "duplicate_skill"
                    else:
                        row["knowledge"] = service["knowledge"] + [skill]
                        row["reflection_status"] = "skill_proposed"
            write_json(period_dir / "editor_candidates.json", candidates)
            best = None
            scored_knowledge = {}
            log("period {}: evaluating {} failure-grounded skill proposals".format(
                period, sum("skill" in candidate for candidate in candidates)))
            for i, candidate in enumerate(candidates):
                if "knowledge" not in candidate:
                    continue
                if candidate["knowledge"] == service["knowledge"]:
                    candidate["no_change"] = True
                    continue
                proposed = {"knowledge": candidate["knowledge"]}
                task = tasks[i]
                task_entry = {"scene": task.source_scene_id, "game": task.task_game_file,
                              "split": task.source_split, "task": task}
                repair = evaluate(executor, proposed, [task_entry], 73000 + period * 1000 + i)
                candidate["triggering_task_repair"] = repair
                if not repair["results"][0]["success"]:
                    candidate["selection_skipped"] = "skill_did_not_repair_triggering_failure"
                    write_json(period_dir / "editor_candidates.json", candidates)
                    continue
                candidate["triggering_task_repaired"] = True
                knowledge_key = tuple(candidate["knowledge"])
                if knowledge_key not in scored_knowledge:
                    scored_knowledge[knowledge_key] = evaluate(executor, proposed, manifest["selection"], 900000)
                score = scored_knowledge[knowledge_key]
                candidate["selection"] = score
                candidate["selection_delta"] = score["success_rate"] - parent_selection["success_rate"]
                write_json(period_dir / "editor_candidates.json", candidates)
                if score["success_rate"] > parent_selection["success_rate"] and (best is None or score["success_rate"] > best["selection"]["success_rate"]):
                    best = candidate
            before = parent_selection["success_rate"]
            if best is not None:
                service = {"knowledge": best["knowledge"]}
                parent_selection = best["selection"]
            summary = {"period": period, "diagnostic_success_rate": diagnostic["success_rate"],
                       "verified_tasks": len(tasks), "baseline_solver_failures": len(tasks),
                       "skill_proposals": sum("skill" in candidate for candidate in candidates),
                       "triggering_task_repairs": sum(candidate.get("triggering_task_repaired", False)
                                                       for candidate in candidates),
                       "selection_before": before, "selection_after": parent_selection["success_rate"],
                       "accepted_experiment": best["experiment_id"] if best else None,
                       "knowledge": service["knowledge"]}
            write_json(period_dir / "service.json", service)
            write_json(period_dir / "summary.json", summary)
            history.append({"period": period, "summary": summary,
                            "hypotheses": [task.hypothesis for task in tasks],
                            "candidate_effects": [{"experiment_id": item["experiment_id"],
                                                   "selection_delta": item.get("selection_delta"),
                                                   "triggering_task_repaired": item.get("triggering_task_repaired"),
                                                   "error": item.get("error")} for item in candidates]})
            period_summaries.append(summary)
            log("period {} complete: {}".format(period, json.dumps(summary, ensure_ascii=False)))
        log("evaluating final ID/OOD tests")
        final_id = evaluate(executor, service, manifest["id_test"], 910000)
        final_ood = evaluate(executor, service, manifest["ood_test"], 920000)
        result = {"periods": period_summaries, "initial_id": initial_id, "final_id": final_id,
                  "initial_ood": initial_ood, "final_ood": final_ood,
                  "id_success_rate_delta": final_id["success_rate"] - initial_id["success_rate"],
                  "ood_success_rate_delta": final_ood["success_rate"] - initial_ood["success_rate"],
                  "initial_service": initial_service, "final_service": service}
        write_json(output / "result.json", result)
        log("completed: ID {:.3f} -> {:.3f}; OOD {:.3f} -> {:.3f}".format(
            initial_id["success_rate"], final_id["success_rate"], initial_ood["success_rate"], final_ood["success_rate"]))


if __name__ == "__main__":
    main()
