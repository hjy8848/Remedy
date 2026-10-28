"""Continue a seeded synthetic-task Solver SFT experiment without manual per-cycle steps."""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file

from experiment_rsi.config import load_config
from experiment_rsi.records import SyntheticTask
from experiment_rsi.solver_sft import train_solver_sft
from run_skills_experiment import (
    evaluate,
    grouped_games,
    log,
    prepare_period,
    worker_pool,
    write_json,
    write_jsonl,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def store_state(path, state):
    temporary = Path(str(path) + ".tmp")
    write_json(temporary, state)
    temporary.replace(path)


def make_manifest(config, seed_manifest, periods, diagnostics, tasks, id_dev_count):
    id_test = seed_manifest["id_test"]
    ood_test = seed_manifest["ood_test"]
    reserved = {entry["scene"] for entry in id_test + ood_test}
    reserved.update(
        entry["scene"]
        for period in seed_manifest["periods"]
        for pool in ("diagnostic", "tasks")
        for entry in period[pool]
    )
    seen = grouped_games(config, "eval_in_distribution")
    available_seen = sorted(set(seen) - reserved)
    rng = random.Random(config["run"]["seed"] + 91)
    rng.shuffle(available_seen)
    if len(available_seen) < id_dev_count:
        raise ValueError("not enough valid_seen scenes for a development set disjoint from final test")
    id_dev_scenes = available_seen[:id_dev_count]
    reserved.update(id_dev_scenes)
    train = grouped_games(config, "train")
    train_scenes = sorted(set(train) - reserved)
    rng.shuffle(train_scenes)
    needed = periods * diagnostics
    if len(train_scenes) < needed:
        raise ValueError("need {} distinct diagnostic training scenes, found {}".format(needed, len(train_scenes)))
    allocations = []
    for index in range(periods):
        selected = train_scenes[index * diagnostics:(index + 1) * diagnostics]
        allocations.append({
            "diagnostic": [{"scene": scene, "game": train[scene]} for scene in selected],
            "tasks": [{"scene": "synthetic-slot-{}-{}".format(index + 2, slot + 1)}
                      for slot in range(tasks)],
        })
    return {
        "id_dev": [{"scene": scene, "game": seen[scene], "split": "eval_in_distribution"}
                   for scene in id_dev_scenes],
        "id_test": id_test,
        "ood_test": ood_test,
        "periods": allocations,
    }


def load_prepared_period(period_dir):
    required = ["period_input.json", "diagnostic_feedback.json", "diagnostic_episodes.jsonl",
                "tasks.jsonl", "solver_sft.jsonl"]
    missing = [name for name in required if not (period_dir / name).is_file()]
    if missing:
        raise RuntimeError("incomplete saved period {}; missing {}".format(period_dir, ", ".join(missing)))
    feedback = read_json(period_dir / "diagnostic_feedback.json")
    diagnostic = {"success_rate": feedback["success_rate"],
                  "results": read_jsonl(period_dir / "diagnostic_episodes.jsonl")}
    tasks = [SyntheticTask(**item) for item in read_jsonl(period_dir / "tasks.jsonl")]
    examples = read_jsonl(period_dir / "solver_sft.jsonl")
    return tasks, examples, diagnostic


def assert_adapter_updated(parent_adapter, adapter_dir):
    parent = load_file(str(Path(parent_adapter) / "adapter_model.safetensors"))
    updated = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    if parent.keys() != updated.keys():
        raise RuntimeError("LoRA adapter tensor keys changed during continued training")
    changed = sum(not torch.equal(parent[name], updated[name]) for name in parent)
    if not changed:
        raise RuntimeError("continued SFT did not change any adapter tensor")
    nonzero_b = sum(
        int(torch.count_nonzero(updated[name]).item() > 0)
        for name in updated if "lora_B" in name
    )
    return changed, nonzero_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--logic-dir", required=True)
    parser.add_argument("--cycles", type=int, default=2, help="additional cycles after the supplied seed run")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--minimum-tasks", type=int, default=8)
    parser.add_argument("--diagnostics", type=int, default=10)
    parser.add_argument("--id-dev", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.cycles, args.tasks, args.minimum_tasks, args.diagnostics, args.id_dev, args.attempts, args.gpus) < 1:
        raise ValueError("cycle, task, diagnostic, evaluation, attempt, and GPU counts must be positive")
    if args.minimum_tasks > args.tasks:
        raise ValueError("minimum-tasks cannot exceed tasks")

    seed_run = Path(args.seed_run).resolve()
    seed_adapter = seed_run / "period-01" / "solver_adapter"
    seed_tasks_path = seed_run / "period-01" / "tasks.jsonl"
    seed_sft_path = seed_run / "period-01" / "solver_sft.jsonl"
    seed_manifest_path = seed_run / "manifest.json"
    seed_baseline_path = seed_run / "initial_test.json"
    for path in (seed_adapter / "adapter_model.safetensors", seed_tasks_path, seed_sft_path,
                 seed_manifest_path, seed_baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_config(args.config)
    config["model"].update({"path": args.model, "revision": "local"})
    seed_tasks = [SyntheticTask(**item) for item in read_jsonl(seed_tasks_path)]
    cumulative_examples = read_jsonl(seed_sft_path)
    seed_manifest = read_json(seed_manifest_path)
    seed_baseline = read_json(seed_baseline_path)
    output = Path(args.out).resolve()
    state_path = output / "state.json"
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError("cannot resume; output directory does not exist: {}".format(output))
        manifest = read_json(output / "manifest.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = make_manifest(config, seed_manifest, args.cycles, args.diagnostics,
                                 args.tasks, args.id_dev)
        write_json(output / "manifest.json", manifest)
        write_json(output / "baseline_test.json", {
            "source": str(seed_baseline_path),
            "id_success_count": seed_baseline["id"]["success_count"],
            "id_task_count": seed_baseline["id"]["task_count"],
            "ood_success_count": seed_baseline["ood"]["success_count"],
            "ood_task_count": seed_baseline["ood"]["task_count"],
            "note": "Base Solver scores were measured before round 1; this test pool was used by the earlier pilot and is held out from the remaining training cycles.",
        })
        write_json(output / "contract.json", {
            "experiment": "Remedy autonomous synthetic-task Solver SFT",
            "seed_run": str(seed_run), "seed_round": 1,
            "additional_cycles": args.cycles, "total_rounds": args.cycles + 1,
            "model": config["model"], "training": config["training"], "run": config["run"],
            "tasks_requested_per_cycle": args.tasks, "minimum_verified_tasks_per_cycle": args.minimum_tasks,
            "generation_attempts_per_slot": args.attempts,
            "diagnostic_tasks_per_cycle": args.diagnostics,
            "dev": "disjoint valid_seen scenes, used each cycle only for progress tracking",
            "final_test": "same held-out ID/OOD pool as seed run; evaluated only after the additional cycles",
            "method": "frozen Designer and base model; continue Solver LoRA SFT on cumulative planner-replayed synthetic-task demonstrations",
            "training_backend": "Transformers Trainer + PEFT",
            "execution": {"inference_gpus": args.gpus, "train_device": args.train_device},
        })
        store_state(state_path, {"stage": "initialized", "completed_round": 1,
                                 "current_adapter": str(seed_adapter), "verified_tasks": len(seed_tasks),
                                 "sft_examples": len(cumulative_examples)})

    if len(manifest["periods"]) != args.cycles:
        raise ValueError("saved manifest does not match requested additional cycle count")
    current_adapter = seed_adapter
    prior_tasks = list(seed_tasks)
    summaries = []

    dev_path = output / "dev_baseline.json"
    if not dev_path.is_file():
        with worker_pool(config, args.gpus, current_adapter) as executor:
            dev_baseline = evaluate(executor, manifest["id_dev"], 61000)
        write_json(dev_path, dev_baseline)
        store_state(state_path, {"stage": "seed_dev_complete", "completed_round": 1,
                                 "current_adapter": str(current_adapter),
                                 "verified_tasks": len(prior_tasks), "sft_examples": len(cumulative_examples)})
    dev_baseline = read_json(dev_path)

    for index, allocation in enumerate(manifest["periods"]):
        period = index + 2
        period_dir = output / "period-{:02d}".format(period)
        adapter_dir = period_dir / "solver_adapter"
        summary_path = period_dir / "summary.json"
        cumulative_path = period_dir / "cumulative_solver_sft.jsonl"
        if summary_path.is_file():
            summary = read_json(summary_path)
            period_tasks = [SyntheticTask(**item) for item in read_jsonl(period_dir / "tasks.jsonl")]
            prior_tasks.extend(period_tasks)
            cumulative_examples = read_jsonl(cumulative_path)
            current_adapter = Path(summary["solver_adapter"])
            summaries.append(summary)
            continue

        if not period_dir.exists():
            with worker_pool(config, args.gpus, current_adapter) as executor:
                period_dir, period_tasks, examples, diagnostic = prepare_period(
                    executor, config, current_adapter, allocation, output, period, args.attempts,
                    args.logic_dir,
                    prior_tasks=prior_tasks, start_index=len(prior_tasks), minimum_tasks=args.minimum_tasks,
                )
        elif (period_dir / "period_input.json").is_file():
            period_tasks, examples, diagnostic = load_prepared_period(period_dir)
            if len(period_tasks) < args.minimum_tasks:
                raise RuntimeError("saved period {} has only {} verified tasks; {} required".format(
                    period, len(period_tasks), args.minimum_tasks))
        else:
            with worker_pool(config, args.gpus, current_adapter) as executor:
                period_dir, period_tasks, examples, diagnostic = prepare_period(
                    executor, config, current_adapter, allocation, output, period, args.attempts,
                    args.logic_dir,
                    prior_tasks=prior_tasks, start_index=len(prior_tasks), minimum_tasks=args.minimum_tasks,
                    resume=True,
                )

        prior_tasks.extend(period_tasks)
        cumulative_examples.extend(examples)
        write_jsonl(cumulative_path, cumulative_examples)
        store_state(state_path, {"stage": "period_{}_prepared".format(period), "completed_round": period - 1,
                                 "current_adapter": str(current_adapter),
                                 "verified_tasks": len(prior_tasks), "sft_examples": len(cumulative_examples)})

        changed = None
        nonzero_b = None
        if adapter_dir.exists():
            metrics_path = adapter_dir / "training_metrics.json"
            if not metrics_path.is_file():
                raise RuntimeError("period {} has an incomplete adapter directory: {}".format(period, adapter_dir))
            metrics = read_json(metrics_path)
            if Path(metrics["parent_adapter"]) != current_adapter:
                raise RuntimeError("period {} adapter was trained from a different parent".format(period))
            changed, nonzero_b = assert_adapter_updated(current_adapter, adapter_dir)
        else:
            log("round {}: fine-tuning cumulative SFT data: {} examples from {} verified tasks".format(
                period, len(cumulative_examples), len(prior_tasks)))
            training_dir = period_dir / "solver_adapter.inprogress"
            if training_dir.exists():
                shutil.rmtree(training_dir)
            train_solver_sft(config, cumulative_path, training_dir,
                             parent_adapter=current_adapter, device=args.train_device)
            training_dir.replace(adapter_dir)
            metrics = read_json(adapter_dir / "training_metrics.json")
            changed, nonzero_b = assert_adapter_updated(current_adapter, adapter_dir)
            log("round {} adapter changed in {} tensors; nonzero LoRA B tensors {}".format(
                period, changed, nonzero_b))
        current_adapter = adapter_dir
        store_state(state_path, {"stage": "period_{}_trained".format(period), "completed_round": period - 1,
                                 "current_adapter": str(current_adapter),
                                 "verified_tasks": len(prior_tasks), "sft_examples": len(cumulative_examples)})

        dev_results_path = period_dir / "dev_results.json"
        if dev_results_path.is_file():
            dev_results = read_json(dev_results_path)
        else:
            with worker_pool(config, args.gpus, current_adapter) as executor:
                dev_results = evaluate(executor, manifest["id_dev"], 61000 + period * 100)
            write_json(dev_results_path, dev_results)
        summary = {
            "round": period,
            "diagnostic_success_rate": diagnostic["success_rate"],
            "new_verified_tasks": len(period_tasks),
            "cumulative_verified_tasks": len(prior_tasks),
            "new_sft_actions": len(examples),
            "cumulative_sft_actions": len(cumulative_examples),
            "train_metrics": metrics["train_metrics"],
            "adapter_tensors_changed": changed,
            "nonzero_lora_b_tensors": nonzero_b,
            "dev_success_count": dev_results["success_count"],
            "dev_task_count": dev_results["task_count"],
            "dev_success_rate": dev_results["success_rate"],
            "dev_delta_from_seed": dev_results["success_rate"] - dev_baseline["success_rate"],
            "solver_adapter": str(current_adapter),
        }
        write_json(summary_path, summary)
        summaries.append(summary)
        store_state(state_path, {"stage": "period_{}_complete".format(period), "completed_round": period,
                                 "current_adapter": str(current_adapter),
                                 "verified_tasks": len(prior_tasks), "sft_examples": len(cumulative_examples)})
        log("round {} complete: {} new verified tasks; dev {}/{}".format(
            period, len(period_tasks), dev_results["success_count"], dev_results["task_count"]))

    final_path = output / "final_test.json"
    if final_path.is_file():
        final_test = read_json(final_path)
    else:
        with worker_pool(config, args.gpus, current_adapter) as executor:
            final_test = {
                "id": evaluate(executor, manifest["id_test"], 910000),
                "ood": evaluate(executor, manifest["ood_test"], 920000),
            }
        write_json(final_path, final_test)
    result = {
        "seed_round": 1,
        "total_rounds": args.cycles + 1,
        "seed_verified_tasks": len(seed_tasks),
        "total_verified_tasks": len(prior_tasks),
        "total_sft_actions": len(cumulative_examples),
        "seed_dev_success_rate": dev_baseline["success_rate"],
        "final_dev_success_rate": summaries[-1]["dev_success_rate"],
        "initial_id": seed_baseline["id"]["success_rate"],
        "final_id": final_test["id"]["success_rate"],
        "initial_ood": seed_baseline["ood"]["success_rate"],
        "final_ood": final_test["ood"]["success_rate"],
        "periods": summaries,
        "final_solver_adapter": str(current_adapter),
    }
    write_json(output / "result.json", result)
    store_state(state_path, {"stage": "complete", "completed_round": args.cycles + 1,
                             "current_adapter": str(current_adapter),
                             "verified_tasks": len(prior_tasks), "sft_examples": len(cumulative_examples)})
    log("completed {} total rounds: ID {:.3f}->{:.3f}; OOD {:.3f}->{:.3f}".format(
        args.cycles + 1, result["initial_id"], result["final_id"],
        result["initial_ood"], result["final_ood"]))


if __name__ == "__main__":
    main()
