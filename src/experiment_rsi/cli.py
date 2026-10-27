import argparse
import json
import random
from pathlib import Path

from .records import EpisodeRecord, EpisodeStep, write_jsonl


def env_smoke(args):
    from .config import load_config

    config = load_config(args.config)
    from .alfworld import AlfworldEpisode

    with AlfworldEpisode(
        config["alfworld"]["config_path"],
        game_file=config["alfworld"].get("game_file"),
        split=config["alfworld"].get("split", "eval_in_distribution"),
    ) as env:
        initial = env.reset()
        if not initial["admissible_commands"]:
            raise RuntimeError("ALFWorld reset produced no admissible commands")
        action = initial["admissible_commands"][0]
        result = env.step(action)
        episode = EpisodeRecord(
            game_file=initial["game_file"],
            initial_observation=initial["observation"],
            steps=[
                EpisodeStep(
                    action=action,
                    observation=result["observation"],
                    admissible_commands=result["admissible_commands"],
                    done=result["done"],
                    won=result["won"],
                )
            ],
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "episodes.jsonl", [episode])
    print(json.dumps({"game_file": episode.game_file, "steps": 1, "won": episode.won}, ensure_ascii=False))


def model_smoke(args):
    from .config import load_config

    config = load_config(args.config)
    from .model import ModelSession

    session = ModelSession(config)
    record = session.generate(
        [{"role": "user", "content": "Reply with the word ready."}],
        max_new_tokens=config["model"]["max_new_tokens"],
        temperature=config["model"]["temperature"],
        top_p=config["model"]["top_p"],
    )
    if len(record["generated_ids"]) != len(record["behavior_logprobs"]):
        raise RuntimeError("generated token and behavior log-probability lengths differ")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "generation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tokens": len(record["generated_ids"]), "text": record["generated_text"]}, ensure_ascii=False))


def train_smoke(args):
    from .config import load_config
    from .model import ModelSession, seed_everything
    from .verl_training import train_with_verl

    config = load_config(args.config)
    session = ModelSession(config)
    messages = [{"role": "user", "content": "Write one short useful sentence."}]
    samples = []
    for index in range(2):
        seed_everything(config["run"]["seed"] + index, session.torch)
        generation = session.generate(
            messages, max_new_tokens=32, temperature=config["model"]["temperature"],
            top_p=config["model"]["top_p"],
        )
        samples.append({
            **generation, "role": "experiment", "group_id": "runtime-smoke",
            "training_eligible": True,
            # Length reward exercises GRPO plumbing, not ALFWorld research quality.
            "reward": float(len(generation["generated_ids"])),
            "task_metrics": {"trajectory_id": str(index), "sample_weight": 1.0,
                             "prompt_messages": messages},
        })
    del session
    result = train_with_verl(config, samples, args.out, "grpo")
    reloaded = ModelSession(config, adapter_path=result["adapter"])
    result["reload_generation"] = reloaded.generate(
        messages, max_new_tokens=16, temperature=0.0, top_p=1.0,
    )
    target = Path(args.out) / "train-smoke.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("backend", "trainer_step", "parameter_update", "adapter")}, ensure_ascii=False))


def synthesize_command(args):
    from .config import load_config
    from .research import synthesize_run

    config = load_config(args.config)
    if args.split:
        config["alfworld"]["split"] = args.split
    tasks = synthesize_run(config, args.out, args.count, args.adapter)
    print(json.dumps({"verified_tasks": len(tasks), "out": str(Path(args.out).resolve())}, ensure_ascii=False))


def build_scene_command(args):
    from .alfworld import AlfworldEpisode
    from .config import load_config
    from .evaluation import replace_native_objective
    from .records import SyntheticTask
    from .scene_builder import build_scene
    from .synthesis import goal_reached, pddl_goal_expression, replay_witness, write_tasks

    config = load_config(args.config)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    scene = build_scene(args.source_game, spec, args.out)
    with AlfworldEpisode(config["alfworld"]["config_path"], game_file=scene["game_file"], split=args.split) as env:
        initial = env.reset()
    result = {"scene": scene, "initial_goal_satisfied": goal_reached(scene["goal_atoms"], initial["facts"]),
              "initial_observation": replace_native_objective(initial["observation"], scene["instruction"]),
              "witness_replay": None}
    if result["initial_goal_satisfied"]:
        raise ValueError("constructed goal is satisfied at environment reset")
    if args.actions:
        actions = json.loads(Path(args.actions).read_text(encoding="utf-8"))
        task = SyntheticTask(
            task_id=scene["task_id"], source_game_file=scene["source_game_file"],
            task_game_file=scene["game_file"], source_scene_id=scene["source_scene_id"],
            instruction=scene["instruction"], goal_kind="place_all",
            goal_atoms=scene["goal_atoms"], source_goal_text=scene["source_goal_text"],
            custom_goal_text=pddl_goal_expression(scene["goal_atoms"]), witness_actions=actions,
            source_split=args.split, generation_mode="constructed", experiment_spec=spec,
        )
        result["witness_replay"] = replay_witness(config, task)
        if result["witness_replay"]["valid"]:
            write_tasks(Path(args.out) / "tasks.jsonl", [task])
    out = Path(args.out)
    (out / "scene.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["witness_replay"] is not None and not result["witness_replay"]["valid"]:
        raise RuntimeError("constructed scene witness did not complete the goal")


def replay_command(args):
    from .config import load_config
    from .research import write_json
    from .synthesis import read_tasks, replay_witness

    config = load_config(args.config)
    tasks = read_tasks(Path(args.run) / "tasks.jsonl")
    results = [{"task_id": task.task_id, **replay_witness(config, task)} for task in tasks]
    write_json(Path(args.run) / "replay.json", results)
    print(json.dumps({"tasks": len(tasks), "replayed": sum(item["valid"] for item in results)}, ensure_ascii=False))


def evaluate_synthesized_command(args):
    from .config import load_config
    from .evaluation import evaluate_tasks, mean_score, mean_task_reward
    from .model import ModelSession
    from .policy import load_service
    from .research import write_json
    from .synthesis import read_tasks

    config = load_config(args.config)
    tasks = read_tasks(args.tasks)
    service = load_service(args.service)
    model = ModelSession(config)
    results = evaluate_tasks(config, service, model, tasks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scores.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    metrics = {
        "success_rate": mean_score(results),
        "mean_task_reward": mean_task_reward(results),
        "task_count": len(results),
    }
    write_json(out / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False))


def research_once_command(args):
    from .config import load_config
    from .research import research_once

    config = load_config(args.config)
    metrics = research_once(
        config,
        args.diagnostic_tasks,
        args.selection_tasks,
        args.reward_tasks,
        args.out,
        args.parent,
        args.edit_adapter,
    )
    print(json.dumps(metrics, ensure_ascii=False))


def research_group_command(args):
    from .config import load_config
    from .research import experiment_group

    config = load_config(args.config)
    metrics = experiment_group(
        config,
        args.selection_tasks,
        args.reward_tasks,
        args.out,
        args.parent,
        args.experiment_adapter,
        args.edit_adapter,
    )
    print(json.dumps(metrics, ensure_ascii=False))


def run_command(args):
    from .config import load_config
    from .research import run_cycles

    config = load_config(args.config)
    result = run_cycles(
        config,
        args.selection_pool,
        args.reward_pool,
        args.periods,
        args.method,
        args.out,
        args.parent,
        args.test_tasks,
        args.experiment_adapter,
        args.edit_adapter,
        args.batches_per_period or config.get("research", {}).get("batches_per_period"),
        args.ood_test_tasks,
    )
    print(json.dumps(result, ensure_ascii=False))


def train_command(args):
    from .train import train_run

    summary = train_run(args.config, args.run, args.out, args.role, args.objective, args.adapter)
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(prog="experiment-rsi")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("env-smoke", env_smoke), ("model-smoke", model_smoke), ("train-smoke", train_smoke)):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--out", required=True)
        command.set_defaults(handler=handler)

    builder = subparsers.add_parser("build-scene")
    builder.add_argument("--config", required=True)
    builder.add_argument("--source-game", required=True)
    builder.add_argument("--spec", required=True)
    builder.add_argument("--out", required=True)
    builder.add_argument("--split", default="train")
    builder.add_argument("--actions", help="JSON list of witness actions to replay")
    builder.set_defaults(handler=build_scene_command)

    synthesize_parser = subparsers.add_parser("synthesize")
    synthesize_parser.add_argument("--config", required=True)
    synthesize_parser.add_argument("--count", type=int, required=True)
    synthesize_parser.add_argument("--split", choices=("train", "eval_in_distribution", "eval_out_of_distribution"))
    synthesize_parser.add_argument("--adapter")
    synthesize_parser.add_argument("--out", required=True)
    synthesize_parser.set_defaults(handler=synthesize_command)

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--config", required=True)
    replay_parser.add_argument("--run", required=True)
    replay_parser.set_defaults(handler=replay_command)

    synthesized_parser = subparsers.add_parser("evaluate-synthesized")
    synthesized_parser.add_argument("--config", required=True)
    synthesized_parser.add_argument("--tasks", required=True)
    synthesized_parser.add_argument("--service", default="seeds/service.json")
    synthesized_parser.add_argument("--out", required=True)
    synthesized_parser.set_defaults(handler=evaluate_synthesized_command)

    research_parser = subparsers.add_parser("research-once")
    research_parser.add_argument("--config", required=True)
    research_parser.add_argument("--diagnostic-tasks", required=True)
    research_parser.add_argument("--selection-tasks", required=True)
    research_parser.add_argument("--reward-tasks", required=True)
    research_parser.add_argument("--parent", default="seeds/service.json")
    research_parser.add_argument("--edit-adapter")
    research_parser.add_argument("--out", required=True)
    research_parser.set_defaults(handler=research_once_command)

    group_parser = subparsers.add_parser("research-group")
    group_parser.add_argument("--config", required=True)
    group_parser.add_argument("--selection-tasks", required=True)
    group_parser.add_argument("--reward-tasks", required=True)
    group_parser.add_argument("--parent", default="seeds/service.json")
    group_parser.add_argument("--experiment-adapter")
    group_parser.add_argument("--edit-adapter")
    group_parser.add_argument("--out", required=True)
    group_parser.set_defaults(handler=research_group_command)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--selection-pool", required=True)
    run_parser.add_argument("--reward-pool", required=True)
    run_parser.add_argument("--periods", type=int, required=True)
    run_parser.add_argument("--method", choices=("rl", "distill", "combined", "designer-only"), required=True)
    run_parser.add_argument("--parent", default="seeds/service.json")
    run_parser.add_argument("--experiment-adapter")
    run_parser.add_argument("--edit-adapter")
    run_parser.add_argument("--test-tasks")
    run_parser.add_argument("--ood-test-tasks")
    run_parser.add_argument("--batches-per-period", type=int)
    run_parser.add_argument("--out", required=True)
    run_parser.set_defaults(handler=run_command)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--role", choices=("edit", "experiment"), required=True)
    train_parser.add_argument("--objective", choices=("grpo", "opsd"), default="grpo")
    train_parser.add_argument("--run", required=True)
    train_parser.add_argument("--adapter")
    train_parser.add_argument("--out", required=True)
    train_parser.set_defaults(handler=train_command)
    args = parser.parse_args()
    args.handler(args)
