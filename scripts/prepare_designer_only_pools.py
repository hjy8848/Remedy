import argparse
import json
import random
from pathlib import Path

from experiment_rsi.alfworld import list_game_files
from experiment_rsi.config import load_config
from experiment_rsi.model import ModelSession
from experiment_rsi.policy import load_service
from experiment_rsi.research import write_json
from experiment_rsi.synthesis import (
    explore_games,
    source_scene_id,
    supported_placement_game,
    write_tasks,
)


def grouped_games(config, split):
    grouped = {}
    for game_file in list_game_files(config["alfworld"]["config_path"], split):
        if supported_placement_game(game_file):
            grouped.setdefault(source_scene_id(game_file), []).append(game_file)
    return {scene: sorted(files) for scene, files in grouped.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent", default="seeds/service.json")
    parser.add_argument("--scenes-per-pool", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["run"]["seed"] if args.seed is None else args.seed
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError("pool output already exists: {}".format(output))
    output.mkdir(parents=True)

    scenes = grouped_games(config, "eval_in_distribution")
    scene_order = list(scenes)
    random.Random(seed).shuffle(scene_order)
    roles = ("selection", "reward", "test")
    role_scenes = {role: [] for role in roles}
    used_scenes = set()
    cursor = 0
    role_tasks = {role: {} for role in roles}
    exploration_rows = []
    exploration_log = output / "exploration.jsonl"
    model_count = config["research"].get("exploration_gpus_per_node", 1)
    models = [ModelSession(config, device="cuda:{}".format(i)) for i in range(model_count)]
    service = load_service(args.parent)

    while any(len(role_tasks[role]) < args.scenes_per_pool for role in roles):
        pending = []
        for role in roles:
            needed = args.scenes_per_pool - len(role_tasks[role])
            while needed and cursor < len(scene_order):
                scene = scene_order[cursor]
                cursor += 1
                if scene in used_scenes:
                    continue
                used_scenes.add(scene)
                role_scenes[role].append(scene)
                pending.extend((role, scene, game_file) for game_file in scenes[scene])
                needed -= 1
        if not pending:
            break

        results = explore_games(
            config,
            [item[2] for item in pending],
            models,
            "eval_in_distribution",
            config["run"]["max_steps"],
            output / "task_games",
            parent_service=service,
            research_feedback={},
            experiment_ids=["pool-{}-{}-{:02d}".format(role, scene, i)
                            for i, (role, scene, _) in enumerate(pending)],
            seed=seed + len(exploration_rows),
        )
        for (role, scene, game_file), explored in zip(pending, results):
            if explored["tasks"] and scene not in role_tasks[role]:
                role_tasks[role][scene] = explored["tasks"][0]
            row = {
                "role": role,
                "source_scene_id": scene,
                "source_game_file": game_file,
                "hypothesis": explored["hypothesis"],
                "hypothesis_error": explored.get("hypothesis_error"),
                "trajectory": explored["trajectory"],
                "actions": explored["actions"],
                "task_candidate_count": explored["task_candidate_count"],
                "verified_task_count": explored["verified_task_count"],
                "task_candidate_records": explored["task_candidate_records"],
                "registered_task_ids": [task.task_id for task in explored["tasks"]],
            }
            exploration_rows.append(row)
            with exploration_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    del models
    verified_scenes = {role: len(role_tasks[role]) for role in roles}
    for role in roles:
        tasks = [role_tasks[role][scene] for scene in role_scenes[role]
                 if scene in role_tasks[role]]
        if len(tasks) < args.scenes_per_pool:
            write_json(output / "pool_manifest.json", {
                "seed": seed,
                "model": config["model"]["path"],
                "revision": config["model"]["revision"],
                "source_split": "eval_in_distribution",
                "scenes_per_pool": args.scenes_per_pool,
                "status": "incomplete",
                "verified_scenes": verified_scenes,
                "roles": {name: sorted(role_tasks[name]) for name in roles},
            })
            raise RuntimeError(
                "{} pool has {} verified scenes; {} required".format(
                    role, len(tasks), args.scenes_per_pool
                )
            )
        write_tasks(output / role / "tasks.jsonl", tasks)

    write_json(output / "pool_manifest.json", {
        "seed": seed,
        "model": config["model"]["path"],
        "revision": config["model"]["revision"],
        "source_split": "eval_in_distribution",
        "scenes_per_pool": args.scenes_per_pool,
        "status": "complete",
        "verified_scenes": verified_scenes,
        "roles": {role: sorted(role_tasks[role]) for role in roles},
    })
    print(json.dumps({role: len(role_tasks[role]) for role in roles}, ensure_ascii=False))


if __name__ == "__main__":
    main()
