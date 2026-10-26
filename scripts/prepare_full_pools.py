import argparse
import json
import random
from pathlib import Path

from experiment_rsi.alfworld import list_game_files
from experiment_rsi.config import load_config
from experiment_rsi.model import ModelSession, seed_everything
from experiment_rsi.policy import load_service
from experiment_rsi.research import write_json
from experiment_rsi.synthesis import (
    explore_game,
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
    parser.add_argument("--selection-scenes", type=int)
    parser.add_argument("--reward-scenes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["run"]["seed"] if args.seed is None else args.seed
    selection_scene_count = (
        config["research"]["selection_seen_scenes"]
        if args.selection_scenes is None else args.selection_scenes
    )
    reward_scene_count = (
        config["research"]["reward_seen_scenes"]
        if args.reward_scenes is None else args.reward_scenes
    )
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError("pool output already exists: {}".format(output))
    output.mkdir(parents=True)

    seen_games = grouped_games(config, "eval_in_distribution")
    unseen_games = grouped_games(config, "eval_out_of_distribution")
    train_games = grouped_games(config, "train")
    if len(seen_games) <= selection_scene_count + reward_scene_count:
        raise ValueError("valid_seen does not leave any independent ID test scenes")

    scene_order = sorted(seen_games)
    random.Random(seed).shuffle(scene_order)
    selection_scenes = set(scene_order[:selection_scene_count])
    reward_start = selection_scene_count
    reward_scenes = set(scene_order[reward_start:reward_start + reward_scene_count])
    id_scenes = set(scene_order[reward_start + reward_scene_count:])
    split_assignments = {
        "selection": {scene: seen_games[scene] for scene in sorted(selection_scenes)},
        "reward": {scene: seen_games[scene] for scene in sorted(reward_scenes)},
        "id_test": {scene: seen_games[scene] for scene in sorted(id_scenes)},
        "ood_test": unseen_games,
    }

    parent = load_service(args.parent)
    model = ModelSession(config)
    exploration_rows = []
    pools = {}
    role_seed_offsets = {"selection": 0, "reward": 10_000,
                         "id_test": 20_000, "ood_test": 30_000}
    for role, scene_games in split_assignments.items():
        source_split = "eval_out_of_distribution" if role == "ood_test" else "eval_in_distribution"
        task_dir = output / role / "task_games"
        tasks = []
        seen_task_ids = set()
        game_index = 0
        for scene, game_files in scene_games.items():
            for game_file in game_files:
                seed_everything(seed + role_seed_offsets[role] + game_index, model.torch)
                explored = explore_game(
                    config,
                    game_file,
                    model,
                    source_split,
                    config["run"]["max_steps"],
                    task_dir,
                    parent_service=parent,
                )
                new_tasks = []
                for task in explored["tasks"]:
                    if task.task_id not in seen_task_ids:
                        seen_task_ids.add(task.task_id)
                        tasks.append(task)
                        new_tasks.append(task.task_id)
                exploration_rows.append({
                    "role": role,
                    "source_scene_id": scene,
                    "source_game_file": game_file,
                    "hypothesis": explored["hypothesis"],
                    "trajectory": explored["trajectory"],
                    "generations": explored["generations"],
                    "task_candidate_count": explored["task_candidate_count"],
                    "verified_task_count": explored["verified_task_count"],
                    "witness_replay_steps": explored["witness_replay_steps"],
                    "registered_task_ids": new_tasks,
                })
                game_index += 1
        role_dir = output / role
        role_dir.mkdir(parents=True, exist_ok=True)
        write_tasks(role_dir / "tasks.jsonl", tasks)
        pools[role] = {
            "task_count": len(tasks),
            "source_game_count": sum(len(files) for files in scene_games.values()),
            "source_scene_count": len(scene_games),
            "task_path": str(role_dir / "tasks.jsonl"),
            "source_scenes": sorted(scene_games),
        }
        if role in ("selection", "reward") and not tasks:
            raise RuntimeError("{} exploration produced no verified tasks".format(role))
        if role in ("id_test", "ood_test") and not tasks:
            raise RuntimeError("{} exploration produced no verified tasks".format(role))

    train_scene_ids = set(train_games)
    allowed_train_scenes = train_scene_ids - set().union(
        selection_scenes, reward_scenes, id_scenes, set(unseen_games)
    )
    pools["training_experiment_source"] = {
        "source_split": "train",
        "source_scene_count": len(allowed_train_scenes),
        "source_game_count": sum(
            len(files) for scene, files in train_games.items()
            if scene in allowed_train_scenes
        ),
        "source_scenes": sorted(allowed_train_scenes),
    }
    with (output / "exploration.jsonl").open("w", encoding="utf-8") as stream:
        for row in exploration_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output / "pool_manifest.json", {
        "seed": seed,
        "pool_generation_model": config["model"]["path"],
        "pool_generation_revision": config["model"]["revision"],
        "source_scene_assignment": {
            role: sorted(scene_games) for role, scene_games in split_assignments.items()
        },
        "pools": pools,
    })
    del model
    print(json.dumps({"out": str(output), "pools": pools}, ensure_ascii=False))


if __name__ == "__main__":
    main()
