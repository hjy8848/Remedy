from pathlib import Path
import random

import yaml


def list_game_files(config_path, split):
    import alfworld.agents.environment as environment

    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    wrapper = environment.get_environment(config["env"]["type"])(config, train_eval=split)
    return sorted(wrapper.game_files)


class AlfworldEpisode:
    """One ALFWorld PDDL game with its true state exposed to the host."""

    def __init__(self, config_path, game_file=None, split="eval_in_distribution"):
        from textworld import EnvInfos
        from textworld.envs import PddlEnv
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        if game_file is None:
            game_file = random.choice(list_game_files(config_path, split))

        self.game_file = str(Path(game_file).resolve())
        self.env = PddlEnv(
            EnvInfos(feedback=True, won=True, admissible_commands=True, facts=True, extras=["gamefile"])
        )
        self.env = AlfredDemangler(self.env, shuffle=False)
        self.env = AlfredInfos(self.env)
        self.env.load(self.game_file)

    @staticmethod
    def _view(state):
        return {
            "observation": state["feedback"],
            "admissible_commands": list(state["admissible_commands"]),
            "facts": [
                {
                    "predicate": fact.name,
                    "arguments": [argument.name for argument in fact.arguments],
                }
                for fact in state["_facts"]
            ],
        }

    def reset(self):
        state = self.env.reset()
        view = self._view(state)
        view["game_file"] = str(Path(state["extra.gamefile"]).resolve())
        self.game_file = view["game_file"]
        return view

    def step(self, action):
        state, score, done = self.env.step(action)
        result = self._view(state)
        result.update({"score": float(score), "done": bool(done), "won": bool(state["won"])})
        return result

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
