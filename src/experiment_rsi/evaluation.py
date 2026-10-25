import re
import statistics

from .alfworld import AlfworldEpisode
from .model import seed_everything
from .policy import make_policy_view
from .synthesis import fact_key, goal_reached
from .worker import CandidateExecutionError, PolicyWorker


TASK_LINE = re.compile(r"your task is to[^\n]*", re.IGNORECASE)
ACTION_OPTION = re.compile(r"(\d+): (.+)")


def replace_native_objective(observation, objective):
    return TASK_LINE.sub("Your task is to {}".format(objective), observation)


def task_progress(task, facts):
    available = {fact_key(fact) for fact in facts}
    completed = sum(fact_key(atom) in available for atom in task.goal_atoms)
    held_objects = {
        arguments[1]
        for predicate, arguments in available
        if predicate == "holds" and len(arguments) == 2
    }
    placed_objects = {atom["arguments"][0] for atom in task.goal_atoms if fact_key(atom) in available}
    partial = sum(
        1 for atom in task.goal_atoms
        if atom["arguments"][0] in held_objects and atom["arguments"][0] not in placed_objects
    )
    return completed, (completed + 0.5 * partial) / len(task.goal_atoms)


def resolve_model_action(selection, admissible_commands, action_choices):
    if selection in admissible_commands:
        return selection
    if re.fullmatch(r"\d+", selection):
        index = int(selection) - 1
        if 0 <= index < len(action_choices) and action_choices[index] in admissible_commands:
            return action_choices[index]
    lines = [line.strip() for line in selection.splitlines() if line.strip()]
    match = ACTION_OPTION.fullmatch(lines[-1]) if lines else None
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(action_choices) and match.group(2) == action_choices[index]:
            if action_choices[index] in admissible_commands:
                return action_choices[index]
    return selection


def run_episode(config, service, model, game_file, task=None, max_steps=None, seed=None):
    task_split = task.source_split if task else config["alfworld"]["split"]
    objective = task.instruction if task else None
    history = []
    trace = []
    success = False
    completed_goal_atoms = 0
    task_reward = 0.0
    candidate_error = None
    step_budget = max_steps if max_steps is not None else config["run"]["max_steps"]
    policy_seed = config["run"]["seed"] if seed is None else seed
    with AlfworldEpisode(
        config["alfworld"]["config_path"],
        game_file=task.task_game_file if task else game_file,
        split=task_split,
    ) as env, PolicyWorker(service, config["worker"], policy_seed) as worker:
        view = env.reset()
        policy_state = {}
        for step_index in range(step_budget):
            observation = view["observation"]
            if objective:
                observation = replace_native_objective(observation, objective)
            policy_view = make_policy_view(
                observation,
                objective,
                history,
                view["admissible_commands"],
                service["knowledge"],
                step_budget - step_index,
            )
            try:
                directive, policy_state = worker.step(policy_view, policy_state)
            except CandidateExecutionError as error:
                candidate_error = str(error)
                break
            generation = None
            if directive["kind"] == "model":
                generation = model.generate(
                    directive["messages"],
                    max_new_tokens=config["model"]["max_new_tokens"],
                    temperature=config["model"]["temperature"],
                    top_p=config["model"]["top_p"],
                )
                action = resolve_model_action(
                    generation["generated_text"].strip(),
                    view["admissible_commands"],
                    policy_state.get("action_choices", view["admissible_commands"]),
                )
            else:
                action = directive["action"]
            result = env.step(action)
            if task:
                completed_goal_atoms, task_reward = task_progress(task, result["facts"])
                success = result["won"] and goal_reached(task.goal_atoms, result["facts"])
            else:
                success = result["won"]
            trace.append(
                {
                    "action": action,
                    "observation": result["observation"],
                    "generation": generation,
                    "policy_kind": directive["kind"],
                    "policy_state": policy_state,
                    "done": result["done"],
                    "won": result["won"],
                    "goal_progress": task_reward if task else None,
                }
            )
            history.append({"action": action, "observation": result["observation"]})
            view = result
            if success or result["done"]:
                break
    return {
        "game_file": str(game_file),
        "task_id": task.task_id if task else None,
        "objective": objective,
        "success": bool(success),
        "completed_goal_atoms": completed_goal_atoms,
        "goal_atom_count": len(task.goal_atoms) if task else None,
        "task_reward": task_reward if task else float(success),
        "candidate_error": candidate_error,
        "steps": len(trace),
        "trace": trace,
    }


def evaluate_tasks(config, service, model, tasks):
    results = []
    for index, task in enumerate(tasks):
        seed = config["run"]["seed"] + index
        seed_everything(seed, model.torch)
        results.append(
            run_episode(config, service, model, task.task_game_file, task=task, seed=seed)
        )
    return results


def mean_score(results):
    if not results:
        raise ValueError("cannot score an empty task set")
    return statistics.mean(float(result["success"]) for result in results)


def mean_task_reward(results):
    if not results:
        raise ValueError("cannot score an empty task set")
    return statistics.mean(float(result["task_reward"]) for result in results)


def choose_candidate(parent_score, candidate_scores):
    best_index = max(range(len(candidate_scores)), key=candidate_scores.__getitem__)
    if candidate_scores[best_index] > parent_score:
        return best_index
    return None


def evidence_removal_reward(with_evidence_deltas, without_evidence_deltas):
    visible_gain = max(0.0, statistics.mean(with_evidence_deltas))
    masked_gain = max(0.0, statistics.mean(without_evidence_deltas))
    return visible_gain - masked_gain
