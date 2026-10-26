import hashlib
from itertools import permutations
import json
import random
import re
import shutil
from pathlib import Path

from .alfworld import AlfworldEpisode
from .records import SyntheticTask


PUT_INTO = re.compile(r"^(put|move)\s+(.+?)\s+(in|on|into|to)\s+(.+)$", re.IGNORECASE)
PDDL_ATOM = re.compile(r"\(\s*([A-Za-z][\w-]*)\s+([^()]+?)\s*\)")
PLACEMENT_OBJECTIVE = re.compile(
    r"Your task is to:\s*put\s+(?:(?:one|two|three|\d+)\s+)?(.+?)\s+"
    r"(?:in/on|in|on|into)\s+(.+?)(?:[.!?]|\n|$)",
    re.IGNORECASE,
)
TAKE_FROM = re.compile(r"^take\s+(.+?)\s+from\s+(.+)$", re.IGNORECASE)


def action_goal(action):
    match = PUT_INTO.match(action.strip().rstrip("."))
    if match:
        preposition = "in" if match.group(3).lower() == "to" else match.group(3)
        return "Put {} {} {}.".format(match.group(2), preposition, match.group(4))
    return "Complete the observed placement goal."


def source_scene_id(game_file):
    metadata_path = Path(game_file).parent / "traj_data.json"
    with metadata_path.open("r", encoding="utf-8") as stream:
        scene = json.load(stream)["scene"]
    return "{}:{}".format(scene["floor_plan"], scene["scene_num"])


def supported_placement_game(game_file):
    task_name = Path(game_file).parents[1].name.lower()
    return task_name.startswith(("pick_and_place_simple-", "pick_two_obj_and_place-"))


def fact_key(fact):
    return fact["predicate"], tuple(fact["arguments"])


def fact_records(state_facts):
    return [
        {"predicate": predicate, "arguments": list(arguments)}
        for predicate, arguments in state_facts
    ]


def pddl_goal_expression(goal_atoms):
    atoms = [
        "({} {})".format(atom["predicate"], " ".join(atom["arguments"]))
        for atom in goal_atoms
    ]
    return "(:goal (and {}))".format(" ".join(atoms))


def pddl_goal_section(problem):
    start = problem.lower().find("(:goal")
    if start < 0:
        raise ValueError("ALFWorld PDDL problem has no :goal section")
    depth = 0
    end = None
    for index in range(start, len(problem)):
        if problem[index] == "(":
            depth += 1
        elif problem[index] == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError("ALFWorld PDDL goal section is not balanced")
    return start, end, problem[start:end]


def pddl_initial_section(problem):
    start = problem.lower().find("(:init")
    if start < 0:
        raise ValueError("ALFWorld PDDL problem has no :init section")
    depth = 0
    for index in range(start, len(problem)):
        if problem[index] == "(":
            depth += 1
        elif problem[index] == ")":
            depth -= 1
            if depth == 0:
                return problem[start:index + 1]
    raise ValueError("ALFWorld PDDL init section is not balanced")


def pddl_goal_satisfied_initially(problem, goal_atoms):
    initial_facts = {
        (match.group(1).lower(), tuple(match.group(2).lower().split()))
        for match in PDDL_ATOM.finditer(pddl_initial_section(problem))
    }
    return all(fact_key(atom) in initial_facts for atom in goal_atoms)


def positive_goal_signature(goal_text):
    operators = {"and", "or", "not", "exists", "forall"}
    return {
        (match.group(1).lower(), tuple(match.group(2).lower().split()))
        for match in PDDL_ATOM.finditer(goal_text)
        if match.group(1).lower() not in operators
    }


def same_source_placement_goal(problem, goal_atoms):
    start, _, source_goal = pddl_goal_section(problem)
    if re.search(r"\(\s*(?:or|forall)\b", source_goal, re.IGNORECASE):
        return False
    goal_without_inequalities = re.sub(
        r"\(\s*not\s+\(\s*=\s+\?\w+\s+\?\w+\s*\)\s*\)",
        "",
        source_goal,
        flags=re.IGNORECASE,
    )
    if re.search(r"\(\s*not\b", goal_without_inequalities, re.IGNORECASE):
        return False

    source_atoms = [
        (match.group(1).lower(), match.group(2).lower().split())
        for match in PDDL_ATOM.finditer(source_goal)
    ]
    if any(predicate not in {"objecttype", "receptacletype", "inreceptacle"} for predicate, _ in source_atoms):
        return False
    relations = [arguments for predicate, arguments in source_atoms if predicate == "inreceptacle"]
    if not relations or len(relations) != len(goal_atoms):
        return False

    object_requirements = {
        arguments[0]: arguments[1]
        for predicate, arguments in source_atoms
        if predicate == "objecttype"
    }
    receptacle_requirements = {
        arguments[0]: arguments[1]
        for predicate, arguments in source_atoms
        if predicate == "receptacletype"
    }
    object_variables = {arguments[0] for arguments in relations}
    receptacle_variables = {arguments[1] for arguments in relations}
    if set(object_requirements) != object_variables or set(receptacle_requirements) != receptacle_variables:
        return False

    init_start = problem.lower().find("(:init")
    if init_start < 0:
        return False
    object_types = {}
    receptacle_types = {}
    for match in PDDL_ATOM.finditer(problem[init_start:start]):
        predicate = match.group(1).lower()
        arguments = match.group(2).lower().split()
        if len(arguments) == 2 and predicate == "objecttype":
            object_types[arguments[0]] = arguments[1]
        elif len(arguments) == 2 and predicate == "receptacletype":
            receptacle_types[arguments[0]] = arguments[1]

    candidate_pairs = [tuple(atom["arguments"]) for atom in goal_atoms]
    inequalities = re.findall(
        r"\(\s*not\s+\(\s*=\s+(\?\w+)\s+(\?\w+)\s*\)\s*\)",
        source_goal,
        re.IGNORECASE,
    )
    for assignment in permutations(candidate_pairs):
        variable_values = {}
        matches = True
        for (object_variable, receptacle_variable), (object_id, receptacle_id) in zip(relations, assignment):
            values = {
                object_variable: object_id.lower(),
                receptacle_variable: receptacle_id.lower(),
            }
            if any(
                variable in variable_values and variable_values[variable] != value
                for variable, value in values.items()
            ):
                matches = False
                break
            variable_values.update(values)
            if object_types.get(object_id.lower()) != object_requirements[object_variable]:
                matches = False
                break
            if receptacle_types.get(receptacle_id.lower()) != receptacle_requirements[receptacle_variable]:
                matches = False
                break
        if matches and all(variable_values[left] != variable_values[right] for left, right in inequalities):
            return True
    return False


def replace_pddl_goal(problem, goal_atoms):
    start, end, _ = pddl_goal_section(problem)
    return problem[:start] + pddl_goal_expression(goal_atoms) + problem[end:]


def task_rejection_reason(problem, goal_atoms):
    if pddl_goal_satisfied_initially(problem, goal_atoms):
        return "goal_already_satisfied_initially"
    _, _, source_goal = pddl_goal_section(problem)
    custom_goal = pddl_goal_expression(goal_atoms)
    if (
        positive_goal_signature(source_goal) == positive_goal_signature(custom_goal)
        or same_source_placement_goal(problem, goal_atoms)
    ):
        return "duplicates_source_task_goal"
    return None


def make_task(game_file, task_dir, source_split, witness_actions, goal_atoms,
              goal_kind, action, object_type=None, experiment_id="",
              hypothesis_id="", hypothesis="", feedback_refs=None,
              diagnostic_target=None):
    source_file = Path(game_file).resolve()
    key = json.dumps(
        {
            "source": str(source_file),
            "goal_atoms": goal_atoms,
            "witness_actions": witness_actions,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    task_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    task_directory = Path(task_dir).resolve() / task_id
    with source_file.open("r", encoding="utf-8") as stream:
        source_data = json.load(stream)
    _, _, source_goal_text = pddl_goal_section(source_data["pddl_problem"])
    custom_goal_text = pddl_goal_expression(goal_atoms)
    if task_rejection_reason(source_data["pddl_problem"], goal_atoms):
        return None
    task_directory.mkdir(parents=True, exist_ok=True)
    for filename in ("traj_data.json", "initial_state.pddl"):
        source = source_file.parent / filename
        if source.exists():
            shutil.copy2(source, task_directory / filename)

    source_data["pddl_problem"] = replace_pddl_goal(source_data["pddl_problem"], goal_atoms)
    source_data["solvable"] = None
    source_data["walkthrough"] = None
    task_game_file = task_directory / "game.tw-pddl"
    with task_game_file.open("w", encoding="utf-8") as stream:
        json.dump(source_data, stream, ensure_ascii=False)

    if goal_kind == "two_same_type_in_receptacle":
        placement = PUT_INTO.match(action.strip().rstrip("."))
        preposition = "in" if placement.group(3).lower() == "to" else placement.group(3)
        object_label = object_type.lower().removesuffix("type")
        instruction = "Put two different {} objects {} {}.".format(
            object_label, preposition, placement.group(4)
        )
    else:
        instruction = action_goal(action)
    return SyntheticTask(
        task_id=task_id,
        source_game_file=str(source_file),
        task_game_file=str(task_game_file),
        source_scene_id=source_scene_id(source_file),
        instruction=instruction,
        goal_kind=goal_kind,
        goal_atoms=goal_atoms,
        source_goal_text=source_goal_text,
        custom_goal_text=custom_goal_text,
        witness_actions=list(witness_actions),
        source_split=source_split,
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        hypothesis=hypothesis,
        feedback_refs=list(feedback_refs or []),
        diagnostic_target=dict(diagnostic_target or {}),
    )


def task_from_dict(value):
    return SyntheticTask(**value)


def read_tasks(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [task_from_dict(json.loads(line)) for line in stream if line.strip()]


def write_tasks(path, tasks):
    from dataclasses import asdict

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for task in tasks:
            stream.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def goal_reached(goal_atoms, facts):
    available = {fact_key(fact) for fact in facts}
    return all(fact_key(atom) in available for atom in goal_atoms)


def mark_task_solvable(task):
    path = Path(task.task_game_file)
    with path.open("r", encoding="utf-8") as stream:
        game_data = json.load(stream)
    game_data["solvable"] = True
    game_data["walkthrough"] = task.witness_actions
    with path.open("w", encoding="utf-8") as stream:
        json.dump(game_data, stream, ensure_ascii=False)


def replay_witness(config, task):
    actions = []
    won = False
    facts = []
    with AlfworldEpisode(
        config["alfworld"]["config_path"],
        game_file=task.task_game_file,
        split=task.source_split,
    ) as env:
        env.reset()
        for action in task.witness_actions:
            result = env.step(action)
            actions.append(action)
            facts = result["facts"]
            won = result["won"]
            if result["done"]:
                break
    valid = won and goal_reached(task.goal_atoms, facts)
    if valid:
        mark_task_solvable(task)
    return {
        "valid": bool(valid),
        "actions": actions,
        "goal_reached": goal_reached(task.goal_atoms, facts),
        "won": won,
    }


def placement_candidates(previous_facts, current_facts):
    new_placements = sorted(
        fact for fact in current_facts - previous_facts if fact[0].lower() == "inreceptacle"
    )
    all_placements = sorted(
        fact for fact in current_facts if fact[0].lower() == "inreceptacle"
    )
    object_types = {
        arguments[0]: arguments[1]
        for predicate, arguments in current_facts
        if predicate.lower() == "objecttype"
    }
    candidates = []
    for placed in new_placements:
        candidates.append(([placed], "single_in_receptacle", object_types.get(placed[1][0])))
        placed_type = object_types.get(placed[1][0])
        if placed_type is None:
            continue
        for other in all_placements:
            if other == placed or other[1][0] == placed[1][0]:
                continue
            same_type = object_types.get(other[1][0]) == placed_type
            same_receptacle = other[1][1] == placed[1][1]
            if same_type and same_receptacle:
                pair = sorted((placed, other))
                candidates.append((pair, "two_same_type_in_receptacle", placed_type))
                break
    return candidates


def task_directed_commands(commands, target_object, target_receptacle,
                           visited_locations, opened_containers, placed_target_objects,
                           carrying_target):
    def command_object(command):
        take = TAKE_FROM.match(command.strip().rstrip("."))
        if take:
            return take.group(1)
        placement = PUT_INTO.match(command.strip().rstrip("."))
        if placement:
            return placement.group(2)
        match = re.match(r"^(?:take|move|put)\s+(.+)$", command, re.IGNORECASE)
        return match.group(1) if match else None

    object_actions = [
        command for command in commands
        if command_object(command) and object_matches(command_object(command), target_object)
        and not any(
            re.search(r"(?<!\w){}(?!\w)".format(re.escape(object_id)), command, re.IGNORECASE)
            for object_id in placed_target_objects
        )
        and (
            command.lower().startswith("take ")
            or target_receptacle in command.lower()
        )
    ]
    if object_actions:
        return object_actions

    if carrying_target:
        return [
            command for command in commands
            if command.lower().startswith("go to ")
            and target_receptacle in command.lower()
        ]

    open_actions = [
        command for command in commands
        if command.lower().startswith("open ") and command.lower() not in opened_containers
    ]
    travel_actions = [
        command for command in commands
        if command.lower().startswith("go to ")
        and command[6:].strip().lower() not in visited_locations
        and target_receptacle not in command.lower()
    ]
    return open_actions + travel_actions


def placement_objective(observation):
    match = PLACEMENT_OBJECTIVE.search(observation)
    if not match:
        return None, None
    target_object = re.sub(r"^(?:some|a|an)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
    return target_object.lower(), match.group(2).strip().lower()


def object_matches(object_name, object_phrase):
    object_name = object_name.strip().lower()
    object_phrase = object_phrase.strip().lower()
    modifiers = {
        "clean", "dirty", "hot", "cold", "cool", "warm", "sliced", "whole",
        "cooked", "raw", "fresh", "stale", "wet", "dry", "washed", "ripe", "unripe",
    }
    phrase_words = object_phrase.split()
    while phrase_words and phrase_words[0] in modifiers:
        phrase_words.pop(0)
    object_words = object_name.split()
    while object_words and object_words[0] in modifiers:
        object_words.pop(0)
    object_core = " ".join(object_words)
    phrase_core = " ".join(phrase_words)
    return (
        object_name == object_phrase
        or object_name.startswith(object_phrase + " ")
        or object_phrase.startswith(object_name + " ")
        or object_core == phrase_core
        or object_core.startswith(phrase_core + " ")
        or phrase_core.startswith(object_core + " ")
    )


def location_matches(location, location_phrase):
    location = location.strip().lower()
    location_phrase = location_phrase.strip().lower()
    return (
        location == location_phrase
        or location.startswith(location_phrase + " ")
        or location_phrase.startswith(location + " ")
    )


def parse_experiment_hypothesis(text, source_object, source_receptacle,
                                observation, commands, research_feedback):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[len("```json"): -len("```")].strip()
    value = json.loads(text)
    required = {
        "hypothesis", "diagnostic_behavior", "focus_object",
        "focus_receptacle", "feedback_refs",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("hypothesis must be JSON with exactly the required five fields")
    for field in ("hypothesis", "diagnostic_behavior", "focus_object", "focus_receptacle"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError("hypothesis field {!r} must be non-empty text".format(field))
        value[field] = value[field].strip()[:240]
    if not isinstance(value["feedback_refs"], list) or not all(
        isinstance(ref, str) for ref in value["feedback_refs"]
    ):
        raise ValueError("feedback_refs must be a list of strings")
    value["feedback_refs"] = list(dict.fromkeys(value["feedback_refs"]))

    available_text = "\n".join([observation] + list(commands)).lower()
    focus_object = value["focus_object"].lower()
    focus_receptacle = value["focus_receptacle"].lower()
    if not object_matches(focus_object, source_object or "") and not re.search(
        r"(?<![\w-]){}(?![\w-])".format(re.escape(focus_object)), available_text
    ):
        raise ValueError("focus_object is not grounded in the current ALFWorld scene")
    if not location_matches(focus_receptacle, source_receptacle or "") and not re.search(
        r"(?<![\w-]){}(?![\w-])".format(re.escape(focus_receptacle)), available_text
    ):
        raise ValueError("focus_receptacle is not grounded in the current ALFWorld scene")
    if source_receptacle and location_matches(focus_receptacle, source_receptacle):
        raise ValueError("focus_receptacle repeats the source task goal")

    valid_refs = set(research_feedback.get("reference_ids", []))
    required_refs = set(research_feedback.get("required_reference_ids", []))
    refs = set(value["feedback_refs"])
    if refs - valid_refs:
        raise ValueError("feedback_refs contains references absent from the supplied feedback")
    if valid_refs and not refs.intersection(required_refs or valid_refs):
        raise ValueError("hypothesis must cite at least one supplied diagnostic or research finding")
    if not valid_refs and refs:
        raise ValueError("feedback_refs must be empty when no feedback was supplied")
    value["feedback_refs"].sort()
    return value


def hypothesis_prompt_messages(parent_service, view, research_feedback, solver_base_model):
    return [
        {
            "role": "system",
            "content": (
                "Design one ALFWorld experiment that probes a capability implicated by the supplied "
                "Solver feedback. Return JSON only, with exactly these fields: hypothesis, "
                "diagnostic_behavior, focus_object, focus_receptacle, feedback_refs. Choose the "
                "object and receptacle from this scene, and choose a receptacle type different from "
                "the native task destination so the witnessed goal is new. Name feedback_refs exactly "
                "as shown in the feedback. Cite at least one finding when feedback contains references; "
                "use an empty list only when no feedback references exist. Do not propose a solution."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "current_service": parent_service,
                    "solver_base_model": solver_base_model,
                    "initial_observation": view["observation"],
                    "admissible_commands": view["admissible_commands"],
                    "research_feedback": research_feedback,
                },
                ensure_ascii=False,
            ),
        },
    ]


def placement_matches_diagnostic_target(action, target):
    match = PUT_INTO.match(action.strip().rstrip("."))
    return bool(
        match
        and object_matches(match.group(2), target["focus_object"])
        and location_matches(match.group(4), target["focus_receptacle"])
    )


def carried_object_commands(commands, object_name, source_receptacle,
                            target_object, target_receptacle, visited_locations):
    excluded = {source_receptacle.strip().lower()}
    matches_target = target_object and object_matches(object_name, target_object)

    placements = []
    for command in commands:
        match = PUT_INTO.match(command.strip().rstrip("."))
        if (
            match
            and object_matches(match.group(2), object_name)
            and not any(location_matches(match.group(4), item) for item in excluded)
            and (not matches_target or location_matches(match.group(4), target_receptacle))
        ):
            placements.append(command)
    if placements:
        return placements

    target_travel = [
        command for command in commands
        if command.lower().startswith("go to ")
        and location_matches(command[6:], target_receptacle)
        and command[6:].strip().lower() not in visited_locations
    ]
    if target_travel:
        return target_travel
    return [
        command for command in commands
        if command.lower().startswith("go to ")
        and not any(location_matches(command[6:], item) for item in excluded)
        and command[6:].strip().lower() not in visited_locations
    ]


def explore_game(config, game_file, model, source_split, max_steps,
                 task_dir, parent_service=None, research_feedback=None,
                 experiment_id="experiment-0", seed=None):
    return explore_games(
        config,
        [game_file],
        [model],
        source_split,
        max_steps,
        task_dir,
        parent_service=parent_service,
        research_feedback=research_feedback,
        experiment_ids=[experiment_id],
        seed=seed,
    )[0]


def _generate_across_models(models, executor, prompts, allowed_texts,
                            max_new_tokens, temperature, top_p):
    outputs = [None] * len(prompts)
    buckets = [[] for _ in models]
    for index in range(len(prompts)):
        buckets[index % len(models)].append(index)
    jobs = []
    for model_index, indices in enumerate(buckets):
        if not indices:
            continue
        model = models[model_index]
        texts = None if allowed_texts is None else [allowed_texts[i] for i in indices]
        args = (
            [prompts[i] for i in indices],
            max_new_tokens,
            temperature,
            top_p,
        )
        if len(models) == 1:
            result = model.generate_batch(*args, allowed_texts_batch=texts)
            for index, item in zip(indices, result):
                outputs[index] = item
        else:
            future = executor.submit(
                model.generate_batch, *args, allowed_texts_batch=texts
            )
            jobs.append((indices, future))
    for indices, future in jobs:
        for index, item in zip(indices, future.result()):
            outputs[index] = item
    return outputs


def explore_games(config, game_files, models, source_split, max_steps,
                  task_dir, parent_service=None, research_feedback=None,
                  experiment_ids=None, seed=None):
    from concurrent.futures import ThreadPoolExecutor
    from .model import seed_everything

    research_feedback = research_feedback or {}
    experiment_ids = experiment_ids or ["experiment-{}".format(i) for i in range(len(game_files))]
    states = []
    pending_tasks = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(models))) as model_executor:
            hypothesis_prompts = []
            for index, game_file in enumerate(game_files):
                env = AlfworldEpisode(
                    config["alfworld"]["config_path"],
                    game_file=game_file,
                    split=source_split,
                )
                state = {
                    "index": index,
                    "env": env,
                    "game_file": game_file,
                    "tasks": [],
                    "task_candidates": [],
                    "task_candidate_count": 0,
                    "verified_task_count": 0,
                    "witness_replay_steps": 0,
                    "actions": [],
                    "action_history": [],
                    "blocked_actions": set(),
                    "visited_locations": set(),
                    "opened_containers": set(),
                    "placed_target_objects": set(),
                    "carrying_object": None,
                    "carrying_source": None,
                    "carrying_target": False,
                    "generations": [],
                    "seen_goals": set(),
                    "task_candidate_records": [],
                    "steps": 0,
                    "active": True,
                    "experiment_id": experiment_ids[index],
                    "research_feedback": research_feedback,
                }
                states.append(state)
                view = env.reset()
                state["view"] = view
                state["previous_facts"] = {fact_key(fact) for fact in view["facts"]}
                state["initial_facts"] = set(state["previous_facts"])
                state["source_object"], state["source_receptacle"] = placement_objective(
                    view["observation"]
                )
                with Path(env.game_file).open("r", encoding="utf-8") as stream:
                    state["source_pddl_problem"] = json.load(stream)["pddl_problem"]
                hypothesis_prompts.append(
                    hypothesis_prompt_messages(
                        parent_service,
                        view,
                        research_feedback,
                        {
                            "name": Path(config["model"]["path"]).name,
                            "revision": config["model"].get("revision"),
                        },
                    )
                )

            seed_everything(
                config["run"]["seed"] if seed is None else seed,
                models[0].torch,
            )
            hypothesis_generations = _generate_across_models(
                models,
                model_executor,
                hypothesis_prompts,
                None,
                config["model"]["max_new_tokens"],
                config["model"]["temperature"],
                config["model"]["top_p"],
            )
            for state, generation in zip(states, hypothesis_generations):
                state["generations"].append(generation)
                state["raw_hypothesis"] = generation["generated_text"].strip()
                try:
                    record = parse_experiment_hypothesis(
                        state["raw_hypothesis"],
                        state["source_object"],
                        state["source_receptacle"],
                        state["view"]["observation"],
                        state["view"]["admissible_commands"],
                        research_feedback,
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    state["hypothesis_error"] = str(error)
                    state["hypothesis"] = state["raw_hypothesis"]
                    state["hypothesis_record"] = None
                    state["active"] = False
                    continue
                state["hypothesis_record"] = record
                state["hypothesis"] = record["hypothesis"]
                state["hypothesis_id"] = state["experiment_id"] + "/hypothesis"
                state["target_object"] = record["focus_object"].lower()
                state["target_receptacle"] = record["focus_receptacle"].lower()

            while any(state["active"] for state in states):
                request_states = []
                request_prompts = []
                request_allowed_texts = []
                for state in states:
                    state["selected_action"] = None
                    if not state["active"]:
                        continue
                    if state["steps"] >= max_steps:
                        state["active"] = False
                        continue
                    view = state["view"]
                    available = [
                        command for command in view["admissible_commands"]
                        if command not in state["blocked_actions"]
                    ]
                    if not available:
                        state["active"] = False
                        continue
                    if state["carrying_object"]:
                        preferred = carried_object_commands(
                            available,
                            state["carrying_object"],
                            state["carrying_source"],
                            state["target_object"],
                            state["target_receptacle"],
                            state["visited_locations"],
                        )
                    elif state["target_object"]:
                        preferred = task_directed_commands(
                            available,
                            state["target_object"],
                            state["target_receptacle"],
                            state["visited_locations"],
                            state["opened_containers"],
                            state["placed_target_objects"],
                            state["carrying_target"],
                        )
                    else:
                        preferred = available
                    preferred = [command for command in preferred if command in available]
                    choices = preferred if state["carrying_object"] and preferred else available
                    state["available_commands"] = choices
                    if len(choices) == 1:
                        state["selected_action"] = choices[0]
                        continue
                    request_states.append(state)
                    request_allowed_texts.append(choices)
                    request_prompts.append([
                        {
                            "role": "system",
                            "content": (
                                "This is ALFWorld control, not free-form question answering. Test the "
                                "hypothesis on its focus object and focus receptacle. Use the supplied "
                                "research feedback and the latest environment result to choose the next "
                                "step. Produce a placement subgoal that was false in the initial state; "
                                "do not return the object to its source or recreate the native task goal. "
                                "Use task-directed suggestions when available. Copy one allowed command "
                                "verbatim and output only that text. Do not repeat an action that produced "
                                "no state change."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Hypothesis: {}\nDiagnostic target: {}\nResearch feedback: {}"
                                "\nObservation:\n{}\nRecent action outcomes:\n{}"
                                "\nTask-directed suggestions:\n{}"
                                "\nCopy one exact command from this JSON list of allowed commands:\n{}"
                            ).format(
                                state["hypothesis"],
                                json.dumps(
                                    {
                                        "diagnostic_behavior": state["hypothesis_record"]["diagnostic_behavior"],
                                        "object": state["target_object"],
                                        "receptacle": state["target_receptacle"],
                                        "feedback_refs": state["hypothesis_record"]["feedback_refs"],
                                    },
                                    ensure_ascii=False,
                                ),
                                json.dumps(research_feedback, ensure_ascii=False),
                                view["observation"],
                                json.dumps(state["action_history"][-5:], ensure_ascii=False),
                                json.dumps(preferred, ensure_ascii=False, indent=2),
                                json.dumps(choices, ensure_ascii=False, indent=2),
                            ),
                        },
                    ])

                if request_prompts:
                    generations = _generate_across_models(
                        models,
                        model_executor,
                        request_prompts,
                        request_allowed_texts,
                        32,
                        config["model"]["temperature"],
                        config["model"]["top_p"],
                    )
                    for state, generation in zip(request_states, generations):
                        state["generations"].append(generation)
                        selection = generation["generated_text"].strip()
                        if selection not in state["available_commands"]:
                            raise ValueError(
                                "constrained proposer returned invalid action {!r}".format(selection)
                            )
                        state["selected_action"] = selection

                stepping = [state for state in states if state["selected_action"] is not None]
                for state in stepping:
                    action = state["selected_action"]
                    view = state["view"]
                    result = state["env"].step(action)
                    state["actions"].append(action)
                    lowered_action = action.lower()
                    if lowered_action.startswith("go to "):
                        state["visited_locations"].add(action[6:].strip().lower())
                    elif lowered_action.startswith("open "):
                        state["opened_containers"].add(lowered_action)
                    take = TAKE_FROM.match(action.strip().rstrip("."))
                    if take:
                        state["carrying_object"] = take.group(1).strip().lower()
                        state["carrying_source"] = take.group(2).strip().lower()
                        state["carrying_target"] = bool(
                            state["target_object"]
                            and object_matches(state["carrying_object"], state["target_object"])
                        )
                        state["visited_locations"].clear()
                    placement = PUT_INTO.match(action.strip().rstrip("."))
                    if placement:
                        placed_object = placement.group(2).strip().lower()
                        if state["carrying_object"] and object_matches(
                            placed_object, state["carrying_object"]
                        ):
                            state["carrying_object"] = None
                            state["carrying_source"] = None
                            state["carrying_target"] = False
                            state["visited_locations"].clear()
                        if state["target_object"] and object_matches(
                            placed_object, state["target_object"]
                        ):
                            state["placed_target_objects"].add(placed_object)
                    current_facts = {fact_key(fact) for fact in result["facts"]}
                    state["action_history"].append(
                        {
                            "action": action,
                            "observation": result["observation"],
                            "new_facts": fact_records(
                                current_facts - state["previous_facts"]
                            ),
                        }
                    )
                    if (
                        current_facts == state["previous_facts"]
                        and result["observation"] == view["observation"]
                    ):
                        state["blocked_actions"].add(action)
                    else:
                        state["blocked_actions"].clear()
                    if PUT_INTO.match(action.strip().rstrip(".")):
                        for facts, goal_kind, object_type in placement_candidates(
                            state["previous_facts"], current_facts
                        ):
                            atoms = fact_records(facts)
                            goal_key = tuple(fact_key(atom) for atom in atoms)
                            if goal_key in state["seen_goals"]:
                                continue
                            state["seen_goals"].add(goal_key)
                            state["task_candidate_count"] += 1
                            candidate_record = {
                                "candidate_id": "{}-task-{:03d}".format(
                                    state["experiment_id"], state["task_candidate_count"]
                                ),
                                "goal_atoms": atoms,
                                "action": action,
                                "hypothesis_id": state.get("hypothesis_id"),
                                "feedback_refs": (
                                    state["hypothesis_record"]["feedback_refs"]
                                    if state.get("hypothesis_record") else []
                                ),
                                "diagnostic_target": {
                                    "object": state.get("target_object"),
                                    "receptacle": state.get("target_receptacle"),
                                },
                                "relevant_to_hypothesis": False,
                                "initial_goal_satisfied": False,
                                "rejection_reason": None,
                            }
                            state["task_candidate_records"].append(candidate_record)
                            if not state.get("hypothesis_record") or not placement_matches_diagnostic_target(
                                action, state["hypothesis_record"]
                            ):
                                candidate_record["rejection_reason"] = "not_diagnostic_target"
                                continue
                            candidate_record["relevant_to_hypothesis"] = True
                            if (
                                set(goal_key).issubset(state["initial_facts"])
                                or pddl_goal_satisfied_initially(state["source_pddl_problem"], atoms)
                            ):
                                candidate_record["initial_goal_satisfied"] = True
                                candidate_record["rejection_reason"] = "goal_already_satisfied_initially"
                                continue
                            rejection_reason = task_rejection_reason(
                                state["source_pddl_problem"], atoms
                            )
                            if rejection_reason:
                                candidate_record["rejection_reason"] = rejection_reason
                                continue
                            task = make_task(
                                state["env"].game_file,
                                task_dir,
                                source_split,
                                state["actions"],
                                atoms,
                                goal_kind,
                                action,
                                object_type,
                                experiment_id=state["experiment_id"],
                                hypothesis_id=state["hypothesis_id"],
                                hypothesis=state["hypothesis"],
                                feedback_refs=state["hypothesis_record"]["feedback_refs"],
                                diagnostic_target=candidate_record["diagnostic_target"],
                            )
                            if task is None:
                                candidate_record["rejection_reason"] = "task_creation_rejected"
                                continue
                            state["task_candidates"].append(task)
                            candidate_record["task_id"] = task.task_id
                            pending_tasks.append((state, task, candidate_record))
                    state["previous_facts"] = current_facts
                    state["view"] = result
                    state["steps"] += 1
                    if result["done"]:
                        state["active"] = False
    finally:
        for state in states:
            state["env"].close()

    if pending_tasks:
        for state, task, candidate_record in pending_tasks:
            replay = replay_witness(config, task)
            state["witness_replay_steps"] += len(replay["actions"])
            candidate_record["witness_replay"] = replay
            if replay["valid"]:
                state["tasks"].append(task)
                state["verified_task_count"] += 1
            else:
                candidate_record["rejection_reason"] = "witness_replay_failed"

    return [
        {
            "tasks": state["tasks"],
            "generations": state["generations"],
            "hypothesis": state["hypothesis"],
            "hypothesis_record": state.get("hypothesis_record"),
            "hypothesis_error": state.get("hypothesis_error"),
            "hypothesis_id": state.get("hypothesis_id"),
            "research_feedback": research_feedback,
            "task_candidate_records": state["task_candidate_records"],
            "diagnostic_target": state.get("hypothesis_record") and {
                "object": state["target_object"],
                "receptacle": state["target_receptacle"],
            },
            "actions": state["actions"],
            "trajectory": state["action_history"],
            "task_candidate_count": state["task_candidate_count"],
            "verified_task_count": state["verified_task_count"],
            "witness_replay_steps": state["witness_replay_steps"],
        }
        for state in states
    ]


def synthesize(config, model, count, task_dir):
    if count < 1:
        raise ValueError("count must be positive")
    split = config["alfworld"]["split"]
    seed_game = config["alfworld"].get("game_file")
    if seed_game is None:
        from .alfworld import list_game_files

        game_files = [
            game_file
            for game_file in list_game_files(config["alfworld"]["config_path"], split)
            if supported_placement_game(game_file)
        ]
    else:
        game_files = [seed_game]

    tasks = []
    seen_scenes = set()
    random.Random(config["run"]["seed"]).shuffle(game_files)
    from .model import seed_everything

    for index, game_file in enumerate(game_files):
        seed_everything(config["run"]["seed"] + index, model.torch)
        exploration = explore_game(
            config,
            game_file,
            model,
            split,
            config["run"]["max_steps"],
            task_dir,
            seed=config["run"]["seed"] + index,
        )
        for task in exploration["tasks"]:
            if task.source_scene_id in seen_scenes:
                continue
            seen_scenes.add(task.source_scene_id)
            tasks.append(task)
            if len(tasks) >= count:
                return tasks
    return tasks
