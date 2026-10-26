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
    r"Your task is to:\s*put\s+(?:(?:one|two|three|\d+)\s+)?(.+?)\s+(?:in|on|into)\s+(.+?)(?:[.!?]|\n|$)",
    re.IGNORECASE,
)


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


def make_task(game_file, task_dir, source_split, witness_actions, goal_atoms,
              goal_kind, action, object_type=None):
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
    if (
        positive_goal_signature(source_goal_text) == positive_goal_signature(custom_goal_text)
        or same_source_placement_goal(source_data["pddl_problem"], goal_atoms)
    ):
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
    target_object_action = re.compile(
        r"^(?:take|move|put)\s+{}(?:\s|$)".format(re.escape(target_object)),
        re.IGNORECASE,
    )
    object_actions = [
        command for command in commands
        if target_object_action.match(command)
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


def explore_game(config, game_file, model, source_split, max_steps,
                 task_dir, parent_service=None):
    tasks = []
    task_candidate_count = 0
    verified_task_count = 0
    witness_replay_steps = 0
    actions = []
    action_history = []
    blocked_actions = set()
    visited_locations = set()
    opened_containers = set()
    placed_target_objects = set()
    carrying_target = False
    generations = []
    seen_goals = set()
    with AlfworldEpisode(
        config["alfworld"]["config_path"], game_file=game_file, split=source_split
    ) as env:
        view = env.reset()
        hypothesis_messages = [
            {
                "role": "system",
                "content": (
                    "Propose one concise experiment hypothesis about an ALFWorld execution "
                    "capability worth probing. State the behavior to probe, not a solution."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_service": parent_service,
                        "initial_observation": view["observation"],
                        "admissible_commands": view["admissible_commands"],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        hypothesis_sample = model.generate(
            hypothesis_messages,
            max_new_tokens=config["model"]["max_new_tokens"],
            temperature=config["model"]["temperature"],
            top_p=config["model"]["top_p"],
        )
        generations.append(hypothesis_sample)
        hypothesis = hypothesis_sample["generated_text"].strip()
        objective = PLACEMENT_OBJECTIVE.search(view["observation"])
        target_object = objective.group(1).strip().lower() if objective else None
        target_receptacle = objective.group(2).strip().lower() if objective else None
        previous_facts = {fact_key(fact) for fact in view["facts"]}
        for _ in range(max_steps):
            if target_object:
                candidate_commands = task_directed_commands(
                    view["admissible_commands"],
                    target_object,
                    target_receptacle,
                    visited_locations,
                    opened_containers,
                    placed_target_objects,
                    carrying_target,
                )
            else:
                candidate_commands = view["admissible_commands"]
            available_commands = [command for command in candidate_commands if command not in blocked_actions]
            if not available_commands:
                break
            if len(available_commands) == 1:
                selected_index = 0
            else:
                command_list = "\n".join(
                    "{}: {}".format(index, command)
                    for index, command in enumerate(available_commands, start=1)
                )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "This is ALFWorld control, not free-form question answering. Read the task "
                            "object and target receptacle from the observation. Explore to investigate "
                            "the stated hypothesis, while choosing commands that move that task object "
                            "toward a concrete placement. Do not act on unrelated objects. Prefer a "
                            "task-directed command over help, look, inventory, or examine when one is "
                            "available. Choose one option numbered from 1 and output only its integer index. "
                            "Use the recent action outcomes; do not repeat a command that produced no "
                            "new observation or state change."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Hypothesis: {}\nObservation:\n{}\nRecent action outcomes:\n{}"
                            "\nValid option range: 1 to {}\nAdmissible commands:\n{}"
                        ).format(
                            hypothesis,
                            view["observation"],
                            json.dumps(action_history[-5:], ensure_ascii=False),
                            len(available_commands),
                            command_list,
                        ),
                    },
                ]
                generation = model.generate(
                    messages,
                    max_new_tokens=4,
                    temperature=config["model"]["temperature"],
                    top_p=config["model"]["top_p"],
                )
                generations.append(generation)
                selection = generation["generated_text"].strip()
                selected_index = None
                if re.fullmatch(r"\d+", selection):
                    selected_index = int(selection) - 1
                else:
                    lines = [line.strip() for line in selection.splitlines() if line.strip()]
                    if lines and re.fullmatch(r"\d+", lines[0]):
                        selected_index = int(lines[0]) - 1
                    else:
                        match = re.fullmatch(r"(\d+): (.+)", lines[-1]) if lines else None
                        if match:
                            numbered_index = int(match.group(1)) - 1
                            if (
                                0 <= numbered_index < len(available_commands)
                                and match.group(2) == available_commands[numbered_index]
                            ):
                                selected_index = numbered_index
                if selected_index is None or not 0 <= selected_index < len(available_commands):
                    correction = model.generate(
                        messages + [
                            {"role": "assistant", "content": selection},
                            {
                                "role": "user",
                                "content": (
                                    "Your reply was not one valid action index. Choose one integer "
                                    "from 1 to {} and output only that integer."
                                ).format(len(available_commands)),
                            },
                        ],
                        max_new_tokens=8,
                        temperature=0.0,
                        top_p=1.0,
                    )
                    generations.append(correction)
                    corrected_selection = correction["generated_text"].strip()
                    if not re.fullmatch(r"\d+", corrected_selection):
                        raise ValueError(
                            "Proposer returned invalid option {!r} and its correction was {!r}"
                            .format(selection, corrected_selection)
                        )
                    selected_index = int(corrected_selection) - 1
                    if not 0 <= selected_index < len(available_commands):
                        raise ValueError(
                            "Proposer returned invalid option {!r} and its correction selected "
                            "option {} from a list of {} commands".format(
                                selection, corrected_selection, len(available_commands)
                            )
                        )
            action = available_commands[selected_index]
            result = env.step(action)
            actions.append(action)
            lowered_action = action.lower()
            if lowered_action.startswith("go to "):
                visited_locations.add(action[6:].strip().lower())
            elif lowered_action.startswith("open "):
                opened_containers.add(lowered_action)
            if target_object and re.match(
                r"^take\s+{}(?:\s|$)".format(re.escape(target_object)), action, re.IGNORECASE
            ):
                carrying_target = True
                visited_locations.clear()
            elif target_object and re.match(
                r"^(?:move|put)\s+{}(?:\s|$)".format(re.escape(target_object)), action, re.IGNORECASE
            ):
                carrying_target = False
                visited_locations.clear()
                placement = PUT_INTO.match(action.strip().rstrip("."))
                placed_target_objects.add(placement.group(2).lower())
            current_facts = {fact_key(fact) for fact in result["facts"]}
            action_history.append(
                {
                    "action": action,
                    "observation": result["observation"],
                    "new_facts": fact_records(current_facts - previous_facts),
                }
            )
            if current_facts == previous_facts and result["observation"] == view["observation"]:
                blocked_actions.add(action)
            else:
                blocked_actions.clear()
            if PUT_INTO.match(action.strip().rstrip(".")):
                for facts, goal_kind, object_type in placement_candidates(previous_facts, current_facts):
                    atoms = fact_records(facts)
                    goal_key = tuple(fact_key(atom) for atom in atoms)
                    if goal_key in seen_goals:
                        continue
                    seen_goals.add(goal_key)
                    task_candidate_count += 1
                    task = make_task(
                        env.game_file,
                        task_dir,
                        source_split,
                        actions,
                        atoms,
                        goal_kind,
                        action,
                        object_type,
                    )
                    if task is None:
                        continue
                    replay = replay_witness(config, task)
                    witness_replay_steps += len(replay["actions"])
                    if replay["valid"]:
                        tasks.append(task)
                        verified_task_count += 1
            previous_facts = current_facts
            view = result
            if result["done"]:
                break
    return {
        "tasks": tasks,
        "generations": generations,
        "hypothesis": hypothesis,
        "actions": actions,
        "trajectory": action_history,
        "task_candidate_count": task_candidate_count,
        "verified_task_count": verified_task_count,
        "witness_replay_steps": witness_replay_steps,
    }


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
        )
        for task in exploration["tasks"]:
            if task.source_scene_id in seen_scenes:
                continue
            seen_scenes.add(task.source_scene_id)
            tasks.append(task)
            if len(tasks) >= count:
                return tasks
    return tasks
