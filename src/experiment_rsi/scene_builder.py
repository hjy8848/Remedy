"""Construct placement experiments from an existing Text ALFWorld game."""

import hashlib
import json
import re
import shutil
from pathlib import Path

from .synthesis import PDDL_ATOM, pddl_goal_section, pddl_goal_satisfied_initially, replace_pddl_goal, source_scene_id


def _section(problem, name):
    start = problem.lower().find("(:" + name)
    if start < 0:
        raise ValueError("ALFWorld PDDL problem has no :{} section".format(name))
    depth = 0
    for index in range(start, len(problem)):
        if problem[index] == "(":
            depth += 1
        elif problem[index] == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise ValueError("ALFWorld PDDL :{} section is not balanced".format(name))


def inspect_scene(problem):
    start, end = _section(problem, "init")
    facts = [(m.group(1).lower(), m.group(2).lower().split())
             for m in PDDL_ATOM.finditer(problem[start:end])]
    pickupable = {a[0] for p, a in facts if p == "pickupable" and len(a) == 1}
    objects, receptacles = {}, {}
    for predicate, args in facts:
        if predicate == "objecttype" and len(args) == 2 and args[0] in pickupable:
            objects.setdefault(args[1].removesuffix("type"), []).append(args[0])
        elif predicate == "receptacletype" and len(args) == 2:
            receptacles.setdefault(args[1].removesuffix("type"), []).append(args[0])
    return {"objects": {k: sorted(v) for k, v in objects.items()},
            "receptacles": {k: sorted(v) for k, v in receptacles.items()},
            "facts": facts}


def _goal_placements(goal):
    if goal["type"] != "place_all":
        raise ValueError("constructed scenes require a place_all goal")
    if "placements" in goal:
        placements = goal["placements"]
    else:
        placements = [{
            "object_type": goal["object_type"],
            "count": goal["count"],
            "destination_type": goal["destination_type"],
        }]
    if not isinstance(placements, list) or not placements:
        raise ValueError("place_all goal requires at least one placement")
    return placements


def build_scene(source_game_file, spec, output_dir):
    source = Path(source_game_file).resolve()
    game = json.loads(source.read_text(encoding="utf-8"))
    problem = game["pddl_problem"]
    scene_id = source_scene_id(source)
    if spec["base_scene"] != scene_id:
        raise ValueError("base_scene must match the selected source scene: {}".format(scene_id))
    objects = spec["objects"]
    if not isinstance(objects, list) or not objects:
        raise ValueError("objects must be a nonempty list")
    if len(objects) > 3:
        raise ValueError("constructed scenes support at most three target object types")
    placements = _goal_placements(spec["goal"])
    object_specs = {}
    for object_spec in objects:
        for label, value in (("object type", object_spec["type"]),
                             ("initial receptacle", object_spec["initial_receptacle"])):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{} must be nonempty text".format(label))
        object_type = object_spec["type"].strip().lower()
        if object_type in object_specs:
            raise ValueError("each object type must appear once in objects")
        count = object_spec["count"]
        if type(count) is not int or not 1 <= count <= 3:
            raise ValueError("each object count must be an integer between 1 and 3")
        object_specs[object_type] = {
            "count": count,
            "initial_type": object_spec["initial_receptacle"].strip().lower(),
        }

    goal_specs = {}
    for placement in placements:
        for label, value in (("goal object type", placement["object_type"]),
                             ("destination type", placement["destination_type"])):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{} must be nonempty text".format(label))
        object_type = placement["object_type"].strip().lower()
        if object_type in goal_specs:
            raise ValueError("each object type must have one destination")
        count = placement["count"]
        if type(count) is not int or not 1 <= count <= 3:
            raise ValueError("each goal count must be an integer between 1 and 3")
        goal_specs[object_type] = {
            "count": count,
            "destination_type": placement["destination_type"].strip().lower(),
        }
    if set(object_specs) != set(goal_specs):
        raise ValueError("objects and goal placements must contain the same object types")
    for object_type, object_spec in object_specs.items():
        if object_spec["count"] != goal_specs[object_type]["count"]:
            raise ValueError("goal count must match the object specification for {}".format(object_type))

    scene = inspect_scene(problem)
    facts = scene["facts"]
    selected_by_type = {}
    initial_by_type = {}
    destination_by_type = {}
    removed = set()
    new_ids_by_type = {}
    for object_type, object_spec in object_specs.items():
        if object_type not in scene["objects"]:
            raise ValueError("source scene has no pickupable {} template".format(object_type))
        initial_type = object_spec["initial_type"]
        destination_type = goal_specs[object_type]["destination_type"]
        if initial_type not in scene["receptacles"] or destination_type not in scene["receptacles"]:
            raise ValueError("initial and destination receptacles must exist in the source scene")
        for receptacle_type in (initial_type, destination_type):
            if ("cancontain", [receptacle_type + "type", object_type + "type"]) not in facts:
                raise ValueError("{} cannot contain {} in this ALFWorld scene".format(receptacle_type, object_type))
        initial_id = scene["receptacles"][initial_type][0]
        destinations = [item for item in scene["receptacles"][destination_type] if item != initial_id]
        if not destinations:
            raise ValueError("initial and destination must be different receptacle instances")
        destination_id = destinations[0]
        locations = [a[1] for p, a in facts if p == "receptacleatlocation" and a[0] == initial_id]
        if not locations:
            raise ValueError("initial receptacle has no navigation location")

        existing = scene["objects"][object_type]
        selected = existing[:object_spec["count"]]
        removed.update(existing[object_spec["count"]:])
        template = existing[0]
        new_ids = [template + "_remedy_{}".format(index + 1)
                   for index in range(object_spec["count"] - len(selected))]
        selected.extend(new_ids)
        selected_by_type[object_type] = selected
        new_ids_by_type[object_type] = new_ids
        initial_by_type[object_type] = {"id": initial_id, "location": locations[0]}
        destination_by_type[object_type] = destination_id

    targeted_objects = {item for values in selected_by_type.values() for item in values}
    targeted_objects.update(removed)
    for predicate, args in facts:
        if predicate == "inreceptacleobject" and any(item in targeted_objects for item in args):
            raise ValueError("objects nested inside movable receptacles are not supported")
        if args and args[0] not in removed and any(item in removed for item in args[1:]):
            raise ValueError("removing an object would leave another object's state inconsistent")

    _, _, source_goal = pddl_goal_section(problem)
    objects_start, objects_end = _section(problem, "objects")
    declarations = problem[objects_start:objects_end]
    for object_id in removed:
        declarations = re.sub(r"(?im)^\s*" + re.escape(object_id) + r"\s+-\s+object\s*$", "", declarations)
    new_ids = [item for values in new_ids_by_type.values() for item in values]
    declarations = declarations[:-1] + "\n" + "\n".join("{} - object".format(i) for i in new_ids) + "\n)"
    problem = problem[:objects_start] + declarations + problem[objects_end:]

    init_start, init_end = _section(problem, "init")
    initial = problem[init_start:init_end]
    selected_objects = {item for values in selected_by_type.values() for item in values}

    def remove_fact(match):
        predicate = match.group(1).lower()
        args = match.group(2).lower().split()
        if any(arg in removed for arg in args):
            return ""
        if args and args[0] in selected_objects and predicate in {"inreceptacle", "objectatlocation"}:
            return ""
        return match.group(0)

    initial = PDDL_ATOM.sub(remove_fact, initial)
    additions = []
    for object_type, ids in new_ids_by_type.items():
        for object_id in ids:
            additions.extend(["(objectType {} {}Type)".format(object_id, object_type),
                              "(pickupable {})".format(object_id)])
    for object_type, ids in selected_by_type.items():
        initial_id = initial_by_type[object_type]["id"]
        location = initial_by_type[object_type]["location"]
        for object_id in ids:
            additions.extend(["(inReceptacle {} {})".format(object_id, initial_id),
                              "(objectAtLocation {} {})".format(object_id, location)])
    initial = initial[:-1] + "\n" + "\n".join(additions) + "\n)"
    problem = problem[:init_start] + initial + problem[init_end:]
    goal_atoms = [
        {"predicate": "inreceptacle", "arguments": [item, destination_by_type[object_type]]}
        for object_type, ids in selected_by_type.items()
        for item in ids
    ]
    if pddl_goal_satisfied_initially(problem, goal_atoms):
        raise ValueError("constructed goal is already satisfied initially")
    problem = replace_pddl_goal(problem, goal_atoms)
    task_id = hashlib.sha256(json.dumps({"source": str(source), "spec": spec}, sort_keys=True).encode()).hexdigest()[:16]
    task_dir = Path(output_dir).resolve() / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    metadata = source.parent / "traj_data.json"
    if metadata.exists():
        shutil.copy2(metadata, task_dir / "traj_data.json")
    (task_dir / "initial_state.pddl").write_text(problem, encoding="utf-8")
    game["pddl_problem"] = problem
    game["solvable"] = None
    game["walkthrough"] = None
    game_file = task_dir / "game.tw-pddl"
    game_file.write_text(json.dumps(game, ensure_ascii=False), encoding="utf-8")
    placement_instructions = []
    for object_type, ids in selected_by_type.items():
        placement = goal_specs[object_type]
        destination_type = placement["destination_type"]
        destination_id = destination_by_type[object_type]
        destination_number = (
            len(scene["receptacles"][destination_type])
            - scene["receptacles"][destination_type].index(destination_id)
        )
        count = len(ids)
        noun = "item" if count == 1 else "items"
        placement_instructions.append((
            "move exactly {} {} {} to {} {}".format(
                count, object_type, noun, destination_type, destination_number
            ),
            count, object_type, destination_type, destination_number,
        ))
    if len(placement_instructions) == 1:
        _, count, object_type, destination_type, destination_number = placement_instructions[0]
        instruction = "Put {} {} in {} {}.".format(
            count, object_type, destination_type, destination_number
        )
    else:
        instruction = "Complete every placement: " + "; then ".join(
            item[0] for item in placement_instructions
        ) + "."
    if len(selected_by_type) == 1:
        only_type = next(iter(selected_by_type))
        object_ids = selected_by_type[only_type]
        initial_receptacle_id = initial_by_type[only_type]["id"]
        destination_id = destination_by_type[only_type]
    else:
        object_ids = selected_by_type
        initial_receptacle_id = initial_by_type
        destination_id = destination_by_type
    return {"task_id": task_id, "game_file": str(game_file), "source_game_file": str(source),
            "source_scene_id": scene_id, "source_goal_text": source_goal,
            "instruction": instruction, "goal_atoms": goal_atoms,
            "object_ids": object_ids, "initial_receptacle_id": initial_receptacle_id,
            "destination_id": destination_id, "placements": placements, "spec": spec}
