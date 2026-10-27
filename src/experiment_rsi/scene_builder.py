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


def build_scene(source_game_file, spec, output_dir):
    source = Path(source_game_file).resolve()
    game = json.loads(source.read_text(encoding="utf-8"))
    problem = game["pddl_problem"]
    scene_id = source_scene_id(source)
    if spec["base_scene"] != scene_id:
        raise ValueError("base_scene must match the selected source scene: {}".format(scene_id))
    objects = spec["objects"]
    goal = spec["goal"]
    if len(objects) != 1 or goal["type"] != "place_all":
        raise ValueError("constructed scenes support one object type and a place_all goal")
    object_spec = objects[0]
    for label, value in (("object type", object_spec["type"]),
                         ("initial receptacle", object_spec["initial_receptacle"]),
                         ("goal object type", goal["object_type"]),
                         ("destination type", goal["destination_type"])):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("{} must be nonempty text".format(label))
    object_type = object_spec["type"].lower()
    count = object_spec["count"]
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError("object count must be an integer between 1 and 3")
    if goal["object_type"].lower() != object_type or goal["count"] != count:
        raise ValueError("goal object type and count must match the object specification")
    scene = inspect_scene(problem)
    if object_type not in scene["objects"]:
        raise ValueError("source scene has no pickupable {} template".format(object_type))
    initial_type = object_spec["initial_receptacle"].lower()
    destination_type = goal["destination_type"].lower()
    if initial_type not in scene["receptacles"] or destination_type not in scene["receptacles"]:
        raise ValueError("initial and destination receptacles must exist in the source scene")
    facts = scene["facts"]
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
    selected = existing[:count]
    removed = set(existing[count:])
    for predicate, args in facts:
        if predicate == "inreceptacleobject" and any(item in existing for item in args):
            raise ValueError("objects nested inside movable receptacles are not supported")
        if args and args[0] not in removed and any(item in removed for item in args[1:]):
            raise ValueError("removing an object would leave another object's state inconsistent")
    template = existing[0]
    for index in range(count - len(selected)):
        selected.append(template + "_remedy_{}".format(index + 1))
    _, _, source_goal = pddl_goal_section(problem)
    objects_start, objects_end = _section(problem, "objects")
    declarations = problem[objects_start:objects_end]
    for object_id in removed:
        declarations = re.sub(r"(?im)^\s*" + re.escape(object_id) + r"\s+-\s+object\s*$", "", declarations)
    new_ids = selected[len(existing):]
    declarations = declarations[:-1] + "\n" + "\n".join("{} - object".format(i) for i in new_ids) + "\n)"
    problem = problem[:objects_start] + declarations + problem[objects_end:]

    init_start, init_end = _section(problem, "init")
    initial = problem[init_start:init_end]
    def remove_fact(match):
        predicate = match.group(1).lower()
        args = match.group(2).lower().split()
        if any(arg in removed for arg in args):
            return ""
        if args and args[0] in selected and predicate in {"inreceptacle", "objectatlocation"}:
            return ""
        return match.group(0)
    initial = PDDL_ATOM.sub(remove_fact, initial)
    additions = []
    for object_id in new_ids:
        additions.extend(["(objectType {} {}Type)".format(object_id, object_type),
                          "(pickupable {})".format(object_id)])
    for object_id in selected:
        additions.extend(["(inReceptacle {} {})".format(object_id, initial_id),
                          "(objectAtLocation {} {})".format(object_id, locations[0])])
    initial = initial[:-1] + "\n" + "\n".join(additions) + "\n)"
    problem = problem[:init_start] + initial + problem[init_end:]
    goal_atoms = [{"predicate": "inreceptacle", "arguments": [item, destination_id]} for item in selected]
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
    return {"task_id": task_id, "game_file": str(game_file), "source_game_file": str(source),
            "source_scene_id": scene_id, "source_goal_text": source_goal,
            "instruction": "Put {} {} in {} {}.".format(
                count, object_type, destination_type,
                len(scene["receptacles"][destination_type]) - scene["receptacles"][destination_type].index(destination_id)),
            "goal_atoms": goal_atoms, "object_ids": selected, "initial_receptacle_id": initial_id,
            "destination_id": destination_id, "spec": spec}
