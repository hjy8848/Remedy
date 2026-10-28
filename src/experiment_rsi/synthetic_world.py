"""Build a new Text ALFWorld placement world from its public domain rules."""

import json
from pathlib import Path


def synthetic_world_messages(feedback):
    """Ask the Designer to propose a world from the ALFWorld type catalogue."""
    from alfworld.gen import constants

    object_types = set(constants.OBJECTS)
    compatibility = {
        name.lower(): sorted(receptacle.lower() for receptacle, objects in
                             constants.VAL_RECEPTACLE_OBJECTS.items() if name in objects)
        for name in sorted(object_types)
    }
    compatibility = {name: receptacles for name, receptacles in compatibility.items()
                     if len(receptacles) >= 2}
    return [
        {"role": "system", "content": (
            "Design an ALFWorld placement task that investigates the Solver feedback. "
            "Create a new room by choosing receptacle types and object groups from the supplied "
            "compatibility catalogue. Use only object types present as catalogue keys, and choose "
            "both receptacles literally from that object's list; absent types and guessed pairs are "
            "invalid. Return only a JSON object with hypothesis, diagnostic_behavior, "
            "feedback_refs, receptacles, and objects. receptacles is a list of distinct receptacle "
            "types. Each objects entry has type, count, initial_receptacle, and destination_receptacle. "
            "Both receptacles must be listed and must be different; the object must be compatible "
            "with both according to that object's catalogue entry. Cite at least one supplied feedback reference. Use the task to investigate "
            "a failure, and do not make it artificially easy. Do not write PDDL."
        )},
        {"role": "user", "content": json.dumps({
            "solver_feedback": feedback,
            "object_to_compatible_receptacles": compatibility,
        }, ensure_ascii=False)},
    ]


def build_synthetic_world(spec, output_dir, logic_dir):
    """Compile an object-placement specification without copying a FloorPlan.

    Receptacles are listed by type, with one instance per type. Each object
    group specifies its initial and destination receptacle types.
    """
    from alfworld.gen import constants

    object_types = {name.lower(): name for name in constants.OBJECTS}
    receptacle_types = {name.lower(): name for name in constants.VAL_RECEPTACLE_OBJECTS}
    receptacles = [name.lower() for name in spec["receptacles"]]
    if not receptacles or len(receptacles) != len(set(receptacles)):
        raise ValueError("receptacles must be a nonempty list of distinct types")
    if any(name not in receptacle_types for name in receptacles):
        raise ValueError("unknown ALFWorld receptacle type")
    groups = spec["objects"]
    if not groups:
        raise ValueError("at least one object group is required")

    object_names = []
    object_records = []
    for group in groups:
        name = group["type"].lower()
        initial = group["initial_receptacle"].lower()
        destination = group["destination_receptacle"].lower()
        count = group["count"]
        if name not in object_types:
            raise ValueError("unknown ALFWorld object type: {}".format(name))
        if name in object_names:
            raise ValueError("each object type must appear in only one group")
        if type(count) is not int or count < 1:
            raise ValueError("object count must be a positive integer")
        if initial not in receptacles or destination not in receptacles:
            raise ValueError("initial and destination receptacles must be declared")
        if initial == destination:
            raise ValueError("initial and destination receptacles must differ")
        canonical = object_types[name]
        for receptacle in (initial, destination):
            if canonical not in constants.VAL_RECEPTACLE_OBJECTS[receptacle_types[receptacle]]:
                raise ValueError("{} cannot contain {}".format(receptacle, name))
        object_names.append(name)
        object_records.append((canonical, count, initial, destination))

    receptacle_ids = {name: '{}_bar_z'.format(receptacle_types[name]) for name in receptacles}
    locations = {name: 'loc_bar_{}'.format(index + 1) for index, name in enumerate(receptacles)}
    object_ids = {
        name.lower(): ['{}_bar_{}'.format(name, index + 1) for index in range(count)]
        for name, count, _, _ in object_records
    }
    obj_types = {name + 'Type' for name, _, _, _ in object_records} | {'KnifeType', 'ButterKnifeType'}
    rec_types = {receptacle_types[name] + 'Type' for name in receptacles} | {
        'SinkBasinType', 'MicrowaveType', 'FridgeType'
    }
    declarations = ['agent1 - agent', 'loc_start - location']
    declarations += ['{} - location'.format(locations[name]) for name in receptacles]
    declarations += ['{} - receptacle'.format(receptacle_ids[name]) for name in receptacles]
    declarations += ['{} - object'.format(item) for ids in object_ids.values() for item in ids]
    declarations += ['{} - otype'.format(name) for name in sorted(obj_types)]
    declarations += ['{} - rtype'.format(name) for name in sorted(rec_types)]
    facts = ['(atLocation agent1 loc_start)']
    for name in receptacles:
        rec_id = receptacle_ids[name]
        canonical = receptacle_types[name]
        facts.extend(['(receptacleType {} {}Type)'.format(rec_id, canonical),
                      '(receptacleAtLocation {} {})'.format(rec_id, locations[name])])
        if canonical in constants.OPENABLE_CLASS_SET:
            facts.append('(openable {})'.format(rec_id))
        for object_type, _, _, _ in object_records:
            if object_type in constants.VAL_RECEPTACLE_OBJECTS[canonical]:
                facts.append('(canContain {}Type {}Type)'.format(canonical, object_type))
    goal_atoms = []
    instructions = []
    for name, count, initial, destination in object_records:
        for item in object_ids[name.lower()]:
            facts.extend(['(objectType {} {}Type)'.format(item, name),
                          '(pickupable {})'.format(item),
                          '(inReceptacle {} {})'.format(item, receptacle_ids[initial]),
                          '(objectAtLocation {} {})'.format(item, locations[initial])])
            goal_atoms.append({'predicate': 'inreceptacle',
                               'arguments': [item.lower(), receptacle_ids[destination].lower()]})
        instructions.append('put {} {} in {} 1'.format(count, name.lower(), destination))
    instruction = ' and '.join(instructions) + '.'
    goal = ['({} {})'.format(atom['predicate'], ' '.join(atom['arguments'])) for atom in goal_atoms]
    problem = '(define (problem remedy-generated)\n(:domain alfred)\n(:objects\n{}\n)\n(:init\n{}\n)\n(:goal (and\n{}\n))\n)\n'.format(
        '\n'.join(declarations), '\n'.join(facts), '\n'.join(goal))
    logic = Path(logic_dir)
    grammar = (logic / 'alfred.twl2').read_text(encoding='utf-8')
    grammar = grammar.replace('Your task is to: UNKNOWN GOAL.', 'Your task is to: ' + instruction, 1)
    game = {'pddl_domain': (logic / 'alfred.pddl').read_text(encoding='utf-8'),
            'grammar': grammar, 'pddl_problem': problem, 'solvable': None, 'walkthrough': None}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    game_file = out / 'game.tw-pddl'
    game_file.write_text(json.dumps(game, ensure_ascii=False), encoding='utf-8')
    (out / 'initial_state.pddl').write_text(problem, encoding='utf-8')
    return {'game_file': str(game_file.resolve()), 'instruction': instruction,
            'goal_atoms': goal_atoms, 'spec': spec}
