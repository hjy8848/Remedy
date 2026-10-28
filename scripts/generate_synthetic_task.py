"""Generate and verify a new Text ALFWorld task without selecting a FloorPlan."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

from experiment_rsi.alfworld import AlfworldEpisode
from experiment_rsi.config import load_config
from experiment_rsi.constructed import _parse_spec
from experiment_rsi.model import ModelSession, seed_everything
from experiment_rsi.solver_sft import collect_solver_sft_examples
from experiment_rsi.synthetic_world import build_synthetic_world, synthetic_world_messages
from experiment_rsi.synthesis import goal_reached
from run_skills_experiment import _planner_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--feedback', required=True)
    parser.add_argument('--logic-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--attempts', type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)
    config['model'].update({'path': args.model, 'revision': 'local'})
    feedback = json.loads(Path(args.feedback).read_text(encoding='utf-8'))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = ModelSession(config, device=args.device)
    messages = synthetic_world_messages(feedback)
    for attempt in range(1, args.attempts + 1):
        seed_everything(config['run']['seed'] + attempt, model.torch)
        generation = model.generate(messages, max_new_tokens=1536,
                                    temperature=config['model']['temperature'],
                                    top_p=config['model']['top_p'], enable_thinking=False)
        text = generation['generated_text']
        (out / ('attempt-{:02d}.txt'.format(attempt))).write_text(text, encoding='utf-8')
        try:
            spec = _parse_spec(text, feedback)
            scene = build_synthetic_world(spec, out / ('world-{:02d}'.format(attempt)), args.logic_dir)
        except (ValueError, KeyError, TypeError) as error:
            print('attempt {} rejected: {}'.format(attempt, error), flush=True)
            messages.extend([{'role': 'assistant', 'content': text},
                             {'role': 'user', 'content': 'The previous proposal was rejected: {}. Correct it using the catalogue.'.format(error)}])
            continue
        print('attempt {}: world constructed'.format(attempt), flush=True)
        break
    else:
        raise RuntimeError('no valid synthetic world after {} attempts'.format(args.attempts))
    del model
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    process = ctx.Process(target=_planner_worker, args=(config, scene, queue))
    process.start()
    process.join(20)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError('planner exceeded the 20-second search budget; solvability is unknown')
    if process.exitcode:
        raise RuntimeError('planner process failed with exit code {}'.format(process.exitcode))
    status, actions = queue.get(timeout=2)
    queue.close()
    if status != 'complete' or not actions:
        raise ValueError('planner failed or goal was already satisfied: {}'.format(status))
    if len(actions) > config['run']['max_steps']:
        raise ValueError('planner solution exceeds the Solver step budget')
    trace = []
    with AlfworldEpisode(config['alfworld']['config_path'], game_file=scene['game_file'], split='train') as env:
        state = env.reset()
        if goal_reached(scene['goal_atoms'], state['facts']):
            raise ValueError('goal already satisfied at reset')
        for action in actions:
            if action not in state['admissible_commands']:
                raise ValueError('planner action is not admissible: {}'.format(action))
            state = env.step(action)
            trace.append({'action': action, 'observation': state['observation'], 'won': state['won']})
        if not state['won'] or not goal_reached(scene['goal_atoms'], state['facts']):
            raise ValueError('planner solution failed ALFWorld replay')
    task = SimpleNamespace(task_id='synthetic-01', task_game_file=scene['game_file'], source_split='train',
                           instruction=scene['instruction'], goal_atoms=scene['goal_atoms'], witness_actions=actions)
    examples = collect_solver_sft_examples(config, [task])
    with (out / 'solver_sft.jsonl').open('w', encoding='utf-8') as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + '\n')
    (out / 'result.json').write_text(json.dumps({
        'spec': spec, 'scene': scene, 'witness_actions': actions, 'replay': trace,
        'replay_valid': True, 'sft_examples': len(examples),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('verified {} steps; created {} SFT examples'.format(len(actions), len(examples)), flush=True)


if __name__ == '__main__':
    main()
