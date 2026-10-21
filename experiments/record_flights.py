#!/usr/bin/env python3
"""Record many flights of one genome, with full trajectories, for failure analysis.

    python experiments/record_flights.py runs/<run> --episodes 1000

Seeds start at 3e9: separate from training (< 1e9), validation (1e9+) and test (2e9+).
Writes results/failure_analysis/<run>/flights.jsonl.gz (header line, then one flight per line).
"""

import argparse
import gzip
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neat_lander import runs  # noqa: E402
from neat_lander.env import CENTRED, make_env, make_net, seed_wind, wind_force  # noqa: E402

SEED0 = 3_000_000_000
_genome = _config = None


def _init(genome, config):
    global _genome, _config
    _genome, _config = genome, config


def fly(seed):
    """One flight: every state, action, the 4 output scores, reward, wind force, and why it ended."""
    env, net = make_env(), make_net(_genome, _config)
    seed_wind(env, seed)
    obs, _ = env.reset(seed=seed)
    states, actions, outputs, rewards, wind = [], [], [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        out = net.activate(obs)
        a = int(np.argmax(out))
        wind.append(round(wind_force(env), 4))
        states.append([round(float(v), 5) for v in obs])
        actions.append(a)
        outputs.append([round(float(v), 4) for v in out])
        obs, r, terminated, truncated, _ = env.step(a)
        rewards.append(round(float(r), 3))
        if len(actions) >= 1000:
            break
    u = env.unwrapped
    if terminated and rewards[-1] > 0:
        outcome, cause = 'landed', 'came to rest'
    elif terminated and abs(obs[0]) >= 1.0:
        outcome, cause = 'crashed', 'left the screen'
    elif terminated:
        outcome, cause = 'crashed', 'body hit the ground'
    else:
        outcome, cause = 'timeout', 'still flying after 1000 steps'
    on_pad = bool(abs(obs[0]) < 0.2)
    centred = bool(outcome == 'landed' and abs(obs[0]) < CENTRED)
    env.close()
    return {'seed': seed, 'outcome': outcome, 'cause': cause, 'reward': round(sum(rewards), 2),
            'steps': len(actions), 'final': [round(float(v), 5) for v in obs], 'on_pad': on_pad,
            'centred': centred, 'states': states, 'actions': actions, 'outputs': outputs, 'rewards': rewards,
            'wind': wind,
            'game_over': bool(getattr(u, 'game_over', False))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--episodes', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=mp.cpu_count())
    args = ap.parse_args()
    genome, config = runs.load_winner(args.run_dir), runs.load_config(args.run_dir)
    name = os.path.basename(os.path.normpath(args.run_dir))
    out_dir = os.path.join(runs.ROOT, 'results', 'failure_analysis', name)
    os.makedirs(out_dir, exist_ok=True)
    seeds = [SEED0 + i for i in range(args.episodes)]
    with mp.Pool(args.workers, initializer=_init, initargs=(genome, config)) as pool:
        flights = pool.map(fly, seeds, chunksize=20)
    path = os.path.join(out_dir, 'flights.jsonl.gz')
    with gzip.open(path, 'wt') as fh:
        fh.write(json.dumps({'run': name, 'genome': genome.key, 'seed0': SEED0,
                             'episodes': len(flights)}) + '\n')
        for f in flights:
            fh.write(json.dumps(f) + '\n')
    counts = {o: sum(f['outcome'] == o for f in flights) for o in ('landed', 'crashed', 'timeout')}
    print(f'{path}: {counts}')


if __name__ == '__main__':
    main()
