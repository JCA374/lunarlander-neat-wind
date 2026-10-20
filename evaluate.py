#!/usr/bin/env python3
"""Test a genome on many windy episodes it never trained on.

    python evaluate.py runs/<run>                    # the run's winner
    python evaluate.py runs/<run>/genomes/x.pkl --episodes 200
"""

import argparse

import numpy as np

from neat_lander import runs
from neat_lander.env import CENTRED, PAD_HALF_WIDTH, make_env, make_net, run_episode
from neat_lander.recorder import network_size

TEST_SEED0 = 2_000_000_000     # different from training and validation seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('target', nargs='?', default=None, help='run dir or genome .pkl')
    ap.add_argument('--episodes', type=int, default=100)
    args = ap.parse_args()
    genome, config, _ = runs.load_genome_arg(args.target or runs.latest_run())

    net, env = make_net(genome, config), make_env()
    eps = [run_episode(net, TEST_SEED0 + i, env=env) for i in range(args.episodes)]
    env.close()
    r = np.array([e['reward'] for e in eps])
    outcomes = [e['outcome'] for e in eps]
    hidden, conns = network_size(genome, config)
    print(f'Network: {hidden} hidden nodes, {conns} enabled connections')
    print(f'{args.episodes} unseen episodes: average reward {r.mean():.1f} (std {r.std():.1f}, '
          f'min {r.min():.0f}, max {r.max():.0f})')
    for o in ('landed', 'crashed', 'timeout'):
        print(f'  {o:8s} {100 * outcomes.count(o) / len(eps):5.1f}%')
    landed = [e for e in eps if e['landed']]
    if landed:
        xs = np.abs([e['x_final'] for e in landed]) / PAD_HALF_WIDTH
        print(f'  centred  {100 * np.mean(xs < CENTRED / PAD_HALF_WIDTH) * len(landed) / len(eps):5.1f}%'
              f'  (landed within {CENTRED} of the middle; the flags are at {PAD_HALF_WIDTH})')
        print(f'  landings between the flags {100 * np.mean(xs <= 1):.0f}%, average distance from the '
              f'middle {xs.mean():.2f} (1.0 = at a flag)')
    print(f'  average fitness (reward + centre bonus) {np.mean([e["fitness"] for e in eps]):.1f}')
    print(f'  average episode length {np.mean([e["steps"] for e in eps]):.0f} steps')
    print('SOLVED' if r.mean() >= config.fitness_threshold else 'not solved',
          f'(threshold {config.fitness_threshold:.0f})')


if __name__ == '__main__':
    main()
