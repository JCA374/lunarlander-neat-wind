#!/usr/bin/env python3
"""Paired comparison of run winners on the same windy flights, with bootstrap confidence intervals.

    python experiments/compare_runs.py A=runs/run_wind_20261020 B=runs/run_wind_b C=runs/run_wind_c

Every winner flies the same 1,000 analysis seeds (3e9+, never used in training, validation or test),
so differences are paired per flight. For winners with loops, a "memory off" variant (every loop
connection disabled) flies the same seeds too. Writes results/comparison/summary.json.
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neat_lander import runs  # noqa: E402
from neat_lander.env import CENTRED, make_env, make_net, run_episode, wind_force  # noqa: E402,F401
from neat_lander.recorder import network_size  # noqa: E402

SEED0 = 3_000_000_000
BOOT = 10_000


def loop_connections(genome):
    """Enabled connections that close a cycle: a -> b where b can already reach a (or a == b)."""
    conns = [k for k, c in genome.connections.items() if c.enabled]
    out = {}
    for a, b in conns:
        out.setdefault(a, []).append(b)

    def reaches(start, goal):
        seen, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n == goal:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(out.get(n, []))
        return False
    return [(a, b) for a, b in conns if a == b or reaches(b, a)]


def _fly(args):
    genome, config, seeds = args
    env, net = make_env(), make_net(genome, config)
    rows = []
    for s in seeds:
        e = run_episode(net, s, env=env, path=True)
        w = np.array(e['wind'])
        rows.append((e['reward'], e['landed'], e['landed'] and abs(e['x_final']) < CENTRED,
                     float(w.mean()) if len(w) else 0.0))
    env.close()
    return rows


def fly_all(pool, genome, config, seeds, workers):
    chunks = [list(c) for c in np.array_split(seeds, workers * 4)]
    parts = pool.map(_fly, [(genome, config, c) for c in chunks])
    rows = [r for p in parts for r in p]
    a = np.array(rows, dtype=float)
    return {'reward': a[:, 0], 'landed': a[:, 1], 'centred': a[:, 2], 'mean_wind': a[:, 3]}


def boot_ci(x, rng):
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('runs', nargs='+', help='NAME=run_dir')
    ap.add_argument('--episodes', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=mp.cpu_count())
    args = ap.parse_args()
    seeds = [SEED0 + i for i in range(args.episodes)]
    rng = np.random.default_rng(0)
    res, info = {}, {}
    with mp.Pool(args.workers) as pool:
        for spec in args.runs:
            name, run_dir = spec.split('=', 1)
            genome, config = runs.load_winner(run_dir), runs.load_config(run_dir)
            with open(runs.path(run_dir, 'genomes', 'winner.json')) as fh:
                w = json.load(fh)
            loops = loop_connections(genome)
            h, c = network_size(genome, config)
            info[name] = {'run': os.path.basename(os.path.normpath(run_dir)), 'generation': w['generation'],
                          'hidden': h, 'connections': c, 'loops': [list(k) for k in loops],
                          'recurrent': not config.genome_config.feed_forward}
            print(f'{name}: generation {w["generation"]}, {h} hidden, {c} connections, {len(loops)} loops', flush=True)
            res[name] = fly_all(pool, genome, config, seeds, args.workers)
            if loops:
                g = copy.deepcopy(genome)
                for k in loops:
                    g.connections[k].enabled = False
                res[name + ' memory off'] = fly_all(pool, g, config, seeds, args.workers)

    out = {'episodes': args.episodes, 'seed0': SEED0, 'winners': info, 'results': {}, 'paired': {},
           'by_wind': {}}
    for name, r in res.items():
        out['results'][name] = {k: float(r[k].mean()) for k in ('reward', 'landed', 'centred')}
        out['results'][name].update({k + '_ci': boot_ci(r[k], rng) for k in ('reward', 'landed', 'centred')})
        mw = r['mean_wind']
        edges = [-16, -7, -2, 2, 7, 16]
        out['by_wind'][name] = [{'range': [lo, hi], 'n': int(((mw >= lo) & (mw < hi)).sum()),
                                 'fail': float(1 - r['landed'][(mw >= lo) & (mw < hi)].mean())
                                 if ((mw >= lo) & (mw < hi)).any() else None}
                                for lo, hi in zip(edges[:-1], edges[1:])]
    names = list(res)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = {k: res[b][k] - res[a][k] for k in ('reward', 'landed', 'centred')}
            out['paired'][f'{b} vs {a}'] = {k: {'mean': float(v.mean()), 'ci': boot_ci(v, rng)} for k, v in d.items()}
    d = os.path.join(runs.ROOT, 'results', 'comparison')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print()
    for name, r in out['results'].items():
        print(f'{name:18s} score {r["reward"]:6.1f} [{r["reward_ci"][0]:.0f}, {r["reward_ci"][1]:.0f}]  '
              f'landed {r["landed"]:.0%}  centred {r["centred"]:.0%}')
    for k, v in out['paired'].items():
        print(f'{k:32s} score {v["reward"]["mean"]:+6.1f} [{v["reward"]["ci"][0]:+.0f}, {v["reward"]["ci"][1]:+.0f}]  '
              f'landed {v["landed"]["mean"]:+.1%}')


if __name__ == '__main__':
    main()
