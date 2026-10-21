"""Why did evolution stall? Checks on the training run itself, for analyze_flights.py.

1. Selection noise: whole population snapshots are re-flown on 100 fresh episodes each. How well does
   the 5-episode training fitness that NEAT selects on rank the genomes by their 100-episode fitness?
2. Progress: the champion's unseen score over the run, with a rolling median.
3. Bloat: network size against the 100-episode fitness.
4. Diversity: species count and the share of the biggest species.
5. Reference controllers: Gymnasium's hand-written (reactive) heuristic, calm and windy, next to the
   evolved winner and the calm-air champion.

Episodes use analysis seeds 3e9 + 100,000 onwards (disjoint from training, validation and test).
"""

import multiprocessing as mp
import os

import numpy as np

from neat_lander import runs
from neat_lander.env import CENTRED, make_env, seed_wind
from neat_lander.fitness import fly

SEED0 = 3_000_100_000
EPISODES = 100
CHUNK = 5               # training uses 5 episodes per genome
_config = None


def _init(config):
    global _config
    _config = config


def _job(args):
    genome, seeds = args
    out = []
    for i in range(0, len(seeds), CHUNK):           # per-5-episode fitness, so noise can be measured
        out.append(fly(genome, _config, seeds[i:i + CHUNK])['fitness'])
    return out


def reevaluate(run_dir, gens, cache):
    """{gen: {'train': 5-ep training fitness, 'true': 100-ep fitness, 'chunks': 20 x 5-ep fitness,
    'hidden', 'conns'}} for every genome of each snapshot generation."""
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)['data'].item()
        if all(g in d for g in gens):
            return d
    config = runs.load_config(run_dir)
    snaps = runs.population_snapshots(run_dir)
    seeds = [SEED0 + i for i in range(EPISODES)]
    out = {}
    with mp.Pool(mp.cpu_count(), initializer=_init, initargs=(config,)) as pool:
        for g in gens:
            pop = runs.load_pickle(snaps[g])['population']
            genomes = list(pop.values())
            chunks = np.array(pool.map(_job, [(gn, seeds) for gn in genomes]))
            out[g] = {'train': np.array([gn.fitness for gn in genomes]), 'chunks': chunks,
                      'true': chunks.mean(axis=1),
                      'hidden': np.array([len(gn.nodes) - config.genome_config.num_outputs for gn in genomes]),
                      'conns': np.array([sum(c.enabled for c in gn.connections.values()) for gn in genomes])}
            print(f'  re-flew generation {g}: {len(genomes)} genomes x {EPISODES} episodes', flush=True)
    np.savez_compressed(cache, data=np.array(out, dtype=object))
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def noise_stats(d):
    """How much of the 5-episode fitness is luck."""
    chunks = d['chunks']
    within = float(np.mean(chunks.std(axis=1)))          # one genome, different 5-episode sets
    between = float(d['true'].std())                       # different genomes, 100 episodes each
    # test-retest: two independent 5-episode scores of the same genome
    retest = float(np.corrcoef(chunks[:, 0], chunks[:, 1])[0, 1])
    top = int(np.argmax(d['train']))
    pct = float(np.mean(d['true'] <= d['true'][top]))     # percentile of the 5-ep best by true fitness
    return {'within_sd': within, 'between_sd': between, 'retest_r': retest,
            'rank_corr_train_true': spearman(d['train'], d['true']),
            'train_best_true_percentile': pct, 'train_best_train': float(d['train'][top]),
            'train_best_true': float(d['true'][top]), 'true_best': float(d['true'].max())}


def heuristic_scores(n=200, seed0=SEED0 + 50_000):
    from gymnasium.envs.box2d.lunar_lander import heuristic
    out = {}
    for wind in (False, True):
        env = make_env(wind=wind)
        R, L, C = [], [], []
        for i in range(n):
            s = seed0 + i
            seed_wind(env, s)
            o, _ = env.reset(seed=s)
            tot, r, term = 0.0, 0.0, False
            for _ in range(1000):
                o, r, term, trunc, _ = env.step(heuristic(env.unwrapped, o))
                tot += r
                if term or trunc:
                    break
            landed = bool(term and r > 0)
            R.append(tot)
            L.append(landed)
            C.append(landed and abs(o[0]) < CENTRED)
        env.close()
        out['wind' if wind else 'calm'] = {'reward': float(np.mean(R)), 'landed': float(np.mean(L)),
                                           'centred': float(np.mean(C))}
    return out
