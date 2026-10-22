"""Fitness: how well does a genome fly?

fitness = the average LunarLander reward over a few episodes, plus a bonus for landing in the middle
between the flags (see env.centre_bonus). Every genome of a generation flies the
*same* starting positions (seeds), so they are compared fairly; the seeds change every generation, so
nobody can specialise on one start. Validation uses 100 fixed seeds that training never sees.
"""

import multiprocessing as mp

import numpy as np
from neat.reporting import BaseReporter

from .env import CENTRED, PAD_HALF_WIDTH, make_env, make_net, run_episode

TRAIN_SEED_BASE = 1234          # training seeds for generation g come from rng(TRAIN_SEED_BASE + g)
TRAIN_SEED_MAX = 10**9          # training seeds are drawn below this ...
VALIDATION_SEEDS = [TRAIN_SEED_MAX + i for i in range(100)]   # ... validation seeds start at it


def training_seeds(generation, episodes):
    rng = np.random.default_rng(TRAIN_SEED_BASE + generation)
    return [int(s) for s in rng.integers(0, TRAIN_SEED_MAX, episodes)]


def reeval_seeds(generation, episodes):
    """Extra training seeds for re-flying a generation's top genomes (a separate stream, still < 1e9)."""
    rng = np.random.default_rng([TRAIN_SEED_BASE, generation, 1])
    return [int(s) for s in rng.integers(0, TRAIN_SEED_MAX, episodes)]


def fly(genome, config, seeds, steady=0.0):
    """Fly a genome on each seed; returns a summary dict (the mean reward is the fitness)."""
    net = make_net(genome, config)
    env = make_env()
    eps = [run_episode(net, s, env=env, steady=steady) for s in seeds]
    env.close()
    return summarize(eps)


SUMMARY_KEYS = ('fitness', 'reward', 'landed', 'centred', 'abs_x', 'crashed', 'steps')


def summarize(eps):
    fitness = np.array([e['fitness'] for e in eps])
    landed = [e for e in eps if e['landed']]
    return {'fitness': float(fitness.mean()),
            'std': float(fitness.std()),
            'reward': float(np.mean([e['reward'] for e in eps])),
            'landed': len(landed) / len(eps),
            'centred': float(np.mean([e['landed'] and abs(e['x_final']) < CENTRED for e in eps])),
            # how far from the middle the landings were (1.0 = at a flag; nan if nothing landed)
            'abs_x': (float(np.mean([abs(e['x_final']) for e in landed]) / PAD_HALF_WIDTH)
                      if landed else float('nan')),
            'crashed': float(np.mean([e['outcome'] == 'crashed' for e in eps])),
            'steps': float(np.mean([e['steps'] for e in eps]))}


# --- multiprocessing workers (the config is sent once per worker, not once per genome) ---------

_config = None


def _init_worker(config):
    global _config
    _config = config


def _fly_job(args):
    genome, seeds, steady = args if len(args) == 3 else (*args, 0.0)
    return fly(genome, _config, seeds, steady)


def curriculum_steady(generation, curriculum):
    """Weight of the steady push in training: 1 until curriculum[0], fading to 0 at curriculum[1]."""
    if not curriculum:
        return 0.0
    g0, g1 = curriculum
    return float(np.clip((g1 - generation) / max(1, g1 - g0), 0.0, 1.0))


class Evaluator(BaseReporter):
    """The fitness function passed to `Population.run`, flying genomes on all CPU cores.

    It is also a reporter, so neat tells it the generation number (which picks the seeds), and it
    keeps each genome's summary in `self.results` for the logger.
    """

    def __init__(self, config, episodes=5, workers=None, reeval_top=0, reeval_episodes=0, curriculum=None):
        self.episodes = episodes
        # Curriculum (g0, g1): training flies a steady one-sided push until g0, blends to the real gusty wind
        # by g1. Validation always flies the real task.
        self.curriculum = curriculum
        # Five windy episodes rank the population well overall but pick lucky genomes at the top, so
        # the best `reeval_top` genomes fly `reeval_episodes` more before elites and species are chosen
        self.reeval_top = reeval_top
        self.reeval_episodes = reeval_episodes
        self.generation = 0
        self.results = {}
        self.pool = mp.Pool(workers or mp.cpu_count(), initializer=_init_worker, initargs=(config,))

    def start_generation(self, generation):
        self.generation = generation

    def __call__(self, genomes, config):
        seeds = training_seeds(self.generation, self.episodes)
        steady = curriculum_steady(self.generation, self.curriculum)
        summaries = self.pool.map(_fly_job, [(g, seeds, steady) for _, g in genomes])
        self.results = {}
        for (gid, genome), s in zip(genomes, summaries):
            genome.fitness = s['fitness']
            self.results[gid] = s
        if self.reeval_top and self.reeval_episodes:
            top = sorted(genomes, key=lambda kv: -kv[1].fitness)[:self.reeval_top]
            extra = self.pool.map(_fly_job, [(g, reeval_seeds(self.generation, self.reeval_episodes), steady)
                                             for _, g in top])
            n0, n1 = self.episodes, self.reeval_episodes
            for (gid, genome), s in zip(top, extra):
                merged = {k: (self.results[gid][k] * n0 + s[k] * n1) / (n0 + n1)
                          for k in s if k not in ('std', 'abs_x')}
                merged.update(std=s['std'], abs_x=s['abs_x'])
                genome.fitness = merged['fitness']
                self.results[gid] = merged

    def validate(self, genome, seeds=VALIDATION_SEEDS):
        """Fly one genome on many seeds in parallel; returns the summary."""
        chunks = [list(c) for c in np.array_split(seeds, min(len(seeds), self.pool._processes))]
        parts = self.pool.map(_fly_job, [(genome, c) for c in chunks])
        weights = np.array([len(c) for c in chunks], dtype=float)
        out = {k: float(np.average([p[k] for p in parts], weights=weights))
               for k in SUMMARY_KEYS if k != 'abs_x'}
        # abs_x averages over landings only
        lw = np.array([p['landed'] for p in parts]) * weights
        out['abs_x'] = (float(np.average([p['abs_x'] for p in parts if p['landed']],
                                         weights=lw[lw > 0])) if lw.sum() else float('nan'))
        return out

    def __getstate__(self):
        # neat checkpoints pickle the species set, which holds the reporters; skip the pool
        return {'episodes': self.episodes, 'generation': self.generation, 'results': {}, 'pool': None,
                'reeval_top': self.reeval_top, 'reeval_episodes': self.reeval_episodes,
                'curriculum': self.curriculum}

    def close(self):
        self.pool.close()
        self.pool.join()
