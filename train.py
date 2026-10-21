#!/usr/bin/env python3
"""Train NEAT on LunarLander-v2 with wind.

    python train.py                          # new run in runs/run_YYYYMMDD_HHMMSS/
    python train.py --generations 300 --workers 8
    python train.py --resume runs/<run>/checkpoints/gen-120   # continue an interrupted run

Each generation:
  1. every genome is turned into a network and flies a few windy episodes
     (fitness = average reward + a bonus for landing in the middle between the flags),
  2. the generation's best genome flies 100 *unseen* episodes (validation),
  3. NEAT keeps the best of each species, breeds and mutates the next generation.
Everything is logged to the run directory as it happens (see neat_lander/runs.py).
"""

import argparse
import itertools
import json
import os
import random
import time

import neat

from neat_lander import runs
from neat_lander.fitness import Evaluator
from neat_lander.recorder import EvolutionRecorder


def build_population(args):
    if args.resume:
        # A checkpoint holds the population right after breeding generation N-1 (= ready for N)
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.resume)))
        config = runs.load_config(run_dir)
        generation, _, population, species_set, rndstate = runs.load_pickle(args.resume)
        random.setstate(rndstate)
        pop = neat.Population(config, (population, species_set, generation))
        # neat-python 0.92 restarts genome ids at 1 after a restore; continue after the largest id
        pop.reproduction.genome_indexer = itertools.count(max(population) + 1)
        pop.species.reporters = pop.reporters   # drop the reporters pickled into the checkpoint
        # ... and new hidden-node ids must not reuse ids already in the population
        last_node = max(n for g in population.values() for n in g.nodes)
        config.genome_config.node_indexer = itertools.count(last_node + 1)
        return pop, config, run_dir
    run_dir = runs.new_run_dir(args.config, args.name, pop_size=args.pop)
    config = runs.load_config(run_dir)
    if args.seed is not None:
        random.seed(args.seed)
    return neat.Population(config), config, run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--config', default='neat-config.ini')
    ap.add_argument('--generations', type=int, default=300, help='total generations of the run')
    ap.add_argument('--episodes', type=int, default=5, help='episodes per genome per generation')
    ap.add_argument('--workers', type=int, default=None, help='CPU processes (default: all)')
    ap.add_argument('--pop', type=int, default=None, help='override pop_size (quick tests)')
    ap.add_argument('--seed', type=int, default=None, help='random seed for NEAT itself')
    ap.add_argument('--name', default=None, help='run directory name')
    ap.add_argument('--resume', default=None, help='checkpoint file to continue from')
    ap.add_argument('--reeval-top', type=int, default=0,
                    help='re-fly the best N genomes of each generation on more episodes (0 = off)')
    ap.add_argument('--reeval-episodes', type=int, default=30, help='extra episodes for --reeval-top')
    args = ap.parse_args()

    pop, config, run_dir = build_population(args)
    start = pop.generation
    remaining = args.generations - start
    if remaining <= 0:
        raise SystemExit(f'Run already has {start} generations (--generations {args.generations}).')
    with open(runs.path(run_dir, 'train_args.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=2)

    evaluator = Evaluator(config, episodes=args.episodes, workers=args.workers,
                          reeval_top=args.reeval_top, reeval_episodes=args.reeval_episodes)
    recorder = EvolutionRecorder(run_dir, config, evaluator, pop.reproduction, args.generations)
    if args.resume:
        recorder.resume_from(start)
    pop.add_reporter(evaluator)
    pop.add_reporter(recorder)
    pop.add_reporter(neat.Checkpointer(generation_interval=10, time_interval_seconds=None,
                                       filename_prefix=runs.path(run_dir, 'checkpoints', 'gen-')))

    print(f'Run: {run_dir}\nPopulation {config.pop_size}, generations {start}..{args.generations - 1}, '
          f'{args.episodes} episodes per genome, solved = 100-episode average >= '
          f'{config.fitness_threshold:.0f}\n')
    t0 = time.time()
    try:
        pop.run(evaluator, remaining)
    finally:
        evaluator.close()

    winner = runs.path(run_dir, 'genomes', 'winner.json')
    with open(winner) as fh:
        w = json.load(fh)
    print(f'\nDone in {(time.time() - t0) / 60:.1f} min.')
    print(f"Winner: generation {w['generation']}, {w['hidden']} hidden nodes, {w['connections']} "
          f"connections, {w['validation']['reward']:.1f} average reward, "
          f"{100 * w['validation']['landed']:.0f}% landings and {100 * w['validation']['centred']:.0f}% "
          f"centred on 100 unseen episodes.")
    if recorder.solved_at is not None:
        print(f'Solved (>= {config.fitness_threshold:.0f}) first at generation {recorder.solved_at}.')
    print(f'\nNext:  python demo.py {run_dir}\n       python visualize_all.py {run_dir}')


if __name__ == '__main__':
    main()
