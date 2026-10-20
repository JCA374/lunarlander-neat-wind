"""EvolutionRecorder: logs everything about a run, every generation, so it can be replayed later.

It writes as it goes (nothing waits for the end of training), so an interrupted run is still usable.
"""

import copy
import csv
import gzip
import json
import os
import time

import numpy as np
from neat.reporting import BaseReporter

from . import runs

FITNESS_COLS = ['generation', 'best', 'mean', 'median', 'worst', 'std', 'species',
                'landers', 'seconds']
COMPLEXITY_COLS = ['generation', 'mean_hidden', 'mean_conns', 'best_hidden', 'best_conns',
                   'max_hidden', 'max_conns']
SPECIES_COLS = ['generation', 'species', 'size', 'age', 'best', 'mean', 'stagnation']
# fitness = reward + centre bonus (what selects the winner); reward = the plain env score (what
# 'solved' is judged on); abs_x = mean landing distance from the middle, 1.0 = at a flag
VALIDATION_COLS = ['generation', 'genome', 'fitness', 'reward', 'landed', 'centred', 'abs_x',
                   'crashed', 'steps', 'hidden', 'conns']
POPULATION_COLS = ['generation', 'genome', 'species', 'fitness', 'landed', 'crashed', 'steps',
                   'hidden', 'conns', 'parent1', 'parent2']


def network_size(genome, config):
    """(hidden nodes, enabled connections)."""
    hidden = len(genome.nodes) - config.genome_config.num_outputs
    conns = sum(1 for c in genome.connections.values() if c.enabled)
    return hidden, conns


class EvolutionRecorder(BaseReporter):
    def __init__(self, run_dir, config, evaluator, reproduction, generations, snapshot_every=10):
        self.run_dir = run_dir
        self.config = config
        self.evaluator = evaluator
        self.reproduction = reproduction
        self.last_generation = generations - 1
        self.snapshot_every = snapshot_every
        self.champions = {}
        self.innovations = {'connections': {}, 'nodes': {}}
        self.winner_score = -np.inf
        self.solved_at = None
        self.t0 = time.time()
        for name, cols in [('fitness', FITNESS_COLS), ('complexity', COMPLEXITY_COLS),
                           ('species', SPECIES_COLS), ('validation', VALIDATION_COLS)]:
            p = runs.path(run_dir, 'data', name + '.csv')
            if not os.path.exists(p):
                with open(p, 'w', newline='') as fh:
                    csv.writer(fh).writerow(cols)
        p = runs.path(run_dir, 'data', 'population.csv.gz')
        if not os.path.exists(p):
            with gzip.open(p, 'wt', newline='') as fh:
                csv.writer(fh).writerow(POPULATION_COLS)

    def __getstate__(self):
        # Checkpoints pickle the reporters (via the species set); the logged state lives on disk
        return {'run_dir': self.run_dir}

    # --- resuming --------------------------------------------------------------------------------

    def resume_from(self, generation):
        """Drop everything logged at or after `generation` (it is about to be re-run)."""
        for name in ('fitness', 'complexity', 'species', 'validation'):
            p = runs.path(self.run_dir, 'data', name + '.csv')
            with open(p) as fh:
                rows = list(csv.reader(fh))
            keep = [rows[0]] + [r for r in rows[1:] if int(r[0]) < generation]
            with open(p, 'w', newline='') as fh:
                csv.writer(fh).writerows(keep)
        p = runs.path(self.run_dir, 'data', 'population.csv.gz')
        with gzip.open(p, 'rt') as fh:
            rows = list(csv.reader(fh))
        with gzip.open(p, 'wt', newline='') as fh:
            csv.writer(fh).writerows([rows[0]] + [r for r in rows[1:] if int(r[0]) < generation])

        p = runs.path(self.run_dir, 'genomes', 'champions.pkl')
        if os.path.exists(p):
            self.champions = {g: c for g, c in runs.load_pickle(p).items() if g < generation}
        p = runs.path(self.run_dir, 'data', 'innovations.json')
        if os.path.exists(p):
            with open(p) as fh:
                inn = json.load(fh)
            self.innovations = {kind: {k: g for k, g in d.items() if g < generation}
                                for kind, d in inn.items()}
        for row in runs.read_csv(self.run_dir, 'validation'):
            if row['fitness'] > self.winner_score:
                self.winner_score = row['fitness']
            if self.solved_at is None and row['reward'] >= self.config.fitness_threshold:
                self.solved_at = row['generation']

    # --- neat reporter hooks ---------------------------------------------------------------------

    def start_generation(self, generation):
        self.generation = generation
        self.gen_start = time.time()

    def post_evaluate(self, config, population, species_set, best_genome):
        gen = self.generation
        results = self.evaluator.results
        fits = np.array([g.fitness for g in population.values()])
        sizes = {gid: network_size(g, config) for gid, g in population.items()}
        best_size = sizes[best_genome.key]

        # Every genome of this generation
        rows = []
        for gid, g in population.items():
            r = results.get(gid, {})
            parents = self.reproduction.ancestors.get(gid, ())
            rows.append([gen, gid, species_set.genome_to_species.get(gid, ''), f'{g.fitness:.2f}',
                         r.get('landed', ''), r.get('crashed', ''), r.get('steps', ''),
                         *sizes[gid], *(list(parents) + ['', ''])[:2]])
        with gzip.open(runs.path(self.run_dir, 'data', 'population.csv.gz'), 'at', newline='') as fh:
            csv.writer(fh).writerows(rows)

        # Species
        species_rows = []
        for sid, s in sorted(species_set.species.items()):
            f = [population[m].fitness for m in s.members if m in population]
            if not f:
                continue
            species_rows.append([gen, sid, len(f), gen - s.created, f'{max(f):.2f}',
                                 f'{np.mean(f):.2f}', gen - s.last_improved])
        self._append('species', species_rows)

        # The generation's champion on 100 unseen episodes
        val = self.evaluator.validate(best_genome)
        self._append('validation', [[gen, best_genome.key, f"{val['fitness']:.2f}",
                                     f"{val['reward']:.2f}", f"{val['landed']:.3f}",
                                     f"{val['centred']:.3f}", f"{val['abs_x']:.3f}",
                                     f"{val['crashed']:.3f}", f"{val['steps']:.1f}", *best_size]])
        self.champions[gen] = copy.deepcopy(best_genome)
        runs.save_pickle(self.champions, runs.path(self.run_dir, 'genomes', 'champions.pkl'))
        if val['fitness'] > self.winner_score:
            self.winner_score = val['fitness']
            runs.save_pickle(best_genome, runs.path(self.run_dir, 'genomes', 'winner.pkl'))
            with open(runs.path(self.run_dir, 'genomes', 'winner.json'), 'w') as fh:
                json.dump({'generation': gen, 'genome': best_genome.key, 'training_fitness':
                           best_genome.fitness, 'validation': val, 'hidden': best_size[0],
                           'connections': best_size[1]}, fh, indent=2)
        if self.solved_at is None and val['reward'] >= config.fitness_threshold:
            self.solved_at = gen

        # Where each gene first appeared (neat-python names a connection gene by its (in, out)
        # pair; that pair plays the role of the "innovation number" from the NEAT paper)
        for g in population.values():
            for (a, b) in g.connections:
                self.innovations['connections'].setdefault(f'{a},{b}', gen)
            for n in g.nodes:
                self.innovations['nodes'].setdefault(str(n), gen)
        with open(runs.path(self.run_dir, 'data', 'innovations.json'), 'w') as fh:
            json.dump(self.innovations, fh)

        # Whole populations (for replaying every pilot of a generation)
        if gen <= 10 or gen % self.snapshot_every == 0 or gen == self.last_generation:
            runs.save_pickle({'population': population,
                              'species': dict(species_set.genome_to_species)},
                             runs.path(self.run_dir, 'populations', f'pop_g{gen:04d}.pkl.gz'))

        landers = np.mean([r['landed'] >= 0.5 for r in results.values()]) if results else 0.0
        secs = time.time() - self.gen_start
        self._append('fitness', [[gen, f'{fits.max():.2f}', f'{fits.mean():.2f}',
                                  f'{np.median(fits):.2f}', f'{fits.min():.2f}', f'{fits.std():.2f}',
                                  len(species_set.species), f'{landers:.3f}', f'{secs:.1f}']])
        self._append('complexity', [[gen, f'{np.mean([s[0] for s in sizes.values()]):.2f}',
                                     f'{np.mean([s[1] for s in sizes.values()]):.2f}', *best_size,
                                     max(s[0] for s in sizes.values()),
                                     max(s[1] for s in sizes.values())]])

        solved = f'  solved@{self.solved_at}' if self.solved_at is not None else ''
        print(f'gen {gen:4d} | best {fits.max():7.1f}  mean {fits.mean():7.1f} | '
              f'champion on 100 new: {val["reward"]:7.1f} ({100 * val["landed"]:3.0f}% land, '
              f'{100 * val["centred"]:3.0f}% centred) | '
              f'species {len(species_set.species):2d} | net {best_size[0]}h/{best_size[1]}c | '
              f'{secs:4.1f}s{solved}', flush=True)

    def _append(self, name, rows):
        with open(runs.path(self.run_dir, 'data', name + '.csv'), 'a', newline='') as fh:
            csv.writer(fh).writerows(rows)
