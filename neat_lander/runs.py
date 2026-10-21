"""Run directories: where training writes and every other script reads.

runs/<name>/
  config.ini                 copy of the NEAT config (written when the run starts)
  data/fitness.csv           one row per generation
  data/complexity.csv        network size per generation
  data/species.csv           one row per species per generation
  data/validation.csv        each generation's champion on 100 unseen episodes
  data/population.csv.gz     every genome of every generation (fitness, species, size, parents)
  data/innovations.json      the generation each connection / node first appeared
  genomes/champions.pkl      {generation: best genome of that generation}
  genomes/winner.pkl         the champion with the best validation score
  populations/pop_gNNNN.pkl.gz   whole populations at selected generations
  checkpoints/               neat checkpoints for --resume
"""

import csv
import glob
import gzip
import os
import pickle
import re
from datetime import datetime

import neat

from .genome import MemoryGenome

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, 'runs')
SUBDIRS = ('data', 'genomes', 'populations', 'checkpoints')


def new_run_dir(config_path, name=None, pop_size=None):
    """Create the run directory and copy the config into it (with an optional pop_size override)."""
    name = name or datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run_dir = os.path.join(RUNS_DIR, name)
    if os.path.exists(os.path.join(run_dir, 'config.ini')):
        raise SystemExit(f'{run_dir} already exists. Use --resume to continue it.')
    for sub in SUBDIRS:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    with open(config_path) as fh:
        text = fh.read()
    if pop_size:
        text = re.sub(r'(?m)^pop_size\s*=.*$', f'pop_size              = {pop_size}', text)
    with open(os.path.join(run_dir, 'config.ini'), 'w') as fh:
        fh.write(text)
    return run_dir


def latest_run():
    runs = sorted(glob.glob(os.path.join(RUNS_DIR, 'run_*')), key=os.path.getmtime)
    if not runs:
        raise SystemExit('No runs found in runs/. Train first: python train.py')
    return runs[-1]


def load_config(path):
    """A neat Config. A [MemoryGenome] section (run D) selects MemoryGenome, [DefaultGenome] the stock one."""
    if os.path.isdir(path):
        path = os.path.join(path, 'config.ini')
    with open(path) as fh:
        genome = MemoryGenome if '[MemoryGenome]' in fh.read() else neat.DefaultGenome
    return neat.Config(genome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, path)


def path(run_dir, *parts):
    return os.path.join(run_dir, *parts)


def read_csv(run_dir, name):
    """Rows of data/<name>.csv as dicts with numbers converted."""
    with open(path(run_dir, 'data', name + '.csv')) as fh:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def read_population(run_dir):
    with gzip.open(path(run_dir, 'data', 'population.csv.gz'), 'rt') as fh:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def load_pickle(p):
    with open(p, 'rb') as fh:
        gz = fh.read(2) == b'\x1f\x8b'     # neat checkpoints are gzipped without a .gz suffix
    with (gzip.open if gz else open)(p, 'rb') as fh:
        return pickle.load(fh)


def save_pickle(obj, p):
    """Write atomically, so a killed run never leaves half a file."""
    opener = gzip.open if p.endswith('.gz') else open
    tmp = p + '.tmp'
    with opener(tmp, 'wb') as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, p)


def load_champions(run_dir):
    return load_pickle(path(run_dir, 'genomes', 'champions.pkl'))


def load_winner(run_dir):
    """The best-validating champion, or the latest champion if training is still running."""
    p = path(run_dir, 'genomes', 'winner.pkl')
    if os.path.exists(p):
        return load_pickle(p)
    champs = load_champions(run_dir)
    return champs[max(champs)]


def population_snapshots(run_dir):
    """{generation: path} of the saved whole populations."""
    out = {}
    for p in glob.glob(path(run_dir, 'populations', 'pop_g*.pkl.gz')):
        out[int(os.path.basename(p)[5:9])] = p
    return dict(sorted(out.items()))


def load_genome_arg(arg):
    """A CLI argument that is a run dir (-> winner) or a genome .pkl (-> its run's config)."""
    if os.path.isdir(arg):
        return load_winner(arg), load_config(arg), arg
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(arg)))
    return load_pickle(arg), load_config(run_dir), run_dir


def _num(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v
