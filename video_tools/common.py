"""
Shared pieces for the video scenes: drawing helpers (Pillow), flying genomes, and `Context`, which
loads a training run and caches the expensive bits (evaluations, replays) on disk.

Ported from Moonlander-Genetic's video_tools/scenes.py; the network drawing lives in
neat_lander/draw.py so the static pictures and the video look the same.
"""

import copy
import hashlib
import json
import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from neat_lander import labels, runs  # noqa: E402
from neat_lander.draw import (AMBER, BG, BLUE, FG, GREEN, MUTED, PANEL, RED, VIOLET,  # noqa: E402,F401
                              font, mix)
from neat_lander.env import (CENTRED, MAX_STEPS, PAD_HALF_WIDTH, VIEW_H, VIEW_W, WIND_POWER,  # noqa: E402,F401
                             make_env, make_net, run_episode)
from neat_lander.recorder import network_size  # noqa: E402
from timeline import Scene  # noqa: E402,F401

W, H, FPS = 1280, 720, 30
TEST_SEED0 = 2_000_000_000          # same test seeds as evaluate.py
ACTION_COLORS = [MUTED, VIOLET, AMBER, (64, 210, 220)]
ABLATION_EPISODES = 40


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------

def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fade(t, start, dur=0.4):
    return ease((t - start) / dur)


def blank():
    return Image.new('RGB', (W, H), BG)


def text(d, xy, s, size=24, fill=FG, a=1.0, bold=False, mono=False, anchor='la', bg=BG):
    if a <= 0:
        return
    d.text(xy, s, font=font(size, bold, mono), fill=mix(bg, fill, a), anchor=anchor)


def heading(d, title, t, sub=None):
    a = fade(t, 0)
    text(d, (60, 42), title, 38, FG, a, bold=True)
    d.rectangle([60, 92, 60 + int(120 * a), 96], fill=AMBER)
    if sub:
        text(d, (60 + 140, 84), sub, 18, MUTED, a)


def paste_frame(img, frame, box, border=None, dim=0.0):
    x, y, w, h = box
    tile = Image.fromarray(frame).resize((w, h), Image.BILINEAR)
    if dim > 0:
        tile = Image.blend(tile, Image.new('RGB', (w, h), BG), dim)
    img.paste(tile, (x, y))
    if border:
        ImageDraw.Draw(img).rectangle([x - 3, y - 3, x + w + 2, y + h + 2], outline=border, width=3)


def frame_at(frames, t, speed=1.0, loop_pause=1.5):
    """Gameplay frame for time t, looping with a pause on the last frame."""
    n = len(frames)
    k = int(t * FPS * speed) % (n + int(loop_pause * FPS))
    return frames[min(k, n - 1)]


def step_at(n, t, speed=1.0, loop_pause=1.5):
    """Index into an n-step recording for time t (same looping as frame_at)."""
    k = int(t * FPS * speed) % (n + int(loop_pause * FPS))
    return min(k, n - 1)


def outcome(ep):
    return {'landed': ('landed', GREEN), 'crashed': ('crashed', RED),
            'timeout': ('timed out', MUTED)}[ep['outcome']]


def wrap(s, width):
    words, lines, cur = s.split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w).strip()
    return lines + [cur] if cur else lines


def wind_gauge(img, box, force, a=1.0):
    """A wind arrow in the top-left corner of a game frame at `box`; length = force / WIND_POWER."""
    if force is None or a <= 0:
        return
    d = ImageDraw.Draw(img)
    x, y, w, h = box
    s = max(0.6, w / 600)
    cx, cy, half = x + 90 * s, y + 24 * s, 70 * s
    d.line([cx - half, cy, cx + half, cy], fill=mix(BG, MUTED, 0.5 * a), width=1)
    tip = cx + half * max(-1.0, min(1.0, force / WIND_POWER))
    if abs(tip - cx) > 3:
        sgn = 1 if tip > cx else -1
        col = mix(BG, (64, 210, 220), a)
        d.line([cx, cy, tip, cy], fill=col, width=max(3, int(6 * s)))
        d.polygon([(tip + sgn * 10 * s, cy), (tip - sgn * 4 * s, cy - 10 * s), (tip - sgn * 4 * s, cy + 10 * s)],
                  fill=col)
    text(d, (cx, cy + 14 * s), 'wind', max(12, int(16 * s)), MUTED, a, anchor='ma')


def paste_flight(img, ep, k, box, border=None, dim=0.0):
    """Frame k of a recorded episode, with its wind arrow."""
    k = min(k, len(ep['frames']) - 1)
    paste_frame(img, ep['frames'][k], box, border, dim)
    if ep.get('wind'):
        wind_gauge(img, box, ep['wind'][min(k, len(ep['wind']) - 1)], 1.0 - dim)


def terrain_frame(seed):
    """The moon surface for `seed`, with the lander moved out of the picture."""
    env = make_env('rgb_array')
    env.reset(seed=seed)
    u = env.unwrapped
    for body in [u.lander] + u.legs:
        body.position = (body.position.x + 1000, body.position.y)
    frame = env.render()
    env.close()
    return frame


# ----------------------------------------------------------------------------
# Flying genomes
# ----------------------------------------------------------------------------

class Run:
    """Everything a scene may want to know about the training run."""

    def __init__(self, run_dir):
        self.dir = run_dir
        self.config = runs.load_config(run_dir)
        self.champions = runs.load_champions(run_dir)
        self.winner = runs.load_winner(run_dir)
        self.fitness = runs.read_csv(run_dir, 'fitness')
        self.validation = runs.read_csv(run_dir, 'validation')
        self.complexity = runs.read_csv(run_dir, 'complexity')
        self.species = runs.read_csv(run_dir, 'species')
        self.population = runs.read_population(run_dir)
        self.snapshots = runs.population_snapshots(run_dir)
        self.G = len(self.fitness)
        self.threshold = self.config.fitness_threshold
        solved = [r['generation'] for r in self.validation if r['reward'] >= self.threshold]
        self.solved_at = solved[0] if solved else None
        self.winner_gen = max(self.validation, key=lambda r: r['fitness'])['generation']

    def size(self, genome):
        return network_size(genome, self.config)

    def snapshot(self, gen):
        return runs.load_pickle(self.snapshots[gen])


def fly(genome, config, seed, frames=True, path=False, trace=False):
    return run_episode(make_net(genome, config), seed, frames=frames, path=path, trace=trace)


def test(genome, config, episodes=100, wind=True):
    env = make_env(wind=wind)
    net = make_net(genome, config)
    eps = [run_episode(net, TEST_SEED0 + i, env=env) for i in range(episodes)]
    env.close()
    return {'landing_rate': 100 * np.mean([e['landed'] for e in eps]),
            'centred_rate': 100 * np.mean([e['landed'] and abs(e['x_final']) < CENTRED for e in eps]),
            'avg_reward': float(np.mean([e['reward'] for e in eps])),
            'avg_steps': float(np.mean([e['steps'] for e in eps]))}


class Context:
    """The run plus cached results. Cache entries are keyed by the winner, so they refresh after
    retraining."""

    def __init__(self, run_dir, cache_dir):
        self.run = Run(run_dir)
        self.config = self.run.config
        self.winner = self.run.winner
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.key = hashlib.sha1(pickle.dumps(sorted(self.winner.connections.items(),
                                                    key=lambda kv: kv[0]))).hexdigest()[:12]
        self.key += f'-{self.run.G}'
        with open(runs.path(run_dir, 'train_args.json')) as fh:
            self.episodes = json.load(fh)['episodes']
        hidden = sorted(k for k in self.winner.nodes if k not in self.config.genome_config.output_keys)
        self.hidden_alias = {k: f'H{i + 1}' for i, k in enumerate(hidden)}
        self.stats = self.cached('stats', lambda: test(self.winner, self.config))
        self.seed = self.cached('seed', self.pick_showcase_seed)
        self.best_ep = fly(self.winner, self.config, self.seed, trace=True)

    def node_name(self, key):
        """Short node name; the winner's hidden nodes are called H1, H2, ..."""
        return self.hidden_alias.get(key) or labels.node_name(key)

    def cached(self, name, fn):
        path = os.path.join(self.cache_dir, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as fh:
                key, value = pickle.load(fh)
            if key == self.key:
                return value
        value = fn()
        with open(path, 'wb') as fh:
            pickle.dump((self.key, value), fh)
        return value

    def pick_showcase_seed(self, candidates=range(1, 200)):
        """A seed where the winner lands near the middle of the pad and then sits still."""
        best = None
        for s in candidates:
            ep = fly(self.winner, self.config, s, frames=False, trace=True)
            if not (ep['landed'] and 180 < ep['steps'] < 500):
                continue
            st, a = np.asarray(ep['states']), np.asarray(ep['actions'])
            settle = int(np.sum((st[:, 6] > 0.5) & (st[:, 7] > 0.5) & (a != 0)))
            key = (settle, round(abs(float(st[-1, 0])), 2))
            if best is None or key < best[0]:
                best = (key, s)
        return best[1] if best else 1

    def ghosts(self, gens, seed):
        """Flight paths of every genome of each generation in `gens`, all flying `seed`."""
        def run():
            out = {}
            for g in gens:
                snap = self.run.snapshot(g)
                eps = []
                for gid, genome in snap['population'].items():
                    ep = fly(genome, self.config, seed, frames=False, path=True)
                    eps.append({'path': ep['path'], 'outcome': ep['outcome'], 'reward': ep['reward'],
                                'species': snap['species'].get(gid), 'fitness': genome.fitness})
                out[g] = eps
            return out
        return self.cached(f'ghosts_{seed}_{"-".join(map(str, gens))}', run)

    def ablation(self, episodes=ABLATION_EPISODES):
        """Landing rate with each enabled connection of the winner switched off, one at a time."""
        def run():
            seeds = [TEST_SEED0 + 500 + i for i in range(episodes)]

            def rate(genome):
                env, net = make_env(), make_net(genome, self.config)
                r = np.mean([run_episode(net, s, env=env)['landed'] for s in seeds])
                env.close()
                return float(r)
            rates = {}
            for key, c in self.winner.connections.items():
                if c.enabled:
                    g = copy.deepcopy(self.winner)
                    g.connections[key].enabled = False
                    rates[key] = rate(g)
            return {'full': rate(self.winner), 'rates': rates,
                    'ranked': sorted(rates.items(), key=lambda kv: kv[1])}
        return self.cached('ablation', run)

    def baseline(self):
        """The no-wind champion (experiments/baseline/) without and with wind, and a test seed where
        it misses the middle in wind while the new winner lands there."""
        def run():
            d = os.path.join(REPO_ROOT, 'experiments', 'baseline')
            genome = runs.load_pickle(os.path.join(d, 'nowind_winner.pkl'))
            config = runs.load_config(os.path.join(d, 'nowind_config.ini'))
            out = {'calm': test(genome, config, wind=False), 'wind': test(genome, config)}
            seed = TEST_SEED0
            for s in range(TEST_SEED0, TEST_SEED0 + 100):
                old = fly(genome, config, s, frames=False)
                new = fly(self.winner, self.config, s, frames=False)
                if (old['outcome'] == 'crashed' and new['landed'] and abs(new['x_final']) < CENTRED
                        and new['steps'] < 450):
                    seed = s
                    break
            out['seed'] = seed
            return out
        out = self.cached('baseline', run)
        d = os.path.join(REPO_ROOT, 'experiments', 'baseline')
        out['genome'] = runs.load_pickle(os.path.join(d, 'nowind_winner.pkl'))
        out['config'] = runs.load_config(os.path.join(d, 'nowind_config.ini'))
        return out
