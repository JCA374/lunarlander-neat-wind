#!/usr/bin/env python3
"""Failure analysis of one genome's flights -> results/failure_analysis/<run>/report.html

    python experiments/analyze_flights.py runs/<run>     # after record_flights.py

One figure per question: what failure looks like, where it happens, what drifts before it, what
predicts it (AUC and partial correlation beyond the network's own inputs), what causes it, and how
it starts. Then the training run itself (training_checks.py): how noisy selection is, whether it
stalled, bloat, diversity, and how a hand-written controller does in the same wind.
Findings are leads, not results: test them by changing the setup and retraining.
"""

import argparse
import gzip
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neat_lander import runs  # noqa: E402
from neat_lander.env import WIND_POWER, make_env, seed_wind  # noqa: E402
import training_checks as TC  # noqa: E402

SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BLUES = LinearSegmentedColormap.from_list('blues', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5',
                                                    '#1c5cab', '#0d366b'])
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2, 'font.family': 'DejaVu Sans',
})

K = 30          # "fails soon" = within K steps
WINDOW = 150    # run-up window drawn before failure
CURRENT = ['x', 'y', 'vx', 'vy', 'angle', 'spin', 'L leg', 'R leg']      # what the network sees
CANDIDATES = ['speed', '|angle|', '|spin|', 'tilt growing', 'distance', 'main duty', 'side duty',
              'switches', 'one leg down', 'margin', 'low & fast', 'low & off-centre', 'time',
              'wind', '|wind|', 'wind away', 'wind with drift', 'gust change', 'wind, last 50']
LABELS = {'x': 'x position', 'y': 'altitude', 'vx': 'horizontal velocity', 'vy': 'vertical velocity',
          'angle': 'angle', 'spin': 'angular velocity', 'L leg': 'left leg contact',
          'R leg': 'right leg contact', 'speed': 'speed |v|', '|angle|': '|angle|', '|spin|': '|angular velocity|',
          'tilt growing': 'angle × angular velocity (tilt growing)', 'distance': 'distance to pad',
          'main duty': 'main engine share, last 20 steps', 'side duty': 'side engine share, last 20 steps',
          'switches': 'action changes, last 20 steps', 'one leg down': 'exactly one leg down',
          'margin': 'best minus 2nd-best output', 'low & fast': 'falling speed below altitude 0.3',
          'low & off-centre': '|x| below altitude 0.3', 'time': 'steps flown',
          'wind': 'wind force (+ = to the right)', '|wind|': 'wind strength |wind|',
          'wind away': 'wind pushing away from the middle (wind × sign x)',
          'wind with drift': 'wind pushing along the drift (wind × vx)',
          'gust change': 'wind change over 20 steps', 'wind, last 50': '|wind| averaged over 50 steps'}


def load(path):
    with gzip.open(path, 'rt') as fh:
        header = json.loads(fh.readline())
        return header, [json.loads(line) for line in fh]


def rolling(a, n=20):
    c = np.cumsum(np.concatenate([[0.0], a]))
    idx = np.arange(1, len(a) + 1)
    lo = np.maximum(0, idx - n)
    return (c[idx] - c[lo]) / (idx - lo)


def signals(f):
    """{name: array over the flight's steps}."""
    s = np.array(f['states'])
    a = np.array(f['actions'])
    o = np.array(f['outputs'])
    srt = np.sort(o, axis=1)
    low = s[:, 1] < 0.3
    w = np.array(f['wind']) / WIND_POWER
    w20 = np.concatenate([np.zeros(min(20, len(w))), w[:-20]]) if len(w) > 20 else np.zeros(len(w))
    d = {k: s[:, i] for i, k in enumerate(CURRENT)}
    d.update({'wind': w, '|wind|': np.abs(w), 'wind away': w * np.sign(s[:, 0]), 'wind with drift': w * s[:, 2],
              'gust change': np.abs(w - w20), 'wind, last 50': rolling(np.abs(w), 50)})
    d.update({'speed': np.hypot(s[:, 2], s[:, 3]), '|angle|': np.abs(s[:, 4]), '|spin|': np.abs(s[:, 5]),
              'tilt growing': s[:, 4] * s[:, 5], 'distance': np.hypot(s[:, 0], s[:, 1]),
              'main duty': rolling((a == 2).astype(float)), 'side duty': rolling(np.isin(a, (1, 3)).astype(float)),
              'switches': rolling(np.concatenate([[0.0], (np.diff(a) != 0).astype(float)])),
              'one leg down': ((s[:, 6] + s[:, 7]) == 1).astype(float), 'margin': srt[:, -1] - srt[:, -2],
              'low & fast': np.where(low, -s[:, 3], 0.0), 'low & off-centre': np.where(low, np.abs(s[:, 0]), 0.0),
              'time': np.arange(len(a), dtype=float)})
    return d


def shaping(f):
    """The game's own score of a state (its potential): the reference line for the AUC chart."""
    s = np.array(f['states'])
    return -(-100 * np.hypot(s[:, 0], s[:, 1]) - 100 * np.hypot(s[:, 2], s[:, 3]) - 100 * np.abs(s[:, 4])
             + 10 * s[:, 6] + 10 * s[:, 7])


def auc(scores, labels):
    order = np.argsort(scores, kind='mergesort')
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def residualize(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name), dpi=130, bbox_inches='tight')
    plt.close(fig)
    return name


def replay_frames(seed, actions, steps):
    """Frames at the given step indices of a recorded flight (same seed + actions = same flight)."""
    env = make_env('rgb_array')
    seed_wind(env, seed)
    env.reset(seed=seed)
    want, out = set(steps), {}
    for t, a in enumerate(actions):
        if t in want:
            out[t] = env.render()
        env.step(a)
    if len(actions) in want or max(steps) >= len(actions):
        out[len(actions)] = env.render()
    env.close()
    return out


SNAP_GENS = [10, 50, 80, 150, 250, 350, 499]


def training_figures(run_dir, out_dir, flights, summary):
    """Figures 7-11: selection noise, progress, bloat, diversity, reference controllers."""
    figs = []
    print('Re-flying population snapshots (cached in reevaluation.npz) ...', flush=True)
    re = TC.reevaluate(run_dir, SNAP_GENS, os.path.join(out_dir, 'reevaluation.npz'))
    noise = {g: TC.noise_stats(re[g]) for g in SNAP_GENS}
    summary['selection_noise'] = noise

    # 7. Is selection noisy?
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    g = 250
    d = re[g]
    ax1.scatter(d['train'], d['true'], s=14, color=SERIES[0], alpha=0.6)
    top = int(np.argmax(d['train']))
    ax1.scatter([d['train'][top]], [d['true'][top]], s=80, color=SERIES[1], zorder=3,
                label=f"NEAT's pick (5-episode best): {d['true'][top]:.0f} on 100")
    best = int(np.argmax(d['true']))
    ax1.scatter([d['train'][best]], [d['true'][best]], s=80, color=SERIES[2], zorder=3,
                label=f"actually best on 100 episodes: {d['true'][best]:.0f}")
    lo = min(d['train'].min(), d['true'].min())
    ax1.plot([lo, 300], [lo, 300], color=INK_2, ls='--', lw=1)
    ax1.set_xlabel('training fitness (5 episodes, what NEAT selects on)')
    ax1.set_ylabel('fitness on 100 fresh episodes')
    ax1.set_title(f'Generation {g}: rank correlation {noise[g]["rank_corr_train_true"]:.2f}')
    ax1.legend(frameon=False, loc='upper left')
    x = np.arange(len(SNAP_GENS))
    ax2.bar(x - 0.2, [noise[g]['within_sd'] for g in SNAP_GENS], 0.4, color=SERIES[1],
            label='luck: SD of one genome over different 5-episode sets')
    ax2.bar(x + 0.2, [noise[g]['between_sd'] for g in SNAP_GENS], 0.4, color=SERIES[0],
            label='skill: SD between genomes (100 episodes each)')
    for i, gg in enumerate(SNAP_GENS):
        ax2.text(i, max(noise[gg]['within_sd'], noise[gg]['between_sd']) + 4,
                 f"r={noise[gg]['retest_r']:.2f}", ha='center', fontsize=8, color=INK_2)
    ax2.set_xticks(x, [str(g) for g in SNAP_GENS])
    ax2.set_xlabel('generation (r = correlation of two independent 5-episode scores of the same genome)')
    ax2.set_title('Luck vs skill in the 5-episode fitness')
    ax2.set_ylim(0, 1.45 * max(noise[g]['between_sd'] for g in SNAP_GENS))
    ax2.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    late = [g for g in SNAP_GENS if g >= 150]
    figs.append(('Is selection noisy?', save(fig, out_dir, '7_noise.png'),
                 'Every genome of a population snapshot re-flown on 100 fresh windy episodes. Left: NEAT ranks genomes '
                 'by 5 episodes (x), but their real skill is the 100-episode score (y). Right: how much a 5-episode score '
                 f'moves by luck alone vs how much genomes really differ. After generation 150 the luck spread is '
                 f'{np.mean([noise[g]["within_sd"] for g in late]):.0f} points and the skill spread '
                 f'{np.mean([noise[g]["between_sd"] for g in late]):.0f}.'))

    # 8. Did it stall?
    val = runs.read_csv(run_dir, 'validation')
    fit = runs.read_csv(run_dir, 'fitness')
    G = len(val)
    gens = np.arange(G)
    v = np.array([r['reward'] for r in val])
    tb = np.array([r['best'] for r in fit])

    def rmed(a, n=25):
        return np.array([np.median(a[max(0, i - n + 1):i + 1]) for i in range(len(a))])
    true_best = [re[g]['true'].max() for g in SNAP_GENS]
    true_med = [np.median(re[g]['true']) for g in SNAP_GENS]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(gens, v, color=SERIES[2], lw=0.8, alpha=0.4)
    ax.plot(gens, rmed(v), color=SERIES[2], label='champion on 100 unseen episodes (game score; 25-gen median)')
    ax.plot(gens, rmed(tb), color=SERIES[1], label='best training fitness, 5 episodes (25-gen median)')
    ax.plot(SNAP_GENS, true_best, 'o-', color=SERIES[0], label='best genome on 100 fresh episodes (re-flown)')
    ax.plot(SNAP_GENS, true_med, 'o--', color=SERIES[6], lw=1.4, label='median genome on 100 fresh episodes')
    ax.axhline(200, color=INK_2, ls=':', lw=1)
    ax.text(G - 1, 203, 'solved', ha='right', fontsize=8, color=INK_2)
    ax.set_xlabel('generation')
    ax.set_title('Progress over the run')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    summary['progress'] = {'val_median_25_at': {int(g): float(rmed(v)[g]) for g in (50, 100, 200, 300, 400, G - 1)},
                           'true_best': dict(zip(map(int, SNAP_GENS), map(float, true_best)))}
    figs.append(('Did it stall?', save(fig, out_dir, '8_progress.png'),
                 'Green: the champion NEAT picked each generation, on unseen episodes. Orange: its lucky 5-episode '
                 'training score. Blue: the best genome that was actually in the population, found by re-flying '
                 'every genome on 100 episodes. The gap between orange and green is luck; the gap between blue and '
                 'green is a better genome NEAT had but did not pick.'))

    # 9. Did the networks bloat?
    cpx = runs.read_csv(run_dir, 'complexity')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6))
    ax1.plot(gens, [r['mean_hidden'] for r in cpx], color=SERIES[6], label='population mean')
    ax1.plot(gens, [r['best_hidden'] for r in cpx], color=SERIES[1], lw=1, alpha=0.7, label='champion')
    ax1.set_title('Hidden nodes over the run')
    ax1.set_xlabel('generation')
    ax1.legend(frameon=False)
    cols = plt.get_cmap('viridis')(np.linspace(0, 0.9, len(SNAP_GENS)))
    for c, g in zip(cols, SNAP_GENS):
        ax2.scatter(re[g]['hidden'] + np.random.default_rng(g).uniform(-0.3, 0.3, len(re[g]['hidden'])),
                    re[g]['true'], s=10, color=c, alpha=0.6, label=f'gen {g}')
    ax2.set_xlabel('hidden nodes')
    ax2.set_ylabel('fitness on 100 fresh episodes')
    ax2.set_title('Does a bigger network fly better?')
    ax2.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    allh = np.concatenate([re[g]['hidden'] for g in SNAP_GENS])
    allt = np.concatenate([re[g]['true'] for g in SNAP_GENS])
    summary['bloat'] = {'rank_corr_hidden_true': TC.spearman(allh, allt),
                        'mean_hidden': {int(g): float(re[g]['hidden'].mean()) for g in SNAP_GENS}}
    figs.append(('Did the networks bloat?', save(fig, out_dir, '9_bloat.png'),
                 f'Rank correlation between hidden nodes and real (100-episode) fitness across all snapshots: '
                 f'{summary["bloat"]["rank_corr_hidden_true"]:+.2f}. Nothing in the fitness charges for size, '
                 'so node and connection mutations pile up.'))

    # 10. Diversity
    sp = runs.read_csv(run_dir, 'species')
    by = {}
    for r in sp:
        by.setdefault(r['generation'], []).append(r['size'])
    nsp = [len(by.get(g, [])) for g in gens]
    share = [max(by.get(g, [0])) / max(1, sum(by.get(g, [1]))) for g in gens]
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.plot(gens, nsp, color=SERIES[0], label='species alive')
    ax2 = ax.twinx()
    ax2.plot(gens, share, color=SERIES[3], lw=1.2, label='share of the largest species')
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)
    ax.set_xlabel('generation')
    ax.set_title('Diversity: how many species, and how dominant the biggest is')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc='upper right')
    fig.tight_layout()
    summary['diversity'] = {'species_median_after_150': float(np.median(nsp[150:])),
                            'largest_share_median_after_150': float(np.median(share[150:]))}
    figs.append(('Is the population diverse?', save(fig, out_dir, '10_diversity.png'),
                 f'After generation 150: median {summary["diversity"]["species_median_after_150"]:.0f} species, the '
                 f'largest holding {100 * summary["diversity"]["largest_share_median_after_150"]:.0f}% of the population. '
                 'The compatibility threshold (2.2) is fixed, while genomes grow.'))

    # 11. Reference controllers
    print('Flying the reference controllers ...', flush=True)
    h = TC.heuristic_scores()
    bdir = os.path.join(runs.ROOT, 'experiments', 'baseline')
    from neat_lander.fitness import fly as fly_genome
    bseeds = [TC.SEED0 + 60_000 + i for i in range(200)]
    old = fly_genome(runs.load_pickle(os.path.join(bdir, 'nowind_winner.pkl')),
                     runs.load_config(os.path.join(bdir, 'nowind_config.ini')), bseeds)
    win = {'reward': float(np.mean([f['reward'] for f in flights])),
           'landed': float(np.mean([f['outcome'] == 'landed' for f in flights])),
           'centred': float(np.mean([f.get('centred', False) for f in flights]))}
    rows = [('hand-written heuristic, calm', h['calm']), ('hand-written heuristic, wind', h['wind']),
            ('calm-air NEAT champion, wind', old), ('wind NEAT winner, wind', win)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.8))
    y = np.arange(len(rows))[::-1]
    colors = [SERIES[4], SERIES[4], SERIES[0], SERIES[2]]
    ax1.barh(y, [r['reward'] for _, r in rows], color=colors)
    ax1.axvline(200, color=INK_2, ls=':', lw=1)
    ax1.set_yticks(y, [n for n, _ in rows])
    ax1.set_title('Average game score')
    ax2.barh(y, [100 * r['landed'] for _, r in rows], color=colors)
    ax2.barh(y, [100 * r['centred'] for _, r in rows], color=INK, alpha=0.25, height=0.4)
    ax2.set_yticks(y, ['' for _ in rows])
    ax2.set_title('Landed % (dark inner bar: landed in the middle)')
    for ax, key, fmt in ((ax1, 'reward', '{:.0f}'), (ax2, 'landed', '{:.0%}')):
        for yy, (_, r) in zip(y, rows):
            ax.text(r[key] * (100 if key == 'landed' else 1), yy, ' ' + fmt.format(r[key]), va='center', fontsize=8)
    fig.tight_layout()
    summary['reference'] = {n: r for n, r in rows}
    figs.append(('Can a controller without memory do it?', save(fig, out_dir, '11_reference.png'),
                 "Gymnasium's own hand-written heuristic reacts only to the current state, like the evolved network. "
                 'It is the reference for what a simple reactive controller manages in this wind.'))
    return figs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('run_dir')
    args = ap.parse_args()
    name = os.path.basename(os.path.normpath(args.run_dir))
    out_dir = os.path.join(runs.ROOT, 'results', 'failure_analysis', name)
    header, flights = load(os.path.join(out_dir, 'flights.jsonl.gz'))
    fails = [f for f in flights if f['outcome'] != 'landed']
    lands = [f for f in flights if f['outcome'] == 'landed']
    sig = [signals(f) for f in flights]
    keys = CURRENT + CANDIDATES
    figures, summary = [], {'episodes': len(flights), 'landed': len(lands),
                            'crashed': sum(f['outcome'] == 'crashed' for f in flights),
                            'timeout': sum(f['outcome'] == 'timeout' for f in flights)}

    # 1. What does failure look like? -----------------------------------------------------------
    show = sorted(fails, key=lambda f: (f['cause'], f['reward']))[:12] + lands[:4]
    fig, axes = plt.subplots(4, 4, figsize=(13, 8.5))
    for ax, f in zip(axes.flat, show):
        fr = replay_frames(f['seed'], f['actions'], [len(f['actions'])])[len(f['actions'])]
        ax.imshow(fr)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        col = SERIES[7] if f['outcome'] == 'crashed' else (SERIES[3] if f['outcome'] == 'timeout' else SERIES[2])
        ax.set_title(f"{f['cause']} · {f['reward']:+.0f}", fontsize=8, color=col, fontweight='normal')
    for ax in axes.flat[len(show):]:
        ax.axis('off')
    fig.suptitle('Final frame of each failure (last row: four landings for comparison)', fontsize=11)
    fig.tight_layout()
    figures.append(('What does failure look like?', save(fig, out_dir, '1_gallery.png'),
                    f'{len(fails)} of {len(flights)} flights failed. Titles give the cause and the episode reward.'))

    # 2. Where does it happen? ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    rng = [[-1, 1], [-0.1, 1.5]]
    norm = np.concatenate([np.array(f['states'])[:, :2] for f in lands])
    late = np.concatenate([np.array(f['states'])[-K:, :2] for f in fails])
    for ax, pts, title in [(axes[0], norm, 'landing flights: all steps'),
                           (axes[1], late, f'failed flights: last {K} steps')]:
        h, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=[40, 32], range=rng, density=True)
        ax.imshow(h.T, origin='lower', extent=[*rng[0], *rng[1]], cmap=BLUES, aspect='auto')
        ax.axvspan(-0.2, 0.2, ymin=0, ymax=0.06, color=SERIES[3], alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('x (pad between ±0.2)')
        ax.set_ylabel('altitude')
        ax.grid(False)
    ax = axes[2]
    causes = sorted({f['cause'] for f in fails})
    for i, c in enumerate(causes):
        p = np.array([f['final'][:2] for f in fails if f['cause'] == c])
        ax.scatter(p[:, 0], p[:, 1], s=28, color=SERIES[[7, 3, 1][i % 3]], label=c)
    ax.set_xlim(*rng[0])
    ax.set_ylim(*rng[1])
    ax.axvspan(-0.2, 0.2, ymin=0, ymax=0.06, color=SERIES[3], alpha=0.4)
    ax.set_title('where each failed flight ended')
    ax.set_xlabel('x')
    ax.legend(frameon=False)
    fig.tight_layout()
    on_pad = np.mean([abs(f['final'][0]) < 0.2 for f in fails])
    summary['failures_ending_over_pad'] = float(on_pad)
    figures.append(('Where does it happen?', save(fig, out_dir, '2_where.png'),
                    f'Darker = more time spent there. {100 * on_pad:.0f}% of failures end above the pad (|x| < 0.2).'))

    # 3. What drifts before failure? ------------------------------------------------------------
    drift_keys = ['vy', 'angle', 'spin', 'speed', 'side duty', 'main duty', 'x', 'switches']
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    normal = {k: np.concatenate([s[k] for s, f in zip(sig, flights) if f['outcome'] == 'landed']) for k in keys}
    departure = {}
    for k in keys:
        M = np.full((len(fails), WINDOW), np.nan)
        for i, f in enumerate(fails):
            v = sig[flights.index(f)][k][-WINDOW:]
            M[i, WINDOW - len(v):] = v
        med = np.nanmedian(M, axis=0)
        p90 = np.percentile(np.abs(normal[k]), 90)
        out = np.where(np.abs(med) > p90)[0]
        departure[k] = int(WINDOW - out[0]) if len(out) else None
        if k in drift_keys:
            ax = axes.flat[drift_keys.index(k)]
            xs = np.arange(-WINDOW, 0)
            col = SERIES[0] if k in CURRENT else SERIES[1]
            ax.fill_between(xs, np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0), color=col, alpha=0.18, lw=0)
            ax.plot(xs, med, color=col)
            ax.axhline(np.median(normal[k]), color=INK_2, ls='--', lw=1)
            ax.set_title(LABELS[k], fontsize=10)
            ax.set_xlabel('steps before failure')
    fig.tight_layout()
    figures.append(('What drifts before failure?', save(fig, out_dir, '3_drift.png'),
                    'Median and interquartile band over the last 150 steps of failed flights. Dashed: median over '
                    'landing flights. Blue = an input the network sees, orange = a derived candidate.'))

    # 4. What predicts failure? -----------------------------------------------------------------
    pos, vals = [], {k: [] for k in keys}
    ref = []
    for s, f in zip(sig, flights):
        n = len(f['actions'])
        if f['outcome'] == 'landed':
            m = np.arange(n) < n - K              # landing flights, minus their final touchdown steps
            lab = np.zeros(n, bool)
        else:
            m = np.ones(n, bool)
            lab = np.arange(n) >= n - K
        pos.append(lab[m])
        for k in keys:
            vals[k].append(s[k][m])
        ref.append(shaping(f)[m])
    labels = np.concatenate(pos)
    stats = {}
    ttf, cand_fail = [], {k: [] for k in keys}
    for f in fails:
        s = sig[flights.index(f)]
        n = len(f['actions'])
        idx = np.arange(max(0, n - WINDOW), n)
        ttf.append(n - idx)
        for k in keys:
            cand_fail[k].append(s[k][idx])
    ttf = np.concatenate(ttf).astype(float)
    X_cur = np.column_stack([np.concatenate(cand_fail[k]) for k in CURRENT])
    y_res = residualize(ttf, X_cur)
    for k in keys:
        v = np.concatenate(vals[k])
        a = auc(v, labels)
        stats[k] = {'auc': max(a, 1 - a), 'direction': 'higher' if a >= 0.5 else 'lower'}
        if k in CANDIDATES:
            c = np.concatenate(cand_fail[k])
            stats[k]['partial_corr'] = -corr(residualize(c, X_cur), y_res)   # + = higher value -> fails sooner
        else:
            stats[k]['partial_corr'] = None
    ref_auc = auc(np.concatenate(ref), labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    order = sorted(keys, key=lambda k: stats[k]['auc'])
    ax1.barh([LABELS[k] for k in order], [stats[k]['auc'] for k in order],
             color=[SERIES[0] if k in CURRENT else SERIES[1] for k in order])
    ax1.axvline(ref_auc, color=INK, ls='--', lw=1)
    ax1.text(ref_auc, len(order) - 0.4, f" game's shaping score {ref_auc:.2f}", fontsize=8, va='bottom')
    ax1.set_xlim(0.5, 1)
    ax1.set_title(f'Warning power: AUC for "fails within {K} steps"')
    for i, k in enumerate(order):
        ax1.text(stats[k]['auc'] + 0.005, i, f"{stats[k]['auc']:.2f}", va='center', fontsize=7, color=INK_2)
    cand = sorted(CANDIDATES, key=lambda k: abs(stats[k]['partial_corr']))
    pc = [stats[k]['partial_corr'] for k in cand]
    ax2.barh([LABELS[k] for k in cand], pc, color=[SERIES[1] if v > 0 else '#b4b2ac' for v in pc])
    ax2.axvline(0, color=INK_2, lw=1)
    for i, v in enumerate(pc):
        ax2.text(v + (0.005 if v >= 0 else -0.005), i, f'{v:+.2f}', va='center', ha='left' if v >= 0 else 'right',
                 fontsize=7, color=INK_2)
    ax2.set_title('Added information beyond the 8 inputs')
    ax2.set_xlabel('partial correlation with "fails sooner"\n(after regressing out the 8 inputs the network sees)')
    fig.tight_layout()
    summary['shaping_auc'] = ref_auc
    summary['signals'] = stats
    figures.append(('What predicts failure?', save(fig, out_dir, '4_predict.png'),
                    'Left: AUC of each signal alone for failing within 30 steps (positives: the last 30 steps of failed '
                    'flights; negatives: landing flights before touchdown). Right: what each derived signal adds about '
                    'time-to-failure once the network\'s own inputs are accounted for.'))

    # 5. What causes it? ------------------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.2))
    cnt = {c: sum(f['cause'] == c for f in fails) for c in causes}
    ax1.barh(list(cnt), list(cnt.values()), color=SERIES[7])
    for i, v in enumerate(cnt.values()):
        ax1.text(v + 0.2, i, str(v), va='center', fontsize=9)
    ax1.set_title(f'Why the {len(fails)} failures ended')
    push_f = [np.hypot(*f['states'][0][2:4]) for f in fails]
    push_l = [np.hypot(*f['states'][0][2:4]) for f in lands]
    bins = np.linspace(0, max(push_f + push_l), 20)
    ax2.hist(push_l, bins=bins, density=True, color=SERIES[2], alpha=0.6, label='landed')
    ax2.hist(push_f, bins=bins, density=True, color=SERIES[7], alpha=0.6, label='failed')
    ax2.set_title('Size of the random starting push')
    ax2.set_xlabel('initial speed')
    ax2.legend(frameon=False)
    mw = np.array([np.mean(f['wind']) for f in flights])
    failed = np.array([f['outcome'] != 'landed' for f in flights])
    edges = [-WIND_POWER, -7, -2, 2, 7, WIND_POWER + 1e-9]
    labels_w, rates, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mw >= lo) & (mw < hi)
        labels_w.append(f'{lo:+.0f}..{min(hi, WIND_POWER):+.0f}')
        rates.append(float(failed[m].mean()) if m.any() else 0.0)
        ns.append(int(m.sum()))
    ax3.bar(range(len(rates)), [100 * r for r in rates], color=[SERIES[7] if r > 0.4 else '#b4b2ac' for r in rates])
    for i, (r, n) in enumerate(zip(rates, ns)):
        ax3.text(i, 100 * r + 2, f'{100 * r:.0f}%\nn={n}', ha='center', fontsize=8)
    ax3.set_xticks(range(len(rates)), labels_w)
    ax3.set_ylim(0, 115)
    ax3.set_xlabel('average wind over the flight (− = to the left)')
    ax3.set_title('Failure rate by steady wind')
    fig.tight_layout()
    summary['causes'] = cnt
    summary['start_push_median'] = {'failed': float(np.median(push_f)), 'landed': float(np.median(push_l))}
    summary['failure_rate_by_mean_wind'] = dict(zip(labels_w, zip(rates, ns)))
    figures.append(('What causes it?', save(fig, out_dir, '5_causes.png'),
                    'Left: how failures end. Middle: whether failures start with a bigger random push. Right: the '
                    'failure rate by the average wind over the whole flight. Gusts that come and go are survivable; '
                    'a push that keeps coming from one side is not. (Wind is zero once a leg touches the ground, so '
                    'the flight average is used instead of the wind at the moment of failure.)'))

    # 6. How does it start? ---------------------------------------------------------------------
    thr = {k: np.percentile(np.abs(normal[k]), 75) for k in ('speed', '|angle|', '|spin|')}
    picks = sorted(fails, key=lambda f: f['reward'])[:3]
    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 2.6 * len(picks)))
    onsets = []
    for r, f in enumerate(picks):
        s = sig[flights.index(f)]
        ok = (s['speed'] <= thr['speed']) & (s['|angle|'] <= thr['|angle|']) & (s['|spin|'] <= thr['|spin|'])
        n = len(f['actions'])
        healthy = int(np.where(ok)[0][-1]) if ok.any() else 0
        onsets.append(n - healthy)
        steps = [healthy, min(n - 1, healthy + 10), min(n - 1, healthy + 30), n]
        frames = replay_frames(f['seed'], f['actions'], steps)
        for c, t in enumerate(steps):
            ax = axes[r, c]
            ax.imshow(frames[t])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            lab = ['last healthy state', '+10 steps', '+30 steps', f"end: {f['cause']}"][c]
            ax.set_title(f'{lab} (step {t})', fontsize=8, fontweight='normal', color=INK_2)
    fig.tight_layout()
    summary['onset_to_failure_steps'] = onsets
    figures.append(('How does it start?', save(fig, out_dir, '6_onset.png'),
                    'The three worst failures. "Healthy" = speed, |angle| and |angular velocity| all inside the '
                    '75th percentile of landing flights.'))

    # 7-11. The training run ----------------------------------------------------------------------
    figures += training_figures(args.run_dir, out_dir, flights, summary)

    # summary + report --------------------------------------------------------------------------
    for k in keys:
        stats[k]['normal_median'] = float(np.median(normal[k]))
        stats[k]['median_at_failure'] = float(np.median([sig[flights.index(f)][k][-1] for f in fails]))
        stats[k]['leaves_normal_range_steps_before'] = departure[k]
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)

    def row(k):
        s = stats[k]
        tag = ' <span class=cur>input</span>' if k in CURRENT else ''
        pc = '' if s['partial_corr'] is None else f"{s['partial_corr']:+.2f}"
        dep = '' if s['leaves_normal_range_steps_before'] is None else s['leaves_normal_range_steps_before']
        return (f"<tr><td>{LABELS[k]}{tag}</td><td>{s['normal_median']:.2f}</td><td>{s['median_at_failure']:.2f}</td>"
                f"<td>{s['auc']:.2f} ({s['direction']})</td><td>{pc}</td><td>{dep}</td></tr>")
    rows = ''.join(row(k) for k in sorted(keys, key=lambda k: -stats[k]['auc']))
    secs = ''.join(f'<section><h2>{t}</h2><p>{d}</p><img src="{img}" alt="{t}"></section>' for t, img, d in figures)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Wind failure analysis</title>
<style>body{{font-family:system-ui,sans-serif;background:{SURFACE};color:{INK};max-width:1200px;margin:24px auto;padding:0 16px}}
h1{{margin-bottom:4px}} .sub{{color:{INK_2}}} img{{max-width:100%;border:1px solid {GRID};border-radius:6px}}
table{{border-collapse:collapse;font-size:14px;margin:12px 0}} td,th{{padding:4px 10px;border-bottom:1px solid {GRID};text-align:right}}
td:first-child,th:first-child{{text-align:left}} .cur{{font-size:11px;color:#fff;background:{SERIES[0]};border-radius:4px;padding:1px 5px}}
section{{margin:28px 0}}</style></head><body>
<h1>Failure analysis: {name} winner (genome {header['genome']})</h1>
<p class="sub">{len(flights)} flights on analysis seeds {header['seed0']:,}+ (never used in training, validation or test):
{summary['landed']} landed, {summary['crashed']} crashed, {summary['timeout']} timed out. "Fails soon" = within {K} steps.</p>
<table><tr><th>signal</th><th>median, landing flights</th><th>median at failure</th><th>AUC (warning)</th>
<th>added info (partial corr.)</th><th>leaves normal range (steps before)</th></tr>{rows}</table>
<p class="sub">AUC of the game's own shaping score of the state: {ref_auc:.2f}. "Leaves normal range" = steps before failure
where the median first exceeds the 90th percentile of |value| on landing flights. Findings are leads: test them by retraining.</p>
{secs}</body></html>"""
    with open(os.path.join(out_dir, 'report.html'), 'w') as fh:
        fh.write(html)
    print(f'Wrote {out_dir}/report.html')


if __name__ == '__main__':
    main()
