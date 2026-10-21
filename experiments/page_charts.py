#!/usr/bin/env python3
"""Inline SVG charts for docs/index.html, from the failure analysis of a run.

    python experiments/page_charts.py runs/<run>      # after analyze_flights.py

Each chart replaces the block between <!-- chart:NAME --> and <!-- /chart:NAME --> in the page.
Colours are the page's CSS variables (--c1/--c2/--c3, --muted, --line, --text), so the charts follow
the light/dark theme; every mark has a <title> tooltip.
"""

import argparse
import gzip
import json
import os
import re
import sys
from html import escape

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neat_lander import runs  # noqa: E402
from neat_lander.env import WIND_POWER  # noqa: E402

PAGE = os.path.join(runs.ROOT, 'docs', 'index.html')
FONT = 'font-family="system-ui, sans-serif"'


def svg(w, h, body, label):
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}" {FONT}>'
            f'{body}</svg>')


def t(x, y, s, size=13, fill='var(--muted)', anchor='start', weight='normal'):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(str(s))}</text>')


def tip(s):
    return f'<title>{escape(s)}</title>'


def bar_top(x, y, w, h, fill, title):
    """A vertical bar from the baseline up, rounded 4px at the data end only."""
    r = min(4, w / 2, h)
    d = (f'M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + w - r:.1f} '
         f'Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z')
    return f'<path d="{d}" fill="{fill}">{tip(title)}</path>'


def bar_right(x, y, w, h, fill, title):
    """A horizontal bar from the baseline to the right, rounded at the data end only."""
    r = min(4, h / 2, max(w, 0))
    d = (f'M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h - r:.1f} '
         f'Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z')
    return f'<path d="{d}" fill="{fill}">{tip(title)}</path>'


def marker(shape, x, y, fill, title, r=5):
    ring = 'stroke="var(--surface)" stroke-width="2"'
    if shape == 'square':
        m = f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1.5" fill="{fill}" {ring}/>'
    elif shape == 'diamond':
        m = (f'<path d="M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y:.1f} L{x:.1f},{y + r + 1:.1f} '
             f'L{x - r - 1:.1f},{y:.1f} Z" fill="{fill}" {ring}/>')
    else:
        m = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" {ring}/>'
    hit = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="transparent"/>'
    return f'<g class="mark">{m}{hit}{tip(title)}</g>'


def steady_wind(flights):
    mw = np.array([np.mean(f['wind']) for f in flights])
    failed = np.array([f['outcome'] != 'landed' for f in flights])
    edges = [-WIND_POWER, -7, -2, 2, 7, WIND_POWER + 1e-9]
    names = ['strong, left', 'moderate, left', 'little or none', 'moderate, right', 'strong, right']
    W, H, x0, y0, ph = 640, 300, 48, 30, 210
    bw = 82
    gap = (W - x0 - 5 * bw) / 5
    body = []
    for v in (0, 50, 100):
        y = y0 + ph - v / 100 * ph
        body.append(f'<line x1="{x0}" x2="{W}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--line)"/>')
        body.append(t(x0 - 8, y + 4, f'{v}%', 12, anchor='end'))
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (mw >= lo) & (mw < hi)
        rate = float(failed[m].mean()) if m.any() else 0.0
        x = x0 + gap / 2 + i * (bw + gap)
        h = max(rate * ph, 1)
        body.append(bar_top(x, y0 + ph - h, bw, h, 'var(--c3)',
                            f'{names[i]} wind: {100 * rate:.0f}% of {m.sum()} flights failed'))
        body.append(t(x + bw / 2, y0 + ph - h - 8, f'{100 * rate:.0f}%', 14, 'var(--text)', 'middle', 'bold'))
        body.append(t(x + bw / 2, y0 + ph + 20, names[i], 12, anchor='middle'))
        body.append(t(x + bw / 2, y0 + ph + 36, f'{m.sum()} flights', 11, anchor='middle'))
    body.append(f'<line x1="{x0}" x2="{W}" y1="{y0 + ph}" y2="{y0 + ph}" stroke="var(--muted)"/>')
    return svg(W, H, ''.join(body), 'Failure rate by the average wind over the flight')


def progress(run_dir, summary):
    val = runs.read_csv(run_dir, 'validation')
    fit = runs.read_csv(run_dir, 'fitness')
    G = len(val)

    def rmed(a, n=25):
        a = np.asarray(a, float)
        return np.array([np.median(a[max(0, i - n + 1):i + 1]) for i in range(len(a))])
    champ = rmed([r['reward'] for r in val])
    train = rmed([r['best'] for r in fit])
    snaps = summary['selection_noise']
    W, H, x0, y0, pw, ph = 720, 330, 48, 20, 560, 260
    lo, hi = -50, 250

    def X(g):
        return x0 + g / (G - 1) * pw

    def Y(v):
        return y0 + ph - (np.clip(v, lo, hi) - lo) / (hi - lo) * ph
    body = []
    for v in range(lo, hi + 1, 50):
        body.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--line)"/>')
        body.append(t(x0 - 8, Y(v) + 4, v, 12, anchor='end'))
    body.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{Y(200):.1f}" y2="{Y(200):.1f}" stroke="var(--muted)" '
                f'stroke-dasharray="2 4"/>')
    body.append(t(x0 + 6, Y(200) - 6, 'solved (200)', 12))
    for g in range(0, G, 100):
        body.append(t(X(g), y0 + ph + 18, g, 12, anchor='middle'))
    body.append(t(x0 + pw / 2, y0 + ph + 36, 'generation', 12, anchor='middle'))

    def path(vals):
        return 'M' + ' L'.join(f'{X(i):.1f},{Y(v):.1f}' for i, v in enumerate(vals))
    body.append(f'<path d="{path(train)}" fill="none" stroke="var(--c1)" stroke-width="2" stroke-dasharray="6 4"/>')
    body.append(f'<path d="{path(champ)}" fill="none" stroke="var(--c2)" stroke-width="2"/>')
    gens = sorted(int(g) for g in snaps)
    pts = [(X(g), Y(snaps[str(g)]['true_best'])) for g in gens]
    body.append('<path d="M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in pts) +
                '" fill="none" stroke="var(--c3)" stroke-width="2"/>')
    for g, (x, y) in zip(gens, pts):
        body.append(marker('square', x, y, 'var(--c3)',
                           f'generation {g}: the best genome in the population scored '
                           f'{snaps[str(g)]["true_best"]:.0f} on 100 fresh episodes'))
    for g in range(0, G, 10):              # hover strips: the two lines' values every 10 generations
        body.append(f'<rect x="{X(g) - 5:.1f}" y="{y0}" width="10" height="{ph}" fill="transparent">'
                    f'{tip(f"generation {g}: best training score {train[g]:.0f} (5 episodes), champion on unseen episodes {champ[g]:.0f}")}</rect>')
    lx = x0 + pw + 10                      # direct labels at the right end, pushed apart to 32px
    labels = sorted([(Y(train[-1]), 'best training score', '(5 lucky episodes)'),
                     (Y(champ[-1]), "NEAT's champion,", 'unseen episodes'),
                     (pts[-1][1], 'best genome it had', '(re-flown, 100 eps)')])
    ys = [labels[0][0]]
    for y, _, _ in labels[1:]:
        ys.append(max(y, ys[-1] + 32))
    for y, (_, a, b) in zip(ys, labels):
        body.append(t(lx, y, a, 12, 'var(--text)'))
        body.append(t(lx, y + 14, b, 11))
    return svg(W + 140, H + 30, ''.join(body), 'Progress over the run')


def luck(summary):
    snaps = summary['selection_noise']
    gens = sorted(int(g) for g in snaps)
    W, H, x0, y0, pw, ph = 720, 300, 48, 20, 600, 230
    lo, hi = -50, 250

    def Y(v):
        return y0 + ph - (np.clip(v, lo, hi) - lo) / (hi - lo) * ph
    step = pw / len(gens)
    body = []
    for v in range(lo, hi + 1, 50):
        body.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--line)"/>')
        body.append(t(x0 - 8, Y(v) + 4, v, 12, anchor='end'))
    for i, g in enumerate(gens):
        s = snaps[str(g)]
        x = x0 + step * (i + 0.5)
        body.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(s["train_best_train"]):.1f}" '
                    f'y2="{Y(s["train_best_true"]):.1f}" stroke="var(--muted)" stroke-width="2"/>')
        body.append(marker('diamond', x, Y(s['train_best_train']), 'var(--c1)',
                           f'generation {g}: NEAT\'s pick scored {s["train_best_train"]:.0f} on its 5 training episodes'))
        body.append(marker('circle', x, Y(s['train_best_true']), 'var(--c2)',
                           f'generation {g}: the same genome on 100 fresh episodes: {s["train_best_true"]:.0f}'))
        body.append(marker('square', x + 14, Y(s['true_best']), 'var(--c3)',
                           f'generation {g}: the best genome actually in the population: {s["true_best"]:.0f}'))
        body.append(t(x + 4, y0 + ph + 18, g, 12, anchor='middle'))
    body.append(t(x0 + pw / 2, y0 + ph + 36, 'generation', 12, anchor='middle'))
    return svg(W, H + 20, ''.join(body), "NEAT's pick against the best genome it had")


def reference(summary):
    ref = summary['reference']
    rows = [('Hand-written controller, calm', ref['hand-written heuristic, calm'], 'var(--muted)'),
            ('Hand-written controller, wind', ref['hand-written heuristic, wind'], 'var(--muted)'),
            ('Calm-air NEAT champion, wind', ref['calm-air NEAT champion, wind'], 'var(--muted)'),
            ('Wind NEAT winner, wind', ref['wind NEAT winner, wind'], 'var(--c3)')]
    W, H, x0, pw, rh = 720, 190, 250, 330, 40
    hi = 260

    def X(v):
        return x0 + max(v, 0) / hi * pw
    body = [f'<line x1="{X(200):.1f}" x2="{X(200):.1f}" y1="6" y2="{12 + rh * len(rows)}" stroke="var(--muted)" '
            f'stroke-dasharray="2 4"/>', t(X(200), H - 8, 'solved (200)', 12, anchor='middle')]
    for i, (name, r, col) in enumerate(rows):
        y = 12 + i * rh
        body.append(t(x0 - 10, y + 19, name, 13, 'var(--text)', 'end'))
        body.append(bar_right(x0, y + 4, X(r['reward']) - x0, 22, col,
                              f'{name}: average score {r["reward"]:.0f}, {100 * r["landed"]:.0f}% landed, '
                              f'{100 * r["centred"]:.0f}% in the middle'))
        body.append(t(X(r['reward']) + 8, y + 20, f'{r["reward"]:.0f}  ·  {100 * r["landed"]:.0f}% landed', 13,
                      'var(--text)'))
    return svg(W, H, ''.join(body), 'Average game score of four controllers in calm air and wind')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('run_dir')
    args = ap.parse_args()
    name = os.path.basename(os.path.normpath(args.run_dir))
    d = os.path.join(runs.ROOT, 'results', 'failure_analysis', name)
    with open(os.path.join(d, 'summary.json')) as fh:
        summary = json.load(fh)
    with gzip.open(os.path.join(d, 'flights.jsonl.gz'), 'rt') as fh:
        fh.readline()
        flights = [json.loads(line) for line in fh]
    charts = {'steady-wind': steady_wind(flights), 'progress': progress(args.run_dir, summary),
              'luck': luck(summary), 'reference': reference(summary)}
    with open(PAGE) as fh:
        page = fh.read()
    for key, content in charts.items():
        pat = re.compile(rf'(<!-- chart:{key} -->).*?(<!-- /chart:{key} -->)', re.S)
        if not pat.search(page):
            raise SystemExit(f'marker <!-- chart:{key} --> not found in {PAGE}')
        page = pat.sub(lambda m: m.group(1) + content + m.group(2), page)
    with open(PAGE, 'w') as fh:
        fh.write(page)
    print(f'updated {len(charts)} charts in {PAGE}')


if __name__ == '__main__':
    main()
