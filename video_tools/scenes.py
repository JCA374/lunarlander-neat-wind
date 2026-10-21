"""
Scenes for the explainer video.

Every scene factory returns a `timeline.Scene`: a render function `render(t, beats)` plus the
default length of each beat. A beat is one narrated section (see timeline.py), so the visuals are
timed by `beats.at(k)` (when beat k starts) instead of fixed seconds.

Parts: the problem (task, reward) · the network and its genome · how NEAT changes genomes
(mutation, crossover, speciation, the generation loop) · the real training run · the result.
"""

import json
import os

import numpy as np
from PIL import Image, ImageDraw

from common import (ABLATION_EPISODES, ACTION_COLORS, AMBER, BG, BLUE, CENTRED, FG, FPS, GREEN, H, MUTED, PANEL,
                    RED, VIEW_H, VIEW_W, VIOLET, W, Scene, blank, ease, fade, fly, frame_at, heading, mix,
                    outcome, paste_flight, step_at, terrain_frame, text, wrap)
from common import REPO_ROOT
from neat_lander.env import CENTRE_BONUS, TURBULENCE_POWER, WIND_POWER
from neat_lander.draw import draw_network, layout
from neat_lander.labels import ACTION_NAMES, INPUT_NAMES

REPO_URL = 'github.com/JCA374/lunarlander-neat-wind'
SPECIES_COLORS = [(84, 212, 140), (96, 168, 255), (255, 184, 64), (190, 120, 255), (240, 96, 96),
                  (64, 210, 220), (255, 130, 190), (170, 200, 90), (255, 150, 90), (150, 150, 255)]


def species_color(sid):
    return SPECIES_COLORS[int(sid) % len(SPECIES_COLORS)] if sid not in (None, '') else MUTED


# ----------------------------------------------------------------------------
# Part 1: the problem
# ----------------------------------------------------------------------------

def scene_title(ctx, name='title'):
    frames = ctx.best_ep['frames']

    def render(t, b):
        img = blank()
        f = frame_at(frames, t)
        tile = Image.fromarray(f).resize((1280, 853), Image.BILINEAR).crop((0, 133, 1280, 853))
        tile = Image.blend(tile, Image.new('RGB', tile.size, BG), 0.62)
        img.paste(tile, (0, 0))
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 270), 'NEAT Lunar Lander in wind', 72, FG, fade(t, 0.2, 0.8), bold=True, anchor='mm')
        text(d, (W // 2, 350), 'Evolving the weights and the shape of a neural network', 30, AMBER,
             fade(t, 0.7, 0.8), anchor='mm')
        return img
    return Scene(name, render, [7.0])


def scene_task(ctx, name='task'):
    """Beats: the game; the 8 inputs; the 4 actions; the wind."""
    ep = ctx.best_ep
    n = len(ep['frames'])

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The problem', t, 'Gymnasium LunarLander-v2, with wind')
        paste_flight(img, ep, step_at(n, t), (60, 140, 660, 440))
        a3 = fade(t, b.at(3))
        if a3 > 0:
            d.rounded_rectangle([60, 600, 720, 680], 10, fill=mix(BG, PANEL, a3))
            text(d, (80, 612), f'wind: a sideways push of up to {WIND_POWER:.0f}, plus turbulence that',
                 19, FG, a3, bg=PANEL)
            text(d, (80, 642), f'twists the lander (up to {TURBULENCE_POWER}). Not one of the 8 inputs.',
                 19, FG, a3, bg=PANEL)
        x0 = 770
        t1, t2 = b.at(1), b.at(2)
        text(d, (x0, 140), '8 inputs (what it senses)', 26, BLUE, fade(t, t1), bold=True)
        for i, nm in enumerate(INPUT_NAMES):
            text(d, (x0 + 10, 182 + 30 * i), f'{i + 1}. {nm}', 21, FG, fade(t, t1 + 0.4 + 0.35 * i))
        text(d, (x0, 450), '4 actions (what it can do)', 26, AMBER, fade(t, t2), bold=True)
        for i, nm in enumerate(ACTION_NAMES):
            text(d, (x0 + 10, 492 + 30 * i), f'{i}. {nm}', 21, FG, fade(t, t2 + 0.6 + 0.6 * i))
        return img
    return Scene(name, render, [5.5, 6.0, 5.0, 8.0])


def scene_reward(ctx, name='reward'):
    """Beats: shaping terms; one-off events; fuel; the centre bonus (ours); 'solved'."""
    items = [(0, '+', 'moving closer to the pad, slowing down, staying level', GREEN),
             (0, '−', 'drifting away, speeding up, tilting', RED),
             (1, '+10', 'each leg that touches the ground', GREEN),
             (1, '+100', 'coming to rest on the ground', GREEN),
             (1, '−100', 'crashing or flying off screen', RED),
             (2, '−0.3', 'every frame the main engine fires', AMBER),
             (2, '−0.03', 'every frame a side engine fires', AMBER),
             (3, f'+0…{CENTRE_BONUS:.0f}', 'our bonus: landing near the middle between the flags', BLUE)]
    ep = ctx.best_ep
    n = len(ep['frames'])

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The reward', t, 'what the game itself scores')
        box = (60, 150, 420, 280)
        paste_flight(img, ep, step_at(n, t), box)
        a = fade(t, b.at(3))
        if a > 0:                       # the bonus: full in the middle, nothing at the flags
            x, y, w, h = box
            cx, half, base = x + w / 2, w / 2 * 0.2, y + h + 70
            d.polygon([(cx - half, base), (cx, base - 50), (cx + half, base)], fill=mix(BG, BLUE, 0.7 * a))
            d.line([x, base, x + w, base], fill=mix(BG, MUTED, a), width=1)
            text(d, (cx, base - 56), f'+{CENTRE_BONUS:.0f}', 16, BLUE, a, bold=True, anchor='mb')
            text(d, (cx, base + 8), 'bonus across the pad, flag to flag', 15, MUTED, a, anchor='ma')
        for k, (beat, v, s, col) in enumerate(items):
            start = b.at(beat) + 0.5 + 0.9 * sum(1 for it in items[:k] if it[0] == beat)
            a = fade(t, start)
            y = 150 + k * 58
            text(d, (540, y), v, 30, col, a, bold=True, mono=True)
            text(d, (680, y + 6), s, 21, FG, a)
        a = fade(t, b.at(4))
        d.rounded_rectangle([540, 612, 1220, 672], 12, fill=mix(BG, PANEL, a))
        text(d, (880, 642), 'a game score of 200 or more counts as solved', 22, AMBER, a,
             anchor='mm', bg=PANEL)
        return img
    return Scene(name, render, [6.0, 6.0, 5.0, 7.0, 5.0])


# ----------------------------------------------------------------------------
# Part 2: the network
# ----------------------------------------------------------------------------

def live_network(img, ctx, genome, ep, k, box, labels=True, r=12):
    """Draw `genome` lit up by step k of the traced episode `ep`; returns the chosen action."""
    values = ep['values'][k]
    action = ep['actions'][k]
    draw_network(img, genome, ctx.config, box, values=values, action=action, labels=labels, r=r,
                 hidden_names=ctx.hidden_alias if genome is ctx.winner else None)
    return action


def action_bars(d, ep, k, x0, y0, a=1.0):
    """The 4 output values as bars, the winner highlighted."""
    vals = [ep['values'][k].get(o, 0.0) for o in range(4)]
    act = ep['actions'][k]
    for o in range(4):
        y = y0 + o * 34
        col = ACTION_COLORS[o] if o == act else mix(BG, MUTED, 0.6)
        text(d, (x0, y), ACTION_NAMES[o], 18, FG if o == act else MUTED, a)
        w = int(140 * (vals[o] + 1) / 2)
        d.rectangle([x0 + 150, y + 3, x0 + 150 + 140, y + 21], outline=mix(BG, PANEL, a))
        d.rectangle([x0 + 150, y + 3, x0 + 150 + w, y + 21], fill=mix(BG, col, a))
        text(d, (x0 + 300, y), f'{vals[o]:+.2f}', 16, FG if o == act else MUTED, a, mono=True)


def scene_network(ctx, name='network'):
    """Beats: the pilot is a network (live); highest output fires; NEAT finds this network."""
    ep = ctx.best_ep
    n = len(ep['actions'])
    hidden, conns = ctx.run.size(ctx.winner)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'The pilot is a neural network', t, 'the evolved winner, flying')
        k = step_at(n, t, speed=0.8)
        paste_flight(img, ep, k, (50, 150, 480, 320))
        live_network(img, ctx, ctx.winner, ep, k, (690, 150, 1060, 560))
        a1 = fade(t, b.at(1))
        if a1 > 0:
            action_bars(d, ep, k, 70, 500, a1)
        a2 = fade(t, b.at(2))
        text(d, (690, 610), f'{hidden} hidden nodes · {conns} connections, '
             'all found by evolution', 20, AMBER, a2)
        text(d, (60, 120), 'inputs light up left, the highest action score fires', 17, MUTED, fade(t, 0.4))
        return img
    return Scene(name, render, [6.0, 6.0, 6.0])


# --- toy genomes for the "how NEAT works" scenes ------------------------------------------------

TOY_LABELS = {1: 'vy', 2: 'angle', 3: 'spin', 4: 'main', 5: 'left', 6: 'h6'}
TOY_POS = {1: (0.0, 0.0), 2: (0.0, 0.5), 3: (0.0, 1.0), 4: (1.0, 0.2), 5: (1.0, 0.8), 6: (0.5, 0.3)}
# innovation number -> (from, to)
TOY_GENES = {1: (1, 4), 2: (2, 5), 3: (3, 5), 4: (2, 4), 5: (3, 4), 6: (1, 6), 7: (6, 4), 8: (3, 6)}


def toy_xy(node, box):
    x0, y0, x1, y1 = box
    u, v = TOY_POS[node]
    return x0 + u * (x1 - x0), y0 + v * (y1 - y0)


def draw_toy(d, genes, box, a=1.0, nodes=None, hl=(), node_a=None, grow=None):
    """genes: {innov: (weight, enabled)}. grow: {innov: 0..1} draws a connection partially."""
    nodes = nodes or sorted({n for i in genes for n in TOY_GENES[i]} | {1, 2, 3, 4, 5})
    for innov, (w, on) in sorted(genes.items()):
        p, q = (toy_xy(n, box) for n in TOY_GENES[innov])
        g = 1.0 if grow is None else grow.get(innov, 1.0)
        q = (p[0] + (q[0] - p[0]) * g, p[1] + (q[1] - p[1]) * g)
        col = AMBER if w > 0 else BLUE
        if not on:
            col = mix(BG, MUTED, 0.35)
        if innov in hl:
            col = FG
        width = 2 if not on else max(3, int(2 + 4 * min(1.0, abs(w) / 1.5)))
        if on:
            d.line([p, q], fill=mix(BG, col, a), width=width)
        else:                                            # dashed: disabled gene
            for s in np.linspace(0, 1, 14)[::2]:
                e = min(1.0, s + 1 / 13)
                d.line([(p[0] + (q[0] - p[0]) * s, p[1] + (q[1] - p[1]) * s),
                        (p[0] + (q[0] - p[0]) * e, p[1] + (q[1] - p[1]) * e)], fill=mix(BG, col, a), width=2)
    for nd in nodes:
        x, y = toy_xy(nd, box)
        na = a * (node_a.get(nd, 1.0) if node_a else 1.0)
        r = 15
        fill = PANEL if nd <= 3 else (mix(PANEL, AMBER, 0.25) if nd >= 6 else mix(PANEL, BLUE, 0.25))
        d.ellipse([x - r, y - r, x + r, y + r], fill=mix(BG, fill, na), outline=mix(BG, FG, na), width=2)
        text(d, (x, y), str(nd), 14, FG, na, bold=True, anchor='mm', bg=fill)
        lab = TOY_LABELS[nd]
        if nd <= 3:
            text(d, (x - 24, y), lab, 17, MUTED, na, anchor='rm')
        elif nd <= 5:
            text(d, (x + 24, y), lab, 17, MUTED, na, anchor='lm')


def gene_card(d, x, y, innov, w, on, a=1.0, border=None, w_cell=96, dim=False):
    """One connection gene: innovation number, from -> to, weight, enabled."""
    src, dst = TOY_GENES[innov]
    fill = mix(BG, PANEL, a * (0.5 if dim else 1.0))
    d.rounded_rectangle([x, y, x + w_cell - 8, y + 96], 8, fill=fill,
                        outline=mix(BG, border, a) if border else None, width=3 if border else 1)
    ta = a * (0.45 if dim else 1.0)
    cx = x + (w_cell - 8) // 2
    text(d, (cx, y + 16), f'#{innov}', 17, AMBER, ta, bold=True, anchor='mm', bg=fill)
    text(d, (cx, y + 40), f'{src} → {dst}', 17, FG, ta, anchor='mm', bg=fill)
    text(d, (cx, y + 62), f'{w:+.1f}', 16, AMBER if w > 0 else BLUE, ta, mono=True, anchor='mm', bg=fill)
    text(d, (cx, y + 82), 'on' if on else 'OFF', 13, GREEN if on else RED, ta, bold=not on, anchor='mm', bg=fill)


def scene_genome(ctx, name='genome'):
    """Beats: genotype (node + connection genes) -> phenotype (network); innovation numbers."""
    genes = {1: (1.2, True), 2: (-0.7, True), 3: (0.9, True), 4: (0.3, True)}
    box = (760, 170, 1080, 430)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'A genome is a recipe for a network', t, 'genotype → phenotype')
        a0 = fade(t, 0.3)
        text(d, (80, 140), 'node genes', 20, BLUE, a0, bold=True)
        for i, nd in enumerate([1, 2, 3, 4, 5]):
            x = 80 + i * 92
            kind = 'input' if nd <= 3 else 'output'
            d.rounded_rectangle([x, 172, x + 84, 232], 8, fill=mix(BG, PANEL, fade(t, 0.5 + 0.15 * i)))
            text(d, (x + 42, 190), f'node {nd}', 15, FG, fade(t, 0.5 + 0.15 * i), anchor='mm', bg=PANEL)
            text(d, (x + 42, 214), f'{kind}', 13, MUTED, fade(t, 0.5 + 0.15 * i), anchor='mm', bg=PANEL)
        text(d, (80, 262), 'connection genes', 20, AMBER, a0, bold=True)
        for i, innov in enumerate(sorted(genes)):
            w, on = genes[innov]
            gene_card(d, 80 + i * 104, 294, innov, w, on, fade(t, 1.0 + 0.3 * i), w_cell=104)
        draw_toy(d, genes, box, a=fade(t, b.at(0) + 2.0, 0.8))
        text(d, (920, 470), 'the network it builds', 17, MUTED, fade(t, 2.0), anchor='ma')
        a1 = fade(t, b.at(1))
        if a1 > 0:
            for i, innov in enumerate(sorted(genes)):
                w, on = genes[innov]
                gene_card(d, 80 + i * 104, 294, innov, w, on, 1.0, border=mix(PANEL, AMBER, a1), w_cell=104)
            lines = ['Every connection gene gets a historical marker, its innovation number,',
                     'the first time that connection appears anywhere in the population.',
                     'Genes with the same number in two genomes have the same origin.']
            for i, s in enumerate(lines):
                text(d, (80, 450 + 34 * i), s, 20, FG, fade(t, b.at(1) + 0.4 + 0.5 * i))
            text(d, (80, 580), 'neat-python names each gene by its (from, to) pair, which plays the same role.',
                 17, MUTED, fade(t, b.at(1) + 2.0))
        return img
    return Scene(name, render, [7.0, 8.0])


# ----------------------------------------------------------------------------
# Part 3: how NEAT changes genomes
# ----------------------------------------------------------------------------

def scene_mutation(ctx, name='mutation'):
    """Beats: intro; weight mutation; add connection; add node."""
    box = (620, 180, 980, 470)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Mutation: small random changes', t, 'three kinds')
        k, lt, dur = b.local(t)
        genes = {1: (1.2, True), 2: (-0.7, True), 3: (0.9, True), 4: (0.3, True)}
        hl, grow, node_a, title, sub = (), None, None, '', ''
        if k >= 1:
            u = ease((t - b.at(1) - 0.8) / 1.5)
            genes[4] = (0.3 + (-0.5 - 0.3) * u, True)
            if k == 1:
                hl = (4,)
                title, sub = 'change a weight', 'most mutations just nudge a weight or a bias'
        if k >= 2:
            u = ease((t - b.at(2) - 0.8) / 1.2)
            genes[5] = (0.8, True)
            if k == 2:
                grow, hl = {5: u}, (5,)
                title, sub = 'add a connection', 'join two nodes that were not connected'
        if k >= 3:
            u = ease((t - b.at(3) - 0.8) / 1.5)
            genes[1] = (1.2, u < 0.3)
            genes[6] = (1.0, True)
            genes[7] = (1.2, True)
            grow = {6: u, 7: u}
            node_a = {6: u}
            hl = (6, 7)
            title, sub = 'add a node', 'split a connection: the old gene is switched off, two new genes replace it'
        if k == 0:
            title, sub = 'the parent', 'this genome will be copied and mutated'
        draw_toy(d, genes, box, nodes=[1, 2, 3, 4, 5] + ([6] if k >= 3 else []), hl=hl, grow=grow,
                 node_a=node_a)
        text(d, (80, 170), title, 30, AMBER, fade(t, b.at(k) + 0.2), bold=True)
        for i, s in enumerate(wrap(sub, 38)):
            text(d, (80, 220 + 30 * i), s, 20, FG, fade(t, b.at(k) + 0.4))
        # gene strip
        x = 80
        for innov in sorted(genes):
            w, on = genes[innov]
            new = (k == 2 and innov == 5) or (k == 3 and innov in (6, 7))
            ca = fade(t, b.at(k) + 1.6) if new else 1.0
            gene_card(d, x, 540, innov, w, on, ca, border=FG if innov in hl else None, dim=not on)
            x += 96
        if k == 3:
            text(d, (80, 655), 'new connection into the node gets weight 1, the one out keeps the old '
                 'weight, so behaviour barely changes', 17, MUTED, fade(t, b.at(3) + 2.2))
        return img
    return Scene(name, render, [5.0, 6.0, 6.0, 8.0])


def scene_crossover(ctx, name='crossover'):
    """Beats: two parents lined up by innovation number; matching/disjoint/excess; the child."""
    A = {1: (1.2, True), 2: (-0.7, True), 3: (0.9, True), 4: (-0.5, True), 5: (0.8, True)}
    B = {1: (1.0, False), 2: (-0.4, True), 3: (1.1, True), 4: (0.2, True), 6: (1.0, True),
         7: (1.0, True), 8: (-0.6, True)}
    cols = list(range(1, 9))
    cw, x0 = 112, 250
    rng = np.random.default_rng(3)
    pick = {i: ('A' if rng.random() < 0.5 else 'B') for i in (1, 2, 3, 4)}
    child = {i: (A if pick.get(i, 'A') == 'A' else B)[i] for i in A}

    def kind(i):
        if i in A and i in B:
            return 'matching', GREEN
        if i > max(A) or i > max(B):
            return 'excess', VIOLET
        return 'disjoint', RED

    def row(d, y, genes, a, label, fit, sel=None):
        text(d, (x0 - 30, y + 30), label, 22, FG, a, bold=True, anchor='ra')
        text(d, (x0 - 30, y + 58), fit, 16, MUTED, a, anchor='ra')
        for c, i in enumerate(cols):
            if i in genes:
                w, on = genes[i]
                border = None
                if sel is not None:
                    border = kind(i)[1] if sel > 0 else None
                gene_card(d, x0 + c * cw, y, i, w, on, a, border=border, w_cell=cw, dim=not on)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Crossover: genes line up by history', t, 'innovation numbers make it possible')
        k = b.index(t)
        a1 = fade(t, b.at(1))
        row(d, 130, A, fade(t, 0.3), 'parent A', 'fitness 120', sel=a1 if k >= 1 else None)
        row(d, 250, B, fade(t, 0.8), 'parent B', 'fitness 60', sel=a1 if k >= 1 else None)
        if k >= 1:
            for c, i in enumerate(cols):
                lab, col = kind(i)
                text(d, (x0 + c * cw + (cw - 8) // 2, 362), lab, 15, col, a1, anchor='ma')
            text(d, (x0, 395), 'matching: both parents have it  ·  disjoint: a gap in the other  ·  '
                 'excess: past the end of the other', 16, MUTED, fade(t, b.at(1) + 1.5))
        a2 = fade(t, b.at(2))
        if a2 > 0:
            text(d, (x0 - 30, 460 + 30), 'child', 22, AMBER, a2, bold=True, anchor='ra')
            for c, i in enumerate(cols):
                if i in child:
                    w, on = child[i]
                    ca = fade(t, b.at(2) + 0.4 + 0.25 * c)
                    src = pick.get(i, 'A')
                    gene_card(d, x0 + c * cw, 460, i, w, on, ca, w_cell=cw, dim=not on)
                    text(d, (x0 + c * cw + (cw - 8) // 2, 566), f'from {src}', 14, MUTED, ca, anchor='ma')
            lines = ['matching genes: picked at random from either parent',
                     'disjoint and excess genes: taken from the fitter parent (A)']
            for i, s in enumerate(lines):
                text(d, (x0, 610 + 30 * i), s, 19, FG, fade(t, b.at(2) + 2.5 + 0.6 * i))
        return img
    return Scene(name, render, [7.0, 8.0, 9.0])


def scene_speciation(ctx, name='speciation'):
    """Beats: the problem (new structure starts weak); compatibility distance; species protect it."""
    c1 = ctx.config.genome_config.compatibility_disjoint_coefficient
    c2 = ctx.config.genome_config.compatibility_weight_coefficient
    thr = ctx.config.species_set_config.compatibility_threshold
    rng = np.random.default_rng(5)
    # an illustrative population: three clusters of genomes in "genome space"
    centers = [(330, 380), (560, 300), (470, 520)]
    pts = [(c, (cx + rng.normal(0, 38), cy + rng.normal(0, 30))) for c, (cx, cy) in enumerate(centers)
           for _ in range((22, 14, 6)[c])]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Speciation protects new ideas', t, 'compete mostly inside your own species')
        k = b.index(t)
        if k == 0:
            lines = ['A mutation that adds a node usually makes a genome worse at first:',
                     'its new weights are not tuned yet.',
                     'Ranked against the whole population, it would be thrown away',
                     'before it ever had the chance to become useful.']
            for i, s in enumerate(lines):
                text(d, (80, 170 + 44 * i), s, 24, FG, fade(t, 0.4 + 0.7 * i))
            return img
        a = fade(t, b.at(1))
        text(d, (80, 140), 'compatibility distance', 22, AMBER, a, bold=True)
        text(d, (80, 180), 'δ = (c₁·D + c₂·Σ|Δw|) / N', 30, FG, a, mono=True)
        for i, s_ in enumerate([f'D: genes that do not match (disjoint + excess), c₁ = {c1:g}',
                                f'Σ|Δw|: weight differences of the matching genes, c₂ = {c2:g}',
                                'N: genes in the larger genome',
                                'worked out for connection genes and node genes (biases), then added',
                                f'same species if δ < {thr:g}']):
            text(d, (90, 240 + 32 * i), s_, 18, MUTED if i < 4 else FG, fade(t, b.at(1) + 0.6 + 0.5 * i))
        if k >= 2:
            a2 = fade(t, b.at(2))
            for c, (x, y) in pts:
                col = mix(BG, species_color(c), a2) if a2 > 0 else MUTED
                d.ellipse([x + 520 - 6, y - 6, x + 520 + 6, y + 6], fill=mix(BG, col, 0.9))
            for c, (cx, cy) in enumerate(centers):
                r = 85
                d.ellipse([cx + 520 - r, cy - r * 0.8, cx + 520 + r, cy + r * 0.8],
                          outline=mix(BG, species_color(c), a2), width=2)
            lines = ['Each species shares its offspring among its members,',
                     'so a big species cannot crowd out a small new one.',
                     'A species that stops improving for 20 generations dies out.']
            for i, s in enumerate(lines):
                text(d, (80, 440 + 34 * i), s, 20, FG, fade(t, b.at(2) + 0.8 + 0.7 * i))
        else:
            for c, (x, y) in pts:
                d.ellipse([x + 520 - 6, y - 6, x + 520 + 6, y + 6], fill=mix(BG, MUTED, fade(t, b.at(1) + 1)))
        return img
    return Scene(name, render, [7.0, 9.0, 9.0])


def scene_loop(ctx, name='loop'):
    """Beats: the generation loop, one highlighted step per beat."""
    pop = ctx.config.pop_size
    eps = ctx.episodes
    steps = [('Evaluate', f'each of the {pop} genomes flies {eps} episodes; fitness = average reward'),
             ('Speciate', 'group similar genomes into species'),
             ('Select', 'inside each species, the best 20% become parents; the 2 best are kept unchanged'),
             ('Reproduce', 'crossover between parents, then mutation, back to a full population')]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'One generation', t, 'repeat a few hundred times')
        k = b.index(t)
        cx, cy, R = 640, 400, 190
        for i, (title, desc) in enumerate(steps):
            ang = -np.pi / 2 + i * np.pi / 2
            x, y = cx + R * 1.55 * np.cos(ang), cy + R * np.sin(ang)
            on = i == k or len(b.durations) == 1
            a = fade(t, 0.3 + 0.3 * i)
            col = AMBER if on else MUTED
            bw, bh = 400, 96
            d.rounded_rectangle([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], 12,
                                fill=mix(BG, PANEL, a), outline=mix(BG, col, a), width=3 if on else 1)
            text(d, (x, y - 26), f'{i + 1}. {title}', 22, col, a, bold=True, anchor='mm', bg=PANEL)
            for j, s in enumerate(wrap(desc, 40)):
                text(d, (x, y + 4 + 22 * j), s, 15, FG if on else MUTED, a, anchor='mm', bg=PANEL)
        text(d, (cx, cy), '↻', 70, mix(BG, AMBER, 0.6), fade(t, 1.5), anchor='mm')
        return img
    return Scene(name, render, [5.0, 4.0, 5.0, 5.0])


def scene_fitness(ctx, name='fitness'):
    """Beats: fitness; fair comparison (same seeds); validation on unseen episodes."""
    eps = ctx.episodes

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Fitness: what "good" means', t)
        d.rounded_rectangle([80, 150, 1200, 230], 12, fill=PANEL)
        text(d, (110, 172), f'fitness = average (reward + centre bonus) over {eps} episodes', 27, FG,
             fade(t, 0.3), mono=True, bg=PANEL)
        items = [(1, 'Every genome in a generation flies the same starting positions in the same wind, so the '
                     'comparison is fair. Starts and wind change every generation, so nobody can memorise one.'),
                 (2, 'A score from 5 windy episodes is noisy: the top genome is often just lucky. So the 10 best '
                     'also fly 30 more before they are kept, and the champion flies 100 episodes no genome trained on.'),
                 (2, f'Solved = an average game score of {ctx.run.threshold:.0f} or more on those 100 unseen '
                     'episodes (without the bonus).')]
        for i, (beat, s) in enumerate(items):
            a = fade(t, b.at(beat) + 0.3 + (0.9 if i == 2 else 0))
            for j, line in enumerate(wrap(s, 80)):
                text(d, (90, 280 + 120 * i + 30 * j), line, 21, AMBER if i == 2 else FG, a)
        return img
    return Scene(name, render, [5.0, 7.0, 9.0])


# ----------------------------------------------------------------------------
# Part 4: the real run
# ----------------------------------------------------------------------------

def _gen_progress(t, b, start_beat, keyframes):
    """Generation shown at time t: `keyframes` = generation range covered by each beat."""
    k, lt, dur = b.local(t)
    if k < start_beat:
        return keyframes[0][0]
    idx = min(k - start_beat, len(keyframes) - 1)
    g0, g1 = keyframes[idx]
    u = ease(lt / max(0.1, dur - 0.8))
    return g0 + (g1 - g0) * u


def scene_gen_fitness(ctx, name='gen_fitness', keyframes=None):
    """Every genome's fitness, one column of dots per generation, coloured by species."""
    run = ctx.run
    G = run.G
    keyframes = keyframes or ((0, 0), (0, min(G - 1, 60)), (min(G - 1, 60), G - 1))
    px0, py0, pw, ph = 130, 140, 940, 440
    fit = np.array([r['best'] for r in run.fitness])
    ylo, yhi = -400.0, max(320.0, float(fit.max()) + 20)
    val = {r['generation']: r for r in run.validation}

    def X(g):
        return px0 + g / max(1, G - 1) * pw

    def Y(v):
        return py0 + ph - (np.clip(v, ylo, yhi) - ylo) / (yhi - ylo) * ph

    dots = Image.new('RGB', (W, H), BG)
    dd = ImageDraw.Draw(dots)
    rng = np.random.default_rng(0)
    for r in run.population:
        x = X(r['generation']) + rng.uniform(-1.2, 1.2)
        y = Y(r['fitness'])
        dd.ellipse([x - 1.6, y - 1.6, x + 1.6, y + 1.6], fill=mix(BG, species_color(r['species']), 0.75))
    best = fit
    mean = np.array([r['mean'] for r in run.fitness])
    nspec = [r['species'] for r in run.fitness]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Every genome, every generation', t,
                f'{G} generations × {ctx.config.pop_size} genomes, coloured by species')
        gp = _gen_progress(t, b, 0, keyframes)
        k = int(gp) + 1
        xcut = int(X(gp) + 3)
        img.paste(dots.crop((px0 - 3, py0, xcut, py0 + ph)), (px0 - 3, py0))
        d.rectangle([px0, py0, px0 + pw, py0 + ph], outline=PANEL, width=2)
        for v in np.arange(np.ceil(ylo / 100) * 100, yhi, 100):
            y = Y(v)
            d.line([xcut, y, px0 + pw, y], fill=PANEL)
            text(d, (px0 - 10, y), f'{v:.0f}' if v > ylo else f'≤{v:.0f}', 15, MUTED, anchor='rm')
        ty = Y(run.threshold)
        d.line([px0, ty, px0 + pw, ty], fill=mix(BG, AMBER, 0.5), width=1)
        text(d, (px0 + 6, ty - 4), f'solved ({run.threshold:.0f})', 14, AMBER, anchor='lb')
        step = 50 if G > 300 else 25
        for g in range(0, G, step):
            text(d, (X(g), py0 + ph + 12), str(g), 15, MUTED, anchor='ma')
        text(d, (px0 + pw / 2, py0 + ph + 34), 'generation', 16, MUTED, anchor='ma')
        text(d, (px0 - 60, py0 - 28), 'fitness', 16, MUTED)
        last = len(b.durations) - 1
        show = fade(t, b.at(last)) if last > 0 else 1.0
        if k > 1 and show > 0:
            pts = [(X(g), Y(v)) for g, v in enumerate(mean[:k])]
            d.line(pts, fill=mix(BG, BLUE, show), width=2)
            vp = [(X(g), Y(val[g]['reward'])) for g in range(k) if g in val]
            if len(vp) > 1:
                d.line(vp, fill=mix(BG, GREEN, show), width=3)
        g = min(k - 1, G - 1)
        x = 1100
        v = val.get(g)
        text(d, (x, 150), f'gen {g}', 30, FG, bold=True)
        text(d, (x, 200), 'species', 15, MUTED)
        text(d, (x, 220), f'{nspec[g]}', 30, FG, bold=True)
        text(d, (x, 272), 'best (5 episodes)', 15, MUTED)
        text(d, (x, 292), f'{best[g]:.0f}', 30, AMBER, bold=True)
        if v:
            text(d, (x, 344), 'champion on', 15, MUTED)
            text(d, (x, 362), '100 unseen', 15, MUTED)
            text(d, (x, 382), f'{v["reward"]:.0f}', 30, GREEN, bold=True)
            text(d, (x, 420), f'{100 * v["landed"]:.0f}% land', 16, GREEN)
            text(d, (x, 442), f'{100 * v["centred"]:.0f}% centred', 16, BLUE)
        if show > 0:
            for i, (col, lab) in enumerate([(BLUE, 'average'), (GREEN, 'champion, unseen')]):
                y = 480 + i * 26
                d.rectangle([x, y + 8, x + 16, y + 11], fill=mix(BG, col, show))
                text(d, (x + 22, y), lab, 15, MUTED, show)
        return img
    return Scene(name, render, [5.0, 7.0, 8.0][:len(keyframes)], min_beats=[0, 6.0, 7.0][:len(keyframes)])


def scene_species(ctx, name='species'):
    """Beats: species sizes stacked over the run (reveal); the story (births and extinctions)."""
    run = ctx.run
    gens = sorted({r['generation'] for r in run.species})
    sids = sorted({r['species'] for r in run.species})
    size = {(r['species'], r['generation']): r['size'] for r in run.species}
    G = len(gens)
    pop = ctx.config.pop_size
    px0, py0, pw, ph = 120, 150, 1040, 430

    def X(g):
        return px0 + g / max(1, G - 1) * pw

    layer = Image.new('RGB', (W, H), BG)
    ld = ImageDraw.Draw(layer)
    base = np.zeros(G)
    for sid in sids:
        s = np.array([size.get((sid, g), 0) for g in gens], dtype=float)
        if s.sum() == 0:
            continue
        top = base + s
        poly = [(X(g), py0 + ph - base[g] / pop * ph) for g in range(G)] + \
               [(X(g), py0 + ph - top[g] / pop * ph) for g in reversed(range(G))]
        ld.polygon(poly, fill=mix(BG, species_color(sid), 0.8))
        base = top
    alive = [r['species'] for r in run.fitness]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Species over the run', t, f'{len(sids)} species appeared; each band is one species')
        gp = _gen_progress(t, b, 0, ((0, G - 1),))
        xcut = int(X(gp)) + 2
        img.paste(layer.crop((px0, py0, xcut, py0 + ph)), (px0, py0))
        d.rectangle([px0, py0, px0 + pw, py0 + ph], outline=PANEL, width=2)
        step = 50 if G > 300 else 25
        for g in range(0, G, step):
            text(d, (X(g), py0 + ph + 12), str(g), 15, MUTED, anchor='ma')
        text(d, (px0 + pw / 2, py0 + ph + 34), 'generation', 16, MUTED, anchor='ma')
        text(d, (px0 - 12, py0), f'{pop}', 15, MUTED, anchor='ra')
        text(d, (px0 - 12, py0 + ph), '0', 15, MUTED, anchor='rb')
        text(d, (px0 - 12, py0 + ph / 2), 'genomes', 15, MUTED, anchor='rm')
        g = int(min(gp, G - 1))
        text(d, (px0, py0 - 30), f'generation {g}: {alive[g]} species', 18, FG)
        return img
    return Scene(name, render, [8.0, 7.0], min_beats=[7.0, 0])


def scene_gen_ghosts(ctx, name='gen_ghosts', gens_by_beat=None):
    """All genomes of a generation flying the same start, drawn as trails."""
    run = ctx.run
    snaps = list(run.snapshots)
    G = run.G

    def near(g):
        return min(snaps, key=lambda s: abs(s - g))
    if gens_by_beat is None:
        late = [near(G // 4), near(G // 2), near(G - 1)]
        gens_by_beat = ((0,), (0, near(5), near(10)), (near(20), near(30), near(50)), tuple(dict.fromkeys(late)))
    gens = sorted({g for gs in gens_by_beat for g in gs})
    seed = ctx.seed
    paths = ctx.ghosts(gens, seed)
    bg = terrain_frame(seed)
    bx, by, bw, bh = 40, 125, 780, 520
    bgimg = Image.fromarray(bg).resize((bw, bh), Image.BILINEAR)
    bgimg = Image.blend(bgimg, Image.new('RGB', (bw, bh), BG), 0.35)
    sx, sy = bw / VIEW_W, bh / VIEW_H
    best_val = np.array([r['reward'] for r in run.validation])
    pop = ctx.config.pop_size

    def slot(t, b):
        k, lt, dur = b.local(t)
        gs = gens_by_beat[min(k, len(gens_by_beat) - 1)]
        if k == 0 and len(gens_by_beat) > 1 and gens_by_beat[1][0] == gs[0]:
            return gs[0], ease(lt / max(1.0, dur)) * 0.6
        n = len(gs)
        per = dur / n
        i = min(int(lt / per), n - 1)
        u = (lt - i * per) / max(0.1, per * 0.75)
        if k == 1 and gs[0] == gens_by_beat[0][0] and i == 0:
            u = 0.6 + 0.4 * u
        return gs[i], min(1.0, u)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, f'One generation, {pop} flights', t, 'every genome starts from the same spot')
        g, u = slot(t, b)
        eps = paths[g]
        layer = bgimg.copy().convert('RGBA')
        over = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
        od = ImageDraw.Draw(over)
        longest = max(len(e['path']) for e in eps)
        upto = int(u * longest) + 1
        counts = {'landed': 0, 'crashed': 0, 'timed out': 0}
        for e in eps:
            p = [(x * sx, y * sy) for x, y in e['path'][:upto]]
            done = upto >= len(e['path'])
            lab, col = outcome(e)
            c = col if done else FG
            if len(p) > 1:
                od.line(p, fill=c + (90,), width=2)
            hx, hy = p[-1]
            od.ellipse([hx - 3, hy - 3, hx + 3, hy + 3], fill=c + (230,))
            if done:
                counts[lab] += 1
        layer = Image.alpha_composite(layer, over).convert('RGB')
        img.paste(layer, (bx, by))
        d.rectangle([bx, by, bx + bw, by + bh], outline=PANEL, width=2)

        x = 860
        text(d, (x, 140), 'generation', 20, MUTED)
        text(d, (x, 165), f'{g}', 64, FG, bold=True)
        for i, (lab, col) in enumerate([('landed', GREEN), ('crashed', RED), ('timed out', MUTED)]):
            y = 270 + i * 44
            text(d, (x, y), lab, 22, col)
            text(d, (x + 330, y), f'{counts[lab]}', 28, col, bold=True, anchor='ra')
        sx0, sy0, sw, sh = x, 450, 340, 150
        d.rectangle([sx0, sy0, sx0 + sw, sy0 + sh], outline=PANEL, width=2)
        lo, hi = -300.0, 300.0
        pts = [(sx0 + i / (G - 1) * sw, sy0 + sh - (np.clip(v, lo, hi) - lo) / (hi - lo) * sh)
               for i, v in enumerate(best_val)]
        d.line(pts, fill=mix(BG, GREEN, 0.7), width=2)
        mx = sx0 + g / (G - 1) * sw
        d.line([mx, sy0, mx, sy0 + sh], fill=FG, width=2)
        text(d, (sx0, sy0 + sh + 8), 'champion on 100 unseen episodes', 15, MUTED)
        return img
    beats = [5.0, 6.0, 7.0, 7.0][:len(gens_by_beat)]
    mins = [0.0] + [2.2 * len(gs) for gs in gens_by_beat[1:]]
    return Scene(name, render, beats, min_beats=mins)


def growth_gens(run, n=6):
    """Generations where the champion's shape changed noticeably, spread over the run."""
    G = run.G
    picks = sorted({0, run.winner_gen, *[int(round(x)) for x in np.linspace(0, G - 1, n)]})
    return [g for g in picks if g in run.champions]


def scene_growth(ctx, name='growth'):
    """Beats: the champion's network at several generations; size over the run."""
    run = ctx.run
    gens = growth_gens(run)
    hid = np.array([r['best_hidden'] for r in run.complexity])
    con = np.array([r['best_conns'] for r in run.complexity])
    mhid = np.array([r['mean_hidden'] for r in run.complexity])
    mcon = np.array([r['mean_conns'] for r in run.complexity])
    G = run.G
    box = (180, 150, 760, 560)
    pics = {}
    for g in gens:
        im = Image.new('RGB', (W, H), BG)
        draw_network(im, run.champions[g], ctx.config, box)
        pics[g] = im

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'How the network changed', t, 'the champion of each generation')
        dur = b.end - 0.5
        u = min(0.999, max(0.0, t / dur)) * (len(gens) - 1)
        i = int(u)
        f = ease((u - i) * 3 - 2)                       # hold, then cross-fade in the last third
        g0, g1 = gens[i], gens[min(i + 1, len(gens) - 1)]
        img.paste(Image.blend(pics[g0], pics[g1], f).crop((0, 120, 860, 620)), (0, 120))
        g = g1 if f > 0.5 else g0
        h, c = run.size(run.champions[g])
        text(d, (60, 600), f'generation {g}', 26, FG, bold=True)
        text(d, (330, 604), f'{h} hidden nodes · {c} connections', 21, AMBER)
        # size over the run
        x0, y0, w, h_ = 880, 170, 340, 160
        for j, (series, mean, lab, col) in enumerate([(con, mcon, 'connections', BLUE),
                                                     (hid, mhid, 'hidden nodes', AMBER)]):
            yy = y0 + j * (h_ + 70)
            lo, hi = 0, max(3, series.max(), mean.max()) * 1.1
            d.rectangle([x0, yy, x0 + w, yy + h_], outline=PANEL, width=2)
            pts = [(x0 + q / (G - 1) * w, yy + h_ - (v - lo) / (hi - lo) * h_) for q, v in enumerate(mean)]
            d.line(pts, fill=mix(BG, MUTED, 0.8), width=2)
            pts = [(x0 + q / (G - 1) * w, yy + h_ - (v - lo) / (hi - lo) * h_) for q, v in enumerate(series)]
            d.line(pts, fill=col, width=2)
            mx = x0 + g / (G - 1) * w
            d.line([mx, yy, mx, yy + h_], fill=FG, width=1)
            text(d, (x0, yy - 24), f'{lab}: champion vs. population average', 15, MUTED)
            text(d, (x0 + w + 8, yy), f'{hi:.0f}', 13, MUTED)
            text(d, (x0 + w + 8, yy + h_ - 14), '0', 13, MUTED)
        return img
    return Scene(name, render, [8.0, 8.0], min_beats=[0, 1.6 * len(gens)])


def scene_evolution(ctx, name='evolution'):
    """Beats: the line-up; playback (champions at different generations flying the same start)."""
    run = ctx.run
    G = run.G
    wanted = [g for g in (0, 10, 25, 50, 100) if g < run.winner_gen]
    picks = [(f'generation {g}', run.champions[g]) for g in wanted]
    picks.append((f'winner (generation {run.winner_gen})', ctx.winner))
    eps = [fly(gn, ctx.config, ctx.seed) for _, gn in picks]
    tw, th, gx, gy = 318, 212, 28, 72
    x0, y0 = (W - (3 * tw + 2 * gx)) // 2, 150

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Evolution in action', t, 'the champion of each generation, same start every time')
        play = max(0.0, t - b.at(1) + 0.3) if b.at(1) < 1e9 else max(0.0, t - 0.5)
        step = int(play * FPS * 1.6)
        for i, ((label, _), ep) in enumerate(zip(picks, eps)):
            r, c = divmod(i, 3)
            x, y = x0 + c * (tw + gx), y0 + r * (th + gy)
            k = min(step, len(ep['frames']) - 1)
            done = step >= len(ep['frames']) - 1
            lab, col = outcome(ep)
            paste_flight(img, ep, k, (x, y, tw, th), border=col if done else PANEL)
            text(d, (x, y - 26), label, 19, AMBER, bold=True)
            if done:
                text(d, (x + tw, y + th + 8), f'{lab}  {ep["reward"]:+.0f}', 17, col, anchor='ra')
        return img
    longest = max(len(e['frames']) for e in eps) / (FPS * 1.6)
    return Scene(name, render, [5.0, max(8.0, longest + 1.5)], min_beats=[0, longest + 1.0])


# ----------------------------------------------------------------------------
# Part 5: the result
# ----------------------------------------------------------------------------

def conn_label(ctx, key):
    a, b = key
    return f'{ctx.node_name(a)} → {ctx.node_name(b)}'


def scene_read(ctx, name='read'):
    """Beats: the winner's strongest connections, explained one by one while it flies."""
    ep = ctx.best_ep
    n = len(ep['actions'])
    abl = ctx.ablation()
    top = abl['ranked'][:3]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Reading the evolved network', t, 'which connections matter?')
        k = step_at(n, t, speed=0.8)
        paste_flight(img, ep, k, (50, 150, 450, 300))
        kb = b.index(t)
        hl = tuple(key for key, _ in top[:kb]) if kb > 0 else ()
        draw_network(img, ctx.winner, ctx.config, (660, 150, 1060, 560), values=ep['values'][k],
                     action=ep['actions'][k], highlight=hl, hidden_names=ctx.hidden_alias)
        action_bars(d, ep, k, 70, 480)
        if kb > 0:
            key, rate = top[min(kb, len(top)) - 1]
            text(d, (660, 600), f'{conn_label(ctx, key)}: weight {ctx.winner.connections[key].weight:+.2f}', 20,
                 AMBER, fade(t, b.at(kb)))
            text(d, (660, 630), f'without it: {100 * rate:.0f}% landings (instead of {100 * abl["full"]:.0f}%)',
                 18, FG, fade(t, b.at(kb) + 0.4))
        return img
    return Scene(name, render, [6.0] + [6.0] * len(top))


def scene_ablation(ctx, name='ablation'):
    """Beats: switch off one connection at a time; most are spare, a few are vital; the hidden nodes off."""
    abl = ctx.ablation()
    ranked = abl['ranked']
    show = ranked[:12]
    nh = len(ctx.hidden_alias)
    x0, y0, bw, rh = 360, 150, 600, 36

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Switch one connection off', t, f'landing rate on {ABLATION_EPISODES} episodes, '
                f'{len(ranked)} connections tried one at a time')
        full = abl['full']
        fx = x0 + full * bw
        for i, (key, r) in enumerate(show):
            a = fade(t, 0.4 + 0.15 * i)
            y = y0 + i * rh
            text(d, (x0 - 14, y + 4), conn_label(ctx, key), 18, FG, a, anchor='ra')
            col = RED if r < full - 0.3 else (AMBER if r < full - 0.1 else GREEN)
            d.rectangle([x0, y + 2, x0 + int(r * bw * ease((t - 0.4 - 0.15 * i) / 0.8)), y + rh - 8],
                        fill=mix(BG, col, a))
            text(d, (x0 + bw + 70, y + 4), f'{100 * r:.0f}%', 17, col, a, anchor='ra')
        d.line([fx, y0 - 10, fx, y0 + len(show) * rh], fill=FG, width=2)
        text(d, (fx, y0 - 14), f'all connections: {100 * full:.0f}%', 15, FG, anchor='mb')
        a = fade(t, b.at(1))
        spare = sum(1 for _, r in ranked if r >= full - 0.05)
        text(d, (80, 616), f'{spare} of {len(ranked)} connections can go with almost no loss; '
             f'a handful carry the whole skill.', 21, AMBER, a)
        a = fade(t, b.at(2))
        if a > 0 and not nh:
            text(d, (80, 652), 'No hidden nodes and no loops: every connection runs straight from an input to an '
                 'action.', 21, VIOLET, a, bold=True)
        if a > 0 and nh:
            off, on = ctx.without_hidden(), ctx.stats
            what = 'the hidden node' if nh == 1 else f'all {nh} hidden nodes'
            text(d, (80, 652), f'Without {what}: {off["landing_rate"]:.0f}% landings, score '
                 f'{off["avg_reward"]:.0f}  (with: {on["landing_rate"]:.0f}%, {on["avg_reward"]:.0f}; '
                 '100 test flights)', 21, VIOLET, a, bold=True)
        return img
    return Scene(name, render, [7.0, 7.0, 7.0])


def scene_harder(ctx, name='harder'):
    """Beats: the no-wind champion and the new winner fly the same windy start; the numbers."""
    base = ctx.baseline()
    seed = base['seed']
    old = fly(base['genome'], base['config'], seed)
    new = fly(ctx.winner, ctx.config, seed)
    s = ctx.stats
    tw, th, y0 = 560, 373, 150
    xs = (60, 660)
    rows = [('no-wind champion, calm air', base['calm'], MUTED),
            ('no-wind champion, in wind', base['wind'], RED),
            ('new winner, in wind', s, GREEN)]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Is the wind hard enough?', t, 'the old no-wind champion against the new winner')
        step = int(max(0.0, t - 0.6) * FPS * 1.4)
        a1 = fade(t, b.at(1))
        for x, (label, ep) in zip(xs, [('old champion (trained without wind)', old), ('new winner', new)]):
            k = min(step, len(ep['frames']) - 1)
            done = step >= len(ep['frames']) - 1
            lab, col = outcome(ep)
            paste_flight(img, ep, k, (x, y0, tw, th), border=col if done else PANEL, dim=0.6 * a1)
            text(d, (x, y0 - 28), label, 19, AMBER, bold=True)
            if done:
                where = 'in the middle' if ep['landed'] and abs(ep['x_final']) < CENTRED else ''
                text(d, (x + tw, y0 + th + 8), f'{lab} {where}  {ep["reward"]:+.0f}', 17, col, anchor='ra')
        if a1 > 0:
            d.rounded_rectangle([140, 200, 1140, 490], 14, fill=mix(BG, PANEL, a1))
            for j, h in enumerate(['landed', 'in the middle', 'game score']):
                text(d, (720 + j * 150, 222), h, 16, MUTED, a1, anchor='ma', bg=PANEL)
            for i, (lab, st, col) in enumerate(rows):
                a = fade(t, b.at(1) + 0.4 + 0.8 * i)
                y = 272 + i * 62
                text(d, (170, y), lab, 22, col, a, bold=True, bg=PANEL)
                for j, v in enumerate([f'{st["landing_rate"]:.0f}%', f'{st["centred_rate"]:.0f}%',
                                       f'{st["avg_reward"]:.0f}']):
                    text(d, (720 + j * 150, y), v, 24, FG, a, anchor='ma', mono=True, bg=PANEL)
            text(d, (W // 2, 456), '100 test episodes each, never seen in training', 16, MUTED, a1, anchor='ma',
                 bg=PANEL)
        return img
    longest = max(len(old['frames']), len(new['frames'])) / (FPS * 1.4)
    return Scene(name, render, [max(7.0, longest + 1.5), 9.0], min_beats=[longest + 1.0, 0])


def _results(*parts):
    with open(os.path.join(REPO_ROOT, 'results', *parts)) as fh:
        return json.load(fh)


def scene_why(ctx, name='why', analysis='run_wind_20261020'):
    """Beats: failure rate by steady wind; NEAT's lucky pick vs the best genome it had (run A)."""
    fa = _results('failure_analysis', analysis, 'summary.json')
    rates = fa['failure_rate_by_mean_wind']
    names = ['strong\nleft', 'moderate\nleft', 'little or\nnone', 'moderate\nright', 'strong\nright']
    noise = fa['selection_noise']
    gens = sorted(int(g) for g in noise)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Why the first run failed', t, '1,000 test flights and 7 re-flown populations')
        # left: failure rate by steady wind
        x0, y0, ph, bw = 90, 220, 300, 78
        text(d, (x0, 145), 'failure rate by the average wind over a flight', 19, MUTED, fade(t, 0.3))
        for i, (k, (r, n)) in enumerate(rates.items()):
            a = fade(t, 0.5 + 0.25 * i)
            x = x0 + i * (bw + 22)
            h = int(ph * r * ease((t - 0.5 - 0.25 * i) / 0.8))
            col = RED if r > 0.4 else mix(BG, MUTED, 0.7)
            d.rounded_rectangle([x, y0 + ph - h, x + bw, y0 + ph], 4, fill=mix(BG, col, a))
            text(d, (x + bw // 2, y0 + ph - h - 30), f'{100 * r:.0f}%', 24, FG, a, bold=True, anchor='ma')
            for j, line in enumerate(names[i].split('\n')):
                text(d, (x + bw // 2, y0 + ph + 10 + 22 * j), line, 17, MUTED, a, anchor='ma')
        # right: lucky pick vs best genome
        a1 = fade(t, b.at(1))
        if a1 > 0:
            px, py, pw2, ph2 = 700, 190, 500, 330
            text(d, (px, 150), 'top genome: 5 lucky flights vs 100 fresh flights', 19, MUTED, a1)
            lo, hi = -50, 250

            def Y(v):
                return py + ph2 - (np.clip(v, lo, hi) - lo) / (hi - lo) * ph2
            for v in (0, 100, 200):
                d.line([px, Y(v), px + pw2, Y(v)], fill=mix(BG, PANEL, a1), width=1)
                text(d, (px - 8, Y(v)), str(v), 15, MUTED, a1, anchor='rm')
            for i, g in enumerate(gens):
                n = noise[str(g)]
                x = px + 30 + i * (pw2 - 60) / (len(gens) - 1)
                aa = fade(t, b.at(1) + 0.3 + 0.2 * i)
                d.line([x, Y(n['train_best_train']), x, Y(n['train_best_true'])], fill=mix(BG, MUTED, aa), width=2)
                d.ellipse([x - 7, Y(n['train_best_train']) - 7, x + 7, Y(n['train_best_train']) + 7],
                          fill=mix(BG, AMBER, aa))
                d.ellipse([x - 7, Y(n['train_best_true']) - 7, x + 7, Y(n['train_best_true']) + 7],
                          fill=mix(BG, GREEN, aa))
                d.rectangle([x + 10, Y(n['true_best']) - 6, x + 22, Y(n['true_best']) + 6], fill=mix(BG, BLUE, aa))
                text(d, (x, py + ph2 + 10), str(g), 15, MUTED, aa, anchor='ma')
            for j, (col, lab) in enumerate([(AMBER, "NEAT's pick, 5 flights"), (GREEN, 'same genome, 100 flights'),
                                            (BLUE, 'best genome it had')]):
                yy = py + ph2 + 50 + 26 * j
                d.ellipse([px, yy + 4, px + 12, yy + 16], fill=mix(BG, col, a1))
                text(d, (px + 22, yy), lab, 17, FG, a1)
        return img
    return Scene(name, render, [8.0, 9.0])


def scene_memory(ctx, name='memory'):
    """Beats: a reflex sees calm air and a steady push as the same state; a loop keeps a running total."""
    def node(d, xy, label, a, r=34, ring=None):
        x, y = xy
        d.ellipse([x - r, y - r, x + r, y + r], fill=mix(BG, PANEL, a), outline=mix(BG, ring or MUTED, a), width=3)
        text(d, (x, y), label, 22, FG, a, anchor='mm', bg=PANEL)

    def arrow(d, p, q, a, col=MUTED, width=3):
        d.line([p, q], fill=mix(BG, col, a), width=width)
        ang = np.arctan2(q[1] - p[1], q[0] - p[0])
        tip = [q, (q[0] - 14 * np.cos(ang - 0.4), q[1] - 14 * np.sin(ang - 0.4)),
               (q[0] - 14 * np.cos(ang + 0.4), q[1] - 14 * np.sin(ang + 0.4))]
        d.polygon(tip, fill=mix(BG, col, a))

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'What a reflex cannot do', t, 'and how memory could help')
        a0 = fade(t, 0.4)
        for i, (title, sub, act) in enumerate([('calm air', 'hovering still', 'do nothing'),
                                               ('steady wind →', 'hovering still', 'keep firing')]):
            x = 120 + i * 300
            aa = fade(t, 0.6 + 1.2 * i)
            d.rounded_rectangle([x, 160, x + 250, 380], 12, fill=mix(BG, PANEL, aa))
            text(d, (x + 125, 185), title, 22, AMBER if i else FG, aa, bold=True, anchor='ma', bg=PANEL)
            text(d, (x + 125, 225), sub, 19, MUTED, aa, anchor='ma', bg=PANEL)
            text(d, (x + 125, 265), 'speed 0 · tilt 0', 19, FG, aa, anchor='ma', bg=PANEL, mono=True)
            text(d, (x + 125, 330), f'needs: {act}', 20, GREEN if i == 0 else RED, aa, anchor='ma', bg=PANEL)
        for j, line in enumerate(['Same 8 inputs, different answer needed.',
                                  'A feed-forward network can only give one',
                                  'answer, so it lets the wind win a little,', 'again and again.']):
            text(d, (120, 420 + 32 * j), line, 21, FG, fade(t, 3.0 + 0.3 * j))
        a1 = fade(t, b.at(1))
        if a1 > 0:
            cx = 820
            node(d, (cx, 300), 'vx', a1)
            node(d, (cx + 160, 300), 'h', a1, ring=AMBER)
            node(d, (cx + 320, 300), 'right', a1, r=40)
            arrow(d, (cx + 36, 300), (cx + 124, 300), a1)
            arrow(d, (cx + 196, 300), (cx + 278, 300), a1)
            d.arc([cx + 125, 200, cx + 195, 280], 150, 390, fill=mix(BG, AMBER, a1), width=4)
            text(d, (cx + 160, 180), 'loop: remembers', 18, AMBER, a1, anchor='ma')
            text(d, (cx - 60, 380), 'h(now) = tanh( h(before) + 0.1·vx(now) )', 18, FG, a1, mono=True)
            text(d, (cx - 60, 425), 'a running total of the drift:', 20, MUTED, a1)
            text(d, (cx - 60, 455), 'the "I" in a PID controller', 20, MUTED, a1)
        return img
    return Scene(name, render, [11.0, 10.0])


def scene_compare(ctx, name='compare'):
    """Beats: A/B/C on the same 1,000 flights; what happened to the loops in run C."""
    cmp_ = _results('comparison', 'summary.json')
    res = cmp_['results']
    rows = [('A', 'first run', MUTED), ('B', 'steadier search', BLUE), ('C', 'B + memory allowed', GREEN)]
    snaps = runs_loops(ctx)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Three runs, the same 1,000 flights', t, 'winners compared flight by flight')
        x0, bw = 330, 520
        for i, (k, lab, col) in enumerate(rows):
            r = res[k]
            a = fade(t, 0.4 + 0.5 * i)
            y = 170 + i * 120
            text(d, (x0 - 20, y + 4), f'{k}: {lab}', 22, FG, a, bold=True, anchor='ra')
            w = int(bw * r['landed'] * ease((t - 0.4 - 0.5 * i) / 0.9))
            d.rounded_rectangle([x0, y, x0 + max(w, 6), y + 34], 4, fill=mix(BG, col, a))
            lo, hi = r['landed_ci']
            d.line([x0 + bw * lo, y + 17, x0 + bw * hi, y + 17], fill=mix(BG, FG, a), width=2)
            text(d, (x0 + bw + 20, y + 2), f'{100 * r["landed"]:.0f}% landed', 22, FG, a)
            text(d, (x0 + bw + 20, y + 32), f'score {r["reward"]:.0f}', 18, MUTED, a)
        text(d, (x0, 530), 'line on each bar: 95% confidence interval (bootstrap)', 16, MUTED, fade(t, 2.0))
        a1 = fade(t, b.at(1))
        if a1 > 0:
            d.rectangle([0, 560, W, H], fill=BG)
            px, py, pw2, ph2 = 330, 590, 520, 80
            text(d, (px - 20, py + 30), 'run C: genomes with a loop', 19, AMBER, a1, anchor='ra')
            pts = [(px + g / 499 * pw2, py + ph2 - v * ph2) for g, v in snaps]
            d.line(pts, fill=mix(BG, AMBER, a1), width=3)
            for g, v in snaps:
                text(d, (px + g / 499 * pw2, py + ph2 - v * ph2 - 22), f'{100 * v:.0f}%', 14, FG, a1, anchor='ma')
                text(d, (px + g / 499 * pw2, py + ph2 + 8), f'gen {g}', 13, MUTED, a1, anchor='ma')
            text(d, (px + pw2 + 20, py + 50), 'evolution dropped the memory', 19, AMBER, a1)
        return img
    return Scene(name, render, [11.0, 9.0])


def runs_loops(ctx):
    """[(generation, share of run C's population with a loop)] from its population snapshots."""
    def run():
        import sys
        sys.path.insert(0, os.path.join(REPO_ROOT, 'experiments'))
        from compare_runs import loop_connections
        from neat_lander import runs
        snaps = runs.population_snapshots(os.path.join(REPO_ROOT, 'runs', 'run_wind_c'))
        out = []
        for g in (0, 50, 100, 200, 300, 400, 499):
            pop = list(runs.load_pickle(snaps[g])['population'].values())
            out.append((g, sum(1 for x in pop if loop_connections(x)) / len(pop)))
        return out
    return ctx.cached('loops_c', run)


def scene_outro(ctx, name='outro'):
    s = ctx.stats
    h, c = ctx.run.size(ctx.winner)
    items = [(f'{s["landing_rate"]:.0f}%', 'landing rate', GREEN),
             (f'{s["centred_rate"]:.0f}%', 'landed in the middle', BLUE),
             (f'{s["avg_reward"]:.0f}', 'average game score', AMBER),
             (f'{h}', 'hidden nodes', VIOLET),
             (f'{c}', 'connections', FG)]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, 'Results', t, '100 new windy episodes, never seen in training')
        for i, (v, lab, col) in enumerate(items):
            x = 160 + i * 240
            a = fade(t, 0.5 + 0.3 * i)
            text(d, (x, 260), v, 72, col, a, bold=True, anchor='mm')
            text(d, (x, 320), lab, 20, MUTED, a, anchor='mm')
        a = fade(t, b.at(1))
        text(d, (W // 2, 450), REPO_URL, 34, FG, a, bold=True, anchor='mm')
        text(d, (W // 2, 505), 'pip install -r requirements.txt  &&  python train.py', 20, MUTED, a,
             mono=True, anchor='mm')
        return img
    return Scene(name, render, [6.0, 6.0])
