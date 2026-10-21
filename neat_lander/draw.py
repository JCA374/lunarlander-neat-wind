"""Drawing a NEAT network with Pillow (used by networks.py and the video).

Layout: inputs on the left, actions on the right, each hidden node one column further right than
the deepest node feeding it. Orange connections are positive, blue negative; thickness is |weight|.
Recurrent genomes: a loop to itself is a ring above the node, a backward connection is a curve below.
"""

import os

from PIL import Image, ImageDraw, ImageFont

from .labels import ACTION_SHORT, INPUT_SHORT

BG = (11, 16, 32)
PANEL = (22, 30, 54)
FG = (232, 236, 245)
MUTED = (140, 152, 180)
AMBER = (255, 184, 64)
BLUE = (96, 168, 255)
GREEN = (84, 212, 140)
RED = (240, 96, 96)
VIOLET = (190, 120, 255)

FONT_DIR = '/usr/share/fonts/truetype/dejavu'
_fonts = {}


def font(size, bold=False, mono=False):
    key = (size, bold, mono)
    if key not in _fonts:
        name = 'DejaVuSansMono' if mono else 'DejaVuSans'
        if bold:
            name += '-Bold'
        _fonts[key] = ImageFont.truetype(os.path.join(FONT_DIR, name + '.ttf'), size)
    return _fonts[key]


def mix(c1, c2, a):
    a = max(0.0, min(1.0, a))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))


def enabled_connections(genome):
    return {k: c for k, c in genome.connections.items() if c.enabled}


def layout(genome, config, box):
    """{node key: (x, y)} inside box = (x0, y0, x1, y1)."""
    gc = config.genome_config
    inputs, outputs = list(gc.input_keys), list(gc.output_keys)
    hidden = [n for n in genome.nodes if n not in outputs]
    conns = enabled_connections(genome)

    depth = {k: 0 for k in inputs}
    for _ in range(len(hidden) + 1):                 # longest path from the inputs
        for h in hidden:
            src = [depth.get(a, 0) for (a, b) in conns if b == h]
            depth[h] = 1 + max(src) if src else 1
    cols = max([depth[h] for h in hidden], default=0) + 1
    for o in outputs:
        depth[o] = cols

    x0, y0, x1, y1 = box
    pos = {}
    for i, k in enumerate(inputs):
        pos[k] = (x0, _spread(i, len(inputs), y0, y1))
    for i, k in enumerate(outputs):
        pos[k] = (x1, _spread(i, len(outputs), y0 + (y1 - y0) * 0.12, y1 - (y1 - y0) * 0.12))
    for c in range(1, cols):
        col = [h for h in hidden if depth[h] == c]
        # order a column by the average height of what feeds each node (fewer crossings)
        def bary(h):
            ys = [pos[a][1] for (a, b) in conns if b == h and a in pos]
            return sum(ys) / len(ys) if ys else (y0 + y1) / 2
        col.sort(key=bary)
        x = x0 + (x1 - x0) * c / cols
        for i, h in enumerate(col):
            pos[h] = (x, _spread(i, len(col), y0 + (y1 - y0) * 0.1, y1 - (y1 - y0) * 0.1))
    return pos


def _spread(i, n, lo, hi):
    return (lo + hi) / 2 if n == 1 else lo + (hi - lo) * i / (n - 1)


def draw_network(img, genome, config, box, values=None, action=None, labels=True, r=11,
                 wmax=3.0, a=1.0, highlight=(), pos=None, bg=BG, hidden_names=None):
    """Draw `genome` into `img` (PIL Image).

    values: {node: value} from a live forward pass; lights up active nodes and connections.
    action: index of the chosen action (outlined). highlight: connection keys / node keys to ring.
    hidden_names: {node: label} written above hidden nodes.
    """
    d = ImageDraw.Draw(img)
    pos = pos or layout(genome, config, box)
    conns = enabled_connections(genome)
    order = sorted(conns.items(), key=lambda kv: abs(kv[1].weight))
    for (src, dst), c in order:
        if src not in pos or dst not in pos:
            continue
        w = c.weight
        strength = min(1.0, abs(w) / wmax)
        col = AMBER if w > 0 else BLUE
        if values is not None:
            flow = min(1.0, abs(values.get(src, 0.0) * w) / wmax)
            colr = mix(PANEL, col, 0.25 + 0.75 * flow)
        else:
            colr = mix(PANEL, col, 0.35 + 0.65 * strength)
        if (src, dst) in highlight:
            colr = FG
        width = max(1, int(1 + 4 * strength))
        (sx, sy), (tx, ty) = pos[src], pos[dst]
        if src == dst:                               # a loop to itself (memory): a ring above the node
            lr = r + 5
            d.ellipse([sx - lr * 0.8, sy - r - 2 * lr, sx + lr * 0.8, sy - r + 2], outline=mix(bg, colr, a),
                      width=width)
        elif tx <= sx:                               # a connection running backwards (recurrent): a curve
            bow = 40 + 0.15 * abs(sx - tx)
            mx, my = (sx + tx) / 2, max(sy, ty) + bow
            pts = [((1 - u) ** 2 * sx + 2 * (1 - u) * u * mx + u * u * tx,
                    (1 - u) ** 2 * sy + 2 * (1 - u) * u * my + u * u * ty) for u in [i / 24 for i in range(25)]]
            d.line(pts, fill=mix(bg, colr, a), width=width, joint='curve')
        else:
            d.line([pos[src], pos[dst]], fill=mix(bg, colr, a), width=width)

    gc = config.genome_config
    for k, (x, y) in pos.items():
        is_out = k in gc.output_keys
        fill = PANEL
        if values is not None:
            v = values.get(k, 0.0)
            fill = mix(PANEL, AMBER if v > 0 else BLUE, min(1.0, abs(v)))
        rr = r + 2 if is_out else r
        outline = FG if (action is not None and is_out and k == action) or k in highlight else MUTED
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=mix(bg, fill, a), outline=mix(bg, outline, a),
                  width=3 if outline is FG else 1)
        if labels:
            if k < 0:
                d.text((x - rr - 8, y), INPUT_SHORT[-k - 1], font=font(15), fill=mix(bg, FG, a),
                       anchor='rm')
            elif hidden_names and k in hidden_names:
                d.text((x, y - rr - 4), hidden_names[k], font=font(14), fill=mix(bg, MUTED, a), anchor='mb')
            elif is_out:
                d.text((x + rr + 8, y), ACTION_SHORT[k], font=font(15, bold=k == action),
                       fill=mix(bg, FG if k == action else MUTED, a), anchor='lm')
    return pos


def network_image(genome, config, size=(900, 560), title=None, subtitle=None):
    """A standalone picture of a network (for networks.py)."""
    img = Image.new('RGB', size, BG)
    w, h = size
    top = 70 if title else 30
    draw_network(img, genome, config, (110, top + 10, w - 90, h - 30))
    d = ImageDraw.Draw(img)
    if title:
        d.text((24, 18), title, font=font(24, bold=True), fill=FG)
    if subtitle:
        d.text((24, 48), subtitle, font=font(15), fill=MUTED)
    return img
