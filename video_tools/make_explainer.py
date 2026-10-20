"""
Explainer video generator

Renders the narrated explainer (docs/explainer.mp4 + .vtt captions) and the still images used by
the web page. Narration comes from voice/explainer/ (voice-text pipeline, see README.md); without
it the video renders silent with default timings, which is handy for previews.

Usage (from the repository root, after training):
    python video_tools/make_explainer.py runs/<run>
    python video_tools/make_explainer.py runs/<run> --stills-only
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import timeline as T  # noqa: E402
import videos_def as V  # noqa: E402

# still name -> (scene name, beat, fraction of the beat)
STILLS = {'poster': ('title', 0, 0.5), 'network': ('network', 1, 0.6), 'genome': ('genome', 1, 0.95),
          'mutation': ('mutation', 3, 0.95), 'crossover': ('crossover', 2, 0.95),
          'speciation': ('speciation', 2, 0.95), 'loop': ('loop', 0, 0.9),
          'training': ('gen_fitness', 2, 0.97), 'species': ('species', 1, 0.95),
          'ghosts': ('gen_ghosts', 3, 0.97), 'growth': ('growth', 1, 0.99),
          'evolution': ('evolution', 1, 0.97), 'read': ('read', 1, 0.6), 'ablation': ('ablation', 1, 0.95),
          'harder': ('harder', 1, 0.97), 'wind': ('task', 3, 0.9), 'bonus': ('reward', 3, 0.95)}


def main():
    ap = argparse.ArgumentParser(description='Render the narrated explainer video.')
    ap.add_argument('run_dir')
    ap.add_argument('--voice', default=os.path.join(C.REPO_ROOT, 'voice', 'explainer'))
    ap.add_argument('--output', default=os.path.join(C.REPO_ROOT, 'docs', 'explainer.mp4'))
    ap.add_argument('--stills', default=os.path.join(C.REPO_ROOT, 'docs', 'img'))
    ap.add_argument('--crf', type=int, default=28)
    ap.add_argument('--stills-only', action='store_true')
    ap.add_argument('--only', nargs='*', help='stills to make (default: all)')
    args = ap.parse_args()

    ctx = C.Context(args.run_dir, os.path.join(C.REPO_ROOT, 'video_tools', 'cache'))
    print(f'Winner: {ctx.stats}')
    scenes = V.explainer(ctx)
    by_name = {s.name: s for s in scenes}
    os.makedirs(args.stills, exist_ok=True)
    for still, (scene, beat, frac) in STILLS.items():
        if args.only and still not in args.only:
            continue
        T.still(by_name[scene], beat, frac, args.voice).save(os.path.join(args.stills, f'{still}.jpg'),
                                                             quality=86)
    run = ctx.run
    h, c = run.size(ctx.winner)
    with open(os.path.join(args.stills, 'stats.json'), 'w') as fh:
        json.dump({**ctx.stats, 'generations': run.G, 'solved_at': run.solved_at,
                   'winner_generation': run.winner_gen, 'hidden': h, 'connections': c,
                   'pop_size': ctx.config.pop_size, 'episodes': ctx.episodes,
                   'species_total': len({r['species'] for r in run.species})}, fh, indent=2)
    if args.stills_only:
        return
    T.render(scenes, args.output, args.voice, crf=args.crf)


if __name__ == '__main__':
    main()
