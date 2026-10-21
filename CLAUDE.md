# CLAUDE.md

A pedagogical NEAT example: neat-python evolves networks that land Gymnasium's LunarLander-v2 **in wind**,
with a fitness bonus for landing in the middle between the flags. Forked from ../lunarlander-neat (no wind).
Clarity and visualisation come before raw performance. README.md is the user-facing guide.

## Layout

- `neat_lander/`: the shared package.
  - `env.py`: `run_episode`, which records frames, paths, wind and per-node values; the wind settings and `centre_bonus`.
  - `fitness.py`: the `Evaluator` (a multiprocessing pool that is also a reporter) and the training/validation seeds.
  - `recorder.py`: `EvolutionRecorder`, which writes every log each generation.
  - `runs.py`: run-dir layout and loaders.
  - `draw.py`: the Pillow network renderer, shared by `networks.py` and the video.
  - `labels.py`: input and action names.
- CLI scripts: `train.py`, `evaluate.py`, `demo.py`, `plots.py`, `networks.py`, `visualize_all.py`. Each takes a run dir and defaults to the latest run.
- `video_tools/`:
  - `timeline.py` is ported unchanged from ../Moonlander-Genetic.
  - `common.py` has the helpers and `Context`, which caches results in `video_tools/cache`, keyed by the winner.
  - `scenes.py` holds one factory per scene.
  - `videos_def.py` lists the scenes in each video.
  - `make_explainer.py` renders into `docs/`. There are no lessons in this repo.
- `video_tools/narration/*.md`: one `## <scene>[.n]` section per beat. The number of beats in a scene must match its `Scene(...)` beat list, or `timeline.plan` raises an error.
- `voice/<slug>/`: voice-text work dirs. Only text is tracked; audio is ignored.
- `docs/`: the GitHub Pages site, served from main.

## Invariants

- Environment: gymnasium 0.29.1 with `LunarLander-v2`, `enable_wind=True`, `WIND_POWER`/`TURBULENCE_POWER` in `env.py`.
- Wind is seeded: gymnasium draws the gust phase from the global numpy RNG, so `seed_wind` sets `wind_idx`/`torque_idx`
  from the episode seed, *before* `reset` (reset already takes one step in the wind).
- Fitness = env reward + centre bonus (up to 50, landing only). `reward` columns are the plain env score; "solved"
  (>= 200) is judged on it. The winner is chosen on validation `fitness`.
- `experiments/baseline/` holds the no-wind champion, used by the video's "harder" scene.
- Three published runs: A `run_wind_20261020` (classic), B `run_wind_b`, C `run_wind_c` (recurrent). The video and page
  use C. `make_net` builds a `RecurrentNetwork` when `feed_forward = False`; `run_episode` resets it every flight.
- `docs/index.html` charts are inline SVG written by `experiments/page_charts.py` between `<!-- chart:NAME -->` markers.
- Seeds: training seeds are below 1e9. The 100 validation seeds start at 1e9, evaluate.py and the video use 2e9+, and the failure analysis (`experiments/`) uses 3e9+. Keep the four sets disjoint.
- Every log is written during training, nothing only at the end. `--resume` truncates the logs back to the checkpoint generation.
- neat-python 0.92 quirks handled in `train.py`:
  - After a restore, genome and node indexers restart. They are re-seeded from the population.
  - Checkpoints pickle the reporters through the species set. `__getstate__` on `Evaluator` and `EvolutionRecorder` keeps the pool out.
- neat-python has no innovation numbers. A connection gene's key `(in, out)` plays that role. The video explains this.
- `runs/` is gitignored except the published run, which is whitelisted in `.gitignore`.
