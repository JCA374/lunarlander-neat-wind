# NEAT Lunar Lander

A small, readable example of **NEAT** (NeuroEvolution of Augmenting Topologies, Stanley &
Miikkulainen 2002) learning to land Gymnasium's **LunarLander-v2**. NEAT evolves both the
weights *and the shape* of a neural network. Every network starts as the simplest possible
one, with the 8 inputs wired straight to the 4 actions. Evolution adds hidden nodes and
connections only when they help.

**Explainer video, lessons and pictures:** https://jca374.github.io/lunarlander-neat-wind/

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install swig && pip install -r requirements.txt

python train.py                      # ~1 h on 8 cores; writes runs/run_YYYYMMDD_HHMMSS/
python demo.py runs/<run>            # watch the winner fly
python visualize_all.py runs/<run>   # plots, network pictures, evaluation
```

## How it works

| Step | What happens | Where |
|---|---|---|
| Genome → network | A genome is a list of node genes and connection genes. neat-python turns it into a feed-forward network. | `neat_lander/env.py` |
| Acting | The network gets the lander's 8 observations and outputs 4 scores. The highest score is the action. | `choose_action` |
| Fitness | The average reward over 5 episodes. Every genome in a generation flies the **same** 5 starts, and the starts change each generation. | `neat_lander/fitness.py` |
| Validation | Each generation's champion also flies **100 episodes no genome trained on**. "Solved" means an average of at least 200 there, not on a lucky training score. | `EvolutionRecorder` |
| Evolution | neat-python speciates, selects within species, does crossover and mutation (add node, add connection, change weights). | `neat-config.ini` |

The config is classic NEAT. It starts with `num_hidden = 0` and `initial_connection = full_direct`, uses one fixed `tanh` activation, a population of 150 and speciation with `compatibility_threshold = 2.2`. Each setting is commented in [`neat-config.ini`](neat-config.ini).

## Scripts

| Script | Purpose |
|---|---|
| `train.py` | Train. Options: `--generations 500 --episodes 5 --workers 8`; `--resume runs/<run>/checkpoints/gen-N` continues an interrupted run. |
| `evaluate.py` | Test a genome on 100 unseen episodes: landing / crash / timeout rates and average reward. |
| `demo.py` | Watch a genome fly. `--generation N` picks the champion of generation N. |
| `plots.py` | `plots/training.png` (fitness, species, network size, landing) and `plots/species.png`. |
| `networks.py` | Draws the champion network every 10 generations and makes `networks/evolution.gif`. |
| `visualize_all.py` | Runs plots, networks and evaluate. |
| `experiments/` | Failure analysis: `record_flights.py` flies 1,000 analysis episodes, then `analyze_flights.py` writes `results/failure_analysis/<run>/report.html`, covering what failure looks like, where it happens, what drifts before it and what predicts it. |
| `video_tools/` | Renders the explainer and lesson videos and the web page images (see below). |

## What a run saves

Everything is written while training runs, so an interrupted run is still usable.

```
runs/<run>/
  config.ini                 the NEAT config, copied when the run starts
  data/fitness.csv           best / mean / worst fitness and species count per generation
  data/complexity.csv        hidden nodes and connections (population and champion)
  data/species.csv           size, age, best fitness and stagnation of each species per generation
  data/validation.csv        each generation's champion on 100 unseen episodes
  data/population.csv.gz     every genome of every generation: fitness, species, size, parents
  data/innovations.json      the generation each connection and node first appeared
  genomes/champions.pkl      the best genome of every generation
  genomes/winner.pkl/.json   the champion with the best validation score
  populations/pop_gNNNN.pkl.gz  whole populations (generations 0-10, then every 10th)
  checkpoints/gen-N          for --resume
```

## Videos and web page

`docs/` is the GitHub Pages site: an explainer video, 7 lessons and the page images. All of it is rendered from a training run with Pillow and ffmpeg:

```bash
python video_tools/make_explainer.py runs/<run>       # docs/explainer.mp4 + docs/img/*
python video_tools/make_lessons.py runs/<run>         # docs/lessons/*
```

The narration scripts are in `video_tools/narration/`, with one `## <scene>[.n]` section per beat. They are voiced locally with Kokoro TTS and checked with Whisper using the `voice-text` pipeline:

```bash
V=~/.claude/skills/voice-text/scripts/voice.sh
$V new video_tools/narration/explainer.md --kind voiceover --slug explainer
$V prep voice/explainer && $V narrate voice/explainer && $V listen voice/explainer && $V export voice/explainer
```

Without narration, the videos render silent with default timings.

## Requirements

Python 3.12, neat-python 0.92, gymnasium 0.29.1 with Box2D (needs `swig`), numpy, matplotlib and Pillow. Rendering the videos also needs ffmpeg and the DejaVu fonts.
