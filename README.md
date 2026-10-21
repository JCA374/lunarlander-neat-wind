# NEAT Lunar Lander in wind

A small, readable example of **NEAT** (NeuroEvolution of Augmenting Topologies, Stanley &
Miikkulainen 2002) learning to land Gymnasium's **LunarLander-v2 in wind**. NEAT evolves both the
weights *and the shape* of a neural network. Every network starts as the simplest possible
one, with the 8 inputs wired straight to the 4 actions. Evolution adds hidden nodes and
connections only when they help.

This is the harder sibling of [lunarlander-neat](https://github.com/JCA374/lunarlander-neat). The no-wind version
was solved by a network with **no hidden nodes**. Here, two things are added to see whether the task gets hard
enough for evolution to *need* hidden structure:

- **Wind.** Gymnasium's built-in wind (`wind_power = 15`) and turbulence (`turbulence_power = 1.5`) push and
  twist the lander with slowly changing gusts. The wind is **not** one of the 8 inputs, so the network can only
  notice it as drift in its own velocity.
- **A centre bonus.** A landing earns up to **+50 fitness**: the full bonus exactly in the middle between the
  flags, falling to 0 at a flag. "Solved" is still judged on the plain game score (an average of 200 or more).

**Explainer video and pictures:** https://jca374.github.io/lunarlander-neat-wind/

## Results: three runs, none solved, no hidden layer needed

| Run | Config | What changed | Winner on 1,000 analysis flights |
|---|---|---|---|
| A `runs/run_wind_20261020` | `neat-config.ini` | classic NEAT | 69% landed, score 160, 1 hidden node |
| B `runs/run_wind_b` | `neat-config-b.ini` + `--reeval-top 10` | top 10 re-flown on 30 extra episodes; fewer structural mutations; `max_stagnation` 40 | 71% landed, score 169, 2 hidden nodes |
| C `runs/run_wind_c` | `neat-config-c.ini` + `--reeval-top 10` | B + `feed_forward = False` (loops, memory) | **80% landed, score 171**, 0 hidden nodes, 0 loops |

None reached an average game score of 200. The video and page use run C's winner. What we learned
(`experiments/analyze_flights.py`, `experiments/compare_runs.py`, and the page's "Why it fails" and "Memory" sections):

- **A steady push from one side is what kills the lander.** Gusts are survivable; a flight with a strong steady
  push failed every time in all three runs. A network that only reacts to the current state sees "hovering in calm
  air" and "hovering against a steady push" as the same input, but they need different actions.
- **Five windy episodes make selection lucky.** In run A the top genome by 5 episodes was much worse on 100, and
  better genomes went unnoticed. Re-flying the top 10 (runs B and C) cut the luck gap from about 90 to about 25
  points and doubled the typical champion's score, but didn't raise the best one.
- **Evolution dropped the memory.** Run C started every genome with 4 loops; by generation 200 only 1% had any,
  and the winner has none. Its edge over B (+9 landings in 100, CI +6 to +13) is not from remembering; with one run
  per setup it may be luck between runs.
- **More generations don't help.** All three runs plateaued after generation 100-200.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install swig && pip install -r requirements.txt

python train.py --generations 500    # ~1.5 h on 8 cores; writes runs/run_YYYYMMDD_HHMMSS/
python demo.py runs/<run>            # watch the winner fly
python visualize_all.py runs/<run>   # plots, network pictures, evaluation
```

## How it works

| Step | What happens | Where |
|---|---|---|
| Genome → network | A genome is a list of node genes and connection genes. neat-python turns it into a feed-forward network. | `neat_lander/env.py` |
| Acting | The network gets the lander's 8 observations and outputs 4 scores. The highest score is the action. | `choose_action` |
| Wind | Gymnasium's gust phase is drawn from the global numpy RNG, not from the reset seed. `seed_wind` ties it to the episode seed, so one seed always means the same start *and* the same wind. | `neat_lander/env.py` |
| Fitness | The average of reward + centre bonus over 5 episodes. Every genome in a generation flies the **same** 5 starts (and winds), and they change each generation. | `neat_lander/fitness.py` |
| Validation | Each generation's champion also flies **100 episodes no genome trained on**. "Solved" means an average game score of at least 200 there, not on a lucky training score. The run's winner is the champion with the best validation fitness (score + bonus). | `EvolutionRecorder` |
| Evolution | neat-python speciates, selects within species, does crossover and mutation (add node, add connection, change weights). | `neat-config.ini` |

The config is classic NEAT. It starts with `num_hidden = 0` and `initial_connection = full_direct`, uses one fixed `tanh` activation, a population of 150 and speciation with `compatibility_threshold = 2.2`. Each setting is commented in [`neat-config.ini`](neat-config.ini).

## Scripts

| Script | Purpose |
|---|---|
| `train.py` | Train. Options: `--generations 500 --episodes 5 --workers 8`; `--resume runs/<run>/checkpoints/gen-N` continues an interrupted run. |
| `evaluate.py` | Test a genome on 100 unseen windy episodes: landing / crash / timeout rates, centred landings and average reward. |
| `demo.py` | Watch a genome fly. `--generation N` picks the champion of generation N. |
| `plots.py` | `plots/training.png` (fitness, species, network size, landing) and `plots/species.png`. |
| `networks.py` | Draws the champion network every 10 generations and makes `networks/evolution.gif`. |
| `visualize_all.py` | Runs plots, networks and evaluate. |
| `experiments/` | Failure analysis: `record_flights.py` flies 1,000 analysis episodes, then `analyze_flights.py` writes `results/failure_analysis/<run>/report.html`, covering what failure looks like, where it happens, what drifts before it and what predicts it. |
| `video_tools/` | Renders the explainer video and the web page images (see below). |

## What a run saves

Everything is written while training runs, so an interrupted run is still usable.

```
runs/<run>/
  config.ini                 the NEAT config, copied when the run starts
  data/fitness.csv           best / mean / worst fitness and species count per generation
  data/complexity.csv        hidden nodes and connections (population and champion)
  data/species.csv           size, age, best fitness and stagnation of each species per generation
  data/validation.csv        each generation's champion on 100 unseen episodes (fitness, game score, landed, centred)
  data/population.csv.gz     every genome of every generation: fitness, species, size, parents
  data/innovations.json      the generation each connection and node first appeared
  genomes/champions.pkl      the best genome of every generation
  genomes/winner.pkl/.json   the champion with the best validation score
  populations/pop_gNNNN.pkl.gz  whole populations (generations 0-10, then every 10th)
  checkpoints/gen-N          for --resume
```

## Videos and web page

`docs/` is the GitHub Pages site: an explainer video and the page images. All of it is rendered from a training run with Pillow and ffmpeg:

```bash
python video_tools/make_explainer.py runs/<run>       # docs/explainer.mp4 + docs/img/*
```

The narration scripts are in `video_tools/narration/`, with one `## <scene>[.n]` section per beat. They are voiced locally with Kokoro TTS and checked with Whisper using the `voice-text` pipeline:

```bash
V=~/.claude/skills/voice-text/scripts/voice.sh
$V new video_tools/narration/explainer.md --kind voiceover --slug explainer
$V prep voice/explainer && $V narrate voice/explainer && $V listen voice/explainer && $V export voice/explainer
```

Without narration, the video renders silent with default timings.

## Requirements

Python 3.12, neat-python 0.92, gymnasium 0.29.1 with Box2D (needs `swig`), numpy, matplotlib and Pillow. Rendering the videos also needs ffmpeg and the DejaVu fonts.
