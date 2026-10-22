# Continue here (paused 2026-10-22)

Two curriculum runs were paused for a reboot. Checkpoints are saved every 10 generations in
`runs/<run>/checkpoints/`; resuming truncates the logs back to the checkpoint, so nothing is duplicated.

| Run | What | Paused at | Latest checkpoint |
|---|---|---|---|
| G `runs/run_wind_g` | run D's setup (`neat-config-d.ini`) + curriculum | gen 578 / 1000 | `gen-569` |
| H `runs/run_wind_h` | run F's leaky memory (`neat-config-f.ini`) + curriculum | gen 617 / 1000 | `gen-609` |

Curriculum: steady one-sided push until gen 300, blending to the real gusty wind by gen 600, real task after.
Both are now past 600, so they train on the real task.

## 1. Resume both runs (4 cores each, in parallel)

The flags must be repeated on resume: `--reeval-top`, `--reeval-episodes` and `--curriculum` are not stored in
the checkpoint.

```bash
cd ~/code/games/lunarlander/lunarlander-neat-wind
source .venv/bin/activate
mkdir -p logs
for r in g h; do
  ck=$(ls -d runs/run_wind_$r/checkpoints/gen-* | sort -t- -k2 -n | tail -1)
  nohup python train.py --resume "$ck" --generations 1000 --workers 4 \
        --reeval-top 10 --reeval-episodes 30 --curriculum 300,600 > logs/run_$r.log 2>&1 &
done
tail -n 2 logs/run_g.log logs/run_h.log    # one "gen N | ..." line per generation
```

Pause and continue without losing anything (as long as the computer stays on):

```bash
pkill -STOP -f "train.py --resume"     # pause
pkill -CONT -f "train.py --resume"     # continue
```

## 2. When both reach generation 1000

1. Compare with run D (the current best: 191 [185, 197], 85% landed) on the same 1,000 flights, plus the
   memory switch-off test:
   `python experiments/compare_runs.py D=runs/run_wind_d G=runs/run_wind_g H=runs/run_wind_h`
   Then move `results/comparison/summary.json` to `results/comparison_gh/` and restore the A-D file
   (`git checkout results/comparison/summary.json`), since the page charts read the A-D comparison.
2. Look at the strong-steady-wind failure rate (`by_wind` in that summary). It was 100% in every run so far;
   it is what the curriculum is meant to fix.
3. If G or H is still improving at gen 1000, keep training (resume with a higher `--generations`).
4. If one beats D: re-render the video (`video_tools/make_explainer.py runs/<run>`, narration in
   `video_tools/narration/explainer.md`, voice with the voice-text pipeline) and update `docs/index.html` and the
   README. If not: add a short result bullet to the page's memory section and a row to the README results table.
5. Whitelist the runs in `.gitignore` (`!runs/run_wind_g/`, `!runs/run_wind_h/`) and commit.

## Where things stand

Runs A-F are summarised in README.md ("Results") and on https://jca374.github.io/lunarlander-neat-wind/
(sections 9 and 10). Best so far: run D. No run has reached an average game score of 200; strong steady
wind beats every run; memory either goes unused (C, D, E) or is used as a smoother (F).
