# Bio-Analysis

Two-photon calcium imaging of mouse motor cortex during a cued water-reaching task, analysed for one question:

> **Which cells lock onto the contralateral reach, can they predict it without data leakage, and are the cells of a
> successful reach the same cells as those of a missed reach, an in-vain reach, or the cue?**

The repository answers it with three building blocks, all leak-free and all cached:

| block | what it does | package |
|---|---|---|
| **clarity** | a model-free score of how clearly a cell locks onto an event type, calibrated against the recording's own null (circular label shifts) | `src/clarity` |
| **prediction** | a leak-free ExtraTrees predictor of the behavior from a causal window of the cell signals, scored on contiguous time blocks | `src/prediction` |
| **contrast** | pairwise comparisons of event types (success vs missed / in vain / cue / walk): do the same cells lock onto both, and do they respond the same way | `src/clarity` |

Everything older (`sps_select`, `rfecv_predict`, shuffled-frame splits) was removed: it leaked through the temporal
autocorrelation of the interpolated signals and scored random walks as well as real cells. See
[doc/context_human.md](doc/context_human.md) for the story and [doc/context_claude.md](doc/context_claude.md) for the
exhaustive technical record.

## Headline results

All numbers are on 45 recordings (29 initialization + 16 memory), behavior `cont` = contralateral successful reach,
causal 5 s window, 5 contiguous blocks with 5 s purge, within-block AUC.

1. **Clarity screens predictability.** The fraction of cells with a significant clarity (z >= 2) predicts the
   leak-free AUC of the recording: Spearman 0.79 (init 0.78, memory hold-out 0.76, p < 1e-9). Random-walk control
   AUC 0.52.
2. **Clear cells carry the information.** ExtraTrees on the clearest 30 % of the cells (ranked on training blocks
   only) reaches 0.70 vs 0.60 on the least-clear 30 % (37/45 wins, p < 1e-6) and 0.68 on a size-matched random set;
   the clearest 30 % keep 98 % of the all-cell AUC (0.715).
3. **In the clear recordings, success, missed and in-vain reaches use the same cells; success and missed respond
   the same way, in vain the opposite way; the cue cells are a separate small population.** Overlap histograms by
   clarity bucket and response-profile panels in `plot/clarity/contrast/cont/`, details in the docs.

## Layout

```
src/
  util/         ABlockLogger, check_folders, fetch_tokens, load_paths, parallel_map
  preprocess/   TARGETS (every binary target), Preprocessor (frame grid), BlockPreprocessor (time-ordered + folds + noise control),
                EventPreprocessor (native-rate event windows)
  clarity/      ClarityEngine (per-cell clarity, cache), ContrastEngine (pairwise targets, cache), ClarityVisualizer (maps, overlap, profiles)
  prediction/   ExtraTreesEngine (leak-free predictor, cache), PredictionSummarizer (rank / subset plots)
  deduction/    Locator / PenaltyComputer / Visualizer (spike-window expansion, unchanged logic)
script/         command-line entry points (argparse)
test/           determinism and cache-reproduction tests
cache/          every engine output: clarity/{init,memo}/{beh}, extratrees/{init,memo}/{beh}[/subset_30], contrast/{init,memo}/{first}, deduce/{beh}
plot/           every figure, svg only: clarity/{map,auc,subset,contrast}/..., deduce/
doc/            context_human.md (walk-through), context_claude.md (full record)
config/         .env.path (ROOT_PATH, INIT_DATA_PATH, MEMO_DATA_PATH, PLOT_PATH, CACHE_PATH)
data/           initialization/ and memory/ with {token}_B.csv (behavior) and {token}_S.csv (signals)
```

Every engine follows the same contract: `compute(token, ...)` for one recording, `collect(...)` for a folder of
recordings (parallel over tokens with `n_workers`), a `_C.json` next to every cached table holding the configuration,
and `rerun=True` to recompute. A cached result is reused only when its configuration matches exactly.

## Running

Use the `regular` conda environment and launch the scripts from `script/` with the repository root on the path:

```bash
cd script
export PYTHONPATH=..        # PowerShell: $env:PYTHONPATH = ".."
python clarity_map.py       --beh cont --view-width -80 80 --n-workers 6      # per-cell event heatmaps / curves, significant cells first
python clarity_auc.py       --beh cont --noise --n-workers 6                  # tokens ranked by ExtraTrees AUC vs clarity fraction (+ random-walk control)
python clarity_subset.py    --beh cont --fraction 0.3 --n-workers 6           # clearest / random / least-clear / all cells
python clarity_contrast.py  --first cont --seconds cont_missed cont_inv cont_inv_near cont_inv_isolated cue_none walk --clear-fraction 0.4
python deduce_3d.py                                                            # spike-window expansion penalties (legacy deduction)
python deduce_viz.py        --token K215_A_D1 --beh cont
```

Every script prints `--help`. The targets accepted by `--beh` / `--first` / `--seconds` are the keys of
`src.preprocess.TARGETS`:

| key | events |
|---|---|
| `cont` | contralateral successful reach (`cont reach`, `cont reaching`) |
| `cont_missed` | contralateral missed reach |
| `cont_inv` | contralateral in-vain reach; `cont_inv_near` within 3 s after a success ended, `cont_inv_isolated` with no success in the 8 s before |
| `cue` | valve / spout cue; `cue_none` a cue followed by no contralateral attempt within 5 s |
| `walk`, `both`, `both_inv`, `cont&inv`, `both&inv` | the historical targets |

Tests:

```bash
cd test
PYTHONPATH=.. python cache_test.py          # the engines reproduce the committed caches
PYTHONPATH=.. python preprocessor_test.py   # determinism of the preprocessors, consistency of the targets
```

## Outputs

| script | cache | plot |
|---|---|---|
| `clarity_map.py` | `cache/clarity/{shard}/{beh}/{token}-{beh}_Z.csv` | `plot/clarity/map/{beh}/{token}-{beh}_{heatmap,curve}.svg` |
| `clarity_auc.py` | `cache/extratrees/{shard}/{beh}/{token}-{beh}[_noise]_R.csv` | `plot/clarity/auc/{beh}/clarity_auc-{beh}[_noise].svg` |
| `clarity_subset.py` | `cache/extratrees/{shard}/{beh}/subset_30/...` | `plot/clarity/subset/{beh}/clarity_subset-{beh}.svg` |
| `clarity_contrast.py` | `cache/contrast/{shard}/{first}/{token}-{first}-{second}_{R,T}.csv` | `plot/clarity/contrast/{first}/clarity_{overlap,profile}-{first}.svg` |
| `deduce_3d.py`, `deduce_viz.py` | `cache/deduce/{beh}/{token}_{N,B,L,R}.json` (not tracked, 5.8 GB) | `plot/deduce/...` |

Reading a map: one panel per cell, the significant cells first in warm colours (sorted by z), the others after in cool
colours; a heatmap stacks the events as rows over the frames around the onset (0.1 s each), a curve averages them;
the grey mask marks the frames outside the clarity window, which the score never saw.

## Protocol in one paragraph

Signals are deconvolved spikes at ~1 Hz, interpolated 10x onto the 0.1 s behavior grid (Akima, which reads the next
native sample, so a "causal" window carries up to 1 s of the future). Neighbouring frames are therefore near copies,
and any shuffled split leaks: the predictor is scored only on contiguous time blocks with a 50-frame purge, the
normalization and the clarity ranking are fitted on the training frames of each fold, the within-block AUC is averaged
over the blocks that hold both classes (the pooled AUC collapses when events cluster in time), and a random-walk control
replaces the cells by random walks of the same shape while keeping the labels. The clarity of a cell is the mean
pairwise Pearson correlation between its +-5 s event windows, z-scored against 1000 circular shifts of the labels of
the same recording; a cell with z >= 2 is significant; the recording descriptor is the fraction of significant cells.
