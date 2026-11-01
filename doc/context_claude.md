# Bio-Analysis: full context for an assistant with no prior context

This document is the complete technical record of the project as of 2026-11-01: the question, the data and its
pitfalls, every experiment that was run (including the ones whose code is not in this repository any more), the
numbers, the interpretation, the open points, and the conventions of the code base. Read it top to bottom before
touching the repository. The human-facing walk-through is `doc/context_human.md`; the repository overview is
`README.md`.

---

## 0. One-paragraph summary

Mouse motor-cortex cells (two-photon calcium, deconvolved spikes at ~1 Hz) were recorded during a cued
water-reaching task. The owner (Tianyi Ma) wants to know which cells lock onto the contralateral reach, whether they
predict it, and whether the cells of a successful reach differ from those of a missed reach, an in-vain reach, or the
cue. Earlier code in this repository scored ~0.9 balanced accuracy but leaked (shuffled frame split on 10x
interpolated signals; random walks scored the same). Under a leak-free block protocol the honest ExtraTrees AUC is
0.715 on 45 recordings. A model-free **clarity** score (event-to-event coherence of a cell's event windows, z-scored
against circular label shifts of the same recording) predicts a recording's AUC with Spearman 0.79 and ranks cells
such that the clearest 30 % beat the least-clear 30 % by 0.10 AUC. Contrasting event types on the same recordings:
success and missed reaches engage the same cells with statistically indistinguishable responses; in-vain reaches
engage largely the same cells but with the opposite response shape; cues without a reach engage a separate small
population. Consuming water is therefore not what the reach cells encode; they encode the cued first reach, and the
reversal in vain is not explained by the cue stimulus nor by the acute aftermath of the previous reach.

---

## 1. The scientific question

Collaborator's framing (translated): "Can a successful water grab be predicted accurately, is a group of cells driving
it, can in-vain and missed grabs be predicted, and if so are the cells selected for success the same as those for
in vain / missed? Are the 'success' cells about extending the paw or about actually drinking?"

Decomposition used here:

1. **Prediction**: can the reach be predicted from the cells without leakage, and how well per recording.
2. **Which cells**: a defensible, training-only criterion for "this cell carries the reach".
3. **Same cells?**: success vs missed (same phase of the trial, no water) vs in vain (later in the trial, water gone)
   vs cue without a reach vs walk.

---

## 2. Data

- Folders `data/initialization` (30 tokens, days D1/D2/D4/D5/D7/D8) and `data/memory` (16 tokens, M1/M2/M3/M4/M5).
  Token pattern `K\d{3}_[AB]_[DM]\d` (mouse, hemisphere/field, day). `fetch_tokens(folder, shard)` lists them.
- `{token}_S.csv`: columns `y0000...` one per cell, index `x0000` = two-photon frame number. Native rate: 1 frame per
  1.001286 s (1.000045 s for K215). Values are deconvolved spikes: 57-61 % zeros, lag-1 autocorrelation ~0 at native
  rate. ~1200 rows per recording.
- `{token}_B.csv`: columns `Behavior`, `Start`, `End` in behavior-camera frames at 7.06 fps. The preprocessors add
  0.5 s (the CSVs already carry +1.5 s) and convert to seconds.
- Annotation names (normalized: lower case, single spaces, trailing "(n)" counter removed by
  `src.preprocess.normalize_name`): `cont reach` / `cont reaching` (successful contralateral reach, 1961 events),
  `cont missed` / `cont miss` (127), `cont in vain` (1547), `walk` (2422), `valve opened` (2813), `spout movement`
  (401), `spout moved after whisker detection` (50), `both reach`, `both in vain`, licking labels, rig labels
  (`noise generation`, `two photon imaging`...).
- K232_A_D1 has no successful reach; several tokens have < 10 reaches (K228_B_D2, K228_A_D1, K232_A_M1, K232_B_M2);
  those are unreliable in every per-token statistic.
- Trial structure (own measurement): a cue (`valve opened`) precedes the reach; 98 % of successful reaches start
  within 3 s of a cue (median 0.57 s); missed reaches too (median 0.42 s); in-vain reaches start median 2.97 s
  after the cue and 68 % begin within 2 s after a successful reach ended (the mouse re-reaches after drinking).
  The first behavior after 1146 of 1922 cues is a reach. Licking annotations are not time-locked to reaches (median
  84 s) and are useless as a consumption marker.

### 2.1 Preprocessing (src/preprocess)

- `Preprocessor` (frame grid): the signal is reindexed onto a 0.1 s grid and Akima-interpolated (then ffill/bfill);
  the labels are painted on the same grid from `Start` to `End` of every positive annotation; the signal is cut at
  the first behavior timestamp and resized (linspace pick or `np.interp`) to the behavior length when `resize=True`.
  Historical options (shuffle, SMOTE, normalizer, tvt split) still exist but are leaky for prediction.
  **The Akima interpolation reads the next native sample: every frame carries up to 1 s of the future.** A strictly
  causal zero-order hold costs 0.02-0.04 AUC (measured on the exploratory branch); the 5 s purge keeps it from being
  a split leak, but for an online-prediction claim quote the pad numbers.
- `BlockPreprocessor`: time-ordered wrapper (`tvt=(1,0,0)`, no shuffle, no normalizer, no SMOTE, `resize=True`);
  exposes `time`, `cells`, `x` (frames x cells, float32), `y` (0/1); `noise=True` replaces `x` by random walks of the
  same shape (the control group); `normalized(frames)` z-scores with the statistics of the given frames only;
  `block_folds(n_blocks=5, purge=50, valid_ratio=0.15)` cuts contiguous blocks with purge gaps and a validation tail;
  static helpers `runs_of`, `within_runs`, `onsets`.
- `EventPreprocessor`: native-rate signal z-scored per cell over the whole recording (label-free), an event table
  with `name`, `start`, `end`, `latency` (seconds since the previous cue), `gap` (seconds since the previous
  successful reach ended); `select(target)` and `windows(target, offsets)` -> (events, cells, offsets) by linear
  interpolation at `start + offsets`.
- `TARGETS` (src/preprocess/target.py): the registry of binary targets, each a set of positive annotation names
  plus an optional rule over the whole behavior table. Keys: `cont`, `cont_inv`, `cont&inv`, `both`, `both_inv`,
  `both&inv`, `walk` (historical), `cont_missed`, `cue`, `cue_none` (cue with no contralateral attempt within 5 s),
  `cont_inv_near` (in vain starting <= 3 s after a success ended), `cont_inv_isolated` (no success ended within the
  8 s before). `LABELS` gives the display names. Adding a target = adding one entry; every engine and script accepts
  it immediately.

---

## 3. The leakage finding (why the old code was removed)

- Old pipeline (`src_select/sps_select.py`, `src_predict/rfecv_*`, deleted on branch `feat_clarity`): shuffled
  frames, 50/50 split, SMOTE, per-cell single-feature voting across 5 models, RFECV. Scores ~0.9.
- Noise control (2026-10-28, notebook kept in the session scratchpad only): random-walk cells score the same 0.90
  balanced accuracy as real cells under the shuffled split. Reason: 10x interpolation makes neighbouring frames near
  copies (lag-1 autocorrelation ~0.99 on the grid), so a shuffled split puts near-duplicates of every test frame in
  the training set. The old `skew` control (column permutation of the training set) cannot detect this.
- Under contiguous blocks the same cells average 0.54 (10 blocks, LGB) vs 0.515 for noise.
- Conclusion adopted by the owner: never evaluate with shuffled frames; always report block-CV scores next to the
  random-walk control; the old selection/prediction code and its caches (`cache/sps_*`, `cache/rfecv_*`) are gone.

---

## 4. The leak-free protocol (src/prediction/extratrees.py)

- 5 contiguous blocks, each tested once; 50 frames (5 s) purged on both sides of the test block; a contiguous
  validation tail (15 %) purged from the training frames (kept for future tuning; not used by ExtraTrees).
- Per fold: z-score with the training frames' mean/std; features = the causal window of the past 50 frames,
  sub-sampled every 5 frames (10 positions x cells); only frames whose whole window lies inside one contiguous run
  of their subset are used (`within_runs`).
- Model: `ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features="sqrt",
  class_weight="balanced_subsample", random_state=42)`. CPU only (scikit-learn trees have no GPU path; the parallelism
  is over tokens with `n_workers` processes and `share_threads` threads per forest).
- Scores: `auc_fold` = mean within-block AUC over the blocks holding both classes (the headline), `auc` = pooled
  out-of-fold AUC (collapses when events cluster in time, e.g. K223_A_D1 pooled 0.18 vs within-block 0.70; report
  it but do not rank by it).
- Cell subsets: `all`; `top` / `bottom` = the `fraction` (30 %) clearest / least-clear cells by the clarity computed
  on the training frames of the fold (`ClarityEngine.rank`, seed + fold index); `random` = a size-matched random
  draw. The ranking never sees the test block.
- Control: `noise=True` (random walks, labels kept). Cache stem `{token}-{beh}[_noise]`, files `_R.csv` (metrics)
  and `_C.json` (configuration: token, beh, subsets, fraction, noise, n_blocks, purge, valid_ratio, window, causal,
  inner_width, n_shift, seed). A cache is reused only when the configuration dict is identical.

Honest performance (exploratory branch `feat_predict_algo`, 29 init tokens with events, causal w50, auc_fold):

| model | real | noise | notes |
|---|---|---|---|
| LGB (frame) | 0.636 | 0.524 | |
| LGB-WIN (window) | 0.676 | 0.509 | |
| CNN | 0.621 | 0.528 | |
| GRU | 0.688 | 0.514 | |
| LSTM | 0.687 | 0.561 | |
| RoPE transformer | 0.678 | 0.522 | |
| DPRIME template | 0.691 | 0.521 | exploits drift (highest noise control) |
| **ExtraTrees window** | **0.730** | **0.499** | +0.054 vs LGB-WIN, 28/29 tokens, p < 0.001 |
| ensemble (ET, LGB-WIN, GRU, DPRIME) | 0.715 | 0.531 | |

Dead ends (do not retry without new data): ROCKET (0.59), GRU size / seed ensembles, validation-tuned LightGBM,
focal loss, 3:1 negative sub-sampling, linear decoders, stacking, causal EMA rate filters, log1p, multi-scale counts,
zero-order-hold interpolation, matched-filter features, centered windows (seeing 2.4 s of the future does not beat
the causal window). No model widens the gap between clear and unclear recordings: stronger models lift the unclear
half too (ET 0.658 vs LGB-WIN 0.583 on the unclear half).

Memory set alone: ET 0.686 (noise 0.550; the label base rate drifts more there, so quote real minus noise).

---

## 5. Clarity (src/clarity/engine.py)

Definition, per recording and target:

1. `event_windows`: for every event onset (first frame of a positive run) whose whole window `inner_width`
   (default -50..+50 frames = +-5 s) lies inside the allowed frames, stack the window of every cell ->
   (cells, events, 101).
2. `coherence`: per cell, the mean pairwise Pearson correlation between its event windows (nan with < 3 events).
3. Null: circularly shift the label vector by a random offset (>= 10 % of the recording and >= the window length)
   1000 times, recompute the coherence of every cell each time; `z = (observed - mean_null) / sd_null`,
   `p = norm.sf(z)` (one-sided Gaussian; the empirical p with floor 1/1001 was too conservative for FDR).
4. A cell is **significant** when z >= 2 (threshold). The recording descriptor is `fraction` = share of significant
   cells (`describe`); `significant` returns the set.
5. Training-only version: `frames` restricts both the observed windows and the shifted labels to the training frames
   of a fold; `rank` orders cells by that z (used by the predictor's subsets).
6. Single cell: `compute_cell(token, cell, beh)` slices one column (the score of a cell does not depend on the
   others; float32 accumulation order makes it differ from the full table in the 4th decimal).
7. Cache `cache/clarity/{shard}/{beh}/{token}-{beh}_Z.csv` (columns token, cell, n_events, coherence, z, p) +
   `_C.json` (token, beh, inner_width, n_shift, seed). `collect` runs tokens in parallel and accepts `callbacks`
   (the visualizer mounts its map rendering there).

Reliability: split-half reliability of the per-token AUC is ~0.90, of the clarity fraction ~0.97, so the ceiling of
any clarity-vs-AUC correlation is ~0.93.

Alternatives that were tried and rejected: `t_contrast` / `t_peak` (t-statistics of the window vs baseline), raw
locked fraction with z >= 3 (0.73 vs 0.79), mean z over cells (0.75), population-pattern coherence (0.80-0.87 on init
but 0.43 on the memory hold-out: a post-hoc selection artefact).

---

## 6. Results on `cont` (successful contralateral reach), 45 tokens

### 6.1 Clarity screens predictability (`script/clarity_auc.py`, plot `plot/clarity/auc/cont/`)

Spearman between the clarity fraction (z >= 2, 1000 shifts, whole recording) and the ExtraTrees `auc_fold` (all
cells): init 29 tokens 0.779 (p 6e-7), memory 16 tokens 0.759 (p 7e-4; this set was a hold-out when the descriptor
was chosen), pooled 45 tokens 0.785 (p 2e-10). Against the mean of three models it was 0.84 on the exploratory
branch. Flagging tokens that reach auc_fold >= 0.75 from the clarity alone: AUC 0.93 (21 of 45 reach 0.75).
Training-block-only clarity gives the same picture (0.72 on the exploratory branch), so it is not a test-block leak.
Random-walk control: real 0.715 vs noise 0.517 mean auc_fold (the memory shard drifts).

Interpretation agreed with the owner: a screening tool, not a predictor. Yes, it is partly tautological ("if the
event pattern is visible, a model will learn it"), but it quantifies that intuition model-free, it is calibrated
against the recording's own null, it transfers to a hold-out set, and it tells which recordings are worth
modelling without training anything. The remaining AUC variance (ceiling 0.93) is distributed multi-cell signal and
block drift (e.g. K228_A_D7: 3 significant cells, auc_fold 0.87).

### 6.2 Clear cells carry the information (`script/clarity_subset.py`, plot `plot/clarity/subset/cont/`)

ExtraTrees, cells ranked per fold on training frames only, 30 % subsets (mean 12.5 of 41.5 cells):

| set | n | all | top | random | bottom | top - bottom | top - all | top - random |
|---|---|---|---|---|---|---|---|---|
| init | 29 | 0.730 | 0.712 | 0.692 | 0.614 | +0.097, 24/29, p 1e-4 | -0.019, p 0.013 | +0.019, p 0.020 |
| memory | 16 | 0.686 | 0.682 | 0.649 | 0.579 | +0.103, 13/16, p 0.006 | -0.004, p 0.53 | +0.033, p 0.039 |
| pooled | 45 | 0.715 | 0.701 | 0.677 | 0.602 | +0.099, 37/45, p 7e-7 | -0.014, p 0.023 | +0.024, 33/45, p 0.002 |

Wording agreed with the owner: "the clearest 30 % keep ~98 % of the all-cell AUC" (not "identical"); "clear cells
carry the information" holds, "only clear cells carry information" does not (random same-size sets are only 0.024
worse because the information is redundant across many cells in clear recordings). Earlier version with a
significance criterion instead of a fraction (z >= 2 clear vs same-size least-clear, LGB-WIN): +0.110, 8/9 tokens,
p 0.008; z >= 1.5: +0.083, 14/16, p 0.0013; only 9/29 tokens have >= 2 significant cells in every fold.

### 6.3 Size-adaptive cell selection (exploratory branch only; not ported)

`select_cells` with two routes, per fold on training frames, stable set = chosen in >= 3/5 folds:
- `fdr`: Benjamini-Hochberg at q 0.05 on the Gaussian p of the clarity z -> non-empty on 16/45 tokens, precision
  0.81 / recall 0.62 for flagging ET AUC >= 0.75; empty on every token with AUC < 0.55 or <= 6 events.
- `wrapper`: ET importance ranking + validation-block AUC curve over k, smallest k within 0.01 of the best, nothing
  when the best validation AUC < 0.6 -> non-empty on 27/45, precision 0.63 / recall 0.81; keeps ~15 of 44 cells at
  -0.018 AUC vs all and +0.046 vs random.
- Agreement: the clarity z ranks wrapper-chosen cells above the rest with AUC 0.77 (122/124 folds); 58 % of them are
  in the top 30 % by clarity. Jaccard of the stable sets is low (fdr keeps far fewer cells): compare by rank, not by
  set. Containment flips with pattern strength (strong tokens: wrapper inside fdr; weak tokens: the reverse).
- Logical status discussed at length with the owner: a label-driven selector A and the clarity ranking are not
  circular (A never sees the clarity), so "A picks the clear cells" is a genuine finding; FDR is a prior-free
  significance call, not a prior. The owner's visually clear cells of K227_A_D7 (y0001, y0006, y0009, y0040, y0044)
  are all selected by fdr.

### 6.4 Collaborator's repository (Juniea/Bio-Analysis, branch `window_any_dedup`, clone at `../Bio-Analysis-side`)

Read-only audit (the execution of external code was refused by the tooling; everything was re-implemented on the
raw files):
- Event-level decoder: every annotation is an event; label 1 = cont reach/reaching, 0 = every other behavior;
  window [-2, +4] s at 1 s steps on the native samples, per-session z-score; cleaning before the split: `window_any`
  (drop negatives whose window touches a reach) + `dedup` (keep the first of overlapping negative windows); 309
  hand-crafted cell statistics + 25 per-session selectivity features cross-fitted; nested CV (outer 5 folds by
  event / trial / 10 contiguous blocks, inner 5 folds for trees + threshold by macro F1); controls: circular shift
  of the signal, session-balanced scores.
- Results (28 sessions): pooled AUC 0.890 event split, 0.885 block split, circular shift 0.68 pooled but 0.50
  session-balanced; real session-balanced 0.79; the pre-event window alone (-3..-1 s) still 0.855.
- Leakage: none at the label level; block split == event split so adjacency leakage is negligible. Caveats:
  cleaning uses the labels of the test events (the evaluated distribution is the cleaned one, and it removes 68 % of
  the in-vain events, the ones adjacent to reaches); earlier "min_" features were session fingerprints (dropped).
- **Trial-phase confound**: her positives (success) are the first cued attempt (0.6 s after the cue), her negatives
  (in vain after cleaning) sit ~3 s after the cue and after a drink. A [-2, +4] s window around an in-vain reach
  therefore contains the tail of the previous success and no cue; the decoder can use "was there a cue just before"
  and "is this the aftermath of a drink" instead of the reach itself. Her 0.855 from the pre-window is this.
- Overlap with ours: her POST cells (0..+4 s selectivity) are almost all inside our z >= 2 cells (K227_A_D7 22/24,
  K227_A_M4 26/26, K227_B_M5 10/10) and our FDR cells inside her POST (16/18, 18/18); her per-session AUC correlates
  0.49 with our ET AUC and 0.59 with our clarity fraction.
- Transfers tested on the frame-level task (her LightGBM params, d'-template, linear decoders, stacking): nothing
  beyond +0.024 (DPRIME, partly drift); her 0.88 comes from the easier event-level task.

---

## 7. Success vs missed vs in vain vs cue (the "same cells?" question)

### 7.1 Design

The phase-matched comparison is success vs **missed** (both the first cued attempt, 0.6 vs 0.4 s after the cue; the
difference is only whether water was obtained). Success vs **in vain** is phase-mismatched (in vain is ~3 s after the
cue, mostly right after a drink). **Cue without a reach** (`cue_none`) isolates the cue. **Walk** is an unrelated
movement. Two confounders are tracked explicitly: the cue (present before success and missed, absent before in vain
and walk) and the aftermath of a previous reach / drink (absent before success and missed, present before near in
vain).

Three complementary measurements, all per recording:

1. **Overlap** (`ContrastEngine.overlap`): the significant cells (z >= 2) of each target; Jaccard and containment
   against the chance Jaccard of independent draws of the same sizes; Spearman of the z-scores across cells.
2. **Response-profile similarity** (`ContrastEngine.similarity`): per cell, response = mean over events of
   (window mean at 0..+1 s minus mean at -2..-1 s) on native-rate windows; Pearson r across cells between the two
   targets; the null relabels the pooled events at random (what two halves of the same kind would give); z_r is the
   distance to that null. r near the null = exchangeable; z_r <= -2 = less similar than two samples of one kind.
3. **Population time course** (`curves`): the mean z of the reference (success-significant) cells around the onset
   of each kind, sampled every 0.25 s with the Akima interpolant between the native samples (display only; the
   statistics above use the 1 s linear sampling).

### 7.2 Numbers from the session's scratch scripts (2026-10-31; same definitions, native-rate windows -2..+4 s)

- Cross-session decoders on cell-agnostic population features (GroupKFold by session) are weak for everything
  (success vs missed 0.47, vs in vain 0.63, shifted controls 0.44 / 0.56): the information is cell-specific and
  within-session.
- Profile similarity, success vs missed: r 0.42 mean (slope 0.97), z ~ +0.5 -> exchangeable. Success vs in vain:
  r 0.01, z -2.4. Success vs walk: r -0.07, z -2.4.
- Cue-aligned: the response of success-trial cells at the cue vs at cues followed by nothing: r -0.02, z -3.7
  (28/43 sessions significantly dissimilar) -> the reach cells do not respond to the cue alone. Within-session
  decoder "cue -> success follows vs nothing" AUC 0.78 (33/43 sessions >= 0.7), i.e. the cells already differ at
  the cue when a reach is about to follow (preparation / engagement).
- Clarity with other targets: missed-locked significant cells are the reach cells (65/75 inside, Spearman of z
  0.70-0.73 where missed is plentiful: K227_A_D7, K227_B_D5). Cue-only events have a small separate population
  (10 % of cells significant vs 35 % for reach; 54/147 inside the reach set, i.e. chance; rho 0.03).
- In-vain shape: in-vain-locked cells overlap the reach cells heavily in strong recordings (Jaccard 0.7-0.9) but
  with a different shape: success is suppressed around the onset (population -0.09 at -2 s, -0.17 at -1 s, ~0 after),
  isolated in vain (no reach within 8 s before) is excited +1..+3 s (+0.25 to +0.33); per-cell shape correlation
  0.03. Near in vain shows the tail of the previous success before the onset (-0.08) and the same excitation after.
- Late window +3..+5 s (consumption period): no success-vs-missed difference (z -0.2); pre-window -3..-2: none
  either (z -0.08).

### 7.3 Numbers from the repository implementation (`script/clarity_contrast.py --first cont`, 2026-11-01)

The script computes the clarity of every second target on the frame grid (caches `cache/clarity/{shard}/{target}`),
the pairwise records (`cache/contrast/{shard}/cont/{token}-cont-{second}_R.csv`) and the curves (`_T.csv`), and
draws `plot/clarity/contrast/cont/clarity_overlap-cont.svg` and `clarity_profile-cont.svg`. The per-pair statistics
are printed by the visualizer's logger; the table below is the printed summary (tokens with >= 3 significant cells
on both sides for the overlap, >= 3 events of both kinds and >= 3 reference cells for the profile).

| pair (reference: success cells) | overlap tokens | containment observed / chance (Wilcoxon p) | pooled shared / smaller set | Spearman of z across cells | profile tokens | r observed / relabelling null | z_r | tokens z_r <= -2 |
|---|---|---|---|---|---|---|---|---|
| success vs missed | 9 | 0.667 / 0.623 (0.57) | 74 / 88 | 0.25 | 14 | 0.442 / 0.378 | +0.43 | 1 (3 tokens z_r >= +2) |
| success vs in vain | 30 | 0.589 / 0.479 (0.001) | 270 / 354 | 0.30 | 40 | -0.011 / 0.338 | -2.54 | 20 |
| success vs in vain (near) | 30 | 0.589 / 0.466 (0.002) | 288 / 368 | 0.35 | 39 | 0.007 / 0.355 | -2.57 | 18 |
| success vs in vain (isolated) | 7 | 0.621 / 0.542 (0.47) | 22 / 37 | 0.02 | 20 | -0.055 / 0.312 | -2.28 | 8 |
| success vs cue (no attempt) | 21 | 0.457 / 0.412 (0.18) | 50 / 110 | 0.04 | 40 | -0.009 / 0.302 | -3.57 | 21 |
| success vs walk | 35 | 0.565 / 0.506 (0.02) | 238 / 375 | 0.14 | 39 | -0.071 / 0.248 | -2.50 | 17 |

Population time course of the success cells (mean z per second, `clarity_profile-cont.svg`): success dips to
-0.08 at -2..-1 s and stays near 0 after the onset; missed follows the same dip and rises to +0.11 at +1 s (within
the error band of success at every second); in vain rises from the onset to +0.2..+0.25 at +2..+5 s, isolated in
vain to +0.3 at +3 s; cue without attempt is flat at 0; walk rises slowly to +0.2 at +4 s.

Reading the overlap columns: pooled over all tokens the containment is only modestly above chance, because the chance
level (success set / all cells) is itself 0.4-0.6 and climbs to 0.9 in the clearest tokens. The plot therefore
bins the tokens by their clarity (fraction of success-significant cells, 15 equal bins, clear = fraction >= 0.4)
and shows one histogram per pair: the bar height is the number of tokens in the bin and the bar colour is their mean
containment, darker = more of the other target's cells inside the success cells (`clarity_overlap-cont.svg`); the
table below groups the bins into the clear and the unclear side, where the conclusion is read:

| pair | clear tokens | containment clear side (chance) | pooled clear side | informative bucket 0.6-0.8 (chance) | unclear tokens | containment unclear side (chance) |
|---|---|---|---|---|---|---|
| success vs missed | 8 | 0.75 (0.66) | 74 / 84 | 0.76 (0.72), 4 tokens | 1 | 0.00 (0.30) |
| success vs in vain | 15 | 0.72 (0.64) | 227 / 269 | 0.87 (0.71), 6 tokens | 15 | 0.46 (0.32), p 0.003 |
| success vs in vain (near) | 14 | 0.81 (0.64), p 0.003 | 248 / 280 | 0.93 (0.71), 5 tokens | 16 | 0.40 (0.32) |
| success vs in vain (isolated) | 4 | 0.74 (0.74) | 17 / 26 | 0.61 (0.74), 1 token | 3 | 0.47 (0.28) |
| success vs cue (no attempt) | 8 | 0.69 (0.65) | 27 / 40 | 0.66 (0.74), 2 tokens | 13 | 0.32 (0.26) |
| success vs walk | 17 | 0.68 (0.64) | 165 / 233 | 0.73 (0.73), 5 tokens | 18 | 0.46 (0.39) |

So in the clear tokens the in-vain cells are the success cells (0.81-0.93 against 0.64-0.71 chance), the missed
cells are the success cells in the tokens that have them (74 of 84 pooled, but only 8 tokens), the cue cells are not
(at or below chance in every clear bucket), and walk is at chance. In the unclear tokens the sets are small and the
containment is near chance for every pair. The decisive measurement for "same response?" remains the profile
similarity: success vs missed sits inside the relabelling null, every other pair sits 2.3-3.6 null standard deviations
below it.

Clarity screens predictability for the other targets as well (`clarity_auc.py --beh ...`, Spearman between the
clarity fraction and the ExtraTrees auc_fold, pooled tokens): cont_inv 0.65 (43 tokens; init 0.78, memory 0.46),
walk 0.62 (45), cont_missed 0.72 (14 tokens with enough events), cue_none 0.37 (42). The clearest / least-clear
subset test on cont_inv (`clarity_subset.py --beh cont_inv`, 43 tokens): all 0.652, top 0.640, random 0.634,
bottom 0.599; top - bottom +0.041 (29/43, p 0.018), top - random +0.006 (n.s.): the in-vain target is less
predictable than the success target and its clear cells are less decisive.

### 7.4 Interpretation (agreed wording)

- **Not water.** Missed reaches (no water) engage the same cells with the same response as successful reaches:
  overlap far above chance, profile correlation inside the relabelling null, no difference in the pre, onset or late
  windows. The two confounders cancel in this pair (both have the cue, both are the first attempt after a quiet
  wait), so this is the clean comparison.
- **Not the paw movement alone.** In-vain reaches are the same movement but the same cells respond in the opposite
  direction (suppression around a cued first reach, excitation after an in-vain reach). Walk shares neither.
- **Not the cue stimulus.** The reach cells are silent at cues that are not followed by a reach, and the cue has its
  own small population at chance overlap with the reach cells. The cue defines the context (water available) but
  does not drive the reach cells.
- **Not the acute aftermath.** Isolated in-vain reaches (no reach in the 8 s before) show the excitation even more
  strongly than the ones right after a drink, so the reversal is a response of the in-vain event itself.
- **Open.** Three candidates for the reversal remain inseparable in the present data: the absence of the cue
  context, the state of having already drunk in that trial, and the movement itself (aimed first reach vs. repeated
  reaching). The discriminating event exists but is rare: in-vain reaches that happen after a cue but before any
  success in the same trial (cued, not yet drunk, in vain). If they look like success / missed (suppression) the
  cause is the drunk state; if they look like ordinary in vain (excitation) the cause is the cue distance or the
  movement.
- Earlier statement to avoid: "success and in vain are two different populations" was a misreading of r ~ 0 (same
  cells, opposite shape -> correlation zero).

---

## 8. Repository layout and conventions

- Packages under `src/` (see README). Every engine: `__init__(sig_folder, beh_folder, seed, log_level)`,
  `compute(token, ...)`, `collect(...)` with `token_filter`, `n_workers`, `rerun`, `tgt_folder`; caches
  `{stem}_X.csv` + `{stem}_C.json`, reused only when the configuration dict matches; `__getstate__` drops the logger
  so engines pickle onto worker processes (`src.util.parallel_map`, joblib loky).
- `src.util.load_paths()` reads `config/.env.path` relative to the package, so modules and scripts work from any
  working directory; scripts still expect `PYTHONPATH=..` from `script/` (or `PYTHONPATH=.` from the root).
- Logging: `ABlockLogger` with keyword calls `logger.info(role=..., message=...)`; `logger.error` does not raise,
  so an explicit `raise RuntimeError` follows it.
- Style: one parameter per line with column-aligned defaults, reST docstrings (`:param x:` lines, blank line,
  `:return:`), lowercase block comments every few lines, `__author__ = "Tianyi Ma"` and explicit `__all__` in every
  `__init__.py`, scripts with an aligned `# config the script parameters` block and
  `logger.warning("start execute x.py")` / `"complete execute x.py"`. Commit messages on master:
  "init: ...", "feature: ...", "fix bug: ...", "refactory: ..."; recent commits "[init] ...". No Co-Authored-By
  lines.
- Plots: svg only; short titles with the key statistic in parentheses, everything else in the logger; colours init
  blue `#4e7abd` (light `#9dbbe0`), memory red `#c43639` (light `#e39b9c`), control grey `#d1d1d1`, highlight blue
  `#56a4d9`; maps use `afmhot` for significant cells and `viridis` for the others with a `#3a3a3a` mask at 45 %
  alpha outside the clarity window; a single dashed line at 0.5 on AUC panels.
- Environment: conda env `regular` (`D:/SW_Anaconda/envs/regular/python.exe`): numpy 2.2, pandas 2.2,
  scikit-learn 1.6, scipy 1.15, matplotlib 3.10, joblib, python-dotenv, imbalanced-learn; 32 CPUs; a GPU exists
  (RTX 4060 8 GB) but nothing in this repository uses it (trees are CPU-only).
- Caches tracked in git: clarity (init/memo; cont, cont_inv, cont_missed, cont_inv_near, cont_inv_isolated,
  cue_none, walk), extratrees (init/memo; cont incl. `_noise` and `subset_30`, and the targets run by the scripts),
  contrast (init/memo; cont). `cache/deduce` (5.8 GB of window JSONs) is untracked.
- Tests: `test/cache_test.py` recomputes the committed clarity / ExtraTrees caches of K227_A_D7 and checks equality
  (float32 precision / exact), and the determinism of the contrast engine; `test/preprocessor_test.py` checks the
  determinism of `Preprocessor` and the consistency of the frame-grid and native-rate targets.

---

## 9. History of the work (branches)

- `master` (9b7aef4): the original repository (`src_deduction`, `src_predict`, `src_select`, `tools`, leaky).
- `feat_predict_algo` (from 9b7aef4): exploratory record under `temp/` (sequence models, clarity, cell selection,
  PDF transfers, model search, hold-out validation, select_cells API); `temp/FINDINGS.md` holds the full tables.
  Kept as the exploratory record; nothing there is needed to run this branch.
- `feat_clarity` (from 9b7aef4): c15c8e1 ported clarity + ExtraTrees as `src_clarity` with two scripts; the current
  commit restructures everything into `src/{util,preprocess,clarity,prediction,deduction}`, deletes the leaky
  selection / prediction code and caches, adds the event-level preprocessing, the contrast engine, the map /
  overlap / profile visualizer, the targets registry, parallel collection, the docs, and re-homes the caches
  (`cache/clarity_init/cont` -> `cache/clarity/init/cont`, etc.; the cached numbers are reproduced bit-for-bit by
  the new code, see `test/cache_test.py`).

---

## 10. How to extend

- New event type: add a `Target` to `TARGETS` (and a label to `LABELS`); run `clarity_map.py --beh <key>`,
  `clarity_auc.py --beh <key>`, `clarity_contrast.py --first cont --seconds <key>`.
- New predictor: mirror `ExtraTreesEngine` (same `predict` / `collect` contract, same cache layout under
  `cache/<name>/{shard}/{beh}`), reuse `BlockPreprocessor.block_folds` and `ClarityEngine.rank` for the subsets,
  and keep `auc_fold` as the score.
- The discriminating experiment of section 7.4 is one more target: in-vain reaches with `latency <= 3 s` and no
  success since the last cue; the rule has access to the whole behavior table.
- Do not reintroduce shuffled splits, do not rank cells on frames outside the training block, and do not quote the
  pooled AUC alone.
