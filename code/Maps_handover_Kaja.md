# maps_paper — Project Handover

**Goal.** Recover the measured orientation-preference (OP) map of macaque V1 Utah
arrays from *resting-state* spontaneous activity, using PCA on the channel
correlation structure. The question is not only "can we do it" but "for which
arrays, how well, how stably across days, and does eye state (EC/EO) matter."

**Data.** Three monkeys (L, N, F). Per monkey, 7-14 V1 Utah arrays of 64
channels. Recording days: L has 3, N has 2, F has 2. Each resting-state
recording is split into eyes-closed (EC) and eyes-open (EO) periods. For every
array a measured OP map exists; channels are labelled *oriented* or *unoriented*
by two selection parameters.

---

## 1. Method (as currently implemented)

Per (monkey, day, array, condition):

1. Load tMUA spikes at 1 ms resolution.
2. Mask to a condition: `all`, `EC`, or `EO`.
3. Bin firing rate at a chosen bin size (50 ms in the first pipeline; swept over
   {20, 50, 100, 250, 500, 750, 1000} ms in the bin-sweep pipeline). **No
   binarisation and no upstate detection** - this is a deliberate simplification
   of the older approach.
4. Keep only *oriented* channels. A channel is unoriented (set to -1) if
   `selectivity_01 < 0.15` OR `num_f0_high_jump > 2` (the "two parameters").
5. Full Pearson correlation matrix of the oriented channels.
6. PCA on that correlation matrix, up to 7 components ("Option A": PCA is fit on
   oriented channels only).
7. For each PC pair (up to 21 pairs from 7 PCs), read each channel's polar angle
   in that 2-D PC plane, then find the rotation offset and optional flip that
   best matches the real OP (fit on oriented channels). This gives a derived OP
   per channel.
8. Error of the derived map vs the real map, in two norms: L1 (mean absolute
   circular error) and L2 (RMSE circular error).

The OP-derivation math (`polar_angles_from_centroid`, `best_rotation_to_op`,
`apply_geometric_map`, `map_errors`) lives in `functions_maps.py` and was taken
faithfully from the original `utah_ec_analysis` module.

### The significance test (important)

Because we choose the *best* of 21 PC pairs, a naive per-pair significance test
is biased. The correct test is **min-over-pairs**:

- real statistic = min over pairs of the real error (best reconstruction),
- null = for each control map, min over pairs of *its* error (each control also
  gets to pick its own best pair),
- percentile = `100 * mean(null_min <= real_min)`.

Low percentile (< 5) means the real reconstruction beats the control. This
correction matters: the min-over-pairs null sits noticeably lower than any
single-pair null, so it is a genuinely stricter test. Both the *per-pair*
percentile (uncorrected, looser - fine for illustration) and the *min-over-pairs*
percentile (strict - use for claims) appear in the notebooks; do not quote the
per-pair number as headline significance.

---

## 2. Files

### Pipeline A - label-permutation null, fixed 50 ms bin
- `params_maps.yml` - parameters (folders, arrays, dates, method, selection,
  `n_perm: 1000`, layouts).
- `functions_maps.py` - **core module, shared by everything**. Data loading,
  FR-matrix construction, correlation, PCA, OP derivation, error metrics,
  permutation generator. If you touch the method, touch it here.
- `generate_permutations.py` - 1000 label-permuted OP maps per array (oriented
  labels shuffled among themselves, unoriented kept at -1). Saved under
  `dataframes/permutations/`.
- `compute_errors.py` - main analysis, parallelised **by monkey** (`--array=0,1,2`).
  Writes `errors_long_{monkey}.pkl` and `pca_scores_{monkey}.pkl`; `merge`
  subcommand combines them.

### Pipeline B - uniform-random null, bin sweep (the current main pipeline)
- `params_binsweep.yml` - adds `method.bin_sizes` and a `control` block
  (`n_control: 2000`, uniform).
- `generate_controls.py` - 2000 **uniform-random** control OP maps per array
  (oriented channels drawn i.i.d. from Uniform[0,180), unoriented kept at -1).
  Saved under `dataframes/controls/`. Generated once per array (independent of
  day/condition/bin).
- `compute_binsweep.py` - main analysis, parallelised **by (monkey, bin)**:
  3 monkeys x 7 bins = 21 tasks (`--array=0-20`). Error rows carry a `bin_ms`
  column and use `source='ctrl'`. Subcommands: task index runs one cell, `map`
  prints the task table, `merge` combines to `errors_long_binsweep.pkl` and
  `pca_scores_binsweep.pkl`.
- `repack_binsweep.py` - **run once after merge**. Pools the 2000 control errors
  into a single vector per (cell, pair, norm), producing
  `errors_compact_binsweep.pkl` (~1000x fewer rows, far faster notebooks). The
  original long file is left untouched.

### SLURM scripts
- `run_maps.sh` / `run_prep.sh` - pipeline A (array job + controls/merge helper).
- `run_binsweep.sh` / `run_binsweep_prep.sh` - pipeline B.
- `run_repack.sh` - the repack step on a compute node (needs high memory, ~200G).

### Notebooks
- `plot_op_maps.ipynb` - pipeline A visualisation: per-pair real-vs-null errors,
  best-pair selection with p-value, derived-vs-real maps on the 8x8 layout,
  pooled summaries, EC-vs-EO scatter, and the cumulative percentile grid.
  Contains the first `build_percentile_df` / min-over-pairs percentile table.
- `plot_binsweep.ipynb` - pipeline B visualisation: the min-over-pairs percentile
  table, per-monkey L1-vs-L2 percentile grids, per-pair boxplots (5/95 marked),
  the "optimal" (min-over-pairs) control distribution histogram, L1-vs-L2
  agreement, and the bin-size scan summary that picks the best bin.
- `explore_pca_pairs_full.ipynb` - the deep-dive notebook (reads the compact
  file). Four sections: (1) example cell with the real map beside the 7x7
  triangle grid (upper = derived maps + per-pair percentile + green borders,
  diagonal = explained variance, lower = PC scatters) plus per-pair boxplots;
  (2) strict min-over-pairs percentile grid; (3) across-day stability of per-pair
  information for arrays with >=25 oriented channels; (4) EC-vs-EO comparison for
  arrays significant in both (own-best and shared-pair derived maps, plus which
  pairs win in each condition). Later cells added: line-plot versions of sections
  3 and 4, day-to-day similarity matrices (L2 distance + top-k Jaccard), pooled
  similarity across arrays per animal, and winning-pair barplots by condition.

### Run order
Pipeline B (current): `generate_controls.py` -> `compute_binsweep.py` (21 tasks)
-> `compute_binsweep.py merge` -> `repack_binsweep.py` -> notebooks.
Chain the dependency on the cluster:
`CTRL=$(sbatch --parsable run_binsweep_prep.sh controls); SWEEP=$(sbatch --parsable --dependency=afterok:$CTRL run_binsweep.sh); sbatch --dependency=afterok:$SWEEP run_binsweep_prep.sh merge`.

---

## 3. What was tried, including dead ends

- **Label-permutation vs uniform-random null.** We started with label
  permutations (`generate_permutations.py`), which preserve the array's OP
  histogram. We then switched to uniform-random controls
  (`generate_controls.py`, 2000 draws) as the primary null, which samples flat
  orientation and is a cleaner "no structure" baseline. Both pipelines still
  exist; pipeline B (uniform) is the one to build on.
- **Naive per-pair significance (rejected).** The first percentile used a single
  pair's own null. This ignores the best-of-21 selection and is optimistically
  biased - it produced far too many "significant" cells. Replaced by the
  min-over-pairs test everywhere it matters. This was the single most important
  correction in the project.
- **PCA on all channels vs oriented only.** Settled on Option A (PCA fit on
  oriented channels only). The full-64 variant was considered and dropped.
- **Bin size.** Rather than guess, we swept 20-1000 ms. See the bin-scan summary
  in `plot_binsweep.ipynb`; pick the bin from that figure rather than assuming.
- **Storage format.** The long-form error DataFrame stores 2000 rows per
  cell/pair/norm and became slow to filter. `repack_binsweep.py` pools controls
  into vectors; the deep-dive notebook is built on the compact format. If memory
  is tight during repack, rewrite it to loop over the 21 per-task files instead
  of loading the merged file at once (noted but not yet done).
- **Engineering gotchas fixed along the way.** A race condition in directory
  creation under concurrent SLURM tasks (fixed with `exist_ok=True` in
  `ensure_dir_exists`). Repeated `DataFrame.apply(..., axis=1)` building tuple
  keys threw "unhashable type: NumpyExtensionArray" on this pandas version;
  replaced with vectorised `groupby` / `MultiIndex.isin`.

---

## 4. Main findings

- **Recovery is real but array-specific.** The OP map is recovered well and
  reproducibly for a subset of arrays, and not at all for others - not an
  all-or-nothing global effect. This is visible in the percentile grids
  (`plot_binsweep.ipynb`, `explore_pca_pairs_full.ipynb` section 2): some monkey
  L arrays (e.g. 13, 14, 16) are deep blue (percentile ~0) across conditions and
  all three days, while others (e.g. L11, both F arrays shown) sit red.
- **Day-to-day reproducibility is the strongest signal.** The same arrays recover
  across independent recording days with a freshly redrawn control. This is the
  most defensible headline because chance would not recur in the same arrays.
- **The map-carrying PC pair is not fixed.** Which PC pair best reconstructs the
  map shifts across days and conditions (section 1 triangle grid; section 3
  across-day profiles). Katarina can reproduce this in a model - adding input
  rotates which PCs the map loads onto. This is a mechanistic point, but be ready
  for the reviewer reading of "unstable winner = noise"; the defence is that the
  null is also min-over-pairs and the recovered arrays are significant despite
  the correction and reproducibly across days.
- **Recovery requires enough oriented channels.** Below a channel count,
  significance is effectively impossible (the null and real overlap). The
  significance-vs-oriented-count analysis suggested a cutoff; grids were then
  filtered to arrays with >~24-25 oriented channels.
- **EC vs EO is mixed, not a clean story (yet).** Some arrays reconstruct better
  in one condition than the other, but not uniformly. Needs the pooled paired
  comparison restricted to recoverable arrays before any claim.
- **Preliminary, promising: orientation-biased up-states.** Katarina observed
  that EC up-states seem to preferentially replay one orientation preference.
  This is the most exciting lead and the least developed - see prospects.

---

## 5. How to continue

Ordered roughly by value and readiness.

1. **Define a single recoverability score per array** pooled across days and
   conditions (e.g. fraction of day/condition cells with min-over-pairs
   percentile < 5, or the median percentile). This turns the grids into a clean
   "N of M arrays recover" statement for the paper.
2. **Explain *which* arrays recover.** Regress the recoverability score on
   oriented-channel count and on a smoothness measure of the real OP map (Moran's
   I - there was Moran's I code in the original `utah_ec_analysis` module). If
   smoothness predicts recoverability, that is a clean mechanistic headline.
3. **Formalise across-day stability.** The similarity matrices (L2 distance +
   top-k Jaccard) and the pooled-per-animal plots exist in
   `explore_pca_pairs_full.ipynb`. With only 3 days for L (day-pairs 1-2, 1-3,
   2-3) and 2 for N/F, keep this descriptive: report whether the same pairs carry
   the map, and whether later days resemble each other more than the first. Do
   not over-claim significance at n=3.
4. **Tie the shifting PC pair to the model.** If the model predicts *which* pair
   should carry the map as a function of input/state, show the model prediction
   next to the data observation. That converts a potential weakness into a
   result.
5. **Develop the up-state replay observation properly.** Before it enters the
   paper it needs: a defined up-state detector; per up-state, a readout of which
   OP is most active; and a test that the distribution of replayed OPs is
   non-uniform *beyond what the array's own OP composition predicts* (if 40% of
   channels prefer ~90 deg, then 40% replaying at 90 deg is nothing). Ideally,
   test whether the replayed-OP bias predicts which arrays are recoverable - that
   would link the two main observations mechanistically. Note this reintroduces
   up-states, which the current pipeline deliberately dropped; keep it as a
   separate analysis rather than folding it back into `compute_binsweep.py`.

### Suggested paper shape
1. Spontaneous tMUA correlation structure, reduced by PCA, recovers the measured
   OP map in a subset of V1 arrays - established with a uniform-random control and
   the selection-corrected min-over-pairs test.
2. Recovery is reproducible across recording days and predicted by
   [oriented-channel count / map smoothness].
3. The map-carrying principal component is not fixed; it shifts with state/input,
   reproduced in a model.
4. (If it holds) Spontaneous up-states preferentially replay specific
   orientations, suggesting the correlation structure arises from discrete
   reinstatement events.

Parts 1-2 are a solid short paper on their own; 3 and 4 are what would make it a
strong one, each contingent on the controls above.

---

## 6. Gotchas / notes for whoever picks this up

- Everything imports from `functions_maps.py`; the method lives there.
- Two params files and two pipelines coexist. Use `params_binsweep.yml` /
  pipeline B unless you specifically want the label-permutation null.
- OP maps and EC/EO indicators are read from their *original* locations
  (`final_OP_maps/`, `ripple_waves/metadata/`), not copied into the project.
- Controls/permutations use a deterministic per-array seed, so they are
  reproducible regardless of run order.
- The compact file (`errors_compact_binsweep.pkl`) is what the deep-dive notebook
  expects - run `repack_binsweep.py` first.
- `day_pair` labels in the stability analysis are *positional* (1-2 = first two
  dates in `dates[monkey]['RS']`); check that ordering matches the intended
  day numbering.
- With all 21 pairs, distance-based similarity is compressed by the ~15
  chance-level pairs that look alike every day; read the *ordering* of
  off-diagonals, not the absolute spread.
