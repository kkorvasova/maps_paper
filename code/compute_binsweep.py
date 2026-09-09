"""
compute_binsweep.py
===================
Main analysis with a FR bin-size sweep and the UNIFORM-RANDOM control.

For each monkey / recording day / V1 array / condition (all / EC / EO) /
FR bin size:

  1. Load tMUA spikes, build the FR matrix at the given bin size for the
     condition (no binarisation, no upstates).
  2. Restrict to ORIENTED channels, full Pearson correlation matrix, PCA
     (up to max_dim components; PCA on oriented channels only).
  3. For each PC pair, derive the geometric OP map (optimal rotation/flip
     fit on oriented channels) and compute L1 + L2 error vs the real OP map.
  4. For each of the 2000 uniform-random control maps, refit the optimal
     rotation/flip in the SAME PC cloud and record its L1 + L2 error.
  5. Save PCA scores and a long-form error DataFrame, both carrying a
     `bin_ms` column.

Parallelisation
---------------
One (monkey, bin_size) per SLURM array task.  With 3 monkeys x 7 bins
there are 21 tasks:

    python compute_binsweep.py $SLURM_ARRAY_TASK_ID    # 0..20

Task index -> (monkey, bin) mapping is row-major over
    monkeys x bin_sizes
i.e. task = monkey_idx * n_bins + bin_idx.  Print the map with:

    python compute_binsweep.py map

Merge per-task outputs afterwards with:

    python compute_binsweep.py merge

Outputs (under df_folder/binsweep/):
    pca_scores_{monkey}_bin{bin_ms}.pkl
    errors_long_{monkey}_bin{bin_ms}.pkl
and after merge:
    pca_scores_binsweep.pkl
    errors_long_binsweep.pkl

Long-form DataFrame columns:
    monkey, date, array, condition, bin_ms, pc_pair, pc_x, pc_y,
    source ('real' or 'ctrl'), ctrl_idx (-1 for real),
    norm ('L1' or 'L2'), error
"""

import os
import sys
import time
import glob
import pickle
import itertools
import numpy as np
import pandas as pd
import yaml

from functions_maps import (
    load_spike_array, load_ec_indicator, load_op_map,
    extract_fr_matrix, compute_correlation_matrix, compute_pca,
    polar_angles_from_centroid, best_rotation_to_op, apply_geometric_map,
    map_errors, pc_pairs, ensure_dir_exists,
)

# ── parameters ────────────────────────────────────────────────────
with open("params_binsweep.yml") as f:
    P = yaml.safe_load(f)

DATA_FOLDER = P['data_folder']
OP_MAP_FOLDER = P['op_map_folder']
EC_EO_FOLDER = P['ec_eo_folder']
DF_FOLDER = P['df_folder']

MONKEYS = P['monkeys']
V1_ARRAYS = P['v1_arrays']
DATES = P['dates']

BIN_SIZES = P['method']['bin_sizes']
SIG_TYPE = P['method']['sig_type']
CORR_METHOD = P['method']['corr_method']
MAX_DIM = P['method']['max_dim']
N_CH = P['method']['n_channels']
CONDITIONS = P['method']['conditions']

SEL_MIN = P['channel_selection']['selectivity_min']
JUMP_MAX = P['channel_selection']['num_jump_max']

CTRL_DIR = os.path.join(DF_FOLDER, 'controls')
OUT_DIR = os.path.join(DF_FOLDER, 'binsweep')
ensure_dir_exists(OUT_DIR)

# task index -> (monkey, bin_ms), row-major over monkeys x bin_sizes
TASK_GRID = list(itertools.product(MONKEYS, BIN_SIZES))


def load_controls(monkey, array):
    """Load the pre-generated uniform control maps for one array, or None."""
    path = os.path.join(CTRL_DIR, f'control_maps_monkey_{monkey}_array{array}.npy')
    if not os.path.exists(path):
        return None
    return np.load(path)


def errors_for_target(scores, pairs, oriented_idx, target_op_full):
    """
    L1/L2 errors for every PC pair for one target OP map (real or control),
    refitting the optimal rotation/flip in the PC cloud.
    Returns dict[label] -> (l1, l2).
    """
    op_oriented = target_op_full[oriented_idx]
    valid = op_oriented >= 0
    out = {}
    for pc_x, pc_y, label in pairs:
        x, y = scores[:, pc_x], scores[:, pc_y]
        phi = polar_angles_from_centroid(x, y)
        best_rot, best_flip = best_rotation_to_op(phi[valid], op_oriented[valid])
        geo = apply_geometric_map(phi, best_rot, best_flip)
        derived_full = np.full(N_CH, -1.0)
        derived_full[oriented_idx] = geo
        l1, l2 = map_errors(derived_full, target_op_full)
        out[label] = (l1, l2)
    return out


def process(monkey, bin_ms):
    """Run the full analysis for one (monkey, bin_ms) and save its files."""
    pca_store = {}
    rows = []
    t_start = time.time()

    for date in DATES[monkey]['RS']:
        try:
            EC_indic = load_ec_indicator(monkey, date, EC_EO_FOLDER)
        except Exception as exc:
            print(f"[skip day] monkey {monkey} {date}: no EC indicator ({exc})")
            EC_indic = None

        for array in V1_ARRAYS[monkey]:
            t_cell = time.time()

            try:
                op_array = load_op_map(
                    monkey, array, OP_MAP_FOLDER,
                    selectivity_min=SEL_MIN, num_jump_max=JUMP_MAX,
                    n_channels=N_CH)
            except Exception as exc:
                print(f"[skip] OP map monkey {monkey} array {array}: {exc}")
                continue

            controls = load_controls(monkey, array)
            if controls is None:
                print(f"[warn] no controls for monkey {monkey} array {array} "
                      f"-- run generate_controls.py")

            t_load = time.time()
            spike_arr = load_spike_array(
                monkey, array, date, DATA_FOLDER, sig_type=SIG_TYPE)
            load_s = time.time() - t_load
            if spike_arr is None:
                print(f"[skip] spikes monkey {monkey} {date} array {array}")
                continue

            for condition in CONDITIONS:
                if condition in ('EC', 'EO') and EC_indic is None:
                    continue

                fr = extract_fr_matrix(spike_arr, EC_indic, condition, bin_ms)
                if fr is None:
                    print(f"[skip] {monkey} {date} arr{array} {condition} "
                          f"bin{bin_ms}: too few bins")
                    continue

                C = compute_correlation_matrix(fr, method=CORR_METHOD)
                scores, evr, mask = compute_pca(C, op_array, max_dim=MAX_DIM)
                if scores is None:
                    print(f"[skip] {monkey} {date} arr{array} {condition} "
                          f"bin{bin_ms}: <3 oriented channels")
                    continue

                oriented_idx = np.where(mask)[0]
                pairs = pc_pairs(MAX_DIM, scores.shape[1])

                key = (monkey, date, int(array), condition, int(bin_ms))
                pca_store[key] = dict(
                    scores=scores, evr=evr, mask=mask,
                    oriented_idx=oriented_idx)

                # real-map errors
                real_err = errors_for_target(scores, pairs, oriented_idx, op_array)
                for pc_x, pc_y, label in pairs:
                    l1, l2 = real_err[label]
                    for nrm, val in (('L1', l1), ('L2', l2)):
                        rows.append(dict(
                            monkey=monkey, date=str(date), array=int(array),
                            condition=condition, bin_ms=int(bin_ms),
                            pc_pair=label, pc_x=pc_x + 1, pc_y=pc_y + 1,
                            source='real', ctrl_idx=-1, norm=nrm, error=val))

                # control-map errors
                if controls is not None:
                    for k in range(controls.shape[0]):
                        ctrl_err = errors_for_target(
                            scores, pairs, oriented_idx, controls[k])
                        for pc_x, pc_y, label in pairs:
                            l1, l2 = ctrl_err[label]
                            for nrm, val in (('L1', l1), ('L2', l2)):
                                rows.append(dict(
                                    monkey=monkey, date=str(date),
                                    array=int(array), condition=condition,
                                    bin_ms=int(bin_ms), pc_pair=label,
                                    pc_x=pc_x + 1, pc_y=pc_y + 1,
                                    source='ctrl', ctrl_idx=k,
                                    norm=nrm, error=val))

            print(f"done  {monkey} {date} arr{array:2d} bin{bin_ms:4d}  "
                  f"load={load_s:5.1f}s  cell={time.time()-t_cell:5.1f}s  "
                  f"rows so far={len(rows)}")

    tag = f"{monkey}_bin{bin_ms}"
    pca_path = os.path.join(OUT_DIR, f'pca_scores_{tag}.pkl')
    with open(pca_path, 'wb') as f:
        pickle.dump(pca_store, f)
    df = pd.DataFrame(rows)
    df_path = os.path.join(OUT_DIR, f'errors_long_{tag}.pkl')
    df.to_pickle(df_path)

    print(f"\n[{tag}] saved {pca_path}  ({len(pca_store)} entries)")
    print(f"[{tag}] saved {df_path}  ({len(df)} rows)")
    print(f"[{tag}] total time {time.time()-t_start:.1f}s")


def merge():
    """Combine per-task outputs into single binsweep files."""
    pca_combined = {}
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'pca_scores_*_bin*.pkl'))):
        with open(path, 'rb') as f:
            pca_combined.update(pickle.load(f))
    out_pca = os.path.join(DF_FOLDER, 'pca_scores_binsweep.pkl')
    with open(out_pca, 'wb') as f:
        pickle.dump(pca_combined, f)
    print(f"merged PCA scores -> {out_pca}  ({len(pca_combined)} entries)")

    dfs = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'errors_long_*_bin*.pkl'))):
        dfs.append(pd.read_pickle(path))
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        out_df = os.path.join(DF_FOLDER, 'errors_long_binsweep.pkl')
        df.to_pickle(out_df)
        print(f"merged errors    -> {out_df}  ({len(df)} rows)")
    else:
        print("no per-task error files found to merge")


def print_map():
    print(f"{len(TASK_GRID)} tasks (monkeys x bin_sizes):")
    for i, (m, b) in enumerate(TASK_GRID):
        print(f"  task {i:2d} -> monkey {m}  bin {b} ms")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "merge":
        merge()
    elif arg == "map":
        print_map()
    elif arg is None:
        # no argument: run every (monkey, bin) in one process
        for m, b in TASK_GRID:
            process(m, b)
    else:
        task = int(arg)
        if task < 0 or task >= len(TASK_GRID):
            raise SystemExit(f"task index {task} out of range "
                             f"(0..{len(TASK_GRID)-1})")
        m, b = TASK_GRID[task]
        print(f"task {task} -> monkey {m}  bin {b} ms")
        process(m, b)
