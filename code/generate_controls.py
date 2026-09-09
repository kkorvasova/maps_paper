"""
generate_controls.py
=====================
Build UNIFORM-RANDOM control OP maps (the new null).

For every monkey / V1 array, take the real measured OP map (with channel
selection applied) and create `n_control` control maps in which every
ORIENTED channel is assigned an orientation drawn i.i.d. from
Uniform[0, 180); UNORIENTED channels keep -1 (and their positions).

This differs from the old label-permutation null: it does NOT preserve the
empirical OP histogram of the array, it samples flat orientation.

Controls are a property of the array only (independent of recording day,
condition and FR bin size), so we generate them once per monkey/array.

Output (under df_folder/controls/):
    control_maps_monkey_{M}_array{A}.npy   [n_control, n_ch]
    control_maps_all.pkl                   dict[(M, array)] -> [n_control, n_ch]

Run:
    python generate_controls.py
"""

import os
import pickle
import numpy as np
import yaml

from functions_maps import load_op_map, ensure_dir_exists

# ── parameters ────────────────────────────────────────────────────
with open("params_binsweep.yml") as f:
    P = yaml.safe_load(f)

OP_MAP_FOLDER = P['op_map_folder']
DF_FOLDER = P['df_folder']
MONKEYS = P['monkeys']
V1_ARRAYS = P['v1_arrays']

SEL_MIN = P['channel_selection']['selectivity_min']
JUMP_MAX = P['channel_selection']['num_jump_max']
N_CH = P['method']['n_channels']

N_CONTROL = P['control']['n_control']
SEED = P['control']['random_seed']

OUT_DIR = os.path.join(DF_FOLDER, 'controls')
ensure_dir_exists(OUT_DIR)


def make_uniform_controls(op_array, n_control, seed):
    """
    Build `n_control` uniform-random control maps. Oriented channels get
    orientation ~ Uniform[0,180); unoriented channels keep -1.
    Returns ndarray [n_control, n_ch].
    """
    op_array = np.asarray(op_array, dtype=float)
    n_ch = len(op_array)
    oriented_idx = np.where(op_array != -1)[0]
    n_oriented = len(oriented_idx)

    rng = np.random.default_rng(seed)
    controls = np.full((n_control, n_ch), -1.0)
    draws = rng.uniform(0.0, 180.0, size=(n_control, n_oriented))
    controls[:, oriented_idx] = draws
    return controls


def main():
    combined = {}
    for monkey in MONKEYS:
        for array in V1_ARRAYS[monkey]:
            try:
                op_array = load_op_map(
                    monkey, array, OP_MAP_FOLDER,
                    selectivity_min=SEL_MIN, num_jump_max=JUMP_MAX,
                    n_channels=N_CH)
            except Exception as exc:
                print(f"[skip] monkey {monkey} array {array}: {exc}")
                continue

            n_oriented = int((op_array != -1).sum())
            array_seed = SEED + hash((monkey, array)) % (2 ** 31)
            controls = make_uniform_controls(op_array, N_CONTROL, array_seed)

            out_path = os.path.join(
                OUT_DIR, f'control_maps_monkey_{monkey}_array{array}.npy')
            np.save(out_path, controls)
            combined[(monkey, array)] = controls

            print(f"monkey {monkey}  array {array:2d}  "
                  f"oriented={n_oriented:2d}  controls={controls.shape}  "
                  f"-> {out_path}")

    combined_path = os.path.join(OUT_DIR, 'control_maps_all.pkl')
    with open(combined_path, 'wb') as f:
        pickle.dump(combined, f)
    print(f"\nSaved combined controls: {combined_path}")
    print(f"Total arrays: {len(combined)}")


if __name__ == "__main__":
    main()
