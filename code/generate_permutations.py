"""
generate_permutations.py
========================
For every monkey / V1 array, load the real measured OP map (with channel
selection applied), build `n_perm` permuted copies (oriented channels
shuffled among themselves, unoriented kept at -1), and save them.

Permutations do NOT depend on recording day or condition -- they are a
property of the real OP map of an array -- so we save one file per
monkey/array, plus a single combined pickle for convenience.

Output (under df_folder/permutations/):
    perm_maps_monkey_{M}_array{A}.npy      [n_perm, n_ch]
    permuted_maps_all.pkl                  dict[(M, array)] -> [n_perm, n_ch]

Run:
    python generate_permutations.py
"""

import os
import pickle
import numpy as np
import yaml

from functions_maps import load_op_map, make_permuted_maps, ensure_dir_exists

# ── parameters ────────────────────────────────────────────────────
with open("params_maps.yml") as f:
    P = yaml.safe_load(f)

OP_MAP_FOLDER = P['op_map_folder']
DF_FOLDER = P['df_folder']
MONKEYS = P['monkeys']
V1_ARRAYS = P['v1_arrays']

SEL_MIN = P['channel_selection']['selectivity_min']
JUMP_MAX = P['channel_selection']['num_jump_max']
N_CH = P['method']['n_channels']

N_PERM = P['permutation']['n_perm']
SEED = P['permutation']['random_seed']

OUT_DIR = os.path.join(DF_FOLDER, 'permutations')
ensure_dir_exists(OUT_DIR)


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
            # Deterministic per-array seed: same array always gives the
            # same permutations regardless of run order.
            array_seed = SEED + hash((monkey, array)) % (2 ** 31)
            perms = make_permuted_maps(op_array, n_perm=N_PERM, seed=array_seed)

            out_path = os.path.join(
                OUT_DIR, f'perm_maps_monkey_{monkey}_array{array}.npy')
            np.save(out_path, perms)
            combined[(monkey, array)] = perms

            print(f"monkey {monkey}  array {array:2d}  "
                  f"oriented={n_oriented:2d}  perms={perms.shape}  -> {out_path}")

    combined_path = os.path.join(OUT_DIR, 'permuted_maps_all.pkl')
    with open(combined_path, 'wb') as f:
        pickle.dump(combined, f)
    print(f"\nSaved combined permutations: {combined_path}")
    print(f"Total arrays: {len(combined)}")


if __name__ == "__main__":
    main()
