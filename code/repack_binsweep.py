"""
repack_binsweep.py
==================
Repack the long-form `errors_long_binsweep.pkl` into a compact format.

The long form stores one row per control map (2000 rows per cell/pair/norm),
repeating all key columns each time -- huge and slow to filter. This repacks
the control errors into a single numpy vector per (cell, pair, norm), giving
a ~2000x smaller table.

Output (under df_folder/):
    errors_compact_binsweep.pkl   a dict with two pandas objects:
        'real'  : DataFrame, one row per
                  (monkey, date, array, condition, bin_ms, pc_pair, norm)
                  with columns pc_x, pc_y, real_error
        'ctrl'  : dict keyed by the same tuple -> np.ndarray [n_ctrl] of
                  control errors

Run once after merge:
    python repack_binsweep.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import yaml

with open("params_binsweep.yml") as f:
    P = yaml.safe_load(f)
DF_FOLDER = P['df_folder']

IN_PATH = os.path.join(DF_FOLDER, 'errors_long_binsweep.pkl')
OUT_PATH = os.path.join(DF_FOLDER, 'errors_compact_binsweep.pkl')

KEY_COLS = ["monkey", "date", "array", "condition", "bin_ms", "pc_pair", "norm"]


def main():
    print("loading", IN_PATH)
    df = pd.read_pickle(IN_PATH)
    print("  rows:", len(df))

    # real: one row per key
    real = (df[df.source == "real"]
            .rename(columns={"error": "real_error"})
            [KEY_COLS + ["pc_x", "pc_y", "real_error"]]
            .reset_index(drop=True))

    # ctrl: pool the 2000 errors into one vector per key
    ctrl = df[df.source == "ctrl"]
    ctrl_vec = {}
    grouped = ctrl.groupby(KEY_COLS, sort=False)["error"]
    for key, s in grouped:
        ctrl_vec[key] = s.to_numpy()

    out = {"real": real, "ctrl": ctrl_vec, "key_cols": KEY_COLS}
    with open(OUT_PATH, "wb") as f:
        pickle.dump(out, f)

    # report sizes
    in_mb = os.path.getsize(IN_PATH) / 1e6
    out_mb = os.path.getsize(OUT_PATH) / 1e6
    n_ctrl = len(next(iter(ctrl_vec.values()))) if ctrl_vec else 0
    print(f"  real rows: {len(real)}  | ctrl keys: {len(ctrl_vec)}  "
          f"| controls per key: {n_ctrl}")
    print(f"saved {OUT_PATH}")
    print(f"  size: {in_mb:.0f} MB -> {out_mb:.0f} MB")


if __name__ == "__main__":
    main()
