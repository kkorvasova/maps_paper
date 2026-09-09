#!/bin/bash
#SBATCH --job-name=binsweep_prep
#SBATCH --mem=16G
#SBATCH -o output/binsweep_prep_%A.out
#SBATCH -e output/binsweep_prep_%A.err
#SBATCH --nodes=1

# One-time, non-parallel steps.
#   sbatch run_binsweep_prep.sh controls   # BEFORE the array job
#   sbatch run_binsweep_prep.sh merge      # AFTER  the array job

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/maps_paper/code

STEP=${1:-controls}

if [ "$STEP" = "controls" ]; then
    python generate_controls.py
elif [ "$STEP" = "merge" ]; then
    python compute_binsweep.py merge
else
    echo "unknown step: $STEP  (use 'controls' or 'merge')"
    exit 1
fi
