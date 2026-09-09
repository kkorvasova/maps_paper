#!/bin/bash
#SBATCH --job-name=op_maps
#SBATCH --array=0,1,2
#SBATCH --mem=200G
#SBATCH -o output/op_maps_%A_%a.out
#SBATCH -e output/op_maps_%A_%a.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory
cd /CSNG/studekat/maps_paper/code

# Main analysis for ONE monkey (task 0 -> L, 1 -> N, 2 -> F).
# Permutations must already exist (run generate_permutations.py first).
python compute_errors.py $SLURM_ARRAY_TASK_ID
