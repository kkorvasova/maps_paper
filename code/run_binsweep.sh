#!/bin/bash
#SBATCH --job-name=binsweep
#SBATCH --array=0-20
#SBATCH --mem=1000G
#SBATCH -o output/binsweep_%A_%a.out
#SBATCH -e output/binsweep_%A_%a.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory
cd /CSNG/studekat/maps_paper/code

# One (monkey, bin_size) per task: 3 monkeys x 7 bins = 21 tasks (0..20).
# Controls must already exist (run generate_controls.py first).
python compute_binsweep.py $SLURM_ARRAY_TASK_ID
