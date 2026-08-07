#!/bin/bash -l

# Configure the resources required
#SBATCH -p a100
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB

# Notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yi.ren@adelaide.edu.au

# Execute the program

module load Anaconda3/2024.06-1

conda activate domain-adaptation
cd "$SLURM_SUBMIT_DIR/.."
python scripts/save_all_checkpoints.py
conda deactivate

