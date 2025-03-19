#!/bin/bash
#SBATCH --job-name=RL_DYLAN
#SBATCH --output=slurm/misc/%j.out
#SBATCH --error=slurm/misc/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=40
#SBATCH --time=5-00:00:00
#SBATCH --mem=400GB