#!/bin/bash
#SBATCH --job-name=SWAG_3
#SBATCH --output=slurm/idm/%j.out
#SBATCH --error=slurm/idm/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --partition=dgxh100
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
#SBATCH --mem=500GB

