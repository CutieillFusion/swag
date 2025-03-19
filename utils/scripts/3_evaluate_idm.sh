#!/bin/bash
#SBATCH --job-name=RL_DYLAN
#SBATCH --output=slurm/idm/%j.out
#SBATCH --error=slurm/idm/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=450GB

cd /data/ai_club/nes_2025/swag

export PYTHONPATH=/data/ai_club/nes_2025/swag:$PYTHONPATH

/data/ai_club/nes_2025/swag/.venv/bin/python idm/evalutate_idm.py --model_path idm/models/$1/best_model.pt --output_dir idm/models/$1 --stride $2