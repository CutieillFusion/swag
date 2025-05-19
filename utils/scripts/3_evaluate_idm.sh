#!/bin/bash
#SBATCH --job-name=SWAG_3
#SBATCH --output=slurm/idm/%j.out
#SBATCH --error=slurm/idm/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --mem=350GB

cd /data/ai_club/nes_2025/swag

export PYTHONPATH=/data/ai_club/nes_2025/swag:$PYTHONPATH

/data/ai_club/nes_2025/swag/.venv/bin/python idm/evaluate_idm.py --model_path idm/models/$1/best_model.pt --output_dir idm/models/$1 --stride $2

# sbatch utils/scripts/3_evaluate_idm.sh 188499 2