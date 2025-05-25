#!/bin/bash
#SBATCH --job-name=SWAG_5
#SBATCH --output=slurm/misc/%j.out
#SBATCH --error=slurm/misc/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=5-00:00:00
#SBATCH --mem=300GB

export PYTHONPATH=/data/ai_club/nes_2025/swag:$PYTHONPATH

/data/ai_club/nes_2025/swag/.venv/bin/python /data/ai_club/nes_2025/swag/idm/run_idm.py --model_path idm/models/$1/best_model.pt

# sbatch utils/scripts/5_label_with_idm.sh 189912