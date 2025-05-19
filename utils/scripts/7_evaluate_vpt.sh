#!/bin/bash
#SBATCH --job-name=RL_DYLAN
#SBATCH --output=slurm/vpt/%j.out
#SBATCH --error=slurm/vpt/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --mem=350GB

cd /data/ai_club/nes_2025/swag

export PYTHONPATH=/data/ai_club/nes_2025/swag:$PYTHONPATH

# /data/ai_club/nes_2025/swag/.venv/bin/python vpt/evaluate_vpt.py --model_path vpt/models/$1/best_model.pt --output_dir vpt/models/$1 --stride $2

/data/ai_club/nes_2025/swag/.venv/bin/python emulator/run_vpt.py --model_path vpt/models/$1/best_model.pt --output_dir emulator/models/$1

# sbatch utils/scripts/7_evaluate_vpt.sh 189783 2