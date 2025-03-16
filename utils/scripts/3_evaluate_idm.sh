#!/bin/bash
#SBATCH --job-name=SWAG_3
#SBATCH --output=slurm/idm/%j.out
#SBATCH --error=slurm/idm/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=200GB

cd /data/ai_club/nes_2025/swag

export PYTHONPATH=/data/ai_club/nes_2025/swag:$PYTHONPATH

srun singularity exec --nv -B /data:/data /data/containers/msoe-pytorch-24.05-py3.sif python idm/evalutate_idm.py --model_path idm/best_model.pt --x 64 --y 60 --embedding_dim 512 --ff_dim 512 --transformer_blocks 2