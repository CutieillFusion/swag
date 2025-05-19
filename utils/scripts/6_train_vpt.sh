#!/bin/bash
#SBATCH --job-name=SWAG_6
#SBATCH --output=slurm/vpt/%j.out
#SBATCH --error=slurm/vpt/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --mem=450GB

cd /data/ai_club/nes_2025/swag

export PYTHONPATH=/data/ai_club/nes_2025/swag:$PYTHONPATH

/data/ai_club/nes_2025/swag/.venv/bin/python vpt/train_vpt.py \
    --job_id $SLURM_JOB_ID \
    --embedding_dim $1 \
    --ff_dim $2 \
    --transformer_blocks $3 \
    --transformer_heads $4 \
    --x $5 \
    --y $6 \
    --learning_rate $7 \
    --weight_decay $8 \
    --feature_channels $9

# sbatch utils/scripts/6_train_vpt.sh 1024 2048 4 8 256 240 0.0002147 0.000001 64,128,128
# /data/ai_club/nes_2025/swag/.venv/bin/python vpt/train_vpt.py \
#    --job_id 0 \
#    --embedding_dim 1024 \
#    --ff_dim 2048 \
#    --transformer_blocks 4 \
#    --transformer_heads 8 \
#    --x 256 \
#    --y 240 \
#    --learning_rate 0.0002147 \
#    --weight_decay 0.000001 \
#    --feature_channels 64,128,128

# /data/ai_club/nes_2025/swag/.venv/bin/python vpt/train_vpt.py \
#    --job_id 0 \
#    --embedding_dim 256 \
#    --ff_dim 256 \
#    --transformer_blocks 1 \
#    --transformer_heads 2 \
#    --x 32 \
#    --y 30 \
#    --learning_rate 0.0002147 \
#    --weight_decay 0.000001 \
#    --feature_channels 16,32,32