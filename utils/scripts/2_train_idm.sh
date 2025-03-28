#!/bin/bash
#SBATCH --job-name=RL_DYLAN
#SBATCH --output=slurm/idm/%j.out
#SBATCH --error=slurm/idm/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --mem=350GB

cd /data/ai_club/nes_2025/swag

export PYTHONPATH=/data/ai_club/nes_2025/swag:$PYTHONPATH

/data/ai_club/nes_2025/swag/.venv/bin/python idm/train_idm.py \
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

# sbatch utils/scripts/2_train_idm.sh 512 2048 2 4 64 60 0.0002147 0.000001 32,64,64
# /data/ai_club/nes_2025/swag/.venv/bin/python idm/train_idm.py \
#    --job_id 2 \
#    --embedding_dim 512 \
#    --ff_dim 2048 \
#    --transformer_blocks 2 \
#    --transformer_heads 4 \
#    --x 64 \
#    --y 60 \
#    --learning_rate 0.0002147 \
#    --weight_decay 0.000001 \
#    --feature_channels 32,64,64