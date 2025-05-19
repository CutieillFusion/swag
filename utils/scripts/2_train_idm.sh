#!/bin/bash
#SBATCH --job-name=RL_DYLAN
#SBATCH --output=slurm/idm/%j.out
#SBATCH --error=slurm/idm/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=undergrad_research
#SBATCH --partition=dgxh100
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --time=5-00:00:00
#SBATCH --mem=1200GB

cd /data/ai_club/nes_2025/swag

export PYTHONPATH=/data/ai_club/nes_2025/swag:$PYTHONPATH

singularity exec \
    --env PYTHONPATH=$PYTHONPATH \
    --nv \
    -B /data/ai_club/nes_2025/swag:/data/ai_club/nes_2025/swag \
    utils/containers/container.sif \
    torchrun --nproc_per_node=8 \
    idm/train_idm.py \
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
# singularity exec \
#     --env PYTHONPATH=$PYTHONPATH \
#     --nv \
#     -B /data/ai_club/nes_2025/swag:/data/ai_club/nes_2025/swag \
#     utils/containers/container.sif \
#     torchrun --nproc_per_node=8 \
#     idm/train_idm.py \
#     --job_id 2 \
#     --embedding_dim 128 \
#     --ff_dim 128 \
#     --transformer_blocks 1 \
#     --transformer_heads 2 \
#     --x 256 \
#     --y 240 \
#     --learning_rate 0.0002147 \
#     --weight_decay 0.000001 \
#     --feature_channels 16,16,16