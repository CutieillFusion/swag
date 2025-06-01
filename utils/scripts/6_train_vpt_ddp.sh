#!/bin/bash
#SBATCH --job-name=SWAG_6
#SBATCH --output=slurm/vpt/%j.out
#SBATCH --error=slurm/vpt/%j.err
#SBATCH --nodes=18
#SBATCH --gpus-per-node=4
#SBATCH --partition=teaching
#SBATCH --account=undergrad_research
#SBATCH --cpus-per-task=60
#SBATCH --time=5-00:00:00
#SBATCH --mem=150GB

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=12355
export OMP_NUM_THREADS=9

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "Nodes:"
echo "$(scontrol show hostnames $SLURM_JOB_NODELIST)"

srun --mpi=none \
  singularity exec --nv \
    --env PYTHONPATH=$PYTHONPATH \
    -B /data/ai_club/nes_2025/swag:/data/ai_club/nes_2025/swag \
    utils/containers/container.sif \
      torchrun \
      --nnodes=$SLURM_NNODES \
      --nproc_per_node=$SLURM_GPUS_PER_NODE \
      --node_rank=$SLURM_NODEID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      --rdzv_id=$SLURM_JOB_ID \
      vpt/train_vpt_ddp.py \
        --config vpt.yaml \
        --job_id $SLURM_JOB_ID