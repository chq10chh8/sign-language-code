#!/bin/bash
#SBATCH --job-name=shubert_full_w8_200k
#SBATCH --account=project_465003161
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=60G
#SBATCH --output=/scratch/project_465003161/YouTube_ASL/logs/slurm_%x_%j.out
#SBATCH --error=/scratch/project_465003161/YouTube_ASL/logs/slurm_%x_%j.err

# 1. 环境准备
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif

# 2. 路径配置
fairseq_root=/scratch/project_465003161/SHuBERT/fairseq
path=$fairseq_root/examples/shubert/config
code_dir=$fairseq_root/examples/shubert
custom_libs=/scratch/project_465003161/YouTube_ASL/custom_python_libs

# 3. 环境变量
export NCCL_SOCKET_IFNAME=hsn0,hsn1
export SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0,hsn1
export NCCL_DEBUG=INFO
export SINGULARITYENV_NCCL_DEBUG=INFO
export HDF5_USE_FILE_LOCKING=FALSE
export SINGULARITYENV_HDF5_USE_FILE_LOCKING=FALSE
export SINGULARITYENV_WANDB_API_KEY="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"
export SINGULARITYENV_PYTHONPATH="$custom_libs:$fairseq_root:$fairseq_root/examples"
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# 4. 日志目录
log_dir=/scratch/project_465003161/YouTube_ASL/training_logs/run_full_w8_iter2
mkdir -p $log_dir

# 5. 启动训练
srun singularity run \
    -B /scratch:/scratch \
    $SIF \
    python3 -m fairseq_cli.hydra_train \
    --config-dir $path \
    --config-name base_full_w56_iter2 \
    common.user_dir=$code_dir \
    checkpoint.save_dir=$log_dir/ckpt \
    common.log_file=$log_dir/log.txt \
    common.tensorboard_logdir=$log_dir/tb \
    distributed_training.distributed_world_size=8 \
    distributed_training.distributed_port=$MASTER_PORT \
    "distributed_training.distributed_init_method=tcp://${MASTER_ADDR}:${MASTER_PORT}" \
    hydra.run.dir=$log_dir/hydra
