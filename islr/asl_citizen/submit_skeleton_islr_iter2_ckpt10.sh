#!/bin/bash
#SBATCH --job-name=skeleton_islr_iter2_ckpt10
#SBATCH --account=project_465003161
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/project_465003161/logs/asl_citizen/skeleton_islr_%j.out
#SBATCH --error=/scratch/project_465003161/logs/asl_citizen/skeleton_islr_%j.err

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260124_092648/lumi-multitorch-full-u24r64f21m43t29-20260124_092648.sif
export SINGULARITYENV_PYTHONPATH="/scratch/project_465003161/mediapipe_env:/scratch/project_465003161/YouTube_ASL/custom_python_libs"

mkdir -p /scratch/project_465003161/logs/asl_citizen
mkdir -p /scratch/project_465003161/asl_citizen_skeleton_islr_results_iter2_ckpt10

srun singularity exec -B /scratch:/scratch $SIF python3 \
    /scratch/project_465003161/SHuBERT/train_islr_lora_skeleton_asl_citizen.py \
    --skeleton_base /scratch/project_465003161/ASL_Citizen_skeleton \
    --train_csv /scratch/project_465003161/ASL_Citizen/ASL_Citizen/splits/train.csv \
    --val_csv /scratch/project_465003161/ASL_Citizen/ASL_Citizen/splits/val.csv \
    --test_csv /scratch/project_465003161/ASL_Citizen/ASL_Citizen/splits/test.csv \
    --checkpoint_path /scratch/project_465003161/YouTube_ASL/training_logs/run_full_w8_iter2/ckpt/checkpoint10.pt \
    --output_dir /scratch/project_465003161/asl_citizen_skeleton_islr_results_iter2_ckpt10 \
    --batch_size 8 \
    --epochs 125 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --max_frames 500 \
    --lora_r 1 \
    --patience 20
