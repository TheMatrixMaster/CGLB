#!/bin/bash
#SBATCH --job-name=dygformer-ncgl
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=long
#SBATCH --time=23:59:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --partition long

# Activate modules
module --force purge
module load python/3 cuda/11.1 pytorch/1.8.1

# Activate virtual environment
source ~/scratch/venvs/dyg/bin/activate

# Hyperparameters and other settings
METHOD=bare
BACKBONE=GCN
DATASET=Arxiv-CL
DATETIME=$(date '+%Y-%m-%d_%H-%M-%S')
INPUT_DIR=/home/mila/s/stephen.lu/scratch/dyg/in
OUTPUT_DIR=/home/mila/s/stephen.lu/scratch/dyg/out

# Run the training script
python train.py \
    --dataset $DATASET \
    --method $METHOD \
    --backbone $BACKBONE \
    --gpu 0 \
    --ILmode taskIL \
    --inter-task-edges False \
    --minibatch False \
    --ori_data_path $INPUT_DIR/raw \
    --data_path $INPUT_DIR/processed \
    --result_path $OUTPUT_DIR/$METHOD_$BACKBONE/$DATETIME \
