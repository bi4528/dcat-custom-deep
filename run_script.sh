#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=gpu-test
#SBATCH --output=logs/gpu-job-%j.out

source ~/venv/ckanenv/bin/activate

module load CUDA/11.8.0
 
srun python './src/test.py'