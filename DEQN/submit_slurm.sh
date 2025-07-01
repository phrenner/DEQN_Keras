#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=1
#SBATCH -p gpu-medium


source /etc/profile
module add miniconda/2023.06
module add cuda/11.8
module add opence/1.10.0
export PYTHONIOENCODING=utf-8
export CUDA_DIR=/usr/shared_apps/packages/cuda-11.8
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/shared_apps/packages/cuda-11.8
python run_deepnet.py ENABLE_TB=False VERBOSE=0

