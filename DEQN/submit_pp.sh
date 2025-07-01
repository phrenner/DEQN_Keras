#!/bin/bash

#SBATCH -p serial
#SBATCH -J post_process
#SBATCH --time=03:00:00
#SBATCH --mem=128G

export OMP_NUM_THREADS=1
source /etc/profile
module add miniconda/2023.06
module add cuda/11.8
module add opence/1.10.0
export PYTHONIOENCODING=utf-8
export CUDA_DIR=/usr/shared_apps/packages/cuda-11.8
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/shared_apps/packages/cuda-11.8

python post_process.py && python impulse_response.py 