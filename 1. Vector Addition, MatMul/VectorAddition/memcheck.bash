#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=mem
#SBATCH --reservation=gpu-class
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=mem.%j.out

cd /scratch/`whoami`/GPUClassS19/HOL2/

set -o xtrace
cuda-memcheck ./vAdd 1000000 2048
cuda-memcheck ./vAdd 1000000 256
