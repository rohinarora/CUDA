#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec
#SBATCH --reservation=GPU-CLASS-SP20
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:1
#SBATCH --output=exec.%j.out

cd /scratch/$USER/GPUClassS20/HOL3/

set -o xtrace
nvidia-smi
./vadd 100000000
