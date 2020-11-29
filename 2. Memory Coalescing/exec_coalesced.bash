#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec_coalesced
#SBATCH --reservation=GPU-CLASS-SP20
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:1
#SBATCH --output=exec_coalesced.%j.out

cd /scratch/$USER/GPUClassS19/HOL3/

set -o xtrace
nvidia-smi
./vadd_coalesced 100000000
