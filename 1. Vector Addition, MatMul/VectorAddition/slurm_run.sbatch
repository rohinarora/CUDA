#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=exec
#SBATCH --reservation=GPU-CLASS-SP20
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:1
#SBATCH --output=exec.%j.out


set -o xtrace
./vAdd 10000000 32
./vAdd 10000000 64
./vAdd 10000000 128
./vAdd 10000000 256
./vAdd 10000000 512
./vAdd 10000000 1024
./vAdd 10000000 2048
./vAdd 1000000 1024
./vAdd 100000 1024
./vAdd 10000 1024
./vAdd 1000 1024
./vAdd 100 1024
./vAdd 10 1024
