#!/bin/bash
#SBATCH --job-name=DeepLMITrain
#SBATCH --output=DeepLMILog.txt
#SBATCH --partition=GPU_nvrtx4090
#SBATCH --mem-per-cpu=20000
#SBATCH --cpus-per-task=2

set -x

rm -f checkpoint_deeplmi.pt
/home/cicek/miniconda3/envs/deeplmi/bin/python -u main_independent_custom.py
