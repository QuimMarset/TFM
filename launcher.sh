#!/bin/bash

#SBATCH --job-name="TFM-EPyMARL"
#SBATCH --qos=bsc_cs
#SBATCH -D .
#SBATCH --output=./outputs/tfm_epymarl_%j.out
#SBATCH --error=./outputs/tfm_epymarl_%j.err
#SBATCH --cpus-per-task=40
#SBATCH --gres gpu:1
#SBATCH --time=30:00:00

module purge; 
module load singularity

singularity exec /apps/SINGULARITY/images/pettingzoo.sif python -c "import mujoco"
