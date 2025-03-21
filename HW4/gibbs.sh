#!/bin/bash
#SBATCH --job-name=gibbs_sampler
#SBATCH --output=gibbs_sampler_%j.out
#SBATCH --error=gibbs_sampler_%j.err
#SBATCH --partition=gpu-common
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB



# Activate virtual environment if needed
source /hpc/dctrl/ma618/torch/bin/activate

# Run the Gibbs sampler script
python GibbsSampler.py
