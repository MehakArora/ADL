#!/bin/bash
#SBATCH --job-name=langevin_dynamics
#SBATCH --output=langevin_dynamics_%j.out
#SBATCH --error=langevin_dynamics_%j.err
#SBATCH --partition=scavenger-gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB



# Activate virtual environment if needed
source /hpc/dctrl/ma618/torch/bin/activate

# Run the Gibbs sampler script
python LangevinDynamics.py
